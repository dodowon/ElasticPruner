# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
# Modified by dodowon, 2026. See THIRD_PARTY_NOTICES.md for details.

"""
Pre-compute teacher logits for the Genetic Algorithm (GA) pruning search.

Background
----
The current GA (the `GeneticSparsityCallback` family) draws only a single sample
(sample_batch_size=1) from the training batch on each call and compares it to the teacher logits to compute fitness.
With few samples the randomness is large, so the GA fitness does not generalize. Yet if the
number of GA samples is increased during training, the teacher forward + forward of every population member
put a heavy load on GPU memory and OOM can occur.

Solution
----
This script runs *once in advance*, separately from training, and:
  1. Samples the specified number (`num_samples`) of items from the training dataloader
     (`build_text_dataloader`),
  2. Feeds them as input to the HuggingFace teacher model created from `genetic.teacher_model_name`
     to generate logits (large models are forwarded in chunks to avoid OOM),
  3. Saves the data and logits to disk in dictionary form (torch binary or pickle).

If the GA callback loads this saved (data, logit) pool and computes fitness over many samples
without a teacher forward at every step, the GA variance can be reduced without memory burden.

See `elasticpruner/scripts/make_teacher_logit_for_ga.sh` for a usage example.

  python elasticpruner/make_teacher_logit_for_ga.py <config.yaml> \
      genetic.teacher_model_name=princeton-nlp/Sheared-LLaMA-1.3B \
      teacher_logit.num_samples=256 \
      teacher_logit.sample_seq_len=128 \
      teacher_logit.chunk_size=4 \
      teacher_logit.output_path=/path/to/teacher_logits.pt
"""

import os
import pickle
import sys
import time
import warnings
from collections import defaultdict

import torch
from omegaconf import OmegaConf as om

from llmfoundry.utils.config_utils import update_batch_size_info
from transformers import AutoModelForCausalLM

import streaming

from elasticpruner.datasets.load_text_dataloader import build_text_dataloader

# Same as the training script: clean up streaming shared memory left over from a previous run
streaming.base.util.clean_stale_shared_memory()


DTYPE_MAP = {
    "float32": torch.float32, "fp32": torch.float32,
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    """Convert a string to a torch.dtype."""
    if isinstance(name, torch.dtype):
        return name
    key = str(name).lower()
    if key not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{name}'. Choose from {list(DTYPE_MAP)}")
    return DTYPE_MAP[key]


def get_teacher_model_name(cfg) -> str:
    """Read genetic.teacher_model_name (or the top-level teacher_model_name if absent)."""
    genetic = cfg.get("genetic", None)
    if genetic is not None and genetic.get("teacher_model_name", None):
        return genetic.teacher_model_name
    name = cfg.get("teacher_model_name", None)
    if name:
        return name
    raise ValueError(
        "A teacher model name is required. "
        "Please pass `genetic.teacher_model_name=<hf-model>`."
    )


def resolve_output_path(cfg, teacher_model_name: str, num_samples: int,
                        sample_seq_len: int, save_format: str) -> str:
    """Determine the save path. Use teacher_logit.output_path as is if present, otherwise generate one automatically."""
    tl_cfg = cfg.get("teacher_logit", {})
    out = tl_cfg.get("output_path", None) if tl_cfg else None
    if out:
        return out

    ext = "pt" if save_format == "torch" else "pkl"
    model_tag = teacher_model_name.rstrip("/").split("/")[-1]
    fname = f"teacher_logits_{model_tag}_n{num_samples}_s{sample_seq_len}.{ext}"

    save_folder = cfg.get("save_folder", None)
    if save_folder and save_folder != "save_dir":
        save_folder = save_folder.replace("{run_name}", cfg.get("run_name", "ga"))
        return os.path.join(save_folder, fname)
    return os.path.join("elasticpruner", "ga_teacher_logits", fname)


def collect_samples(train_loader, num_samples: int, sample_seq_len: int):
    """Collect num_samples sequences from the dataloader and return them as dict[str, Tensor].

    After gathering all tensor fields of each batch (input_ids, labels, set, ...),
    2D tensors (input_ids, labels) are truncated to sample_seq_len.
    """
    collected = defaultdict(list)
    n_collected = 0

    for batch in train_loader:
        if "input_ids" not in batch:
            raise KeyError(f"'input_ids' is missing from the batch. keys={list(batch.keys())}")
        bs = batch["input_ids"].shape[0]
        take = min(bs, num_samples - n_collected)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                collected[k].append(v[:take].detach().cpu())
        n_collected += take
        print(f"  collected {n_collected}/{num_samples} samples", flush=True)
        if n_collected >= num_samples:
            break

    if n_collected < num_samples:
        warnings.warn(
            f"Only {n_collected} samples were collected, fewer than the {num_samples} requested "
            f"(the dataloader was exhausted)."
        )

    data = {k: torch.cat(v, dim=0) for k, v in collected.items()}

    # Reduce the sequence length (2D tensors such as input_ids, labels)
    for k in list(data.keys()):
        if data[k].ndim == 2 and data[k].shape[1] > sample_seq_len:
            data[k] = data[k][:, :sample_seq_len].contiguous()

    return data


@torch.no_grad()
def compute_teacher_logits(teacher, input_ids: torch.Tensor, chunk_size: int,
                           device: str, save_dtype: torch.dtype) -> torch.Tensor:
    """Compute logits with the teacher model. The batch dimension is split into chunks for the forward pass to avoid OOM.

    The logits of each chunk are immediately moved to CPU and cast to save_dtype so that GPU memory
    does not accumulate.
    """
    teacher.eval()
    total = input_ids.shape[0]
    logits_chunks = []

    for i in range(0, total, chunk_size):
        chunk = input_ids[i:i + chunk_size].to(device)
        out = teacher(chunk)
        logits = out.logits if hasattr(out, "logits") else out["logits"]
        logits_chunks.append(logits.to(device="cpu", dtype=save_dtype))
        del out, logits, chunk
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"  teacher forward {min(i + chunk_size, total)}/{total}", flush=True)

    return torch.cat(logits_chunks, dim=0)


def main(cfg):
    print("Start making teacher logits for GA", flush=True)

    # ── 0. Parse settings ──
    tl_cfg = cfg.get("teacher_logit", {})
    num_samples = int(tl_cfg.get("num_samples", 256)) if tl_cfg else 256
    sample_seq_len = int(tl_cfg.get("sample_seq_len", 128)) if tl_cfg else 128
    chunk_size = int(tl_cfg.get("chunk_size", 4)) if tl_cfg else 4
    device = (tl_cfg.get("device", "cuda") if tl_cfg else "cuda")
    compute_dtype = resolve_dtype(tl_cfg.get("compute_dtype", "float16") if tl_cfg else "float16")
    save_dtype = resolve_dtype(tl_cfg.get("save_dtype", "float16") if tl_cfg else "float16")
    save_format = (tl_cfg.get("save_format", "torch") if tl_cfg else "torch")
    assert save_format in ("torch", "pickle"), \
        f"save_format must be 'torch' or 'pickle' (got {save_format})"

    if device.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA is not available, so the device is changed to cpu.")
        device = "cpu"
        compute_dtype = torch.float32  # float32 is safe on cpu

    teacher_model_name = get_teacher_model_name(cfg)
    output_path = resolve_output_path(cfg, teacher_model_name, num_samples,
                                      sample_seq_len, save_format)

    print(f"  teacher_model_name = {teacher_model_name}", flush=True)
    print(f"  num_samples={num_samples}, sample_seq_len={sample_seq_len}, "
          f"chunk_size={chunk_size}", flush=True)
    print(f"  device={device}, compute_dtype={compute_dtype}, save_dtype={save_dtype}", flush=True)
    print(f"  save_format={save_format}, output_path={output_path}", flush=True)

    # ── 1. Build the dataloader (same way as the training script) ──
    cfg = update_batch_size_info(cfg)
    assert getattr(cfg.callbacks.data_loading, "set_names", None) is not None, \
        "Please specify callbacks.data_loading.set_names (domain names) in the config."

    if cfg.callbacks.data_loading.dynamic:
        cfg.model.set_names = cfg.callbacks.data_loading.set_names

    print("Building train loader...", flush=True)
    train_loader = build_text_dataloader(
        cfg.train_loader,
        cfg.device_train_batch_size,
        cfg.callbacks.data_loading.dynamic,
        cfg.callbacks.data_loading.set_names,
        proportion=cfg.callbacks.data_loading.proportion,
    )

    # ── 2. Sample the data ──
    print(f"Collecting {num_samples} samples from dataloader...", flush=True)
    data = collect_samples(train_loader, num_samples, sample_seq_len)
    print("  collected tensors:", {k: tuple(v.shape) for k, v in data.items()}, flush=True)

    # ── 3. Load the teacher model ──
    print(f"Loading teacher model: {teacher_model_name} ...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    teacher.to(device)
    teacher.eval()
    print("  teacher model loaded", flush=True)

    # ── 4. Compute teacher logits (chunked forward) ──
    print("Computing teacher logits...", flush=True)
    start = time.time()
    logits = compute_teacher_logits(teacher, data["input_ids"], chunk_size, device, save_dtype)
    elapsed = time.time() - start
    print(f"  teacher logits: shape={tuple(logits.shape)}, dtype={logits.dtype}, "
          f"elapsed={elapsed:.1f}s", flush=True)

    # ── 5. Build and save the result dict ──
    result = {k: v for k, v in data.items()}  # input_ids, labels, set, ...
    result["logits"] = logits
    result["meta"] = {
        "teacher_model_name": teacher_model_name,
        "num_samples": int(data["input_ids"].shape[0]),
        "sample_seq_len": int(data["input_ids"].shape[1]),
        "vocab_size": int(logits.shape[-1]),
        "compute_dtype": str(compute_dtype),
        "save_dtype": str(save_dtype),
        "set_names": list(cfg.callbacks.data_loading.set_names),
        "created_at": time.time(),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Saving to {output_path} (format={save_format}) ...", flush=True)
    if save_format == "torch":
        torch.save(result, output_path)
    else:
        with open(output_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"Done. Saved {result['meta']['num_samples']} samples "
          f"({size_mb:.1f} MB) to {output_path}", flush=True)


if __name__ == "__main__":
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
