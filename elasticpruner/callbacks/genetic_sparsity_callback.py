# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

import os
import time
import random
from typing import Any, Dict, List
import math

import torch
from composer import Callback, Logger, State
from composer.loggers import Logger
from composer.utils import dist
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from types import SimpleNamespace
from copy import deepcopy

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from elasticpruner.utils.gentic_algo_new_total_sparsity import AirGeneticSparsity, change_target_mask_size
from elasticpruner.utils.genetic_algo_eval_loss import AirGeneticEvalLoss
from elasticpruner.utils.genetic_contraint import AirGeneticSparsityConstraint, AirGeneticSparsityConstraintLayerWise
from elasticpruner.utils.genetic_contraint_w_head import AirGeneticSparsityConstraintHead
from elasticpruner.utils.distribution_test import get_beta_dist_value, find_beta_param_for_peak
from elasticpruner.ga_dataset.eval_dataset import GAEvalDataset

def get_normal_dist_value(current_step, total_steps=3200, peak_step=640):
    """
    Returns a value on the normal distribution curve based on the current step.
    
    :param current_step: The current step
    :param total_steps: Total number of steps (default: 3200)
    :param peak_step: The step that is both the mean and the maximum (1.0) (default: 640)
    :return: A normal distribution value between 0.0 and 1.0
    """
    mu = peak_step
    
    # Set the standard deviation (sigma) so that the value is nearly 0 at the target total step (3200)
    # (3200 - 640) / 3 = about 853.33
    sigma = (total_steps - peak_step) / 3.0 
    
    # Exception handling: avoid errors if the total step and the peak step are equal
    if sigma == 0:
        return 1.0 if current_step == peak_step else 0.0

    # Apply the Gaussian (normal distribution) formula (maximum value of 1.0)
    value = math.exp(-((current_step - mu) ** 2) / (2 * (sigma ** 2)))
    
    return value

def calculate_max_step_size(total_steps, interval, target_change=4.5):
    """
    Computes the optimal MAX_STEP_SIZE needed to reach the target cumulative change
    when using a polynomial schedule (k=2).
    
    Args:
        total_steps (int): Total number of training steps
        interval (int): Call interval of the enforcement function (e.g., 100)
        target_change (float): Target amount of parameter change.
                               The default is set generously to 4.5 so that +/- 4.0 is reliably reached.
    
    Returns:
        float: The computed MAX_STEP_SIZE
    """
    total_calls = total_steps // interval
    
    # Edge case: when the number of calls is 0 (interval is larger than the total steps)
    if total_calls == 0:
        return target_change 
        
    # Formula: MAX_STEP_SIZE = (3 * Target) / N
    max_step_size = (3.0 * target_change) / total_calls
    
    return max_step_size

class GeneticSparsityCallback(Callback):
    """
    Runs the genetic algorithm at eval_end after a certain step
    to search for the optimal per-module sparsity (d_model, intermediate_size, n_layers),
    and a Callback that permanently applies the result to the mask of the actual l0_module.
    """
    def __init__(
        self,
        teacher_model_name: str = "meta-llama/Llama-2-7b-hf",
        target_params: int = 1_345_423_360,
        sim_func: dict = None,
        start_step: int = 640,
        population_size: int = 500,
        origin_module_cfg: dict = None,
        init_cfg: dict = None,
        range_ratio: float = 0.1,
        sample_batch_size: int = 4,
        save_folder: str = None,
        seed: int = 42,
        pruning_cfg: dict = None
    ) -> None:
        self.teacher_model_name = teacher_model_name
        self.target_params = target_params
        self.sim_func_cfg = sim_func or {"func_name": "kl"}
        self.start_step = start_step
        self.population_size = population_size
        self.origin_module_cfg = origin_module_cfg or {
            "n_layers": 32, "d_model": 4096, "n_heads": 32, "intermediate_size": 11008
        }
        self.init_cfg = init_cfg or {
            "n_layers": 24, "d_model": 1024, "intermediate_size": 2816
        }
        self.range_ratio = range_ratio
        self.sample_batch_size = sample_batch_size
        self.save_folder = save_folder
        self.seed = seed
        self.pruning_cfg = pruning_cfg

        self.current_step = 0
        self.total_step = int(pruning_cfg.get("total_step", 3200))
        self.eval_interval = int(pruning_cfg.get("eval_interval", 100))
        # lazy-loaded
        self.teacher_model = None
        self.applied = False  # whether it has already been applied
        self.best_config = None

        self.is_gradual = pruning_cfg.get("is_gradual", False)
        self.is_linear = pruning_cfg.get("is_linear", False)
        self.fixed_head = pruning_cfg.get("fixed_head", True)
        self.ga_step_func = pruning_cfg.get("ga_step_func", None)
        self.ga_data = pruning_cfg.get("ga_data",None)
        self.ga_data_seq_len = pruning_cfg.get("ga_data_seq_len",128)
        print("-----ga step func :", self.ga_step_func, " ", self.ga_step_func == "beta")
        print("------ga data len :", self.ga_data_seq_len)

        self.max_step_size = calculate_max_step_size(self.total_step, self.eval_interval, 4.5)
        self.is_test = False
        self.lag_warm_up_step = 640
        self.curr_ga_step_size = 0

    def _lazy_load_teacher(self, device="cpu"):
        """Load the teacher model on the first call"""
        if self.teacher_model is not None:
            return
        print(f"[GeneticSparsity] Loading teacher model: {self.teacher_model_name}", flush=True)
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            self.teacher_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        self.teacher_model.to(device)
        self.teacher_model.eval()
        print(f"[GeneticSparsity] Teacher model loaded on {device}", flush=True)


    def plug_in_pruned_steps(self, state: State, logger: Logger):
        """Insert pruned_steps into the batch (PruningCallback pattern)"""
        if getattr(state.model.model, "l0_module", None) is not None:
            input_ids = state.batch["input_ids"]
            state.batch["pruned_steps"] = torch.LongTensor(
                [state.timestamp.batch.value] * len(input_ids)
            ).to(input_ids.device)

    def batch_start(self, state: State, logger: Logger):
        self.plug_in_pruned_steps(state, logger)

    def eval_batch_start(self, state: State, logger: Logger):
        self.plug_in_pruned_steps(state, logger)

    def eval_end(self, state: State, logger: Logger) -> None:
        step = state.timestamp.batch.value
        l0_module = state.model.model.l0_module
        self.current_step = step

        if l0_module is None:
            return

        # skip if before start_step
        if step < self.start_step:
            print(f"[GeneticSparsity] step={step} < start_step={self.start_step}, skipping.", flush=True)
            return

        # skip if already applied (run only once)
        if self.applied:
            print(f"[GeneticSparsity] step={step}: already applied, skipping.", flush=True)
            return

        print(f"\n{'='*60}", flush=True)
        print(f"[GeneticSparsity] step={step}: Starting genetic sparsity search", flush=True)
        print(f"{'='*60}", flush=True)

        # ── 1. Log the current mask state ──
        with FSDP.summon_full_params(state.model, writeback=True, recurse=True):
            for k in l0_module.masks:
                score, sparsity = l0_module.masks[k].calculate_expected_score_sparsity()
                print(f"  {k}: {sparsity.mean().item():.4f}")

        # ── 2. Lazy-load the Teacher model (rank 0 only) ──
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0 and self.ga_data is None:
            self._lazy_load_teacher(device="cpu")

        # ── 3. Prepare a sample batch from the eval data ──
        if self.ga_data is not None:
            # Use the pre-saved teacher logit pool → must be built as a dict for the student forward.
            # (If a bare tensor is passed, `for k in data` below would iterate over a tensor instead of a dict)
            if type(self.ga_data) is str:
                self.ga_data = torch.load(self.ga_data)
            model_device = next(state.model.parameters()).device
            data = {"input_ids": self.ga_data["input_ids"].to(model_device)}
            if "labels" in self.ga_data:
                data["labels"] = self.ga_data["labels"].to(model_device)
            bs = data["input_ids"].shape[0]
            data["pruned_steps"] = torch.zeros(bs, dtype=torch.long, device=model_device)
            print("use teacher logit path")
        else:
            data = self._get_sample_batch(state)
        if data is None:
            print("[GeneticSparsity] Failed to get sample batch, skipping.", flush=True)
            return

        # ── 4. Set up sim_func ──
        sim_fc = deepcopy(self.sim_func_cfg)
        if "topk_rate" in sim_fc and isinstance(sim_fc["topk_rate"], float) and sim_fc["topk_rate"] < 1.0:
            sim_fc["topk_rate"] = int(32000 * sim_fc["topk_rate"])

        # ── 5. Pre-compute the Teacher output (forward on rank 0 only, then broadcast) ──
        model_param = next(state.model.parameters())
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        # Reduce the sequence length (slice all 2D tensors together)
        # input_ids/labels are integer tensors, so they must not be cast to a float dtype (only match the device).
        max_seq_len = self.ga_data_seq_len
        for k in data:
            if isinstance(data[k], torch.Tensor) and data[k].ndim == 2:
                data[k] = data[k][:, :max_seq_len].to(device=model_param.device)
        if self.ga_data is not None:
            self.ga_data['logits'] = self.ga_data['logits'][:, :max_seq_len].to(device=model_param.device)

        print(f"[GeneticSparsity] Computing teacher output (rank 0 only, seq_len={max_seq_len})...", flush=True)
        # First check the output shape with one student model forward pass
        state.model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=model_param.dtype):
            dummy_output = state.model(data)
        logits_shape = dummy_output['logits'].shape
        logits_dtype = dummy_output['logits'].dtype
        logits_device = dummy_output['logits'].device
        if rank == 0 and self.ga_data is None:
            with torch.no_grad():
                print("start teacher forward")
                teacher_out = self.teacher_model(data['input_ids'].to('cpu'))
                teacher_logits = teacher_out['logits'].to(device=logits_device, dtype=logits_dtype)
                print("end teacher forward")
        elif rank == 0 and self.ga_data is not None:
            #teacher_logits = torch.empty(logits_shape, dtype=logits_dtype, device=logits_device)
            teacher_logits = self.ga_data['logits'].to(device=logits_device, dtype=logits_dtype)
            #teacher_logits = self.ga_data['logits'].to(device="cpu", dtype=logits_dtype)
        else:
            teacher_logits = torch.empty(logits_shape, dtype=logits_dtype, device=logits_device)


        if world_size > 1:
            torch.distributed.broadcast(teacher_logits, src=0)

        teacher_output = {'logits': teacher_logits}
        print(f"[GeneticSparsity] Teacher output ready (shape={logits_shape})", flush=True)

        # ── 6. Create and run AirGeneticSparsity ──
        # Synchronize the seed so that all ranks create the same population in an FSDP environment
        sync_seed = self.seed + step
        random.seed(sync_seed)
        torch.manual_seed(sync_seed)
        torch.cuda.manual_seed(sync_seed)
        print(f"[GeneticSparsity] Synchronized seed={sync_seed} across ranks", flush=True)

        #genetic = AirGeneticSparsity(
        #    model=state.model,
        #    teacher_model=self.teacher_model,
        #    target_params=self.target_params,
        #    sim_func=sim_fc,
        #)
        if self.fixed_head:
            genetic = AirGeneticSparsityConstraint(
                    model=state.model,
                    teacher_model=self.teacher_model,
                    target_params=self.target_params,
                    sim_func=sim_fc,
                    )
        else:
            print("Use Not Fixed Head GA")
            genetic = AirGeneticSparsityConstraintHead(
                    model=state.model,
                    teacher_model=self.teacher_model,
                    target_params=self.target_params,
                    sim_func=sim_fc,
                    )
            
        genetic.origin_module_cfg = deepcopy(self.origin_module_cfg)
        genetic.range_ratio = self.range_ratio
        # Pass the pre-computed teacher output to genetic (no teacher forward in apply_config)
        genetic.target_output = teacher_output

        init_cfg = deepcopy(self.init_cfg)
        start_time = time.time()

        print(f"[GeneticSparsity] Initializing population (size={self.population_size})...", flush=True)
        genetic.intialize_population(self.population_size, init_cfg)

        print(f"[GeneticSparsity] Running evolution...", flush=True)
        # with FSDP.summon_full_params(state.model, writeback=True, recurse=True):
        genetic_output = genetic.evolution(data, self.population_size)

        ## added code to check whether it works without summon
        # for mask_name in l0_module.masks:
        #     z = l0_module.masks[mask_name].z_loga
        #     t = torch.zeros_like(z)
        #     torch.distributed.all_reduce(t := z.clone(), op=torch.distributed.ReduceOp.MAX)
        #     torch.distributed.all_reduce(z_min := z.clone(), op=torch.distributed.ReduceOp.MIN)
        #     assert torch.allclose(t, z_min), f"{mask_name} diverged across ranks"
        elapsed = time.time() - start_time

        best_indiv = genetic_output["best_indiv"]
        # If best_indiv differs slightly per rank due to FP non-determinism, similarity→current_interval
        # becomes misaligned, check_start_ga() turns True at different steps per rank → GA collective desync →
        # an NCCL deadlock occurs. Broadcast rank 0's best_indiv to all ranks to keep lockstep.
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            obj_list = [best_indiv]
            torch.distributed.broadcast_object_list(obj_list, src=0)
            best_indiv = obj_list[0]
        print(f"[GeneticSparsity] Evolution completed in {elapsed:.1f}s", flush=True)
        print(f"[GeneticSparsity] Best individual: {best_indiv}", flush=True)

        # ── 6. Permanently apply the best config to the actual mask ──
        int_best = {k: int(v, 2) if isinstance(v, str) else v for k, v in best_indiv.items()}
        # best_config = genetic.infer_missing_llama_config(self.target_params, **int_best) ## air: inter_missing seems unnecessary since it is already complete
        best_config = int_best

        if best_config is None:
            print("[GeneticSparsity] Could not infer valid config from best individual, skipping apply.", flush=True)
            state.model.train()
            return

        self.best_config = best_config
        print(f"[GeneticSparsity] Applying best config to masks: {best_config}", flush=True)

        best_ns = SimpleNamespace(**best_config)

        # Update target_model_info
        target_module_cfg = l0_module.set_model_info(best_ns, 3)
        l0_module.target_model_info = target_module_cfg

        # Permanently apply target_mask_size of each mask
        change_target_mask_size(l0_module, best_ns)

        # Directly manipulate z_loga to force the expected sparsity to the value found by GA
        with FSDP.summon_full_params(state.model, writeback=True, recurse=True):
            if self.pruning_cfg.get("is_gradual", True):
                self._force_gradually_mask_sparsity(l0_module)

        # Also update target_sparsity
        for mask_name in l0_module.masks:
            mask = l0_module.masks[mask_name]
            if mask.target_mask_size is not None:
                new_sparsity = 1.0 - mask.target_mask_size / mask.mask_size
                mask.target_sparsity = new_sparsity
                score, actual_sparsity = mask.calculate_expected_score_sparsity()
                print(f"  {mask_name}: target_mask_size={mask.target_mask_size}, "
                      f"target_sparsity={new_sparsity:.4f}, "
                      f"expected_sparsity={actual_sparsity.mean().item():.4f}", flush=True)

        with FSDP.summon_full_params(state.model, writeback=True, recurse=True):
            for k in l0_module.masks:
                score, sparsity = l0_module.masks[k].calculate_expected_score_sparsity()
                print(f"after force  {k}: {sparsity.mean().item():.4f}")

        # Recompute the prunable target model size and the overall target sparsity
        l0_module.prunable_target_model_size = l0_module.calculate_prunable_model_size(l0_module.target_model_info)
        # l0_module.target_sparsity = 1 - l0_module.prunable_target_model_size / l0_module.prunable_model_size
        print(f"  Overall target_sparsity: {l0_module.target_sparsity:.4f}", flush=True)

        # self.applied = True ## must be applied when running only once

        # ── 7. Record logs ──
        log_metrics = {
            "metrics/eval/genetic_sparsity/step": step,
            "metrics/eval/genetic_sparsity/elapsed_sec": elapsed,
            "metrics/eval/genetic_sparsity/target_sparsity": l0_module.target_sparsity,
            "metrics/eval/genetic_sparsity/finish_generation": genetic.finish_generation,
            "metrics/eval/genetic_sparsity/curr_ga_step_size": self.curr_ga_step_size,
        }
        for k, v in best_config.items():
            log_metrics[f"metrics/eval/genetic_sparsity/best_{k}"] = v
        logger.log_metrics(log_metrics)

        # ── 8. Save results ──
        # if self.save_folder:
        #     save_dir = self.save_folder.replace("{run_name}", getattr(state, "run_name", "default"))
        #     os.makedirs(save_dir, exist_ok=True)
        #     save_path = os.path.join(save_dir, f"genetic_best_config_s{step}.pt")
        #     torch.save({
        #         "best_indiv": best_indiv,
        #         "best_config": best_config,
        #         "target_sparsity": l0_module.target_sparsity,
        #     }, save_path)
        #     print(f"[GeneticSparsity] Saved result to {save_path}", flush=True)

        # Restore training mode
        state.model.train()

        print(f"[GeneticSparsity] ✅ Mask sparsity applied successfully!", flush=True)
        print(f"{'='*60}\n", flush=True)
        if self.is_test:
            exit()

    def _force_mask_sparsity(self, l0_module):
        """
        Directly manipulates z_loga to force the expected sparsity to match target_mask_size.
        - Bottom num_zeros entries → set to PRUNE_VALUE (-4.0) (pruning)
        - Top target_mask_size entries → set to KEEP_VALUE (+4.0) (keeping)
        Both sides must be set so the mask changes exactly regardless of the previous training state.
        """
        PRUNE_VALUE = -4.0
        KEEP_VALUE = 4.0

        with torch.no_grad():
            for mask_name in l0_module.masks:
                mask = l0_module.masks[mask_name]
                print(f"  [force_sparsity] {mask_name}: target={mask.target_mask_size}, size={mask.mask_size}", flush=True)
                if mask.target_mask_size is None:
                    continue

                num_zeros = mask.mask_size - mask.target_mask_size
                num_keep = mask.target_mask_size
                if num_zeros <= 0:
                    # No pruning needed: keep everything
                    mask.z_loga.data.fill_(KEEP_VALUE)
                    print(f"  [force_sparsity] {mask_name}: no pruning needed "
                          f"(target={mask.target_mask_size}, size={mask.mask_size}), "
                          f"all set to {KEEP_VALUE}", flush=True)
                    continue

                if mask.z_loga.ndim == 1:
                    # 1D mask (layer, hidden)
                    # Top keep entries → KEEP_VALUE
                    print("keep", num_keep, "prune", num_zeros, mask.z_loga.data.shape, flush=True)
                    _, keep_indices = torch.topk(mask.z_loga, k=num_keep, largest=True)
                    # Bottom prune entries → PRUNE_VALUE
                    _, prune_indices = torch.topk(mask.z_loga, k=num_zeros, largest=False)
                    mask.z_loga.data[keep_indices] = KEEP_VALUE
                    mask.z_loga.data[prune_indices] = PRUNE_VALUE
                    print(f"  [force_sparsity] {mask_name}: "
                          f"keep {num_keep} → {KEEP_VALUE}, "
                          f"prune {num_zeros} → {PRUNE_VALUE} "
                          f"(total={mask.mask_size})", flush=True)
                else:
                    # 2D mask (head, intermediate) - handled per row (layer)
                    for i in range(mask.z_loga.shape[0]):
                        _, keep_indices = torch.topk(mask.z_loga[i], k=num_keep, largest=True)
                        _, prune_indices = torch.topk(mask.z_loga[i], k=num_zeros, largest=False)
                        mask.z_loga.data[i, keep_indices] = KEEP_VALUE
                        mask.z_loga.data[i, prune_indices] = PRUNE_VALUE
                    print(f"  [force_sparsity] {mask_name}: "
                          f"keep {num_keep} → {KEEP_VALUE}, "
                          f"prune {num_zeros} → {PRUNE_VALUE} "
                          f"per row (rows={mask.z_loga.shape[0]})", flush=True)

    def _get_sample_batch(self, state: State):
        """Generate sample data using the recent batch of state (broadcast rank 0's data to all ranks)"""
        try:
            data = {k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in state.batch.items()}
            # Slice the batch size
            data = {k: v[:self.sample_batch_size] if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()}
            model_device = next(state.model.parameters()).device
            data = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()}
            # Set pruned_steps
            batch_size = data['input_ids'].shape[0]
            data['pruned_steps'] = torch.zeros(batch_size, dtype=torch.long, device=model_device)

            # FSDP: broadcast rank 0's data to all ranks so the same input is used
            # (different input → different score → selection diverges → forward count mismatch → avoids deadlock)
            if dist.is_initialized():
                for k in data:
                    if isinstance(data[k], torch.Tensor):
                        torch.distributed.broadcast(data[k], src=0)

            return data
        except Exception as e:
            print(f"[GeneticSparsity] Error getting sample batch: {e}", flush=True)
            return None

    def _force_gradually_mask_sparsity(self, l0_module):
        """
        Directly manipulates z_loga to force the expected sparsity to match target_mask_size.
        - Bottom num_zeros entries → subtract step_size (pruning)
        - Top target_mask_size entries → add step_size (keeping)
        Both sides must be set so the mask changes exactly regardless of the previous training state.
        """
        #PRUNE_VALUE = -4.0
        #KEEP_VALUE = 4.7
        KEEP_VALUE = math.log(1e4)
        max_step_value = math.log(1e4)
        
        
        if self.is_linear:
            #step_size = 4.5 * self.current_step / self.total_step
            step_size = max_step_value * self.current_step / self.total_step
            
            # step_size = 2 # test
            print(f"step_size: {step_size}")
        else:
            step_size = max_step_value * get_normal_dist_value(self.current_step, self.total_step, peak_step=self.lag_warm_up_step)
        #if self.ga_step_func == "beta":
        #    alpha = 1.5
        #    beta = find_beta_param_for_peak(alpha=alpha, target_peak_step=640, total_steps=self.total_step)
        #    step_size = max_step_value * get_beta_dist_value(self.current_step, self.total_step, alpha=alpha, beta_param=beta)

        #step_size = max_step_value * self.current_step / self.total_step
        #step_size = self.get_ga_step_size(self.current_step, max_step_value, self.total_step)
        #step_size = max_step_value * get_normal_dist_value(self.current_step, self.total_step, peak_step=self.lag_warm_up_step)
        self.curr_ga_step_size = step_size

        with torch.no_grad():
            for mask_name in l0_module.masks:
                mask = l0_module.masks[mask_name]
                print(f"  [force_sparsity] {mask_name}: target={mask.target_mask_size}, size={mask.mask_size}", flush=True)
                if mask.target_mask_size is None:
                    continue

                num_zeros = mask.mask_size - mask.target_mask_size
                num_keep = mask.target_mask_size

                if mask.z_loga.ndim == 1:
                    # 1D mask (layer, hidden)
                    # Top keep entries → KEEP_VALUE
                    print("keep", num_keep, "prune", num_zeros, mask.z_loga.data.shape, flush=True)
                    _, keep_indices = torch.topk(mask.z_loga, k=num_keep, largest=True)
                    # Bottom prune entries → PRUNE_VALUE
                    _, prune_indices = torch.topk(mask.z_loga, k=num_zeros, largest=False)
                    #mask.z_loga.data[keep_indices] += (step_size)
                    #mask.z_loga.data[prune_indices] -= (step_size)
                    mask.z_loga.data[keep_indices] = torch.clamp(mask.z_loga.data[keep_indices] + step_size, max=KEEP_VALUE)
                    mask.z_loga.data[prune_indices] = torch.clamp(mask.z_loga.data[prune_indices] - step_size, min=-KEEP_VALUE)
                    print(f"  [force_sparsity] {mask_name}: "
                          f"keep {num_keep} → add {step_size}, "
                          f"prune {num_zeros} → sub {step_size} "
                          f"(total={mask.mask_size})", flush=True)
                else:
                    # 2D mask (head, intermediate) - handled per row (layer)
                    for i in range(mask.z_loga.shape[0]):
                        _, keep_indices = torch.topk(mask.z_loga[i], k=num_keep, largest=True)
                        _, prune_indices = torch.topk(mask.z_loga[i], k=num_zeros, largest=False)
                        #mask.z_loga.data[i, keep_indices] += (step_size)
                        #mask.z_loga.data[i, prune_indices] -= (step_size)
                        mask.z_loga.data[i, keep_indices] = torch.clamp(mask.z_loga.data[i, keep_indices] + step_size, max=KEEP_VALUE)
                        mask.z_loga.data[i, prune_indices] = torch.clamp(mask.z_loga.data[i, prune_indices] - step_size, min=-KEEP_VALUE)
                    print(f"  [force_sparsity] {mask_name}: "
                          f"keep {num_keep} → add {step_size}, "
                          f"prune {num_zeros} → sub {step_size} "
                          f"per row (rows={mask.z_loga.shape[0]})", flush=True)
                    print(f" {mask_name} mask : {mask.z_loga}, {mask.z_loga.shape}", flush=True)
    def get_ga_step_size(self, curr_step: int, max_step_size: float, max_train_step: int = 3200) -> float:
        """
        Applies 1-Cycle cosine scheduling and returns the GA step_size for the current step.
    
        Args:
            curr_step (int): Current training step
            max_step_size (float): Maximum mask change (step_size)
            max_train_step (int): Total training steps (default 3200)
        
        Returns:
            float: The computed current step_size
        """
        # Clip so the current step does not exceed the maximum step (safeguard)
        curr_step = min(curr_step, max_train_step)
    
        # Compute progress (0.0 ~ 1.0)
        progress = curr_step / max_train_step
    
        # 1-Cycle cosine formula: (1 - cos(2 * pi * progress)) / 2
        # A curve that is 0 at progress 0, 1 at 0.5, and 0 at 1.0
        cosine_term = 1 - math.cos(2 * math.pi * progress)
        step_size = max_step_size * (cosine_term / 2)
    
        return step_size



class GeneticEvalFitCallback(GeneticSparsityCallback):
    def __init__(self, ga_eval_data_path=None, n=1,seed=42, ga_eval_datasets=None, **kwargs):
        super().__init__(**kwargs)
        self.ga_eval_dataset = GAEvalDataset(data_dir=ga_eval_data_path, n=n, seed=seed, datasets=ga_eval_datasets)
        self.data_n = n
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs.get("teacher_model_name", "meta-llama/Llama-2-7b-hf"))
        self.is_test = False
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_layer_wise = False

    def eval_end(self, state: State, logger: Logger) -> None:
        step = state.timestamp.batch.value
        l0_module = state.model.model.l0_module
        self.current_step = step

        if l0_module is None:
            return

        # skip if before start_step
        if step < self.start_step:
            print(f"[GeneticSparsity] step={step} < start_step={self.start_step}, skipping.", flush=True)
            return

        # skip if already applied (run only once)
        if self.applied:
            print(f"[GeneticSparsity] step={step}: already applied, skipping.", flush=True)
            return

        print(f"\n{'='*60}", flush=True)
        print(f"[GeneticSparsity] step={step}: Starting genetic sparsity search", flush=True)
        print(f"{'='*60}", flush=True)

        # ── 1. Log the current mask state ──
        with FSDP.summon_full_params(l0_module, writeback=False):
            print("[GeneticSparsity] Current mask sparsity:", flush=True)
            for k in l0_module.masks:
                score, sparsity = l0_module.masks[k].calculate_expected_score_sparsity()
                print(f"  {k}: {sparsity.mean().item():.4f}", flush=True)


        # ── 3. Prepare a sample batch from the eval data ──
        data = self.ga_eval_dataset.get_samples_flat()
        input_answer_data_list = [sample["input_answer"] for sample in data]
        tokenized_data_list = self.tokenizer(input_answer_data_list, return_tensors="pt", padding=True, truncation=True)
        if data is None:
            print("[GeneticSparsity] Failed to get sample batch, skipping.", flush=True)
            return

        # ── 4. Set up sim_func ──
        sim_fc = deepcopy(self.sim_func_cfg)
        if "topk_rate" in sim_fc and isinstance(sim_fc["topk_rate"], float) and sim_fc["topk_rate"] < 1.0:
            sim_fc["topk_rate"] = int(32000 * sim_fc["topk_rate"])

        # ── 5. Pre-compute the Teacher output (forward on rank 0 only, then broadcast) ──
        model_param = next(state.model.parameters())
        model_device = model_param.device
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        # Reduce the sequence length (slice all 2D tensors together)
        # max_seq_len = 128
        # for k in data:
        #     if isinstance(data[k], torch.Tensor) and data[k].ndim == 2:
        #         data[k] = data[k][:, :max_seq_len]
        data = tokenized_data_list
        data['pruned_steps'] = torch.tensor([0]*len(data['input_ids']), device=model_device)
        data['labels'] = data['input_ids'].clone()
        # print(data[0])

        # Move sample to the exact device of the model to avoid moving the FSDP model to CPU
        sample_batch = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        
        # print(f"[GeneticSparsity] Computing teacher output (rank 0 only, seq_len={max_seq_len})...", flush=True)
        # First check the output shape with one student model forward pass
        state.model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=model_param.dtype):
            dummy_output = state.model(sample_batch)
        logits_shape = dummy_output['logits'].shape
        logits_dtype = dummy_output['logits'].dtype
        logits_device = dummy_output['logits'].device

        # ── 6. Create and run AirGeneticSparsity ──
        # Synchronize the seed so that all ranks create the same population in an FSDP environment
        sync_seed = self.seed + step
        random.seed(sync_seed)
        torch.manual_seed(sync_seed)
        torch.cuda.manual_seed(sync_seed)
        print(f"[GeneticSparsity] Synchronized seed={sync_seed} across ranks", flush=True)

        #genetic = AirGeneticEvalLoss(
        #    model=state.model,
        #    teacher_model=self.teacher_model,
        #    target_params=self.target_params,
        #)
        if self.use_layer_wise:
            print("[Callback] use layer wise ga")
            genetic = AirGeneticSparsityConstraintLayerWise(
                    model=state.model,
                    teacher_model=self.teacher_model,
                    target_params=self.target_params,
                    sim_func=sim_fc,
                    is_logit=False,
                    num_layer_groups=10,   # optional: default 10
                    layer_step=2,          # optional: default 2
                      # max_layer=...,       # optional: if unspecified, origin_module_cfg['n_layers'] is used
                    ) 

        else:
            genetic = AirGeneticSparsityConstraint(
                    model=state.model,
                    teacher_model=self.teacher_model,
                    target_params=self.target_params,
                    sim_func=sim_fc,
                    is_logit=False
                    )
        genetic.origin_module_cfg = deepcopy(self.origin_module_cfg)
        genetic.range_ratio = self.range_ratio
        genetic.is_logit = False


        init_cfg = deepcopy(self.init_cfg)
        start_time = time.time()

        print(f"[GeneticSparsity] Initializing population (size={self.population_size})...", flush=True)
        genetic.intialize_population(self.population_size, init_cfg)

        print(f"[GeneticSparsity] Running evolution...", flush=True)
        genetic_output = genetic.evolution(data, self.population_size)
        elapsed = time.time() - start_time

        best_indiv = genetic_output["best_indiv"]
        ## test
        # best_indiv = {'d_model': '11100000000', 'n_heads': '1110', 'n_layers': '1100', 'intermediate_size': '10110001100', 'total_params': '10101011111000111111100000000', 'vocab_size': '111110100000000'}
        print(f"[GeneticSparsity] Evolution completed in {elapsed:.1f}s", flush=True)
        print(f"[GeneticSparsity] Best individual: {best_indiv}", flush=True)

        # ── 6. Permanently apply the best config to the actual mask ──
        int_best = {k: int(v, 2) if isinstance(v, str) else v for k, v in best_indiv.items()}
        # best_config = genetic.infer_missing_llama_config(self.target_params, **int_best) ## air: inter_missing seems unnecessary since it is already complete
        best_config = int_best

        if best_config is None:
            print("[GeneticSparsity] Could not infer valid config from best individual, skipping apply.", flush=True)
            state.model.train()
            return

        self.best_config = best_config
        print(f"[GeneticSparsity] Applying best config to masks: {best_config}", flush=True)

        best_ns = SimpleNamespace(**best_config)

        # Update target_model_info
        target_module_cfg = l0_module.set_model_info(best_ns, 3)
        l0_module.target_model_info = target_module_cfg

        # Permanently apply target_mask_size of each mask
        change_target_mask_size(l0_module, best_ns)

        # Directly manipulate z_loga to force the expected sparsity to the value found by GA
        with FSDP.summon_full_params(state.model, writeback=True, recurse=True):
            if self.pruning_cfg.get("is_gradual", True):
                self._force_gradually_mask_sparsity(l0_module)
            else:
                self._force_mask_sparsity(l0_module)
        # self._force_mask_sparsity(l0_module)

        # Also update target_sparsity
        for mask_name in l0_module.masks:
            mask = l0_module.masks[mask_name]
            if mask.target_mask_size is not None:
                new_sparsity = 1.0 - mask.target_mask_size / mask.mask_size
                mask.target_sparsity = new_sparsity
                score, actual_sparsity = mask.calculate_expected_score_sparsity()
                print(f"  {mask_name}: target_mask_size={mask.target_mask_size}, "
                      f"target_sparsity={new_sparsity:.4f}, "
                      f"expected_sparsity={actual_sparsity.mean().item():.4f}", flush=True)

        # Recompute the prunable target model size and the overall target sparsity
        l0_module.prunable_target_model_size = l0_module.calculate_prunable_model_size(l0_module.target_model_info)
        # l0_module.target_sparsity = 1 - l0_module.prunable_target_model_size / l0_module.prunable_model_size
        print(f"  Overall target_sparsity: {l0_module.target_sparsity:.4f}", flush=True)

        # self.applied = True ## must be applied when running only once

        # ── 7. Record logs ──
        log_metrics = {
            "metrics/eval/genetic_sparsity/step": step,
            "metrics/eval/genetic_sparsity/elapsed_sec": elapsed,
            "metrics/eval/genetic_sparsity/target_sparsity": l0_module.target_sparsity,
            "metrics/eval/genetic_sparsity/finish_generation": genetic.finish_generation,
        }
        for k, v in best_config.items():
            log_metrics[f"metrics/eval/genetic_sparsity/best_{k}"] = v
        logger.log_metrics(log_metrics)

        # ── 8. Save results ──
        # if self.save_folder:
        #     save_dir = self.save_folder.replace("{run_name}", getattr(state, "run_name", "default"))
        #     os.makedirs(save_dir, exist_ok=True)
        #     save_path = os.path.join(save_dir, f"genetic_best_config_s{step}.pt")
        #     torch.save({
        #         "best_indiv": best_indiv,
        #         "best_config": best_config,
        #         "target_sparsity": l0_module.target_sparsity,
        #     }, save_path)
        #     print(f"[GeneticSparsity] Saved result to {save_path}", flush=True)

        # Restore training mode
        state.model.train()

        print(f"[GeneticSparsity] ✅ Mask sparsity applied successfully!", flush=True)
        print(f"{'='*60}\n", flush=True)
        if self.is_test:
            print(f"[GeneticSparsity] Exiting program after applying best config. {best_config}", flush=True)
            exit()

class GeneticDynamicIntervalCallback(GeneticSparsityCallback):
    """
    A Callback that dynamically adjusts the GA execution interval.
    - If the GA result (best_config) differs a lot from the previous result → shorten the interval (run more often)
    - If the GA result is similar to the previous result → lengthen the interval (run less often)
    """
    def __init__(
        self,
        base_interval: int = 400,
        min_interval: int = 100,
        max_interval: int = 1600,
        increase_factor: float = 1.5,
        decrease_factor: float = 0.5,
        similarity_threshold_high: float = 0.90,
        similarity_threshold_low: float = 0.80,
        compare_keys: List[str] = None,
        warmup_ga_runs: int = 1,
        warmup_end_step: int = 640,
        start_ga_at_peak = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.base_interval = int(base_interval)
        self.min_interval = int(min_interval)
        self.max_interval = int(max_interval)
        self.current_interval = int(base_interval)
        self.increase_factor = float(increase_factor)
        self.decrease_factor = float(decrease_factor)
        self.similarity_threshold_high = float(similarity_threshold_high)
        self.similarity_threshold_low = float(similarity_threshold_low)
        self.compare_keys = compare_keys or ["d_model", "n_heads", "n_layers", "intermediate_size"]
        # warmup_ga_runs: run GA with a fixed base_interval up to this count, then adjust dynamically
        self.warmup_ga_runs = int(warmup_ga_runs)
        self.peak_step = warmup_end_step
        print("target :",self.target_params)
        print("origin :", self.origin_module_cfg)

        self.last_ga_step: int = -1
        self.last_similarity: float = None
        self.ga_run_count: int = 0
        self.start_ga_at_peak = start_ga_at_peak
        # Override the parent's self.applied=False to True → so Composer's normal eval_end call does not trigger GA by itself
        self.applied = True

    def check_start_ga(self) -> bool:
        """Decide whether to run GA now, based on the dynamic interval.
        - Before start_step: False
        - First run (last_ga_step < 0): True if at or past start_step
        - Otherwise: True if current step - last GA step >= current_interval
        """
        if self.current_step == self.total_step:
            return True
        if self.current_step == self.peak_step and self.start_ga_at_peak:
            return True
        if self.current_step < self.start_step:
            return False
        if self.last_ga_step < 0:
            return True
        return (self.current_step - self.last_ga_step) >= self.current_interval

    def _calculate_config_similarity(self, prev_config: dict, curr_config: dict) -> float:
        """Compute the similarity [0,1] between the previous best_config and the current best_config.
        One minus the mean of the relative L1 difference of each key.
        1.0 = identical, 0.0 = differs by 100% or more on average.
        """
        if prev_config is None or curr_config is None:
            return 0.0

        relative_diffs = []
        for key in self.compare_keys:
            if key not in prev_config or key not in curr_config:
                continue
            prev_val = float(prev_config[key])
            curr_val = float(curr_config[key])
            if prev_val == 0:
                continue
            relative_diffs.append(abs(curr_val - prev_val) / prev_val)

        if not relative_diffs:
            return 1.0

        avg_diff = sum(relative_diffs) / len(relative_diffs)
        return max(0.0, 1.0 - avg_diff)

    def _update_interval(self, similarity: float) -> None:
        """Adjust the interval based on similarity.
        - similarity >= high  : results are similar → interval *= increase_factor (cap: max_interval)
        - similarity <  low   : results differ → interval *= decrease_factor (floor: min_interval)
        - otherwise           : keep
        """
        old = self.current_interval
        param_a = 0.2
        ## 0615 dynamic threshold decision part
        if self.ga_step_func == "gaussian":
            low_threshold = self.similarity_threshold_low - param_a *get_normal_dist_value(self.current_step, total_steps=3200, peak_step=self.lag_warm_up_step) 
        elif self.ga_step_func == "beta":
            alpha = 1.5
            beta = find_beta_param_for_peak(alpha=alpha, target_peak_step=640, total_steps=self.total_step)
            beta_value = get_beta_dist_value(self.current_step, self.total_step, alpha=alpha, beta_param=beta)
            low_threshold = self.similarity_threshold_low - param_a * beta_value 
        else:
            low_threshold = self.similarity_threshold_low

        if self.is_linear:
            low_threshold = self.similarity_threshold_low - self.current_step / self.total_step


        ### 0615 end
        if similarity >= self.similarity_threshold_high:
            self.current_interval = min(
                max(int(round(self.current_interval * self.increase_factor)), old + 1),
                self.max_interval,
            )
            reason = f"similarity={similarity:.4f} >= {self.similarity_threshold_high} (increase)"
        elif similarity < low_threshold:
            self.current_interval = max(
                min(int(round(self.current_interval * self.decrease_factor)), old - 1),
                self.min_interval,
            )
            reason = f"similarity={similarity:.4f} < {low_threshold} (decrease)"
        else:
            reason = f"similarity={similarity:.4f} (in dead-zone, unchanged)"
        print(
            f"[DynamicInterval] {reason}: interval {old} → {self.current_interval} "
            f"(min={self.min_interval}, max={self.max_interval})",
            flush=True,
        )

    def after_train_batch(self, state: State, logger: Logger) -> None:
        self.current_step = state.timestamp.batch.value
        # Ensure True at every batch so that Composer's normal eval_end call does not trigger GA
        self.applied = True

        if not self.check_start_ga() :
            return
        if self.ga_run_count >= 8: ## specify the maximum number of runs
            return

        # Snapshot of the previous best_config before running GA
        prev_config = deepcopy(self.best_config) if self.best_config is not None else None

        # Bypass the applied check inside eval_end to force GA to run
        self.applied = False
        self.eval_end(state, logger)
        self.applied = True

        self.last_ga_step = self.current_step
        self.ga_run_count += 1
        in_warmup = self.ga_run_count <= self.warmup_ga_runs

        # Adjust the interval after comparing GA results
        log_metrics = {
            "metrics/dynamic_interval/current_interval": self.current_interval,
            "metrics/dynamic_interval/last_ga_step": self.last_ga_step,
            "metrics/dynamic_interval/ga_run_count": self.ga_run_count,
        }
        if in_warmup:
            # Warmup period: only compute and log similarity, and fix the interval to base
            self.current_interval = self.base_interval
            if prev_config is not None and self.best_config is not None:
                similarity = self._calculate_config_similarity(prev_config, self.best_config)
                self.last_similarity = similarity
                log_metrics["metrics/dynamic_interval/similarity"] = similarity
            print(
                f"[DynamicInterval] Warmup GA run {self.ga_run_count}/{self.warmup_ga_runs} "
                f"at step={self.current_step}, fixed interval={self.base_interval}",
                flush=True,
            )
            log_metrics["metrics/dynamic_interval/current_interval"] = self.current_interval
        elif prev_config is not None and self.best_config is not None:
            similarity = self._calculate_config_similarity(prev_config, self.best_config)
            self.last_similarity = similarity
            self._update_interval(similarity)
            log_metrics["metrics/dynamic_interval/similarity"] = similarity
            # Re-apply the value after the interval adjustment
            log_metrics["metrics/dynamic_interval/current_interval"] = self.current_interval
        else:
            print(
                f"[DynamicInterval] First GA run at step={self.current_step}, "
                f"interval={self.current_interval} (no prior config to compare)",
                flush=True,
            )

        logger.log_metrics(log_metrics)


class GeneticDynamicIntervalAtEvalCallback(GeneticDynamicIntervalCallback):
    def after_train_batch(self, state: State, logger: Logger) -> None:
        return 

    def eval_end(self, state, logger):
        self.current_step = state.timestamp.batch.value
        # Ensure True at every batch so that Composer's normal eval_end call does not trigger GA

        if not self.check_start_ga() :
            return

        # Snapshot of the previous best_config before running GA
        prev_config = deepcopy(self.best_config) if self.best_config is not None else None

        # Bypass the applied check inside eval_end to force GA to run
        self.applied = False
        super().eval_end(state, logger)

        self.last_ga_step = self.current_step
        self.ga_run_count += 1
        in_warmup = self.ga_run_count <= self.warmup_ga_runs

        # Adjust the interval after comparing GA results
        log_metrics = {
            "metrics/dynamic_interval/current_interval": self.current_interval,
            "metrics/dynamic_interval/last_ga_step": self.last_ga_step,
            "metrics/dynamic_interval/ga_run_count": self.ga_run_count,
        }
        if in_warmup:
            # Warmup period: only compute and log similarity, and fix the interval to base
            self.current_interval = self.base_interval
            if prev_config is not None and self.best_config is not None:
                similarity = self._calculate_config_similarity(prev_config, self.best_config)
                self.last_similarity = similarity
                log_metrics["metrics/dynamic_interval/similarity"] = similarity
            print(
                f"[DynamicInterval] Warmup GA run {self.ga_run_count}/{self.warmup_ga_runs} "
                f"at step={self.current_step}, fixed interval={self.base_interval}",
                flush=True,
            )
            log_metrics["metrics/dynamic_interval/current_interval"] = self.current_interval
        elif prev_config is not None and self.best_config is not None:
            similarity = self._calculate_config_similarity(prev_config, self.best_config)
            self.last_similarity = similarity
            self._update_interval(similarity)
            log_metrics["metrics/dynamic_interval/similarity"] = similarity
            # Re-apply the value after the interval adjustment
            log_metrics["metrics/dynamic_interval/current_interval"] = self.current_interval
        else:
            print(
                f"[DynamicInterval] First GA run at step={self.current_step}, "
                f"interval={self.current_interval} (no prior config to compare)",
                flush=True,
            )

        logger.log_metrics(log_metrics)

