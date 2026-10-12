# ElasticPruner

A project for finding the optimal architecture of an LLM.

A genetic algorithm (GA) searches for combinations of the number of layers, number of
attention heads, FFN intermediate dimension, and hidden dimension within a target
parameter budget. The discovered structure is then realized through L0 mask training
to structurally prune LLaMA-2 family models.

This is a derivative work based on
[Sheared LLaMA (princeton-nlp/LLM-Shearing)](https://github.com/princeton-nlp/LLM-Shearing).

## License

| File | Description |
|---|---|
| [`LICENSE`](LICENSE) | MIT (entire repository) |
| [`LICENSE-APACHE`](LICENSE-APACHE) | Apache-2.0 (applies to some files) |
| [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | **Per-file origin and license table** |

Some files retain the Apache-2.0 headers from MosaicML Examples / MosaicML Streaming
as-is. These headers have not been modified.

---


## 1. Installation

```bash
bash install_pip.sh
```

The benchmark step (Step 5) uses a separate environment — see [Step 5](#step-5-benchmarks).

## 2. Data Preparation

Place RedPajama (or data in the same format) in per-domain folders, with jsonl bundles
inside each domain, and convert it to MDS format. Seven domains are used:
`cc, github, book, stackexchange, wiki, arxiv, c4-rp`

> **Required external file:** `tokenize_single_file.py` reads `tokenizer.model` from the
> working directory (the LLaMA-2 SentencePiece tokenizer). It is not included in this
> repository, so you must place it in `elasticpruner/data/` yourself.

- The data used for pruning and training in this research is the data shared by the authors of Sheared LLaMA (https://github.com/princeton-nlp/LLM-Shearing/tree/main).

```bash
cd elasticpruner/data

# 1) Generate the jsonl file list -> jsonl_list.txt
python get_all_jsonl.py {data_root}

# 2) Tokenize everything (saved as numpy)
bash tokenize_all_files.sh

# 3) Sample into eval / for_prune / for_ft and encode as MDS
bash sample_all_domains.sh
```

Resulting directory structure:

```
eval/        Per-domain validation data + eval_merge (all domains merged, required)
for_prune/   For Steps 1–2 (mask training)
for_ft/      For Step 3 (after train)
```

`eval_merge` is created with `merge_data.py` (e.g., `elasticpruner/data/eval_merge.sh`):

```bash
python -m elasticpruner.data.merge_data \
    --input_dir /path/to/mds_data/eval \
    --output_dir /path/to/mds_data/for_ft \
    --output_split eval_merge \
    --split_names arxiv c4-rp cc github stackexchange book wiki
```

The dataset for GA fitness evaluation (`genetic.ga_eval_data_path`) is downloaded separately:

```bash
python ga_eval_data/download_datasets.py --save_dir /path/to/ga_eval_data
```

## 3. Preparing the Original Model

Convert the HuggingFace model to the Composer key format
(see `elasticpruner/model_preparation.sh`).

```bash
python3 -m elasticpruner.utils.composer_to_hf save_hf_to_composer \
    meta-llama/Llama-2-7b-hf  /path/to/models/llama-2-7b-composer/state_dict.pt
```

---

# Pipeline

| Step | Script | Entry point |
|---|---|---|
| 0 (optional) | `elasticpruner/scripts/make_teacher_logit_for_ga.sh` | `elasticpruner/make_teacher_logit_for_ga.py` |
| 1 — Mask training | `elasticpruner/scripts/pruning_only_mask_7b_genetic.sh` | `elasticpruner/train_only_mask_w_genetic.py` |
| 2 — Pruning with the mask | `elasticpruner/utils/post_prune.sh` | `elasticpruner.utils.post_pruning_air` |
| 3 — After train | `elasticpruner/scripts/pruning_train_ours_7b.sh` | `elasticpruner/train_om_after_pruned.py` |
| 4 — HF conversion | `elasticpruner/scripts/composer_to_hf_llama.sh` | `elasticpruner.utils.composer_to_hf` |
| 5 — Benchmarks | `evaluation/run_*.sh`, `evaluation/auto_bench_v3_seed.py` | lm-evaluation-harness |

Variants for the 1.3B target: `pruning_only_mask_1_3b_genetic.sh`, `pruning_train_ours_1_3b.sh`

## Step 0: Pre-generate GA teacher logits (optional)

When computing GA fitness as KL divergence (against teacher logits), a pool of logits is
generated in advance to avoid running the teacher forward pass at every step. This is
required if you use `genetic.is_logit=true` in Step 1.

```bash
# Needs to be replaced: PROJ_DIR, DATA_DIR, OUTPUT_DIR
bash elasticpruner/scripts/make_teacher_logit_for_ga.sh
```

Main arguments (`teacher_logit.*`): `num_samples=32`, `sample_seq_len=4096`,
`compute_dtype=float16`. Output is `OUTPUT_DIR/llama2-7b-teacher_logits_*.pt`.

## Step 1: Pruning mask training

While training the L0 masks (head / intermediate / layer / hidden), the GA searches for a
structure within the target parameter budget (`genetic.target_params`).

```bash
# Needs to be replaced: PROJ_DIR, DATA_DIR, OUTPUT_DIR, MODEL_PATH,
#                       ga_eval_data_path, genetic.teacher_logit_path
bash elasticpruner/scripts/pruning_only_mask_7b_genetic.sh
```

Default configuration for 7B → 2.7B:

| Item | Value |
|---|---|
| Target structure | `d_model=2560, n_heads=20, n_layers=32, intermediate_size=6912` |
| `genetic.target_params` | `2745000000` |
| Training budget | `max_duration=3200ba`, `max_seq_len=4096` |
| Domain proportions | `[0.67, 0.045, 0.045, 0.02, 0.045, 0.025, 0.15]` (dynamically updated by doremi) |
| GA settings | `sim_func=kl`, `ga_step_func=gaussian`, `is_linear=true`, `is_dynamic_interval=true`, `start_step=400` |

Output: `OUTPUT_DIR/{run_name}/ep0-ba3200-rank0.pt` and `config.pt`

> **Record the final structure selected by the GA from the logs.** You must enter it
> manually in Step 2 (the `target_mask_size=...` log printed by
> `genetic_sparsity_callback.py`).

## Step 2: Prune the model with the mask

Apply the mask to actually remove weights, and re-index the layers.

> ### ⚠️ This step requires source modification
> The final structure discovered by the GA is not saved in the checkpoint
> (`target_model_info` / `target_mask_size` are plain attributes, so they are not included
> in the `state_dict`). For this reason, the target structure is **hardcoded** in
> `last_config` in `elasticpruner/utils/post_pruning_air.py`. You must edit it by hand to
> the values you confirmed in the Step 1 logs.
>
> ```python
> last_config = {
>         "n_layers": 24, "n_heads": 24,
>         "intermediate_size": 5636, "d_model": 3584,
>         "vocab_size": 32000
>         }
> ```
> See [`HARDCODED_PATHS.md`](HARDCODED_PATHS.md) for details.

```bash
python -m elasticpruner.utils.post_pruning_air prune_and_save_model \
    /path/to/output/mask_train/{run_name}/ep0-ba3200-rank0.pt
```

The `config.pt` in the same directory is read as well. The output is saved next to the
input file as `pruned-ep0-ba3200-rank0.pt`. (A wrapper, `elasticpruner/utils/post_prune.sh`,
is provided.)

## Step 3: After train the pruned model

Recover the performance lost to pruning through continued pre-training. Training is done
without a mask module by setting `model.l0_module=null`.

```bash
# Needs to be replaced: PROJ_DIR, DATA_DIR, OUTPUT_DIR, prune_run_name (Step 2 output directory)
bash elasticpruner/scripts/pruning_train_ours_7b.sh
```

`config_file` uses the **target size** configuration (for 7B→2.7B, `elasticpruner/configs/llama2/2.7b_ours.yaml`).
`lr=1e-4`, `max_duration=3200ba`, `t_warmup=320ba`.

> **Known issue:** As of commit `3e8a5a7`, it has been reported that during 2.7B after
> train, the loss increases around 80–100ba and then becomes NaN.

## Step 4: Convert to an HF model

Convert the Composer checkpoint to the `LlamaForCausalLM` format.
**You must pass the actual pruned structure exactly as arguments.**

`elasticpruner/scripts/composer_to_hf_llama.sh` contains a 1.3B example
(`hidden_size=2048, num_attention_heads=16, num_hidden_layers=24, intermediate_size=5504`).

To check the parameter count (accepts either a checkpoint `.pt` or an HF directory):

```bash
python count_params.py /path/to/hf_model
```

## Step 5: Benchmarks

The scripts in `evaluation/` use the `lm_eval` CLI from
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

```bash
pip install -r evaluation/eval_requirement.txt   # lm-eval==0.4.9.1
```

> The training environment is pinned to `transformers==4.33.0`, which may conflict with
> `lm-eval==0.4.9.1`. **We recommend running benchmarks in a separate virtual environment.**

### Running individually

Replace `pretrained=` in each script with the HF output path from Step 4, then run it.

| Script | Tasks | n-shot | batch |
|---|---|---:|---|
| `evaluation/run_zero_shot.sh` | winogrande, piqa, sciq, arc_easy, logiqa, lambada | 0 | auto |
| `evaluation/run_mmlu.sh` | mmlu | 5 | auto |
| `evaluation/run_hellaswag.sh` | hellaswag | 10 | auto |
| `evaluation/run_arc_c.sh` | arc_challenge | 25 | auto |
| `evaluation/run_boolq.sh` | boolq | 32 | 16 |
| `evaluation/run_nq.sh` | nq_open | 32 | auto |

```bash
cd evaluation
bash run_zero_shot.sh      # Results are saved to ./results
```

### Running multiple models and seeds automatically

`auto_bench_v3_seed.py` reads the six scripts above, substitutes `pretrained=`, appends
`--seed`, and runs them sequentially. Edit the settings at the bottom of the file before
running:

| Variable | Description |
|---|---|
| `target_models` | List of HF model paths to evaluate |
| `target_benchmarks` | List of `run_*.sh` scripts to run |
| `seed_list` | Default `[0, 1234, 42]` |
| `save_path` | Result JSON path (a `_seed{N}` suffix is appended per seed) |

```bash
cd evaluation
python auto_bench_v3_seed.py
```

### Aggregating results

```bash
# Result JSON -> Markdown table
python result_json_to_table.py evaluation/save_path_seed0.json

# Mean / standard deviation across seeds (edit run_name and seed_keys in the file first)
python calc_stat_bench.py
```

---

## Repository Structure

```
.
├── elasticpruner/
│   ├── train_only_mask_w_genetic.py   Step 1: GA + mask training
│   ├── train_om_after_pruned.py       Step 3: after train
│   ├── make_teacher_logit_for_ga.py   Step 0: teacher logit generation
│   ├── train.py                       Original Sheared-LLaMA entry point (baseline)
│   ├── models/
│   │   ├── composer_llama.py          LLaMA model (where the pruning mask is applied)
│   │   ├── l0_module.py               L0 mask module
│   │   └── l0_module_ours_only_total.py  Version with total parameter budget constraint
│   ├── callbacks/
│   │   ├── genetic_sparsity_callback.py  GA search callback (structure selection/application)
│   │   ├── air_callback.py            Sparsity schedule / diagnostics
│   │   └── dynamic_loading_callback.py   Dynamic domain proportion updates (doremi)
│   ├── utils/
│   │   ├── post_pruning_air.py        Step 2: mask-applied pruning
│   │   ├── composer_to_hf.py          Step 4: HF conversion
│   │   ├── genetic_algo_eval_loss.py  GA fitness evaluation
│   │   └── gentic_algo_new_total_sparsity.py  GA core
│   ├── data/                          Data tokenization / sampling / MDS encoding
│   ├── datasets/                      Streaming dataloaders
│   ├── configs/llama2/                Per-model-size configurations
│   └── scripts/                       Per-step execution scripts
├── evaluation/                        Step 5: lm-eval benchmarks
├── data/download_dataset.py           Training data download
├── ga_eval_data/download_datasets.py  GA fitness evaluation data download
├── count_params.py                    Parameter count check
├── result_json_to_table.py            Benchmark results -> Markdown table
└── calc_stat_bench.py                 Per-seed mean / standard deviation
```

## Reference Documents

| File | Description |
|---|---|
| [`HARDCODED_PATHS.md`](HARDCODED_PATHS.md) | Full list of paths and hardcoded values that must be replaced |
| [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | Per-file origin and license |
| [`CLEANUP_PLAN.md`](CLEANUP_PLAN.md) | List of files removed during public-release cleanup, and why |
