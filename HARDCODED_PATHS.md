# Paths and Hardcoded Values to Replace

For public release, all of the author's personal absolute paths have been replaced with `/path/to/...` placeholders.
**Replace them to match your own environment before running.**

## Security Check Results

| Item | Result |
|---|---|
| API keys / tokens / passwords | **None** (exhaustive search for `api_key`, `secret`, `hf_*`, `sk-*`, `WANDB_API` patterns) |
| Hardcoded wandb entity / project name | **None** (only a local save directory is specified) |
| Personal absolute paths | **None** — all replaced with placeholders |
| Server addresses / emails | **None** |

---

## 1. Placeholder List

| Placeholder | Meaning |
|---|---|
| `/path/to/ElasticPruner` | Root of this repository (`PROJ_DIR`) |
| `/path/to/mds_data` | Root of the MDS-converted training data (contains `for_prune`, `for_ft`, `eval` subdirectories) |
| `/path/to/models` | Parent folder of the original models in Composer key format |
| `/path/to/output` | Root for checkpoint and log output |
| `/path/to/teacher_logits` | Save location of the stage 0 teacher logits |
| `/path/to/ga_eval_data` | Dataset for GA fitness evaluation |
| `/path/to/hf_model` | Stage 4 HF conversion output (input to stage 5 evaluation) |

## 2. Replacement Locations by Stage

### Common to all training scripts (`elasticpruner/scripts/*.sh`)

| Variable | Placeholder |
|---|---|
| `PROJ_DIR` | `/path/to/ElasticPruner` |
| `DATA_DIR` | `/path/to/mds_data/for_prune` |
| `OUTPUT_DIR` | `/path/to/output/...` |
| `MODEL_PATH` | `/path/to/models` |

### Additional Items by Stage

| Stage | File | Line | Variable |
|---|---|---:|---|
| 0 | `elasticpruner/scripts/make_teacher_logit_for_ga.sh` | 15 | `OUTPUT_DIR=/path/to/teacher_logits` |
| 1 | `elasticpruner/scripts/pruning_only_mask_7b_genetic.sh` | 23 | `path=$MODEL_PATH/llama-2-7b-composer/state_dict.pt` |
| 1 | 〃 | 27 | `ga_eval_data_path=/path/to/ga_eval_data` |
| 1 | 〃 | 126 | `genetic.teacher_logit_path=/path/to/teacher_logits/...pt` (stage 0 output) |
| 2 | `elasticpruner/utils/post_prune.sh` | 12 | `MODEL_PATH=/path/to/output/mask_train/<run_name>/ep0-ba3200-rank0.pt` |
| 3 | `elasticpruner/scripts/pruning_train_ours_7b.sh` | 25 | `prune_run_name=/path/to/output/prune_run` (stage 2 output directory) |
| 4 | `elasticpruner/scripts/composer_to_hf_llama.sh` | 4 | `MODEL_DIR=/path/to/output/after_train_run` |
| 5 | `evaluation/run_*.sh` (6 files) | 5 each | `--model_args pretrained=/path/to/hf_model` |
| 5 | `evaluation/auto_bench_v3_seed.py` | 147 | `target_models` list |

The same applies to the 1.3B variant scripts (`pruning_only_mask_1_3b_genetic.sh`, `pruning_train_ours_1_3b.sh`).

### config YAML (`data_local`, `model.path`)

The training scripts override it with `data_local=`, so it is harmless as long as you use the scripts,
but it must be replaced if you use the YAML directly.

| File | Key |
|---|---|
| `elasticpruner/configs/llama2/7b.yaml` | `data_local`, `model.path` |
| `elasticpruner/configs/llama2/1.3b_for_pruning.yaml` | `data_local`, `model.path` |
| `elasticpruner/configs/llama3/8b.yaml` | `data_local`, `model.path` |
| `elasticpruner/configs/llama2/{125m,1.3b,1.3b_ours,1.3b_genetic,2.7b_ours,370m,370m_ours,3b}.yaml` | `data_local` |

### Data Preparation Utilities (optional)

| File | Variable |
|---|---|
| `elasticpruner/data/eval_merge.sh` | `--input_dir`, `--output_dir` |
| `elasticpruner/utils/insert_ref_loss_dataset.py` | `SOURCE_ROOT`, `TARGET_ROOT` |
| `elasticpruner/utils/merge_mds_parts.py` | `SRC_ROOT`, `DST_ROOT` |
| `elasticpruner/model_preparation.sh` | `OUTPUT_PATH` |

---

## 3. [Important] Hardcoded Target Structure in `post_pruning_air.py`

This is a functional limitation of stage 2 (pruning the model with the mask). **Replacing paths alone does not resolve it.**

The final structure found by the genetic algorithm is recorded by `GeneticSparsityCallback` during execution
only as **plain Python attributes**: `l0_module.target_model_info` and each mask's `target_mask_size`.
These values are neither buffers nor parameters, so **they are not saved in the checkpoint `state_dict`.**

As a result, `post_pruning_air.py` cannot recover the target structure from the saved checkpoint alone,
and the result values of a specific experiment are hardcoded in the source (`elasticpruner/utils/post_pruning_air.py`):

```python
last_config = {
        "n_layers": 24,
        "n_heads": 24,
        "intermediate_size": 5636,
        "d_model": 3584,
        "vocab_size": 32000
        }
```

**To reproduce with a different target size, you must edit these values yourself.** The correct values can be found in the stage 1 (mask training)
log, in the final structure printed by the GA (the
`target_mask_size=...` log in `genetic_sparsity_callback.py`).

> Reproducibility would improve if the structure were accepted as a CLI argument or JSON. It is currently left unmodified
> to preserve the original behavior.

## 4. Other Hardcoded Values in the Code

| File | Content | Impact |
|---|---|---|
| `calc_stat_bench.py` | `run_name` in `__main__` is reassigned several times (only the last line takes effect). `seed_keys = [0, 1234, 42]` | You must specify the result file names to aggregate yourself |
| `evaluation/auto_bench_v3_seed.py` | `target_models`, `target_benchmarks`, `seed_list`, `save_path` | The evaluation targets are specified inside the file |
| `elasticpruner/callbacks/air_callback.py` | Graph save path `/path/to/output/images` | Saving the debug graph may fail if the directory does not exist |
| `elasticpruner/models/l0_module*.py` | `om.load(...)` paths in `__main__` (`configs/llama/7b.yaml`, `configs/pythia/410m.yaml`) | Development demo code from the original. This path layout does not exist in this repository, but it is not executed on module import, so the pipeline is unaffected |

## 5. Sheared-LLaMA References Kept as Model Identifiers

The following are **actual HuggingFace model IDs / baseline model names**, so changing them would break the code
or make it factually wrong. They were kept intentionally.

| File | Content |
|---|---|
| `elasticpruner/model_preparation.sh` | `HF_MODEL_NAME=princeton-nlp/Sheared-LLaMA-1.3B` |
| `elasticpruner/scripts/pruning_only_mask_1_3b_genetic.sh` | `genetic.teacher_model_name=princeton-nlp/Sheared-LLaMA-1.3B` |
| `elasticpruner/scripts/make_teacher_logit_for_ga.sh` | Teacher model candidates in the comments |
| `elasticpruner/make_teacher_logit_for_ga.py` | Usage example in the docstring |
| `calc_stat_bench.py` | Comparison baseline run name (`390m-sheared-llama-...`) |
