# Third-Party Notices

ElasticPruner is a derivative work based on [Princeton NLP, LLM-Shearing](https://github.com/princeton-nlp/LLM-Shearing)
(MIT License).

The root of the original repository is MIT-licensed, but some files retain the unchanged
**Apache License 2.0** headers from MosaicML Examples / MosaicML Streaming.
**Those headers and copyright notices have not been changed in any way.**

- Full MIT text: [`LICENSE`](LICENSE)
- Full Apache-2.0 text: [`LICENSE-APACHE`](LICENSE-APACHE)

> **Notice on directory and package renaming**
> 
> In the public release, the Python package was renamed `llmshearing` → `elasticpruner`,
> and the wrapping directory `LLM-Shearing/` was flattened into the repository root.
> Therefore, the `Original path` column in the tables below refers to the **path in the original repository**.
> The renaming is a modification permitted by the MIT license, and the copyright notices and attribution text
> (`Based on princeton-nlp/LLM-Shearing ...`) were kept as-is in all applicable files.

## Classification Method

1. Map our path to the original path (`elasticpruner/X` → `llmshearing/X`) and compare bytes
   → if identical, `Original, unmodified`; otherwise `Original, modified`
2. If there is no corresponding file, compute the **line-level containment** against all original files;
   if it is 0.30 or higher, treat it as `Derived` from that original (license headers we injected are excluded from the comparison)
3. Files with an Apache-2.0 header are `Derived (Apache-2.0 code)` regardless of containment
4. If none of the above applies, `Newly written`
5. **Any file whose classification is uncertain is classified as modified/derived**

## Header Rules Applied

| Category | Header handling |
|---|---|
| Original, unmodified | No change (original author's header kept) |
| Derived (Apache-2.0) | Existing Apache-2.0 header **kept**, with `# Modified by dodowon, 2026.` added below it |
| Original, modified / Derived (MIT) | `# Based on princeton-nlp/LLM-Shearing (MIT License), Copyright (c) 2023 Princeton Natural Language Processing` + `# Modified by ...` |
| Newly written | `# Copyright 2026 dodowon` + `# SPDX-License-Identifier: MIT` |

## Summary

| Category | Files | Applied license |
|---|---:|---|
| Original, unmodified | 12 | Original author's license kept (MIT) |
| Original, modified | 20 | MIT (original copyright notice) + modification notice |
| Derived (Apache-2.0 code) | 6 | Apache-2.0 (header kept) + modification notice |
| Derived (MIT code) | 16 | MIT (original copyright notice) + modification notice |
| Newly written | 35 | MIT (Copyright 2026 dodowon) |
| **Total** | **89** | |

Documentation files (`*.md`), the full license texts (`LICENSE`, `LICENSE-APACHE`), and `.gitignore` are excluded from the tables.


## Original, unmodified (12)

| File | Original path | License | Original copyright notice |
|---|---|---|---|
| `elasticpruner/callbacks/callbacks.py` | LLM-Shearing `llmshearing/callbacks/callbacks.py` | MIT | — |
| `elasticpruner/callbacks/pruning_callback.py` | LLM-Shearing `llmshearing/callbacks/pruning_callback.py` | MIT | — |
| `elasticpruner/data/get_all_jsonl.py` | LLM-Shearing `llmshearing/data/get_all_jsonl.py` | MIT | — |
| `elasticpruner/data/llama_tokenizer.py` | LLM-Shearing `llmshearing/data/llama_tokenizer.py` | MIT | — |
| `elasticpruner/data/sample.py` | LLM-Shearing `llmshearing/data/sample.py` | MIT | — |
| `elasticpruner/data/sample_all_domains.sh` | LLM-Shearing `llmshearing/data/sample_all_domains.sh` | MIT | — |
| `elasticpruner/data/tokenize_all_files.sh` | LLM-Shearing `llmshearing/data/tokenize_all_files.sh` | MIT | — |
| `elasticpruner/data/tokenize_single_file.py` | LLM-Shearing `llmshearing/data/tokenize_single_file.py` | MIT | — |
| `elasticpruner/datasets/index.py` | LLM-Shearing `llmshearing/datasets/index.py` | Apache-2.0 | Copyright 2023 MosaicML Streaming authors |
| `elasticpruner/datasets/partition.py` | LLM-Shearing `llmshearing/datasets/partition.py` | MIT | — |
| `elasticpruner/models/metrics.py` | LLM-Shearing `llmshearing/models/metrics.py` | MIT | — |
| `elasticpruner/scripts/cleanshm.sh` | LLM-Shearing `llmshearing/scripts/cleanshm.sh` | MIT | — |

## Original, modified (20)

| File | Original path | License | Original copyright notice |
|---|---|---|---|
| `elasticpruner/callbacks/dynamic_loading_callback.py` | LLM-Shearing `llmshearing/callbacks/dynamic_loading_callback.py` | MIT | — |
| `elasticpruner/configs/llama2/1.3b.yaml` | LLM-Shearing `llmshearing/configs/llama2/1.3b.yaml` | MIT | — |
| `elasticpruner/configs/llama2/370m.yaml` | LLM-Shearing `llmshearing/configs/llama2/370m.yaml` | MIT | — |
| `elasticpruner/configs/llama2/3b.yaml` | LLM-Shearing `llmshearing/configs/llama2/3b.yaml` | MIT | — |
| `elasticpruner/configs/llama2/7b.yaml` | LLM-Shearing `llmshearing/configs/llama2/7b.yaml` | MIT | — |
| `elasticpruner/data/merge_data.py` | LLM-Shearing `llmshearing/data/merge_data.py` | MIT | — |
| `elasticpruner/datasets/load_text_dataloader.py` | LLM-Shearing `llmshearing/datasets/load_text_dataloader.py` | MIT | — |
| `elasticpruner/datasets/state.py` | LLM-Shearing `llmshearing/datasets/state.py` | MIT | — |
| `elasticpruner/models/composer_llama.py` | LLM-Shearing `llmshearing/models/composer_llama.py` | MIT | — |
| `elasticpruner/models/composer_pythia.py` | LLM-Shearing `llmshearing/models/composer_pythia.py` | MIT | — |
| `elasticpruner/models/l0_module.py` | LLM-Shearing `llmshearing/models/l0_module.py` | MIT | — |
| `elasticpruner/scripts/continue_pretrain.sh` | LLM-Shearing `llmshearing/scripts/continue_pretrain.sh` | MIT | — |
| `elasticpruner/scripts/launch.sh` | LLM-Shearing `llmshearing/scripts/launch.sh` | MIT | — |
| `elasticpruner/scripts/pruning.sh` | LLM-Shearing `llmshearing/scripts/pruning.sh` | MIT | — |
| `elasticpruner/scripts/srun_launch.sh` | LLM-Shearing `llmshearing/scripts/srun_launch.sh` | MIT | — |
| `elasticpruner/utils/composer_to_hf.py` | LLM-Shearing `llmshearing/utils/composer_to_hf.py` | MIT | — |
| `elasticpruner/utils/post_pruning_processing.py` | LLM-Shearing `llmshearing/utils/post_pruning_processing.py` | MIT | — |
| `elasticpruner/utils/utils.py` | LLM-Shearing `llmshearing/utils/utils.py` | MIT | — |
| `requirement.txt` | LLM-Shearing `requirement.txt` | MIT | — |
| `setup.py` | LLM-Shearing `setup.py` | MIT | — |

## Derived (Apache-2.0 code) (6)

| File | Original path | License | Original copyright notice |
|---|---|---|---|
| `elasticpruner/datasets/streaming_dataset.py` | LLM-Shearing `llmshearing/datasets/streaming_dataset.py` | Apache-2.0 | Copyright 2023 MosaicML Streaming authors |
| `elasticpruner/make_teacher_logit_for_ga.py` | LLM-Shearing `llmshearing/train.py` (containment 0.10) | Apache-2.0 | Copyright 2022 MosaicML Examples authors |
| `elasticpruner/models/model_registry.py` | LLM-Shearing `llmshearing/models/model_registry.py` | Apache-2.0 | Copyright 2022 MosaicML Examples authors |
| `elasticpruner/train.py` | LLM-Shearing `llmshearing/train.py` | Apache-2.0 | Copyright 2022 MosaicML Examples authors |
| `elasticpruner/train_om_after_pruned.py` | LLM-Shearing `llmshearing/train.py` (containment 0.82) | Apache-2.0 | Copyright 2022 MosaicML Examples authors |
| `elasticpruner/train_only_mask_w_genetic.py` | LLM-Shearing `llmshearing/train.py` (containment 0.66) | Apache-2.0 | Copyright 2022 MosaicML Examples authors |

## Derived (MIT code) (16)

| File | Original path | License | Original copyright notice |
|---|---|---|---|
| `elasticpruner/configs/llama2/1.3b_for_pruning.yaml` | LLM-Shearing `llmshearing/configs/llama2/7b.yaml` (containment 0.82) | MIT | — |
| `elasticpruner/configs/llama2/1.3b_genetic.yaml` | LLM-Shearing `llmshearing/configs/llama2/1.3b.yaml` (containment 0.91) | MIT | — |
| `elasticpruner/configs/llama2/1.3b_ours.yaml` | LLM-Shearing `llmshearing/configs/llama2/1.3b.yaml` (containment 0.95) | MIT | — |
| `elasticpruner/configs/llama2/125m.yaml` | LLM-Shearing `llmshearing/configs/llama2/1.3b.yaml` (containment 0.90) | MIT | — |
| `elasticpruner/configs/llama2/2.7b_ours.yaml` | LLM-Shearing `llmshearing/configs/llama2/1.3b.yaml` (containment 0.91) | MIT | — |
| `elasticpruner/configs/llama2/370m_ours.yaml` | LLM-Shearing `llmshearing/configs/llama2/370m.yaml` (containment 0.91) | MIT | — |
| `elasticpruner/configs/llama3/8b.yaml` | LLM-Shearing `llmshearing/configs/llama2/7b.yaml` (containment 0.92) | MIT | — |
| `elasticpruner/models/composer_llama3.py` | LLM-Shearing `llmshearing/models/composer_llama.py` (containment 0.88) | MIT | — |
| `elasticpruner/models/l0_module_llama3_ours.py` | LLM-Shearing `llmshearing/models/l0_module.py` (containment 0.68) | MIT | — |
| `elasticpruner/models/l0_module_ours_only_total.py` | LLM-Shearing `llmshearing/models/l0_module.py` (containment 0.66) | MIT | — |
| `elasticpruner/scripts/make_teacher_logit_for_ga.sh` | LLM-Shearing `llmshearing/scripts/pruning.sh` (containment 0.33) | MIT | — |
| `elasticpruner/scripts/pruning_only_mask_1_3b_genetic.sh` | LLM-Shearing `llmshearing/scripts/pruning.sh` (containment 0.67) | MIT | — |
| `elasticpruner/scripts/pruning_only_mask_7b_genetic.sh` | LLM-Shearing `llmshearing/scripts/pruning.sh` (containment 0.67) | MIT | — |
| `elasticpruner/scripts/pruning_train_ours_1_3b.sh` | LLM-Shearing `llmshearing/scripts/pruning.sh` (containment 0.69) | MIT | — |
| `elasticpruner/scripts/pruning_train_ours_7b.sh` | LLM-Shearing `llmshearing/scripts/pruning.sh` (containment 0.70) | MIT | — |
| `elasticpruner/utils/post_pruning_air.py` | LLM-Shearing `llmshearing/utils/post_pruning_processing.py` (containment 0.65) | MIT | — |

## Newly written (35)

| File | License |
|---|---|
| `calc_stat_bench.py` | MIT (Copyright 2026 dodowon) |
| `count_params.py` | MIT (Copyright 2026 dodowon) |
| `data/download_dataset.py` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/callbacks/air_callback.py` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/callbacks/ema_z_score_callback.py` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/callbacks/genetic_sparsity_callback.py` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/callbacks/random_selection_callback.py` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/callbacks/sab_callback.py` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/callbacks/sparsity_revert_diagnostic_callback.py` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/data/eval_merge.sh` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/ga_dataset/eval_dataset.py` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/model_preparation.sh` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/scripts/composer_to_hf_llama.sh` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/utils/distribution_test.py` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/utils/genetic_algo_eval_loss.py` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/utils/genetic_contraint.py` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/utils/genetic_contraint_w_head.py` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/utils/genetic_math.py` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/utils/genetic_utils.py` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/utils/gentic_algo_new_total_sparsity.py` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/utils/greedy_selection.py` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/utils/insert_ref_loss_dataset.py` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/utils/merge_mds_parts.py` | MIT (Copyright 2026 dodowon) |
| `elasticpruner/utils/post_prune.sh` | MIT (Copyright 2026 dodowon) |
| `evaluation/auto_bench_v3_seed.py` | MIT (Copyright 2026 dodowon) |
| `evaluation/eval_requirement.txt` | MIT (Copyright 2026 dodowon) |
| `evaluation/run_arc_c.sh` | MIT (Copyright 2026 dodowon) |
| `evaluation/run_boolq.sh` | MIT (Copyright 2026 dodowon) |
| `evaluation/run_hellaswag.sh` | MIT (Copyright 2026 dodowon) |
| `evaluation/run_mmlu.sh` | MIT (Copyright 2026 dodowon) |
| `evaluation/run_nq.sh` | MIT (Copyright 2026 dodowon) |
| `evaluation/run_zero_shot.sh` | MIT (Copyright 2026 dodowon) |
| `ga_eval_data/download_datasets.py` | MIT (Copyright 2026 dodowon) |
| `install_pip.sh` | MIT (Copyright 2026 dodowon) |
| `result_json_to_table.py` | MIT (Copyright 2026 dodowon) |
