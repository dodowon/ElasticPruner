# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

# Stage 2: Actually prune the model by applying the trained mask.
# The input is the stage 1 (mask training) checkpoint; config.pt in the same directory is read together with it.
# The output is saved next to the input file as pruned-<filename>.
#
# Note: The target structure is hardcoded in last_config in post_pruning_air.py.
#       Check the structure selected by the GA in the stage 1 log and edit it first.
#       See section 3 of HARDCODED_PATHS.md for details.

MODEL_PATH=/path/to/output/mask_train/<run_name>/ep0-ba3200-rank0.pt

python -m elasticpruner.utils.post_pruning_air prune_and_save_model $MODEL_PATH
