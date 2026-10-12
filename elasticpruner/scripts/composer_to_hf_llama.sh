# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

MODEL_DIR=/path/to/output/after_train_run
MODEL_PATH=$MODEL_DIR/ep0-ba3200-rank0.pt
OUTPUT_PATH=$MODEL_DIR/hf-latest_rank0_3200ba
MODEL_CLASS=LlamaForCausalLM
HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=16
NUM_HIDDEN_LAYERS=24
INTERMEDIATE_SIZE=5504
MODEL_NAME=ElasticPruner-1.3B_3200ba

python3 -m elasticpruner.utils.composer_to_hf save_composer_to_hf $MODEL_PATH $OUTPUT_PATH \
        model_class=${MODEL_CLASS} \
        hidden_size=${HIDDEN_SIZE} \
        num_attention_heads=${NUM_ATTENTION_HEADS} \
        num_hidden_layers=${NUM_HIDDEN_LAYERS} \
        intermediate_size=${INTERMEDIATE_SIZE} \
        num_key_value_heads=${NUM_ATTENTION_HEADS} \
        _name_or_path=${MODEL_NAME}

