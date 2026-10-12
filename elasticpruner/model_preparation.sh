# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

HF_MODEL_NAME=princeton-nlp/Sheared-LLaMA-1.3B
OUTPUT_PATH=/path/to/models/sheared-llama-1.3b/state_dict.pt

# Create the necessary directory if it doesn't exist
mkdir -p $(dirname $OUTPUT_PATH)

# Convert the Hugging Face model to Composer key format
python3 -m elasticpruner.utils.composer_to_hf save_hf_to_composer $HF_MODEL_NAME $OUTPUT_PATH


