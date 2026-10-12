# Based on princeton-nlp/LLM-Shearing (MIT License), Copyright (c) 2023 Princeton Natural Language Processing
# Modified by dodowon, 2026. See THIRD_PARTY_NOTICES.md for details.

# Script that pre-generates and saves the teacher logit pool before GA training
# (reference: scripts/pruning_only_mask_1_3b_genetic.sh)
#
# Samples num_samples items from the training dataloader, feeds them to the teacher model
# to generate logits, and saves (data, logit) to a single file.
# The GA callback loads the saved file and uses it to compute fitness over many
# samples without a teacher forward pass at every step.

# Please specify the working folder
PROJ_DIR=/path/to/ElasticPruner
DATA_DIR=/path/to/mds_data/for_prune
OUTPUT_DIR=/path/to/teacher_logits
MAIN_SCRIPT=${PROJ_DIR}/elasticpruner/make_teacher_logit_for_ga.py

from_model=1.3b # source model size (basis for dataloader/domain setup)
config_file=${PROJ_DIR}/elasticpruner/configs/llama2/${from_model}_for_pruning.yaml

# data setup
data_local=${DATA_DIR}

# basic setup
max_seq_len=4096
global_train_batch_size=32

# dynamic loading setup (sample according to domain ratios, same as training)
dynamic=True
set_names=[cc,github,book,stackexchange,wiki,arxiv,c4-rp]
proportion=[0.67,0.045,0.045,0.02,0.045,0.025,0.15]
update_type=doremi

# teacher logit generation settings
#teacher_model_name=princeton-nlp/Sheared-LLaMA-1.3B
teacher_model_name=meta-llama/Llama-2-7b-hf
num_samples=32        # number of sequences to draw in advance
sample_seq_len=4096     # sequence length (128 recommended, same as GA)
chunk_size=1           # teacher forward split unit (reduce on OOM)
device=cuda
compute_dtype=float16  # teacher forward compute dtype (saves GPU memory)
save_dtype=float16     # saved logit dtype (saves disk)
save_format=torch      # torch | pickle

mkdir -p ${OUTPUT_DIR}
output_path=${OUTPUT_DIR}/llama2-7b-teacher_logits_${from_model}_n${num_samples}_s${sample_seq_len}.pt

# Run as a single process (single GPU) — so multiple processes do not overwrite the same file
CUDA_VISIBLE_DEVICES=1 python $MAIN_SCRIPT \
    $config_file \
    data_local=${data_local} \
    max_seq_len=${max_seq_len} \
    global_train_batch_size=${global_train_batch_size} \
    callbacks.data_loading.dynamic=${dynamic} \
    callbacks.data_loading.set_names=${set_names} \
    callbacks.data_loading.proportion=${proportion} \
    callbacks.data_loading.update_type=${update_type} \
    train_loader.num_workers=0 \
    train_loader.prefetch_factor=null \
    train_loader.persistent_workers=false \
    genetic.teacher_model_name=${teacher_model_name} \
    teacher_logit.num_samples=${num_samples} \
    teacher_logit.sample_seq_len=${sample_seq_len} \
    teacher_logit.chunk_size=${chunk_size} \
    teacher_logit.device=${device} \
    teacher_logit.compute_dtype=${compute_dtype} \
    teacher_logit.save_dtype=${save_dtype} \
    teacher_logit.save_format=${save_format} \
    teacher_logit.output_path=${output_path}
