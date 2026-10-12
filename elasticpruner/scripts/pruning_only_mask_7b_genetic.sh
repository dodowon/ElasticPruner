# Based on princeton-nlp/LLM-Shearing (MIT License), Copyright (c) 2023 Princeton Natural Language Processing
# Modified by dodowon, 2026. See THIRD_PARTY_NOTICES.md for details.

# pruning llama2 7b -> 3b or 1.3b

# Please specify the working folder
PROJ_DIR=/path/to/ElasticPruner
LAUNCH_SCRIPT=${PROJ_DIR}/elasticpruner/scripts/launch.sh
DATA_DIR=/path/to/mds_data/for_prune
OUTPUT_DIR=/path/to/output/mask_train
TRAIN_SCRIPT=${PROJ_DIR}/elasticpruner/train_only_mask_w_genetic.py
MODEL_PATH=/path/to/models

# Specify $PROJ_DIR in scripts/launch.sh and scripts/srun_launch.sh if using slurm

test=False

from_model=7b # source model size
# to_model=1.3b # target model size
to_model=2.7b
config_file=${PROJ_DIR}/elasticpruner/configs/llama2/${from_model}.yaml
#path=$MODEL_PATH/sheared-llama-1.3b/state_dict.pt
path=$MODEL_PATH/llama-2-7b-composer/state_dict.pt

# data setup
data_local=${DATA_DIR}
ga_eval_data_path=/path/to/ga_eval_data

# basic setup
max_seq_len=4096
device_train_microbatch_size=4
global_train_batch_size=32
device_eval_batch_size=8

# learning setup
lr=1e-4 # learning rate for the main parameters
max_duration=3200ba # 0.42B tokens
save_interval=3200ba # save in the end
t_warmup=320ba # 10% learning rate warmup 

# dynamic loading setup
dynamic=True
set_names=[cc,github,book,stackexchange,wiki,arxiv,c4-rp] # domain names
proportion=[0.67,0.045,0.045,0.02,0.045,0.025,0.15] # initial proportion of RP, make sure that the sum(proportion) = 1
# doremi: update weights with exponential descent
# constant: keep the weights constant
update_type=doremi 
if [[ $to_model == 1.3b ]]; then
    target_loss=[1.9643,0.7459,2.1393,1.6117,1.7590,1.4449,2.1251] # 1.3b predicted loss from scaling law
elif [[ $to_model == 2.7b ]]; then
    target_loss=[1.8712,0.6883,2.0325,1.5353,1.6297,1.3560,2.0328] # 2.7b predicted loss from scaling law
elif [[ $to_model == 370m ]]; then
    target_loss=[2.1401,0.8694,2.3625,1.7791,2.047,1.6637,2.3139] # 410m predicted loss from scaling law
fi
eval_split_name=eval_merge # eval on all domains
eval_target_model=false # evaluate on the current model, not the target model, otherwise the loss will be inaccurate
eval_interval=400ba # eval every 50 batches and update the loading proportion


# pruning setup
lag_lr=1.0 # learning rate or l0_module
# lagr_warmup=640ba # 20% sparsity warmup
lagr_warmup=640ba ## air
if [[ $to_model == 1.3b ]]; then
    target_d_model=2048; target_n_heads=16; target_n_layers=24; target_intermediate_size=5504
elif [[ $to_model == 2.7b ]]; then
    target_d_model=2560; target_n_heads=20; target_n_layers=32; target_intermediate_size=6912
elif [[ $to_model == 370m ]]; then
    target_d_model=1024; target_n_heads=8; target_n_layers=24; target_intermediate_size=2816
fi

# save directroy
run_name=llama2_${from_model}_pruning_scaling_${update_type}_to${to_model}_sl${max_seq_len}_ours_om_0727_flexhead_ga_linear_flexintv_max_step_ga-max-len_32-512
save_dir=${OUTPUT_DIR}/${run_name}
wandb_dir=${save_dir} # save locally

if [[ $test == True ]]; then t=00-01:00:00; else t=00-20:00:00; fi

# Run in bash, it will automatically use resources available in the current environment


# Run with slurm    
#sbatch --job-name ${run_name} \
#    --nodes=4 \
#    --gpus-per-node=2 \
#    --mem=512gb \
#    --cpus-per-task=8 \
#    --time $t \
    #model.l0_module.target_model.target_params=2700000000 \
composer $TRAIN_SCRIPT \
    $config_file \
    run_name=${run_name} \
    data_local=${data_local} \
    eval_loader.dataset.split=${eval_split_name} \
    global_train_batch_size=${global_train_batch_size} \
    device_train_microbatch_size=${device_train_microbatch_size} \
    device_eval_batch_size=${device_eval_batch_size} \
    max_seq_len=${max_seq_len} \
    max_duration=${max_duration} \
    eval_first=false \
    scheduler.t_warmup=${t_warmup} \
    save_folder=${save_dir} \
    loggers.wandb.init_kwargs.dir=${wandb_dir} \
    eval_interval=${eval_interval} \
    save_interval=${save_interval} \
    optimizer.lr=${lr} \
    optimizer.lag_lr=${lag_lr} \
    model.path=${path} \
    model.l0_module.lagrangian_warmup_steps=${lagr_warmup} \
    model.l0_module.pruning_modules='[head,intermediate,layer,hidden]' \
    model.l0_module.eval_target_model=${eval_target_model} \
    model.l0_module.target_model.d_model=${target_d_model} \
    model.l0_module.target_model.n_heads=${target_n_heads} \
    model.l0_module.target_model.n_layers=${target_n_layers} \
    model.l0_module.target_model.intermediate_size=${target_intermediate_size} \
    callbacks.data_loading.dynamic=${dynamic} \
    callbacks.data_loading.set_names=${set_names} \
    callbacks.data_loading.proportion=${proportion} \
    callbacks.data_loading.update_type=${update_type} \
    callbacks.data_loading.target_loss=${target_loss} \
    train_loader.num_workers=0 \
    train_loader.prefetch_factor=null \
    train_loader.persistent_workers=false \
	genetic.is_use=true \
    genetic.teacher_model_name=meta-llama/Llama-2-7b-hf \
	genetic.teacher_logit_path=/path/to/teacher_logits/llama2-7b-teacher_logits_1.3b_n32_s4096.pt \
    genetic.sim_func=kl \
    genetic.ga_eval_data_path=${ga_eval_data_path} \
	genetic.is_logit=true \
	genetic.is_dynamic_interval=true \
    genetic.target_params=2745000000 \
	genetic.is_linear=true \
	genetic.fixed_head=false \
	genetic.ga_step_func=gaussian \
	genetic.ga_data_seq_len=512 \
	genetic.start_step=400 \
	genetic.is_gradual=true \
    autoresume=false
