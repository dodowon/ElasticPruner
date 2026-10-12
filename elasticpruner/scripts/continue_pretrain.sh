# Based on princeton-nlp/LLM-Shearing (MIT License), Copyright (c) 2023 Princeton Natural Language Processing
# Modified by dodowon, 2026. See THIRD_PARTY_NOTICES.md for details.

# pruning llama2 7b -> 3b or 1.3b

PROJ_DIR=/path/to/ElasticPruner
DATA_DIR=/path/to/mds_data/for_ft
OUTPUT_DIR=/path/to/output
LAUNCH_SCRIPT=${PROJ_DIR}/elasticpruner/scripts/launch.sh
TRAIN_SCRIPT=${PROJ_DIR}/elasticpruner/train.py

# test=True

model=1.3b # target model size
config_file=${PROJ_DIR}/elasticpruner/configs/llama2/${model}_genetic.yaml
prune_run_name=/path/to/output/prune_run
path=${prune_run_name}/pruned-latest-rank0.pt # path to the 
# pruned model
#path=/path/to/output/prune_run/changedkeys-ep0-ba3200-rank0.pt

# data setup
data_local=${DATA_DIR}

# basic setup
max_seq_len=4096
device_train_microbatch_size=16
global_train_batch_size=256
device_eval_batch_size=8

# learning setup
lr=5e-5 # learning rate for the main parameters
max_duration=48000ba # 50B tokens
save_interval=2000ba # save every 3200ba
t_warmup=1440ba # 3% learning rate warmup 

# dynamic loading setup
dynamic=True
#set_names=[cc,github,book,stackexchange,wiki,arxiv,c4-rp]
set_names=[cc,github,stackexchange,wiki,arxiv,c4] # domain names
#proportion=[0.2192,0.0002,0.0791,0.0064,0.0096,0.001,0.6845] # final proportion of pruning
# proportion=[0.2194,0.0791,0.0064,0.0096,0.001,0.6845] # final proportion of pruning before 1206
proportion=[0.2380,0.0002,0.0070,0.0104,0.0011,0.7433]
# doremi: update weights with exponential descent
# constant: keep the weights constant
update_type=doremi 
if [[ $model == 1.3b ]]; then
    target_loss=[1.9643,0.7459,2.1393,1.6117,1.4449,2.1251] # 1.3b predicted loss from scaling law
    # target_loss=[1.9643,2.1393,1.6117,1.7590,1.4449,2.1251] # 1.3b predicted loss from scaling law
    # target_loss=[1.9617,0.9055,1.6999,1.6844,1.5878,2.0942] # new reference loss from origin model
else
    target_loss=[1.8712,0.6883,2.0325,1.5353,1.3560,2.0328] # 2.7b predicted loss from scaling law
    # target_loss=[1.8712,2.0325,1.5353,1.6297,1.3560,2.0328] # 2.7b predicted loss from scaling law
    # target_loss=[]
fi
eval_split_name=eval_merge # eval on all domains
eval_interval=400ba # eval every 50 batches and update the loading proportion


# save directroy
run_name=${prune_run_name}_ft${max_duration}_kl_20_18_6154_2304
save_dir=${OUTPUT_DIR}/${run_name}
wandb_dir=${save_dir} # save locally

if [[ $test == True ]]; then t=00-01:00:00; else t=01-00:00:00; fi

# Run with slurm
# sbatch -p cli \
#     --job-name ${run_name} \
#     --nodes=8 \
#     --gpus-per-node=2 \
#     --mem=512gb \
#     --cpus-per-task=8 \
#     --time $t \
#     $LAUNCH_SCRIPT \
     

# Run in bash, it will automatically use resources available in the current environment
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
    eval_first=true \
    scheduler.t_warmup=${t_warmup} \
    save_folder=${save_dir} \
    loggers.wandb.init_kwargs.dir=${wandb_dir} \
    eval_interval=${eval_interval} \
    save_interval=${save_interval} \
    optimizer.lr=${lr} \
    model.l0_module=null \
    model.path=${path} \
    callbacks.data_loading.dynamic=${dynamic} \
    callbacks.data_loading.set_names=${set_names} \
    callbacks.data_loading.proportion=${proportion} \
    callbacks.data_loading.update_type=${update_type} \
    callbacks.data_loading.target_loss=${target_loss} \
    train_loader.num_workers=0 \
    train_loader.prefetch_factor=null \
    train_loader.persistent_workers=false \
    autoresume=false

# checking eval_first
