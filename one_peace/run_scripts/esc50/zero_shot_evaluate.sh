#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6081
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4

config_dir=../../run_scripts
path=../../checkpoints/one-peace.pt
data=../../dataset/esc50/esc50.tsv
selected_cols=uniq_id,audio,text,duration
results_path=../../results/esc50
gen_subset='esc50'

torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    --config-dir=${config_dir} \
    --config-name=evaluate \
    common_eval.path=${path} \
    common_eval.results_path=${results_path} \
    task._name=audio_text_retrieval \
    task.use_template=true \
    task.data=${data} \
    task.valid_file=../../dataset/esc50/esc50_label.json \
    task.selected_cols=${selected_cols} \
    task.head_type=al \
    task.zero_shot=true \
    dataset.gen_subset=${gen_subset} \
    model._name=one_peace_retrieval \
    common_eval.model_overrides="{'model':{'_name':'one_peace_retrieval'},'task':{'_name':'audio_text_retrieval'}}"


