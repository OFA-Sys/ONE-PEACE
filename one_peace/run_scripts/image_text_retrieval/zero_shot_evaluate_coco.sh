#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6081
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4

config_dir=../../run_scripts
path=../../checkpoints/one-peace.pt
data=../../dataset/mscoco/test_new.tsv
selected_cols=image_id,image,caption
results_path=../../results/mscoco
gen_subset='test'

torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    --config-dir=${config_dir} \
    --config-name=evaluate \
    common_eval.path=${path} \
    common_eval.results_path=${results_path} \
    task._name=image_text_retrieval \
    task.data=${data} \
    task.valid_file=../../dataset/mscoco/test_texts.json \
    task.selected_cols=${selected_cols} \
    task.head_type=vl \
    task.zero_shot=true \
    dataset.gen_subset=${gen_subset} \
    model._name=one_peace_retrieval \
    common.bf16=false common.memory_efficient_bf16=false \
    common_eval.model_overrides="{'model':{'_name':'one_peace_retrieval'},'task':{'_name':'image_text_retrieval'}}"


