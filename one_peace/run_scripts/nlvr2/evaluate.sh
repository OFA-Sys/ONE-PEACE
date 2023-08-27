#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6081
export CUDA_VISIBLE_DEVICES=1,2,3
export GPUS_PER_NODE=3

config_dir=../../run_scripts
path=../../checkpoints/finetune_nlvr2.pt
task_name=nlvr2
model_name=one_peace_classify
selected_cols=uniq_id,text,image1,image2,label
results_path=../../results/nlvr2


data=../../dataset/nlvr2/dev.tsv
gen_subset='dev'
torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    --config-dir=${config_dir} \
    --config-name=evaluate \
    common_eval.path=${path} \
    common_eval.results_path=${results_path} \
    task._name=${task_name} \
    model._name=${model_name} \
    dataset.gen_subset=${gen_subset} \
    common_eval.model_overrides="{'task': {'data': '${data}', 'selected_cols': '${selected_cols}', 'bpe_dir': '../../utils/BPE'}}"

data=../../dataset/nlvr2/test1.tsv
gen_subset='test1'
torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    --config-dir=${config_dir} \
    --config-name=evaluate \
    common_eval.path=${path} \
    common_eval.results_path=${results_path} \
    task._name=${task_name} \
    model._name=${model_name} \
    dataset.gen_subset=${gen_subset} \
    common_eval.model_overrides="{'task': {'data': '${data}', 'selected_cols': '${selected_cols}', 'bpe_dir': '../../utils/BPE'}}"


