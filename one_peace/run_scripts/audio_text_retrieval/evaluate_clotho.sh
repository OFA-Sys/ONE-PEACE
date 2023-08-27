#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6081
export CUDA_VISIBLE_DEVICES=1,2,3
export GPUS_PER_NODE=3

config_dir=../../run_scripts
path=../../checkpoints/finetune_al_retrieval.pt
task_name=audio_text_retrieval
model_name=one_peace_retrieval
data=../../dataset/clotho/clotho_evaluation_new.tsv
selected_cols=uniq_id,audio,text,duration
valid_file=../../dataset/clotho/evaluation_texts.json
results_path=../../results/clotho
gen_subset='test'

torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
  --config-dir=${config_dir} \
  --config-name=evaluate \
  common_eval.path=${path} \
  common_eval.results_path=${results_path} \
  task._name=${task_name} \
  model._name=${model_name} \
  dataset.gen_subset=${gen_subset} \
  dataset.batch_size=8 \
  common_eval.model_overrides="{'task': {'data': '${data}', 'selected_cols': '${selected_cols}', 'bpe_dir': '../../utils/BPE', 'valid_file': '${valid_file}'}, 'model': {'copy_rel_pos_table': 'true'}}"
