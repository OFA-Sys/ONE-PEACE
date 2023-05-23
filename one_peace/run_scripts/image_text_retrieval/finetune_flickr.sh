#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6081
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4

config_dir=./
config_name=base
data=../../dataset/flickr30k/flickr_train.tsv
valid_data=../../dataset/flickr30k/flickr_val_new.tsv
valid_file=../../dataset/flickr30k/flickr30k_val_texts.json
max_epoch=20
lr=[7e-5]
drop_path_rate=0.4

torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../train.py \
    --config-dir=${config_dir} \
    --config-name=${config_name} \
    task.data=${data} \
    task.valid_data=${valid_data} \
    task.valid_file=${valid_file} \
    optimization.max_epoch=${max_epoch} \
    optimization.lr=${lr} \
    model.encoder.drop_path_rate=${drop_path_rate}