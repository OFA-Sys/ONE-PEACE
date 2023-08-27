#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6081
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS_PER_NODE=8

config_dir=./
config_name=base
data=../../dataset/refcoco/train.tsv
valid_data=../../dataset/refcoco/val.tsv

torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../train.py \
    --config-dir=${config_dir} \
    --config-name=${config_name} \
    task.data=${data} \
    task.valid_data=${valid_data}