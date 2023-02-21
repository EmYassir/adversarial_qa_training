#!/bin/bash

## General
export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=4
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NUM_PROC=8

## CACHE
export HF_HOME=/media/data/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME

# Paths
export ROOT_DIR=~/qa_gan_project
export MAIN=${ROOT_DIR}/tools/run_trainer_ds.py
export CONFIG_PATH=${ROOT_DIR}/cfg/full_gan_bert_base_config.json
export OUTPUT_DIR=/media/data/yassir/output/test_training_loop


## Deepspeed
export INCLUDE="localhost:0,1,2,3,4,5,6,7"
export DS_HOSTFILE=${ROOT_DIR}/deepspeed_cfg/hostfile
export DS_CONFIG=${ROOT_DIR}/deepspeed_cfg/ds_config_zero2.json
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+2000))

# Original settings
export DATE=$(date '+%Y-%m-%d')
export BATCH_SIZE=1
export EPOCHS=4


deepspeed --hostfile ${DS_HOSTFILE} --include=${INCLUDE} --master_port=${MASTER_PORT} ${MAIN}  \
    --dataset_path /media/data/yassir/datasets/NaturalQuestions/dpr_Natural_Questions \
    --answerability_heuristic \
    --config_path $CONFIG_PATH \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --deepspeed \
    --deepspeed_config $DS_CONFIG
