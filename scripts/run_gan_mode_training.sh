#!/bin/bash

## Hardware
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

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
export MAIN=${ROOT_DIR}/tools/run_trainer_gan_mode.py
export CONFIG_PATH=${ROOT_DIR}/cfg/full_gan_bert_base_config.json
export DATASET_PATH=/media/data/yassir/datasets/QuasarT/QuasarT_ds
export OUTPUT_DIR=/media/data/yassir/output/test_training_gan_mode_sgd

# Training settings
export BATCH_SIZE=1
export EPOCHS=5
export GEN_ROUNDS=4
export DIS_ROUNDS=3

# Other hyper-parameters
export LR=1e-3
export WD=0.
export OPT="adamw"


python ${MAIN}  \
    --dataset_path ${DATASET_PATH} \
    --config_path ${CONFIG_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --num_gen_rounds ${GEN_ROUNDS} \
    --num_dis_rounds ${DIS_ROUNDS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --optimizer ${OPT} \
    --overwrite_output_dir \
    --device_map 3 4 5
