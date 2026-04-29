#!/usr/bin/env bash

OUTPUT_PATH="./Checkpoints"  # Path to save the output
DATA_PATH="../Data/PretrainData"  # Path to the step_one dataset
SCRIPT_PATH=./train_stage_one.py
CONFIG_PATH=./Configs/step1_dual.yaml

deepspeed \
    --num_node=1 \
    --num_gpus=2 \
    "$SCRIPT_PATH" \
    -c \
    "$CONFIG_PATH" \
    --batch-size 4 \
    --workers 2 \
    --data-path "$DATA_PATH" \
    --output "$OUTPUT_PATH" \
    --accelerator "npu" \
    --enable-amp True \
    --use-checkpoint \
    --wandb False \
    --name "stage1"
