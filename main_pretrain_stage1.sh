OUTPUT_PATH="./Checkpoint/test"  # Path to save the output
DATA_PATH="/root/shared-nvme/data/PretrainData"  # Path to the step_one dataset
SCRIPT_PATH=./main_pretrain_stage1.py
CONFIG_PATH=./Configs/train.yaml

deepspeed \
    --num_node=1 \
    --num_gpus=8 \
    $SCRIPT_PATH \
    -c \
    $CONFIG_PATH \
    --batch-size 8 \
    --workers 4 \
    --data-path $DATA_PATH \
    --output $OUTPUT_PATH \
    --accelerator "gpu" \
    --enable-amp True \
    --use-checkpoint \