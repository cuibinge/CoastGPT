OUTPUT_PATH=""  # Path to save the output
DATA_PATH=""  # Path to the step_one dataset
SCRIPT_PATH=./train_stage_one.py
CONFIG_PATH=./Configs/train_dual.yaml

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0

deepspeed \
    --num_nodes=1 \
    --num_gpus=8 \
    $SCRIPT_PATH \
    -c \
    $CONFIG_PATH \
    --batch-size 8 \
    --workers 4 \
    --data-path $DATA_PATH \
    --output $OUTPUT_PATH \
    --accelerator "npu" \
    --enable-amp True \
    --use-checkpoint \
