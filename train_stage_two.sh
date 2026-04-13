MODEL_PATH="" # Path to the Stage1 model
OUTPUT_PATH=""  # Path to save the output
DATA_PATH=""  # Path to the Stage 2 dataset
CONFIG_PATH=./Configs/step2_dual.yaml
SCRIPT_PATH=./train_stage_two.py

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0

deepspeed \
    --num_nodes=1 \
    --num_gpus=8 \
    $SCRIPT_PATH \
    -c \
    $CONFIG_PATH \
    --batch-size 4 \
    --workers 2 \
    --data-path $DATA_PATH \
    --output $OUTPUT_PATH \
    --accelerator "npu" \
    --enable-amp True \
    --use-checkpoint \
