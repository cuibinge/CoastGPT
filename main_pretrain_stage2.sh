MODEL_PATH="./Checkpoint/LHRS/Stage1/FINAL.pt" # Path to the Stage1 model
OUTPUT_PATH="./Checkpoint/test_stage2"  # Path to save the output
DATA_PATH="/root/shared-nvme/data/Stage2Data_old"  # Path to the Stage 2 dataset
CONFIG_PATH=./Configs/step2.yaml
SCRIPT_PATH=./main_pretrain_stage2.py

deepspeed \
    --num_node=1 \
    --num_gpus=8 \
    $SCRIPT_PATH \
    -c \
    $CONFIG_PATH \
    --batch-size 4 \
    --workers 2 \
    --data-path $DATA_PATH \
    --output $OUTPUT_PATH \
    --accelerator "gpu" \
    --enable-amp True \
    --use-checkpoint \