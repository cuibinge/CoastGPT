MODEL_PATH="" # step_two model路径
OUTPUT_PATH=""  # output路径
DATA_PATH=""  #  step_two 的数据集
CONFIG_PATH=../Config/step2.yaml
SCRIPT_PATH=../train_step_two.py

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