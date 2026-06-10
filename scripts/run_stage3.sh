#!/bin/bash
set -e

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate PyTorch-2.1.0

export HF_HOME=/home/ma-user/work/CoastGPT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0

cd /home/ma-user/work/CoastGPT

OUTPUT_PATH="./output/stage3/mixed_v3"
CONFIG_PATH=./Configs/step3_dual.yaml
DATA_PATH="./MixedStage3Data_v2"

echo "============================================"
echo "Stage 3 Training (2 NPU) - $(date)"
echo "Model: ./FINAL.pt"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_PATH"
echo "2 NPU, batch=2/GPU, accum=8, eff_batch=32, epochs=4"
echo "============================================"

deepspeed     --num_nodes=1     --num_gpus=2     train_stage_three.py     -c $CONFIG_PATH     --batch-size 2     --workers 1     --accumulation-steps 8     --epochs 4     --model-path ./FINAL.pt     --data-path $DATA_PATH     --auto-build-geojson-data False     --geojson-priority True     --output $OUTPUT_PATH     --accelerator npu     --enable-amp True     --use-checkpoint     --wandb False     --ckpt-period 4000

echo "Training finished at $(date) with exit code: $?"
