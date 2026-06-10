#!/bin/bash
set -e

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate PyTorch-2.1.0

export HF_HOME=/home/ma-user/work/CoastGPT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=1

cd /home/ma-user/work/CoastGPT
rm -rf ./runs/moe_diag_v2
mkdir -p ./runs/moe_diag_v2

MODEL_PATH=output/stage2/checkpoints/iter_2879_consolidated.pt
OUTPUT_PATH="./runs/moe_diag_v2"
CONFIG_PATH=./Configs/step3_dual.yaml
SCRIPT_PATH=./train_stage_three.py
MERGED_DATA_PATH="./MixedStage3Data_v2"

echo "=== Launching MoE diagnostic training v2 ==="
echo "Config: moe_warmup_steps=200, router_noise=0.2, gate_temperature=1.5"
echo "Log: ./runs/moe_diag_v2/train.log"

timeout 2700 deepspeed --num_nodes=1 --num_gpus=2 ${SCRIPT_PATH}   -c ${CONFIG_PATH}   --batch-size 2   --workers 1   --accumulation-steps 8   --epochs 1   --model-path ${MODEL_PATH}   --data-path ${MERGED_DATA_PATH}   --auto-build-geojson-data False   --geojson-priority True   --output ${OUTPUT_PATH}   --accelerator npu   --enable-amp True   --use-checkpoint   --wandb False   --ckpt-period 5000   2>&1 | tee ${OUTPUT_PATH}/train.log
