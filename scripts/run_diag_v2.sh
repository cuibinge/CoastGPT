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
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
timeout 3600 deepspeed --num_nodes=1 --num_gpus=2 ./train_stage_three.py   -c ./Configs/step3_dual.yaml   --batch-size 2 --workers 1 --accumulation-steps 8 --epochs 1   --model-path output/stage2/checkpoints/iter_2879_consolidated.pt   --data-path ./MixedStage3Data_v2   --auto-build-geojson-data False --geojson-priority True   --output ./runs/moe_diag_v2   --accelerator npu --enable-amp True --use-checkpoint   --wandb False --ckpt-period 5000   2>&1 | tee ./runs/moe_diag_v2/train.log
