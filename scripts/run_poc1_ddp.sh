#!/bin/bash
# PoC-1 Aquaculture Instance Head — 2-NPU DDP training via torchrun.
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
source /home/ma-user/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate PyTorch-2.1.0 2>/dev/null || true

export HF_HOME=/home/ma-user/work/CoastGPT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

CONFIG="${1:-configs/poc_aqua_instance.yaml}"
NUM_GPUS="${NUM_GPUS:-2}"

echo "=== PoC-1 2-NPU DDP Training ==="
echo "Config: ${CONFIG}"
echo "GPUs:   ${NUM_GPUS}"
echo ""

torchrun --nproc_per_node="${NUM_GPUS}" --master_port=29500 \
  scripts/poc_stage_one_det.py --config "${CONFIG}" --device npu
