#!/bin/bash
# A2 8-NPU training: Soft edge target + Multi-scale Deep Supervision
# Launches poc_stage_edge_ds.py with DeepSpeed ZeRO-0 (DDP only).

set -e

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate PyTorch-2.1.0

export HF_HOME=/home/ma-user/work/CoastGPT/hf_cache
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

CONFIG_PATH="${1:-configs/poc3_edge_a2_soft_multiscale.yaml}"
OUTPUT_PATH="${2:-outputs/poc3_edge/a2_8npu}"

echo "============================================"
echo "PoC-3 A2: 8-NPU DeepSpeed Training"
echo "Config:  $CONFIG_PATH"
echo "Output:  $OUTPUT_PATH"
echo "============================================"

deepspeed \
    --num_nodes=1 \
    --num_gpus=8 \
    scripts/poc_stage_edge_ds.py \
    -c "$CONFIG_PATH" \
    --batch-size 2 \
    --workers 2 \
    --accumulation-steps 4 \
    --epochs 30 \
    --output "$OUTPUT_PATH" \
    --accelerator npu \
    --enable-amp False
