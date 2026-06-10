#!/bin/bash
# CoastGPT Stage 3 Multi-Scale Fine-Tune
# Uses dynamic resolution: train on 224/280/336/392 with epoch-level size switching.
# Requires Phase 1 resolution-agnostic refactor applied (Configs use default_input_size).

set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
source /home/ma-user/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate PyTorch-2.1.0 2>/dev/null || true

export HF_HOME=/home/ma-user/work/CoastGPT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# ---- Configurable settings ----
CONFIG="${CONFIG:-Configs/step3_dual.yaml}"
MODEL_PATH="${MODEL_PATH:-./output/stage3/multiscale_v1/FINAL.pt}"
DATA_PATH="${DATA_PATH:-./MixedStage3Data_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/stage3/multiscale_v1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUM_STEPS="${ACCUM_STEPS:-8}"
EPOCHS="${EPOCHS:-12}"
NUM_GPUS="${NUM_GPUS:-2}"

# Enable multi-scale in config by overriding
# The config has multi_scale.enabled=false by default; set to true for this run.
echo "=== CoastGPT Stage 3 Multi-Scale Fine-Tune ==="
echo "Config:       ${CONFIG}"
echo "Output:       ${OUTPUT_DIR}"
echo "Data:         ${DATA_PATH}"
echo "GPUs:         ${NUM_GPUS}"
echo "Batch/GPU:    ${BATCH_SIZE}"
echo "Accum steps:  ${ACCUM_STEPS}"
echo "Epochs:       ${EPOCHS}"
echo "Multi-scale:  ENABLED (224/280/336/392)"
echo ""

# Multi-scale fine-tune with epoch-level size switching.
# The training script reads multi_scale from config; edit step3_dual.yaml
# to set multi_scale.enabled=true before running, or pass overrides.
deepspeed --num_nodes=1 --num_gpus="${NUM_GPUS}" train_stage_three.py \
  -c "${CONFIG}" \
  --batch-size "${BATCH_SIZE}" \
  --workers 1 \
  --accumulation-steps "${ACCUM_STEPS}" \
  --epochs "${EPOCHS}" \
  --model-path "${MODEL_PATH}" \
  --data-path "${DATA_PATH}" \
  --output "${OUTPUT_DIR}" \
  --accelerator npu \
  --enable-amp True \
  --use-checkpoint \
  --wandb False \
  --multi-scale

echo ""
echo "=== Training complete ==="
echo "Final checkpoint: ${MODEL_PATH}"
