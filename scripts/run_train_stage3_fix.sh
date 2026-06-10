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

MODEL_PATH=output/stage2/checkpoints/iter_2399_consolidated.pt
OUTPUT_PATH="./runs/stage3_fix_moe_lora32_geojson15x"
CONFIG_PATH=./Configs/step3_dual.yaml
SCRIPT_PATH=./train_stage_three.py
MERGED_DATA_PATH="./MixedStage3Data_v2"

mkdir -p "$OUTPUT_PATH"

echo "============================================"
echo "Stage 3 Retraining (Fixed Config)"
echo "Start: $(date)"
echo "Model: $MODEL_PATH"
echo "Data: $MERGED_DATA_PATH"
echo "Output: $OUTPUT_PATH"
echo "Config: $CONFIG_PATH"
echo "2 NPU, batch=2/GPU, accum=8, eff_batch=32"
echo "Changes vs original:"
echo "  aux_balance_coef: 8.0 -> 1.0"
echo "  mm_moe_aux_weight: 0.02 -> 0.15"
echo "  lora_r: 128 -> 32"
echo "  lora_alpha: 256 -> 64"
echo "  epochs: 4 -> 2"
echo "  WEIGHT_DICT: GeoJSON 15x, NWPU 0.25x"
echo "============================================"

deepspeed     --num_nodes=1     --num_gpus=2     $SCRIPT_PATH     -c $CONFIG_PATH     --batch-size 2     --workers 1     --accumulation-steps 8     --epochs 2     --model-path $MODEL_PATH     --data-path $MERGED_DATA_PATH     --auto-build-geojson-data False     --geojson-priority True     --output $OUTPUT_PATH     --accelerator "npu"     --enable-amp True     --use-checkpoint     --wandb False     --ckpt-period 5000

EXIT_CODE=$?
echo "============================================"
echo "Training finished at $(date) with exit code: $EXIT_CODE"
echo "============================================"
