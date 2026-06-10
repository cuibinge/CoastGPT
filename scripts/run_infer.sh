#!/bin/bash
#
# CoastGPT Stage3 推理脚本
#
# 用法:
#   bash run_infer.sh <图片路径> [prompt] [输出路径] [样本ID]
#
# 示例:
#   bash run_infer.sh ./GF1_PMS1_E119.4_N35.3.png
#   bash run_infer.sh ./test.jpg "检测海水养殖区" ./results.json "sample_01"
#

set -e

IMAGE="${1:?请指定图片路径}"
PROMPT="${2:-检测这张遥感影像中的海岸带地物，输出GeoJSON格式的标注结果。}"
OUTPUT="${3:-./infer_results.json}"
SAMPLE_ID="${4:-$(basename "$IMAGE" | sed 's/\.[^.]*$//')}"
MODEL_PATH="${5:-./output/stage3/mixed_v3/checkpoints/FINAL.pt}"
CONFIG="${6:-Configs/step3_dual.yaml}"
MAX_TOKENS="${7:-256}"

# 环境初始化
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate PyTorch-2.1.0

export HF_HOME=/home/ma-user/work/CoastGPT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

cd /home/ma-user/work/CoastGPT

echo "========================================"
echo "CoastGPT Stage3 Inference"
echo "========================================"
echo "Image:       $IMAGE"
echo "Prompt:      $PROMPT"
echo "Output:      $OUTPUT"
echo "Sample ID:   $SAMPLE_ID"
echo "Model:       $MODEL_PATH"
echo "Config:      $CONFIG"
echo "Max tokens:  $MAX_TOKENS"
echo "========================================"

python -u run_infer.py \
  --image "$IMAGE" \
  --prompt "$PROMPT" \
  --output "$OUTPUT" \
  --sample-id "$SAMPLE_ID" \
  --model-path "$MODEL_PATH" \
  --config "$CONFIG" \
  --max-new-tokens "$MAX_TOKENS"

echo "Done. Result: $OUTPUT"
