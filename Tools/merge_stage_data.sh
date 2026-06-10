#!/bin/bash
# 合并阶段二和阶段三数据到统一目录，用于混合训练。
# 用法：bash Tools/merge_stage_data.sh <stage2目录> <stage3目录> <输出目录>

STAGE2_DIR="${1:-../Stage2Data}"
STAGE3_DIR="${2:-./output/stage3_geojson_multiturn}"
OUTPUT_DIR="${3:-./MixedStage3Data}"

mkdir -p "$OUTPUT_DIR"

echo "链接阶段二数据: $STAGE2_DIR ..."
for f in "$STAGE2_DIR"/*.json; do
    [ -e "$f" ] || continue
    dst="$OUTPUT_DIR/$(basename "$f")"
    [ -e "$dst" ] || ln -s "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$dst"
done
for d in "$STAGE2_DIR"/*_Image; do
    [ -e "$d" ] || continue
    dst="$OUTPUT_DIR/$(basename "$d")"
    [ -e "$dst" ] || ln -s "$(cd "$(dirname "$d")" && pwd)/$(basename "$d")" "$dst"
done

echo "链接阶段三多轮数据: $STAGE3_DIR ..."
for f in "$STAGE3_DIR"/*.json; do
    [ -e "$f" ] || continue
    dst="$OUTPUT_DIR/$(basename "$f")"
    [ -e "$dst" ] || ln -s "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$dst"
done
for d in "$STAGE3_DIR"/*_Image; do
    [ -e "$d" ] || continue
    dst="$OUTPUT_DIR/$(basename "$d")"
    [ -e "$dst" ] || ln -s "$(cd "$(dirname "$d")" && pwd)/$(basename "$d")" "$dst"
done

echo "合并完成，输出目录: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR/"
echo "训练时使用 --data-path $OUTPUT_DIR"
