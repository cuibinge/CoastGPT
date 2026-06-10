MODEL_PATH=./FINAL.pt
OUTPUT_PATH="./output/stage3/multiturn_mixed"
RAW_DATA_ROOT="../GeoJsonData"
GEOJSON_OUTPUT_ROOT="./output/stage3_geojson_multiturn"
CONFIG_PATH=./Configs/step3_dual.yaml
SCRIPT_PATH=./train_stage_three.py
MERGED_DATA_PATH="./MixedStage3Data"

# # Step 0: Build multi-turn GeoJSON data
# python Tools/build_gf2_geojson_dataset.py \
#     --gf2-root $RAW_DATA_ROOT \
#     --output-dir $GEOJSON_OUTPUT_ROOT \
#     --sizes 128 \
#     --image-subdir Image_TrueColor \
#     --compact-answer \
#     --normalize-coords \
#     --multiturn \
#     --max-chars-per-turn 3000 \
#     --prompt-variants 1

# # Step 1: Merge Stage 2 and Stage 3 data
# bash Tools/merge_stage_data.sh ../Stage2Data $GEOJSON_OUTPUT_ROOT $MERGED_DATA_PATH

# Step 2: Train with mixed data
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0

deepspeed \
    --num_nodes=1 \
    --num_gpus=8 \
    $SCRIPT_PATH \
    -c \
    $CONFIG_PATH \
    --batch-size 4 \
    --workers 2 \
    --model-path $MODEL_PATH \
    --data-path $MERGED_DATA_PATH \
    --raw-data-root $RAW_DATA_ROOT \
    --auto-build-geojson-data False \
    --geojson-output-root $GEOJSON_OUTPUT_ROOT \
    --geojson-priority True \
    --output $OUTPUT_PATH \
    --accelerator "npu" \
    --enable-amp True \
    --use-checkpoint \
    --weight-sample
