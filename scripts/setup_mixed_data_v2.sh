#!/bin/bash
set -e

ORIG=/home/ma-user/work/CoastGPT/MixedStage3Data
NEW=/home/ma-user/work/CoastGPT/MixedStage3Data_v2
LANDCLASS=/home/ma-user/work/土地分类/stage3_geojson_landclass

mkdir -p "$NEW"

echo "=== Linking original MixedStage3Data ==="
for f in "$ORIG"/*.json "$ORIG"/*_Image; do
    name=$(basename "$f")
    if [ -L "$f" ]; then
        target=$(readlink -f "$f")
        ln -sf "$target" "$NEW/$name"
    elif [ -f "$f" ] || [ -d "$f" ]; then
        target=$(cd "$(dirname "$f")" && pwd)/$(basename "$f")
        ln -sf "$target" "$NEW/$name"
    fi
done

echo "=== Adding land class data ==="
ln -sf "$LANDCLASS/GF_geojson_train.json" "$NEW/GF_landclass_train.json"
ln -sf "$LANDCLASS/GF_geojson_train_Image" "$NEW/GF_landclass_train_Image"
ln -sf "$LANDCLASS/coord_transform_train.json" "$NEW/coord_transform_landclass.json"

echo "=== Result ==="
ls -la "$NEW/"
echo "Done: MixedStage3Data_v2 created"
