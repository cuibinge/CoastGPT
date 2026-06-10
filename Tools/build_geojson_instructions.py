"""
Build Stage-3 GeoJSON instruction data from raw geojson corpora.

Expected input layout:
    data/train/*.geojson
    data/train/<stem>_Image/*.png

Output layout:
    stage3_data/GF_geojson_train.json
    stage3_data/GF_geojson_train_Image/*
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_PROMPTS = (
    "<image>\n[DET] Extract the target features from this remote sensing image and output a valid GeoJSON FeatureCollection. Return JSON only.",
    "<image>\n[DET] Generate an editable GeoJSON FeatureCollection for ArcGIS from this remote sensing image. Return JSON only.",
    "<image>\n[DET] Output the extracted feature information for this image as a GeoJSON FeatureCollection with geometry and properties. Return JSON only.",
)


def normalize_feature_collection(item: Dict) -> Dict:
    features = []
    raw_type = str(item.get("type", "")).strip()

    if raw_type == "FeatureCollection" and isinstance(item.get("features"), list):
        features = item["features"]
    elif raw_type == "Feature":
        features = [item]
    elif isinstance(item.get("features"), list):
        features = item["features"]
    elif "geometry" in item:
        features = [{
            "type": "Feature",
            "geometry": item.get("geometry"),
            "properties": item.get("properties", {}),
        }]

    clean_features = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict) or not geometry.get("type"):
            continue
        properties = feature.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        clean_features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": properties,
        })

    out = {
        "type": "FeatureCollection",
        "features": clean_features,
    }
    if isinstance(item.get("name"), str) and item.get("name"):
        out["name"] = item["name"]
    return out


def pick_caption(item: Dict) -> str:
    if not isinstance(item, dict):
        return ""
    for feature in item.get("features", []):
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties", {})
        if not isinstance(properties, dict):
            continue
        for key in ("caption1", "caption2", "caption3"):
            value = properties.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def build_conversations(item: Dict, compact: bool = True) -> List[Dict[str, str]]:
    feature_collection = normalize_feature_collection(item)
    if not feature_collection["features"]:
        return []

    answer = json.dumps(
        feature_collection,
        ensure_ascii=False,
        separators=(",", ":") if compact else None,
        indent=None if compact else 2,
    )
    convs = [{"Question": prompt, "Answer": answer} for prompt in DEFAULT_PROMPTS]

    caption = pick_caption(item)
    if caption:
        convs.append(
            {
                "Question": (
                    "<image>\n[DET] Based on the following scene description, extract the target "
                    "features and output a GeoJSON FeatureCollection. Return JSON only.\n"
                    f"{caption}"
                ),
                "Answer": answer,
            }
        )
    return convs


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(src, dst)
        return

    try:
        os.link(src, dst)
        return
    except OSError:
        pass

    try:
        os.symlink(src.resolve(), dst)
        return
    except OSError:
        shutil.copy2(src, dst)


def process_split(
    data_root: Path,
    output_dir: Path,
    split: str,
    copy_images: bool = False,
):
    geojson_files = sorted(data_root.glob("*.geojson"))
    if not geojson_files:
        print(f"[WARN] No .geojson files found in {data_root}")
        return

    image_output_dir = output_dir / f"GF_geojson_{split}_Image"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    skipped = 0

    for geojson_path in geojson_files:
        img_folder = data_root / f"{geojson_path.stem}_Image"
        if not img_folder.exists():
            print(f"[WARN] Image folder not found: {img_folder}, skip")
            continue

        with geojson_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        items = raw.get("data", [])
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict) and not isinstance(raw.get("data"), list):
            items = [raw]

        for item in items:
            if not isinstance(item, dict):
                skipped += 1
                continue

            name = str(item.get("name", "")).strip()
            if not name:
                skipped += 1
                continue

            img_src = img_folder / f"{name}.png"
            if not img_src.exists():
                skipped += 1
                continue

            convs = build_conversations(item, compact=True)
            if not convs:
                skipped += 1
                continue

            img_dst = image_output_dir / img_src.name
            link_or_copy(img_src, img_dst, copy_images=copy_images)
            all_samples.append({"name": name, "conv": convs})

    json_out = output_dir / f"GF_geojson_{split}.json"
    with json_out.open("w", encoding="utf-8") as f:
        json.dump({"data": all_samples}, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[OK] split={split}, samples={len(all_samples)} -> {json_out}")
    if skipped:
        print(f"[WARN] skipped={skipped}")


def main():
    parser = argparse.ArgumentParser(description="Build GeoJSON instruction data for Stage-3 training")
    parser.add_argument("--data-root", default="data/train", help="Directory containing .geojson and *_Image folders")
    parser.add_argument("--output-dir", default="stage3_data", help="Root output directory for Stage-3 data")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split name")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of linking")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    if not data_root.exists():
        print(f"[ERROR] data-root not found: {data_root}")
        sys.exit(1)

    process_split(
        data_root=data_root,
        output_dir=output_dir,
        split=args.split,
        copy_images=bool(args.copy),
    )


if __name__ == "__main__":
    main()
