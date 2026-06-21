"""Build generic GeoJSON instruction data without feature-specific rules."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _load_geojson(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("type") == "FeatureCollection":
        return obj
    if obj.get("type") == "Feature":
        return {"type": "FeatureCollection", "features": [obj]}
    if "geometry" in obj:
        return {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": obj["geometry"], "properties": obj.get("properties", {})}]}
    return None


def _iter_pairs(root: Path) -> Iterable[tuple[Path, Path]]:
    images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    geojson_by_stem = {p.stem: p for p in root.rglob("*.geojson") if p.is_file()}
    for image_path in images:
        geojson_path = geojson_by_stem.get(image_path.stem)
        if geojson_path is not None:
            yield image_path, geojson_path


def process_split(
    data_root,
    output_dir,
    split: str = "train",
    copy_images: bool = True,
    image_folder_name: str = "Generic_Image",
) -> Path:
    """Create a generic instruction JSON from image/GeoJSON pairs."""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    image_dir = output_dir / image_folder_name
    image_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, (image_path, geojson_path) in enumerate(_iter_pairs(data_root)):
        geojson_obj = _load_geojson(geojson_path)
        if geojson_obj is None:
            continue
        target_name = f"{split}_{idx:08d}{image_path.suffix.lower()}"
        if copy_images:
            shutil.copy2(image_path, image_dir / target_name)
        else:
            target_name = str(image_path.resolve())
        answer = json.dumps(geojson_obj, ensure_ascii=True, separators=(",", ":"))
        records.append(
            {
                "name": target_name,
                "conv": [
                    {
                        "Question": (
                            "<image>[DET] Output only the requested vector objects as a valid GeoJSON "
                            "FeatureCollection. Treat requested features as open vocabulary queries. "
                            "Do not assign non-requested regions to any known category. Return JSON only."
                        ),
                        "Answer": answer,
                    }
                ],
                "features": geojson_obj.get("features", []),
            }
        )

    output_path = output_dir / f"Generic_geojson_{split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"data": records}, f, ensure_ascii=True)
    return output_path
