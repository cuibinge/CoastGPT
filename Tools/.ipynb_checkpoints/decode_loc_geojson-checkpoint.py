"""Post-process Stage-3 GeoJSON predictions back to lon/lat.

When the dataset is built with ``--normalize-coords`` (and optionally
``--quantize-coords N``), the model emits FeatureCollections whose coordinates
are either floats in [0, 1] or string tokens of the form ``<loc_xxx>``. This
script consumes raw model outputs together with the per-sample
``tile_transform`` stored in the dataset JSON and writes a *georeferenced*
GeoJSON FeatureCollection that ArcGIS / QGIS can open directly.

Usage::

    python -m Tools.decode_loc_geojson \
        --dataset-json /path/to/GF2_geojson_train.json \
        --predictions  /path/to/predictions.json \
        --output-dir   /path/to/decoded
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.loc_tokens import (  # noqa: E402
    DEFAULT_LOC_BINS,
    decode_feature_collection,
)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the outermost JSON object from a possibly noisy model output."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--default-bins",
        type=int,
        default=DEFAULT_LOC_BINS,
        help="Fallback for samples that lack quantize_bins metadata.",
    )
    return parser.parse_args()


def _load_predictions(path: Path) -> Dict[str, str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    pred_map: Dict[str, str] = {}
    if isinstance(obj, dict) and isinstance(obj.get("predictions"), list):
        obj = obj["predictions"]
    if isinstance(obj, dict):
        for key, value in obj.items():
            pred_map[str(key)] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return pred_map
    if isinstance(obj, list):
        for record in obj:
            if not isinstance(record, dict):
                continue
            sample_id = record.get("sample_id") or record.get("id") or record.get("name")
            if sample_id is None:
                continue
            text = record.get("pred") or record.get("prediction") or record.get("output") or record.get("text")
            if text is None:
                continue
            pred_map[str(sample_id)] = str(text)
    return pred_map


def main() -> None:
    args = parse_args()
    dataset = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    entries: List[Dict[str, Any]] = (
        dataset.get("data", []) if isinstance(dataset, dict) else dataset
    )
    pred_map = _load_predictions(args.predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    decoded = 0
    skipped_missing = 0
    skipped_no_transform = 0
    skipped_unparseable = 0
    for entry in entries:
        sample_id = entry.get("sample_id") or entry.get("name")
        if sample_id is None:
            continue
        pred_text = pred_map.get(str(sample_id))
        if pred_text is None:
            skipped_missing += 1
            continue
        transform = entry.get("tile_transform")
        if not isinstance(transform, dict):
            skipped_no_transform += 1
            continue
        bins = int(entry.get("quantize_bins") or 0) or int(args.default_bins)
        encoding = str(entry.get("coord_encoding", "")).lower()
        if encoding != "loc_tokens":
            bins = 0  # treat as float-normalised coordinates
        fc = _extract_json(pred_text)
        if fc is None:
            skipped_unparseable += 1
            continue
        try:
            decoded_fc = decode_feature_collection(fc, transform=transform, quantize_bins=bins)
        except Exception:
            skipped_unparseable += 1
            continue
        out_path = args.output_dir / f"{sample_id}.geojson"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(decoded_fc, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        decoded += 1

    print(
        json.dumps(
            {
                "decoded": decoded,
                "skipped_missing_prediction": skipped_missing,
                "skipped_no_transform": skipped_no_transform,
                "skipped_unparseable": skipped_unparseable,
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
