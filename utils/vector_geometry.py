"""Generic vector-object helpers for GeoJSON supervision."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Tuple


def _iter_positions(coords: Any) -> Iterable[Tuple[float, float]]:
    if isinstance(coords, (list, tuple)):
        if len(coords) >= 2 and all(isinstance(v, (int, float)) for v in coords[:2]):
            yield float(coords[0]), float(coords[1])
            return
        for item in coords:
            yield from _iter_positions(item)


def _ring_closure_error(coords: Any) -> List[float]:
    errors: List[float] = []
    if not isinstance(coords, list):
        return errors
    if coords and isinstance(coords[0], list) and coords[0] and isinstance(coords[0][0], (int, float)):
        if len(coords) >= 2:
            first = coords[0]
            last = coords[-1]
            if len(first) >= 2 and len(last) >= 2:
                errors.append(math.hypot(float(first[0]) - float(last[0]), float(first[1]) - float(last[1])))
        return errors
    for item in coords:
        errors.extend(_ring_closure_error(item))
    return errors


def _geometry_stats(geometry: Dict[str, Any]) -> Dict[str, float]:
    coords = geometry.get("coordinates")
    points = list(_iter_positions(coords))
    closure_errors = _ring_closure_error(coords)
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
    else:
        width = 0.0
        height = 0.0
    return {
        "point_count": float(len(points)),
        "bbox_area": float(max(width, 0.0) * max(height, 0.0)),
        "closure_error": float(sum(closure_errors) / max(len(closure_errors), 1)),
    }


def extract_vector_stats(text_or_obj: Any) -> Dict[str, float]:
    """Extract domain-neutral geometry statistics from a GeoJSON object."""
    if isinstance(text_or_obj, str):
        try:
            obj = json.loads(text_or_obj)
        except Exception:
            return {"feature_count": 0.0, "point_count": 0.0, "bbox_area": 0.0, "closure_error": 0.0}
    else:
        obj = text_or_obj

    if not isinstance(obj, dict):
        return {"feature_count": 0.0, "point_count": 0.0, "bbox_area": 0.0, "closure_error": 0.0}

    if obj.get("type") == "FeatureCollection":
        features = obj.get("features", [])
    elif obj.get("type") == "Feature":
        features = [obj]
    else:
        features = [{"geometry": obj}]

    total_points = 0.0
    total_area = 0.0
    total_closure = 0.0
    valid_features = 0.0
    for feature in features:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry", feature)
        if not isinstance(geometry, dict):
            continue
        stats = _geometry_stats(geometry)
        if stats["point_count"] <= 0:
            continue
        valid_features += 1.0
        total_points += stats["point_count"]
        total_area += stats["bbox_area"]
        total_closure += stats["closure_error"]

    return {
        "feature_count": valid_features,
        "point_count": total_points,
        "bbox_area": total_area,
        "closure_error": total_closure / max(valid_features, 1.0),
    }


def build_vector_token_weights(
    labels,
    tokenizer,
    ignore_index: int,
    structure_weight: float = 2.0,
    coordinate_weight: float = 1.5,
):
    """Create per-token weights for generic vector-object language targets."""
    import torch

    weights = torch.ones_like(labels, dtype=torch.float32)
    weights = torch.where(labels.eq(ignore_index), torch.zeros_like(weights), weights)

    structure_tokens = [
        "{",
        "}",
        "[",
        "]",
        ":",
        ",",
        '"type"',
        '"FeatureCollection"',
        '"Feature"',
        '"geometry"',
        '"coordinates"',
        '"properties"',
        '"Point"',
        '"LineString"',
        '"Polygon"',
        '"MultiPolygon"',
    ]
    structure_ids = set()
    for token in structure_tokens:
        ids = tokenizer(token, add_special_tokens=False).input_ids
        structure_ids.update(int(i) for i in ids)

    for token_id in structure_ids:
        weights = torch.where(labels.eq(token_id), weights.new_full((), float(structure_weight)), weights)

    digit_ids = set()
    for token in list("0123456789.-"):
        ids = tokenizer(token, add_special_tokens=False).input_ids
        digit_ids.update(int(i) for i in ids)
    for token_id in digit_ids:
        weights = torch.where(labels.eq(token_id), torch.maximum(weights, weights.new_full((), float(coordinate_weight))), weights)

    return weights
