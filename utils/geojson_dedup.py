"""GeoJSON feature deduplication — pure rule module, no torch dependency.

Geometry family dispatch:
  - "area":  Polygon, MultiPolygon → IoU dedup
  - "line":  LineString, MultiLineString → buffered IoU dedup
  - "point": Point, MultiPoint → distance dedup

Two-layer dedup:
  1. dedup_within_source: LLM self-dedup (stricter thresholds)
  2. dedup_cross_source:  detection vs LLM (design-doc thresholds, known wins)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from shapely.geometry import shape as shapely_shape

    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


# ---------------------------------------------------------------------------
# Geometry family classification
# ---------------------------------------------------------------------------

_AREA_TYPES = {"Polygon", "MultiPolygon"}
_LINE_TYPES = {"LineString", "MultiLineString"}
_POINT_TYPES = {"Point", "MultiPoint"}


def classify_geometry_family(geom_type: str) -> str:
    """Map a GeoJSON geometry type to its family: 'area', 'line', or 'point'."""
    if geom_type in _AREA_TYPES:
        return "area"
    if geom_type in _LINE_TYPES:
        return "line"
    if geom_type in _POINT_TYPES:
        return "point"
    raise ValueError(f"Unknown geometry type: {geom_type!r}")


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class DedupReport:
    """Result of a dedup operation."""
    kept_features: List[dict]
    removed_features: List[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    # stats keys: n_input, n_kept, n_removed, n_internal_merged, n_cross_removed


# ---------------------------------------------------------------------------
# Per-family dedup strategies
# ---------------------------------------------------------------------------

def _iou_dedup(
    feats_a: List[dict],
    feats_b: List[dict],
    iou_threshold: float,
    keep_a_on_conflict: bool,
    **kwargs,
) -> Tuple[List[dict], List[dict]]:
    """Deduplicate two lists of area features by IoU.

    Returns (kept, removed). If keep_a_on_conflict=True, features in feats_a
    always win over feats_b when IoU > threshold.
    """
    if not HAS_SHAPELY:
        return (feats_a + feats_b, [])

    kept = list(feats_a)
    removed = []

    for fb in feats_b:
        try:
            geom_b = shapely_shape(fb["geometry"])
        except Exception:
            removed.append(fb)
            continue

        is_dup = False
        for fa in kept:
            try:
                geom_a = shapely_shape(fa["geometry"])
            except Exception:
                continue
            try:
                intersection = geom_a.intersection(geom_b).area
                union = geom_a.union(geom_b).area
                iou = intersection / union if union > 0 else 0.0
            except Exception:
                iou = 0.0
            if iou > iou_threshold:
                if keep_a_on_conflict:
                    is_dup = True  # drop fb
                else:
                    # Merge: keep higher confidence
                    conf_a = fa.get("properties", {}).get("confidence", 0.5)
                    conf_b = fb.get("properties", {}).get("confidence", 0.5)
                    if conf_b > conf_a:
                        kept.remove(fa)
                        kept.append(fb)
                    is_dup = True
                break

        if is_dup:
            removed.append(fb)
        else:
            kept.append(fb)

    return kept, removed


def _buffer_iou_dedup(
    feats_a: List[dict],
    feats_b: List[dict],
    buffer_px: float,
    iou_threshold: float,
    keep_a_on_conflict: bool,
    **kwargs,
) -> Tuple[List[dict], List[dict]]:
    """Deduplicate line features using buffered IoU.

    buffer_px: buffer radius in the coordinate units (degrees for WGS84).
    """
    if not HAS_SHAPELY:
        return (feats_a + feats_b, [])

    kept = list(feats_a)
    removed = []

    for fb in feats_b:
        try:
            geom_b = shapely_shape(fb["geometry"]).buffer(buffer_px)
        except Exception:
            removed.append(fb)
            continue

        is_dup = False
        for fa in kept:
            try:
                geom_a = shapely_shape(fa["geometry"]).buffer(buffer_px)
            except Exception:
                continue
            try:
                intersection = geom_a.intersection(geom_b).area
                union = geom_a.union(geom_b).area
                iou = intersection / union if union > 0 else 0.0
            except Exception:
                iou = 0.0
            if iou > iou_threshold:
                if keep_a_on_conflict:
                    is_dup = True
                else:
                    len_a = len(fa.get("geometry", {}).get("coordinates", []))
                    len_b = len(fb.get("geometry", {}).get("coordinates", []))
                    if len_b > len_a:
                        kept.remove(fa)
                        kept.append(fb)
                    is_dup = True
                break

        if is_dup:
            removed.append(fb)
        else:
            kept.append(fb)

    return kept, removed


def _distance_dedup(
    feats_a: List[dict],
    feats_b: List[dict],
    distance_threshold: float,
    keep_a_on_conflict: bool,
    **kwargs,
) -> Tuple[List[dict], List[dict]]:
    """Deduplicate point features by Euclidean distance."""
    if not HAS_SHAPELY:
        return (feats_a + feats_b, [])

    kept = list(feats_a)
    removed = []

    for fb in feats_b:
        try:
            geom_b = shapely_shape(fb["geometry"])
        except Exception:
            removed.append(fb)
            continue

        is_dup = False
        for fa in kept:
            try:
                geom_a = shapely_shape(fa["geometry"])
                dist = geom_a.distance(geom_b)
            except Exception:
                dist = float("inf")
            if dist < distance_threshold:
                if keep_a_on_conflict:
                    is_dup = True
                else:
                    conf_a = fa.get("properties", {}).get("confidence", 0.5)
                    conf_b = fb.get("properties", {}).get("confidence", 0.5)
                    if conf_b > conf_a:
                        kept.remove(fa)
                        kept.append(fb)
                    is_dup = True
                break

        if is_dup:
            removed.append(fb)
        else:
            kept.append(fb)

    return kept, removed


# Strategy dispatch table
_FAMILY_DEDUP_FN = {
    "area":  _iou_dedup,
    "line":  _buffer_iou_dedup,
    "point": _distance_dedup,
}

_FAMILY_EXTRA_KWARGS = {
    "area":  {},
    "line":  {"buffer_px": 1e-5},   # ~1m at equator in degrees
    "point": {},
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DEFAULT_INTERNAL_THRESHOLDS = {
    "area":  0.7,       # IoU
    "line":  0.6,       # buffered IoU
    "point": 5e-6,      # distance (~0.5m in degrees)
}

DEFAULT_CROSS_THRESHOLDS = {
    "area":  0.5,       # IoU (from design doc §5.3)
    "line":  0.5,       # buffered IoU
    "point": 1e-5,      # distance (~1m in degrees)
}


def dedup_within_source(
    features: List[dict],
    thresholds: Optional[dict] = None,
) -> Tuple[List[dict], DedupReport]:
    """Deduplicate features within a single source (LLM self-dedup).

    Args:
        features: List of GeoJSON Feature dicts.
        thresholds: Per-family threshold dict, defaults to DEFAULT_INTERNAL_THRESHOLDS.

    Returns:
        (deduped_features, DedupReport)
    """
    if thresholds is None:
        thresholds = dict(DEFAULT_INTERNAL_THRESHOLDS)

    # Group by geometry family
    by_family: Dict[str, List[dict]] = {"area": [], "line": [], "point": []}
    for feat in features:
        try:
            family = classify_geometry_family(feat["geometry"]["type"])
        except (KeyError, ValueError):
            continue
        by_family[family].append(feat)

    n_input = len(features)
    kept_all = []
    removed_all = []

    for family, feats in by_family.items():
        if not feats:
            continue
        threshold = thresholds.get(family, 0.5)
        dedup_fn = _FAMILY_DEDUP_FN[family]
        kwargs = dict(_FAMILY_EXTRA_KWARGS.get(family, {}))
        # Self-dedup: feats_a starts empty, each feature compared against kept
        kept, removed = dedup_fn(
            feats_a=[],
            feats_b=feats,
            iou_threshold=threshold if family in ("area", "line") else 0.0,
            distance_threshold=threshold if family == "point" else 0.0,
            keep_a_on_conflict=False,  # merge by confidence
            **kwargs,
        )
        kept_all.extend(kept)
        removed_all.extend(removed)

    n_kept = len(kept_all)
    n_removed = len(removed_all)
    report = DedupReport(
        kept_features=kept_all,
        removed_features=removed_all,
        stats={
            "n_input": n_input,
            "n_kept": n_kept,
            "n_removed": n_removed,
            "n_internal_merged": n_removed,
        },
    )
    return kept_all, report


def dedup_cross_source(
    det_features: List[dict],
    llm_features: List[dict],
    thresholds: Optional[dict] = None,
) -> Tuple[List[dict], DedupReport]:
    """Deduplicate detection head features vs LLM fallback features.

    On conflict, detection features always win (known-first policy).

    Args:
        det_features: Features from detection heads (known classes).
        llm_features: Features from LLM fallback (unknown classes, already self-deduped).
        thresholds: Per-family threshold dict, defaults to DEFAULT_CROSS_THRESHOLDS.

    Returns:
        (merged_features, DedupReport)
    """
    if thresholds is None:
        thresholds = dict(DEFAULT_CROSS_THRESHOLDS)

    # Group LLM features by geometry family
    llm_by_family: Dict[str, List[dict]] = {"area": [], "line": [], "point": []}
    for feat in llm_features:
        try:
            family = classify_geometry_family(feat["geometry"]["type"])
        except (KeyError, ValueError):
            continue
        llm_by_family[family].append(feat)

    # Group det features by geometry family
    det_by_family: Dict[str, List[dict]] = {"area": [], "line": [], "point": []}
    for feat in det_features:
        try:
            family = classify_geometry_family(feat["geometry"]["type"])
        except (KeyError, ValueError):
            continue
        det_by_family[family].append(feat)

    n_llm_input = len(llm_features)
    kept_all = list(det_features)
    removed_all = []

    for family in ("area", "line", "point"):
        llm_feats = llm_by_family.get(family, [])
        if not llm_feats:
            continue
        det_feats = det_by_family.get(family, [])
        threshold = thresholds.get(family, 0.5)
        dedup_fn = _FAMILY_DEDUP_FN[family]
        kwargs = dict(_FAMILY_EXTRA_KWARGS.get(family, {}))

        keep_after, cross_removed = dedup_fn(
            feats_a=det_feats,  # known sources — kept on conflict
            feats_b=llm_feats,  # LLM sources — dropped on conflict
            iou_threshold=threshold if family in ("area", "line") else 0.0,
            distance_threshold=threshold if family == "point" else 0.0,
            keep_a_on_conflict=True,
            **kwargs,
        )
        # Only add the LLM features that survived (det features are already in kept_all)
        newly_kept = [f for f in keep_after if f not in det_feats]
        kept_all.extend(newly_kept)
        removed_all.extend(cross_removed)

    n_cross_removed = len(removed_all)
    report = DedupReport(
        kept_features=kept_all,
        removed_features=removed_all,
        stats={
            "n_det_input": len(det_features),
            "n_llm_input": n_llm_input,
            "n_final": len(kept_all),
            "n_cross_removed": n_cross_removed,
        },
    )
    return kept_all, report
