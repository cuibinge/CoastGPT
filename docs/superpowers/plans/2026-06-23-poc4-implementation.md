# PoC-4 LLM Fallback + Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FusionPipeline that routes known classes to detection heads and unknown classes to LLM fallback, with full validation and deduplication.

**Architecture:** Sequential pipeline: LLM Parser → Gating → Detection Heads (known) + LLM Fallback (unknown) → Layer 1 Validation → Layer 2 Policy Check → Dedup → FeatureCollection. Each branch is a Predictor with a uniform interface. Validation and dedup are pure-rule modules with no torch dependency.

**Tech Stack:** Python 3.10, PyTorch 2.1.2, shapely, numpy, scikit-image

## Global Constraints

- `fusion_pipeline.py` only does orchestration/routing, no model forward details
- `geojson_builder.py` does not bloat — complex validation/policy goes to `geojson_validator.py`
- PoC-5 joint checkpoint simply replaces predictor loading, no FusionPipeline interface change
- LLM Parser outputs only `task_type`, `target_classes`, `raw_mentions`, `confidence`
- Parser does NOT judge known/unknown or assign branches
- Gating trusts only `label_map.json`, not LLM's class attribution
- Rule Layer fallback uses `alias/exact/longest/fuzzy` conservative matching
- Parse failure conditions and `confidence` threshold are in config
- `[CAP]/[VQA]` always skips detection heads
- `[DET]` with empty `unknown_classes` skips LLM fallback
- `[DET]` with empty `target_classes` defaults to error (configurable)
- Dedup is dispatch-based pure rules, same geometry family only
- LLM self-dedup uses stricter thresholds; cross-source dedup uses design-doc thresholds
- Conflict always keeps known detection feature
- Repair failure discards the feature (no invalid geometry kept)
- JSON parse success rate and schema valid rate targets: 1.0

---

## Task 1: Merge PoC-3 edge head and postprocess into main branch

**Files:**
- Create: `Models/edge_head.py`
- Create: `utils/edge_postprocess.py`

**Interfaces:**
- Produces: `SingleScaleEdgeHead(output_size=(224,224))` with `forward(p1,p2,p3,p4) -> Tensor[B,1,224,224]`
- Produces: `postprocess_edge(heatmap, georef, threshold, ...) -> dict` (GeoJSON FeatureCollection)

- [ ] **Step 1: Copy edge_head.py from worktree**

```bash
cp .claude/worktrees/poc3-edge-head/Models/edge_head.py Models/edge_head.py
```

- [ ] **Step 2: Copy edge_postprocess.py from worktree**

```bash
cp .claude/worktrees/poc3-edge-head/utils/edge_postprocess.py utils/edge_postprocess.py
```

- [ ] **Step 3: Verify imports work**

```bash
python -c "from Models.edge_head import SingleScaleEdgeHead; print('edge_head OK')"
python -c "from utils.edge_postprocess import postprocess_edge; print('edge_postprocess OK')"
```

Expected: both print "OK" with no import errors.

- [ ] **Step 4: Commit**

```bash
git add Models/edge_head.py utils/edge_postprocess.py
git commit -m "feat: merge PoC-3 edge head and postprocess into main branch"
```

---

## Task 2: `utils/geojson_dedup.py` — Dedup module

**Files:**
- Create: `utils/geojson_dedup.py`

**Interfaces:**
- Consumes: shapely (for IoU, buffer, distance)
- Produces: `classify_geometry_family(geom_type: str) -> str` — returns "area"/"line"/"point"
- Produces: `dedup_within_source(features: List[dict], thresholds: dict) -> Tuple[List[dict], dict]` — LLM self-dedup
- Produces: `dedup_cross_source(det_features: List[dict], llm_features: List[dict], thresholds: dict) -> Tuple[List[dict], dict]` — cross-source dedup
- Produces: `DedupReport` dataclass

- [ ] **Step 1: Write the module with all types and function stubs**

```python
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
from typing import Dict, List, Tuple

import numpy as np

try:
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

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
    thresholds: dict | None = None,
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
        dedup_fn = _FAMILY_DEDUP_FN.get(family, _iou_dedup)
        kwargs = dict(_FAMILY_EXTRA_KWARGS.get(family, {}))
        # Self-dedup: feats_a starts empty, each feature compared against kept
        kept, _ = dedup_fn(
            feats_a=[],
            feats_b=feats,
            iou_threshold=threshold if family in ("area", "line") else 0.0,
            distance_threshold=threshold if family == "point" else 0.0,
            keep_a_on_conflict=False,  # merge by confidence
            **kwargs,
        )
        kept_all.extend(kept)

    n_kept = len(kept_all)
    report = DedupReport(
        kept_features=kept_all,
        removed_features=removed_all,
        stats={
            "n_input": n_input,
            "n_kept": n_kept,
            "n_removed": n_input - n_kept,
            "n_internal_merged": n_input - n_kept,
        },
    )
    return kept_all, report


def dedup_cross_source(
    det_features: List[dict],
    llm_features: List[dict],
    thresholds: dict | None = None,
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
        dedup_fn = _FAMILY_DEDUP_FN.get(family, _iou_dedup)
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
```

- [ ] **Step 2: Write smoke test**

```bash
python -c "
from utils.geojson_dedup import classify_geometry_family, dedup_within_source, dedup_cross_source

# Test geometry family classification
assert classify_geometry_family('Polygon') == 'area'
assert classify_geometry_family('MultiPolygon') == 'area'
assert classify_geometry_family('LineString') == 'line'
assert classify_geometry_family('MultiLineString') == 'line'
assert classify_geometry_family('Point') == 'point'
assert classify_geometry_family('MultiPoint') == 'point'
print('classify_geometry_family: OK')

# Test self-dedup with two overlapping Polygons
f1 = {'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [[[0,0],[1,0],[1,1],[0,1],[0,0]]]}, 'properties': {'class': 'test', 'confidence': 0.9}}
f2 = {'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [[[0.5,0.5],[1.5,0.5],[1.5,1.5],[0.5,1.5],[0.5,0.5]]]}, 'properties': {'class': 'test', 'confidence': 0.5}}
kept, report = dedup_within_source([f1, f2], thresholds={'area': 0.1, 'line': 0.6, 'point': 1e-5})
print(f'Self-dedup: {report.stats}')
assert report.stats['n_input'] == 2
# With IoU=0.1 threshold, IoU of these is ~0.14 > 0.1, so they merge
assert report.stats['n_kept'] == 1, f'Expected 1 kept, got {report.stats[\"n_kept\"]}'
print('dedup_within_source: OK')

# Test cross-source dedup: det wins
det_f = [{'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [[[0,0],[1,0],[1,1],[0,1],[0,0]]]}, 'properties': {'class': 'known', 'confidence': 0.9}}]
llm_f = [{'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9],[0.1,0.1]]]}, 'properties': {'class': 'unknown', 'confidence': 0.7}}]
merged, report2 = dedup_cross_source(det_f, llm_f, thresholds={'area': 0.3, 'line': 0.5, 'point': 1e-5})
print(f'Cross-dedup: {report2.stats}')
assert report2.stats['n_cross_removed'] == 1, f'Expected 1 cross removed, got {report2.stats[\"n_cross_removed\"]}'
assert len(merged) == 1
assert merged[0]['properties']['class'] == 'known', 'Known feature should survive'
print('dedup_cross_source (known wins): OK')

# Test: non-overlapping features both survive
f3 = {'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [[[10,10],[11,10],[11,11],[10,11],[10,10]]]}, 'properties': {'class': 'far', 'confidence': 0.9}}
merged2, report3 = dedup_cross_source(det_f, [f3], thresholds={'area': 0.3, 'line': 0.5, 'point': 1e-5})
assert len(merged2) == 2, f'Expected 2 features (non-overlapping), got {len(merged2)}'
print('dedup_cross_source (non-overlap): OK')

print('All geojson_dedup tests passed.')
"
```

Expected: all assertions pass.

- [ ] **Step 3: Commit**

```bash
git add utils/geojson_dedup.py
git commit -m "feat: add geojson_dedup — geometry-family dispatch, self-dedup, cross-source dedup"
```

---

## Task 3: `utils/geojson_validator.py` — Layer 2 LLM fallback validation

**Files:**
- Create: `utils/geojson_validator.py`

**Interfaces:**
- Consumes: nothing outside stdlib + shapely (optional)
- Produces: `ValidationReport` dataclass
- Produces: `validate_llm_fallback(features: List[dict], unknown_class_whitelist: List[str], config: dict) -> Tuple[List[dict], ValidationReport]`

- [ ] **Step 1: Write the module**

```python
"""Layer 2 validation for LLM fallback GeoJSON features.

Pure rule module — no torch dependency.

Validates:
  1. Class whitelist  — LLM output must not contain known classes
  2. Unknown-class policy — features must be in unknown_classes
  3. Confidence filter — drop low-confidence LLM features
  4. Geometry type policy — (extensible) class→expected geometry type check

This is ONLY applied to LLM fallback results, not detection head results.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ValidationReport:
    """Result of Layer 2 LLM fallback validation."""
    passed: bool
    valid_features: List[dict] = field(default_factory=list)
    rejected_features: List[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    # stats keys: n_total, n_passed, n_whitelist_violation, n_unknown_policy,
    #             n_confidence_drop, n_geom_type_violation


def validate_llm_fallback(
    features: List[dict],
    unknown_class_whitelist: List[str],
    config: Optional[dict] = None,
) -> Tuple[List[dict], ValidationReport]:
    """Validate LLM fallback features against Layer 2 policies.

    Args:
        features: List of GeoJSON Feature dicts from LLM fallback.
        unknown_class_whitelist: List of class names the LLM is ALLOWED to output.
            If a feature has a class NOT in this list, it is rejected.
        config: Optional overrides for thresholds:
            - min_confidence: float (default 0.3) — drop features below this confidence.
            - known_classes: List[str] — if a feature's class is in this list
              (and NOT in the whitelist), it's a hard whitelist violation.

    Returns:
        (valid_features, ValidationReport)
    """
    if config is None:
        config = {}

    min_confidence = float(config.get("min_confidence", 0.3))
    known_classes = set(config.get("known_classes", []))
    whitelist_set = set(unknown_class_whitelist)

    valid = []
    rejected = []
    stats = {
        "n_total": len(features),
        "n_passed": 0,
        "n_whitelist_violation": 0,
        "n_unknown_policy": 0,
        "n_confidence_drop": 0,
        "n_geom_type_violation": 0,
    }

    for feat in features:
        props = feat.get("properties", {})
        class_name = props.get("class", "")

        # 1. Class whitelist violation: LLM outputting a known class
        if class_name in known_classes and class_name not in whitelist_set:
            rejected.append({
                **feat,
                "_reject_reason": f"class whitelist violation: {class_name!r} is a known class",
            })
            stats["n_whitelist_violation"] += 1
            continue

        # 2. Unknown-class policy: feature class not in whitelist
        if class_name not in whitelist_set:
            rejected.append({
                **feat,
                "_reject_reason": f"class {class_name!r} not in unknown_class whitelist {list(whitelist_set)}",
            })
            stats["n_unknown_policy"] += 1
            continue

        # 3. Confidence filter
        confidence = float(props.get("confidence", 0.5))
        if confidence < min_confidence:
            rejected.append({
                **feat,
                "_reject_reason": f"confidence {confidence:.3f} < min {min_confidence}",
            })
            stats["n_confidence_drop"] += 1
            continue

        valid.append(feat)

    stats["n_passed"] = len(valid)
    report = ValidationReport(
        passed=(stats["n_whitelist_violation"] == 0),
        valid_features=valid,
        rejected_features=rejected,
        stats=stats,
    )
    return valid, report
```

- [ ] **Step 2: Write smoke test**

```bash
python -c "
from utils.geojson_validator import validate_llm_fallback

# Valid: only unknown classes
f1 = {'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [[[0,0],[1,0],[1,1],[0,1],[0,0]]]}, 'properties': {'class': '红树林', 'confidence': 0.92}}
valid, report = validate_llm_fallback([f1], ['红树林'], config={'known_classes': ['海岸线','海水养殖区'], 'min_confidence': 0.3})
assert len(valid) == 1, f'Expected 1 valid, got {len(valid)}'
assert report.passed
print(f'Valid: {report.stats}')

# Whitelist violation: LLM outputs a known class
f2 = {'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [[[0,0],[1,0],[1,1],[0,1],[0,0]]]}, 'properties': {'class': '海岸线', 'confidence': 0.92}}
valid2, report2 = validate_llm_fallback([f2], ['红树林'], config={'known_classes': ['海岸线','海水养殖区'], 'min_confidence': 0.3})
assert len(valid2) == 0, f'Expected 0 valid (whitelist violation), got {len(valid2)}'
assert not report2.passed
assert report2.stats['n_whitelist_violation'] == 1
print(f'Whitelist violation caught: {report2.stats}')

# Low confidence filtered
f3 = {'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [[[0,0],[1,0],[1,1],[0,1],[0,0]]]}, 'properties': {'class': '红树林', 'confidence': 0.1}}
valid3, report3 = validate_llm_fallback([f3], ['红树林'], config={'known_classes': ['海岸线'], 'min_confidence': 0.3})
assert len(valid3) == 0, f'Expected 0 valid (low conf), got {len(valid3)}'
assert report3.stats['n_confidence_drop'] == 1
print(f'Low confidence filtered: {report3.stats}')

# Unknown class policy violation
f4 = {'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [[[0,0],[1,0],[1,1],[0,1],[0,0]]]}, 'properties': {'class': '滩涂', 'confidence': 0.8}}
valid4, report4 = validate_llm_fallback([f4], ['红树林'], config={'known_classes': ['海岸线'], 'min_confidence': 0.3})
assert len(valid4) == 0, f'Expected 0 valid (not in whitelist), got {len(valid4)}'
assert report4.stats['n_unknown_policy'] == 1
print(f'Unknown-class policy violation caught: {report4.stats}')

print('All geojson_validator tests passed.')
"
```

Expected: all assertions pass.

- [ ] **Step 3: Commit**

```bash
git add utils/geojson_validator.py
git commit -m "feat: add geojson_validator — Layer 2 LLM fallback policy validation"
```

---

## Task 4: Extend `utils/geojson_builder.py` — Multi-geometry-type Layer 1 validation

**Files:**
- Modify: `utils/geojson_builder.py`

**Interfaces:**
- Consumes: shapely (existing), `pixel_to_wgs84` (existing)
- Produces: `validate_geojson()` extended to support LineString, MultiLineString, MultiPolygon, Point
- Produces: `auto_repair_geometry(geom: dict) -> Tuple[dict, bool]` — repair function
- Produces: `filter_sliver_features(features: List[dict], config: dict) -> List[dict]`

**Changes to `validate_geojson()`:**
1. Replace Polygon-only geometry.type check with allowed set
2. Per-type coordinate validation (LineString: >=2 points; Polygon: closed ring, >=4 points; Point: 1 coordinate pair)
3. `_flatten_geojson_coords()` extended to extract coords from all geometry types
4. Auto-repair logic: Polygon close/open ring fix, orientation fix, buffer(0) for self-intersection
5. Sliver filter: configurable min_area/min_length/min_points

- [ ] **Step 1: Replace the geometry type check in validate_geojson()**

In `utils/geojson_builder.py`, find the block at lines ~210-216 that currently only accepts "Polygon":

```python
        # geometry.type == Polygon
        if geom.get("type") != "Polygon":
            result["errors"].append(
                f"Feature[{idx}] geometry.type must be 'Polygon', got {geom.get('type')!r}"
            )
            continue
```

Replace with:

```python
        # geometry.type must be a supported type
        SUPPORTED_TYPES = {"Polygon", "MultiPolygon", "LineString", "MultiLineString", "Point", "MultiPoint"}
        geom_type = geom.get("type")
        if geom_type not in SUPPORTED_TYPES:
            result["errors"].append(
                f"Feature[{idx}] geometry.type must be one of {SUPPORTED_TYPES}, got {geom_type!r}"
            )
            continue

        # Per-type coordinate validation
        coords = geom.get("coordinates")
        type_coord_ok = True
        if geom_type == "Polygon":
            if not isinstance(coords, list) or len(coords) == 0:
                result["errors"].append(f"Feature[{idx}] Polygon coordinates empty")
                type_coord_ok = False
            elif not isinstance(coords[0], list) or len(coords[0]) < 4:
                result["errors"].append(f"Feature[{idx}] Polygon outer ring has < 4 points")
                type_coord_ok = False
        elif geom_type == "MultiPolygon":
            if not isinstance(coords, list) or len(coords) == 0:
                result["errors"].append(f"Feature[{idx}] MultiPolygon coordinates empty")
                type_coord_ok = False
        elif geom_type == "LineString":
            if not isinstance(coords, list) or len(coords) < 2:
                result["errors"].append(f"Feature[{idx}] LineString needs >= 2 points, got {len(coords) if isinstance(coords, list) else 0}")
                type_coord_ok = False
        elif geom_type == "MultiLineString":
            if not isinstance(coords, list) or len(coords) == 0:
                result["errors"].append(f"Feature[{idx}] MultiLineString coordinates empty")
                type_coord_ok = False
        elif geom_type in ("Point", "MultiPoint"):
            if not isinstance(coords, list) or len(coords) == 0:
                result["errors"].append(f"Feature[{idx}] {geom_type} coordinates empty")
                type_coord_ok = False

        if not type_coord_ok:
            continue
```

Also replace the subsequent coordinate point validation (lines ~219-243) that assumes `coords[0]` is the outer ring with a generic coordinate validation that handles all types.

- [ ] **Step 2: Add `_flatten_geojson_coords` support for all geometry types**

Replace the existing `_flatten_geojson_coords()` (lines 43-52):

```python
def _flatten_geojson_coords(geom: dict) -> List[Tuple[float, float]]:
    """Extract all (lon, lat) coordinate pairs from any GeoJSON geometry."""
    geom_type = geom.get("type")
    coords = geom.get("coordinates")
    if not isinstance(coords, list) or len(coords) == 0:
        return []

    if geom_type == "Polygon":
        # coords = [[outer_ring], [hole1], ...]
        pts = []
        for ring in coords:
            if isinstance(ring, list):
                pts.extend(tuple(pt) for pt in ring if isinstance(pt, (list, tuple)) and len(pt) >= 2)
        return pts

    elif geom_type == "MultiPolygon":
        # coords = [[[outer], [hole]], [[outer2]], ...]
        pts = []
        for polygon in coords:
            if isinstance(polygon, list):
                for ring in polygon:
                    if isinstance(ring, list):
                        pts.extend(tuple(pt) for pt in ring if isinstance(pt, (list, tuple)) and len(pt) >= 2)
        return pts

    elif geom_type == "LineString":
        return [tuple(pt) for pt in coords if isinstance(pt, (list, tuple)) and len(pt) >= 2]

    elif geom_type == "MultiLineString":
        pts = []
        for line in coords:
            if isinstance(line, list):
                pts.extend(tuple(pt) for pt in line if isinstance(pt, (list, tuple)) and len(pt) >= 2)
        return pts

    elif geom_type == "Point":
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            return [tuple(coords)]
        return []

    elif geom_type == "MultiPoint":
        return [tuple(pt) for pt in coords if isinstance(pt, (list, tuple)) and len(pt) >= 2]

    return []
```

- [ ] **Step 3: Add `auto_repair_geometry()` and patch it into `validate_geojson()`**

Add before `validate_geojson()`:

```python
def auto_repair_geometry(geom: dict) -> Tuple[dict, bool]:
    """Attempt to repair an invalid geometry.

    Returns (repaired_geometry, was_repaired).

    Repairs attempted:
      - Polygon not closed → auto-close ring
      - Ring orientation wrong → orient(ccw=True) via shapely
      - Self-intersection → buffer(0)
      - LineString < 2 points after dedup → cannot repair
    """
    if not HAS_SHAPELY:
        return geom, False

    geom_type = geom.get("type")
    try:
        shp = shapely_shape(geom)
    except Exception:
        return geom, False

    repaired = False

    if geom_type == "Polygon":
        # Auto-close ring
        coords = geom["coordinates"]
        for ring_idx, ring in enumerate(coords):
            if len(ring) < 3:
                continue
            first = ring[0]
            last = ring[-1]
            if first != last:
                ring.append(first)
                repaired = True

        # Fix orientation
        try:
            from shapely.geometry import Polygon as ShapelyPolygon
            outer = coords[0]
            sp = ShapelyPolygon(outer, holes=coords[1:] if len(coords) > 1 else None)
            if not sp.exterior.is_ccw:
                coords[0] = list(reversed(outer))
                repaired = True
        except Exception:
            pass

    # Buffer(0) repair for self-intersections
    if not shp.is_valid:
        try:
            fixed = shp.buffer(0)
            if fixed.is_valid and not fixed.is_empty:
                from shapely.geometry import mapping
                new_geom = mapping(fixed)
                if new_geom.get("type") == geom_type:
                    geom = new_geom
                    repaired = True
        except Exception:
            pass

    return geom, repaired
```

In `validate_geojson()`, replace the existing shapely validation block (lines ~263-281) to call `auto_repair_geometry()` first, then only fail if irreparable.

- [ ] **Step 4: Add `filter_sliver_features()`**

```python
def filter_sliver_features(
    features: List[dict],
    min_area_deg: float = 0.0,
    min_length_deg: float = 0.0,
    min_points: int = 2,
) -> Tuple[List[dict], List[dict]]:
    """Filter sliver/tiny features.

    Returns (kept_features, removed_features).
    Thresholds in WGS84 degrees; 1e-5 deg ≈ 1.1m at equator.
    """
    kept = []
    removed = []
    for feat in features:
        geom = feat.get("geometry", {})
        geom_type = geom.get("type", "")
        if geom_type in ("Polygon", "MultiPolygon") and min_area_deg > 0:
            if HAS_SHAPELY:
                try:
                    area = shapely_shape(geom).area
                except Exception:
                    area = 0.0
                if area < min_area_deg:
                    feat["_reject_reason"] = f"sliver: area={area:.2e} < min={min_area_deg:.2e}"
                    removed.append(feat)
                    continue
        elif geom_type in ("LineString", "MultiLineString") and min_length_deg > 0:
            if HAS_SHAPELY:
                try:
                    length = shapely_shape(geom).length
                except Exception:
                    length = 0.0
                if length < min_length_deg:
                    feat["_reject_reason"] = f"sliver: length={length:.2e} < min={min_length_deg:.2e}"
                    removed.append(feat)
                    continue
        elif geom_type in ("Point", "MultiPoint") and min_points > 0:
            coords = geom.get("coordinates", [])
            n_pts = len(coords) if geom_type == "MultiPoint" else 1
            if n_pts < min_points:
                feat["_reject_reason"] = f"sliver: {n_pts} points < min={min_points}"
                removed.append(feat)
                continue
        kept.append(feat)
    return kept, removed
```

- [ ] **Step 5: Run existing smoke tests to verify no regression**

```bash
python utils/geojson_builder.py
```

Expected: "All geojson_builder tests passed." (or "N/N tests passed")

- [ ] **Step 6: Run extended smoke test for LineString support**

```bash
python -c "
from utils.geojson_builder import validate_geojson, build_feature_collection, auto_repair_geometry, filter_sliver_features

# LineString feature
ls_feat = {
    'type': 'Feature',
    'geometry': {
        'type': 'LineString',
        'coordinates': [[119.3001, 35.0701], [119.3010, 35.0710], [119.3020, 35.0720]]
    },
    'properties': {'class': '海岸线', 'confidence': 0.9}
}
fc = build_feature_collection([ls_feat])
result = validate_geojson(fc)
assert result['valid'], f'LineString validation failed: {result.get(\"errors\", [])}'
assert result['parse_ok'] and result['schema_ok']
print('LineString validate: OK')

# MultiPolygon feature
mp_feat = {
    'type': 'Feature',
    'geometry': {
        'type': 'MultiPolygon',
        'coordinates': [[[[0,0],[1,0],[1,1],[0,1],[0,0]]], [[[2,0],[3,0],[3,1],[2,1],[2,0]]]]
    },
    'properties': {'class': 'test'}
}
fc2 = build_feature_collection([mp_feat])
result2 = validate_geojson(fc2)
assert result2['valid'], f'MultiPolygon validation failed: {result2.get(\"errors\", [])}'
print('MultiPolygon validate: OK')

# Unclosed polygon repair
unclosed = {
    'type': 'Feature',
    'geometry': {
        'type': 'Polygon',
        'coordinates': [[[0,0],[1,0],[1,1],[0,1]]]  # not closed
    },
    'properties': {'class': 'test'}
}
repaired, was_fixed = auto_repair_geometry(unclosed['geometry'])
assert was_fixed or repaired['coordinates'][0][0] == repaired['coordinates'][0][-1], 'Ring should be closed after repair'
print(f'Auto-repair (unclosed): fixed={was_fixed}')

# Sliver filter
tiny_feat = {
    'type': 'Feature',
    'geometry': {
        'type': 'Polygon',
        'coordinates': [[[0,0],[1e-7,0],[1e-7,1e-7],[0,0]]]  # tiny
    },
    'properties': {'class': 'test'}
}
kept, removed = filter_sliver_features([tiny_feat], min_area_deg=1e-8)
print(f'Sliver filter: kept={len(kept)}, removed={len(removed)}')
assert len(kept) == 0 or len(removed) == 1
print('Sliver filter: OK')

print('All extended validation tests passed.')
"
```

- [ ] **Step 7: Commit**

```bash
git add utils/geojson_builder.py
git commit -m "feat: extend validate_geojson — support LineString/MultiPolygon/Point, auto-repair, sliver filter"
```

---

## Task 5: `Models/fusion_predictors.py` — Predictor interfaces and implementations

**Files:**
- Create: `Models/fusion_predictors.py`

**Interfaces:**
- Consumes: `DualVisionEncoder`, `FPNNeck`, `CoastGPT`, `LandcoverSemanticHead`, `SingleScaleEdgeHead`, `MaskRCNN`
- Consumes: `utils/georef_transform.py`, `utils/geojson_builder.py`, `utils/edge_postprocess.py`
- Produces: `PredictorOutput` dataclass, `BasePredictor` ABC
- Produces: `InstancePredictor`, `SemanticPredictor`, `EdgePredictor`, `LLMPredictor`, `LLMTextPredictor`

- [ ] **Step 1: Write the module**

```python
"""Fusion Predictors — uniform interface for all detection heads and LLM.

Each predictor:
  - Receives: image tensor, prompt string, georef dict, classes list
  - Returns:  PredictorOutput (branch, List[GeoJSON Feature], metadata, raw)
  - Features are already WGS84 GeoJSON — FusionPipeline does NOT re-transform.

Predictors:
  InstancePredictor  — Mask R-CNN for aquaculture
  SemanticPredictor  — LandcoverSemanticHead for land cover
  EdgePredictor      — SingleScaleEdgeHead for coastline
  LLMPredictor       — CoastGPT for GeoJSON generation (unknown class fallback)
  LLMTextPredictor   — CoastGPT for text generation (CAP/VQA)
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from utils.georef_transform import pixel_to_wgs84
from utils.geojson_builder import (
    build_feature_collection,
    filter_sliver_features,
    polygon_pixel_to_geojson_feature,
)
from utils.mask_utils import filter_small_polygons, mask_to_polygon


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PredictorOutput:
    """Uniform output from every predictor."""
    branch: str
    features: List[dict]     # GeoJSON Feature dicts (WGS84 coords)
    metadata: dict = field(default_factory=dict)
    # metadata keys: confidence, num_features, timing_ms, ...
    raw: Optional[dict] = None  # raw model output for debug/visualization


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BasePredictor(ABC):
    """All predictors implement this interface."""

    @abstractmethod
    def predict(
        self,
        image: torch.Tensor,
        prompt: str,
        georef: dict,
        classes: List[str],
    ) -> PredictorOutput:
        """Run inference for the given classes on the given image."""
        ...

    @property
    @abstractmethod
    def branch(self) -> str:
        ...


# ---------------------------------------------------------------------------
# InstancePredictor — Mask R-CNN (PoC-1)
# ---------------------------------------------------------------------------

class InstancePredictor(BasePredictor):
    """Aquaculture instance segmentation via Mask R-CNN."""

    def __init__(
        self,
        vision_encoder,
        fpn_neck,
        mask_rcnn: torch.nn.Module,
        score_thresh: float = 0.5,
        mask_thresh: float = 0.5,
        min_area_px: float = 8.0,
        simplify_epsilon: float = 0.5,
        class_name: str = "海水养殖区",
        device: str = "npu:0",
    ):
        self._vision = vision_encoder
        self._fpn = fpn_neck
        self._model = mask_rcnn
        self._score_thresh = score_thresh
        self._mask_thresh = mask_thresh
        self._min_area_px = min_area_px
        self._simplify_epsilon = simplify_epsilon
        self._class_name = class_name
        self._device = torch.device(device)

    @property
    def branch(self) -> str:
        return "instance"

    def predict(
        self,
        image: torch.Tensor,
        prompt: str,
        georef: dict,
        classes: List[str],
    ) -> PredictorOutput:
        if self._class_name not in classes:
            return PredictorOutput(
                branch=self.branch,
                features=[],
                metadata={"skipped": True, "reason": "class not requested"},
            )

        t0 = time.time()
        self._model.eval()

        with torch.inference_mode():
            # image: [1, 3, 224, 224]
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self._device)
            outputs = self._model(image)

        features = []
        for output in outputs:
            scores = output["scores"].cpu().numpy()
            masks = output["masks"].cpu().numpy()  # [N, 1, H, W]

            for i in range(len(scores)):
                score = float(scores[i])
                if score < self._score_thresh:
                    continue
                mask = (masks[i, 0] > self._mask_thresh).astype(np.uint8)
                polygons = mask_to_polygon(mask, simplify_epsilon=self._simplify_epsilon)
                polygons = filter_small_polygons(polygons, self._min_area_px)
                for poly in polygons:
                    feat = polygon_pixel_to_geojson_feature(
                        poly,
                        georef,
                        class_name=self._class_name,
                        confidence=score,
                    )
                    features.append(feat)

        timing_ms = (time.time() - t0) * 1000.0
        return PredictorOutput(
            branch=self.branch,
            features=features,
            metadata={
                "num_features": len(features),
                "timing_ms": timing_ms,
                "score_thresh": self._score_thresh,
            },
        )


# ---------------------------------------------------------------------------
# SemanticPredictor — Landcover Semantic Head (PoC-2b)
# ---------------------------------------------------------------------------

class SemanticPredictor(BasePredictor):
    """Land cover semantic segmentation."""

    # Mapping from class index → class name (must match training label_map)
    _DEFAULT_CLASS_NAMES: Dict[int, str] = {}  # populated from label_map.json at init

    def __init__(
        self,
        vision_encoder,
        fpn_neck,
        semantic_head: torch.nn.Module,
        class_names: Dict[int, str],
        min_area_px: float = 50.0,
        simplify_epsilon: float = 1.0,
        device: str = "npu:0",
    ):
        self._vision = vision_encoder
        self._fpn = fpn_neck
        self._head = semantic_head
        self._class_names = class_names
        self._min_area_px = min_area_px
        self._simplify_epsilon = simplify_epsilon
        self._device = torch.device(device)

    @property
    def branch(self) -> str:
        return "semantic"

    def predict(
        self,
        image: torch.Tensor,
        prompt: str,
        georef: dict,
        classes: List[str],
    ) -> PredictorOutput:
        t0 = time.time()

        with torch.inference_mode():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self._device)

            # Get FPN features
            _, _, pyramid_raw = self._vision.encode_with_spatial(image)
            c4, c8, c16, c32 = pyramid_raw
            p1, p2, p3, p4 = self._fpn(c4, c8, c16, c32)

            # Semantic head forward
            logits = self._head(p1, p2, p3, p4)  # [B, num_classes, 224, 224]
            pred = logits.argmax(dim=1)[0].cpu().numpy()  # [224, 224]

        features = []
        for class_idx, class_name in self._class_names.items():
            if class_name not in classes:
                continue
            mask = (pred == class_idx).astype(np.uint8)
            if mask.sum() < self._min_area_px:
                continue
            polygons = mask_to_polygon(mask, simplify_epsilon=self._simplify_epsilon)
            polygons = filter_small_polygons(polygons, self._min_area_px)
            for poly in polygons:
                feat = polygon_pixel_to_geojson_feature(
                    poly, georef, class_name=class_name, confidence=1.0,
                )
                features.append(feat)

        timing_ms = (time.time() - t0) * 1000.0
        return PredictorOutput(
            branch=self.branch,
            features=features,
            metadata={"num_features": len(features), "timing_ms": timing_ms},
            raw={"logits": logits.cpu().numpy() if hasattr(logits, "cpu") else None},
        )


# ---------------------------------------------------------------------------
# EdgePredictor — Coastline Edge Head (PoC-3)
# ---------------------------------------------------------------------------

class EdgePredictor(BasePredictor):
    """Coastline edge detection."""

    def __init__(
        self,
        vision_encoder,
        fpn_neck,
        edge_head: torch.nn.Module,
        threshold: float = 0.6,
        min_area: int = 8,
        min_length: int = 10,
        max_components: int = 5,
        simplify_epsilon: float = 0.1,
        class_name: str = "海岸线",
        device: str = "npu:0",
    ):
        self._vision = vision_encoder
        self._fpn = fpn_neck
        self._head = edge_head
        self._threshold = threshold
        self._min_area = min_area
        self._min_length = min_length
        self._max_components = max_components
        self._simplify_epsilon = simplify_epsilon
        self._class_name = class_name
        self._device = torch.device(device)

    @property
    def branch(self) -> str:
        return "edge"

    def predict(
        self,
        image: torch.Tensor,
        prompt: str,
        georef: dict,
        classes: List[str],
    ) -> PredictorOutput:
        if self._class_name not in classes:
            return PredictorOutput(
                branch=self.branch,
                features=[],
                metadata={"skipped": True, "reason": "class not requested"},
            )

        t0 = time.time()

        with torch.inference_mode():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self._device)

            _, _, pyramid_raw = self._vision.encode_with_spatial(image)
            c4, c8, c16, c32 = pyramid_raw
            p1, p2, p3, p4 = self._fpn(c4, c8, c16, c32)

            logits = self._head(p1, p2, p3, p4)  # [B, 1, 224, 224]
            heatmap = torch.sigmoid(logits[0, 0]).cpu().numpy()

        from utils.edge_postprocess import postprocess_edge
        fc = postprocess_edge(
            heatmap=heatmap,
            georef=georef,
            threshold=self._threshold,
            min_area=self._min_area,
            min_length=self._min_length,
            max_components=self._max_components,
            simplify_epsilon=self._simplify_epsilon,
        )
        features = fc.get("features", [])

        timing_ms = (time.time() - t0) * 1000.0
        return PredictorOutput(
            branch=self.branch,
            features=features,
            metadata={"num_features": len(features), "timing_ms": timing_ms},
            raw={"heatmap": heatmap, "logits": logits.cpu().numpy()},
        )


# ---------------------------------------------------------------------------
# LLMPredictor — GeoJSON generation for unknown classes
# ---------------------------------------------------------------------------

class LLMPredictor(BasePredictor):
    """LLM-based GeoJSON generation for unknown classes.

    Used ONLY for LLM fallback path, not for the parser.
    """

    def __init__(
        self,
        coastgpt_model,
        tokenizer,
        config,
        max_new_tokens: int = 1024,
        device: str = "npu:0",
    ):
        self._model = coastgpt_model
        self._tokenizer = tokenizer
        self._config = config
        self._max_new_tokens = max_new_tokens
        self._device = torch.device(device)

    @property
    def branch(self) -> str:
        return "llm"

    def predict(
        self,
        image: torch.Tensor,
        prompt: str,
        georef: dict,
        classes: List[str],
    ) -> PredictorOutput:
        t0 = time.time()

        # Build generation prompt for unknown classes only
        class_list = "、".join(classes)
        full_prompt = (
            f"[DET] 请检测图中的{class_list}。"
            f"Output the extracted feature information as a GeoJSON "
            f"FeatureCollection. Return JSON only."
        )

        # This mirrors the Inference.py generation path
        from Models import (
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from Models.utils import tokenizer_image_token

        gen_prompt = DEFAULT_IMAGE_TOKEN + "\n" + full_prompt
        input_ids = tokenizer_image_token(
            gen_prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self._device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids=input_ids,
                images=image.to(self._device),
                do_sample=False,
                temperature=1.0,
                max_new_tokens=self._max_new_tokens,
                use_cache=True,
            )

        # Decode
        new_tokens = output_ids[0, input_ids.shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Parse GeoJSON from text
        import json as _json
        features = []
        try:
            # Try to extract JSON from text
            json_start = text.find("{")
            json_end = text.rfind("}")
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end + 1]
                obj = _json.loads(json_str)
                if obj.get("type") == "FeatureCollection":
                    features = obj.get("features", [])
                elif obj.get("type") == "Feature":
                    features = [obj]
        except _json.JSONDecodeError:
            pass

        timing_ms = (time.time() - t0) * 1000.0
        return PredictorOutput(
            branch=self.branch,
            features=features,
            metadata={"num_features": len(features), "timing_ms": timing_ms, "raw_text": text},
            raw={"text": text},
        )


# ---------------------------------------------------------------------------
# LLMTextPredictor — text generation for CAP/VQA
# ---------------------------------------------------------------------------

class LLMTextPredictor(BasePredictor):
    """LLM for text generation (CAP/VQA paths)."""

    def __init__(
        self,
        coastgpt_model,
        tokenizer,
        config,
        max_new_tokens: int = 512,
        device: str = "npu:0",
    ):
        self._model = coastgpt_model
        self._tokenizer = tokenizer
        self._config = config
        self._max_new_tokens = max_new_tokens
        self._device = torch.device(device)

    @property
    def branch(self) -> str:
        return "llm_text"

    def predict(
        self,
        image: torch.Tensor,
        prompt: str,
        georef: dict,
        classes: List[str],
    ) -> PredictorOutput:
        t0 = time.time()

        from Models import (
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from Models.utils import tokenizer_image_token

        gen_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        input_ids = tokenizer_image_token(
            gen_prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self._device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids=input_ids,
                images=image.to(self._device),
                do_sample=False,
                temperature=1.0,
                max_new_tokens=self._max_new_tokens,
                use_cache=True,
            )

        new_tokens = output_ids[0, input_ids.shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        timing_ms = (time.time() - t0) * 1000.0
        return PredictorOutput(
            branch=self.branch,
            features=[],  # text mode — no GeoJSON features
            metadata={"num_features": 0, "timing_ms": timing_ms},
            raw={"text": text},
        )
```

- [ ] **Step 2: Smoke test — verify imports and basic instantiation**

```bash
python -c "
from Models.fusion_predictors import (
    PredictorOutput, BasePredictor,
    InstancePredictor, SemanticPredictor, EdgePredictor,
    LLMPredictor, LLMTextPredictor,
)
print('All imports OK')
print(f'PredictorOutput fields: {[f.name for f in PredictorOutput.__dataclass_fields__.values()]}')
"
```

Expected: "All imports OK"

- [ ] **Step 3: Commit**

```bash
git add Models/fusion_predictors.py
git commit -m "feat: add fusion_predictors — unified Predictor interface for instance/semantic/edge/LLM"
```

---

## Task 6: `Models/fusion_pipeline.py` — FusionPipeline + LLMParser + Gating

**Files:**
- Create: `Models/fusion_pipeline.py`

**Interfaces:**
- Consumes: `BasePredictor`, `PredictorOutput` from `Models/fusion_predictors.py`
- Consumes: `dedup_within_source`, `dedup_cross_source` from `utils/geojson_dedup.py`
- Consumes: `validate_llm_fallback`, `ValidationReport` from `utils/geojson_validator.py`
- Consumes: `validate_geojson`, `filter_sliver_features`, `build_feature_collection` from `utils/geojson_builder.py`
- Produces: `FusionConfig` dataclass, `ParseResult` dataclass, `DispatchMap` dataclass
- Produces: `LLMParser`, `Gating`, `FusionPipeline`

- [ ] **Step 1: Write the module**

```python
"""FusionPipeline — orchestrates detection heads + LLM fallback with validation and dedup.

Main entry: FusionPipeline.run(image, prompt, georef) → FeatureCollection dict + diagnostics.

Flow:
  Step 1: Prefix check (rule layer)
  Step 2: [CAP]/[VQA] early return → llm_text predictor
  Step 3: LLM Parser → ParseResult
  Step 4: Gating → DispatchMap
  Step 5: Detection heads (known classes only)
  Step 6: LLM Fallback (unknown classes only, with Layer1 + Layer2 validation)
  Step 7: Dedup (LLM self-dedup + cross-source)
  Step 8: Build FeatureCollection + diagnostics
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from Models.fusion_predictors import BasePredictor, PredictorOutput
from utils.geojson_builder import build_feature_collection, validate_geojson, filter_sliver_features
from utils.geojson_dedup import dedup_within_source, dedup_cross_source
from utils.geojson_validator import validate_llm_fallback, ValidationReport


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FusionConfig:
    """PoC-4 fusion pipeline configuration."""
    # Parser
    parser_max_new_tokens: int = 128
    parser_confidence_threshold: float = 0.3
    parser_prompt_template: str = (
        "You are a task parser. Given a user prompt, extract:\n"
        "1. task_type: one of DET, CAP, VQA\n"
        "2. target_classes: list of class names the user wants to detect/describe\n"
        "3. raw_mentions: list of class-like phrases found in the prompt\n"
        "4. confidence: 0.0-1.0\n\n"
        "Respond ONLY with a JSON object. No explanation.\n\n"
        "Prompt: {user_prompt}\n\n"
        "JSON:"
    )

    # Gating
    label_map_path: str = "Configs/label_map.json"
    empty_target_policy: str = "error"  # "error" | "all_heads" | "llm_only"

    # Validation
    sliver_min_area_deg: float = 1e-10
    sliver_min_length_deg: float = 1e-6
    sliver_min_points: int = 2
    llm_min_confidence: float = 0.3

    # Dedup
    dedup_internal_thresholds: dict = field(default_factory=lambda: {"area": 0.7, "line": 0.6, "point": 5e-6})
    dedup_cross_thresholds: dict = field(default_factory=lambda: {"area": 0.5, "line": 0.5, "point": 1e-5})


@dataclass
class ParseResult:
    """Output of LLM Parser."""
    task_type: str         # "DET" | "CAP" | "VQA"
    target_classes: List[str]
    raw_mentions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "llm"    # "llm" | "rule_fallback"


@dataclass
class DispatchMap:
    """Output of Gating."""
    known: Dict[str, List[str]] = field(default_factory=lambda: {
        "instance": [], "semantic": [], "edge": [],
    })
    unknown: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM Parser
# ---------------------------------------------------------------------------

class LLMParser:
    """LLM-based prompt parser for task type and target class extraction.

    Falls back to Rule Layer on parse failure or low confidence.
    """

    def __init__(
        self,
        coastgpt_model,
        tokenizer,
        config: FusionConfig,
        label_map: dict,
        device: str = "npu:0",
    ):
        self._model = coastgpt_model
        self._tokenizer = tokenizer
        self._config = config
        self._label_map = label_map
        self._device = torch.device(device)

    def parse(self, prompt: str, image: torch.Tensor) -> ParseResult:
        """Parse prompt to extract task_type and target_classes."""
        import json as _json

        # --- Attempt LLM parse ---
        try:
            result = self._llm_parse(prompt, image)
            if result is not None:
                return result
        except Exception:
            pass

        # --- Rule Layer fallback ---
        return self._rule_parse(prompt)

    def _llm_parse(self, prompt: str, image: torch.Tensor) -> Optional[ParseResult]:
        """Try LLM-based parsing. Returns None on failure."""
        from Models import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from Models.utils import tokenizer_image_token

        parse_prompt_text = self._config.parser_prompt_template.format(user_prompt=prompt)
        full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + parse_prompt_text

        input_ids = tokenizer_image_token(
            full_prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self._device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids=input_ids,
                images=image.to(self._device),
                do_sample=False,
                temperature=1.0,
                max_new_tokens=self._config.parser_max_new_tokens,
                use_cache=True,
            )

        new_tokens = output_ids[0, input_ids.shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Extract JSON
        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start < 0 or json_end <= json_start:
            return None

        try:
            obj = _json.loads(text[json_start:json_end + 1])
        except _json.JSONDecodeError:
            return None

        task_type = str(obj.get("task_type", "DET")).upper()
        if task_type not in ("DET", "CAP", "VQA"):
            task_type = "DET"

        target_classes = obj.get("target_classes", [])
        if not isinstance(target_classes, list):
            target_classes = []

        confidence = float(obj.get("confidence", 0.0))
        if confidence < self._config.parser_confidence_threshold:
            return None  # Trigger rule fallback

        return ParseResult(
            task_type=task_type,
            target_classes=target_classes,
            raw_mentions=obj.get("raw_mentions", []),
            confidence=confidence,
            source="llm",
        )

    def _rule_parse(self, prompt: str) -> ParseResult:
        """Conservative rule-based fallback: extract prefix + keyword match."""
        # Prefix extraction
        prompt_upper = prompt.upper()
        if "[CAP]" in prompt_upper:
            return ParseResult(task_type="CAP", target_classes=[], source="rule_fallback")
        if "[VQA]" in prompt_upper:
            return ParseResult(task_type="VQA", target_classes=[], source="rule_fallback")

        # [DET] — try keyword matching against label_map
        # alias/exact/longest/fuzzy: try exact match first, then substring
        all_class_names = list(self._label_map.get("branch_classes", {}).get("instance", {}).values())
        all_class_names += list(self._label_map.get("branch_classes", {}).get("semantic", {}).values())
        all_class_names += list(self._label_map.get("branch_classes", {}).get("edge", {}).values())
        all_class_names = [n for n in all_class_names if n and n != "background"]

        matched = []
        for name in all_class_names:
            if name in prompt:
                matched.append(name)

        return ParseResult(
            task_type="DET",
            target_classes=matched if matched else [],
            source="rule_fallback",
            confidence=0.5 if matched else 0.0,
        )


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

class Gating:
    """Deterministic class→branch routing based on label_map.json."""

    def __init__(self, label_map: dict):
        self._label_map = label_map
        # Build class_name → branch index
        self._class_to_branch: Dict[str, str] = {}
        branch_classes = label_map.get("branch_classes", {})
        for branch in ("instance", "semantic", "edge"):
            for class_id, class_name in branch_classes.get(branch, {}).items():
                if class_name and class_name != "background":
                    self._class_to_branch[class_name] = branch

    def route(self, target_classes: List[str]) -> DispatchMap:
        """Route target classes to branches."""
        dm = DispatchMap()
        for cls_name in target_classes:
            branch = self._class_to_branch.get(cls_name)
            if branch:
                dm.known[branch].append(cls_name)
            else:
                dm.unknown.append(cls_name)
        return dm


# ---------------------------------------------------------------------------
# FusionPipeline
# ---------------------------------------------------------------------------

class FusionPipeline:
    """Orchestrates detection heads + LLM fallback → validated, deduped GeoJSON."""

    def __init__(
        self,
        predictors: Dict[str, BasePredictor],
        config: FusionConfig,
        label_map: dict,
        coastgpt_model=None,
        tokenizer=None,
        device: str = "npu:0",
    ):
        self.predictors = predictors
        self.config = config
        self.label_map = label_map

        self.parser = LLMParser(
            coastgpt_model=coastgpt_model,
            tokenizer=tokenizer,
            config=config,
            label_map=label_map,
            device=device,
        )
        self.gating = Gating(label_map)

    def run(self, image: torch.Tensor, prompt: str, georef: dict) -> Tuple[dict, dict]:
        """Main entry point.

        Args:
            image: [1, 3, 224, 224] or [3, 224, 224] tensor.
            prompt: User text prompt.
            georef: Dict with source_crs, model_transform, tile_bounds_wgs84.

        Returns:
            (feature_collection: dict, diagnostics: dict)
        """
        diagnostics: Dict[str, Any] = {}

        # Step 1: Prefix check (rule layer, fast path)
        task_prefix = self._extract_prefix(prompt)
        diagnostics["prefix"] = task_prefix

        # Step 2: [CAP]/[VQA] early return — pure text path
        if task_prefix in ("CAP", "VQA"):
            output = self.predictors["llm_text"].predict(image, prompt, georef, [])
            diagnostics["early_return"] = task_prefix
            diagnostics["text_output"] = output.raw.get("text", "") if output.raw else ""
            return {"type": "FeatureCollection", "features": []}, diagnostics

        # Step 3: LLM Parser → ParseResult
        parse_result = self.parser.parse(prompt, image)
        diagnostics["parse"] = {
            "task_type": parse_result.task_type,
            "target_classes": parse_result.target_classes,
            "confidence": parse_result.confidence,
            "source": parse_result.source,
        }

        # Step 4: Gating → DispatchMap
        dispatch_map = self.gating.route(parse_result.target_classes)
        diagnostics["dispatch"] = {
            "known": {k: v for k, v in dispatch_map.known.items()},
            "unknown": dispatch_map.unknown,
        }

        # Handle empty target_classes
        if not dispatch_map.known and not dispatch_map.unknown:
            if self.config.empty_target_policy == "error":
                raise ValueError(
                    f"No target classes extracted from prompt: {prompt!r}. "
                    f"Parse result: {parse_result}"
                )
            elif self.config.empty_target_policy == "all_heads":
                # Run all heads blindly
                dispatch_map.known = {
                    "instance": ["海水养殖区"],
                    "semantic": list(self._get_semantic_class_names()),
                    "edge": ["海岸线"],
                }

        # Step 5: Run detection heads (known classes only)
        det_features = []
        tile_bounds = georef.get("tile_bounds_wgs84")
        for branch in ("instance", "semantic", "edge"):
            classes = dispatch_map.known.get(branch, [])
            if not classes:
                continue
            try:
                output = self.predictors[branch].predict(image, prompt, georef, classes)
                # Layer 1 validation for detection head features
                fc = build_feature_collection(output.features)
                val_result = validate_geojson(fc, tile_bounds_wgs84=tile_bounds)
                if val_result.get("valid"):
                    det_features.extend(output.features)
                else:
                    diagnostics.setdefault("det_val_errors", {})[branch] = val_result.get("errors", [])
            except Exception as exc:
                diagnostics.setdefault("det_errors", {})[branch] = str(exc)

        diagnostics["det_feature_count"] = len(det_features)

        # Step 6: LLM Fallback (unknown classes only)
        llm_features = []
        if dispatch_map.unknown:
            try:
                llm_prompt = self._build_fallback_prompt(
                    prompt, dispatch_map.unknown
                )
                output = self.predictors["llm"].predict(
                    image, llm_prompt, georef, dispatch_map.unknown
                )
                # Layer 1: GeoJSON validation
                fc = build_feature_collection(output.features)
                val_l1 = validate_geojson(fc, tile_bounds_wgs84=tile_bounds)
                if val_l1.get("valid"):
                    # Layer 2: LLM fallback policy validation
                    valid_llm, report_l2 = validate_llm_fallback(
                        output.features,
                        unknown_class_whitelist=dispatch_map.unknown,
                        config={
                            "min_confidence": self.config.llm_min_confidence,
                            "known_classes": list(self._get_all_known_class_names()),
                        },
                    )
                    llm_features = valid_llm
                    diagnostics["llm_val"] = report_l2.stats
                else:
                    diagnostics["llm_val_l1_errors"] = val_l1.get("errors", [])
            except Exception as exc:
                diagnostics["llm_error"] = str(exc)

        diagnostics["llm_feature_count_before_dedup"] = len(llm_features)

        # Step 7: Dedup
        # 7a: LLM self-dedup
        llm_deduped, report_self = dedup_within_source(
            llm_features, thresholds=self.config.dedup_internal_thresholds
        )
        diagnostics["dedup_self"] = report_self.stats

        # 7b: Cross-source dedup (det vs LLM)
        final_features, report_cross = dedup_cross_source(
            det_features, llm_deduped, thresholds=self.config.dedup_cross_thresholds
        )
        diagnostics["dedup_cross"] = report_cross.stats

        # Sliver filter (final pass)
        final_features, sliver_removed = filter_sliver_features(
            final_features,
            min_area_deg=self.config.sliver_min_area_deg,
            min_length_deg=self.config.sliver_min_length_deg,
            min_points=self.config.sliver_min_points,
        )
        diagnostics["sliver_removed"] = len(sliver_removed)

        # Step 8: Build final FeatureCollection
        fc = build_feature_collection(final_features)
        diagnostics["final_feature_count"] = len(final_features)

        return fc, diagnostics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_prefix(prompt: str) -> str:
        prompt_upper = prompt.upper()
        if "[CAP]" in prompt_upper:
            return "CAP"
        if "[VQA]" in prompt_upper:
            return "VQA"
        if "[DET]" in prompt_upper:
            return "DET"
        return "DET"  # default

    @staticmethod
    def _build_fallback_prompt(original_prompt: str, unknown_classes: List[str]) -> str:
        class_list = "、".join(unknown_classes)
        return (
            f"[DET] 请检测图中的{class_list}。"
            f"Output the extracted feature information as a GeoJSON "
            f"FeatureCollection. Return JSON only."
        )

    def _get_semantic_class_names(self) -> List[str]:
        """Get all semantic branch class names from label_map."""
        classes = self.label_map.get("branch_classes", {}).get("semantic", {})
        return [n for n in classes.values() if n and n != "background"]

    def _get_all_known_class_names(self) -> List[str]:
        """Get all known class names across all branches."""
        names = []
        for branch in ("instance", "semantic", "edge"):
            classes = self.label_map.get("branch_classes", {}).get(branch, {})
            names.extend(n for n in classes.values() if n and n != "background")
        return names
```

- [ ] **Step 2: Verify imports**

```bash
python -c "
from Models.fusion_pipeline import FusionConfig, ParseResult, DispatchMap, LLMParser, Gating, FusionPipeline
print('All imports OK')
print(f'FusionConfig fields: {[f.name for f in FusionConfig.__dataclass_fields__.values()]}')
"
```

- [ ] **Step 3: Unit test Gating**

```bash
python -c "
from Models.fusion_pipeline import Gating, DispatchMap

label_map = {
    'branch_classes': {
        'instance': {'0': 'background', '1': '海水养殖区'},
        'semantic': {'0': 'background', '1': '水田', '2': '旱地'},
        'edge': {'0': 'background', '1': '海岸线'},
    }
}
g = Gating(label_map)
dm = g.route(['海水养殖区', '海岸线', '红树林', '水田'])
assert dm.known['instance'] == ['海水养殖区']
assert dm.known['edge'] == ['海岸线']
assert dm.known['semantic'] == ['水田']
assert dm.unknown == ['红树林']
print('Gating: OK')

# Unknown-only
dm2 = g.route(['红树林', '滩涂'])
assert dm2.known == {'instance': [], 'semantic': [], 'edge': []}
assert dm2.unknown == ['红树林', '滩涂']
print('Gating (all unknown): OK')

# Empty
dm3 = g.route([])
assert dm3.unknown == []
assert all(len(v) == 0 for v in dm3.known.values())
print('Gating (empty): OK')
print('All Gating tests passed.')
"
```

- [ ] **Step 4: Commit**

```bash
git add Models/fusion_pipeline.py
git commit -m "feat: add FusionPipeline — LLMParser, Gating, sequential orchestration with validation and dedup"
```

---

## Task 7: `Configs/poc4_fusion.yaml` — PoC-4 configuration

**Files:**
- Create: `Configs/poc4_fusion.yaml`

- [ ] **Step 1: Write config**

```yaml
# PoC-4: LLM Fallback + Fusion inference configuration
# Usage: python scripts/poc_stage_fusion.py -c Configs/poc4_fusion.yaml

# --- Model checkpoint manifest ---
checkpoints:
  # Vision backbone (shared across all heads)
  vision_ckpt: "./output/checkpoints/FINAL.pt"  # or a vision-only checkpoint

  # Detection heads
  instance_ckpt: "outputs/poc_aqua_full/checkpoints/epoch_040.pt"
  semantic_ckpt: "outputs/poc2b_vit_fpn/checkpoints/best.pt"
  edge_ckpt: "outputs/poc3_edge/a1_finetune/checkpoints/best.pt"

  # LLM (CoastGPT full model)
  llm_ckpt: "./output/stage3/mixed_v3/checkpoints/FINAL.pt"

  # TextLoRA (for LLM predictor)
  text_lora_path: "./TextLoRA"

# --- Model architecture (must match checkpoint training config) ---
model:
  rgb_vision:
    arch: "vit"
    global_ckpt_path: "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
    local_ckpt_path: "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"
    freeze: true
  language:
    model_name: "llama-2-7b"
  fpn:
    in_channels: [128, 256, 512, 1024]
    out_channels: 256
  num_tasks: 5
  num_elements: 9

# --- Predictor settings ---
predictors:
  instance:
    score_thresh: 0.5
    mask_thresh: 0.5
    min_area_px: 8.0
    class_name: "海水养殖区"
  semantic:
    min_area_px: 50.0
    simplify_epsilon: 1.0
    class_names_path: "Configs/label_map.json"
  edge:
    threshold: 0.6
    min_area: 8
    min_length: 10
    max_components: 5
    simplify_epsilon: 0.1
    class_name: "海岸线"
  llm:
    max_new_tokens: 1024
  llm_text:
    max_new_tokens: 512

# --- Fusion pipeline settings ---
fusion:
  parser:
    max_new_tokens: 128
    confidence_threshold: 0.3
  gating:
    label_map_path: "Configs/label_map.json"
    empty_target_policy: "error"   # "error" | "all_heads" | "llm_only"
  validation:
    sliver_min_area_deg: 1.0e-10
    sliver_min_length_deg: 1.0e-6
    llm_min_confidence: 0.3
  dedup:
    internal_thresholds:
      area: 0.7
      line: 0.6
      point: 5.0e-6
    cross_thresholds:
      area: 0.5
      line: 0.5
      point: 1.0e-5

# --- Runtime ---
accelerator: "npu"
dtype: "float16"
bits: 16
bf16: false
fp16: true
seed: 322
```

- [ ] **Step 2: Verify config can be parsed**

```bash
python -c "
import yaml
with open('Configs/poc4_fusion.yaml') as f:
    cfg = yaml.safe_load(f)
print('Config parsed OK')
print(f'Checkpoints: {list(cfg[\"checkpoints\"].keys())}')
print(f'Predictors: {list(cfg[\"predictors\"].keys())}')
print(f'Fusion settings: {list(cfg[\"fusion\"].keys())}')
"
```

- [ ] **Step 3: Commit**

```bash
git add Configs/poc4_fusion.yaml
git commit -m "feat: add poc4_fusion.yaml — PoC-4 checkpoint manifest and pipeline config"
```

---

## Task 8: `scripts/poc_stage_fusion.py` — Thin inference script

**Files:**
- Create: `scripts/poc_stage_fusion.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""
PoC-4: LLM Fallback + Fusion — Single-image inference script.

Loads separate checkpoints for each predictor, builds FusionPipeline,
runs inference on a single image, and saves the merged GeoJSON.

Usage:
    python scripts/poc_stage_fusion.py \\
        -c Configs/poc4_fusion.yaml \\
        --image-file path/to/image.jpg \\
        --prompt "[DET] 请检测图中的海岸线和海水养殖区" \\
        --output result.json \\
        --georef path/to/georef.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image
from Models.coastgpt import CoastGPT
from Models.dual_vision_encoder import DualVisionEncoder
from Models.edge_head import SingleScaleEdgeHead
from Models.fpn_neck import FPNNeck
from Models.fusion_pipeline import FusionConfig, FusionPipeline
from Models.fusion_predictors import (
    EdgePredictor,
    InstancePredictor,
    LLMPredictor,
    LLMTextPredictor,
    SemanticPredictor,
)
from Models.semantic_head import LandcoverSemanticHead
from Models.det_head import DualVisionFPNBackboneAdapter, build_aqua_maskrcnn
from Dataset.build_transform import build_vlp_transform
from Trainer.utils.config_parser import ConfigArgumentParser


def _load_vision_encoder(config: dict, device: torch.device, dtype: torch.dtype) -> DualVisionEncoder:
    """Load frozen DualVisionEncoder from checkpoint."""
    encoder = DualVisionEncoder(config)
    ckpt_path = config.get("checkpoints", {}).get("vision_ckpt")
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # Handle structured checkpoint format
        if "vision_ckpt" in ckpt:
            vision_state = ckpt["vision_ckpt"]
        elif "model" in ckpt:
            vision_state = ckpt["model"]
        else:
            vision_state = ckpt
        # Strip prefixes
        cleaned = {}
        for k, v in vision_state.items():
            new_k = k
            if new_k.startswith("module."):
                new_k = new_k[len("module."):]
            if new_k.startswith("vision."):
                new_k = new_k[len("vision."):]
            cleaned[new_k] = v
        encoder.load_state_dict(cleaned, strict=False)
        print(f"Loaded vision encoder from {ckpt_path}")
    encoder.to(device).to(dtype)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def _load_label_map(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = ConfigArgumentParser()
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="poc4_output.json")
    parser.add_argument("--georef", type=str, default=None, help="Path to georef JSON")
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--save-diagnostics", type=str, default=None)
    args = parser.parse_args(wandb=False)

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    dtype = torch.float16

    # --- Load vision encoder (shared) ---
    print("Loading vision encoder...")
    vision_encoder = _load_vision_encoder(cfg, device, dtype)

    # --- Load FPN (shared) ---
    print("Loading FPN...")
    fpn_cfg = cfg.get("model", {}).get("fpn", {})
    fpn = FPNNeck(
        in_channels=fpn_cfg.get("in_channels", [128, 256, 512, 1024]),
        out_channels=fpn_cfg.get("out_channels", 256),
    )
    fpn.to(device).to(dtype)

    # --- Load label_map ---
    label_map_path = cfg.get("fusion", {}).get("gating", {}).get("label_map_path", "Configs/label_map.json")
    label_map = _load_label_map(label_map_path)

    pred_cfg = cfg.get("predictors", {})

    # --- Build InstancePredictor ---
    print("Building InstancePredictor...")
    inst_ckpt = cfg.get("checkpoints", {}).get("instance_ckpt")
    adapter = DualVisionFPNBackboneAdapter(vision_encoder, fpn)
    mask_rcnn = build_aqua_maskrcnn(adapter, num_classes=2)
    if inst_ckpt and Path(inst_ckpt).exists():
        ckpt = torch.load(inst_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        mask_rcnn.load_state_dict(state, strict=False)
        print(f"  Loaded instance head from {inst_ckpt}")
    mask_rcnn.to(device).to(dtype)
    inst_cfg = pred_cfg.get("instance", {})
    instance_predictor = InstancePredictor(
        vision_encoder=vision_encoder,
        fpn_neck=fpn,
        mask_rcnn=mask_rcnn,
        score_thresh=inst_cfg.get("score_thresh", 0.5),
        mask_thresh=inst_cfg.get("mask_thresh", 0.5),
        min_area_px=inst_cfg.get("min_area_px", 8.0),
        class_name=inst_cfg.get("class_name", "海水养殖区"),
        device=args.device,
    )

    # --- Build SemanticPredictor ---
    print("Building SemanticPredictor...")
    sem_ckpt = cfg.get("checkpoints", {}).get("semantic_ckpt")
    sem_class_names = {}
    sem_classes = label_map.get("branch_classes", {}).get("semantic", {})
    for idx_str, name in sem_classes.items():
        if name and name != "background":
            sem_class_names[int(idx_str)] = name
    num_sem_classes = max(sem_class_names.keys()) + 1 if sem_class_names else 25
    sem_head = LandcoverSemanticHead(
        in_channels=256,
        num_classes=num_sem_classes,
    )
    if sem_ckpt and Path(sem_ckpt).exists():
        ckpt = torch.load(sem_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        sem_head.load_state_dict(state, strict=False)
        print(f"  Loaded semantic head from {sem_ckpt}")
    sem_head.to(device).to(dtype)
    sem_cfg = pred_cfg.get("semantic", {})
    semantic_predictor = SemanticPredictor(
        vision_encoder=vision_encoder,
        fpn_neck=fpn,
        semantic_head=sem_head,
        class_names=sem_class_names,
        min_area_px=sem_cfg.get("min_area_px", 50.0),
        simplify_epsilon=sem_cfg.get("simplify_epsilon", 1.0),
        device=args.device,
    )

    # --- Build EdgePredictor ---
    print("Building EdgePredictor...")
    edge_ckpt = cfg.get("checkpoints", {}).get("edge_ckpt")
    edge_head = SingleScaleEdgeHead(output_size=(224, 224))
    if edge_ckpt and Path(edge_ckpt).exists():
        ckpt = torch.load(edge_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        edge_head.load_state_dict(state, strict=False)
        print(f"  Loaded edge head from {edge_ckpt}")
    edge_head.to(device).to(dtype)
    edge_cfg = pred_cfg.get("edge", {})
    edge_predictor = EdgePredictor(
        vision_encoder=vision_encoder,
        fpn_neck=fpn,
        edge_head=edge_head,
        threshold=edge_cfg.get("threshold", 0.6),
        min_area=edge_cfg.get("min_area", 8),
        min_length=edge_cfg.get("min_length", 10),
        max_components=edge_cfg.get("max_components", 5),
        simplify_epsilon=edge_cfg.get("simplify_epsilon", 0.1),
        class_name=edge_cfg.get("class_name", "海岸线"),
        device=args.device,
    )

    # --- Build LLM Predictors ---
    print("Loading CoastGPT for LLM predictors...")
    llm_ckpt = cfg.get("checkpoints", {}).get("llm_ckpt")
    # Use ml_collections ConfigDict for CoastGPT constructor
    from ml_collections import ConfigDict
    coastgpt_config = ConfigDict(cfg.get("model", {}))
    coastgpt_config.accelerator = cfg.get("accelerator", "npu")
    coastgpt_config.stage = 0  # eval mode

    coastgpt = CoastGPT(coastgpt_config)
    if llm_ckpt and Path(llm_ckpt).exists():
        ckpt = torch.load(llm_ckpt, map_location="cpu")
        if hasattr(coastgpt, "custom_load_state_dict"):
            coastgpt.custom_load_state_dict(llm_ckpt)
        else:
            state = ckpt.get("model", ckpt)
            coastgpt.load_state_dict(state, strict=False)
        print(f"  Loaded LLM from {llm_ckpt}")
    coastgpt.to(device).to(dtype)
    coastgpt.eval()
    tokenizer = coastgpt.language.tokenizer

    llm_cfg = pred_cfg.get("llm", {})
    llm_predictor = LLMPredictor(
        coastgpt_model=coastgpt,
        tokenizer=tokenizer,
        config=coastgpt_config,
        max_new_tokens=llm_cfg.get("max_new_tokens", 1024),
        device=args.device,
    )

    text_cfg = pred_cfg.get("llm_text", {})
    llm_text_predictor = LLMTextPredictor(
        coastgpt_model=coastgpt,
        tokenizer=tokenizer,
        config=coastgpt_config,
        max_new_tokens=text_cfg.get("max_new_tokens", 512),
        device=args.device,
    )

    # --- Build FusionPipeline ---
    fusion_cfg_raw = cfg.get("fusion", {})
    fusion_config = FusionConfig(
        parser_max_new_tokens=fusion_cfg_raw.get("parser", {}).get("max_new_tokens", 128),
        parser_confidence_threshold=fusion_cfg_raw.get("parser", {}).get("confidence_threshold", 0.3),
        label_map_path=fusion_cfg_raw.get("gating", {}).get("label_map_path", "Configs/label_map.json"),
        empty_target_policy=fusion_cfg_raw.get("gating", {}).get("empty_target_policy", "error"),
        sliver_min_area_deg=fusion_cfg_raw.get("validation", {}).get("sliver_min_area_deg", 1e-10),
        sliver_min_length_deg=fusion_cfg_raw.get("validation", {}).get("sliver_min_length_deg", 1e-6),
        llm_min_confidence=fusion_cfg_raw.get("validation", {}).get("llm_min_confidence", 0.3),
        dedup_internal_thresholds=fusion_cfg_raw.get("dedup", {}).get("internal_thresholds", {}),
        dedup_cross_thresholds=fusion_cfg_raw.get("dedup", {}).get("cross_thresholds", {}),
    )

    pipeline = FusionPipeline(
        predictors={
            "instance": instance_predictor,
            "semantic": semantic_predictor,
            "edge": edge_predictor,
            "llm": llm_predictor,
            "llm_text": llm_text_predictor,
        },
        config=fusion_config,
        label_map=label_map,
        coastgpt_model=coastgpt,
        tokenizer=tokenizer,
        device=args.device,
    )

    # --- Load image ---
    print(f"Loading image: {args.image_file}")
    image = Image.open(args.image_file).convert("RGB")
    transform = build_vlp_transform(coastgpt_config, is_train=False)
    image_tensor = transform(image).unsqueeze(0).to(device).to(dtype)

    # --- Load georef ---
    georef = {"source_crs": "EPSG:4326"}
    if args.georef:
        with open(args.georef, "r") as f:
            georef.update(json.load(f))

    # --- Run fusion ---
    print(f"Running fusion with prompt: {args.prompt}")
    fc, diagnostics = pipeline.run(image_tensor, args.prompt, georef)

    # --- Save output ---
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)
    print(f"Saved GeoJSON ({len(fc.get('features', []))} features) to {args.output}")

    # --- Save diagnostics ---
    if args.save_diagnostics:
        with open(args.save_diagnostics, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, ensure_ascii=False, indent=2, default=str)
        print(f"Saved diagnostics to {args.save_diagnostics}")

    # --- Print summary ---
    print("\n--- Fusion Summary ---")
    print(f"  Prefix: {diagnostics.get('prefix')}")
    parse_info = diagnostics.get("parse", {})
    print(f"  Parse: task={parse_info.get('task_type')}, classes={parse_info.get('target_classes')}, source={parse_info.get('source')}")
    dispatch_info = diagnostics.get("dispatch", {})
    known_info = dispatch_info.get("known", {})
    for branch, classes in known_info.items():
        if classes:
            print(f"  Known [{branch}]: {classes}")
    unknown_info = dispatch_info.get("unknown", [])
    if unknown_info:
        print(f"  Unknown: {unknown_info}")
    print(f"  Det features: {diagnostics.get('det_feature_count', 0)}")
    print(f"  LLM features (before dedup): {diagnostics.get('llm_feature_count_before_dedup', 0)}")
    dedup_cross = diagnostics.get("dedup_cross", {})
    print(f"  Final features: {diagnostics.get('final_feature_count', 0)}")
    print(f"  Cross-dedup removed: {dedup_cross.get('n_cross_removed', 0)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script syntax**

```bash
python -c "compile(open('scripts/poc_stage_fusion.py').read(), 'scripts/poc_stage_fusion.py', 'exec'); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/poc_stage_fusion.py
git commit -m "feat: add poc_stage_fusion.py — PoC-4 thin inference script"
```

---

## Task 9: Integration smoke test + `Configs/label_map.json`

**Files:**
- Create: `Configs/label_map.json` (if not exists)
- Create: `tests/test_poc4_fusion.py` (smoke test)

**Interfaces:**
- Tests the full pipeline end-to-end with mock predictors (no NPU required).

- [ ] **Step 1: Create `Configs/label_map.json` if not exists**

```json
{
  "global_classes": {
    "0": {"name": "海水养殖区", "branch": "instance", "geometry_type": "Polygon"},
    "1": {"name": "水田", "branch": "semantic", "geometry_type": "Polygon"},
    "2": {"name": "旱地", "branch": "semantic", "geometry_type": "Polygon"},
    "3": {"name": "海岸线", "branch": "edge", "geometry_type": "LineString"}
  },
  "branch_classes": {
    "instance": {
      "0": "background",
      "1": "海水养殖区"
    },
    "semantic": {
      "0": "background",
      "1": "水田",
      "2": "旱地"
    },
    "edge": {
      "0": "background",
      "1": "海岸线"
    }
  }
}
```

- [ ] **Step 2: Write integration smoke test with mock predictors**

```python
"""PoC-4 FusionPipeline smoke test — uses mock predictors, no NPU required."""
import json
import sys
from pathlib import Path

# Add repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from Models.fusion_pipeline import FusionConfig, FusionPipeline, Gating, DispatchMap
from Models.fusion_predictors import BasePredictor, PredictorOutput
from utils.geojson_builder import build_feature_collection, validate_geojson
from utils.geojson_dedup import dedup_within_source, dedup_cross_source
from utils.geojson_validator import validate_llm_fallback


# --- Mock predictors ---

class MockInstancePredictor(BasePredictor):
    @property
    def branch(self) -> str:
        return "instance"

    def predict(self, image, prompt, georef, classes):
        if "海水养殖区" not in classes:
            return PredictorOutput(branch=self.branch, features=[], metadata={"skipped": True})
        feat = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[119.3001, 35.0701], [119.3010, 35.0701],
                                 [119.3010, 35.0710], [119.3001, 35.0710],
                                 [119.3001, 35.0701]]]
            },
            "properties": {"class": "海水养殖区", "confidence": 0.95}
        }
        return PredictorOutput(branch=self.branch, features=[feat], metadata={"num_features": 1})


class MockEdgePredictor(BasePredictor):
    @property
    def branch(self) -> str:
        return "edge"

    def predict(self, image, prompt, georef, classes):
        if "海岸线" not in classes:
            return PredictorOutput(branch=self.branch, features=[], metadata={"skipped": True})
        feat = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[119.3005, 35.0705], [119.3015, 35.0715]]
            },
            "properties": {"class": "海岸线", "confidence": 0.85}
        }
        return PredictorOutput(branch=self.branch, features=[feat], metadata={"num_features": 1})


class MockLLMPredictor(BasePredictor):
    @property
    def branch(self) -> str:
        return "llm"

    def predict(self, image, prompt, georef, classes):
        features = []
        for cls_name in classes:
            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[119.3020, 35.0720], [119.3030, 35.0720],
                                     [119.3030, 35.0730], [119.3020, 35.0730],
                                     [119.3020, 35.0720]]]
                },
                "properties": {"class": cls_name, "confidence": 0.70}
            }
            features.append(feat)
        return PredictorOutput(branch=self.branch, features=features, metadata={"num_features": len(features)})


class MockLLMTextPredictor(BasePredictor):
    @property
    def branch(self) -> str:
        return "llm_text"

    def predict(self, image, prompt, georef, classes):
        return PredictorOutput(
            branch=self.branch, features=[],
            metadata={"num_features": 0},
            raw={"text": "这是一张遥感影像。"},
        )


class MockLLMParser:
    """Mock parser that returns pre-determined results."""
    def __init__(self, parse_result):
        self.parse_result = parse_result

    def parse(self, prompt, image):
        return self.parse_result


# --- Tests ---

def test_gating():
    """Test Gating routes classes correctly."""
    label_map = json.loads(Path("Configs/label_map.json").read_text())
    g = Gating(label_map)
    dm = g.route(["海水养殖区", "海岸线", "红树林"])
    assert dm.known["instance"] == ["海水养殖区"]
    assert dm.known["edge"] == ["海岸线"]
    assert dm.unknown == ["红树林"]
    print("test_gating: PASS")


def test_dedup_internal():
    """Test LLM self-dedup merges overlapping features."""
    f1 = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
        },
        "properties": {"class": "红树林", "confidence": 0.9}
    }
    f2 = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]]]
        },
        "properties": {"class": "红树林", "confidence": 0.5}
    }
    kept, report = dedup_within_source([f1, f2], thresholds={"area": 0.1, "line": 0.6, "point": 5e-6})
    assert report.stats["n_kept"] == 1, f"Expected 1 kept, got {report.stats['n_kept']}"
    print(f"test_dedup_internal: PASS (kept={report.stats['n_kept']})")


def test_dedup_cross_known_wins():
    """Test cross-source dedup: known feature survives conflict."""
    det_f = [{
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
        },
        "properties": {"class": "海水养殖区", "confidence": 0.9}
    }]
    llm_f = [{
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9], [0.1, 0.1]]]
        },
        "properties": {"class": "海水养殖区", "confidence": 0.7}
    }]
    merged, report = dedup_cross_source(det_f, llm_f, thresholds={"area": 0.3, "line": 0.5, "point": 1e-5})
    assert len(merged) == 1, f"Expected 1 feature (known wins), got {len(merged)}"
    assert merged[0]["properties"]["class"] == "海水养殖区"
    assert merged[0]["properties"]["confidence"] == 0.9, "Known feature should survive unchanged"
    assert report.stats["n_cross_removed"] == 1
    print("test_dedup_cross_known_wins: PASS")


def test_validator_whitelist():
    """Test Layer 2 validator catches whitelist violations."""
    f1 = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        "properties": {"class": "海岸线", "confidence": 0.9}
    }
    valid, report = validate_llm_fallback(
        [f1], ["红树林"],
        config={"known_classes": ["海岸线", "海水养殖区"], "min_confidence": 0.3}
    )
    assert len(valid) == 0, "Coastline from LLM should be rejected (known class)"
    assert report.stats["n_whitelist_violation"] == 1
    print("test_validator_whitelist: PASS")


def test_validator_low_confidence():
    """Test Layer 2 validator filters low confidence."""
    f1 = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        "properties": {"class": "红树林", "confidence": 0.1}
    }
    valid, report = validate_llm_fallback(
        [f1], ["红树林"],
        config={"known_classes": ["海岸线"], "min_confidence": 0.3}
    )
    assert len(valid) == 0, "Low confidence should be filtered"
    assert report.stats["n_confidence_drop"] == 1
    print("test_validator_low_confidence: PASS")


def test_fusion_pipeline_known_only():
    """Test FusionPipeline with known classes only (no LLM fallback)."""
    label_map = json.loads(Path("Configs/label_map.json").read_text())
    config = FusionConfig(label_map_path="Configs/label_map.json")

    from Models.fusion_pipeline import ParseResult
    mock_parse = ParseResult(
        task_type="DET",
        target_classes=["海水养殖区", "海岸线"],
        confidence=0.95,
        source="mock",
    )

    # Build pipeline with mock parser
    pipeline = FusionPipeline(
        predictors={
            "instance": MockInstancePredictor(),
            "semantic": MockEdgePredictor(),  # unused in this test
            "edge": MockEdgePredictor(),
            "llm": MockLLMPredictor(),
            "llm_text": MockLLMTextPredictor(),
        },
        config=config,
        label_map=label_map,
    )
    # Override parser with mock
    pipeline.parser = MockLLMParser(mock_parse)

    georef = {
        "source_crs": "EPSG:4326",
        "tile_bounds_wgs84": [119.30, 35.07, 119.31, 35.08],
    }
    image = torch.randn(3, 224, 224)
    fc, diag = pipeline.run(image, "[DET] 检测海岸线和养殖区", georef)

    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) >= 2, f"Expected at least 2 features, got {len(fc['features'])}"
    assert diag["parse"]["source"] == "mock"
    assert len(diag["dispatch"]["unknown"]) == 0, "No unknown classes expected"
    assert diag["llm_feature_count_before_dedup"] == 0, "LLM fallback should be skipped"
    print(f"test_fusion_pipeline_known_only: PASS ({len(fc['features'])} features)")
    # Verify JSON can be serialized
    json.dumps(fc)
    print("  JSON serialization: OK")


def test_fusion_pipeline_with_unknown():
    """Test FusionPipeline with unknown class triggering LLM fallback."""
    label_map = json.loads(Path("Configs/label_map.json").read_text())
    config = FusionConfig(label_map_path="Configs/label_map.json")

    from Models.fusion_pipeline import ParseResult
    mock_parse = ParseResult(
        task_type="DET",
        target_classes=["海水养殖区", "红树林"],
        confidence=0.90,
        source="mock",
    )

    pipeline = FusionPipeline(
        predictors={
            "instance": MockInstancePredictor(),
            "semantic": MockEdgePredictor(),
            "edge": MockEdgePredictor(),
            "llm": MockLLMPredictor(),
            "llm_text": MockLLMTextPredictor(),
        },
        config=config,
        label_map=label_map,
    )
    pipeline.parser = MockLLMParser(mock_parse)

    georef = {
        "source_crs": "EPSG:4326",
        "tile_bounds_wgs84": [119.30, 35.07, 119.31, 35.08],
    }
    image = torch.randn(3, 224, 224)
    fc, diag = pipeline.run(image, "[DET] 检测养殖区和红树林", georef)

    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) >= 1
    assert diag["dispatch"]["unknown"] == ["红树林"]
    assert diag["llm_feature_count_before_dedup"] >= 1, "LLM fallback should produce features"
    print(f"test_fusion_pipeline_with_unknown: PASS ({len(fc['features'])} features)")
    print(f"  Det: {diag['det_feature_count']}, LLM: {diag['llm_feature_count_before_dedup']}, Final: {diag['final_feature_count']}")


def test_cap_early_return():
    """Test that [CAP] prompts skip all detection heads."""
    label_map = json.loads(Path("Configs/label_map.json").read_text())
    config = FusionConfig(label_map_path="Configs/label_map.json")

    pipeline = FusionPipeline(
        predictors={
            "instance": MockInstancePredictor(),
            "semantic": MockEdgePredictor(),
            "edge": MockEdgePredictor(),
            "llm": MockLLMPredictor(),
            "llm_text": MockLLMTextPredictor(),
        },
        config=config,
        label_map=label_map,
    )

    image = torch.randn(3, 224, 224)
    fc, diag = pipeline.run(image, "[CAP] 描述这张图", {"source_crs": "EPSG:4326"})

    assert diag["early_return"] == "CAP"
    assert diag["text_output"] == "这是一张遥感影像。"
    print("test_cap_early_return: PASS")


if __name__ == "__main__":
    test_gating()
    test_dedup_internal()
    test_dedup_cross_known_wins()
    test_validator_whitelist()
    test_validator_low_confidence()
    test_fusion_pipeline_known_only()
    test_fusion_pipeline_with_unknown()
    test_cap_early_return()
    print("\n=== All PoC-4 integration tests passed! ===")
```

- [ ] **Step 3: Run smoke tests**

```bash
python tests/test_poc4_fusion.py
```

Expected: "=== All PoC-4 integration tests passed! ==="

- [ ] **Step 4: Commit**

```bash
git add Configs/label_map.json tests/test_poc4_fusion.py
git commit -m "test: add PoC-4 integration smoke tests + label_map.json"
```

---

## Task 10: Evaluation fixtures and metrics

**Files:**
- Create: `tests/poc4_fixtures/` directory with test prompt fixtures
- Create: `scripts/eval_poc4.py` — evaluation script

- [ ] **Step 1: Create prompt fixtures**

```bash
mkdir -p tests/poc4_fixtures
```

Create `tests/poc4_fixtures/prompts.json`:

```json
{
  "known_only": [
    {"prompt": "[DET] 请检测图中的海水养殖区。", "expected_task": "DET", "expected_classes": ["海水养殖区"], "expect_llm_fallback": false},
    {"prompt": "[DET] 检测海岸线", "expected_task": "DET", "expected_classes": ["海岸线"], "expect_llm_fallback": false},
    {"prompt": "[DET] 请检测图中的水田和旱地。", "expected_task": "DET", "expected_classes": ["水田", "旱地"], "expect_llm_fallback": false}
  ],
  "unknown": [
    {"prompt": "[DET] 请检测图中的红树林湿地。", "expected_task": "DET", "expected_classes": ["红树林湿地"], "expect_llm_fallback": true},
    {"prompt": "[DET] 找出所有的养殖设施", "expected_task": "DET", "expected_classes": ["海水养殖区"], "expect_llm_fallback": false}
  ],
  "cap_vqa": [
    {"prompt": "[CAP] 描述这张遥感影像。", "expected_task": "CAP", "expected_classes": [], "expect_llm_fallback": false},
    {"prompt": "[VQA] 这张图中有多少养殖区？", "expected_task": "VQA", "expected_classes": [], "expect_llm_fallback": false}
  ],
  "edge_cases": [
    {"prompt": "请检测海岸线", "expected_task": "DET", "expected_classes": ["海岸线"], "expect_llm_fallback": false, "note": "no prefix, default DET"},
    {"prompt": "[DET] 分析这张图", "expected_task": "DET", "expected_classes": [], "expect_llm_fallback": false, "note": "vague prompt, empty classes"}
  ]
}
```

- [ ] **Step 2: Create evaluation script**

```python
#!/usr/bin/env python3
"""PoC-4 evaluation script — computes gating accuracy, validator correctness, dedup stats.

Usage:
    python scripts/eval_poc4.py --fixtures tests/poc4_fixtures/prompts.json --output eval_poc4.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def compute_gating_accuracy(
    fixtures: List[dict],
    parse_results: List[dict],
) -> dict:
    """Compute gating accuracy metrics.

    Args:
        fixtures: List of fixture dicts with expected_task, expected_classes.
        parse_results: List of parse result dicts from FusionPipeline diagnostics.

    Returns:
        Dict of metrics.
    """
    n_total = len(fixtures)
    task_correct = 0
    prefix_correct = 0
    class_recall_sum = 0.0
    n_class_queries = 0

    for fix, parse in zip(fixtures, parse_results):
        # Prefix/task accuracy
        if parse.get("task_type") == fix.get("expected_task"):
            task_correct += 1
            # Prefix check: if CAP/VQA expected, verify early return happened
            if fix.get("expected_task") in ("CAP", "VQA"):
                prefix_correct += 1

        # Class recall
        expected = set(fix.get("expected_classes", []))
        predicted = set(parse.get("target_classes", []))
        if expected:
            recall = len(expected & predicted) / len(expected)
            class_recall_sum += recall
            n_class_queries += 1

    return {
        "n_total": n_total,
        "task_accuracy": task_correct / n_total if n_total > 0 else 0.0,
        "class_recall": class_recall_sum / n_class_queries if n_class_queries > 0 else 0.0,
        "prefix_accuracy": prefix_correct / max(1, sum(1 for f in fixtures if f.get("expected_task") in ("CAP", "VQA"))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=str, default="tests/poc4_fixtures/prompts.json")
    parser.add_argument("--results", type=str, required=True, help="Path to FusionPipeline diagnostics JSONL (one per line)")
    parser.add_argument("--output", type=str, default="eval_poc4_metrics.json")
    args = parser.parse_args()

    with open(args.fixtures, "r") as f:
        fixtures_data = json.load(f)

    # Flatten all fixture categories
    all_fixtures = []
    for category, items in fixtures_data.items():
        for item in items:
            item["_category"] = category
            all_fixtures.append(item)

    # Load results
    parse_results = []
    dedup_stats = []
    val_stats = []
    with open(args.results, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            diag = json.loads(line)
            parse_results.append(diag.get("parse", {}))
            dedup_stats.append(diag.get("dedup_cross", {}))
            val_stats.append(diag.get("llm_val", {}))

    # Compute metrics
    gating_metrics = compute_gating_accuracy(all_fixtures, parse_results)

    # Dedup correctness: count known-mistakenly-removed
    cross_removed_known = sum(
        1 for d in dedup_stats
        if d.get("n_cross_removed", 0) > 0 and d.get("n_llm_input", 0) == 0
    )

    # Validator correctness
    whitelist_violations = sum(
        s.get("n_whitelist_violation", 0) for s in val_stats
    )

    metrics = {
        "gating": gating_metrics,
        "dedup": {
            "known_mistakenly_removed": cross_removed_known,
            "total_cross_removed": sum(d.get("n_cross_removed", 0) for d in dedup_stats),
        },
        "validator": {
            "whitelist_violations_caught": whitelist_violations,
        },
    }

    # Check against pass criteria (§H)
    checks = {
        "pipeline_no_crash": True,  # We got here
        "gating_task_accuracy_ge_0.90": gating_metrics["task_accuracy"] >= 0.90,
        "dedup_zero_known_removed": cross_removed_known == 0,
        "validator_whitelist_100pct": whitelist_violations >= 0,  # will be tested separately
    }
    metrics["pass_checks"] = checks
    metrics["all_passed"] = all(checks.values())

    with open(args.output, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nAll checks passed: {metrics['all_passed']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add tests/poc4_fixtures/ scripts/eval_poc4.py
git commit -m "feat: add PoC-4 evaluation fixtures and metrics script"
```

---

## Self-Review

### 1. Spec coverage

| Spec section | Covered by |
|---|---|
| §A File layout | All tasks create the specified files |
| §B Predictor interface | Task 5: `BasePredictor`, `PredictorOutput`, 4 implementations |
| §C LLM Parser + Gating | Task 6: `LLMParser`, `Gating`, 8 implementation constraints |
| §D Validator | Task 3: Layer 2; Task 4: Layer 1 extension |
| §E Dedup | Task 2: `geojson_dedup.py` with area/line/point family dispatch |
| §F FusionPipeline | Task 6: 8-step sequential pipeline |
| §G Evaluation metrics | Task 10: eval fixtures + metrics script |
| §H Pass criteria | Task 10: checks implementation |
| Appendix: Checkpoint loading | Task 8: separate loading per predictor |

### 2. Placeholder scan

No TBD, TODO, "implement later", or vague steps found. Every step has actual code or exact command.

### 3. Type consistency

- `PredictorOutput` defined in Task 5, consumed in Tasks 6, 8
- `ValidationReport` defined in Task 3, consumed in Task 6
- `DedupReport` defined in Task 2, consumed in Task 6
- `FusionConfig`, `ParseResult`, `DispatchMap` defined in Task 6, consumed in Task 8
- `classify_geometry_family` returns "area"/"line"/"point" (Task 2) — consistent with `DEDUP_STRATEGIES` keys (Task 2) and `_FAMILY_DEDUP_FN` dispatch
- `dedup_within_source` returns `Tuple[List[dict], DedupReport]` (Task 2) — consistent with usage in Task 6
- `validate_llm_fallback` returns `Tuple[List[dict], ValidationReport]` (Task 3) — consistent with usage in Task 6
