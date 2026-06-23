"""Layer 2 validation for LLM fallback GeoJSON features.

Pure rule module — no torch dependency.

Validates:
  1. Class whitelist  — LLM output must not contain known classes
  2. Unknown-class policy — features must be in unknown_classes
  3. Confidence filter — drop low-confidence LLM features
  4. Geometry type policy — (extensible) class->expected geometry type check

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
