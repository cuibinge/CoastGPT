"""Evaluation helpers for PoC-3 coastline batch inference."""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np


def parse_threshold_values(raw: Optional[str]) -> List[float]:
    """Parse threshold values from a comma list or start:end:step range."""
    if raw is None or str(raw).strip() == "":
        return []

    text = str(raw).strip()
    if ":" in text:
        parts = [float(p.strip()) for p in text.split(":")]
        if len(parts) != 3:
            raise ValueError("threshold range must use start:end:step")
        start, end, step = parts
        if step <= 0:
            raise ValueError("threshold range step must be positive")
        values = []
        value = start
        epsilon = step / 1000.0
        while value <= end + epsilon:
            values.append(round(value, 6))
            value += step
        return values

    return [float(p.strip()) for p in text.split(",") if p.strip()]


def aggregate_metric_records(records: Iterable[dict]) -> Dict[str, float]:
    """Average numeric metric fields across records."""
    records = list(records)
    if not records:
        return {}

    keys = set()
    for record in records:
        keys.update(record.keys())

    aggregate = {}
    for key in sorted(keys):
        vals = [
            record[key]
            for record in records
            if isinstance(record.get(key), (int, float, np.integer, np.floating))
            and not isinstance(record.get(key), bool)
        ]
        if vals:
            aggregate[key] = float(np.mean(vals))
    return aggregate


def choose_best_thresholds(
    records: Iterable[dict],
    metric: str,
    group_key: Optional[str] = None,
) -> Dict[str, dict]:
    """Choose the threshold with the best mean metric per group."""
    buckets = defaultdict(list)
    for record in records:
        group = str(record.get(group_key, "global")) if group_key else "global"
        if "threshold" not in record or metric not in record:
            continue
        buckets[(group, float(record["threshold"]))].append(record)

    grouped_scores = defaultdict(list)
    for (group, threshold), bucket in buckets.items():
        aggregate = aggregate_metric_records(bucket)
        grouped_scores[group].append({
            "threshold": threshold,
            metric: float(aggregate.get(metric, 0.0)),
            "n_samples": len(bucket),
        })

    best = {}
    for group, options in grouped_scores.items():
        options.sort(key=lambda row: (row.get(metric, 0.0), -row["threshold"]), reverse=True)
        best[group] = options[0]
    return best
