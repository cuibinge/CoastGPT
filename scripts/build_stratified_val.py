#!/usr/bin/env python3
"""Build size-stratified train/val split ensuring 128/256/512 val coverage.

Constraints:
  - Split by source_image_key for 128/256 tiles (no spatial leakage)
  - Split by grid parity for 512 tiles (all from 1 source image)
  - All 24 classes represented in val where possible
  - Target: 512 val >= 80 tiles
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from Dataset.landcover_tile_grouping import (
    scan_landcover_directories,
    group_tiles_by_spatial_key,
    build_merged_samples,
    extract_source_image_key,
)


def analyze_split(train: List[dict], val: List[dict]) -> dict:
    """Report per-size and per-class statistics for a split."""
    report = {"by_size": {}, "by_class": defaultdict(lambda: {"train": 0, "val": 0})}

    for tag, samples in [("train", train), ("val", val)]:
        size_cnt = Counter()
        for s in samples:
            sz = f"{s['original_size'][0]}x{s['original_size'][1]}"
            size_cnt[sz] += 1
            for ct in s.get("class_tifs", []):
                report["by_class"][ct["dlmc"]][tag] += 1
        for sz, cnt in size_cnt.items():
            report["by_size"].setdefault(sz, {})[tag] = cnt

    return report


def print_split_report(report: dict):
    """Pretty-print split analysis."""
    print("\n--- By size ---")
    for sz in sorted(report["by_size"].keys()):
        info = report["by_size"][sz]
        total = info.get("train", 0) + info.get("val", 0)
        val_pct = info.get("val", 0) / max(total, 1) * 100
        print(f"  {sz}: train={info.get('train', 0)}, val={info.get('val', 0)} "
              f"({val_pct:.1f}%), total={total}")

    print("\n--- By class (top-level) ---")
    by_class = report["by_class"]
    for cls_name in sorted(by_class.keys()):
        info = by_class[cls_name]
        print(f"  {cls_name}: train={info['train']}, val={info['val']}")

    # Summary
    n_val_classes = sum(1 for info in by_class.values() if info["val"] > 0)
    n_zero_val = sum(1 for info in by_class.values() if info["val"] == 0)
    print(f"\n  Classes with val > 0: {n_val_classes}")
    if n_zero_val > 0:
        missing = [k for k, v in by_class.items() if v["val"] == 0]
        print(f"  Classes with val == 0: {n_zero_val} ({missing})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-512-val", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Path to save train/val sample IDs JSON")
    args = parser.parse_args()

    print("Loading merged samples...")
    raw = scan_landcover_directories()
    groups = group_tiles_by_spatial_key(raw)
    merged = build_merged_samples(raw, groups)
    print(f"  Total merged: {len(merged)}")

    # Group by source image key
    key_to_samples: Dict[str, List[dict]] = defaultdict(list)
    for s in merged:
        key = extract_source_image_key(s["image_path"])
        key_to_samples[key].append(s)

    keys = sorted(key_to_samples.keys())
    print(f"  Unique source images: {len(keys)}")

    # Classify keys by their size composition
    keys_512 = [k for k in keys if any(
        tuple(s["original_size"]) == (512, 512) for s in key_to_samples[k]
    )]
    keys_128_256 = [k for k in keys if k not in keys_512]

    print(f"  Keys with 512 tiles: {len(keys_512)}")
    print(f"  Keys with 128/256 only: {len(keys_128_256)}")

    rng = np.random.RandomState(args.seed)

    # --- Split 128/256 by source image ---
    rng.shuffle(keys_128_256)
    n_val_keys = max(1, int(len(keys_128_256) * args.val_ratio))
    val_keys_128_256 = set(keys_128_256[:n_val_keys])

    # --- Split 512 by grid parity (checkerboard pattern avoids spatial leakage) ---
    train_samples: List[dict] = []
    val_samples: List[dict] = []

    for key in keys_128_256:
        dest = val_samples if key in val_keys_128_256 else train_samples
        dest.extend(key_to_samples[key])

    for key in keys_512:
        val_512 = []
        train_512 = []
        for s in key_to_samples[key]:
            grid_str = s["grid"]  # e.g. "R071C040"
            r_str, c_str = grid_str.replace("R", "").split("C")
            gx, gy = int(r_str), int(c_str)
            if (gx + gy) % 2 == 0:
                val_512.append(s)
            else:
                train_512.append(s)

        # Ensure we have at least min_512_val in val
        if len(val_512) < args.min_512_val:
            # Move some from train to val (alternate pattern)
            needed = args.min_512_val - len(val_512)
            val_512.extend(train_512[:needed])
            train_512 = train_512[needed:]

        train_samples.extend(train_512)
        val_samples.extend(val_512)
        print(f"  512 split: {len(train_512)} train, {len(val_512)} val")

    # --- Report ---
    report = analyze_split(train_samples, val_samples)
    print_split_report(report)

    # --- Save ---
    if args.output:
        output = {
            "train_sample_ids": [s["sample_id"] for s in train_samples],
            "val_sample_ids": [s["sample_id"] for s in val_samples],
            "train_count": len(train_samples),
            "val_count": len(val_samples),
            "val_ratio_128_256": n_val_keys / max(len(keys_128_256), 1),
            "seed": args.seed,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nSaved split to {out_path}")


if __name__ == "__main__":
    main()
