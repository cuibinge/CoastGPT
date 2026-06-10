#!/usr/bin/env python3
"""Pre-build all landcover targets from binary TIFs and save as .pt cache.

Loads per-class binary TIFs, resizes to 224x224 with nearest-neighbor,
merges into multi-class target mask, and saves as {sample_id}.pt.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from Dataset.landcover_tile_grouping import (
    scan_landcover_directories, group_tiles_by_spatial_key, build_merged_samples,
)
from Dataset.landcover_label_map import IGNORE_INDEX

CACHE_DIR = Path("data/landcover_target_cache")


def build_target_from_tifs(class_tifs, image_size=224):
    """Load per-class binary TIFs, resize, merge into target mask."""
    target = np.full((image_size, image_size), IGNORE_INDEX, dtype=np.uint8)
    for ct in class_tifs:
        tif = Image.open(ct["tif_path"])
        if tif.size != (image_size, image_size):
            tif = tif.resize((image_size, image_size), Image.NEAREST)
        binary = np.array(tif, dtype=np.uint8)
        mask = binary > 0
        target[mask] = ct["train_id"]
    return target


def main(image_size=224):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Scanning landcover data (binary TIFs)...", flush=True)
    raw = scan_landcover_directories()
    groups = group_tiles_by_spatial_key(raw)
    merged = build_merged_samples(raw, groups)
    print(f"Total samples: {len(merged)}", flush=True)

    built = 0
    skipped = 0

    for idx, sample in enumerate(merged):
        sample_id = sample["sample_id"]
        cache_path = CACHE_DIR / f"{sample_id}.pt"

        if cache_path.exists():
            skipped += 1
            continue

        target = build_target_from_tifs(sample["class_tifs"], image_size)

        torch.save({
            "target": torch.from_numpy(target.astype(np.int64)),
            "sample_id": sample_id,
        }, str(cache_path))

        built += 1

        if (idx + 1) % 2000 == 0:
            print(f"  Progress: {idx+1}/{len(merged)} (built={built}, skipped={skipped})",
                  flush=True)

    print(f"Done. Built: {built}, Skipped (cached): {skipped}", flush=True)
    print(f"Cache: {CACHE_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', type=int, default=224, help='Model input size')
    args = parser.parse_args()
    main(args.image_size)
