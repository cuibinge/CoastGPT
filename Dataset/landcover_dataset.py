"""
LandcoverSemanticDataset: PyTorch Dataset for partial-label semantic segmentation.

Loads land cover tiles, merges per-class binary TIFs to multi-class target masks
with ignore_index=255 for unlabeled pixels.

Split by source image (derived from filename) to prevent spatial leakage.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Dataset.landcover_tile_grouping import (
    extract_source_image_key,
    scan_landcover_directories,
    group_tiles_by_spatial_key,
    build_merged_samples,
)
from Dataset.landcover_label_map import IGNORE_INDEX, BACKGROUND_ID, num_classes


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LandcoverSemanticDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for land cover semantic segmentation.

    Each sample returns:
        image: Tensor[3, 224, 224] float32, range [0, 1]
        target: Tensor[224, 224] int64, with IGNORE_INDEX=255 for unlabeled pixels
        meta: dict with sample_id, known_classes, etc.
    """

    def __init__(
        self,
        merged_samples: List[dict],
        image_size: int = 224,
        cache_dir: Optional[str] = None,
    ):
        self.image_size = image_size
        self.samples = merged_samples
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if not self.samples:
            warnings.warn("LandcoverSemanticDataset initialized with 0 samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        image = self._load_image(sample["image_path"])

        sample_id = sample["sample_id"]
        cache_path = self.cache_dir / f"{sample_id}.pt" if self.cache_dir else None
        if cache_path and cache_path.exists():
            target = torch.load(str(cache_path), map_location="cpu")["target"]
            target = target.numpy() if isinstance(target, torch.Tensor) else target
        else:
            target = self._build_target_from_tifs(sample)

        georef = self._build_georef(sample)

        meta = {
            "sample_id": sample["sample_id"],
            "image_path": sample["image_path"],
            "known_classes": sample["known_classes"],
            "train_ids": sample["train_ids"],
            "original_size": sample["original_size"],
            "model_input_size": [self.image_size, self.image_size],
            "source_crs": georef["source_crs"],
            "model_transform": georef["model_transform"],
            "sensor": sample.get("sensor", "GF1"),
        }

        target_tensor = torch.from_numpy(target.astype(np.int64))
        return {"image": image, "target": target_tensor, "meta": meta}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_target_from_tifs(self, sample: dict) -> np.ndarray:
        """Load per-class binary TIFs, resize, and merge into multi-class target.

        Each class TIF is a single-channel uint8 image (0/255).
        Resize to model input size with nearest-neighbor to preserve label boundaries.
        Assign train_id where TIF > 0.
        Unlabeled pixels remain IGNORE_INDEX (255).
        Overlapping pixels: last write wins.
        """
        H = W = self.image_size
        target = np.full((H, W), IGNORE_INDEX, dtype=np.uint8)

        for ct in sample["class_tifs"]:
            tif = Image.open(ct["tif_path"])
            if tif.size != (self.image_size, self.image_size):
                tif = tif.resize((W, H), Image.NEAREST)
            binary = np.array(tif, dtype=np.uint8)
            mask = binary > 0
            target[mask] = ct["train_id"]

        return target

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image: resize to model input, normalize to [0,1]."""
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _build_georef(self, sample: dict) -> dict:
        """Build georef dict for coordinate conversion (GeoJSON export)."""
        from Dataset.rasterize_geojson import compute_model_transform_from_bounds

        bounds = sample.get("tile_bounds_wgs84")
        if bounds is not None:
            model_transform = compute_model_transform_from_bounds(
                tuple(bounds), (self.image_size, self.image_size)
            )
        else:
            model_transform = None

        return {
            "source_crs": "EPSG:4326",
            "model_transform": model_transform,
        }

    # ------------------------------------------------------------------
    # Dataset statistics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Compute dataset statistics."""
        total = len(self.samples)
        single_cls = sum(1 for s in self.samples if len(s["known_classes"]) == 1)
        multi_cls = total - single_cls

        by_size: Dict[str, int] = {}
        for s in self.samples:
            sz = f"{s['original_size'][0]}x{s['original_size'][1]}"
            by_size[sz] = by_size.get(sz, 0) + 1

        class_counts: Dict[str, int] = {}
        for s in self.samples:
            for cls_name in s["known_classes"]:
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        return {
            "total": total,
            "single_class_tiles": single_cls,
            "multi_class_tiles": multi_cls,
            "by_size": by_size,
            "class_counts": class_counts,
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def landcover_collate_fn(batch: List[dict]) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    """Collate function: stacks images and targets, leaves metas as list."""
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = torch.stack([item["target"] for item in batch], dim=0)
    metas = [item["meta"] for item in batch]
    return images, targets, metas


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------


def split_by_source_image(
    merged_samples: List[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Split merged samples by source image key to prevent spatial leakage.

    Args:
        merged_samples: Output from build_merged_samples().
        val_ratio: Fraction of unique source images for validation.
        seed: Random seed for reproducibility.

    Returns:
        (train_samples, val_samples) lists.
    """
    key_to_samples: Dict[str, List[dict]] = {}
    for s in merged_samples:
        key = extract_source_image_key(s["image_path"])
        key_to_samples.setdefault(key, []).append(s)

    keys = sorted(key_to_samples.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(keys)

    n_val_keys = max(1, int(len(keys) * val_ratio))
    val_keys = set(keys[:n_val_keys])

    train_samples: List[dict] = []
    val_samples: List[dict] = []

    for key in keys:
        if key in val_keys:
            val_samples.extend(key_to_samples[key])
        else:
            train_samples.extend(key_to_samples[key])

    return train_samples, val_samples


# ---------------------------------------------------------------------------
# Main: build and inspect dataset
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Building land cover dataset from binary TIFs...")
    print("  Scanning directories...")
    raw = scan_landcover_directories()
    print(f"  Scanned {len(raw)} raw samples")

    groups = group_tiles_by_spatial_key(raw)
    print(f"  Grouped into {len(groups)} spatial tiles")

    merged = build_merged_samples(raw, groups)
    print(f"  Merged: {len(merged)} tile samples")

    train_samples, val_samples = split_by_source_image(merged, val_ratio=0.2, seed=42)
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = LandcoverSemanticDataset(train_samples)
    val_ds = LandcoverSemanticDataset(val_samples)

    summary = train_ds.summary()
    print(f"\n  Train summary:")
    print(f"    Total: {summary['total']}")
    print(f"    Multi-class: {summary['multi_class_tiles']}")
    print(f"    By size: {summary['by_size']}")

    # Smoke test: load one sample
    print("\n  Smoke test: loading first sample...")
    item = train_ds[0]
    print(f"    image shape: {item['image'].shape}")
    print(f"    image dtype: {item['image'].dtype}")
    print(f"    target shape: {item['target'].shape}")
    print(f"    target dtype: {item['target'].dtype}")
    print(f"    sample_id: {item['meta']['sample_id']}")
    print(f"    known_classes: {item['meta']['known_classes']}")

    unique = torch.unique(item["target"]).tolist()
    print(f"    target unique: {sorted(unique)}")
    n_ignore = (item["target"] == IGNORE_INDEX).sum().item()
    n_fg = (item["target"] != IGNORE_INDEX).sum().item()
    print(f"    ignore pixels: {n_ignore}, foreground pixels: {n_fg}")

    # Collate smoke test
    print("\n  Smoke test: collate batch of 4...")
    batch = landcover_collate_fn([train_ds[i] for i in range(4)])
    imgs, tgts, metas = batch
    print(f"    images: {imgs.shape}, targets: {tgts.shape}, metas: {len(metas)}")
    assert imgs.shape[0] == 4 and imgs.shape[1] == 3
    assert imgs.shape[2] == imgs.shape[3]  # square
    assert tgts.shape[0] == 4
    assert tgts.shape[1] == imgs.shape[2]  # same spatial dims
    assert tgts.shape[2] == imgs.shape[3]

    print("\nAll landcover_dataset checks passed.")
