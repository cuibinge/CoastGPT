"""
AquaPoCDataset: loads GF2 tiles and converts GeoJSON labels to torchvision
Mask R-CNN detection targets in model pixel space.

PoC-1 aquaculture instance detection pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.georef_transform import (
    clip_pixel_coords,
    pixel_to_wgs84,
    resize_georef,
    round_trip_check,
    wgs84_to_pixel,
)
from utils.mask_utils import bbox_from_mask, rasterize_polygon


class AquaPoCDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset that loads GF2 satellite image tiles and converts
    GeoJSON polygon/multipolygon labels into torchvision Mask R-CNN targets.

    Each sample from the manifest contains:
      - image_path: relative path under data_root
      - label_path: relative path under data_root (GeoJSON FeatureCollection)
      - original_size: [width, height] of the original tile (varies per tile)
      - original_transform: 6-element GDAL affine for WGS84 <-> pixel
      - source_crs: "EPSG:4326"
      - has_object, num_features, geom_types, etc.

    The dataset performs online georef adjustment (resize_georef),
    WGS84-to-pixel coordinate conversion, polygon rasterization, and bbox
    extraction, producing targets directly in model pixel space.
    """

    def __init__(
        self,
        manifest_path: str,
        data_root: str = "/home/ma-user/work/GeoJsonData",
        image_size: int = 224,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        if not isinstance(self.samples, list):
            raise ValueError(f"Manifest must be a JSON list, got {type(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # ---- 1. Load and preprocess image ----
        img_path = self.data_root / sample["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)
        image_tensor = image_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

        # ---- 2. Compute model_transform after resize ----
        original_size = sample["original_size"]  # [width, height]
        original_transform = sample["original_transform"]
        model_transform, resize_scale = resize_georef(
            tuple(original_size),
            (self.image_size, self.image_size),
            original_transform,
        )
        georef = {
            "source_crs": sample["source_crs"],
            "model_transform": model_transform,
        }

        # ---- 3. Parse GeoJSON label ----
        label_path = self.data_root / sample["label_path"]
        raw_num_features = sample.get("num_features", 0)

        boxes: List[List[int]] = []
        masks_list: List[torch.Tensor] = []
        labels: List[int] = []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                geojson = json.load(f)
            features = geojson.get("features", [])

            for feature in features:
                geom = feature.get("geometry")
                if geom is None:
                    continue
                geom_type = geom.get("type")
                coords = geom.get("coordinates")

                if geom_type == "Polygon":
                    # coords = [outer_ring, hole1, hole2, ...]
                    rings = coords
                    pixel_rings = self._polygon_to_pixel(rings, georef)
                    self._add_instance(pixel_rings, boxes, masks_list, labels)

                elif geom_type == "MultiPolygon":
                    # coords = [[outer1, hole1a, ...], [outer2, hole2a, ...], ...]
                    for polygon_rings in coords:
                        pixel_rings = self._polygon_to_pixel(polygon_rings, georef)
                        self._add_instance(pixel_rings, boxes, masks_list, labels)

                # Skip other geometry types (Point, LineString, etc.)
        else:
            # Label file not found - treat as negative sample
            raw_num_features = max(raw_num_features, 1)  # force error trigger below

        # ---- 4. Filter empty instances ----
        valid_indices = []
        for i, m in enumerate(masks_list):
            if m.sum() > 0:
                valid_indices.append(i)

        boxes = [boxes[i] for i in valid_indices]
        masks_list = [masks_list[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]

        # ---- 5. Non-empty features but all masks empty -> ValueError ----
        if raw_num_features > 0 and len(masks_list) == 0:
            raise ValueError(
                f"Sample {sample['sample_id']} has {raw_num_features} features "
                f"but no valid rasterized instances. Check georef or label data."
            )

        # ---- 6. Build target dict (torchvision Mask R-CNN format) ----
        N = len(masks_list)
        if N > 0:
            # Stack masks: list of [H,W] -> [N, H, W]
            masks_stacked = torch.stack(masks_list, dim=0)

            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),  # [N, 4] xyxy
                "labels": torch.tensor(labels, dtype=torch.int64),  # [N]
                "masks": masks_stacked.to(torch.uint8),            # [N, H, W]
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": torch.tensor(
                    [int(m.sum()) for m in masks_list], dtype=torch.float32
                ),  # [N]
                "iscrowd": torch.zeros(N, dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64),
                "masks": torch.empty(
                    (0, self.image_size, self.image_size), dtype=torch.uint8
                ),
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": torch.empty((0,), dtype=torch.float32),
                "iscrowd": torch.empty((0,), dtype=torch.int64),
            }

        # ---- 7. Build meta dict ----
        meta = {
            "image_path": str(img_path),
            "source_crs": sample["source_crs"],
            "original_transform": original_transform,
            "model_transform": model_transform,
            "original_size": original_size,
            "model_input_size": [self.image_size, self.image_size],
            "resize_scale": list(resize_scale),
            "sample_id": sample["sample_id"],
            "tile_bounds_wgs84": sample.get("tile_bounds_wgs84"),
            "sensor": sample.get("sensor"),
            "source_image_id": sample.get("source_image_id"),
        }

        # ---- 8. Return ----
        return {"image": image_tensor, "target": target, "meta": meta}

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _polygon_to_pixel(
        self,
        rings: List[List[List[float]]],
        georef: dict,
    ) -> List[List[Tuple[float, float]]]:
        """Convert GeoJSON Polygon rings from WGS84 to pixel space with clipping."""
        pixel_rings = []
        for ring in rings:
            coords_wgs84 = [(lon, lat) for lon, lat in ring]
            coords_pixel = wgs84_to_pixel(coords_wgs84, georef)
            coords_pixel = clip_pixel_coords(
                coords_pixel, self.image_size, self.image_size
            )
            pixel_rings.append(coords_pixel)
        return pixel_rings

    def _add_instance(
        self,
        pixel_rings: List[List[Tuple[float, float]]],
        boxes: List[List[int]],
        masks_list: List[torch.Tensor],
        labels: List[int],
    ) -> None:
        """Rasterize polygon (outer + holes) and compute bbox.

        Appends results to the mutable `boxes`, `masks_list`, and `labels`
        lists in-place. If the rasterized mask is empty (no pixels), the
        instance is silently skipped.
        """
        outer_ring = pixel_rings[0]
        holes = pixel_rings[1:] if len(pixel_rings) > 1 else None
        mask = rasterize_polygon(
            outer_ring, self.image_size, self.image_size, holes
        )
        mask_tensor = torch.from_numpy(mask)
        bbox = bbox_from_mask(mask)
        if bbox is None:
            return
        boxes.append(bbox)
        masks_list.append(mask_tensor)
        labels.append(1)


# ======================================================================
# Collate function
# ======================================================================

def poc_collate_fn(
    batch: List[dict],
) -> Tuple[List[torch.Tensor], List[dict], List[dict]]:
    """
    Collate function for AquaPoCDataset batches.

    Returns images, targets, and metas as separate lists (no stacking).
    This is compatible with torchvision Mask R-CNN which expects lists
    of variable-size targets.
    """
    images = [item["image"] for item in batch]
    targets = [item["target"] for item in batch]
    metas = [item["meta"] for item in batch]
    return images, targets, metas


# ======================================================================
# Dry-run validation
# ======================================================================

def dataset_dry_run(
    dataset: AquaPoCDataset,
    num_samples: int = 3,
) -> None:
    """
    Validate the dataset pipeline on a few random samples.

    Prints sample_id, image shape, num boxes, labels, first bbox center
    in WGS84, round-trip error for the first feature's coordinates, and
    resize_scale.
    """
    import random

    if len(dataset) == 0:
        print("Dataset is empty, nothing to validate.")
        return

    rng = random.Random(42)
    indices = rng.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        print(f"\n{'='*70}")
        item = dataset[idx]
        meta = item["meta"]
        target = item["target"]
        image = item["image"]

        print(f"  sample_id:      {meta['sample_id']}")
        print(f"  image shape:    {list(image.shape)}")
        print(f"  num boxes:      {len(target['boxes'])}")
        print(f"  labels:         {target['labels'].tolist()}")
        print(f"  resize_scale:   {meta['resize_scale']}")
        print(f"  sensor:         {meta['sensor']}")

        if len(target["boxes"]) > 0:
            first_bbox = target["boxes"][0].tolist()
            center_pixel = [
                (first_bbox[0] + first_bbox[2]) / 2.0,
                (first_bbox[1] + first_bbox[3]) / 2.0,
            ]
            georef = {
                "source_crs": meta["source_crs"],
                "model_transform": meta["model_transform"],
            }
            center_wgs84 = pixel_to_wgs84([tuple(center_pixel)], georef)
            print(f"  first bbox:     {first_bbox}")
            print(f"  bbox center WGS84: ({center_wgs84[0][0]:.8f}, {center_wgs84[0][1]:.8f})")
            mask_area = int(target["masks"][0].sum().item())
            print(f"  first mask area: {mask_area} px")

            # Round-trip check on the first bbox corners
            x1, y1, x2, y2 = first_bbox
            corners_wgs84 = pixel_to_wgs84(
                [(float(x1), float(y1)), (float(x2), float(y2))], georef
            )
            err = round_trip_check(corners_wgs84, georef)
            print(f"  round-trip error: {err:.2e} deg")
        else:
            print(f"  (negative sample - no instances)")

    # Summary statistics
    total_boxes = 0
    positive = 0
    empty = 0
    for i in range(len(dataset)):
        t = dataset[i]["target"]
        if len(t["boxes"]) > 0:
            total_boxes += len(t["boxes"])
            positive += 1
        else:
            empty += 1
    print(f"\n{'='*70}")
    print(f"  Summary: {len(dataset)} samples, {positive} positive, "
          f"{empty} negative, {total_boxes} total instances")
    print(f"{'='*70}")


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="AquaPoCDataset dry-run / validation")
    p.add_argument(
        "--manifest",
        type=str,
        default="data/poc_aqua/train.json",
        help="Path to manifest JSON (relative or absolute)",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default="/home/ma-user/work/GeoJsonData",
        help="Root directory for images and labels",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run dataset validation on random samples",
    )
    p.add_argument("--num-samples", type=int, default=3)
    args = p.parse_args()

    # Resolve manifest path relative to repo root if needed
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = _REPO_ROOT / manifest_path

    ds = AquaPoCDataset(
        manifest_path=str(manifest_path),
        data_root=args.data_root,
        image_size=224,
    )
    print(f"Loaded {len(ds)} samples from {manifest_path}")

    if args.dry_run:
        dataset_dry_run(ds, num_samples=args.num_samples)
