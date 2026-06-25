"""Coastline edge detection Dataset for PoC-3.

Loads coastline tiles, generates edge GT from Binary TIF labels (primary)
or GeoJSON LineString labels (fallback). Returns image + edge target tensors.

Manifest builder scans coastline data directories and emits a JSON manifest
with per-tile metadata, split by source image to prevent spatial leakage.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
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

from utils.georef_transform import resize_georef, wgs84_to_pixel, round_trip_check


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LINE_WIDTH_TRAIN = 3   # Hard band training target width in pixels
LINE_WIDTH_EVAL = 1    # Centerline width for evaluation
DENSIFY_STEP = 0.5     # Max pixel step for polyline densification
IMAGE_SIZE = 224  # Default model input size (override via dataset image_size param)
SOFT_EDGE_SIGMA = 1.0  # Gaussian sigma for soft edge target (A2)
SOFT_EDGE_RADIUS = 3   # Truncation radius for Gaussian kernel (A2)


# ---------------------------------------------------------------------------
# Manifest Builder
# ---------------------------------------------------------------------------


def _extract_tile_name(fname: str) -> str:
    """Extract tile name stem without variant suffix.

    GF coastline naming convention:
      Image:  <stem>_<size>_Orig_WRZ.tif
      Binary: <stem>_<size>_Binary_WRZ.tif
      GeoJSON: <stem>_<size>_Label_WRZ.geojson
    Returns the base stem so labels can be derived as <stem>_Binary_WRZ.tif etc.
    """
    for variant in ('_Orig_WRZ', '_True_WRZ', '_False_WRZ'):
        for ext in ('.tif', '.jpg', '.png'):
            if fname.endswith(variant + ext):
                return fname[:-len(variant + ext)]
    # Fallback: strip extension only
    for ext in ['.tif', '.jpg', '.png']:
        if fname.lower().endswith(ext):
            return fname[:-len(ext)]
    return fname
    return fname


def scan_coastline_directory(root: str) -> List[dict]:
    """Scan coastline directories and emit tile metadata.

    Handles both Level 1 and Level 2 directory structures.

    Expected structure:
      .../Patches/<category>/<sensor>/<size>/Image_Orig/*.tif
      .../Patches/<category>/<sensor>/<size>/Label_GeoJSON/*.geojson
      .../Patches/<category>/<sensor>/<size>/Label_Binary/*.tif
    """
    tiles = []
    root = Path(root)

    for image_path in sorted(root.rglob("Image_Orig/*.tif")):
        if image_path.name.startswith("._"):
            continue

        tile_name = _extract_tile_name(image_path.name)

        # Locate sister label files
        label_dir = image_path.parent.parent
        label_geojson = label_dir / "Label_GeoJSON" / f"{tile_name}_Label_WRZ.geojson"
        label_binary = label_dir / "Label_Binary" / f"{tile_name}_Binary_WRZ.tif"

        # Also check for readable fallback images (JPG in TrueColor/FalseColor dirs)
        image_true_jpg = label_dir / "Image_TrueColor" / f"{tile_name}_True_WRZ.jpg"
        image_false_jpg = label_dir / "Image_FalseColor" / f"{tile_name}_False_WRZ.jpg"

        # Determine tile size from directory name
        size_dir = label_dir.name
        try:
            tile_size = int(size_dir.replace("Size_", ""))
        except ValueError:
            tile_size = 256

        # Determine category from path components
        parts = image_path.parts
        category = "海岸线"
        shoreline_type = ""
        for p in parts:
            if p in (
                "海岸线", "砂质岸线", "基岩岸线", "建设围堤",
                "河口岸线", "港口岸线", "生物岸线", "盐田围堤",
            ):
                category = p
                if p != "海岸线":
                    shoreline_type = p

        # Parse tile name for source-scene and spatial metadata
        source_crs = "EPSG:4326"
        tile_bounds = None

        # GF tile naming: ..._R###C###_<size>_...
        match = re.search(r'_R(\d+)C(\d+)_', tile_name)
        grid_row = int(match.group(1)) if match else None
        grid_col = int(match.group(2)) if match else None

        has_object = False
        num_linestrings = 0
        geojson_valid = False

        if label_geojson.exists():
            try:
                with open(label_geojson) as f:
                    gj = json.load(f)
                if isinstance(gj, dict) and gj.get("type") == "FeatureCollection":
                    features = gj.get("features", [])
                    has_object = len(features) > 0
                    num_linestrings = len(features)
                    geojson_valid = True

                    # Derive tile_bounds from all LineString coordinates
                    if tile_bounds is None and features:
                        all_coords = []
                        for feat in features:
                            geom = feat.get("geometry", {})
                            coords = geom.get("coordinates", [])
                            if geom.get("type") == "LineString":
                                all_coords.extend(coords)
                            elif geom.get("type") == "MultiLineString":
                                for line in coords:
                                    all_coords.extend(line)
                        if all_coords:
                            lons = [pt[0] for pt in all_coords]
                            lats = [pt[1] for pt in all_coords]
                            tile_bounds = [min(lons), min(lats), max(lons), max(lats)]
            except (json.JSONDecodeError, OSError):
                pass

        binary_label_path = str(label_binary) if label_binary.exists() else None
        img_path = str(image_path)

        # Build fallback image paths (readable JPG if TIF fails)
        fallback_img_paths = []
        if image_true_jpg.exists():
            fallback_img_paths.append(str(image_true_jpg))
        if image_false_jpg.exists():
            fallback_img_paths.append(str(image_false_jpg))

        # Derive original_transform from tile_bounds (EPSG:4326, center-of-pixel)
        if tile_bounds is not None:
            w, h = tile_size, tile_size
            x_res = (tile_bounds[2] - tile_bounds[0]) / w
            y_res = (tile_bounds[3] - tile_bounds[1]) / h
            if x_res > 0 and y_res > 0:
                original_transform = [x_res, 0.0, tile_bounds[0], 0.0, -y_res, tile_bounds[3]]
            else:
                original_transform = [1e-5, 0.0, 0.0, 0.0, -1e-5, 0.0]
        else:
            original_transform = [1e-5, 0.0, 0.0, 0.0, -1e-5, 0.0]

        tile = {
            "sample_id": tile_name,
            "image_path": img_path,
            "fallback_image_paths": fallback_img_paths,
            "label_geojson_path": str(label_geojson) if label_geojson.exists() else None,
            "binary_label_path": binary_label_path,
            "category": category,
            "shoreline_type": shoreline_type or category,
            "source_crs": source_crs,
            "original_size": [tile_size, tile_size],
            "model_input_size": [IMAGE_SIZE, IMAGE_SIZE],
            "original_transform": original_transform,
            "tile_bounds_wgs84": tile_bounds,
            "grid_row": grid_row,
            "grid_col": grid_col,
            "has_object": has_object,
            "num_linestrings": num_linestrings,
            "geojson_valid": geojson_valid,
        }
        tiles.append(tile)

    return tiles


def build_coastline_manifest(
    data_roots: List[str],
    output_path: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """Build a manifest JSON from coastline data directories.

    Splits by source image (from tile name) to prevent spatial leakage.

    Returns:
        dict with 'train' and 'val' lists of tile dicts.
    """
    rng = np.random.RandomState(seed)
    all_tiles = []

    for root in data_roots:
        tiles = scan_coastline_directory(root)
        all_tiles.extend(tiles)
        print(f"  {root}: {len(tiles)} tiles")

    print(f"Total: {len(all_tiles)} tiles")

    # Group by source image (extracted from tile name prefix before _R...C...)
    source_groups: Dict[str, List[dict]] = {}
    for tile in all_tiles:
        name = tile["sample_id"]
        match = re.search(r'^(.*)_R\d+C\d+', name)
        if match:
            source_key = match.group(1)
        else:
            source_key = name
        source_groups.setdefault(source_key, []).append(tile)

    source_keys = sorted(source_groups.keys())
    rng.shuffle(source_keys)

    n_val = max(1, int(len(source_keys) * val_ratio))
    val_keys = set(source_keys[:n_val])
    train_keys = set(source_keys[n_val:])

    train_tiles = []
    val_tiles = []
    for key, tiles in source_groups.items():
        if key in val_keys:
            val_tiles.extend(tiles)
        else:
            train_tiles.extend(tiles)

    print(f"Split: {len(train_tiles)} train, {len(val_tiles)} val "
          f"({len(train_keys)}/{len(val_keys)} source groups)")

    manifest = {"train": train_tiles, "val": val_tiles}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Manifest saved to {output_path}")

    return manifest


# ---------------------------------------------------------------------------
# Edge target generation
# ---------------------------------------------------------------------------


def densify_polyline(
    points: List[Tuple[float, float]],
    max_step: float = DENSIFY_STEP,
) -> List[Tuple[float, float]]:
    """Densify a polyline so consecutive points are within max_step pixels.

    Args:
        points: List of (col, row) pixel coordinates.
        max_step: Maximum step distance in pixels.

    Returns:
        Densified list of (col, row) coordinates.
    """
    if len(points) < 2:
        return points

    dense = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        dense.append(p0)

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dist = math.sqrt(dx * dx + dy * dy)
        n = max(1, int(math.ceil(dist / max_step)))

        for j in range(1, n):
            t = j / n
            dense.append((p0[0] + t * dx, p0[1] + t * dy))

    dense.append(points[-1])
    return dense


def draw_edge_map(
    coords: List[Tuple[float, float]],
    size: Tuple[int, int] = (224, 224),
    line_width: int = LINE_WIDTH_TRAIN,
) -> np.ndarray:
    """Rasterize a polyline into an edge map.

    Uses skimage.draw.line_aa for anti-aliased line drawing, then
    binary_dilation for line width.

    Args:
        coords: Pixel coordinates (col, row).
        size: (H, W) of output edge map.
        line_width: Width of drawn edge in pixels.

    Returns:
        edge_map: [H, W] float32 binary map.
    """
    from skimage.draw import line_aa
    from scipy.ndimage import binary_dilation

    edge_map = np.zeros(size, dtype=np.float32)
    half_w = max(1, line_width // 2)

    for i in range(len(coords) - 1):
        c0, r0 = coords[i]
        c1, r1 = coords[i + 1]

        rr, cc, val = line_aa(int(r0), int(c0), int(r1), int(c1))

        valid = (rr >= 0) & (rr < size[1]) & (cc >= 0) & (cc < size[0])
        rr, cc, val = rr[valid], cc[valid], val[valid]

        edge_map[rr, cc] = np.maximum(edge_map[rr, cc], val)

    if line_width > 1:
        y, x = np.ogrid[-half_w:half_w + 1, -half_w:half_w + 1]
        kernel = (x * x + y * y) <= half_w * half_w
        edge_map = binary_dilation(edge_map > 0, structure=kernel).astype(np.float32)

    return np.clip(edge_map, 0.0, 1.0)


def _hard_to_soft_edge_target(
    hard_target: np.ndarray,
    sigma: float = SOFT_EDGE_SIGMA,
    radius: int = SOFT_EDGE_RADIUS,
) -> np.ndarray:
    """Convert hard binary edge target to soft Gaussian edge target (A2).

    Computes distance transform from every pixel to the nearest edge pixel,
    then applies exp(-d² / 2σ²), truncated to 0 beyond `radius` pixels.

    Edge pixels (distance=0) get value 1.0; pixels farther than radius get 0.0.
    This provides a softer learning signal and encourages the model to produce
    responses that fade with distance from the true coastline.

    Args:
        hard_target: [H, W] float32 binary edge map in [0, 1].
        sigma: Gaussian standard deviation in pixels (default 1.0).
        radius: Truncation radius in pixels (default 3).

    Returns:
        soft_target: [H, W] float32 soft edge map in [0, 1].
    """
    from scipy.ndimage import distance_transform_edt

    hard_binary = (hard_target > 0.5).astype(np.uint8)

    # Distance from each pixel to the nearest foreground (edge) pixel
    d_fg = distance_transform_edt(1 - hard_binary)

    # Soft target: decays with distance from edge in Gaussian profile
    # Foreground pixels (d_fg=0) get value 1.0, falling to 0 at `radius` px
    soft = np.exp(-(d_fg ** 2) / (2.0 * sigma * sigma + 1e-8))
    soft = np.where(d_fg <= radius, soft, 0.0)

    return soft.astype(np.float32)


def generate_edge_target_from_geojson(
    label_geojson_path: str,
    georef: dict,
    line_width: int = LINE_WIDTH_TRAIN,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Generate edge GT from GeoJSON LineString labels."""
    with open(label_geojson_path) as f:
        gj = json.load(f)

    sz = float(image_size)
    edge_map = np.zeros((image_size, image_size), dtype=np.float32)

    for feat in gj.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") not in ("LineString", "MultiLineString"):
            continue

        coords_wgs84 = geom.get("coordinates", [])
        if geom.get("type") == "LineString":
            lines = [coords_wgs84]
        else:
            lines = coords_wgs84

        for line in lines:
            if len(line) < 2:
                continue
            coords_pixel = wgs84_to_pixel(
                [(lon, lat) for lon, lat in line], georef
            )
            coords_pixel = [
                (max(0.0, min(sz - 0.001, col)), max(0.0, min(sz - 0.001, row)))
                for col, row in coords_pixel
            ]
            coords_pixel = densify_polyline(coords_pixel)
            line_map = draw_edge_map(coords_pixel, (image_size, image_size), line_width)
            edge_map = np.maximum(edge_map, line_map)

    return edge_map


def generate_edge_target_from_binary_tif(
    binary_tif_path: str,
    line_width: int = LINE_WIDTH_TRAIN,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Generate edge GT from Binary TIF label (PRIMARY GT SOURCE)."""
    binary = Image.open(binary_tif_path).convert("L")
    binary = binary.resize((image_size, image_size), Image.NEAREST)
    binary_np = (np.array(binary) > 128).astype(np.float32)

    if line_width > 1 and binary_np.sum() > 0:
        from skimage.morphology import skeletonize
        from scipy.ndimage import binary_dilation

        skeleton = skeletonize(binary_np.astype(bool))
        half_w = line_width // 2
        y, x = np.ogrid[-half_w:half_w + 1, -half_w:half_w + 1]
        kernel = (x * x + y * y) <= half_w * half_w
        binary_np = binary_dilation(skeleton, structure=kernel).astype(np.float32)

    return binary_np


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CoastlineEdgeDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for coastline edge detection.

    Each sample returns:
        image: Tensor[3, 224, 224] float32, range [0, 1]
        target: Tensor[1, 224, 224] float32, edge target
        meta: dict with sample_id, georef, etc.
    """

    def __init__(
        self,
        tiles: List[dict],
        line_width: int = LINE_WIDTH_TRAIN,
        soft_edge_sigma: float = 0.0,
        soft_edge_radius: int = SOFT_EDGE_RADIUS,
        image_size: int = IMAGE_SIZE,
        unified_label_dir: Optional[str] = None,
    ):
        self.tiles = tiles
        self.line_width = line_width
        self.soft_edge_sigma = soft_edge_sigma
        self.soft_edge_radius = soft_edge_radius
        self.image_size = image_size
        self.unified_label_dir = Path(unified_label_dir) if unified_label_dir else None

        if not self.tiles:
            warnings.warn("CoastlineEdgeDataset initialized with 0 tiles")

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> dict:
        tile = self.tiles[idx]

        image = self._load_image(tile)
        georef = self._build_georef(tile)

        # P3-L: Unified labels from GeoJSON (single source of truth)
        sz = self.image_size
        if self.unified_label_dir is not None:
            sid = tile.get("sample_id", "")
            soft_path = self.unified_label_dir / "edge_soft" / f"{sid}.npy"
            if soft_path.exists():
                target = np.load(soft_path).astype(np.float32)
                if target.shape != (sz, sz):
                    from PIL import Image
                    target = np.array(Image.fromarray((target*255).astype(np.uint8)).resize((sz, sz), Image.NEAREST)).astype(np.float32) / 255.0
            else:
                target = np.zeros((sz, sz), dtype=np.float32)
        elif tile.get("binary_label_path"):
            try:
                target = generate_edge_target_from_binary_tif(
                    tile["binary_label_path"], self.line_width, sz
                )
            except Exception:
                target = np.zeros((sz, sz), dtype=np.float32)
        elif tile.get("label_geojson_path"):
            try:
                target = generate_edge_target_from_geojson(
                    tile["label_geojson_path"], georef, self.line_width, sz
                )
            except Exception:
                target = np.zeros((sz, sz), dtype=np.float32)
        else:
            target = np.zeros((sz, sz), dtype=np.float32)

        # A2: Convert hard target to soft Gaussian edge target
        if self.soft_edge_sigma > 0 and target.sum() > 0:
            target = _hard_to_soft_edge_target(
                target, sigma=self.soft_edge_sigma, radius=self.soft_edge_radius,
            )

        meta = {
            "sample_id": tile["sample_id"],
            "image_path": tile["image_path"],
            "source_crs": tile.get("source_crs", "EPSG:4326"),
            "original_transform": tile.get("original_transform", [1e-5, 0, 0, 0, -1e-5, 0]),
            "model_transform": georef["model_transform"],
            "original_size": tile.get("original_size", [128, 128]),
            "model_input_size": [self.image_size, self.image_size],
            "resize_scale": georef["resize_scale"],
            "has_edge": bool(tile.get("has_object", False)),
            "num_linestrings": tile.get("num_linestrings", 0),
        }

        return {
            "image": torch.from_numpy(image).float(),
            "target": torch.from_numpy(target).float().unsqueeze(0),
            "meta": meta,
        }

    def _load_image(self, tile: dict) -> np.ndarray:
        """Load 4-band multispectral image and resize to self.image_size x self.image_size.

        Primary: tifffile reads the 4-band Image_Orig TIF (R,G,B,NIR).
        Fallback: PIL reads TrueColor/FalseColor JPG (3-band RGB).

        Returns:
            arr: [3, 224, 224] float32 in [0, 1].
        """
        primary_path = tile.get("image_path", "")

        # Try tifffile for 4-band TIF first
        try:
            import tifffile
            tifffile_logger = logging.getLogger("tifffile")
            orig_level = tifffile_logger.level
            tifffile_logger.setLevel(logging.ERROR)
            try:
                arr = tifffile.imread(primary_path)
            finally:
                tifffile_logger.setLevel(orig_level)
            # arr: [H, W, 4] (R, G, B, NIR) — take RGB channels
            if arr.ndim == 3 and arr.shape[-1] >= 3:
                arr = arr[..., :3]
            arr = self._resize_array(arr)
            arr = np.clip(arr, 0, None)
            if arr.max() > 1.0:
                arr = arr / 255.0
            arr = np.clip(arr, 0.0, 1.0)
            return arr.transpose(2, 0, 1).astype(np.float32)
        except Exception:
            pass

        # Fallback: try PIL on JPG fallbacks
        for fb_path in tile.get("fallback_image_paths", []):
            try:
                img = Image.open(fb_path).convert("RGB")
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                return arr.transpose(2, 0, 1)
            except Exception:
                continue

        # Last resort: try PIL on primary path (works for some TIFs)
        try:
            img = Image.open(primary_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            return arr.transpose(2, 0, 1)
        except Exception:
            pass

        # Return zeros if nothing works
        warnings.warn(f"Could not load image for {tile.get('sample_id', '?')}")
        return np.zeros((3, self.image_size, self.image_size), dtype=np.float32)

    def _resize_array(self, arr: np.ndarray) -> np.ndarray:
        """Resize a [H, W, C] array to image_size via PIL."""
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.array(img, dtype=np.float32)

    def _build_georef(self, tile: dict) -> dict:
        """Build georef dict with model_transform and resize_scale."""
        original_size = tile.get("original_size", [128, 128])
        original_transform = tile.get("original_transform", [1e-5, 0, 0, 0, -1e-5, 0])

        if original_transform is None or len(original_transform) < 6:
            bounds = tile.get("tile_bounds_wgs84")
            if bounds:
                w, h = original_size
                x_res = (bounds[2] - bounds[0]) / w
                y_res = (bounds[3] - bounds[1]) / h
                original_transform = [x_res, 0.0, bounds[0], 0.0, -y_res, bounds[3]]
            else:
                original_transform = [1e-5, 0.0, 0.0, 0.0, -1e-5, 0.0]

        model_transform, (sx, sy) = resize_georef(
            (original_size[0], original_size[1]),
            (self.image_size, self.image_size),
            original_transform,
        )

        return {
            "source_crs": tile.get("source_crs", "EPSG:4326"),
            "model_transform": model_transform,
            "resize_scale": (sx, sy),
        }


def coastline_collate_fn(batch: List[dict]) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    """Collate function for CoastlineEdgeDataset."""
    images = torch.stack([item["image"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    metas = [item["meta"] for item in batch]
    return images, targets, metas


# ---------------------------------------------------------------------------
# Smoketest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("CoastlineEdgeDataset smoketest")
    print("=" * 60)

    roots = [
        "/home/ma-user/work/Stage3Data/海岸线/RS-海岸线二级/Patches",
        "/home/ma-user/work/Stage3Data/海岸线/RS-海岸线一级/Patches",
    ]

    # 1. Build manifest
    print("\n[1] Building manifest...")
    manifest = build_coastline_manifest(
        roots,
        output_path="/tmp/poc3_coastline_manifest_test.json",
        val_ratio=0.2,
    )
    print(f"  Train tiles: {len(manifest['train'])}")
    print(f"  Val tiles:   {len(manifest['val'])}")

    # 2. Spot-check manifest structure
    print("\n[2] Spot-check manifest entries...")
    for split_name, tiles in [("train", manifest["train"]), ("val", manifest["val"])]:
        for tile in tiles[:2]:
            has_bin = tile.get("binary_label_path") is not None
            has_gj = tile.get("label_geojson_path") is not None
            print(f"  [{split_name}] {tile['sample_id']}  "
                  f"Binary={'OK' if has_bin else '--'}  "
                  f"GeoJSON={'OK' if has_gj else '--'}  "
                  f"has_obj={tile['has_object']}")

    # 3. Load dataset
    print("\n[3] Loading dataset (first 4 train tiles)...")
    ds = CoastlineEdgeDataset(manifest["train"][:4], line_width=LINE_WIDTH_TRAIN)
    print(f"  len(ds) = {len(ds)}")

    # 4. Inspect first sample
    print("\n[4] Inspecting sample 0...")
    sample = ds[0]
    img = sample["image"]
    tgt = sample["target"]
    meta = sample["meta"]
    print(f"  image:  shape={tuple(img.shape)}  dtype={img.dtype}  "
          f"range=[{img.min().item():.3f}, {img.max().item():.3f}]")
    print(f"  target: shape={tuple(tgt.shape)}  dtype={tgt.dtype}  "
          f"fg_ratio={tgt.mean().item():.4f}")
    print(f"  meta:   sample_id={meta['sample_id']}  "
          f"has_edge={meta['has_edge']}  "
          f"resize_scale=({meta['resize_scale'][0]:.3f}, {meta['resize_scale'][1]:.3f})")

    # 5. Verify all 4 samples load without error and have valid shapes
    print("\n[5] Verifying all 4 samples...")
    all_ok = True
    for i in range(len(ds)):
        s = ds[i]
        img_shape = tuple(s["image"].shape)
        tgt_shape = tuple(s["target"].shape)
        ok = (img_shape == (3, 224, 224)) and (tgt_shape == (1, 224, 224))
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{i}] {s['meta']['sample_id']}: image={img_shape} target={tgt_shape} {status}")
    assert all_ok, "Shape check FAILED"

    # 6. Test collate function
    print("\n[6] Testing collate_fn...")
    images, targets, metas = coastline_collate_fn([ds[i] for i in range(min(len(ds), 4))])
    print(f"  batched images:  {tuple(images.shape)}")
    print(f"  batched targets: {tuple(targets.shape)}")
    print(f"  metas:           {len(metas)} entries")
    assert images.ndim == 4, "Batched images must be 4D"
    assert targets.ndim == 4, "Batched targets must be 4D"

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
