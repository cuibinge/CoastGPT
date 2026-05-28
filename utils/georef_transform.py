"""Coordinate transform utilities for WGS84 <-> model pixel space.

Conventions:
- coords_wgs84: (lon, lat) tuples in EPSG:4326
- coords_pixel: (col, row) in model pixel space (224x224), pixel corner 0-indexed
- transform: GDAL-order affine [a, b, c, d, e, f]
"""
from typing import List, Tuple
import numpy as np

try:
    from affine import Affine
except ImportError:
    Affine = None


def resize_georef(
    original_size: Tuple[int, int],
    target_size: Tuple[int, int],
    original_transform: List[float],
) -> Tuple[List[float], Tuple[float, float]]:
    """
    Compute model_transform after image resize.
    Uses orig_affine * Affine.scale(sx, sy) for rotation/shear compatibility.

    Args:
        original_size: [width, height] of original image
        target_size: [width, height] of model input
        original_transform: GDAL affine [a, b, c, d, e, f]

    Returns:
        model_transform: list[float] of 6 affine coefficients
        resize_scale: (sx, sy)
    """
    if Affine is None:
        raise ImportError("affine library required: pip install affine")

    sx = original_size[0] / target_size[0]
    sy = original_size[1] / target_size[1]

    orig_affine = Affine(*original_transform)
    model_affine = orig_affine * Affine.scale(sx, sy)

    return [model_affine.a, model_affine.b, model_affine.c, model_affine.d, model_affine.e, model_affine.f], (sx, sy)


def wgs84_to_pixel(
    coords_wgs84: List[Tuple[float, float]],
    georef: dict,
) -> List[Tuple[float, float]]:
    """
    Convert WGS84 (lon, lat) -> model pixel (col, row).

    georef must contain:
        model_transform: list[float] of 6 affine coefficients
        source_crs: str, e.g. "EPSG:4326"
    """
    if Affine is None:
        raise ImportError("affine library required: pip install affine")

    if georef.get("source_crs", "").strip().upper() == "EPSG:4326":
        # Direct WGS84 -> pixel (no CRS transform needed)
        model_affine = Affine(*georef["model_transform"])
        inv_affine = ~model_affine
        return [(inv_affine * (lon, lat)) for lon, lat in coords_wgs84]
    else:
        raise NotImplementedError(
            f"CRS transform not implemented for {georef.get('source_crs')}"
        )


def pixel_to_wgs84(
    coords_pixel: List[Tuple[float, float]],
    georef: dict,
) -> List[Tuple[float, float]]:
    """
    Convert model pixel (col, row) -> WGS84 (lon, lat).
    """
    if Affine is None:
        raise ImportError("affine library required: pip install affine")

    if georef.get("source_crs", "").strip().upper() == "EPSG:4326":
        model_affine = Affine(*georef["model_transform"])
        return [model_affine * (col, row) for col, row in coords_pixel]
    else:
        raise NotImplementedError(
            f"CRS transform not implemented for {georef.get('source_crs')}"
        )


def round_trip_check(
    coords_wgs84: List[Tuple[float, float]],
    georef: dict,
) -> float:
    """
    WGS84 -> pixel -> WGS84 round-trip error.
    Returns mean error in degrees.
    Target: < 1e-6 degree.
    """
    coords_pixel = wgs84_to_pixel(coords_wgs84, georef)
    coords_back = pixel_to_wgs84(coords_pixel, georef)

    errors = []
    for (lon1, lat1), (lon2, lat2) in zip(coords_wgs84, coords_back):
        err = np.sqrt((lon1 - lon2)**2 + (lat1 - lat2)**2)
        errors.append(err)

    return float(np.mean(errors))


def clip_pixel_coords(
    coords_pixel: List[Tuple[float, float]],
    width: int = 224,
    height: int = 224,
) -> List[Tuple[float, float]]:
    """Clip pixel coordinates to image bounds [0, width) x [0, height)."""
    return [
        (max(0.0, min(width - 1e-9, col)), max(0.0, min(height - 1e-9, row)))
        for col, row in coords_pixel
    ]


if __name__ == "__main__":
    # GF2-style transform: ~1m/pixel at this latitude
    # 256x256 original, 224x224 model
    original_transform = [1e-5, 0.0, 119.300000, 0.0, -1e-5, 35.072560]
    model_transform, (sx, sy) = resize_georef(
        (256, 256), (224, 224), original_transform
    )

    georef = {
        "source_crs": "EPSG:4326",
        "model_transform": model_transform,
    }

    # Test round trip on tile corners
    test_coords = [
        (119.300000, 35.072560),  # top-left
        (119.302560, 35.072560),  # top-right
        (119.302560, 35.070000),  # bottom-right
        (119.300000, 35.070000),  # bottom-left
    ]

    err = round_trip_check(test_coords, georef)
    print(f"Round-trip error: {err:.2e} degrees")
    assert err < 1e-6, f"Round-trip error too large: {err}"

    # Test pixel conversion
    pixel_coords = wgs84_to_pixel(test_coords, georef)
    print(f"Pixel coords: {pixel_coords}")
    print("All checks passed.")
