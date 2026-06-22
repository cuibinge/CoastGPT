"""Lightweight multi-band TIFF loading helpers.

This file intentionally avoids importing CoastGPT model modules. Dataset tests
can import it without triggering language model, DeepSpeed, or NPU setup.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def _to_chw_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[-1] > 16:
        pass
    elif arr.ndim == 3:
        arr = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"unsupported raster shape: {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _resolve_output_size(output_size: Optional[Union[int, Sequence[int]]]) -> Optional[Tuple[int, int]]:
    if output_size is None:
        return None
    if isinstance(output_size, int):
        return int(output_size), int(output_size)
    if len(output_size) != 2:
        raise ValueError(f"output_size must be int or pair, got {output_size}")
    return int(output_size[0]), int(output_size[1])


def load_multiband_tensor(
    path: Union[Path, str],
    output_size: Optional[Union[int, Sequence[int]]] = None,
    max_channels: int = 4,
) -> Optional[torch.Tensor]:
    """Load TIFF as raw float32 ``[C,H,W]`` tensor.

    Non-TIFF paths return ``None`` so mixed datasets can keep using RGB only.
    Values are clipped to non-negative finite floats, but not normalized; the
    model-side adapter applies robust normalization before projection.
    """

    path = Path(path)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return None
    if max_channels <= 0:
        raise ValueError(f"max_channels must be positive, got {max_channels}")

    import tifffile

    arr = _to_chw_array(tifffile.imread(str(path)))
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    if arr.shape[0] > max_channels:
        arr = arr[:max_channels]
    tensor = torch.from_numpy(arr.copy()).float()

    size = _resolve_output_size(output_size)
    if size is not None and tuple(tensor.shape[-2:]) != size:
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor


def stack_optional_multiband(items: List[Optional[torch.Tensor]]) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Stack optional multi-band tensors and return validity mask."""

    ref = next((item for item in items if torch.is_tensor(item)), None)
    valid = torch.tensor([torch.is_tensor(item) for item in items], dtype=torch.bool)
    if ref is None:
        return None, valid

    stacked = []
    for item in items:
        if torch.is_tensor(item):
            if item.shape != ref.shape:
                raise ValueError(f"multiband tensor shape mismatch: {item.shape} != {ref.shape}")
            stacked.append(item.float())
        else:
            stacked.append(torch.zeros_like(ref, dtype=torch.float32))
    return torch.stack(stacked, dim=0), valid
