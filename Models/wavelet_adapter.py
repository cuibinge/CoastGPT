"""Small DWT-based adapter utilities for remote-sensing ablations.

This module is intentionally self-contained: importing it must not import the
rest of CoastGPT, because the main model package can trigger heavyweight NPU
and DeepSpeed setup during unit tests.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
HighFreq = Tuple[Tensor, Tensor, Tensor]


def compute_dwt_levels(
    source_gsd: float,
    target_gsd: float,
    policy: str = "nearest",
    spatial_shape: Optional[Sequence[int]] = None,
    min_size: int = 4,
    max_levels: Optional[int] = None,
) -> Tuple[int, float]:
    """Return dyadic DWT levels and resulting effective GSD.

    DWT can only change GSD by powers of two. For non-power ratios such as
    0.8m -> 8m, callers must choose whether to undershoot, overshoot, or use
    the nearest dyadic level.
    """

    if source_gsd <= 0:
        raise ValueError(f"source_gsd must be positive, got {source_gsd}")
    if target_gsd <= 0:
        raise ValueError(f"target_gsd must be positive, got {target_gsd}")
    if min_size <= 0:
        raise ValueError(f"min_size must be positive, got {min_size}")

    ratio = target_gsd / source_gsd
    if ratio <= 1.0:
        levels = 0
    else:
        raw_level = math.log2(ratio)
        if policy == "floor":
            levels = math.floor(raw_level)
        elif policy == "ceil":
            levels = math.ceil(raw_level)
        elif policy == "nearest":
            levels = round(raw_level)
        else:
            raise ValueError(f"unsupported level policy: {policy}")
        levels = max(0, int(levels))

    if spatial_shape is not None and levels > 0:
        h, w = int(spatial_shape[-2]), int(spatial_shape[-1])
        min_hw = max(1, min(h, w))
        shape_limited = max(0, int(math.floor(math.log2(min_hw / float(min_size)))))
        levels = min(levels, shape_limited)

    if max_levels is not None:
        levels = min(levels, max(0, int(max_levels)))

    return levels, float(source_gsd * (2 ** levels))


def _as_batched(x: Tensor) -> Tuple[Tensor, bool]:
    if x.ndim == 3:
        return x.unsqueeze(0), True
    if x.ndim == 4:
        return x, False
    raise ValueError(f"expected [C,H,W] or [B,C,H,W], got shape {tuple(x.shape)}")


def _restore_batch(x: Tensor, squeezed: bool) -> Tensor:
    return x.squeeze(0) if squeezed else x


def pad_to_multiple(x: Tensor, multiple: int) -> Tuple[Tensor, Tuple[int, int]]:
    """Pad H/W on bottom/right so both are divisible by ``multiple``."""

    if multiple <= 1:
        return x, (int(x.shape[-2]), int(x.shape[-1]))

    xb, squeezed = _as_batched(x)
    original_hw = (int(xb.shape[-2]), int(xb.shape[-1]))
    pad_h = (multiple - original_hw[0] % multiple) % multiple
    pad_w = (multiple - original_hw[1] % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, original_hw

    mode = "reflect"
    if xb.shape[-2] == 1 or xb.shape[-1] == 1:
        mode = "replicate"
    padded = F.pad(xb, (0, pad_w, 0, pad_h), mode=mode)
    return _restore_batch(padded, squeezed), original_hw


def crop_to_hw(x: Tensor, hw: Tuple[int, int]) -> Tensor:
    """Crop a tensor back to the supplied H/W."""

    h, w = int(hw[0]), int(hw[1])
    return x[..., :h, :w]


def haar_dwt2(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """One-level orthonormal Haar DWT for even-sized tensors."""

    if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
        raise ValueError(f"haar_dwt2 requires even H/W, got {tuple(x.shape[-2:])}")

    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]

    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, lh, hl, hh


def haar_idwt2(ll: Tensor, lh: Tensor, hl: Tensor, hh: Tensor) -> Tensor:
    """Inverse of :func:`haar_dwt2`."""

    out_shape = list(ll.shape)
    out_shape[-2] *= 2
    out_shape[-1] *= 2
    out = ll.new_empty(out_shape)

    out[..., 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
    out[..., 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
    out[..., 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
    out[..., 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
    return out


def multi_level_haar_dwt(x: Tensor, levels: int) -> Tuple[Tensor, List[HighFreq], Tuple[int, int]]:
    """Apply multi-level Haar DWT to the LL branch."""

    if levels < 0:
        raise ValueError(f"levels must be non-negative, got {levels}")
    if levels == 0:
        return x, [], (int(x.shape[-2]), int(x.shape[-1]))

    current, original_hw = pad_to_multiple(x, 2 ** levels)
    high_freqs: List[HighFreq] = []
    for _ in range(levels):
        ll, lh, hl, hh = haar_dwt2(current)
        high_freqs.append((lh, hl, hh))
        current = ll
    return current, high_freqs, original_hw


def multi_level_haar_idwt(ll: Tensor, high_freqs: List[HighFreq]) -> Tensor:
    """Reconstruct an image from LL plus high-frequency subbands."""

    current = ll
    for lh, hl, hh in reversed(high_freqs):
        current = haar_idwt2(current, lh, hl, hh)
    return current


def robust_normalize(x: Tensor, low_q: float = 0.02, high_q: float = 0.98, eps: float = 1e-6) -> Tensor:
    """Per-sample, per-channel robust normalization to [0, 1]."""

    xb, squeezed = _as_batched(torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0))
    flat = xb.flatten(-2)
    lo = torch.quantile(flat, low_q, dim=-1).unsqueeze(-1).unsqueeze(-1)
    hi = torch.quantile(flat, high_q, dim=-1).unsqueeze(-1).unsqueeze(-1)
    scaled = (xb - lo) / (hi - lo).clamp_min(eps)
    scaled = scaled.clamp(0.0, 1.0)
    return _restore_batch(scaled, squeezed)


def project_multiband_to_rgb(x: Tensor, assume_bgrn: bool = True) -> Tensor:
    """Deterministic C->3 projection for smoke tests and visual ablations."""

    xb, squeezed = _as_batched(x)
    c = xb.shape[1]
    if c >= 4 and assume_bgrn:
        rgb = xb[:, [2, 1, 0], ...]
    elif c >= 3:
        rgb = xb[:, :3, ...]
    elif c == 2:
        rgb = torch.stack((xb[:, 0], xb[:, 1], 0.5 * (xb[:, 0] + xb[:, 1])), dim=1)
    elif c == 1:
        rgb = xb.repeat(1, 3, 1, 1)
    else:
        raise ValueError("input must have at least one channel")
    return _restore_batch(robust_normalize(rgb), squeezed)


def high_frequency_energy_to_rgb(high_freqs: List[HighFreq]) -> Optional[Tensor]:
    """Summarize the latest DWT high-frequency bands as a 3-channel image."""

    if not high_freqs:
        return None
    lh, hl, hh = high_freqs[-1]
    energy = torch.sqrt(lh.square() + hl.square() + hh.square() + 1e-12)
    return project_multiband_to_rgb(energy, assume_bgrn=False)


class MultiBandDirectAdapter(nn.Module):
    """Learnable per-pixel multi-band adapter.

    The module starts as the current deterministic BGR->RGB projection when
    possible, then training can assign weight to extra bands such as NIR.
    """

    def __init__(
        self,
        in_channels: int,
        target_channels: int = 3,
        output_size: Optional[Tuple[int, int]] = None,
        init: str = "rgb_from_bgrn",
        bias: bool = True,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if target_channels <= 0:
            raise ValueError(f"target_channels must be positive, got {target_channels}")
        self.in_channels = int(in_channels)
        self.target_channels = int(target_channels)
        self.output_size = output_size
        self.init = init
        self.proj = nn.Conv2d(self.in_channels, self.target_channels, kernel_size=1, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        if self.init == "rgb_from_bgrn" and self.target_channels == 3:
            if self.in_channels >= 4:
                mapping = (2, 1, 0)
            elif self.in_channels >= 3:
                mapping = (0, 1, 2)
            elif self.in_channels == 2:
                mapping = (0, 1, 1)
            else:
                mapping = (0, 0, 0)
            with torch.no_grad():
                for out_ch, in_ch in enumerate(mapping):
                    self.proj.weight[out_ch, in_ch, 0, 0] = 1.0
        elif self.init == "kaiming":
            nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
        elif self.init != "zeros":
            raise ValueError(f"unsupported adapter init: {self.init}")

    def extra_band_weight_l1(self) -> Tensor:
        if self.in_channels <= self.target_channels:
            return self.proj.weight.new_zeros(())
        return self.proj.weight[:, self.target_channels :, :, :].abs().sum()

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, object]]:
        proj_weight = self.proj.weight
        xb, squeezed = _as_batched(x.to(device=proj_weight.device, dtype=proj_weight.dtype))
        if xb.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {xb.shape[1]}")

        output = self.proj(xb)
        if self.output_size is not None and output.shape[-2:] != tuple(self.output_size):
            output = F.interpolate(output, size=self.output_size, mode="bilinear", align_corners=False)

        output = _restore_batch(output, squeezed)
        meta: Dict[str, object] = {
            "adapter": "learnable_direct",
            "init": self.init,
            "uses_extra_bands": self.in_channels > self.target_channels,
            "input_shape": list(x.shape),
            "output_shape": list(output.shape),
            "extra_band_weight_l1": float(self.extra_band_weight_l1().detach().cpu().item()),
        }
        return output, meta


def apply_multiband_adapter(
    adapter: nn.Module,
    multiband: Tensor,
    reference_rgb: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, object]]:
    """Normalize raw multi-band input, apply adapter, and align to RGB reference."""

    param = next(adapter.parameters(), None)
    x = robust_normalize(multiband)
    if param is not None:
        x = x.to(device=param.device, dtype=param.dtype)

    output, meta = adapter(x)
    if reference_rgb is None:
        return output, meta

    target_hw = tuple(reference_rgb.shape[-2:])
    if output.shape[-2:] != target_hw:
        output_batched, squeezed = _as_batched(output)
        output_dtype = output_batched.dtype
        output_batched = F.interpolate(
            output_batched.float(),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        ).to(dtype=output_dtype)
        output = _restore_batch(output_batched, squeezed)

    output = output.to(device=reference_rgb.device, dtype=reference_rgb.dtype)
    meta = dict(meta)
    meta["reference_shape"] = list(reference_rgb.shape)
    meta["output_shape"] = list(output.shape)
    return output, meta


class WaveletAdapter(nn.Module):
    """Deterministic adapter used to run the first DWT ablation steps.

    Modes:
      - ``direct``: robust multi-band C->3 projection, no DWT.
      - ``dwt_ll``: DWT LL followed by C->3 projection.
      - ``dwt_ll_hf``: LL projection blended with latest-level HF energy.
    """

    def __init__(
        self,
        mode: str = "direct",
        source_gsd: float = 1.0,
        target_gsd: float = 8.0,
        level_policy: str = "nearest",
        output_size: Optional[Tuple[int, int]] = None,
        hf_weight: float = 0.25,
    ):
        super().__init__()
        if mode not in {"direct", "dwt_ll", "dwt_ll_hf"}:
            raise ValueError(f"unsupported adapter mode: {mode}")
        self.mode = mode
        self.source_gsd = float(source_gsd)
        self.target_gsd = float(target_gsd)
        self.level_policy = level_policy
        self.output_size = output_size
        self.hf_weight = float(hf_weight)

    def forward(self, x: Tensor, source_gsd: Optional[float] = None) -> Tuple[Tensor, Dict[str, object]]:
        xb, squeezed = _as_batched(x)
        gsd = float(self.source_gsd if source_gsd is None else source_gsd)

        levels = 0
        effective_gsd = gsd
        if self.mode == "direct":
            output = project_multiband_to_rgb(xb)
        else:
            levels, effective_gsd = compute_dwt_levels(
                gsd,
                self.target_gsd,
                policy=self.level_policy,
                spatial_shape=xb.shape[-2:],
            )
            ll, high_freqs, _ = multi_level_haar_dwt(xb, levels)
            output = project_multiband_to_rgb(ll)
            if self.mode == "dwt_ll_hf":
                hf_rgb = high_frequency_energy_to_rgb(high_freqs)
                if hf_rgb is not None:
                    if hf_rgb.shape[-2:] != output.shape[-2:]:
                        hf_rgb = F.interpolate(hf_rgb, size=output.shape[-2:], mode="bilinear", align_corners=False)
                    output = ((1.0 - self.hf_weight) * output + self.hf_weight * hf_rgb).clamp(0.0, 1.0)

        if self.output_size is not None and output.shape[-2:] != tuple(self.output_size):
            output = F.interpolate(output, size=self.output_size, mode="bilinear", align_corners=False)
            output = output.clamp(0.0, 1.0)

        output = _restore_batch(output, squeezed)
        meta: Dict[str, object] = {
            "mode": self.mode,
            "source_gsd": gsd,
            "target_gsd": self.target_gsd,
            "levels": int(levels),
            "effective_gsd": float(effective_gsd),
            "input_shape": list(x.shape),
            "output_shape": list(output.shape),
        }
        return output, meta
