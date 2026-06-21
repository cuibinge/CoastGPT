"""Selective pseudo-background loss helpers for partial-label segmentation."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def normalized_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return entropy normalized to [0, 1] across the class dimension."""
    num_classes = probs.shape[1]
    entropy = -(probs * (probs + eps).log()).sum(dim=1)
    return entropy / np.log(max(num_classes, 2))


def scheduled_background_lambda(
    base_lambda: float,
    *,
    epoch: int,
    warmup_epochs: int = 0,
    ramp_epochs: int = 0,
) -> float:
    """Return background-prior weight after warmup and linear ramp."""
    if base_lambda <= 0:
        return 0.0
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(base_lambda)
    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / ramp_epochs))
    return float(base_lambda) * progress


def build_foreground_protection_mask(
    target: torch.Tensor,
    *,
    ignore_index: int = 255,
    background_id: int = 0,
    radius_px: int = 8,
) -> torch.Tensor:
    """Dilate labeled foreground pixels to avoid sampling nearby ignore pixels."""
    foreground = (target != ignore_index) & (target != background_id)
    if radius_px <= 0:
        return foreground

    kernel = 2 * radius_px + 1
    foreground_f = foreground.float().unsqueeze(1)
    protected = F.max_pool2d(
        foreground_f,
        kernel_size=kernel,
        stride=1,
        padding=radius_px,
    )
    return protected.squeeze(1).bool()


def build_pseudo_background_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int = 255,
    background_id: int = 0,
    p_bg_threshold: float = 0.7,
    entropy_threshold: float = 0.25,
    protect_radius_px: int = 8,
) -> torch.Tensor:
    """Select ignore pixels that are likely true background.

    A pixel is selected only when it is ignored by GT, far enough from labeled
    foreground, predicted as background with high confidence, and low entropy.
    The selection mask is detached from the graph; gradients flow only through
    the BCE term at selected pixels.
    """
    with torch.no_grad():
        probs = torch.softmax(logits.detach(), dim=1)
        p_bg = probs[:, background_id]
        entropy = normalized_entropy(probs)
        ignore_mask = target == ignore_index
        protected = build_foreground_protection_mask(
            target,
            ignore_index=ignore_index,
            background_id=background_id,
            radius_px=protect_radius_px,
        )
        return (
            ignore_mask
            & (~protected)
            & (p_bg >= p_bg_threshold)
            & (entropy <= entropy_threshold)
        )


def selective_pseudo_background_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int = 255,
    background_id: int = 0,
    p_bg_threshold: float = 0.7,
    entropy_threshold: float = 0.25,
    protect_radius_px: int = 8,
    max_ignore_ratio: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """BCE on high-confidence pseudo-background pixels in ignore regions."""
    pseudo_bg = build_pseudo_background_mask(
        logits,
        target,
        ignore_index=ignore_index,
        background_id=background_id,
        p_bg_threshold=p_bg_threshold,
        entropy_threshold=entropy_threshold,
        protect_radius_px=protect_radius_px,
    )
    log_p_bg = torch.log_softmax(logits, dim=1)[:, background_id]

    batch_size = target.shape[0]
    losses = []
    active_tiles = 0
    total_candidates = 0
    total_sampled = 0

    for b in range(batch_size):
        candidate_mask = pseudo_bg[b]
        n_candidates = int(candidate_mask.sum().item())
        total_candidates += n_candidates
        if n_candidates == 0:
            continue

        labeled_count = int((target[b] != ignore_index).sum().item())
        if labeled_count > 0:
            max_sample = max(1, int(labeled_count * max_ignore_ratio))
        else:
            max_sample = max(1, int(n_candidates * max_ignore_ratio))
        n_sample = min(max_sample, n_candidates)

        candidate_indices = candidate_mask.nonzero(as_tuple=False)
        perm = torch.randperm(n_candidates, device=logits.device)[:n_sample]
        sampled = candidate_indices[perm]
        losses.append(-log_p_bg[b][sampled[:, 0], sampled[:, 1]].mean())
        active_tiles += 1
        total_sampled += n_sample

    stats = {
        "pseudo_bg_active_tile_ratio": active_tiles / batch_size if batch_size > 0 else 0.0,
        "pseudo_bg_candidate_pixel_count": float(total_candidates),
        "pseudo_bg_sampled_pixel_count": float(total_sampled),
    }

    if not losses:
        return logits.sum() * 0.0, stats

    return torch.stack(losses).mean(), stats
