"""GDF loss functions for PoC-3 A3.

Provides:
  - masked_smooth_l1: SmoothL1 loss with validity mask
  - gdf_loss: composite GDF loss (vector + distance + valid + consistency)
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# GDF Loss
# =============================================================================


def gdf_loss(
    field_pred: torch.Tensor,
    gdf_target: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    lambda_vec: float = 1.0,
    lambda_dist: float = 0.5,
    lambda_valid: float = 0.5,
    lambda_cons: float = 0.0,
    valid_pos_weight: Optional[float] = None,
) -> dict[str, torch.Tensor]:
    """Composite GDF loss.

    Args:
        field_pred: [B, 4, H, W] raw field prediction.
            channel 0: dx_raw
            channel 1: dy_raw
            channel 2: log_dist_raw
            channel 3: valid_logit
        gdf_target: [B, 4, H, W] GDF target.
            channel 0: dx_norm  ∈ [-1, 1]
            channel 1: dy_norm  ∈ [-1, 1]
            channel 2: log_dist_norm ∈ [0, 1]
            channel 3: valid ∈ {0, 1}
        loss_mask: [B, 1, H, W] optional mask (1=supervise, 0=ignore).
                   If None, uses gdf_target[:, 3] as mask.
        lambda_vec: Weight for vector (dx, dy) loss.
        lambda_dist: Weight for distance loss.
        lambda_valid: Weight for valid classification loss.
        lambda_cons: Weight for consistency loss (0 = disabled).
        valid_pos_weight: Optional pos_weight for BCEWithLogits on valid.

    Returns:
        Dict with 'total', 'loss_vec', 'loss_dist', 'loss_valid',
        'loss_consistency', and debug values.
    """
    B, C, H, W = field_pred.shape
    assert C == 4, f"Expected 4-channel field, got {C}"

    # Parse channels
    pred_dx = field_pred[:, 0:1]       # [B, 1, H, W]
    pred_dy = field_pred[:, 1:2]
    pred_log_dist = field_pred[:, 2:3]
    pred_valid_logit = field_pred[:, 3:4]

    gt_dx = gdf_target[:, 0:1]
    gt_dy = gdf_target[:, 1:2]
    gt_log_dist = gdf_target[:, 2:3]
    gt_valid = gdf_target[:, 3:4]

    # Loss mask
    if loss_mask is None:
        loss_mask = gt_valid.clone()
    mask = loss_mask.float()

    # Auto-compute pos_weight for valid loss
    if valid_pos_weight is None and mask.sum() > 0:
        n_pos = gt_valid.sum()
        n_neg = gt_valid.numel() - n_pos
        valid_pos_weight = float(n_neg / max(n_pos, 1))

    # ---- Vector loss (SmoothL1, only in valid region) ----
    vec_diff = torch.cat([pred_dx - gt_dx, pred_dy - gt_dy], dim=1)
    vec_mask = mask.expand(-1, 2, -1, -1)
    loss_vec = _masked_smooth_l1(vec_diff, vec_mask, normalize=True)

    # ---- Distance loss (SmoothL1, only in valid region) ----
    # Apply sigmoid to get log_dist in [0,1]
    pred_log_dist_prob = torch.sigmoid(pred_log_dist)
    dist_diff = pred_log_dist_prob - gt_log_dist
    loss_dist = _masked_smooth_l1(dist_diff, mask, normalize=True)

    # ---- Valid loss (BCEWithLogits, all pixels) ----
    loss_valid = F.binary_cross_entropy_with_logits(
        pred_valid_logit.squeeze(1),
        gt_valid.squeeze(1),
        pos_weight=torch.tensor(valid_pos_weight, device=field_pred.device)
        if valid_pos_weight is not None else None,
    )

    # ---- Consistency loss (optional) ----
    loss_cons = torch.tensor(0.0, device=field_pred.device)
    if lambda_cons > 0:
        # ||pred_vec|| should approximate pred_dist
        pred_vec_norm = torch.sqrt(
            torch.tanh(pred_dx) ** 2 + torch.tanh(pred_dy) ** 2 + 1e-8
        )  # in [-√2, √2] range after tanh
        # Scale to [0, 1] range comparable to log_dist
        pred_vec_norm_scaled = pred_vec_norm / 1.414
        loss_cons = F.smooth_l1_loss(
            pred_vec_norm_scaled * mask,
            pred_log_dist_prob * mask,
            reduction='sum',
        ) / max(mask.sum(), 1)

    # ---- Total ----
    total = (
        lambda_vec * loss_vec
        + lambda_dist * loss_dist
        + lambda_valid * loss_valid
        + lambda_cons * loss_cons
    )

    return {
        "total": total,
        "loss_vec": loss_vec,
        "loss_dist": loss_dist,
        "loss_valid": loss_valid,
        "loss_consistency": loss_cons,
        "valid_pos_weight": torch.tensor(
            valid_pos_weight if valid_pos_weight is not None else 1.0,
            device=field_pred.device,
        ),
    }


def _masked_smooth_l1(
    diff: torch.Tensor,
    mask: torch.Tensor,
    normalize: bool = True,
    beta: float = 1.0,
) -> torch.Tensor:
    """SmoothL1 loss computed only over masked region.

    Args:
        diff: [B, C, H, W] prediction - target.
        mask: [B, 1 or C, H, W] binary mask.
        normalize: If True, divide by mask sum (mean over valid).
                   If False, divide by total pixels (per-pixel loss).
        beta: SmoothL1 beta parameter.

    Returns:
        Scalar loss.
    """
    if mask.shape[1] != diff.shape[1]:
        mask = mask.expand(-1, diff.shape[1], -1, -1)

    loss = F.smooth_l1_loss(
        diff * mask,
        torch.zeros_like(diff),
        reduction='sum',
        beta=beta,
    )

    if normalize:
        denom = max(mask.sum(), 1)
    else:
        denom = diff.shape[0] * diff.shape[2] * diff.shape[3]

    return loss / denom


# =============================================================================
# Unit tests
# =============================================================================

if __name__ == "__main__":
    print("=== GDF Loss Unit Tests ===")

    B, C, H, W = 2, 4, 224, 224

    # Create synthetic GDF target
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    # Simple line target
    edge = np.zeros((H, W), dtype=np.float32)
    edge[100, 50:170] = 1.0
    non_edge = 1 - edge.astype(np.uint8)
    dist, indices = distance_transform_edt(non_edge, return_indices=True)
    nr, nc = indices[0].astype(np.float32), indices[1].astype(np.float32)
    rr, cc = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    R = 32
    dx = np.clip((nc - cc) / R, -1, 1)
    dy = np.clip((nr - rr) / R, -1, 1)
    ld = np.log1p(np.minimum(dist, R)) / np.log1p(R)
    valid = (dist <= R).astype(np.float32)
    dx[edge > 0] = 0; dy[edge > 0] = 0; ld[edge > 0] = 0

    target_np = np.stack([dx, dy, ld, valid], axis=0).astype(np.float32)
    target = torch.from_numpy(target_np).unsqueeze(0).repeat(B, 1, 1, 1)

    # Test 1: Perfect prediction → near-zero loss
    print("\n[1] Perfect prediction...")
    perfect_field = target.clone()
    # Raw field: apply inverse of activation
    # dx/dy: apply atanh (tanh maps [-1,1] so raw is unbounded)
    # log_dist: apply logit (sigmoid^-1)
    # valid: keep as logit for BCEWithLogits
    eps = 1e-6
    dx_clamped = torch.clamp(perfect_field[:, 0:1], -1 + eps, 1 - eps)
    dy_clamped = torch.clamp(perfect_field[:, 1:2], -1 + eps, 1 - eps)
    ld_clamped = torch.clamp(perfect_field[:, 2:3], eps, 1 - eps)
    perfect_field[:, 0:1] = torch.atanh(dx_clamped)
    perfect_field[:, 1:2] = torch.atanh(dy_clamped)
    perfect_field[:, 2:3] = torch.logit(ld_clamped, eps=eps)
    # valid: set logit to +10 for valid=1, -10 for valid=0
    perfect_field[:, 3:4] = torch.where(
        perfect_field[:, 3:4] > 0.5,
        torch.full_like(perfect_field[:, 3:4], 10.0),
        torch.full_like(perfect_field[:, 3:4], -10.0),
    )

    result = gdf_loss(perfect_field, target)
    print(f"  total: {result['total'].item():.4f} (expect near 0)")
    assert result["total"].item() < 0.5, f"Perfect pred should have low loss: {result['total'].item():.4f}"
    print("  ✓")

    # Test 2: Random prediction → higher loss
    print("\n[2] Random prediction...")
    random_field = torch.randn(B, 4, H, W)
    result_r = gdf_loss(random_field, target)
    print(f"  total: {result_r['total'].item():.4f} (expect > 1)")
    assert result_r["total"].item() > result["total"].item(), "Random should be higher than perfect"
    print("  ✓")

    # Test 3: Empty tile (all valid=0)
    print("\n[3] Empty tile...")
    empty_target = torch.zeros(B, 4, H, W)
    empty_target[:, 2] = 1.0  # log_dist = 1
    empty_mask = torch.zeros(B, 1, H, W)
    result_e = gdf_loss(random_field, empty_target, loss_mask=empty_mask)
    print(f"  total: {result_e['total'].item():.4f}")
    print(f"  loss_vec: {result_e['loss_vec'].item():.4f} (expect 0)")
    assert result_e["loss_vec"].item() < 0.01, "Empty mask should zero vector loss"
    print("  ✓")

    # Test 4: Consistency loss
    print("\n[4] Consistency loss...")
    result_c = gdf_loss(random_field, target, lambda_cons=0.1)
    print(f"  loss_consistency: {result_c['loss_consistency'].item():.4f}")
    print("  ✓")

    print("\n=== ALL GDF LOSS TESTS PASSED ===")
