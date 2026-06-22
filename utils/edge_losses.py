"""Edge detection loss functions for PoC-3 coastline edge head.

Provides:
  - SoftDiceLoss: binary soft Dice loss
  - FocalLoss: binary Focal Loss with alpha/gamma
  - edge_bce_dice_loss: BCE + Dice composite (A0)
  - edge_focal_dice_loss: Focal + Dice composite (A1)
  - DeepSupervisedEdgeLoss: multi-scale side-output wrapper (A2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _squeeze_channel(x: torch.Tensor) -> torch.Tensor:
    """Squeeze channel dim if input is 4D [B, 1, H, W]."""
    return x.squeeze(1) if x.dim() == 4 else x


def _binary_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    """Inlined binary Focal Loss (no nn.Module allocation)."""
    logits = _squeeze_channel(logits)
    target = _squeeze_channel(target)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pt = torch.exp(-bce)
    focal_weight = (1.0 - pt) ** gamma
    alpha_t = target * alpha + (1.0 - target) * (1.0 - alpha)
    return (alpha_t * focal_weight * bce).mean()


def _soft_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Inlined soft Dice loss (no nn.Module allocation)."""
    pred = _squeeze_channel(pred)
    target = _squeeze_channel(target)
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return (1.0 - dice).mean()


class SoftDiceLoss(nn.Module):
    """Binary soft Dice loss.

    Args:
        eps: Smoothing term to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute soft Dice loss.

        Args:
            pred: [B, 1, H, W] or [B, H, W] after sigmoid.
            target: [B, 1, H, W] or [B, H, W] in [0, 1].
        """
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)

        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return (1.0 - dice).mean()


class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss with BCEWithLogits.

    Args:
        alpha: Foreground class weight (background gets 1-alpha).
        gamma: Focusing parameter. Higher = more focus on hard examples.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute binary Focal Loss.

        Args:
            logits: [B, 1, H, W] raw logits.
            target: [B, 1, H, W] in [0, 1].
        """
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = (1.0 - pt) ** self.gamma

        alpha_t = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


def edge_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A0: BCEWithLogits + Dice composite for edge detection.

    Args:
        logits: [B, 1, H, W] raw logits.
        target: [B, 1, H, W] edge target in [0, 1].

    Returns:
        total_loss, loss_bce, loss_dice
    """
    logits_flat = _squeeze_channel(logits)
    target_flat = _squeeze_channel(target)

    loss_bce = F.binary_cross_entropy_with_logits(logits_flat, target_flat)
    probs = torch.sigmoid(logits_flat)
    loss_dice = _soft_dice_loss(probs, target_flat)
    total = loss_bce + loss_dice
    return total, loss_bce, loss_dice


def edge_focal_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A1: Focal + Dice composite for edge detection.

    Returns:
        total_loss, loss_focal, loss_dice
    """
    loss_focal = _binary_focal_loss(logits, target, alpha, gamma)
    probs = torch.sigmoid(logits)
    loss_dice = _soft_dice_loss(probs, target)
    total = loss_focal + loss_dice
    return total, loss_focal, loss_dice


class DeepSupervisedEdgeLoss(nn.Module):
    """A2: Deep-supervised loss for multi-scale edge head.

    Computes Focal+Dice on each side output (upsampled to 224) and on the
    fused output, then returns a weighted sum.

    Args:
        alpha: Focal Loss alpha.
        gamma: Focal Loss gamma.
        fused_weight: Weight for fused output loss.
        side_weights: List of 4 weights for [side1, side2, side3, side4].
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        fused_weight: float = 1.0,
        side_weights: tuple[float, float, float, float] = (0.5, 0.3, 0.2, 0.1),
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.fused_weight = fused_weight
        self.side_weights = side_weights

    def forward(
        self,
        fused_logits: torch.Tensor,
        side_logits_list: list[torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute deep-supervised edge loss.

        Args:
            fused_logits: [B, 1, 224, 224] fused output.
            side_logits_list: List of [B, 1, 224, 224] side outputs,
                              ordered [side1, side2, side3, side4].
            target: [B, 1, 224, 224] GT edge target.

        Returns:
            Dict with total, loss_fused, loss_side1..4.
        """
        total = fused_logits.new_tensor(0.0)

        loss_fused, _, _ = edge_focal_dice_loss(
            fused_logits, target, self.alpha, self.gamma
        )
        total = total + self.fused_weight * loss_fused

        result = {"loss_fused": loss_fused}

        for i, (side_logits, w) in enumerate(
            zip(side_logits_list, self.side_weights), start=1
        ):
            loss_side, _, _ = edge_focal_dice_loss(
                side_logits, target, self.alpha, self.gamma
            )
            total = total + w * loss_side
            result[f"loss_side{i}"] = loss_side

        result["total"] = total
        return result


if __name__ == "__main__":
    B, C, H, W = 2, 1, 224, 224
    logits = torch.randn(B, C, H, W)
    target = torch.zeros(B, C, H, W)
    target[:, :, 100:120, 50:170] = 1.0

    total_bce, bce, dice = edge_bce_dice_loss(logits, target)
    print(f"BCE+Dice: total={total_bce.item():.4f}, bce={bce.item():.4f}, dice={dice.item():.4f}")

    total_focal, focal, dice2 = edge_focal_dice_loss(logits, target)
    print(f"Focal+Dice: total={total_focal.item():.4f}, focal={focal.item():.4f}, dice={dice2.item():.4f}")

    # Deep supervision smoke test
    ds_loss = DeepSupervisedEdgeLoss()
    fused = torch.randn(B, 1, 224, 224)
    sides = [torch.randn(B, 1, 224, 224) for _ in range(4)]
    result = ds_loss(fused, sides, target)
    print(f"DeepSup: total={result['total'].item():.4f}, fused={result['loss_fused'].item():.4f}")
    print("All loss checks passed.")
