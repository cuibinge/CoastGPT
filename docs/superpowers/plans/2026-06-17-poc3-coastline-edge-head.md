# PoC-3 Coastline Edge Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement coastline edge detection head (A0 minimum closure) on frozen DualVisionEncoder + ViT-FPN, producing GeoJSON LineString/MultiLineString output from edge heatmaps.

**Architecture:** Follows PoC-2b pattern — independent utility modules (losses, postprocess, metrics), a Dataset that generates edge GT from GeoJSON/Binary TIF, a SingleScaleEdgeHead model, and a single-NPU training script. ViT-FPN is reused. Vision encoder is frozen. No DeepSpeed, no LLM, no EpochBasedTrainer.

**Tech Stack:** PyTorch 2.1.2, torchvision, numpy, PIL, shapely, scikit-image (skeletonize), affine, pyproj

**Design spec:** `docs/superpowers/specs/2026-06-17-poc3-coastline-edge-head-design.md`

**File map:**

| File | Action | Purpose |
|---|---|---|
| `utils/edge_losses.py` | Create | FocalLoss, SoftDiceLoss, BCE+Dice, deep-supervised composite |
| `Models/edge_head.py` | Create | SingleScaleEdgeHead, MultiScaleEdgeHead |
| `utils/edge_postprocess.py` | Create | threshold, skeletonize, graph extraction, simplify, GeoJSON export |
| `utils/coastline_metrics.py` | Create | buffered-F1, Chamfer, Hausdorff, length ratio |
| `Dataset/coastline_dataset.py` | Create | CoastlineEdgeDataset, manifest builder |
| `configs/poc3_edge_a0_closure.yaml` | Create | A0 BCE+Dice single-scale config |
| `scripts/poc_stage_edge.py` | Create | train/eval entrypoint (single NPU) |

**Dependency order:** Tasks 1-4 are independent. Task 5 depends on Task 3 (uses postprocess in dry-run). Task 6 is independent. Task 7 depends on all.

---

### Task 1: Create `utils/edge_losses.py`

**Files:**
- Create: `utils/edge_losses.py`

- [ ] **Step 1: Write the losses module**

```python
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
    if logits.dim() == 4:
        logits_flat = logits.squeeze(1)
    else:
        logits_flat = logits
    if target.dim() == 4:
        target_flat = target.squeeze(1)
    else:
        target_flat = target

    loss_bce = F.binary_cross_entropy_with_logits(logits_flat, target_flat)
    probs = torch.sigmoid(logits)
    loss_dice = SoftDiceLoss()(probs, target)
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
    loss_focal = BinaryFocalLoss(alpha=alpha, gamma=gamma)(logits, target)
    probs = torch.sigmoid(logits)
    loss_dice = SoftDiceLoss()(probs, target)
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
            Dict with total, loss_fused, loss_side1..4, loss_focal, loss_dice.
        """
        total = torch.tensor(0.0, device=fused_logits.device)

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
```

- [ ] **Step 2: Run smoke test**

```bash
cd /home/ma-user/work/CoastGPT && python utils/edge_losses.py
```

Expected: prints loss values with `total=`, all finite.

- [ ] **Step 3: Commit**

```bash
git add utils/edge_losses.py && git commit -m "feat: add edge loss functions (BCE+Dice, Focal+Dice, DeepSup) for PoC-3"
```

---

### Task 2: Create `Models/edge_head.py`

**Files:**
- Create: `Models/edge_head.py`

- [ ] **Step 1: Write the edge head module**

```python
"""Coastline edge detection heads for PoC-3.

Provides:
  - SingleScaleEdgeHead: concatenates FPN P1-P4 at 56x56, outputs 1-channel logit (A0/A1).
  - MultiScaleEdgeHead: HED-style side outputs from each FPN level (A2).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_ch: int, out_ch: int, kernel_size: int = 3) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SingleScaleEdgeHead(nn.Module):
    """Single-scale edge head: FPN P1-P4 → concat at 56×56 → Conv → 1×1 → logit.

    Architecture:
        P1 [B,256,56,56]
        P2 [B,256,28,28] → upsample 56
        P3 [B,256,14,14] → upsample 56
        P4 [B,256,7,7]   → upsample 56
        concat → [B,1024,56,56]
        → 3×3 Conv 1024→256 + BN + ReLU
        → 3×3 Conv 256→128 + BN + ReLU
        → 1×1 Conv 128→1
        → bilinear upsample to output_size (default 224×224)

    Args:
        in_channels: Channels per FPN level (default 256).
        decoder_channels: [mid_channels, pre_output_channels].
        output_size: (H, W) to upsample logits to (default 224×224).
    """

    def __init__(
        self,
        in_channels: int = 256,
        decoder_channels: tuple[int, int] = (256, 128),
        output_size: tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.output_size = output_size

        fused_channels = in_channels * 4  # 1024

        self.conv1 = _conv_bn_relu(fused_channels, decoder_channels[0])
        self.conv2 = _conv_bn_relu(decoder_channels[0], decoder_channels[1])
        self.cls_conv = nn.Conv2d(decoder_channels[1], 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            p1: [B, 256, 56, 56]
            p2: [B, 256, 28, 28]
            p3: [B, 256, 14, 14]
            p4: [B, 256, 7, 7]

        Returns:
            edge_logit: [B, 1, output_H, output_W]
        """
        h, w = p1.shape[2], p1.shape[3]

        p2_up = F.interpolate(p2, size=(h, w), mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=False)

        fused = torch.cat([p1, p2_up, p3_up, p4_up], dim=1)  # [B, 1024, 56, 56]

        x = self.conv1(fused)   # [B, 256, 56, 56]
        x = self.conv2(x)       # [B, 128, 56, 56]
        x = self.cls_conv(x)    # [B, 1, 56, 56]

        logits = F.interpolate(
            x, size=self.output_size, mode='bilinear', align_corners=False
        )
        return logits


class MultiScaleEdgeHead(nn.Module):
    """HED-style multi-scale edge head with side outputs (A2).

    Each FPN level produces a side-output edge logit, which is upsampled
    to output_size. Side outputs are concatenated and fused via 1×1 Conv.

    Architecture:
        P1 → side_conv1 → side1_logit → upsample 224
        P2 → side_conv2 → side2_logit → upsample 224
        P3 → side_conv3 → side3_logit → upsample 224
        P4 → side_conv4 → side4_logit → upsample 224
        concat → fuse_conv(1×1) → fused_logit

    Args:
        in_channels: Channels per FPN level (default 256).
        output_size: (H, W) for all outputs.
    """

    def __init__(
        self,
        in_channels: int = 256,
        output_size: tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.output_size = output_size

        # Side output convolutions (one per FPN level): 3×3 Conv 256→128 + 1×1 Conv 128→1
        self.side_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, kernel_size=1),
            )
            for _ in range(4)
        ])

        # Fuse convolution: 4 side outputs → 1 fused logit
        self.fuse_conv = nn.Conv2d(4, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            p1-p4: FPN feature maps at [56, 28, 14, 7] spatial sizes.

        Returns:
            Dict with:
                'fused': [B, 1, H_out, W_out] fused edge logit.
                'side1'..'side4': [B, 1, H_out, W_out] per-level side logits.
        """
        features = [p1, p2, p3, p4]
        sides = []
        for feat, side_conv in zip(features, self.side_convs):
            side_logit = side_conv(feat)  # [B, 1, Hi, Wi]
            side_up = F.interpolate(
                side_logit, size=self.output_size, mode='bilinear', align_corners=False
            )
            sides.append(side_up)

        fused = self.fuse_conv(torch.cat(sides, dim=1))

        return {
            "fused": fused,
            "side1": sides[0],
            "side2": sides[1],
            "side3": sides[2],
            "side4": sides[3],
        }


if __name__ == "__main__":
    B = 2
    p1 = torch.randn(B, 256, 56, 56)
    p2 = torch.randn(B, 256, 28, 28)
    p3 = torch.randn(B, 256, 14, 14)
    p4 = torch.randn(B, 256, 7, 7)

    # Single-scale
    single = SingleScaleEdgeHead(output_size=(224, 224))
    logits = single(p1, p2, p3, p4)
    print(f"SingleScaleEdgeHead: {logits.shape}")
    assert logits.shape == (B, 1, 224, 224), f"Expected [2,1,224,224], got {logits.shape}"
    n_params = sum(p.numel() for p in single.parameters())
    print(f"  Params: {n_params:,}")

    # Multi-scale
    multi = MultiScaleEdgeHead(output_size=(224, 224))
    outputs = multi(p1, p2, p3, p4)
    print(f"MultiScaleEdgeHead:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
        assert v.shape == (B, 1, 224, 224), f"Expected [2,1,224,224], got {v.shape}"
    n_params_m = sum(p.numel() for p in multi.parameters())
    print(f"  Params: {n_params_m:,}")

    print("All shape checks passed.")
```

- [ ] **Step 2: Run shape check**

```bash
cd /home/ma-user/work/CoastGPT && python Models/edge_head.py
```

Expected: prints shapes, all assertions pass.

- [ ] **Step 3: Commit**

```bash
git add Models/edge_head.py && git commit -m "feat: add SingleScaleEdgeHead and MultiScaleEdgeHead for PoC-3"
```

---

### Task 3: Create `utils/edge_postprocess.py`

**Files:**
- Create: `utils/edge_postprocess.py`

- [ ] **Step 1: Write the postprocess module**

```python
"""Edge heatmap postprocessing for PoC-3 coastline detection.

Converts edge logits → binary mask → skeleton → polylines → GeoJSON.
"""

import math
from typing import List, Optional, Tuple

import numpy as np

try:
    from skimage.morphology import skeletonize, remove_small_objects
    from skimage.measure import label as connected_label
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import linemerge
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


def heatmap_to_binary(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 8,
) -> np.ndarray:
    """Convert sigmoid heatmap to binary mask.

    Args:
        heatmap: [H, W] float32 in [0, 1].
        threshold: Binarization threshold.
        min_area: Minimum connected component area in pixels.

    Returns:
        binary: [H, W] uint8 binary mask.
    """
    binary = (heatmap >= threshold).astype(np.uint8)
    if min_area > 0 and HAS_SKIMAGE:
        binary = remove_small_objects(binary.astype(bool), min_size=min_area)
        binary = binary.astype(np.uint8)
    return binary


def binary_to_skeleton(binary: np.ndarray) -> np.ndarray:
    """Skeletonize binary edge mask.

    Args:
        binary: [H, W] uint8 binary mask.

    Returns:
        skeleton: [H, W] uint8 skeleton (1px wide).
    """
    if HAS_SKIMAGE:
        return skeletonize(binary.astype(bool)).astype(np.uint8)
    else:
        raise ImportError("scikit-image required: pip install scikit-image")


def skeleton_to_paths(
    skeleton: np.ndarray,
    min_length: int = 10,
    max_components: int = 5,
) -> List[List[Tuple[float, float]]]:
    """Extract polylines from skeleton image via connected components.

    Each connected component is converted to a path by tracing the skeleton.
    Short components (< min_length) are dropped.

    Args:
        skeleton: [H, W] uint8 skeleton.
        min_length: Minimum path length in pixels.
        max_components: Maximum number of paths to return (top-k by length).

    Returns:
        List of paths, each path is a list of (row, col) pixel coordinates.
    """
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image required")

    labels = connected_label(skeleton, connectivity=2, background=0)
    paths = []

    for label_id in range(1, labels.max() + 1):
        mask = (labels == label_id)
        coords = np.argwhere(mask)
        if len(coords) < min_length:
            continue
        # Sort by one axis for a rough ordered path; then refine
        # Use a simple tracing approach: find endpoints and trace
        path = _trace_component(mask)
        if len(path) >= 2:
            paths.append(path)

    # Sort by length descending, take top-k
    paths.sort(key=len, reverse=True)
    paths = paths[:max_components]

    # Filter again after tracing
    paths = [p for p in paths if len(p) >= min_length]
    return paths


def _trace_component(mask: np.ndarray) -> List[Tuple[float, float]]:
    """Trace a single skeleton component into an ordered path.

    Finds an endpoint (pixel with 1 neighbor), then follows the path.

    Args:
        mask: [H, W] boolean mask of the component.

    Returns:
        Ordered list of (row, col) coordinates.
    """
    coords = np.argwhere(mask)
    if len(coords) < 2:
        return [(float(r), float(c)) for r, c in coords]

    # Build adjacency: for each pixel, find neighbors in the component
    h, w = mask.shape
    neighbors = {}
    for r, c in coords:
        key = (int(r), int(c))
        nbrs = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    nbrs.append((nr, nc))
        neighbors[key] = nbrs

    if not neighbors:
        return []

    # Find endpoint (pixel with exactly 1 neighbor)
    endpoints = [k for k, v in neighbors.items() if len(v) == 1]
    if not endpoints:
        # No clear endpoint — pick the one with fewest neighbors
        endpoints = [min(neighbors.keys(), key=lambda k: len(neighbors[k]))]

    start = endpoints[0]
    path = [start]
    visited = {start}

    current = start
    while True:
        nbrs = [n for n in neighbors.get(current, []) if n not in visited]
        if not nbrs:
            break
        # Prefer straight continuation
        if len(path) >= 2:
            prev = path[-2]
            dr_prev = current[0] - prev[0]
            dc_prev = current[1] - prev[1]
            # Sort by direction continuity
            nbrs.sort(
                key=lambda n: abs((n[0] - current[0]) - dr_prev)
                + abs((n[1] - current[1]) - dc_prev)
            )
        next_px = nbrs[0]
        path.append(next_px)
        visited.add(next_px)
        current = next_px

    return [(float(r), float(c)) for r, c in path]


def simplify_path(
    path: List[Tuple[float, float]],
    epsilon: float = 1.0,
) -> List[Tuple[float, float]]:
    """Douglas-Peucker simplification.

    Args:
        path: List of (row, col) pixel coordinates.
        epsilon: Maximum distance in pixels.

    Returns:
        Simplified path.
    """
    if len(path) <= 2:
        return path

    # Find point with maximum distance
    dmax = 0.0
    index = 0
    end = len(path) - 1

    for i in range(1, end):
        d = _perpendicular_distance(path[i], path[0], path[end])
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        left = simplify_path(path[:index + 1], epsilon)
        right = simplify_path(path[index:], epsilon)
        return left[:-1] + right
    else:
        return [path[0], path[-1]]


def _perpendicular_distance(
    pt: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> float:
    """Distance from pt to line segment (line_start, line_end)."""
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    if dx == 0 and dy == 0:
        return math.sqrt((pt[0] - line_start[0]) ** 2 + (pt[1] - line_start[1]) ** 2)

    t = ((pt[0] - line_start[0]) * dx + (pt[1] - line_start[1]) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))

    proj = (line_start[0] + t * dx, line_start[1] + t * dy)
    return math.sqrt((pt[0] - proj[0]) ** 2 + (pt[1] - proj[1]) ** 2)


def paths_to_geojson(
    paths: List[List[Tuple[float, float]]],
    georef: dict,
    sample_id: str = "",
    class_name: str = "海岸线",
) -> dict:
    """Convert pixel paths to GeoJSON FeatureCollection.

    Args:
        paths: List of paths, each a list of (row, col) pixel coords.
        georef: Dict with 'model_transform', 'source_crs'.
        sample_id: Sample identifier.
        class_name: Feature class name.

    Returns:
        GeoJSON FeatureCollection dict.
    """
    from utils.georef_transform import pixel_to_wgs84

    if not paths:
        return {
            "type": "FeatureCollection",
            "features": [],
        }

    features = []
    for path in paths:
        # Convert row,col → col,row (georef uses col,row)
        pixel_coords = [(col, row) for row, col in path]

        # pixel → WGS84
        wgs84_coords = pixel_to_wgs84(pixel_coords, georef)

        # Convert to [lon, lat] GeoJSON format
        coords = [[lon, lat] for lon, lat in wgs84_coords]

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
            "properties": {
                "class": class_name,
                "sample_id": sample_id,
                "length_px": len(path),
            },
        })

    if len(features) == 1:
        fc_type = "LineString"
    else:
        fc_type = "MultiLineString"

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def postprocess_edge(
    heatmap: np.ndarray,
    georef: dict,
    threshold: float = 0.5,
    min_area: int = 8,
    min_length: int = 10,
    max_components: int = 5,
    simplify_epsilon: float = 1.0,
    sample_id: str = "",
) -> dict:
    """Full edge postprocessing pipeline: heatmap → GeoJSON FeatureCollection.

    Args:
        heatmap: [H, W] sigmoid probabilities.
        georef: Georeference dict for pixel→WGS84.
        threshold: Binarization threshold.
        min_area: Minimum component area.
        min_length: Minimum LineString length in pixels.
        max_components: Maximum number of paths.
        simplify_epsilon: Douglas-Peucker epsilon in pixels.
        sample_id: Sample identifier.

    Returns:
        GeoJSON FeatureCollection dict.
    """
    binary = heatmap_to_binary(heatmap, threshold=threshold, min_area=min_area)
    skeleton = binary_to_skeleton(binary)
    paths = skeleton_to_paths(
        skeleton, min_length=min_length, max_components=max_components
    )
    # Simplify each path
    paths = [simplify_path(p, epsilon=simplify_epsilon) for p in paths]
    # Filter again after simplification
    paths = [p for p in paths if len(p) >= 2]

    return paths_to_geojson(paths, georef, sample_id=sample_id)


if __name__ == "__main__":
    print("Testing edge_postprocess...")
    if not HAS_SKIMAGE:
        print("WARNING: scikit-image not installed, skeletonize disabled")
    if not HAS_SHAPELY:
        print("WARNING: shapely not installed")

    # Create a synthetic heatmap with a diagonal line
    heatmap = np.zeros((224, 224), dtype=np.float32)
    for i in range(50, 170):
        heatmap[i, i] = 0.9
        heatmap[i, i + 1] = 0.3

    if HAS_SKIMAGE:
        binary = heatmap_to_binary(heatmap, threshold=0.5, min_area=4)
        print(f"Binary: fg_px={binary.sum()}, max={binary.max()}")

        skeleton = binary_to_skeleton(binary)
        print(f"Skeleton: fg_px={skeleton.sum()}")

        paths = skeleton_to_paths(skeleton, min_length=5, max_components=3)
        print(f"Paths: {len(paths)} components")
        for i, p in enumerate(paths):
            print(f"  Path {i}: {len(p)} points, start={p[0]}, end={p[-1]}")

        simplified = [simplify_path(p, epsilon=1.0) for p in paths]
        for i, p in enumerate(simplified):
            print(f"  Simplified {i}: {len(p)} points")

    print("Done.")
```

- [ ] **Step 2: Run smoke test**

```bash
cd /home/ma-user/work/CoastGPT && python utils/edge_postprocess.py
```

Expected: prints binary/skeleton/path info, no errors.

- [ ] **Step 3: Commit**

```bash
git add utils/edge_postprocess.py && git commit -m "feat: add edge postprocess (skeletonize, path extract, simplify, GeoJSON) for PoC-3"
```

---

### Task 4: Create `utils/coastline_metrics.py`

**Files:**
- Create: `utils/coastline_metrics.py`

- [ ] **Step 1: Write the metrics module**

```python
"""Coastline edge detection metrics for PoC-3.

Provides:
  - pixel_edge_metrics: pixel-level precision/recall/F1 on edge maps.
  - buffered_f1: geometry-buffered F1 score.
  - chamfer_distance: mean nearest-neighbor distance.
  - hausdorff_distance: max nearest-neighbor distance.
"""

import numpy as np
from typing import Dict, List, Tuple


def pixel_edge_metrics(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
) -> Dict[str, float]:
    """Pixel-level edge detection metrics.

    Args:
        pred_binary: [H, W] uint8 binary edge prediction.
        gt_binary: [H, W] uint8 binary edge GT.

    Returns:
        Dict with precision, recall, f1, foreground_ratio_pred, foreground_ratio_gt.
    """
    pred = pred_binary.astype(bool)
    gt = gt_binary.astype(bool)

    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "pixel_precision": float(precision),
        "pixel_recall": float(recall),
        "pixel_f1": float(f1),
        "pred_fg_ratio": float(pred.mean()),
        "gt_fg_ratio": float(gt.mean()),
    }


def _coords_from_binary(binary: np.ndarray) -> np.ndarray:
    """Extract (row, col) coordinates of foreground pixels."""
    return np.argwhere(binary.astype(bool)).astype(np.float32)


def chamfer_distance(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
) -> float:
    """Chamfer distance: mean of nearest-neighbor distances.

    Args:
        pred_coords: [N, 2] predicted (row, col) coordinates.
        gt_coords: [M, 2] GT (row, col) coordinates.

    Returns:
        Mean Chamfer distance in pixels.
    """
    if len(pred_coords) == 0 and len(gt_coords) == 0:
        return 0.0
    if len(pred_coords) == 0:
        return float(np.inf)
    if len(gt_coords) == 0:
        return float(np.inf)

    # pred → GT
    diff_p2g = pred_coords[:, None, :] - gt_coords[None, :, :]  # [N, M, 2]
    dist_p2g = np.sqrt((diff_p2g ** 2).sum(axis=2)).min(axis=1)  # [N]

    # GT → pred
    diff_g2p = gt_coords[:, None, :] - pred_coords[None, :, :]  # [M, N, 2]
    dist_g2p = np.sqrt((diff_g2p ** 2).sum(axis=2)).min(axis=1)  # [M]

    return float((dist_p2g.mean() + dist_g2p.mean()) / 2.0)


def hausdorff_distance(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """Hausdorff distance (default: 95th percentile, i.e. robust HD).

    Args:
        pred_coords: [N, 2] predicted coordinates.
        gt_coords: [M, 2] GT coordinates.
        percentile: Percentile for robust HD. 100 = standard HD.

    Returns:
        Hausdorff distance in pixels.
    """
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return float(np.inf)

    diff = pred_coords[:, None, :] - gt_coords[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))  # [N, M]
    p2g = dists.min(axis=1)  # [N]
    g2p = dists.min(axis=0)  # [M]
    all_dists = np.concatenate([p2g, g2p])

    if percentile >= 100:
        return float(all_dists.max())
    return float(np.percentile(all_dists, percentile))


def buffered_f1(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
    buffer_px: int = 1,
) -> Dict[str, float]:
    """Compute buffered-F1 by dilating GT and prediction.

    Args:
        pred_binary: [H, W] uint8 edge prediction.
        gt_binary: [H, W] uint8 edge GT.
        buffer_px: Buffer radius in pixels.

    Returns:
        Dict with precision, recall, f1.
    """
    from scipy.ndimage import binary_dilation

    kernel = _disk_kernel(buffer_px)

    pred = pred_binary.astype(bool)
    gt = gt_binary.astype(bool)

    gt_buf = binary_dilation(gt, structure=kernel) if buffer_px > 0 else gt
    pred_buf = binary_dilation(pred, structure=kernel) if buffer_px > 0 else pred

    tp = (pred & gt_buf).sum()
    fp = (pred & ~gt_buf).sum()
    fn = (~pred_buf & gt).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        f"buffered_precision_{buffer_px}px": float(precision),
        f"buffered_recall_{buffer_px}px": float(recall),
        f"buffered_f1_{buffer_px}px": float(f1),
    }


def _disk_kernel(radius: int) -> np.ndarray:
    """Create a disk-shaped structuring element."""
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= radius * radius


def compute_all_edge_metrics(
    pred_heatmap: np.ndarray,
    gt_heatmap: np.ndarray,
    threshold: float = 0.3,
) -> Dict[str, float]:
    """Compute all edge metrics from heatmaps.

    Args:
        pred_heatmap: [H, W] sigmoid prediction.
        gt_heatmap: [H, W] binary GT edge map (0/1 or 0/255).
        threshold: Binarization threshold for pred.

    Returns:
        Combined metrics dict.
    """
    pred_bin = (pred_heatmap >= threshold).astype(np.uint8)
    gt_bin = (gt_heatmap > 0.5).astype(np.uint8) if gt_heatmap.max() > 1 else gt_heatmap.astype(np.uint8)

    metrics = {}
    metrics.update(pixel_edge_metrics(pred_bin, gt_bin))

    pred_coords = _coords_from_binary(pred_bin)
    gt_coords = _coords_from_binary(gt_bin)

    metrics["chamfer_distance_px"] = chamfer_distance(pred_coords, gt_coords)
    metrics["hausdorff_95_px"] = hausdorff_distance(pred_coords, gt_coords, percentile=95)
    metrics["hausdorff_100_px"] = hausdorff_distance(pred_coords, gt_coords, percentile=100)

    for buf in [1, 3]:
        metrics.update(buffered_f1(pred_bin, gt_bin, buffer_px=buf))

    metrics["pred_coord_count"] = len(pred_coords)
    metrics["gt_coord_count"] = len(gt_coords)

    return metrics


if __name__ == "__main__":
    # Smoke test on synthetic data
    gt = np.zeros((224, 224), dtype=np.uint8)
    gt[100:120, 50:170] = 1

    pred = np.zeros((224, 224), dtype=np.float32)
    pred[100:120, 50:170] = 0.8
    pred[102:118, 55:165] = 0.9

    metrics = compute_all_edge_metrics(pred, gt, threshold=0.3)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("Metrics smoke test passed.")
```

- [ ] **Step 2: Run smoke test**

```bash
cd /home/ma-user/work/CoastGPT && python utils/coastline_metrics.py
```

Expected: prints metrics with reasonable values (pixel_f1 near 0.9 for aligned data), no errors.

- [ ] **Step 3: Commit**

```bash
git add utils/coastline_metrics.py && git commit -m "feat: add coastline edge metrics (buffered-F1, Chamfer, Hausdorff) for PoC-3"
```

---

### Task 5: Create `Dataset/coastline_dataset.py`

**Files:**
- Create: `Dataset/coastline_dataset.py`

- [ ] **Step 1: Write the Dataset and manifest builder**

```python
"""Coastline edge detection Dataset for PoC-3.

Loads coastline tiles, generates edge GT from GeoJSON LineString labels
(or Binary TIF fallback), returns image + edge target tensors.

Manifest builder scans coastline data directories and emits a JSON manifest
with per-tile metadata.
"""

from __future__ import annotations

import json
import math
import os
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
IMAGE_SIZE = 224       # Model input size


# ---------------------------------------------------------------------------
# Manifest Builder
# ---------------------------------------------------------------------------

def _extract_tile_name(fname: str) -> str:
    """Extract tile name stem without suffix."""
    for ext in ['.tif', '.jpg', '.png']:
        if fname.endswith(ext):
            return fname[:-len(ext)]
    return fname


def scan_coastline_directory(root: str) -> List[dict]:
    """Scan coastline directories and emit tile metadata.

    Handles both Level 1 (海岸线/) and Level 2 (7 sub-types) directory structures.

    Expected structure:
      .../Patches/<category>/<sensor>/<size>/Image_Orig/*.tif
      .../Patches/<category>/<sensor>/<size>/Label_GeoJSON/*.geojson
      .../Patches/<category>/<sensor>/<size>/Label_Binary/*.tif
    """
    tiles = []
    root = Path(root)

    for image_path in root.rglob("Image_Orig/*.tif"):
        if image_path.name.startswith("._"):
            continue

        tile_name = _extract_tile_name(image_path.name)

        # Derive paths
        label_geojson = image_path.parent.parent / "Label_GeoJSON" / f"{tile_name}_Label_WRZ.geojson"
        label_binary = image_path.parent.parent / "Label_Binary" / f"{tile_name}_Binary_WRZ.tif"

        # Derive image subdirs
        tc_path = image_path.parent.parent / "Image_TrueColor" / f"{tile_name}_True_WRZ.jpg"
        fc_path = image_path.parent.parent / "Image_FalseColor" / f"{tile_name}_False_WRZ.jpg"

        # Determine size from path
        size_dir = image_path.parent.parent.name  # e.g., "Size_128"
        try:
            tile_size = int(size_dir.replace("Size_", ""))
        except ValueError:
            tile_size = 256

        # Determine category from path components
        parts = image_path.parts
        category = "海岸线"
        shoreline_type = ""
        for p in parts:
            if p in ("海岸线", "砂质岸线", "基岩岸线", "建设围堤", "河口岸线", "港口岸线", "生物岸线", "盐田围堤"):
                category = p
                if p != "海岸线":
                    shoreline_type = p

        # Try to load tile_bounds from GeoJSON (if available)
        tile_bounds = None
        source_crs = "EPSG:4326"
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
            except (json.JSONDecodeError, OSError):
                pass

        # Derive tile bounds from tile name (R/C grid for GF tiles)
        # GF tile naming: ..._R###C###_<size>_...
        import re
        match = re.search(r'_R(\d+)C(\d+)_', tile_name)
        if match:
            # Approximate tile bounds from row/col — for EPSG:4326 tiles at ~1m GSD:
            # Each tile covers nominally 128/256 * ~1e-5 degrees
            # For more accuracy, use source image bounds + R/C offset
            # For now, we use GeoJSON feature bounds as fallback
            pass

        # Binary TIF availability
        binary_label_path = str(label_binary) if label_binary.exists() else None

        # Image path: prefer TIF, fallback to TrueColor
        img_path = str(image_path)
        if not os.path.exists(img_path):
            if tc_path.exists():
                img_path = str(tc_path)
            elif fc_path.exists():
                img_path = str(fc_path)

        tile = {
            "sample_id": tile_name,
            "image_path": img_path,
            "label_geojson_path": str(label_geojson) if label_geojson.exists() else None,
            "binary_label_path": binary_label_path,
            "category": category,
            "shoreline_type": shoreline_type or category,
            "source_crs": source_crs,
            "original_size": [tile_size, tile_size],
            "model_input_size": [IMAGE_SIZE, IMAGE_SIZE],
            "has_object": has_object,
            "num_linestrings": num_linestrings,
            "geojson_valid": geojson_valid,
            "label_source": "geojson_answer" if geojson_valid else "none",
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

    # Group by source image (extracted from tile name)
    source_groups: Dict[str, List[dict]] = {}
    for tile in all_tiles:
        # Extract source image key: everything before "_R###C###"
        name = tile["sample_id"]
        import re
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

    Uses a simple scanline approach: for each pixel within line_width/2 of
    any segment, set to 1.

    Args:
        coords: Pixel coordinates (col, row).
        size: (H, W) of output edge map.
        line_width: Width of drawn edge in pixels.

    Returns:
        edge_map: [H, W] float32 binary map.
    """
    from skimage.draw import line_aa

    edge_map = np.zeros(size, dtype=np.float32)
    half_w = max(1, line_width // 2)

    for i in range(len(coords) - 1):
        c0, r0 = coords[i]
        c1, r1 = coords[i + 1]

        rr, cc, val = line_aa(int(r0), int(c0), int(r1), int(c1))

        # Clip to bounds
        valid = (rr >= 0) & (rr < size[1]) & (cc >= 0) & (cc < size[0])
        rr, cc, val = rr[valid], cc[valid], val[valid]

        edge_map[cc, rr] = np.maximum(edge_map[cc, rr], val)

    # Dilate for line width
    if line_width > 1:
        from scipy.ndimage import binary_dilation
        # Create disk kernel
        y, x = np.ogrid[-half_w:half_w + 1, -half_w:half_w + 1]
        kernel = (x * x + y * y) <= half_w * half_w
        edge_map = binary_dilation(edge_map > 0, structure=kernel).astype(np.float32)

    return np.clip(edge_map, 0.0, 1.0)


def generate_edge_target_from_geojson(
    label_geojson_path: str,
    georef: dict,
    line_width: int = LINE_WIDTH_TRAIN,
) -> np.ndarray:
    """Generate edge GT from GeoJSON LineString labels.

    Args:
        label_geojson_path: Path to GeoJSON FeatureCollection.
        georef: Georeference dict with model_transform, source_crs.
        line_width: Edge line width in pixels.

    Returns:
        edge_target: [224, 224] float32 binary/soft edge map.
    """
    with open(label_geojson_path) as f:
        gj = json.load(f)

    edge_map = np.zeros((224, 224), dtype=np.float32)

    for feat in gj.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") not in ("LineString", "MultiLineString"):
            continue

        coords_wgs84 = geom.get("coordinates", [])
        if geom.get("type") == "LineString":
            lines = [coords_wgs84]
        else:  # MultiLineString
            lines = coords_wgs84

        for line in lines:
            if len(line) < 2:
                continue
            # WGS84 → pixel
            coords_pixel = wgs84_to_pixel(
                [(lon, lat) for lon, lat in line], georef
            )
            # Clip to bounds
            coords_pixel = [
                (max(0.0, min(223.999, col)), max(0.0, min(223.999, row)))
                for col, row in coords_pixel
            ]
            # Densify
            coords_pixel = densify_polyline(coords_pixel)
            # Draw
            line_map = draw_edge_map(coords_pixel, (224, 224), line_width)
            edge_map = np.maximum(edge_map, line_map)

    return edge_map


def generate_edge_target_from_binary_tif(
    binary_tif_path: str,
    line_width: int = LINE_WIDTH_TRAIN,
) -> np.ndarray:
    """Generate edge GT from Binary TIF label.

    The TIF is already a binary mask; this resizes it to 224×224 and
    optionally skeletonizes + re-dilates to the desired line width.

    Args:
        binary_tif_path: Path to binary TIF.
        line_width: Target line width.

    Returns:
        edge_target: [224, 224] float32 binary edge map.
    """
    from PIL import Image

    binary = Image.open(binary_tif_path).convert("L")
    binary = binary.resize((224, 224), Image.NEAREST)
    binary_np = (np.array(binary) > 128).astype(np.float32)

    if line_width > 1 and binary_np.sum() > 0:
        # Skeletonize then re-dilate
        from skimage.morphology import skeletonize
        skeleton = skeletonize(binary_np.astype(bool))
        from scipy.ndimage import binary_dilation
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
    ):
        self.tiles = tiles
        self.line_width = line_width

        if not self.tiles:
            warnings.warn("CoastlineEdgeDataset initialized with 0 tiles")

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> dict:
        tile = self.tiles[idx]

        # Load image
        image = self._load_image(tile["image_path"])

        # Build georef
        georef = self._build_georef(tile)

        # Generate edge target
        if tile.get("binary_label_path"):
            try:
                target = generate_edge_target_from_binary_tif(
                    tile["binary_label_path"], self.line_width
                )
            except Exception:
                target = np.zeros((224, 224), dtype=np.float32)
        elif tile.get("label_geojson_path"):
            try:
                target = generate_edge_target_from_geojson(
                    tile["label_geojson_path"], georef, self.line_width
                )
            except Exception:
                target = np.zeros((224, 224), dtype=np.float32)
        else:
            target = np.zeros((224, 224), dtype=np.float32)

        meta = {
            "sample_id": tile["sample_id"],
            "image_path": tile["image_path"],
            "source_crs": tile.get("source_crs", "EPSG:4326"),
            "original_transform": tile.get("original_transform", [1e-5, 0, 0, 0, -1e-5, 0]),
            "model_transform": georef["model_transform"],
            "original_size": tile.get("original_size", [128, 128]),
            "model_input_size": [IMAGE_SIZE, IMAGE_SIZE],
            "resize_scale": georef["resize_scale"],
            "has_edge": bool(tile.get("has_object", False)),
            "num_linestrings": tile.get("num_linestrings", 0),
        }

        return {
            "image": torch.from_numpy(image).float(),
            "target": torch.from_numpy(target).float().unsqueeze(0),  # [1, H, W]
            "meta": meta,
        }

    def _load_image(self, path: str) -> np.ndarray:
        """Load image and resize to IMAGE_SIZE×IMAGE_SIZE."""
        img = Image.open(path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)  # [3, H, W]

    def _build_georef(self, tile: dict) -> dict:
        """Build georef dict with model_transform and resize_scale."""
        original_size = tile.get("original_size", [128, 128])
        original_transform = tile.get("original_transform", [1e-5, 0, 0, 0, -1e-5, 0])

        if original_transform is None or len(original_transform) < 6:
            # Fallback: derive from tile_bounds if available
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
            (IMAGE_SIZE, IMAGE_SIZE),
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


if __name__ == "__main__":
    print("Testing coastline dataset...")

    # Build a small manifest from Level 2 directory
    roots = [
        "/home/ma-user/work/Stage3Data/海岸线/RS-海岸线二级/Patches",
        "/home/ma-user/work/Stage3Data/海岸线/RS-海岸线一级/Patches",
    ]
    manifest = build_coastline_manifest(
        roots,
        output_path="/tmp/poc3_coastline_manifest_test.json",
        val_ratio=0.2,
    )
    print(f"Train: {len(manifest['train'])}, Val: {len(manifest['val'])}")

    # Smoke test dataset on 4 training tiles
    ds = CoastlineEdgeDataset(manifest["train"][:4])
    print(f"Dataset: {len(ds)} samples")
    sample = ds[0]
    print(f"  image: {sample['image'].shape}, range=[{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"  target: {sample['target'].shape}, fg_ratio={sample['target'].mean():.4f}")
    print(f"  meta: {sample['meta']['sample_id']}, has_edge={sample['meta']['has_edge']}")
    print("Dataset smoke test passed.")
```

- [ ] **Step 2: Run dataset smoke test (manifest build + Dataset dry-run)**

```bash
cd /home/ma-user/work/CoastGPT && python Dataset/coastline_dataset.py
```

Expected: scans coastline directories, prints tile counts, builds manifest, loads 4 samples with valid image/target shapes.

- [ ] **Step 3: Commit**

```bash
git add Dataset/coastline_dataset.py && git commit -m "feat: add CoastlineEdgeDataset and manifest builder for PoC-3"
```

---

### Task 6: Create `configs/poc3_edge_a0_closure.yaml`

**Files:**
- Create: `configs/poc3_edge_a0_closure.yaml`

- [ ] **Step 1: Write A0 config**

```yaml
# PoC-3 A0: Pipeline closure with BCE+Dice single-scale Edge Head
# Goal: verify GT generation, training, skeleton, GeoJSON output.
# NOT a performance baseline — results should NOT be used for performance claims.

experiment:
  name: poc3_a0_bce_dice_closure
  output_dir: outputs/poc3_edge/a0_bce_dice_closure
  seed: 42
  stage: P3-A0

data:
  roots:
    - /home/ma-user/work/Stage3Data/海岸线/RS-海岸线二级/Patches
    - /home/ma-user/work/Stage3Data/海岸线/RS-海岸线一级/Patches
  manifest_path: /home/ma-user/work/CoastGPT/outputs/poc3_edge/coastline_manifest.json
  image_size: 224
  val_ratio: 0.2
  val_split_seed: 42
  num_workers: 2
  line_width_train: 3
  line_width_eval: 1

model:
  alignment_dim: 1024

  rgb_vision:
    arch: dual
    global_encoder_name: dinov3_vitl16
    local_source: dino
    local_encoder_name: convnext_base
    local_ckpt_path: ./dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth
    freeze_global: true
    freeze_local: true
    global_ckpt_path: ./dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
    input_size: [224, 224]
    physical_prompt_dim: 4096
    patch_dropout: 0.0
    input_patchnorm: false
    tune_pooler: false
    attn_pooler:
      num_query: 144
      num_attn_heads: 16
      num_layers: 6

  vision_checkpoint: ./output/stage2/checkpoints/iter_2879_consolidated.pt

  fpn:
    in_channels: [128, 256, 512, 1024]
    out_channels: 256
    vit_in_channels: 1024

  edge_head:
    type: single_scale
    in_channels: 256
    decoder_channels: [256, 128]
    output_size: [224, 224]

train:
  device: npu
  epochs: 20
  batch_size: 2
  lr_fpn: 5.0e-5
  lr_edge_head: 1.0e-4
  weight_decay: 1.0e-4
  max_grad_norm: 1.0
  log_interval: 20
  val_interval: 5
  save_interval: 5
  accum_steps: 1
  freeze_vision: true
  precision: bf16

loss:
  type: bce_dice
  # Focal+Dice params not used in A0 but declared for forward-compat
  focal_alpha: 0.75
  focal_gamma: 2.0

postprocess:
  mode: minimal
  threshold: 0.5
  min_component_area: 8
  min_line_length_px: 10
  max_components: 5
  simplify_epsilon_px: 1.0

eval:
  export_overlay: true
  export_geojson: true
  max_val_batches: 0    # 0 = all
  max_overlay_samples: 12
  threshold_sweep: false
```

- [ ] **Step 2: Commit**

```bash
git add configs/poc3_edge_a0_closure.yaml && git commit -m "feat: add PoC-3 A0 BCE+Dice single-scale config"
```

---

### Task 7: Create `scripts/poc_stage_edge.py`

**Files:**
- Create: `scripts/poc_stage_edge.py`

- [ ] **Step 1: Write the training script**

```python
#!/usr/bin/env python3
"""
PoC-3 Coastline Edge Head Training Script.

Orchestrates the PoC-3 training pipeline:
  - Builds coastline manifest from data directories
  - Loads frozen DualVisionEncoder from checkpoint
  - Builds ViT-FPN + Edge Head
  - Trains on coastline edge data
  - Evaluates with pixel metrics and buffered-F1
  - Exports overlay visualizations and GeoJSON predictions

Single NPU (or CPU), no DeepSpeed.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import numpy as np
import torch
import torch.nn as nn
import yaml
from ml_collections import ConfigDict
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.dual_vision_encoder import DualVisionEncoder
from Models.fpn_neck import FPNNeck
from Models.edge_head import SingleScaleEdgeHead
from Dataset.coastline_dataset import (
    CoastlineEdgeDataset,
    coastline_collate_fn,
    build_coastline_manifest,
)
from utils.edge_losses import edge_bce_dice_loss
from utils.edge_postprocess import postprocess_edge
from utils.coastline_metrics import compute_all_edge_metrics
from utils.georef_transform import round_trip_check, wgs84_to_pixel


# =============================================================================
# Config loading
# =============================================================================


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config from {config_path}")
    return cfg


# =============================================================================
# Vision encoder
# =============================================================================


def clean_vision_state_dict(state_dict: dict) -> dict:
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        if new_k.startswith("vision."):
            new_k = new_k[len("vision."):]
        cleaned[new_k] = v
    return cleaned


def build_vision_encoder(model_cfg: ConfigDict, ckpt_path: str) -> DualVisionEncoder:
    print("Building DualVisionEncoder...")
    vision = DualVisionEncoder(model_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "vision_ckpt" in ckpt:
            state_dict = ckpt["vision_ckpt"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    state_dict = clean_vision_state_dict(state_dict)
    model_keys = set(vision.state_dict().keys())
    matched = len(model_keys & set(state_dict.keys()))
    ratio = matched / max(len(model_keys), 1)

    print(f"  Matched {matched}/{len(model_keys)} vision params ({ratio:.1%})")
    vision.load_state_dict(state_dict, strict=False)
    return vision


# =============================================================================
# Device
# =============================================================================


def resolve_device(device_str: str) -> torch.device:
    if device_str == "npu":
        try:
            import torch_npu  # noqa: F401
            return torch.device("npu:0")
        except (ImportError, RuntimeError):
            return torch.device("cpu")
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# =============================================================================
# Checkpointing
# =============================================================================


def save_checkpoint(
    fpn: FPNNeck,
    edge_head: SingleScaleEdgeHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_dir: str,
) -> str:
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"

    torch.save(
        {
            "epoch": epoch,
            "fpn": fpn.state_dict(),
            "edge_head": edge_head.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        str(ckpt_path),
    )
    print(f"Checkpoint saved to {ckpt_path}")
    return str(ckpt_path)


# =============================================================================
# Training loop
# =============================================================================


def train_epoch(
    fpn: FPNNeck,
    edge_head: SingleScaleEdgeHead,
    vision: DualVisionEncoder,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 20,
    max_grad_norm: float = 1.0,
) -> float:
    vision.eval()
    fpn.train()
    edge_head.train()

    total_loss_sum = 0.0
    total_steps = 0

    for batch_idx, (images, targets, metas) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            image_seq, g_grid, pyramid_raw = vision.encode_with_spatial(images)

        c4, c8, c16, c32 = pyramid_raw
        p1, p2, p3, p4 = fpn(c4, c8, c16, c32, vit_feat=g_grid if fpn.has_vit else None)
        logits = edge_head(p1, p2, p3, p4)

        total_loss, loss_bce, loss_dice = edge_bce_dice_loss(logits, targets)

        total_loss.backward()

        if max_grad_norm > 0:
            trainable = list(fpn.parameters()) + list(edge_head.parameters())
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)

        optimizer.step()

        total_loss_sum += float(total_loss.item())
        total_steps += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss_sum / total_steps
            fg_ratio = float(targets.mean().item()) if targets.numel() > 0 else 0.0
            print(
                f"Epoch {epoch:3d} | Step {batch_idx + 1:5d} | "
                f"Avg Loss: {avg_loss:.4f} | BCE: {loss_bce.item():.3f} "
                f"Dice: {loss_dice.item():.3f} | GT_fg: {fg_ratio:.4f}"
            )

    avg_loss = total_loss_sum / max(total_steps, 1)
    print(f"Epoch {epoch:3d} complete | Avg Loss: {avg_loss:.4f}")
    return avg_loss


# =============================================================================
# Validation
# =============================================================================


@torch.no_grad()
def validate(
    vision: DualVisionEncoder,
    fpn: FPNNeck,
    edge_head: SingleScaleEdgeHead,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    epoch: int,
    threshold: float = 0.5,
    max_batches: int = 0,
) -> dict:
    vision.eval()
    fpn.eval()
    edge_head.eval()

    val_output_dir = Path(output_dir) / "vis" / f"val_epoch_{epoch:03d}"
    val_output_dir.mkdir(parents=True, exist_ok=True)
    geojson_dir = val_output_dir / "geojson"
    geojson_dir.mkdir(exist_ok=True)

    all_metrics: List[dict] = []
    overlay_count = 0
    max_overlay = 12

    for batch_idx, (images, targets, metas) in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        images = images.to(device)
        targets_np = targets[:, 0].numpy()  # [B, H, W]

        image_seq, g_grid, pyramid_raw = vision.encode_with_spatial(images)
        c4, c8, c16, c32 = pyramid_raw
        p1, p2, p3, p4 = fpn(c4, c8, c16, c32, vit_feat=g_grid if fpn.has_vit else None)
        logits = edge_head(p1, p2, p3, p4)

        probs = torch.sigmoid(logits)
        probs_np = probs[:, 0].cpu().numpy()  # [B, H, W]

        for i in range(images.shape[0]):
            # Compute metrics
            metrics = compute_all_edge_metrics(
                probs_np[i], targets_np[i], threshold=threshold
            )
            metrics["sample_id"] = metas[i].get("sample_id", f"batch{batch_idx}_idx{i}")
            all_metrics.append(metrics)

            # Save overlays (limited count)
            if overlay_count < max_overlay:
                _save_edge_overlay(
                    images[i].cpu(),
                    targets_np[i],
                    probs_np[i],
                    metas[i],
                    epoch,
                    val_output_dir,
                    overlay_count,
                    threshold=threshold,
                )
                overlay_count += 1

            # Export GeoJSON (first 5 samples)
            if overlay_count <= 5:
                georef = {
                    "source_crs": metas[i].get("source_crs", "EPSG:4326"),
                    "model_transform": metas[i].get("model_transform", [1e-5, 0, 0, 0, -1e-5, 0]),
                }
                fc = postprocess_edge(
                    probs_np[i],
                    georef,
                    threshold=threshold,
                    min_length=10,
                    max_components=5,
                    simplify_epsilon=1.0,
                    sample_id=metas[i].get("sample_id", ""),
                )
                geo_path = geojson_dir / f"{metas[i].get('sample_id', f'sample_{i}')}.geojson"
                with open(geo_path, 'w', encoding='utf-8') as f:
                    json.dump(fc, f, ensure_ascii=False, indent=2)

    # Aggregate metrics
    agg = _aggregate_metrics(all_metrics)
    print(f"Validation epoch {epoch}: "
          f"pixel_f1={agg.get('pixel_f1', 0):.4f}, "
          f"buffered_f1_1px={agg.get('buffered_f1_1px', 0):.4f}, "
          f"buffered_f1_3px={agg.get('buffered_f1_3px', 0):.4f}, "
          f"chamfer={agg.get('chamfer_distance_px', 0):.2f}px, "
          f"n_samples={len(all_metrics)}")

    return agg


def _aggregate_metrics(metrics_list: List[dict]) -> dict:
    if not metrics_list:
        return {}
    agg = {}
    for key in metrics_list[0]:
        vals = [m[key] for m in metrics_list if isinstance(m.get(key), (int, float))]
        if vals:
            agg[key] = float(np.mean(vals))
    return agg


def _save_edge_overlay(
    image: torch.Tensor,
    gt: np.ndarray,
    pred: np.ndarray,
    meta: dict,
    epoch: int,
    output_dir: Path,
    sample_idx: int,
    threshold: float = 0.5,
):
    """Save GT, prediction, and skeleton overlay images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample_id = meta.get("sample_id", f"sample_{sample_idx}")

    # Image: [3, H, W] → [H, W, 3]
    img = image.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: image, GT, GT overlay
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt, cmap="gray")
    axes[0, 1].set_title("GT Edge")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(img)
    axes[0, 2].imshow(gt, cmap="Reds", alpha=0.5)
    axes[0, 2].set_title("GT Overlay")
    axes[0, 2].axis("off")

    # Row 2: pred heatmap, pred binary, pred skeleton
    axes[1, 0].imshow(pred, cmap="hot", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Pred Heatmap (thresh={threshold})")
    axes[1, 0].axis("off")

    pred_bin = (pred >= threshold).astype(np.uint8)
    axes[1, 1].imshow(pred_bin, cmap="gray")
    axes[1, 1].set_title("Pred Binary")
    axes[1, 1].axis("off")

    # Skeleton
    try:
        from utils.edge_postprocess import heatmap_to_binary, binary_to_skeleton
        binary = heatmap_to_binary(pred, threshold=threshold, min_area=8)
        skeleton = binary_to_skeleton(binary)
        axes[1, 2].imshow(img)
        axes[1, 2].imshow(skeleton, cmap="Greens", alpha=0.7)
        axes[1, 2].set_title("Pred Skeleton Overlay")
    except Exception:
        axes[1, 2].imshow(img)
        axes[1, 2].set_title("Skeleton (skimage not available)")
    axes[1, 2].axis("off")

    plt.suptitle(f"Epoch {epoch} — {sample_id}")
    plt.tight_layout()
    save_path = output_dir / f"{sample_id}_overlay.png"
    plt.savefig(str(save_path), dpi=100, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="PoC-3 Coastline Edge Head Training")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to YAML config")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (npu, cuda, cpu)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--build-manifest-only", action="store_true",
                        help="Build manifest and exit")
    args = parser.parse_args()

    cfg_raw = load_config(args.config)
    cfg = ConfigDict(cfg_raw)

    output_dir = args.output or cfg.get("experiment.output_dir", "outputs/poc3_edge/a0")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device_str = args.device or cfg.get("train.device", "cpu")
    device = resolve_device(device_str)
    print(f"Device: {device}")

    # Build manifest
    print("Building coastline manifest...")
    manifest_path = cfg.get("data.manifest_path",
                             str(Path(output_dir) / "coastline_manifest.json"))
    manifest = build_coastline_manifest(
        data_roots=cfg.get("data.roots", []),
        output_path=manifest_path,
        val_ratio=cfg.get("data.val_ratio", 0.2),
        seed=cfg.get("data.val_split_seed", 42),
    )

    if args.build_manifest_only:
        print("Manifest built. Exiting.")
        return

    # Build vision encoder
    model_cfg = ConfigDict(cfg_raw.get("model", {}))
    vision = build_vision_encoder(
        model_cfg,
        cfg.get("model.vision_checkpoint",
                "./output/stage2/checkpoints/iter_2879_consolidated.pt")
    )
    vision = vision.to(device)
    vision.eval()
    for p in vision.parameters():
        p.requires_grad = False

    # Build FPN
    fpn_cfg = cfg.get("model.fpn", {})
    fpn = FPNNeck(
        in_channels=fpn_cfg.get("in_channels", [128, 256, 512, 1024]),
        out_channels=fpn_cfg.get("out_channels", 256),
        vit_in_channels=fpn_cfg.get("vit_in_channels", 1024),
    )
    fpn = fpn.to(device)
    print(f"FPN params: {sum(p.numel() for p in fpn.parameters()):,}")

    # Build Edge Head
    edge_head = SingleScaleEdgeHead(
        in_channels=cfg.get("model.edge_head.in_channels", 256),
        decoder_channels=tuple(cfg.get("model.edge_head.decoder_channels", [256, 128])),
        output_size=tuple(cfg.get("model.edge_head.output_size", [224, 224])),
    )
    edge_head = edge_head.to(device)
    print(f"Edge Head params: {sum(p.numel() for p in edge_head.parameters()):,}")

    # Datasets
    line_width_train = cfg.get("data.line_width_train", 3)
    train_ds = CoastlineEdgeDataset(manifest["train"], line_width=line_width_train)
    val_ds = CoastlineEdgeDataset(manifest["val"], line_width=cfg.get("data.line_width_eval", 1))

    batch_size = args.batch_size or cfg.get("train.batch_size", 2)
    num_workers = cfg.get("data.num_workers", 2)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=coastline_collate_fn,
        pin_memory=(device_str != "cpu"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=coastline_collate_fn,
    )
    print(f"Data: {len(train_ds)} train, {len(val_ds)} val samples")

    # Optimizer
    trainable = list(fpn.parameters()) + list(edge_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": fpn.parameters(), "lr": cfg.get("train.lr_fpn", 5e-5)},
            {"params": edge_head.parameters(), "lr": cfg.get("train.lr_edge_head", 1e-4)},
        ],
        weight_decay=cfg.get("train.weight_decay", 1e-4),
    )

    epochs = args.epochs or cfg.get("train.epochs", 20)
    val_interval = cfg.get("train.val_interval", 5)
    save_interval = cfg.get("train.save_interval", 5)
    log_interval = cfg.get("train.log_interval", 20)
    max_grad_norm = cfg.get("train.max_grad_norm", 1.0)
    threshold = cfg.get("postprocess.threshold", 0.5)
    max_val_batches = cfg.get("eval.max_val_batches", 0)

    print(f"Training: {epochs} epochs, batch={batch_size}, lr_fpn={cfg.get('train.lr_fpn', 5e-5)}, "
          f"lr_edge={cfg.get('train.lr_edge_head', 1e-4)}")
    print(f"Loss: BCE+Dice (A0)")

    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        avg_loss = train_epoch(
            fpn, edge_head, vision, train_loader, optimizer, device,
            epoch=epoch, log_interval=log_interval, max_grad_norm=max_grad_norm,
        )

        if epoch % val_interval == 0:
            metrics = validate(
                vision, fpn, edge_head, val_loader, device,
                output_dir=output_dir, epoch=epoch,
                threshold=threshold, max_batches=max_val_batches,
            )
            f1_1px = metrics.get("buffered_f1_1px", 0)
            if f1_1px > best_f1:
                best_f1 = f1_1px
                save_checkpoint(fpn, edge_head, optimizer, 0, output_dir)  # as best
                print(f"  New best buffered-F1@1px: {best_f1:.4f}")

        if epoch % save_interval == 0:
            save_checkpoint(fpn, edge_head, optimizer, epoch, output_dir)

    # Final checkpoint
    save_checkpoint(fpn, edge_head, optimizer, epochs, output_dir)
    print(f"Training complete. Best buffered-F1@1px: {best_f1:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run manifest build to verify data scanning**

```bash
cd /home/ma-user/work/CoastGPT && python scripts/poc_stage_edge.py \
  --config configs/poc3_edge_a0_closure.yaml --build-manifest-only
```

Expected: scans directories, prints tile counts (~954 tiles), builds manifest JSON at `outputs/poc3_edge/a0_bce_dice_closure/coastline_manifest.json`.

- [ ] **Step 3: Run dry-run training on CPU (1 epoch, 2 batches)**

```bash
cd /home/ma-user/work/CoastGPT && python scripts/poc_stage_edge.py \
  --config configs/poc3_edge_a0_closure.yaml --device cpu --epochs 1 --batch-size 2
```

Expected: loads vision encoder, builds FPN + edge head, runs 1 training epoch with decreasing loss, runs validation, exports overlays to `outputs/poc3_edge/a0_bce_dice_closure/vis/`.

- [ ] **Step 4: Commit**

```bash
git add scripts/poc_stage_edge.py && git commit -m "feat: add PoC-3 edge head training script (A0 BCE+Dice single-NPU)"
```

