"""
FPN Neck + DualVisionFPNBackboneAdapter + Mask R-CNN builder.

Provides the detection head for PoC-1:
  - FPNNeck: converts ConvNeXt pyramid features (c4/c8/c16/c32) to unified
    256-channel P1-P4 feature maps via 1x1 lateral convs + top-down fusion.
  - DualVisionFPNBackboneAdapter: wraps a frozen DualVisionEncoder + FPNNeck
    into a torchvision-compatible backbone (outputs OrderedDict).
  - build_aqua_maskrcnn(): constructs a torchvision MaskRCNN with the adapter.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.dual_vision_encoder import DualVisionEncoder
from Models.fpn_neck import FPNNeck


class DualVisionFPNBackboneAdapter(nn.Module):
    """
    Wrap frozen DualVisionEncoder + FPNNeck as torchvision Mask R-CNN backbone.

    torchvision's GeneralizedRCNNTransform batches List[Tensor] into ImageList,
    then passes batched Tensor[B, 3, H, W] to backbone.forward().

    Input:  Tensor[B, 3, 224, 224]    (NOT List[Tensor])
    Output: OrderedDict[str, Tensor]
            {"0": P2, "1": P3, "2": P4}   # for Mask R-CNN featmap_names
    """

    def __init__(self, vision_encoder: DualVisionEncoder, fpn_neck: FPNNeck):
        super().__init__()
        self.vision = vision_encoder
        self.fpn = fpn_neck
        self.out_channels = fpn_neck.out_channels

        # Freeze vision
        self.vision.eval()
        for p in self.vision.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        """Override to keep vision encoder permanently in eval mode."""
        super().train(mode)
        self.vision.eval()
        return self

    def forward(self, images: torch.Tensor) -> OrderedDict:
        """
        Args:
            images: Tensor[B, 3, 224, 224]  (batched by GeneralizedRCNNTransform)
        """
        with torch.no_grad():
            image_seq, g_grid, pyramid_raw = self.vision.encode_with_spatial(images)

        c4, c8, c16, c32 = pyramid_raw
        p1, p2, p3, p4 = self.fpn(c4, c8, c16, c32)

        return OrderedDict({
            "0": p2,  # [B, 256, 28, 28]
            "1": p3,  # [B, 256, 14, 14]
            "2": p4,  # [B, 256, 7, 7]
        })


def _build_anchor_sizes(
    relative_scales: Tuple[Tuple[float, ...], ...],
    input_h: int,
    input_w: int,
) -> Tuple[Tuple[int, ...], ...]:
    """Convert relative anchor scales to absolute pixel sizes.

    Args:
        relative_scales: e.g. ((0.0714, 0.1429), (0.1429, 0.2857), (0.2857, 0.4286, 0.5714)).
        input_h, input_w: Current input spatial dimensions.

    Returns:
        Absolute anchor sizes, e.g. ((16, 32), (32, 64), (64, 96, 128)) for 224 input.
    """
    base = min(input_h, input_w)
    return tuple(
        tuple(max(1, round(s * base)) for s in scales)
        for scales in relative_scales
    )


def build_aqua_maskrcnn(
    backbone_adapter: DualVisionFPNBackboneAdapter,
    num_classes: int = 2,
    # New: relative anchor scales (preferred for resolution-agnostic training)
    anchor_relative_scales: Optional[Tuple[Tuple[float, ...], ...]] = None,
    # Old: absolute anchor sizes (backward compat)
    anchor_sizes: Optional[Tuple[Tuple[int, ...], ...]] = None,
    aspect_ratios: Tuple[Tuple[float, ...], ...] = ((0.5, 1.0, 2.0, 3.0),) * 3,
    rpn_pre_nms_top_n_train: int = 512,
    rpn_post_nms_top_n_train: int = 128,
    rpn_pre_nms_top_n_test: int = 256,
    rpn_post_nms_top_n_test: int = 64,
    rpn_nms_thresh: float = 0.7,
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.5,
    box_detections_per_img: int = 50,
    image_mean: List[float] = None,
    image_std: List[float] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
) -> MaskRCNN:
    """
    Build a torchvision Mask R-CNN with a custom FPN backbone adapter.
    num_classes includes background (2 for foreground object + background).
    """
    if min_size is None:
        min_size = 224
    if max_size is None:
        max_size = min_size

    if image_mean is None:
        image_mean = [0.0, 0.0, 0.0]
    if image_std is None:
        image_std = [1.0, 1.0, 1.0]

    # Resolve anchor sizes: prefer relative scales if provided
    if anchor_relative_scales is not None:
        anchor_sizes = _build_anchor_sizes(anchor_relative_scales, min_size, min_size)
    elif anchor_sizes is None:
        # Legacy default (224 inputs)
        anchor_sizes = ((16, 32), (32, 64), (64, 96, 128))

    anchors_per_level = [
        len(size_level) * len(ratio_level)
        for size_level, ratio_level in zip(anchor_sizes, aspect_ratios)
    ]
    if len(set(anchors_per_level)) != 1:
        raise ValueError(
            "torchvision RPNHead requires every FPN level to have the same "
            f"number of anchors per location, got {anchors_per_level}. "
            "Use equal-length anchor sizes/aspect ratios per level."
        )

    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
    )

    # featmap_names must match OrderedDict keys from adapter
    featmap_names = ["0", "1", "2"]

    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=featmap_names,
        output_size=7,
        sampling_ratio=2,
    )

    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=featmap_names,
        output_size=14,
        sampling_ratio=2,
    )

    model = MaskRCNN(
        backbone_adapter,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool,
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
        rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
        rpn_nms_thresh=rpn_nms_thresh,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img,
        image_mean=image_mean,
        image_std=image_std,
        min_size=min_size,
        max_size=max_size,
    )

    return model


if __name__ == "__main__":
    # Test FPN shapes only (no DualVisionEncoder needed)
    fpn = FPNNeck(
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
    )
    print(f"FPN params: {sum(p.numel() for p in fpn.parameters()):,}")

    c4 = torch.randn(2, 128, 56, 56)
    c8 = torch.randn(2, 256, 28, 28)
    c16 = torch.randn(2, 512, 14, 14)
    c32 = torch.randn(2, 1024, 7, 7)

    p1, p2, p3, p4 = fpn(c4, c8, c16, c32)
    print(f"P1: {list(p1.shape)}")
    print(f"P2: {list(p2.shape)}")
    print(f"P3: {list(p3.shape)}")
    print(f"P4: {list(p4.shape)}")

    assert list(p1.shape) == [2, 256, 56, 56]
    assert list(p2.shape) == [2, 256, 28, 28]
    assert list(p3.shape) == [2, 256, 14, 14]
    assert list(p4.shape) == [2, 256, 7, 7]
    print("FPN shape check passed.")
