"""
Feature Pyramid Network neck for ConvNeXt pyramid features.

Shared module used by PoC-1 (Instance Head), PoC-2 (Semantic Head),
PoC-3 (Edge Head), and Stage 3/4 multi-head training.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNNeck(nn.Module):
    """
    Feature Pyramid Network neck for ConvNeXt pyramid features.

    Input:
        c4:  [B, 128, 56, 56]
        c8:  [B, 256, 28, 28]
        c16: [B, 512, 14, 14]
        c32: [B, 1024, 7, 7]

    Processing:
        1x1 Conv to unify channels to 256
        Top-down upsampling with lateral connections

    Output:
        P1: [B, 256, 56, 56]
        P2: [B, 256, 28, 28]
        P3: [B, 256, 14, 14]
        P4: [B, 256, 7, 7]
    """

    def __init__(
        self,
        in_channels: List[int] = (128, 256, 512, 1024),
        out_channels: int = 256,
    ):
        super().__init__()
        self.out_channels = out_channels

        # 1x1 Conv to unify channel dimensions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, kernel_size=1)
            for ch in in_channels
        ])

        # Post-merge smoothing convolutions (one per level)
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(len(in_channels))
        ])

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        c4: torch.Tensor,
        c8: torch.Tensor,
        c16: torch.Tensor,
        c32: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Lateral: 1x1 conv on each level
        lat4 = self.lateral_convs[0](c4)   # [B, 256, 56, 56]
        lat8 = self.lateral_convs[1](c8)   # [B, 256, 28, 28]
        lat16 = self.lateral_convs[2](c16) # [B, 256, 14, 14]
        lat32 = self.lateral_convs[3](c32) # [B, 256, 7, 7]

        # Top-down
        P4 = self.smooth_convs[3](lat32)
        P3 = self.smooth_convs[2](
            lat16 + F.interpolate(P4, size=lat16.shape[-2:], mode='nearest')
        )
        P2 = self.smooth_convs[1](
            lat8 + F.interpolate(P3, size=lat8.shape[-2:], mode='nearest')
        )
        P1 = self.smooth_convs[0](
            lat4 + F.interpolate(P2, size=lat4.shape[-2:], mode='nearest')
        )

        return P1, P2, P3, P4
