"""
LandcoverSemanticHead: per-pixel semantic segmentation decoder.

Takes FPN P1-P4 feature maps, fuses them, and outputs per-pixel class logits.

Architecture:
    P1 [B,256,56,56], P2 [B,256,28,28], P3 [B,256,14,14], P4 [B,256,7,7]
    → upsample P2/P3/P4 to 56×56
    → concat [B, 1024, 56, 56]
    → 3×3 Conv 256 + BN + ReLU
    → 3×3 Conv 128 + BN + ReLU
    → 1×1 Conv num_classes
    → bilinear upsample to (output_size)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LandcoverSemanticHead(nn.Module):
    """Semantic segmentation head operating on FPN feature pyramid.

    Args:
        in_channels: Number of channels per FPN level (default 256).
        num_classes: Number of output classes (including background).
        output_size: (H, W) to upsample logits to (default 224×224).
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 25,
        output_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.output_size = output_size

        # After concat: 4 * in_channels = 1024
        fused_channels = in_channels * 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(fused_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.cls_conv = nn.Conv2d(128, num_classes, kernel_size=1)

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
            logits: [B, num_classes, output_H, output_W]
        """
        # Upsample P2-P4 to P1 spatial size (56×56)
        h, w = p1.shape[2], p1.shape[3]  # 56, 56

        p2_up = F.interpolate(p2, size=(h, w), mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=False)

        # Fuse
        fused = torch.cat([p1, p2_up, p3_up, p4_up], dim=1)  # [B, 1024, 56, 56]

        x = self.conv1(fused)   # [B, 256, 56, 56]
        x = self.conv2(x)       # [B, 128, 56, 56]
        x = self.cls_conv(x)    # [B, num_classes, 56, 56]

        # Upsample to target output size
        logits = F.interpolate(
            x, size=self.output_size, mode='bilinear', align_corners=False
        )
        return logits


if __name__ == "__main__":
    B = 2
    num_classes = 25

    model = LandcoverSemanticHead(
        in_channels=256,
        num_classes=num_classes,
        output_size=(224, 224),
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"LandcoverSemanticHead params: {n_params:,}")

    p1 = torch.randn(B, 256, 56, 56)
    p2 = torch.randn(B, 256, 28, 28)
    p3 = torch.randn(B, 256, 14, 14)
    p4 = torch.randn(B, 256, 7, 7)

    logits = model(p1, p2, p3, p4)
    print(f"Input:  P1={list(p1.shape)}, P2={list(p2.shape)}, P3={list(p3.shape)}, P4={list(p4.shape)}")
    print(f"Output: {list(logits.shape)}")

    assert list(logits.shape) == [B, num_classes, 224, 224], f"Unexpected shape: {logits.shape}"
    print("LandcoverSemanticHead shape check passed.")
