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
