"""Gravitational Distance Field head for PoC-3 A3.

Provides:
  - SingleScaleGDFHead: FPN P1-P4 → concat → Conv decoder → 4-channel field
  - MultiScaleGDFHead: HED-style side outputs (A3-2, deferred)
"""

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


class SingleScaleGDFHead(nn.Module):
    """Single-scale GDF head: FPN P1-P4 → concat → Conv decoder → 4-channel field.

    Architecture (same as SingleScaleEdgeHead but 4 output channels):
        P1 [B,256,56,56]
        P2 [B,256,28,28] → upsample 56
        P3 [B,256,14,14] → upsample 56
        P4 [B,256,7,7]   → upsample 56
        concat → [B,1024,56,56]
        → 3×3 Conv 1024→256 + BN + ReLU
        → 3×3 Conv 256→128 + BN + ReLU
        → 1×1 Conv 128→4
        → bilinear upsample to output_size

    Output channels:
        0: dx_norm     ∈ [-1, 1] (tanh applied in loss/postprocess)
        1: dy_norm     ∈ [-1, 1]
        2: log_dist_norm ∈ [0, 1] (sigmoid applied in loss/postprocess)
        3: valid_logit ∈ R (raw, BCEWithLogits applied in loss)

    Args:
        in_channels: Channels per FPN level (default 256).
        decoder_channels: [mid_channels, pre_output_channels].
        output_size: (H, W) to upsample field to (default 224×224).
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
        self.out_channels = 4  # dx, dy, log_dist, valid_logit

        fused_channels = in_channels * 4  # 1024

        self.conv1 = _conv_bn_relu(fused_channels, decoder_channels[0])
        self.conv2 = _conv_bn_relu(decoder_channels[0], decoder_channels[1])
        self.cls_conv = nn.Conv2d(decoder_channels[1], self.out_channels, kernel_size=1)

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
                'field': [B, 4, H_out, W_out] raw field prediction
                  - field[:, 0] = dx_raw   (apply tanh for [-1,1])
                  - field[:, 1] = dy_raw   (apply tanh for [-1,1])
                  - field[:, 2] = log_dist_raw (apply sigmoid for [0,1])
                  - field[:, 3] = valid_logit  (raw, for BCEWithLogits)
        """
        h, w = p1.shape[2], p1.shape[3]

        p2_up = F.interpolate(p2, size=(h, w), mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=False)

        fused = torch.cat([p1, p2_up, p3_up, p4_up], dim=1)

        x = self.conv1(fused)
        x = self.conv2(x)
        x = self.cls_conv(x)

        field = F.interpolate(
            x, size=self.output_size, mode='bilinear', align_corners=False
        )
        return {"field": field}


class MultiScaleGDFHead(nn.Module):
    """HED-style multi-scale GDF head with side outputs (A3-2, deferred).

    Each FPN level produces a 4-channel field side output.
    Placeholder — not used in A3-0/A3-1.
    """

    def __init__(
        self,
        in_channels: int = 256,
        output_size: tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.output_size = output_size

        self.side_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 4, kernel_size=1),
            )
            for _ in range(4)
        ])
        self.fuse_conv = nn.Conv2d(16, 4, kernel_size=1)
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

    def forward(self, p1, p2, p3, p4) -> dict[str, torch.Tensor]:
        features = [p1, p2, p3, p4]
        sides = []
        for feat, side_conv in zip(features, self.side_convs):
            side_field = side_conv(feat)
            side_up = F.interpolate(
                side_field, size=self.output_size, mode='bilinear', align_corners=False
            )
            sides.append(side_up)
        fused = self.fuse_conv(torch.cat(sides, dim=1))
        return {
            "field": fused,
            "side1": sides[0], "side2": sides[1],
            "side3": sides[2], "side4": sides[3],
        }


if __name__ == "__main__":
    B = 2
    p1 = torch.randn(B, 256, 56, 56)
    p2 = torch.randn(B, 256, 28, 28)
    p3 = torch.randn(B, 256, 14, 14)
    p4 = torch.randn(B, 256, 7, 7)

    head = SingleScaleGDFHead(output_size=(224, 224))
    out = head(p1, p2, p3, p4)
    field = out["field"]
    print(f"SingleScaleGDFHead: field={list(field.shape)}")
    assert field.shape == (B, 4, 224, 224), f"Expected [2,4,224,224], got {list(field.shape)}"
    n = sum(p.numel() for p in head.parameters())
    print(f"  Params: {n:,}")

    multi = MultiScaleGDFHead(output_size=(224, 224))
    mout = multi(p1, p2, p3, p4)
    print(f"MultiScaleGDFHead: field={list(mout['field'].shape)}")
    assert mout["field"].shape == (B, 4, 224, 224)
    nm = sum(p.numel() for p in multi.parameters())
    print(f"  Params: {nm:,}")

    print("All shape checks passed.")
