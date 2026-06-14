"""CoastGPT Landcover Semantic Head — standalone PyTorch module."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LandcoverSemanticHead(nn.Module):
    """Semantic segmentation head operating on FPN P1-P4 features.

    Args:
        in_channels: Channels per FPN level (default 256).
        num_classes: Number of output classes including background.
        output_size: (H, W) to upsample logits to.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 25,
        output_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.output_size = output_size

        fused_channels = in_channels * 4  # 1024 after concat

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
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self, p1: torch.Tensor, p2: torch.Tensor,
        p3: torch.Tensor, p4: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            p1: [B, 256, H/4, W/4]
            p2: [B, 256, H/8, W/8]
            p3: [B, 256, H/16, W/16]
            p4: [B, 256, H/32, W/32]

        Returns:
            logits: [B, num_classes, output_H, output_W]
        """
        h, w = p1.shape[2], p1.shape[3]

        p2_up = F.interpolate(p2, size=(h, w), mode="bilinear", align_corners=False)
        p3_up = F.interpolate(p3, size=(h, w), mode="bilinear", align_corners=False)
        p4_up = F.interpolate(p4, size=(h, w), mode="bilinear", align_corners=False)

        fused = torch.cat([p1, p2_up, p3_up, p4_up], dim=1)  # [B, 1024, h, w]

        x = self.conv1(fused)   # [B, 256, h, w]
        x = self.conv2(x)       # [B, 128, h, w]
        x = self.cls_conv(x)    # [B, num_classes, h, w]

        if self.output_size is not None:
            upsample_size = self.output_size
        else:
            upsample_size = (h * 4, w * 4)

        logits = F.interpolate(x, size=upsample_size, mode="bilinear", align_corners=False)
        return logits
