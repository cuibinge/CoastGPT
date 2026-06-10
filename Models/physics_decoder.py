import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Lateral(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PhysicsDecoder(nn.Module):
    """FPN/U-Net style decoder producing multi-scale and full-resolution physical maps.

    Inputs:
      - fused_spatial: [B, C, Hf, Wf] fused features aligned to global grid
      - pyramid_raw: list of feature maps from backbone at strides ~ 1/4,1/8,1/16
      - input_size: (H, W) of the model input image for full-resolution mapping

    Outputs:
      Dict with keys: 'phy_full', 'phy_s1', 'phy_s2', 'phy_s3' (scales ~1/4,1/8,1/16)
    """

    def __init__(self, out_channels: int = 1, dec_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
        self.dec_channels = dec_channels

        # Lateral projections for up to 3 scales
        self.lateral_1 = Lateral(in_ch=dec_channels, out_ch=dec_channels)
        self.lateral_2 = Lateral(in_ch=dec_channels, out_ch=dec_channels)
        self.lateral_3 = Lateral(in_ch=dec_channels, out_ch=dec_channels)

        # Fusion convolution after upsampling and addition
        self.fuse_1 = ConvBlock(dec_channels, dec_channels)
        self.fuse_2 = ConvBlock(dec_channels, dec_channels)
        self.fuse_3 = ConvBlock(dec_channels, dec_channels)

        # Output heads for each scale
        self.head_s1 = nn.Conv2d(dec_channels, out_channels, kernel_size=1)
        self.head_s2 = nn.Conv2d(dec_channels, out_channels, kernel_size=1)
        self.head_s3 = nn.Conv2d(dec_channels, out_channels, kernel_size=1)
        self.head_full = nn.Conv2d(dec_channels, out_channels, kernel_size=1)

        # Project fused spatial to decoder channel size
        self.fused_proj = nn.Conv2d(in_channels=dec_channels, out_channels=dec_channels, kernel_size=1)

    def forward(
        self,
        fused_spatial: torch.Tensor,
        pyramid_raw: List[torch.Tensor],
        input_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        B, Cf, Hf, Wf = fused_spatial.shape

        # Ensure decoder channel size
        x_top = self.fused_proj(fused_spatial)

        # Build pyramid by downsampling fused_spatial
        s1 = x_top
        s2 = F.avg_pool2d(x_top, kernel_size=2, stride=2)
        s3 = F.avg_pool2d(s2, kernel_size=2, stride=2)
        lateral_feats = [s1, s2, s3]

        # Identify resolutions for scales (assume s1 highest resolution)
        s1 = self.lateral_1(lateral_feats[0])
        s2 = self.lateral_2(lateral_feats[1])
        s3 = self.lateral_3(lateral_feats[2])

        # Top-down fusion
        s2_up = F.interpolate(s3, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        s2 = self.fuse_2(s2 + s2_up)

        s1_up = F.interpolate(s2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        s1 = self.fuse_1(s1 + s1_up)

        # Predict per-scale physical maps
        phy_s3 = self.head_s3(s3)
        phy_s2 = self.head_s2(s2)
        phy_s1 = self.head_s1(s1)

        # Full-resolution mapping aligned to input_size
        full_feat = F.interpolate(s1, size=input_size, mode="bilinear", align_corners=False)
        phy_full = self.head_full(full_feat)

        return {
            "phy_s1": phy_s1,
            "phy_s2": phy_s2,
            "phy_s3": phy_s3,
            "phy_full": phy_full,
        }


def sobel_grad(x: torch.Tensor) -> torch.Tensor:
    # Compute simple gradient magnitude using Sobel filters
    kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kernel_x, padding=1, groups=x.shape[1])
    gy = F.conv2d(x, kernel_y, padding=1, groups=x.shape[1])
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


def pixelwise_mse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, gt)


def edge_preserve_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    gp = sobel_grad(pred)
    gg = sobel_grad(gt)
    return F.l1_loss(gp, gg)


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """Total variation regularizer over spatial dimensions.
    Args: x [B, C, H, W]
    Returns: scalar TV loss
    """
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw


def consistency_loss_multiscale(
    phy_full: torch.Tensor,
    phy_s1: torch.Tensor,
    phy_s2: torch.Tensor,
    phy_s3: torch.Tensor,
) -> torch.Tensor:
    """Encourage consistency between full-res and upsampled multi-scale predictions."""
    H, W = phy_full.shape[-2], phy_full.shape[-1]
    s1_up = F.interpolate(phy_s1, size=(H, W), mode="bilinear", align_corners=False)
    s2_up = F.interpolate(phy_s2, size=(H, W), mode="bilinear", align_corners=False)
    s3_up = F.interpolate(phy_s3, size=(H, W), mode="bilinear", align_corners=False)
    l = F.mse_loss(phy_full, s1_up) + F.mse_loss(phy_full, s2_up) + F.mse_loss(phy_full, s3_up)
    return l / 3.0


def spectral_loss_sam(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Spectral Angle Mapper (SAM) loss per pixel across channels.
    Args:
        pred: [B, C, H, W]
        gt:   [B, C, H, W]
    Returns:
        mean spectral angle (radians) across all pixels
    """
    B, C, H, W = pred.shape
    p = pred.permute(0, 2, 3, 1).reshape(-1, C)
    g = gt.permute(0, 2, 3, 1).reshape(-1, C)
    dot = (p * g).sum(dim=1)
    pn = torch.norm(p, dim=1)
    gn = torch.norm(g, dim=1)
    cos = torch.clamp(dot / (pn * gn + eps), min=-1.0, max=1.0)
    ang = torch.acos(cos)
    return ang.mean()