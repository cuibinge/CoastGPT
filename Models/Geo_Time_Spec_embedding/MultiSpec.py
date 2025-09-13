# -*- coding: utf-8 -*-

# GF-1 PMS band metadata encoding + FiLM conditioning (PyTorch)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# 1) GF-1 PMS 波段元数据（单位：μm）。FWHM 近似取为带宽。
#   PAN: 0.45–0.90
#   Blue: 0.45–0.52
#   Green:0.52–0.59
#   Red:  0.63–0.69
#   NIR:  0.77–0.89
#   注：FWHM≈高响应近似为矩形时的带宽；如有 SRF，请用进阶方法替换。
# ------------------------------------------------------------
GF1_PMS_META = [
    {"name": "PAN",   "wl_low": 0.45, "wl_high": 0.90},  # 可选是否使用
    {"name": "Blue",  "wl_low": 0.45, "wl_high": 0.52},
    {"name": "Green", "wl_low": 0.52, "wl_high": 0.59},
    {"name": "Red",   "wl_low": 0.63, "wl_high": 0.69},
    {"name": "NIR",   "wl_low": 0.77, "wl_high": 0.89},
]


def build_gf1_band_tensors(include_pan: bool = False, device: torch.device = torch.device("cpu")):
    """
    返回:
      lambdas: (C,)  中心波长 [μm]
      fwhms:   (C,)  近似FWHM（=带宽）[μm]
      names:   list[str] 通道名称，顺序与上两者一致
    """
    metas = GF1_PMS_META[:]  # 拷贝
    if not include_pan:
        metas = [m for m in metas if m["name"] != "PAN"]

    lambdas, fwhms, names = [], [], []
    for m in metas:
        wl_c = 0.5 * (m["wl_low"] + m["wl_high"])
        bw   = m["wl_high"] - m["wl_low"]  # 近似FWHM
        lambdas.append(wl_c)
        fwhms.append(bw)
        names.append(m["name"])

    lambdas = torch.tensor(lambdas, dtype=torch.float32, device=device)  # (C,)
    fwhms   = torch.tensor(fwhms,   dtype=torch.float32, device=device)  # (C,)
    return lambdas, fwhms, names


# ------------------------------------------------------------
# 2) 波长/FWHM 连续编码（Fourier 特征）→ 通道嵌入向量
# ------------------------------------------------------------
class WavelengthEncoder(nn.Module):
    """
    将 (λ, FWHM) 编码为通道嵌入 e_c ∈ R^emb_dim
    使用 Fourier/positional 编码增强“连续波长”可泛化性。
    """
    def __init__(self, L: int = 6, emb_dim: int = 32):
        super().__init__()
        self.L = L
        # 输入维度：lambda 的 2L（sin/cos）+ (lambda/FWHM) 的 2L + FWHM 本身(1) = 4L + 1
        in_dim = 4 * L + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, emb_dim), nn.ReLU()
        )

    def _fourier_feats(self, v: torch.Tensor, L: int):
        """ v: (C,)  -> (C, 2L) """
        feats = []
        for k in range(L):
            w = (2.0 ** k) * torch.pi
            feats.append(torch.sin(w * v))
            feats.append(torch.cos(w * v))
        return torch.stack(feats, dim=-1)  # (C, 2L)

    def forward(self, lambdas: torch.Tensor, fwhms: torch.Tensor) -> torch.Tensor:
        """
        lambdas/fwhms: (C,)
        返回 e: (C, emb_dim)
        """
        # 数值尺度简单归一化到 ~[0,1] 区间，提升稳定性（按可见-近红外常见范围）
        lam = (lambdas - 0.4) / 2.1         # 0.4~2.5 μm 粗归一化
        fwh = torch.clamp(fwhms, min=1e-6)
        ratio = lam / fwh

        z = torch.cat([
            self._fourier_feats(lam, self.L),
            self._fourier_feats(ratio, self.L),
            fwh.unsqueeze(-1)
        ], dim=-1)  # (C, 4L+1)

        e = self.mlp(z)  # (C, emb_dim)
        return e


# ------------------------------------------------------------
# 3) Band-FiLM：用通道嵌入生成 γ/β，对特征做仿射调制
# ------------------------------------------------------------
class BandFiLM(nn.Module):
    def __init__(self, emb_dim: int = 32):
        super().__init__()
        self.gamma_head = nn.Linear(emb_dim, 1)
        self.beta_head  = nn.Linear(emb_dim, 1)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)  原始或预处理后的影像张量
        e: (C, emb_dim)  通道嵌入
        return: (B, C, H, W)
        """
        C = x.size(1)
        assert e.size(0) == C, "嵌入通道数与输入通道数不一致"
        gamma = self.gamma_head(e).view(1, C, 1, 1) + 1.0
        beta  = self.beta_head(e).view(1, C, 1, 1)
        return gamma * x + beta


# ------------------------------------------------------------
# 4) 一个前端封装：把 GF-1 波段元数据 → 嵌入 → FiLM
# ------------------------------------------------------------
class GF1SpectralConditioner(nn.Module):
    """
    用法：
        cond = GF1SpectralConditioner(include_pan=False, emb_dim=32)
        x = cond(x)  # x:(B,C,H,W); C=4(不含PAN)或5(含PAN)
    """
    def __init__(self, include_pan: bool = False, emb_dim: int = 32, L: int = 6):
        super().__init__()
        self.include_pan = include_pan
        self.encoder = WavelengthEncoder(L=L, emb_dim=emb_dim)
        self.film    = BandFiLM(emb_dim=emb_dim)

        # 这里预先保存元数据张量（也可在 forward 里现算）
        lambdas, fwhms, names = build_gf1_band_tensors(include_pan=include_pan)
        # 注册为 buffer，自动跟随 device / dtype
        self.register_buffer("lambdas", lambdas, persistent=False)
        self.register_buffer("fwhms",   fwhms,   persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将元数据搬到 x.device
        lambdas = self.lambdas.to(device=x.device, dtype=torch.float32)
        fwhms   = self.fwhms.to(device=x.device,   dtype=torch.float32)

        e = self.encoder(lambdas, fwhms)        # (C, emb_dim)
        x = self.film(x, e)                     # (B, C, H, W)
        return x


# ------------------------------------------------------------
# 5) 简要示例
# ------------------------------------------------------------
if __name__ == "__main__":
    # 假设使用 4 通道（Blue/Green/Red/NIR），输入大小 (B=2, C=4, H=256, W=256)
    B, C, H, W = 2, 4, 256, 256
    x = torch.randn(B, C, H, W)

    cond = GF1SpectralConditioner(include_pan=False, emb_dim=32, L=6)
    y = cond(x)  # y 即已做通道条件化，可接后续 UNet/ResNet/Transformer 等主干
    print("Conditioned shape:", y.shape)


# ------------------------------------------------------------
# 进阶：由 SRF 自动计算中心波长与 FWHM（可替换上面的近似）
# ------------------------------------------------------------
import numpy as np

def center_and_fwhm_from_srf(wavelength_nm: np.ndarray, resp: np.ndarray):
    """
    输入:
      wavelength_nm: (L,) 波长（nm）
      resp: (L,)     SRF 响应（0~1）
    返回:
      center: 有效中心波长（μm）= ∫λ·SRF / ∫SRF
      fwhm:   半高宽（μm）：响应≥max(resp)/2 的 λ 区间宽度
    """
    wl_um = wavelength_nm.astype(np.float64) / 1000.0
    r = resp.astype(np.float64)
    r = np.clip(r, 0, None) + 1e-12

    center = (wl_um * r).sum() / r.sum()
    half = r.max() * 0.5
    above = wl_um[r >= half]
    fwhm  = (above.max() - above.min()) if above.size > 0 else (wl_um[-1] - wl_um[0]) * 0.5
    return float(center), float(fwhm)


# 用法（示例）：
# wl_nm = ...  # (L,)
# resp_blue = ...  # (L,)
# lam_blue, fwhm_blue = center_and_fwhm_from_srf(wl_nm, resp_blue)
# 把每个波段的 (lam, fwhm) 计算出来，替换 build_gf1_band_tensors 的结果即可。