#!/usr/bin/env python3
"""
DWT 异构遥感数据统一 — 离线验证脚本

验证要点:
  1. DWT 多级分解对分辨率对齐的效果 (0.8m GF2 vs 8m GF1)
  2. LL 子带跨传感器光谱融合 (不同波段数 → 统一 3 通道)
  3. IDWT 重建后与原始 RGB 的视觉对比

用法:
  python scripts/verify_dwt_fusion.py
"""

import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")
import logging
logging.getLogger("tifffile").setLevel(logging.ERROR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pywt

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 1. 数据加载
# ---------------------------------------------------------------------------

SENSOR_CONFIGS = {
    "GF2": {
        "root": "/home/ma-user/work/Stage3Data/养殖区/GF2/Size_512/Image_Orig",
        "gsd_m": 0.8,
        "bands": ["Blue", "Green", "Red", "NIR"],
        "dtype_norm": "per_band_minmax",
    },
    "GF1": {
        "root": "/home/ma-user/work/Stage3Data/养殖区/GF1/Size_128/Image_Orig",
        "gsd_m": 8.0,
        "bands": ["Blue", "Green", "Red", "NIR"],
        "dtype_norm": "per_band_minmax",
    },
    "GF6": {
        "root": "/home/ma-user/work/Stage3Data/养殖区/GF6/Size_512/Image_Orig",
        "gsd_m": 2.0,
        "bands": ["Blue", "Green", "Red", "NIR"],
        "dtype_norm": "per_band_minmax",
    },
}


def sample_one_tif(sensor: str) -> Tuple[str, np.ndarray]:
    """从指定传感器的 Image_Orig 中随机取一张 TIF, 归一化到 [0,1]."""
    import tifffile

    cfg = SENSOR_CONFIGS[sensor]
    root = cfg["root"]
    tifs = [f for f in os.listdir(root) if f.endswith(".tif") and not f.startswith(".")]
    if not tifs:
        raise RuntimeError(f"No TIFs in {root}")
    path = os.path.join(root, np.random.RandomState(42).choice(tifs))
    name = os.path.basename(path)

    arr = tifffile.imread(path).astype(np.float32)
    arr = np.clip(arr, 0, None)  # remove nodata sentinels

    h, w, c = arr.shape
    # per-band min-max norm
    for b in range(c):
        bmin, bmax = arr[:, :, b].min(), arr[:, :, b].max()
        if bmax > bmin + 1e-6:
            arr[:, :, b] = (arr[:, :, b] - bmin) / (bmax - bmin)
        else:
            arr[:, :, b] = 0.0

    return name, arr


# ---------------------------------------------------------------------------
# 2. DWT 工具
# ---------------------------------------------------------------------------


def dwt2_tensor(x: np.ndarray, wavelet: str = "haar") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2D DWT per channel: x[C, H, W] → LL, LH, HL, HH 各 [C, H/2, W/2]."""
    c, h, w = x.shape
    ll = np.zeros((c, h // 2, w // 2), dtype=np.float32)
    lh = np.zeros_like(ll)
    hl = np.zeros_like(ll)
    hh = np.zeros_like(ll)
    for i in range(c):
        coeffs = pywt.dwt2(x[i], wavelet, mode="symmetric")
        ll[i] = coeffs[0]
        lh[i], hl[i], hh[i] = coeffs[1]
    return ll, lh, hl, hh


def idwt2_tensor(ll: np.ndarray, lh: np.ndarray, hl: np.ndarray, hh: np.ndarray, wavelet: str = "haar") -> np.ndarray:
    """Inverse 2D DWT per channel: LL+LH+HL+HH → [C, H, W]."""
    c, h_half, w_half = ll.shape
    result = np.zeros((c, h_half * 2, w_half * 2), dtype=np.float32)
    for i in range(c):
        result[i] = pywt.idwt2((ll[i], (lh[i], hl[i], hh[i])), wavelet, mode="symmetric")
    return result


def compute_levels(source_gsd: float, target_gsd: float = 8.0) -> int:
    """计算对齐到 target_gsd 需要的 DWT 级数."""
    if source_gsd >= target_gsd:
        return 0
    levels = 0
    while source_gsd * (2 ** levels) < target_gsd:
        levels += 1
    return levels


def pad_to_even(x: np.ndarray, factor: int = 1) -> Tuple[np.ndarray, Tuple[int, int]]:
    """填充使 H, W 为 2^factor 的倍数, 返回 (padded, (orig_h, orig_w))."""
    _, h, w = x.shape
    div = 2 ** factor
    pad_h = (div - h % div) % div
    pad_w = (div - w % div) % div
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    padded = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    return padded, (h, w)


def crop_back(x: np.ndarray, orig_size: Tuple[int, int]) -> np.ndarray:
    """裁切回原始尺寸."""
    orig_h, orig_w = orig_size
    return x[:, :orig_h, :orig_w] if x.shape[1] != orig_h or x.shape[2] != orig_w else x


def multi_level_dwt(x: np.ndarray, levels: int, wavelet: str = "haar") -> Tuple[np.ndarray, List[dict], Tuple[int, int]]:
    """多级 DWT: x[C, H, W] → x_ll + 高频 + 原始尺寸."""
    high_freqs = []
    current, orig_size = pad_to_even(x, levels)
    for lv in range(levels):
        ll, lh, hl, hh = dwt2_tensor(current, wavelet)
        high_freqs.append({"level": lv, "lh": lh, "hl": hl, "hh": hh, "shape": current.shape})
        current = ll
    return current, high_freqs, orig_size


def multi_level_idwt(ll: np.ndarray, high_freqs: List[dict], wavelet: str = "haar") -> np.ndarray:
    """多级 IDWT: 从最终 LL + 高频列表恢复原始分辨率."""
    current = ll.copy()
    for hf in reversed(high_freqs):
        current = idwt2_tensor(current, hf["lh"], hf["hl"], hf["hh"], wavelet)
    return current


# ---------------------------------------------------------------------------
# 3. 跨传感器光谱融合 (简化版: query-based attention)
# ---------------------------------------------------------------------------


def spectral_fusion_ll(ll_list: List[np.ndarray], names: List[str], target_ch: int = 3) -> np.ndarray:
    """
    将多个传感器的 LL 子带融合为统一的 target_ch 通道图像.

    当前简化实现: 对每个 LL 取其前 3 个波段 (RGB), 按均值融合.
    完整版: 可学习 cross-attention query → target_ch 通道.
    """
    rgbs = []
    for ll, name in zip(ll_list, names):
        c = ll.shape[0]
        if c >= 3:
            rgb = ll[:3]  # B, G, R
        elif c == 2:
            # SAR: VV/VH → 复制填充为 3 通道
            rgb = np.stack([ll[0], ll[1], (ll[0] + ll[1]) * 0.5], axis=0)
        else:
            rgb = np.tile(ll[0:1], (3, 1, 1))
        rgbs.append(rgb)

    # 对齐空间尺寸 → 取最小尺寸
    target_h = min(r.shape[1] for r in rgbs)
    target_w = min(r.shape[2] for r in rgbs)
    aligned = []
    for r in rgbs:
        if r.shape[1] != target_h or r.shape[2] != target_w:
            from scipy.ndimage import zoom
            factors = (1, target_h / r.shape[1], target_w / r.shape[2])
            aligned.append(zoom(r, factors, order=1))
        else:
            aligned.append(r)

    # 简单均值融合
    fused = np.mean(aligned, axis=0)  # [3, H, W]
    return np.clip(fused, 0, 1)


# ---------------------------------------------------------------------------
# 4. 可视化
# ---------------------------------------------------------------------------


def to_display(arr_3ch: np.ndarray) -> np.ndarray:
    """CHW → HWC uint8 for matplotlib."""
    img = arr_3ch.transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def plot_results(
    original: np.ndarray,
    fused: np.ndarray,
    sensor_name: str,
    gsd: float,
    levels: int,
    high_freqs: List[dict],
    out_path: str,
):
    """对比可视化: 原始 RGB | LL | 融合结果 | 高频 + 差值."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Row 0: 原始 → DWT 分解
    orig_disp = to_display(original[:3])
    axes[0, 0].imshow(orig_disp)
    axes[0, 0].set_title(f"Original {sensor_name}\n({original.shape[0]} bands, {gsd}m, {original.shape[1]}×{original.shape[2]})", fontsize=9)

    # LL after DWT
    show_levels = min(levels, 3)
    ll_show, _ = pad_to_even(original[:3].copy(), show_levels)
    for _ in range(show_levels):
        ll_show, _, _, _ = dwt2_tensor(ll_show)
    axes[0, 1].imshow(to_display(ll_show))
    axes[0, 1].set_title(f"DWT×{show_levels} LL\n({ll_show.shape[1]}×{ll_show.shape[2]})", fontsize=9)

    # HF: concat LH|HL|HH of last level
    if high_freqs and levels > 0:
        hf = high_freqs[-1]
        hf_cat = []
        for key in ["lh", "hl", "hh"]:
            hf_d = hf[key][:3] if hf[key].shape[0] >= 3 else np.tile(hf[key][0:1], (3, 1, 1))
            hf_n = (hf_d - hf_d.min()) / (hf_d.max() - hf_d.min() + 1e-6)
            hf_cat.append(hf_n)
        axes[0, 2].imshow(to_display(np.concatenate(hf_cat, axis=2)))
        axes[0, 2].set_title(f"HF (LH|HL|HL) Lv{levels}", fontsize=9)
    else:
        axes[0, 2].axis("off")

    # Row 1: Fused result
    fused_disp = to_display(fused)
    axes[1, 0].imshow(fused_disp)
    axes[1, 0].set_title(f"DWT Fused (3ch)\n({fused.shape[1]}×{fused.shape[2]})", fontsize=9)

    # Diff
    min_h = min(orig_disp.shape[0], fused_disp.shape[0])
    min_w = min(orig_disp.shape[1], fused_disp.shape[1])
    diff = np.abs(orig_disp[:min_h, :min_w].astype(np.float32) -
                  fused_disp[:min_h, :min_w].astype(np.float32))
    diff = np.clip(diff * 3, 0, 255).astype(np.uint8)
    axes[1, 1].imshow(diff)
    axes[1, 1].set_title("Diff (×3)", fontsize=9)

    # Fused DWT LL|LH|HL|HH
    fused_pad, _ = pad_to_even(fused, 1)
    l_rgb, lh_r, hl_r, hh_r = dwt2_tensor(fused_pad)
    show_cat = np.concatenate([l_rgb, lh_r[:3] * 3, hl_r[:3] * 3, hh_r[:3] * 3], axis=2)
    axes[1, 2].imshow(to_display(np.clip(show_cat, 0, 1)))
    axes[1, 2].set_title("Fused DWT (LL|LH|HL|HH)", fontsize=9)

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle(f"DWT Fusion: {sensor_name} @ {gsd}m → {levels}× DWT → unified space",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "output/dwt_verify"))
    parser.add_argument("--target-gsd", type=float, default=8.0)
    parser.add_argument("--wavelet", default="haar")
    parser.add_argument("--sensors", nargs="+", default=["GF2", "GF1", "GF6"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("DWT Multi-Sensor Fusion Verification")
    print(f"Target GSD: {args.target_gsd}m, Wavelet: {args.wavelet}")
    print("=" * 70)

    # === 逐个传感器分析 ===
    all_lls = []
    all_names = []

    for sensor in args.sensors:
        cfg = SENSOR_CONFIGS[sensor]
        name, arr = sample_one_tif(sensor)
        h, w, c = arr.shape   # tifffile returns [H, W, C]
        arr_chw = arr.transpose(2, 0, 1)  # → [C, H, W]

        levels = compute_levels(cfg["gsd_m"], args.target_gsd)

        print(f"\n--- {sensor}: {name} ---")
        print(f"  Shape: {c}ch × {h}×{w}, GSD: {cfg['gsd_m']}m")
        print(f"  DWT levels to {args.target_gsd}m: {levels}")
        print(f"  Effective GSD after DWT: {cfg['gsd_m'] * (2**levels):.1f}m")

        # DWT 分解
        ll, high_freqs, orig_size = multi_level_dwt(arr_chw, levels, args.wavelet)
        print(f"  Final LL shape: {ll.shape} ({ll.shape[1]}×{ll.shape[2]})")

        # 存储 LL 用于融合
        all_lls.append(ll)
        all_names.append(sensor)

        # IDWT 重建（验证可逆性）
        reconstructed_full = multi_level_idwt(ll, high_freqs, args.wavelet)
        reconstructed = crop_back(reconstructed_full, orig_size)
        arr_cropped = arr_chw[:, :orig_size[0], :orig_size[1]]
        rec_error = np.mean((arr_cropped - reconstructed) ** 2)
        print(f"  Reconstruction MSE: {rec_error:.8f}")

        # 对每个传感器单独出图
        fused_single = spectral_fusion_ll([ll], [sensor])
        arr_for_plot = arr_chw[:, :orig_size[0], :orig_size[1]]
        plot_results(arr_for_plot, fused_single, sensor, cfg["gsd_m"], levels, high_freqs,
                     os.path.join(args.output_dir, f"dwt_{sensor}.png"))

    # === 跨传感器融合 ===
    if len(all_lls) >= 2:
        print(f"\n--- Cross-Sensor Fusion ---")
        print(f"  LL shapes: {[(n, list(ll.shape)) for n, ll in zip(all_names, all_lls)]}")

        fused_multi = spectral_fusion_ll(all_lls, all_names)

        fig, axes = plt.subplots(1, len(all_lls) + 1, figsize=(4 * (len(all_lls) + 1), 4))
        for i, (ll, name) in enumerate(zip(all_lls, all_names)):
            rgb = ll[:3] if ll.shape[0] >= 3 else np.tile(ll[0:1], (3, 1, 1))
            axes[i].imshow(to_display(rgb))
            axes[i].set_title(f"{name} LL\n{list(ll.shape)}", fontsize=9)
            axes[i].axis("off")
        axes[-1].imshow(to_display(fused_multi))
        axes[-1].set_title(f"Fused (mean)\n{list(fused_multi.shape)}", fontsize=9)
        axes[-1].axis("off")
        plt.suptitle("Cross-Sensor DWT Fusion", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fusion_path = os.path.join(args.output_dir, "dwt_cross_sensor_fusion.png")
        plt.savefig(fusion_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fusion_path}")

    # === 多分辨率对齐演示 ===
    if len(args.sensors) >= 2:
        print(f"\n--- Multi-Resolution Alignment Demo ---")
        fig2, axes2 = plt.subplots(2, max(2, len(args.sensors)), figsize=(4 * max(2, len(args.sensors)), 8))
        for i, sensor in enumerate(args.sensors):
            cfg = SENSOR_CONFIGS[sensor]
            name, arr = sample_one_tif(sensor)
            arr_chw = arr.transpose(2, 0, 1)
            rgb = arr_chw[:3] if arr_chw.shape[0] >= 3 else np.tile(arr_chw[0:1], (3, 1, 1))

            axes2[0, i].imshow(to_display(rgb))
            axes2[0, i].set_title(f"{sensor} @ {cfg['gsd_m']}m\n{rgb.shape[1]}×{rgb.shape[2]}", fontsize=9)
            axes2[0, i].axis("off")

            # DWT aligned
            lv = compute_levels(cfg["gsd_m"], args.target_gsd)
            ll, _, _ = multi_level_dwt(arr_chw, lv)
            ll_rgb = ll[:3] if ll.shape[0] >= 3 else np.tile(ll[0:1], (3, 1, 1))
            axes2[1, i].imshow(to_display(ll_rgb))
            axes2[1, i].set_title(f"DWT×{lv} → ~{cfg['gsd_m']*(2**lv):.1f}m\n{ll_rgb.shape[1]}×{ll_rgb.shape[2]}", fontsize=9)
            axes2[1, i].axis("off")

        # Hide unused subplots
        for j in range(len(args.sensors), axes2.shape[1]):
            axes2[0, j].axis("off")
            axes2[1, j].axis("off")

        plt.suptitle(f"Resolution Alignment via DWT → Target @ {args.target_gsd}m", fontsize=12, fontweight="bold")
        plt.tight_layout()
        align_path = os.path.join(args.output_dir, "dwt_resolution_alignment.png")
        plt.savefig(align_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {align_path}")

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
