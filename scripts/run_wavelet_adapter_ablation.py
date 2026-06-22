#!/usr/bin/env python3
"""Run deterministic WaveletAdapter ablations on multi-band TIFF samples.

This script is a Phase-0/Phase-1 smoke harness. It does not train CoastGPT and
does not claim downstream quality. It verifies that RGB baseline, direct
multi-band projection, DWT LL, and DWT LL+HF produce stable tensors and
comparable visual artifacts.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


_REPO_ROOT = Path(__file__).resolve().parents[1]
_WAVELET_PATH = _REPO_ROOT / "Models" / "wavelet_adapter.py"


def _load_wavelet_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("wavelet_adapter", _WAVELET_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wavelet_adapter = _load_wavelet_module()
WaveletAdapter = wavelet_adapter.WaveletAdapter
MultiBandDirectAdapter = wavelet_adapter.MultiBandDirectAdapter
project_multiband_to_rgb = wavelet_adapter.project_multiband_to_rgb
robust_normalize = wavelet_adapter.robust_normalize


SENSOR_CONFIGS: Dict[str, Dict[str, object]] = {
    "GF2": {
        "root": "/home/ma-user/work/Stage3Data/养殖区/GF2/Size_512/Image_Orig",
        "gsd": 0.8,
    },
    "GF1": {
        "root": "/home/ma-user/work/Stage3Data/养殖区/GF1/Size_128/Image_Orig",
        "gsd": 8.0,
    },
    "GF6": {
        "root": "/home/ma-user/work/Stage3Data/养殖区/GF6/Size_512/Image_Orig",
        "gsd": 2.0,
    },
}


def load_tif_as_chw(path: Path) -> torch.Tensor:
    """Load a TIFF as float32 [C,H,W], preserving all bands."""

    import tifffile

    arr = tifffile.imread(str(path))
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[-1] > 16:
        pass
    elif arr.ndim == 3:
        arr = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"unsupported TIFF shape {arr.shape} for {path}")

    arr = arr.astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    return torch.from_numpy(arr.copy())


OutputSize = Union[int, Sequence[int]]


def _output_hw(output_size: OutputSize) -> Tuple[int, int]:
    if isinstance(output_size, int):
        return output_size, output_size
    if len(output_size) != 2:
        raise ValueError(f"output_size must be int or pair, got {output_size}")
    return int(output_size[0]), int(output_size[1])


def _resize_rgb(x: torch.Tensor, output_size: OutputSize) -> torch.Tensor:
    hw = _output_hw(output_size)
    if x.shape[-2:] == hw:
        return x
    return F.interpolate(x, size=hw, mode="bilinear", align_corners=False).clamp(0.0, 1.0)


def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    finite = torch.isfinite(x)
    nan_ratio = 1.0 - float(finite.float().mean().item())
    safe = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "nan_ratio": nan_ratio,
        "min": float(safe.min().item()),
        "max": float(safe.max().item()),
        "mean": float(safe.mean().item()),
        "std": float(safe.std(unbiased=False).item()),
    }


def _save_tensor_png(x: torch.Tensor, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = x.detach().cpu().float().clamp(0.0, 1.0)
    if img.ndim == 4:
        img = img[0]
    if img.shape[0] != 3:
        raise ValueError(f"expected 3-channel tensor for PNG, got {tuple(img.shape)}")
    arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _rgb_baseline(x: torch.Tensor, output_size: OutputSize) -> Tuple[torch.Tensor, Dict[str, object]]:
    y = project_multiband_to_rgb(x)
    y = _resize_rgb(y, output_size)
    meta: Dict[str, object] = {
        "mode": "rgb_baseline",
        "levels": 0,
        "effective_gsd": None,
        "output_shape": list(y.shape),
    }
    return y, meta


def run_modes_on_tensor(
    x: torch.Tensor,
    sample_id: str,
    sensor: str,
    source_gsd: float,
    target_gsd: float,
    output_size: OutputSize,
    modes: Sequence[str],
    level_policy: str = "nearest",
    save_dir: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """Run configured modes on one batched tensor and return metric records."""

    records: List[Dict[str, object]] = []
    for mode in modes:
        started = time.perf_counter()
        if mode == "rgb_baseline":
            y, meta = _rgb_baseline(x, output_size)
        elif mode == "learnable_direct":
            x_for_adapter = robust_normalize(x)
            adapter = MultiBandDirectAdapter(
                in_channels=x_for_adapter.shape[1],
                target_channels=3,
                output_size=_output_hw(output_size),
            )
            y, meta = adapter(x_for_adapter)
        else:
            adapter = WaveletAdapter(
                mode=mode,
                source_gsd=source_gsd,
                target_gsd=target_gsd,
                level_policy=level_policy,
                output_size=_output_hw(output_size),
            )
            y, meta = adapter(x)
        latency_ms = (time.perf_counter() - started) * 1000.0

        stats = _tensor_stats(y)
        record: Dict[str, object] = {
            "sample_id": sample_id,
            "sensor": sensor,
            "adapter": mode,
            "source_gsd": source_gsd,
            "target_gsd": target_gsd,
            "level_policy": level_policy,
            "levels": meta.get("levels"),
            "effective_gsd": meta.get("effective_gsd"),
            "input_shape": list(x.shape),
            "output_shape": list(y.shape),
            "latency_ms": round(latency_ms, 4),
        }
        if "extra_band_weight_l1" in meta:
            record["extra_band_weight_l1"] = meta["extra_band_weight_l1"]
        if "uses_extra_bands" in meta:
            record["uses_extra_bands"] = meta["uses_extra_bands"]
        record.update(stats)
        records.append(record)

        if save_dir is not None:
            _save_tensor_png(y, save_dir / sensor / f"{sample_id}__{mode}.png")

    return records


def _iter_tifs(root: Path, limit: int, seed: int) -> Iterable[Path]:
    tifs = sorted([p for p in root.iterdir() if p.suffix.lower() in {".tif", ".tiff"}])
    rng = random.Random(seed)
    rng.shuffle(tifs)
    if limit > 0:
        tifs = tifs[:limit]
    return tifs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "output" / "wavelet_adapter_ablation"))
    parser.add_argument("--sensors", nargs="+", default=["GF2", "GF1", "GF6"], choices=sorted(SENSOR_CONFIGS))
    parser.add_argument("--modes", nargs="+", default=["rgb_baseline", "direct", "learnable_direct", "dwt_ll", "dwt_ll_hf"])
    parser.add_argument("--target-gsd", type=float, default=8.0)
    parser.add_argument("--level-policy", default="nearest", choices=["floor", "ceil", "nearest"])
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=2, help="samples per sensor; <=0 means all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    image_dir = output_dir / "images" if args.save_images else None

    all_records: List[Dict[str, object]] = []
    with metrics_path.open("w", encoding="utf-8") as f:
        for sensor in args.sensors:
            cfg = SENSOR_CONFIGS[sensor]
            root = Path(str(cfg["root"]))
            if not root.exists():
                print(f"[WARN] skip {sensor}: missing root {root}")
                continue
            for tif_path in _iter_tifs(root, args.limit, args.seed):
                sample_id = tif_path.stem
                tensor = load_tif_as_chw(tif_path).unsqueeze(0)
                records = run_modes_on_tensor(
                    tensor,
                    sample_id=sample_id,
                    sensor=sensor,
                    source_gsd=float(cfg["gsd"]),
                    target_gsd=args.target_gsd,
                    output_size=args.output_size,
                    modes=args.modes,
                    level_policy=args.level_policy,
                    save_dir=image_dir,
                )
                for record in records:
                    record["path"] = str(tif_path)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                all_records.extend(records)

    summary = {
        "num_records": len(all_records),
        "metrics_path": str(metrics_path),
        "output_dir": str(output_dir),
        "modes": list(args.modes),
        "sensors": list(args.sensors),
        "target_gsd": args.target_gsd,
        "level_policy": args.level_policy,
        "output_size": args.output_size,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
