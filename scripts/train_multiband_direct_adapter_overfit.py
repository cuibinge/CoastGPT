#!/usr/bin/env python3
"""Tiny overfit check for the learnable multi-band direct adapter.

This is a lightweight pre-integration experiment. It trains only a 1x1
MultiBandDirectAdapter to fit a deterministic target that explicitly depends on
the fourth band. Passing this check means the adapter path can learn to use NIR;
it is not a downstream task result.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
MultiBandDirectAdapter = wavelet_adapter.MultiBandDirectAdapter
robust_normalize = wavelet_adapter.robust_normalize
project_multiband_to_rgb = wavelet_adapter.project_multiband_to_rgb


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


def make_nir_teacher_target(x: torch.Tensor) -> torch.Tensor:
    """Return a 3-channel target that forces use of the fourth input band."""

    if x.ndim != 4:
        raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
    if x.shape[1] < 4:
        raise ValueError("NIR teacher target requires at least 4 channels")
    return torch.stack(
        [
            x[:, 3],
            0.75 * x[:, 3] + 0.25 * x[:, 1],
            x[:, 2],
        ],
        dim=1,
    )


def train_adapter_to_target(
    batch: torch.Tensor,
    steps: int,
    lr: float,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[MultiBandDirectAdapter, Dict[str, object]]:
    torch.manual_seed(seed)
    x = batch.to(device=device, dtype=torch.float32)
    target = make_nir_teacher_target(x)
    adapter = MultiBandDirectAdapter(in_channels=x.shape[1], target_channels=3, output_size=None).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=0.0)

    with torch.no_grad():
        initial_pred, _ = adapter(x)
        initial_loss = F.mse_loss(initial_pred, target).item()

    history: List[float] = []
    started = time.perf_counter()
    for _ in range(int(steps)):
        pred, _ = adapter(x)
        loss = F.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.detach().cpu().item()))

    with torch.no_grad():
        final_pred, meta = adapter(x)
        final_loss = F.mse_loss(final_pred, target).item()

    result: Dict[str, object] = {
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "loss_ratio": float(final_loss / max(initial_loss, 1e-12)),
        "steps": int(steps),
        "lr": float(lr),
        "device": device,
        "input_shape": list(batch.shape),
        "target_shape": list(target.shape),
        "elapsed_sec": round(time.perf_counter() - started, 4),
        "extra_band_weight_l1": meta["extra_band_weight_l1"],
        "loss_history_first": history[:5],
        "loss_history_last": history[-5:],
    }
    return adapter, result


def run_synthetic_overfit(
    seed: int = 42,
    steps: int = 120,
    batch_size: int = 8,
    image_size: int = 8,
    lr: float = 0.08,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    x = torch.rand(batch_size, 4, image_size, image_size, dtype=torch.float32)
    _, result = train_adapter_to_target(x, steps=steps, lr=lr, device="cpu", seed=seed)
    result["source"] = "synthetic"
    return result


def load_tif_as_chw(path: Path) -> torch.Tensor:
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


def iter_sensor_tifs(sensor: str, limit: int, seed: int) -> Iterable[Path]:
    cfg = SENSOR_CONFIGS[sensor]
    root = Path(str(cfg["root"]))
    if not root.exists():
        raise FileNotFoundError(f"missing sensor root: {root}")
    tifs = sorted([p for p in root.iterdir() if p.suffix.lower() in {".tif", ".tiff"}])
    rng = random.Random(seed)
    rng.shuffle(tifs)
    if limit > 0:
        tifs = tifs[:limit]
    return tifs


def load_sensor_batch(sensor: str, limit: int, image_size: int, seed: int) -> Tuple[torch.Tensor, List[str]]:
    tensors: List[torch.Tensor] = []
    paths: List[str] = []
    for path in iter_sensor_tifs(sensor, limit=limit, seed=seed):
        tensor = load_tif_as_chw(path)
        if tensor.shape[0] < 4:
            continue
        tensor = robust_normalize(tensor[:4])
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        tensors.append(tensor)
        paths.append(str(path))
    if not tensors:
        raise RuntimeError(f"no >=4-band TIFF samples found for {sensor}")
    return torch.stack(tensors, dim=0), paths


def save_overfit_preview(adapter: MultiBandDirectAdapter, batch: torch.Tensor, out_path: Path) -> None:
    from PIL import Image, ImageDraw

    adapter_cpu = adapter.to("cpu").eval()
    x = batch[:1].cpu()
    with torch.no_grad():
        initial_rgb = project_multiband_to_rgb(x)
        target = make_nir_teacher_target(x).clamp(0.0, 1.0)
        pred, _ = adapter_cpu(x)
        pred = pred.clamp(0.0, 1.0)

    panels = [
        ("rgb_init", initial_rgb[0]),
        ("nir_teacher", target[0]),
        ("adapter_pred", pred[0]),
    ]
    thumb = 192
    label_h = 26
    canvas = Image.new("RGB", (thumb * len(panels), thumb + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, tensor) in enumerate(panels):
        arr = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB").resize((thumb, thumb))
        x0 = idx * thumb
        draw.text((x0 + 6, 7), label, fill=(0, 0, 0))
        canvas.paste(img, (x0, label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", default="GF2", choices=sorted(SENSOR_CONFIGS))
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "output" / "multiband_direct_overfit"))
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        result = run_synthetic_overfit(
            seed=args.seed,
            steps=args.steps,
            batch_size=args.limit,
            image_size=args.image_size,
            lr=args.lr,
        )
        paths: List[str] = []
    else:
        batch, paths = load_sensor_batch(args.sensor, args.limit, args.image_size, args.seed)
        adapter, result = train_adapter_to_target(
            batch,
            steps=args.steps,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
        )
        result["source"] = "sensor"
        result["sensor"] = args.sensor
        result["sample_paths"] = paths
        save_overfit_preview(adapter, batch, output_dir / f"{args.sensor}_preview.png")

    result["output_dir"] = str(output_dir)
    metrics_path = output_dir / ("synthetic_metrics.json" if args.synthetic else f"{args.sensor}_metrics.json")
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"metrics_path={metrics_path}")


if __name__ == "__main__":
    main()
