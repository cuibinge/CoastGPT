"""Utilities for semantic validation overlay export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image


IGNORE_COLOR = np.array([64, 64, 64], dtype=np.uint8)
CLASS_COLORS = np.array(
    [
        [0, 0, 0],
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
        [174, 199, 232],
        [255, 187, 120],
        [152, 223, 138],
        [197, 176, 213],
        [196, 156, 148],
        [247, 182, 210],
        [199, 199, 199],
        [219, 219, 141],
        [158, 218, 229],
        [255, 152, 150],
        [255, 255, 51],
        [158, 202, 225],
        [107, 174, 214],
        [66, 146, 198],
        [33, 113, 181],
    ],
    dtype=np.uint8,
)


OverlayKey = Tuple[Tuple[int, ...], Tuple[str, ...]]


def overlay_group_key(meta: dict) -> OverlayKey:
    """Group overlays by tile size and known class set."""
    size = tuple(int(v) for v in meta.get("original_size", []) or [])
    classes = tuple(sorted(str(v) for v in meta.get("known_classes", []) or []))
    return size, classes


def select_diverse_overlay_indices(
    metas: Sequence[dict],
    seen_keys: Set[OverlayKey],
    saved_sample_ids: Set[str],
    max_new: int,
) -> List[int]:
    """Select a deterministic, diverse subset from one validation batch."""
    if max_new <= 0:
        return []

    selected: List[int] = []
    batch_keys: Set[OverlayKey] = set()

    for idx, meta in enumerate(metas):
        if len(selected) >= max_new:
            break
        sample_id = str(meta.get("sample_id", f"sample_{idx}"))
        key = overlay_group_key(meta)
        if sample_id in saved_sample_ids:
            continue
        if key in seen_keys or key in batch_keys:
            continue
        selected.append(idx)
        batch_keys.add(key)

    for idx, meta in enumerate(metas):
        if len(selected) >= max_new:
            break
        sample_id = str(meta.get("sample_id", f"sample_{idx}"))
        if idx in selected or sample_id in saved_sample_ids:
            continue
        selected.append(idx)

    return selected


def colorize_mask(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """Convert a train-id mask to RGB colors."""
    mask = np.asarray(mask)
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(CLASS_COLORS):
        colored[mask == class_id] = color
    colored[mask == ignore_index] = IGNORE_COLOR
    return colored


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an HWC image in uint8 or [0, 1] float format to uint8."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def build_overlay_metadata(
    meta: dict,
    target: np.ndarray,
    pred: np.ndarray,
    *,
    epoch: int,
    batch_idx: int,
    batch_position: int,
    overlay_filename: str,
) -> Dict[str, object]:
    """Build source metadata for one semantic overlay image."""
    return {
        "sample_id": str(meta.get("sample_id", f"s_{batch_idx}_{batch_position}")),
        "overlay_filename": overlay_filename,
        "panel_layout": ["image", "ground_truth", "prediction"],
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "batch_position": int(batch_position),
        "image_path": meta.get("image_path"),
        "known_classes": _json_list(meta.get("known_classes", [])),
        "train_ids": _json_list(meta.get("train_ids", [])),
        "original_size": _json_list(meta.get("original_size", [])),
        "model_input_size": _json_list(meta.get("model_input_size", [])),
        "sensor": meta.get("sensor"),
        "target_unique_train_ids": _unique_ints(target),
        "prediction_unique_train_ids": _unique_ints(pred),
    }


def save_semantic_overlays(
    images_np: np.ndarray,
    targets_np: np.ndarray,
    preds_np: np.ndarray,
    metas: Sequence[dict],
    *,
    epoch: int,
    batch_idx: int,
    output_dir: Path,
    max_total: int,
    state: Optional[MutableMapping[str, object]] = None,
) -> List[str]:
    """Save diverse semantic overlay PNGs and per-sample metadata files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = state if state is not None else {}
    seen_keys = state.setdefault("seen_keys", set())
    saved_sample_ids = state.setdefault("saved_sample_ids", set())
    saved_count = int(state.setdefault("saved_count", 0))
    remaining = max(0, max_total - saved_count)

    indices = select_diverse_overlay_indices(
        metas,
        seen_keys,  # type: ignore[arg-type]
        saved_sample_ids,  # type: ignore[arg-type]
        remaining,
    )
    saved: List[str] = []

    for idx in indices:
        meta = metas[idx]
        sample_id = str(meta.get("sample_id", f"s_{batch_idx}_{idx}"))
        filename_stem = _safe_filename(sample_id)
        overlay_filename = f"{filename_stem}.png"

        image = image_to_uint8(images_np[idx])
        target = targets_np[idx]
        pred = preds_np[idx]
        panel = np.hstack([image, colorize_mask(target), colorize_mask(pred)])
        Image.fromarray(panel).save(str(output_dir / overlay_filename))

        metadata = build_overlay_metadata(
            meta,
            target,
            pred,
            epoch=epoch,
            batch_idx=batch_idx,
            batch_position=idx,
            overlay_filename=overlay_filename,
        )
        metadata_path = output_dir / f"{filename_stem}.meta.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        seen_keys.add(overlay_group_key(meta))  # type: ignore[union-attr]
        saved_sample_ids.add(sample_id)  # type: ignore[union-attr]
        saved.append(sample_id)

    state["saved_count"] = saved_count + len(saved)
    return saved


def _unique_ints(values: np.ndarray) -> List[int]:
    return sorted(int(v) for v in np.unique(values))


def _json_list(values: Iterable[object]) -> List[object]:
    return [_json_scalar(v) for v in values]


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)
