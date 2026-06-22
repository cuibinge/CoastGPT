import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.semantic_overlay import (
    build_overlay_metadata,
    save_semantic_overlays,
    select_diverse_overlay_indices,
)


def test_select_diverse_overlay_indices_prefers_unseen_class_size_groups():
    metas = [
        {"sample_id": "a1", "original_size": [128, 128], "known_classes": ["沿海滩涂"]},
        {"sample_id": "a2", "original_size": [128, 128], "known_classes": ["沿海滩涂"]},
        {"sample_id": "b1", "original_size": [128, 128], "known_classes": ["盐田"]},
        {"sample_id": "c1", "original_size": [256, 256], "known_classes": ["沿海滩涂"]},
    ]
    seen_keys = {((128, 128), ("沿海滩涂",))}

    indices = select_diverse_overlay_indices(metas, seen_keys, set(), max_new=2)

    assert indices == [2, 3]


def test_build_overlay_metadata_keeps_source_and_unique_labels_json_safe():
    meta = {
        "sample_id": "022632_1cls",
        "image_path": "/data/tile.jpg",
        "known_classes": ["沿海滩涂"],
        "train_ids": [22],
        "original_size": [256, 256],
        "model_input_size": [224, 224],
        "sensor": "GF1",
    }
    target = np.array([[255, 22], [22, 22]], dtype=np.uint8)
    pred = np.array([[0, 22], [23, 22]], dtype=np.int64)

    result = build_overlay_metadata(
        meta,
        target,
        pred,
        epoch=5,
        batch_idx=0,
        batch_position=1,
        overlay_filename="022632_1cls.png",
    )

    assert result["sample_id"] == "022632_1cls"
    assert result["image_path"] == "/data/tile.jpg"
    assert result["panel_layout"] == ["image", "ground_truth", "prediction"]
    assert result["target_unique_train_ids"] == [22, 255]
    assert result["prediction_unique_train_ids"] == [0, 22, 23]
    json.dumps(result, ensure_ascii=False)


def test_save_semantic_overlays_writes_png_metadata_and_updates_state(tmp_path):
    images_np = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    images_np[0, :, :] = [255, 0, 0]
    images_np[1, :, :] = [0, 255, 0]
    targets_np = np.array(
        [
            [[22, 255], [22, 22]],
            [[23, 23], [255, 23]],
        ],
        dtype=np.uint8,
    )
    preds_np = np.array(
        [
            [[22, 22], [0, 22]],
            [[23, 0], [23, 23]],
        ],
        dtype=np.int64,
    )
    metas = [
        {
            "sample_id": "first",
            "image_path": "/data/first.jpg",
            "known_classes": ["沿海滩涂"],
            "original_size": [128, 128],
        },
        {
            "sample_id": "second",
            "image_path": "/data/second.jpg",
            "known_classes": ["盐田"],
            "original_size": [256, 256],
        },
    ]
    state = {"seen_keys": set(), "saved_sample_ids": set(), "saved_count": 0}

    saved = save_semantic_overlays(
        images_np,
        targets_np,
        preds_np,
        metas,
        epoch=5,
        batch_idx=0,
        output_dir=tmp_path,
        max_total=2,
        state=state,
    )

    assert saved == ["first", "second"]
    assert state["saved_count"] == 2
    assert (tmp_path / "first.png").exists()
    assert (tmp_path / "first.meta.json").exists()

    image = Image.open(tmp_path / "first.png")
    assert image.size == (6, 2)

    metadata = json.loads((tmp_path / "first.meta.json").read_text(encoding="utf-8"))
    assert metadata["image_path"] == "/data/first.jpg"
    assert metadata["overlay_filename"] == "first.png"
