import json
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aqua_poc_dataset_direct",
    ROOT / "Dataset" / "aqua_poc_dataset.py",
)
aqua_poc_dataset = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(aqua_poc_dataset)
AquaPoCDataset = aqua_poc_dataset.AquaPoCDataset


def _write_sample(tmp_path, width=96, height=64):
    data_root = tmp_path / "data"
    data_root.mkdir()
    image_path = data_root / "tile.png"
    mask_path = data_root / "mask.png"
    label_path = data_root / "label.geojson"

    Image.fromarray(np.full((height, width, 3), 128, dtype=np.uint8)).save(image_path)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[10:30, 20:50] = 255
    Image.fromarray(mask).save(mask_path)
    label_path.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

    manifest = [
        {
            "image_path": image_path.name,
            "label_path": label_path.name,
            "binary_label_path": mask_path.name,
            "original_size": [width, height],
            "original_transform": [0, 1, 0, 0, 0, -1],
            "source_crs": "EPSG:4326",
            "sample_id": "sample_0",
        }
    ]
    manifest_path = data_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return data_root, manifest_path


def test_aqua_dataset_resizes_to_fixed_input_size(tmp_path):
    data_root, manifest_path = _write_sample(tmp_path)
    dataset = AquaPoCDataset(
        manifest_path=str(manifest_path),
        data_root=str(data_root),
        image_size=128,
    )

    sample = dataset[0]

    assert tuple(sample["image"].shape[-2:]) == (128, 128)
    assert tuple(sample["target"]["masks"].shape[-2:]) == (128, 128)
    assert sample["meta"]["model_input_size"] == [128, 128]


def test_aqua_dataset_uses_native_size_when_image_size_zero(tmp_path):
    data_root, manifest_path = _write_sample(tmp_path, width=96, height=64)
    dataset = AquaPoCDataset(
        manifest_path=str(manifest_path),
        data_root=str(data_root),
        image_size=0,
    )

    sample = dataset[0]

    assert tuple(sample["image"].shape[-2:]) == (64, 96)
    assert tuple(sample["target"]["masks"].shape[-2:]) == (64, 96)
    assert sample["meta"]["model_input_size"] == [96, 64]
