import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import ml_collections
import torch


_MODULE_PATH = Path(__file__).resolve().parents[1] / "Dataset" / "build_transform.py"
_SPEC = importlib.util.spec_from_file_location("build_transform", _MODULE_PATH)
build_transform = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_transform)
build_vlp_transform = build_transform.build_vlp_transform


def test_deterministic_resize_train_transform_outputs_fixed_image_size():
    cfg = ml_collections.ConfigDict()
    cfg.rgb_vision = ml_collections.ConfigDict({"arch": "dual"})
    cfg.transform = ml_collections.ConfigDict(
        {
            "default_input_size": [16, 16],
            "rand_aug": "rand-m5-n2-mstd0.5-inc1",
            "deterministic_resize": True,
        }
    )

    transform = build_vlp_transform(cfg, is_train=True)
    image = Image.fromarray(np.arange(29 * 11 * 3, dtype=np.uint8).reshape(11, 29, 3), mode="RGB")
    output = transform(image)
    output_again = transform(image)

    assert "RandomResizedCrop" not in repr(transform)
    assert "RandAugment" not in repr(transform)
    assert tuple(output.shape) == (3, 16, 16)
    assert torch.equal(output, output_again)
    assert torch.isfinite(output).all()
