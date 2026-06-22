import importlib.util
from pathlib import Path

import torch


def _load_dual_vision_encoder_module():
    script_path = Path(__file__).resolve().parents[1] / "Models" / "dual_vision_encoder.py"
    spec = importlib.util.spec_from_file_location("dual_vision_encoder", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_interpolate_pos_embed_grid_preserves_low_precision_dtype():
    module = _load_dual_vision_encoder_module()
    patch_pe = torch.randn(1, 8, 4, 4, dtype=torch.bfloat16)

    resized = module._interpolate_pos_embed_grid(patch_pe, size=(8, 6))

    assert resized.shape == (1, 8, 8, 6)
    assert resized.dtype == torch.bfloat16
