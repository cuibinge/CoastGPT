import importlib.util
from pathlib import Path

import numpy as np
import torch


_MODULE_PATH = Path(__file__).resolve().parents[1] / "Dataset" / "multiband_source.py"
_SPEC = importlib.util.spec_from_file_location("multiband_source", _MODULE_PATH)
multiband_source = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(multiband_source)


def test_load_multiband_tensor_preserves_four_tiff_bands_and_resizes(tmp_path):
    import tifffile

    tif_path = tmp_path / "sample.tif"
    arr = np.arange(7 * 9 * 4, dtype=np.float32).reshape(7, 9, 4)
    tifffile.imwrite(tif_path, arr)

    tensor = multiband_source.load_multiband_tensor(tif_path, output_size=(8, 10), max_channels=4)

    assert tensor.shape == (4, 8, 10)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert tensor.max() > 1.0


def test_load_multiband_tensor_returns_none_for_non_tiff(tmp_path):
    png_path = tmp_path / "sample.png"
    png_path.write_bytes(b"not actually an image")

    assert multiband_source.load_multiband_tensor(png_path, output_size=(8, 8)) is None


def test_stack_optional_multiband_replaces_missing_with_zeros():
    first = torch.ones(4, 8, 8)
    second = None

    stacked, valid = multiband_source.stack_optional_multiband([first, second])

    assert stacked.shape == (2, 4, 8, 8)
    assert torch.equal(valid, torch.tensor([True, False]))
    assert stacked[0].sum() > 0
    assert stacked[1].sum() == 0
