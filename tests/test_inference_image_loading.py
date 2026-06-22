import importlib.util
from pathlib import Path

from PIL import Image


def _load_inference_module():
    script_path = Path(__file__).resolve().parents[1] / "Inference.py"
    spec = importlib.util.spec_from_file_location("inference_module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resize_large_image_for_inference_caps_pixels_and_keeps_rgb():
    module = _load_inference_module()
    image = Image.new("RGB", (400, 300), color=(1, 2, 3))

    resized = module._resize_large_image_for_inference(
        image,
        max_pixels=10_000,
        max_side=128,
    )

    assert resized.mode == "RGB"
    assert resized.width * resized.height <= 10_000
    assert max(resized.size) <= 128
    assert resized.size[0] < image.size[0]
