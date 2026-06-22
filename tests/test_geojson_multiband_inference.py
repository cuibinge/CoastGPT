from pathlib import Path

import ml_collections
import numpy as np
import torch


def test_build_multiband_tensor_for_generation_can_zero_extra_band(tmp_path):
    import tifffile
    from Tools import run_geojson_batch_eval

    tif_path = tmp_path / "sample.tif"
    arr = np.zeros((5, 6, 4), dtype=np.float32)
    arr[..., 0] = 1.0
    arr[..., 1] = 2.0
    arr[..., 2] = 3.0
    arr[..., 3] = 9.0
    tifffile.imwrite(tif_path, arr)

    config = ml_collections.ConfigDict(
        {
            "multiband_inference_mode": "zero-extra",
            "multiband_max_channels": 4,
        }
    )
    image_tensor = torch.zeros(1, 3, 2, 2)

    multiband, valid = run_geojson_batch_eval._build_multiband_tensor_for_generation(
        config=config,
        image_path=tif_path,
        image_tensor=image_tensor,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert valid.tolist() == [True]
    assert multiband.shape == (1, 4, 5, 6)
    assert torch.all(multiband[:, 0] == 1.0)
    assert torch.all(multiband[:, 1] == 2.0)
    assert torch.all(multiband[:, 2] == 3.0)
    assert torch.all(multiband[:, 3] == 0.0)


def test_build_multiband_tensor_for_generation_off_mode_returns_none(tmp_path):
    from Tools import run_geojson_batch_eval

    config = ml_collections.ConfigDict({"multiband_inference_mode": "off"})
    multiband, valid = run_geojson_batch_eval._build_multiband_tensor_for_generation(
        config=config,
        image_path=Path(tmp_path / "sample.png"),
        image_tensor=torch.zeros(1, 3, 2, 2),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert multiband is None
    assert valid is None


def test_build_multiband_tensor_for_generation_skips_tiff_with_too_few_channels(tmp_path):
    import tifffile
    from Tools import run_geojson_batch_eval

    tif_path = tmp_path / "rgb.tif"
    tifffile.imwrite(tif_path, np.zeros((5, 6, 3), dtype=np.float32))
    config = ml_collections.ConfigDict(
        {
            "multiband_inference_mode": "auto",
            "multiband_max_channels": 4,
            "wavelet_adapter": {"in_channels": 4},
        }
    )

    multiband, valid = run_geojson_batch_eval._build_multiband_tensor_for_generation(
        config=config,
        image_path=tif_path,
        image_tensor=torch.zeros(1, 3, 2, 2),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert multiband is None
    assert valid is None


def test_ensure_image_token_does_not_duplicate_existing_token():
    from Tools import run_geojson_batch_eval

    prompt = "<image>\n[DET] detect targets"
    result = run_geojson_batch_eval._ensure_image_token(
        prompt,
        tune_im_start=False,
        default_image_token="<image>",
        default_im_start_token="<im_start>",
        default_im_end_token="<im_end>",
    )

    assert result == prompt
    assert result.count("<image>") == 1


def test_ensure_image_token_adds_plain_token_when_missing():
    from Tools import run_geojson_batch_eval

    result = run_geojson_batch_eval._ensure_image_token(
        "[DET] detect targets",
        tune_im_start=False,
        default_image_token="<image>",
        default_im_start_token="<im_start>",
        default_im_end_token="<im_end>",
    )

    assert result == "<image>\n[DET] detect targets"


def test_ensure_image_token_adds_wrapped_token_when_tuned():
    from Tools import run_geojson_batch_eval

    result = run_geojson_batch_eval._ensure_image_token(
        "[DET] detect targets",
        tune_im_start=True,
        default_image_token="<image>",
        default_im_start_token="<im_start>",
        default_im_end_token="<im_end>",
    )

    assert result == "<im_start><image><im_end>\n[DET] detect targets"


def test_infer_decode_prompt_len_when_output_contains_prompt_prefix():
    from Tools import run_geojson_batch_eval

    output_ids = torch.tensor([[11, 22, 33, 44]])
    input_ids = torch.tensor([[11, 22]])

    assert run_geojson_batch_eval._infer_decode_prompt_len(output_ids, input_ids) == 2


def test_infer_decode_prompt_len_when_output_is_generated_only():
    from Tools import run_geojson_batch_eval

    output_ids = torch.tensor([[33, 44, 55]])
    input_ids = torch.tensor([[11, 22]])

    assert run_geojson_batch_eval._infer_decode_prompt_len(output_ids, input_ids) == 0
