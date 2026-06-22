import math
import importlib.util
from pathlib import Path

import numpy as np
import torch

_MODULE_PATH = Path(__file__).resolve().parents[1] / "Models" / "wavelet_adapter.py"
_SPEC = importlib.util.spec_from_file_location("wavelet_adapter", _MODULE_PATH)
wavelet_adapter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wavelet_adapter)

WaveletAdapter = wavelet_adapter.WaveletAdapter
MultiBandDirectAdapter = wavelet_adapter.MultiBandDirectAdapter
apply_multiband_adapter = wavelet_adapter.apply_multiband_adapter
compute_dwt_levels = wavelet_adapter.compute_dwt_levels
crop_to_hw = wavelet_adapter.crop_to_hw
multi_level_haar_dwt = wavelet_adapter.multi_level_haar_dwt
multi_level_haar_idwt = wavelet_adapter.multi_level_haar_idwt


def test_compute_dwt_levels_records_effective_gsd_for_non_power_ratio():
    levels, effective_gsd = compute_dwt_levels(0.8, 8.0, policy="nearest")
    assert levels == 3
    assert math.isclose(effective_gsd, 6.4, rel_tol=1e-6)

    levels, effective_gsd = compute_dwt_levels(0.8, 8.0, policy="ceil")
    assert levels == 4
    assert math.isclose(effective_gsd, 12.8, rel_tol=1e-6)

    levels, effective_gsd = compute_dwt_levels(8.0, 8.0, policy="ceil")
    assert levels == 0
    assert math.isclose(effective_gsd, 8.0, rel_tol=1e-6)


def test_multi_level_haar_dwt_reconstructs_odd_spatial_input_after_crop():
    x = torch.linspace(0.0, 1.0, steps=2 * 4 * 17 * 19, dtype=torch.float32).reshape(2, 4, 17, 19)

    ll, high_freqs, original_hw = multi_level_haar_dwt(x, levels=2)
    reconstructed = multi_level_haar_idwt(ll, high_freqs)
    cropped = crop_to_hw(reconstructed, original_hw)

    assert cropped.shape == x.shape
    assert torch.allclose(cropped, x, atol=1e-6)


def test_wavelet_adapter_modes_return_finite_three_channel_tensors():
    x = torch.rand(2, 4, 17, 19, dtype=torch.float32)

    for mode in ("direct", "dwt_ll", "dwt_ll_hf"):
        adapter = WaveletAdapter(
            mode=mode,
            source_gsd=0.8,
            target_gsd=8.0,
            level_policy="nearest",
            output_size=(16, 16),
        )

        y, meta = adapter(x)

        assert y.shape == (2, 3, 16, 16)
        assert torch.isfinite(y).all()
        assert meta["mode"] == mode
        assert meta["output_shape"] == [2, 3, 16, 16]
        assert "effective_gsd" in meta


def test_multiband_direct_adapter_rgb_initialization_and_shape():
    x = torch.rand(2, 4, 11, 13, dtype=torch.float32)
    adapter = MultiBandDirectAdapter(in_channels=4, target_channels=3, output_size=(16, 16))

    y, meta = adapter(x)

    expected = torch.nn.functional.interpolate(x[:, [2, 1, 0]], size=(16, 16), mode="bilinear", align_corners=False)
    assert y.shape == (2, 3, 16, 16)
    assert torch.allclose(y, expected, atol=1e-6)
    assert meta["adapter"] == "learnable_direct"
    assert meta["uses_extra_bands"] is True
    assert meta["input_shape"] == [2, 4, 11, 13]
    assert meta["output_shape"] == [2, 3, 16, 16]


def test_multiband_direct_adapter_casts_input_to_projection_dtype():
    x = torch.rand(1, 4, 5, 5, dtype=torch.float32)
    adapter = MultiBandDirectAdapter(in_channels=4, target_channels=3, output_size=None).to(dtype=torch.bfloat16)

    y, _ = adapter(x)

    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y.float()).all()


def test_multiband_direct_adapter_can_learn_nir_dependent_target():
    torch.manual_seed(7)
    x = torch.rand(8, 4, 6, 6, dtype=torch.float32)
    target = torch.stack(
        [
            x[:, 3],
            0.75 * x[:, 3] + 0.25 * x[:, 1],
            x[:, 2],
        ],
        dim=1,
    )
    adapter = MultiBandDirectAdapter(in_channels=4, target_channels=3, output_size=None)
    opt = torch.optim.AdamW(adapter.parameters(), lr=0.08, weight_decay=0.0)

    with torch.no_grad():
        initial_loss = torch.nn.functional.mse_loss(adapter(x)[0], target).item()

    for _ in range(120):
        pred, _ = adapter(x)
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    final_loss = torch.nn.functional.mse_loss(adapter(x)[0], target).item()
    assert final_loss < initial_loss * 0.05
    assert final_loss < 1e-3


def test_apply_multiband_adapter_normalizes_raw_values_and_matches_reference():
    x = torch.arange(2 * 4 * 9 * 11, dtype=torch.float32).reshape(2, 4, 9, 11) * 100.0
    reference = torch.zeros(2, 3, 16, 16, dtype=torch.float16)
    adapter = MultiBandDirectAdapter(in_channels=4, target_channels=3, output_size=None)

    y, meta = apply_multiband_adapter(adapter, x, reference_rgb=reference)

    assert y.shape == reference.shape
    assert y.dtype == reference.dtype
    assert torch.isfinite(y).all()
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0
    assert meta["adapter"] == "learnable_direct"


def test_apply_multiband_adapter_resizes_bfloat16_output_in_float32(monkeypatch):
    x = torch.arange(1 * 4 * 5 * 5, dtype=torch.float32).reshape(1, 4, 5, 5)
    reference = torch.zeros(1, 3, 8, 8, dtype=torch.bfloat16)
    adapter = MultiBandDirectAdapter(in_channels=4, target_channels=3, output_size=None).to(dtype=torch.bfloat16)
    seen_dtypes = []
    original_interpolate = wavelet_adapter.F.interpolate

    def checked_interpolate(input_tensor, *args, **kwargs):
        seen_dtypes.append(input_tensor.dtype)
        assert input_tensor.dtype == torch.float32
        return original_interpolate(input_tensor, *args, **kwargs)

    monkeypatch.setattr(wavelet_adapter.F, "interpolate", checked_interpolate)

    y, meta = apply_multiband_adapter(adapter, x, reference_rgb=reference)

    assert seen_dtypes == [torch.float32]
    assert y.shape == reference.shape
    assert y.dtype == torch.bfloat16
    assert meta["reference_shape"] == list(reference.shape)


def test_ablation_helpers_load_tif_and_collect_mode_metrics(tmp_path):
    import tifffile

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_wavelet_adapter_ablation.py"
    spec = importlib.util.spec_from_file_location("run_wavelet_adapter_ablation", script_path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    tif_path = tmp_path / "sample.tif"
    arr = np.arange(5 * 7 * 4, dtype=np.float32).reshape(5, 7, 4)
    tifffile.imwrite(tif_path, arr)

    tensor = script.load_tif_as_chw(tif_path)
    assert tensor.shape == (4, 5, 7)

    records = script.run_modes_on_tensor(
        tensor.unsqueeze(0),
        sample_id="sample",
        sensor="GF2",
        source_gsd=0.8,
        target_gsd=8.0,
        output_size=16,
        modes=("rgb_baseline", "direct", "learnable_direct", "dwt_ll"),
    )

    assert [record["adapter"] for record in records] == ["rgb_baseline", "direct", "learnable_direct", "dwt_ll"]
    assert all(record["output_shape"] == [1, 3, 16, 16] for record in records)
    assert all(record["nan_ratio"] == 0.0 for record in records)
    assert all(0.0 <= record["min"] <= record["max"] <= 1.0 for record in records)
    learnable = [record for record in records if record["adapter"] == "learnable_direct"][0]
    assert learnable["extra_band_weight_l1"] == 0.0


def test_multiband_overfit_script_synthetic_training_reduces_loss():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_multiband_direct_adapter_overfit.py"
    spec = importlib.util.spec_from_file_location("train_multiband_direct_adapter_overfit", script_path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    result = script.run_synthetic_overfit(seed=3, steps=80, batch_size=8, image_size=8, lr=0.08)

    assert result["initial_loss"] > 0.01
    assert result["final_loss"] < result["initial_loss"] * 0.1
    assert result["final_loss"] < 1e-3
    assert result["extra_band_weight_l1"] > 0.5


def _load_coastgpt_class_with_stubs(monkeypatch):
    import sys
    import types

    root = Path(__file__).resolve().parents[1]

    models_pkg = types.ModuleType("Models")
    models_pkg.__path__ = [str(root / "Models")]
    monkeypatch.setitem(sys.modules, "Models", models_pkg)

    deepspeed_mod = types.ModuleType("deepspeed")
    deepspeed_utils = types.ModuleType("deepspeed.utils")
    zero_to_fp32 = types.ModuleType("deepspeed.utils.zero_to_fp32")
    zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed_mod)
    monkeypatch.setitem(sys.modules, "deepspeed.utils", deepspeed_utils)
    monkeypatch.setitem(sys.modules, "deepspeed.utils.zero_to_fp32", zero_to_fp32)

    peft_mod = types.ModuleType("peft")
    peft_mod.PeftModel = object
    monkeypatch.setitem(sys.modules, "peft", peft_mod)

    wavelet_mod = types.ModuleType("Models.wavelet_adapter")
    wavelet_mod.MultiBandDirectAdapter = object
    wavelet_mod.apply_multiband_adapter = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "Models.wavelet_adapter", wavelet_mod)

    vision_mod = types.ModuleType("Models.vision_model")
    vision_mod.VisionModel = object
    dual_mod = types.ModuleType("Models.dual_vision_encoder")
    dual_mod.DualVisionEncoder = object
    lang_mod = types.ModuleType("Models.language_model")
    lang_mod.LanguageModel = object
    emb_mod = types.ModuleType("Models.embedding_model_r1")
    emb_mod.EmbeddingModel = object
    monkeypatch.setitem(sys.modules, "Models.vision_model", vision_mod)
    monkeypatch.setitem(sys.modules, "Models.dual_vision_encoder", dual_mod)
    monkeypatch.setitem(sys.modules, "Models.language_model", lang_mod)
    monkeypatch.setitem(sys.modules, "Models.embedding_model_r1", emb_mod)

    phys_mod = types.ModuleType("Models.physics_decoder")
    phys_mod.PhysicsDecoder = object
    monkeypatch.setitem(sys.modules, "Models.physics_decoder", phys_mod)

    constraints_mod = types.ModuleType("Models.physics_constraints")
    constraints_mod.compute_rte_rrs_gt = lambda *args, **kwargs: None
    constraints_mod.compute_sar_sigma0_gt = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "Models.physics_constraints", constraints_mod)

    module_path = root / "Models" / "coastgpt.py"
    spec = importlib.util.spec_from_file_location("Models.coastgpt", module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "Models.coastgpt", module)
    spec.loader.exec_module(module)
    return module.CoastGPT


def test_coastgpt_generate_uses_multiband_adapter_before_encoding(monkeypatch):
    CoastGPT = _load_coastgpt_class_with_stubs(monkeypatch)

    class DummyLanguage:
        def generate(self, **kwargs):
            self.kwargs = kwargs
            return kwargs["image_embedding"]

    class DummyModel:
        def __init__(self):
            self.language = DummyLanguage()
            self.seen_image = None
            self.adapter_called = False

        def _maybe_apply_wavelet_adapter(self, data, out):
            self.adapter_called = True
            data["rgb"] = data["multiband"][:, :3] + 5.0

        def encode_image(self, image, pool):
            self.seen_image = image.detach().clone()
            return image + 100.0

    dummy = DummyModel()
    images = torch.zeros(1, 3, 2, 2)
    multiband = torch.ones(1, 4, 2, 2)
    input_ids = torch.ones(1, 3, dtype=torch.long)

    result = CoastGPT.generate(
        dummy,
        input_ids=input_ids,
        images=images,
        multiband=multiband,
        valid_multiband=torch.tensor([True]),
        do_sample=False,
    )

    assert dummy.adapter_called is True
    assert torch.allclose(dummy.seen_image, torch.full_like(images, 6.0))
    assert torch.allclose(result, torch.full_like(images, 106.0))
    assert "multiband" not in dummy.language.kwargs
    assert "valid_multiband" not in dummy.language.kwargs
