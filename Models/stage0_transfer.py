from pathlib import Path
from typing import Mapping, Optional, Union

import torch


_WEIGHT_KEYS = (
    "visual_proj.1.weight",
    "stage0_contrastive.visual_proj.1.weight",
    "module.stage0_contrastive.visual_proj.1.weight",
)
_NORM_WEIGHT_KEYS = (
    "visual_proj.0.weight",
    "stage0_contrastive.visual_proj.0.weight",
    "module.stage0_contrastive.visual_proj.0.weight",
)
_NORM_BIAS_KEYS = (
    "visual_proj.0.bias",
    "stage0_contrastive.visual_proj.0.bias",
    "module.stage0_contrastive.visual_proj.0.bias",
)


def _candidate_state_dicts(checkpoint: Mapping):
    yield checkpoint
    for key in ("module", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            yield value
    other = checkpoint.get("other_ckpt")
    if isinstance(other, Mapping):
        yield other
        stage0 = other.get("stage0_contrastive")
        if isinstance(stage0, Mapping):
            yield stage0
    stage0 = checkpoint.get("stage0_contrastive")
    if isinstance(stage0, Mapping):
        yield stage0


def extract_stage0_visual_proj_weight(checkpoint: Mapping) -> torch.Tensor:
    for state in _candidate_state_dicts(checkpoint):
        for key in _WEIGHT_KEYS:
            value = state.get(key)
            if torch.is_tensor(value):
                return value
    keys = []
    for state in _candidate_state_dicts(checkpoint):
        keys.extend(str(key) for key in state.keys())
    sample = ", ".join(sorted(keys)[:20])
    raise KeyError(
        "Stage0 visual projection weight not found. Expected one of "
        f"{_WEIGHT_KEYS}; checkpoint keys sample: {sample}"
    )


def _extract_optional_tensor(checkpoint: Mapping, keys) -> Optional[torch.Tensor]:
    for state in _candidate_state_dicts(checkpoint):
        for key in keys:
            value = state.get(key)
            if torch.is_tensor(value):
                return value
    return None


def extract_stage0_visual_proj_norm(checkpoint: Mapping):
    return (
        _extract_optional_tensor(checkpoint, _NORM_WEIGHT_KEYS),
        _extract_optional_tensor(checkpoint, _NORM_BIAS_KEYS),
    )


def copy_stage0_visual_proj_to_moe(model, weight: torch.Tensor) -> int:
    projection = model.multimodal.projection
    experts = getattr(projection, "experts", None)
    if experts is None:
        raise AttributeError("model.multimodal.projection has no experts to initialize")

    copied = 0
    for idx, expert in enumerate(experts):
        out_proj = getattr(expert, "out_proj", None)
        if out_proj is None or not hasattr(out_proj, "weight"):
            raise AttributeError(f"expert {idx} has no out_proj.weight")
        if tuple(out_proj.weight.shape) != tuple(weight.shape):
            raise ValueError(
                f"Stage0 visual_proj weight shape {tuple(weight.shape)} does not match "
                f"expert {idx} out_proj weight shape {tuple(out_proj.weight.shape)}"
            )
        with torch.no_grad():
            out_proj.weight.copy_(weight.to(device=out_proj.weight.device, dtype=out_proj.weight.dtype))
            if getattr(out_proj, "bias", None) is not None:
                out_proj.bias.zero_()
        copied += 1
    return copied


def load_stage0_projection_into_moe(model, checkpoint_path: Union[str, Path]) -> int:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    weight = extract_stage0_visual_proj_weight(checkpoint)
    copied = copy_stage0_visual_proj_to_moe(model, weight)

    norm_weight, norm_bias = extract_stage0_visual_proj_norm(checkpoint)
    if norm_weight is not None or norm_bias is not None:
        for expert in getattr(model.multimodal.projection, "experts", []):
            norm = getattr(expert, "pre_out_norm", None)
            if norm is None:
                continue
            with torch.no_grad():
                if norm_weight is not None and hasattr(norm, "weight"):
                    if tuple(norm.weight.shape) != tuple(norm_weight.shape):
                        raise ValueError(
                            f"Stage0 visual_proj norm weight shape {tuple(norm_weight.shape)} "
                            f"does not match expert pre_out_norm shape {tuple(norm.weight.shape)}"
                        )
                    norm.weight.copy_(norm_weight.to(device=norm.weight.device, dtype=norm.weight.dtype))
                if norm_bias is not None and hasattr(norm, "bias") and norm.bias is not None:
                    if tuple(norm.bias.shape) != tuple(norm_bias.shape):
                        raise ValueError(
                            f"Stage0 visual_proj norm bias shape {tuple(norm_bias.shape)} "
                            f"does not match expert pre_out_norm bias shape {tuple(norm.bias.shape)}"
                        )
                    norm.bias.copy_(norm_bias.to(device=norm.bias.device, dtype=norm.bias.dtype))
    return copied


def freeze_for_stage0_alignment(model) -> None:
    stage0 = getattr(model, "stage0_contrastive", None)
    if stage0 is None:
        raise AttributeError("model.stage0_contrastive is required for Stage0 alignment")

    for param in model.multimodal.parameters():
        param.requires_grad_(False)
    for param in stage0.parameters():
        param.requires_grad_(True)
