import logging
import os
from pathlib import Path

import ml_collections.config_dict

from Trainer.utils import ConfigArgumentParser, str2bool
from train_stage_two import (
    apply_npu_stability_overrides,
    init_wandb,
    patch_deepspeed_zero_grad_norm_for_npu,
    patch_vector_norm_for_npu,
    save_config,
    setup_environment,
    train_model,
)

logger = logging.getLogger("train")


def _norm_path(path_value) -> str:
    if path_value is None:
        return ""
    try:
        text = str(Path(str(path_value)))
    except Exception:
        text = str(path_value)
    return os.path.normcase(os.path.normpath(text))


def _is_gf_size_root(path_value) -> bool:
    if path_value is None:
        return False
    try:
        root = Path(str(path_value))
    except Exception:
        return False
    if not root.exists() or not root.is_dir():
        return False
    return any(child.is_dir() and child.name.startswith("Size_") for child in root.iterdir())


def _is_gf_collection_root(path_value) -> bool:
    if path_value is None:
        return False
    try:
        root = Path(str(path_value))
    except Exception:
        return False
    if not root.exists() or not root.is_dir():
        return False
    return any(child.is_dir() and _is_gf_size_root(child) for child in root.iterdir())


def _is_gf2_root(path_value) -> bool:
    return _is_gf_size_root(path_value) or _is_gf_collection_root(path_value)


def _parse_gf2_sizes(value):
    text = str(value or "").strip()
    if not text or text.lower() == "all":
        return None
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return parts or None


def _infer_gf2_output_root(raw_root: Path, gf2_sizes, image_subdir: str) -> Path:
    size_tag = "all" if gf2_sizes is None else "-".join(str(part) for part in gf2_sizes)
    image_tag = str(image_subdir).replace("Image_", "").lower()
    if _is_gf_collection_root(raw_root):
        sensor_names = sorted(
            child.name.lower() for child in raw_root.iterdir()
            if child.is_dir() and _is_gf_size_root(child)
        )
        sensor_tag = "-".join(sensor_names) if sensor_names else "multi"
        return raw_root / f"stage3_geojson_{sensor_tag}_{size_tag}_{image_tag}"
    return raw_root / f"stage3_geojson_{size_tag}_{image_tag}"


def parse_option():
    parser = ConfigArgumentParser()

    # Basic parameters
    parser.add_argument("--batch-size", type=int, help="Batch size per device")
    parser.add_argument("--data-path", type=str, help="Path to Stage-3 dataset root")
    parser.add_argument("--eval-data-path", type=str, help="Path to evaluate dataset")
    parser.add_argument("--workers", type=int, default=8, help="Workers of dataloader")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--resume-path", type=str, default=None, help="Resume checkpoint path")
    parser.add_argument("--model-path", type=str, default=None, help="Pretrained checkpoint path")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Max gradient norm. Set 0 to disable clipping.")
    parser.add_argument("--use-checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--enable-amp", type=str2bool, default=False, help="Enable mixed precision")
    parser.add_argument("--output", default="output", type=str, metavar="PATH", help="Root of output folder")
    parser.add_argument("--seed", type=int, default=322, help="Random seed")
    parser.add_argument("--inf_sampler", type=str2bool, default=False, help="Use Infinite loader")
    parser.add_argument("--torch-compile", type=str2bool, default=False, help="Use torch.compile")
    parser.add_argument("--epochs", type=int, default=None, help="Override max training epochs from config")
    parser.add_argument(
        "--ckpt-period",
        type=int,
        default=None,
        help="Checkpoint save period. Set <= 0 to disable intermediate checkpoints.",
    )
    parser.add_argument(
        "--max-debug-iters",
        type=int,
        default=0,
        help="If > 0, run only this many iterations for quick sanity check",
    )

    # Stage-3 options
    parser.add_argument(
        "--geojson-priority",
        type=str2bool,
        default=None,
        help="Prefer GeoJSON localization targets when auto-building conversations",
    )
    parser.add_argument(
        "--raw-data-root",
        type=str,
        default=None,
        help="Raw data root containing train/ and val/ folders (default: data)",
    )
    parser.add_argument(
        "--auto-build-geojson-data",
        type=str2bool,
        default=None,
        help="Build GeoJSON instruction dataset from raw data before training",
    )
    parser.add_argument(
        "--geojson-output-root",
        type=str,
        default=None,
        help="Output root for generated GeoJSON instruction dataset (default: stage3_data)",
    )
    parser.add_argument(
        "--geojson-copy-images",
        type=str2bool,
        default=None,
        help="Copy images when building GeoJSON instruction dataset (else symlink first)",
    )
    parser.add_argument(
        "--geojson-build-val",
        type=str2bool,
        default=None,
        help="Also build val split into stage3_data (train split is always built)",
    )
    parser.add_argument(
        "--gf2-sizes",
        type=str,
        default=None,
        help=(
            "Comma-separated tile sizes to include when raw-data-root points to a GF sensor root "
            "or a parent directory such as 养殖区数据集. Default: 512"
        ),
    )
    parser.add_argument(
        "--gf2-image-subdir",
        type=str,
        default=None,
        help=(
            "Image variant directory to use when raw-data-root points to a GF sensor root "
            "or a parent directory such as 养殖区数据集. Default: Image_FalseColor"
        ),
    )
    parser.add_argument(
        "--wavelet-adapter-enabled",
        type=str2bool,
        default=None,
        help="Enable multi-band wavelet adapter input path",
    )
    parser.add_argument(
        "--wavelet-adapter-mode",
        type=str,
        default=None,
        help="Wavelet adapter mode, currently learnable_direct",
    )
    parser.add_argument(
        "--wavelet-adapter-in-channels",
        type=int,
        default=None,
        help="Number of raw multi-band channels consumed by the adapter",
    )
    parser.add_argument(
        "--wavelet-adapter-multiband-channels",
        type=int,
        default=None,
        help="Maximum TIFF bands loaded by the dataset",
    )

    # W&B parameters
    parser.add_argument("--wandb", type=str2bool, default=False, help="Enable wandb logging")
    parser.add_argument("--entity", type=str, default="pumpkinn", help="Wandb entity")
    parser.add_argument("--project", type=str, default="MultiModal", help="Wandb project")
    parser.add_argument("--job-type", type=str, default="vlm_test", help="Wandb job_type")
    parser.add_argument("--tags", type=str, default="MultiModal", nargs="+", help="Wandb tags")
    parser.add_argument("--name", type=str, default="first_run", help="Wandb run name")
    parser.add_argument("--notes", type=str, default=None, help="Wandb run notes")

    # Hardware parameters
    parser.add_argument("--accelerator", default="npu", type=str, choices=["cpu", "gpu", "mps", "npu"])
    parser.add_argument("--local_rank", type=int)

    return parser.parse_args(wandb=True)


def apply_stage3_defaults(config: ml_collections.config_dict.ConfigDict):
    stage_value = int(getattr(config, "stage", 3))
    if not hasattr(config, "stage"):
        config.stage = stage_value
    if stage_value < 3:
        logger.warning("Config stage=%s is lower than 3. Forcing stage=3 for Stage-3 training.", stage_value)
        config.stage = 3

    if getattr(config, "geojson_priority", None) is None:
        config.geojson_priority = True

    if getattr(config, "raw_data_root", None) is None:
        config.raw_data_root = "data"
    raw_root = Path(str(getattr(config, "raw_data_root", "data")))
    gf2_mode = _is_gf2_root(raw_root)
    if getattr(config, "gf2_sizes", None) is None:
        if _is_gf_collection_root(raw_root):
            config.gf2_sizes = "all"
        else:
            config.gf2_sizes = "512" if gf2_mode else None
    if getattr(config, "gf2_image_subdir", None) is None:
        config.gf2_image_subdir = "Image_FalseColor"
    if getattr(config, "geojson_output_root", None) is None:
        if gf2_mode:
            config.geojson_output_root = str(
                _infer_gf2_output_root(
                    raw_root=raw_root,
                    gf2_sizes=_parse_gf2_sizes(getattr(config, "gf2_sizes", None)),
                    image_subdir=str(getattr(config, "gf2_image_subdir", "Image_FalseColor")),
                )
            )
        else:
            config.geojson_output_root = "stage3_data"
    if getattr(config, "geojson_copy_images", None) is None:
        config.geojson_copy_images = False
    if getattr(config, "geojson_build_val", None) is None:
        config.geojson_build_val = False
    if getattr(config, "auto_build_geojson_data", None) is None:
        config.auto_build_geojson_data = True
    if getattr(config, "ckpt_period", None) is None:
        config.ckpt_period = 0

    wavelet_cfg = getattr(config, "wavelet_adapter", ml_collections.config_dict.ConfigDict())
    if getattr(config, "wavelet_adapter_enabled", None) is not None:
        wavelet_cfg.enabled = bool(config.wavelet_adapter_enabled)
    if getattr(config, "wavelet_adapter_mode", None) is not None:
        wavelet_cfg.mode = str(config.wavelet_adapter_mode)
    if getattr(config, "wavelet_adapter_in_channels", None) is not None:
        wavelet_cfg.in_channels = int(config.wavelet_adapter_in_channels)
    if getattr(config, "wavelet_adapter_multiband_channels", None) is not None:
        wavelet_cfg.multiband_channels = int(config.wavelet_adapter_multiband_channels)
    config.wavelet_adapter = wavelet_cfg

    # Stage-3 on NPU is much more stable in bf16 than fp16.
    if str(getattr(config, "accelerator", "")).lower() == "npu":
        use_fp16 = bool(getattr(config, "fp16", False))
        use_bf16 = bool(getattr(config, "bf16", False))
        if use_fp16 and not use_bf16:
            logger.warning(
                "Detected fp16-only setting on NPU for Stage-3; force bf16=True and fp16=False for stability."
            )
            config.fp16 = False
            config.bf16 = True
        grad_clip = getattr(config, "max_grad_norm", None)
        grad_clip = 0.0 if grad_clip is None else float(grad_clip)
        if grad_clip <= 0.0:
            config.max_grad_norm = 0.0
        else:
            logger.warning(
                "Stage-3 on NPU: force max_grad_norm=0.0 to avoid DeepSpeed ZeRO grad-norm all-reduce instability."
            )
            config.max_grad_norm = 0.0

    if not getattr(config, "data_path", None):
        if gf2_mode:
            config.data_path = str(config.geojson_output_root)
        else:
            config.data_path = os.path.join(str(config.raw_data_root), "train")
        logger.info("No --data-path provided, default to %s", config.data_path)

    return config


def maybe_build_geojson_data(config: ml_collections.config_dict.ConfigDict):
    if not bool(getattr(config, "auto_build_geojson_data", False)):
        return config

    raw_root = Path(str(getattr(config, "raw_data_root", "data")))
    output_root = Path(str(getattr(config, "geojson_output_root", "stage3_data")))

    if _is_gf2_root(raw_root):
        try:
            from Tools.build_gf2_geojson_dataset import build_dataset
        except Exception as exc:
            logger.warning("Skip GF2 auto-build: failed to import Tools.build_gf2_geojson_dataset (%s).", exc)
            return config

        # Coordinate encoding policy:
        #   loc_tokens.enabled=True  -> quantise to <loc_*> tokens (only do this
        #     if you have separately verified the embed_tokens save path actually
        #     captures trained values; otherwise the new tokens never learn).
        #   loc_tokens.enabled=False -> normalised [0, 1] float coordinates that
        #     work with the original 32000-token vocabulary.
        loc_cfg = getattr(config, "loc_tokens", None)
        loc_enabled = bool(loc_cfg is not None and getattr(loc_cfg, "enabled", False))
        loc_bins = int(getattr(loc_cfg, "num_bins", 1000)) if loc_cfg is not None else 0
        quantize_coords = loc_bins if loc_enabled else 0
        normalize_coords = True  # always normalise; only quantisation is optional
        logger.info(
            "Auto-building Stage-3 GF GeoJSON dataset: %s -> %s (normalize_coords=%s, quantize_coords=%s)",
            raw_root, output_root, normalize_coords, quantize_coords,
        )
        build_dataset(
            gf2_root=raw_root,
            output_dir=output_root,
            size_filter=_parse_gf2_sizes(getattr(config, "gf2_sizes", None)),
            image_subdir=str(getattr(config, "gf2_image_subdir", "Image_FalseColor")),
            label_subdir="Label_GeoJSON",
            copy_images=bool(getattr(config, "geojson_copy_images", False)),
            compact_answer=True,
            keep_crs=False,
            normalize_coords=normalize_coords,
            quantize_coords=quantize_coords,
        )

        generated_train_json = output_root / "GF2_geojson_train.json"
        if generated_train_json.exists():
            config.data_path = str(output_root)
            logger.info("GF auto-build finished. data-path switched to generated dataset root: %s", config.data_path)
        else:
            logger.warning("GF auto-build finished but generated train json not found: %s", generated_train_json)
        return config

    train_root = raw_root / "train"
    val_root = raw_root / "val"

    if not train_root.exists():
        logger.warning(
            "Skip auto-build: train split path does not exist: %s. Continue with data-path=%s",
            train_root,
            getattr(config, "data_path", None),
        )
        return config

    try:
        from Tools.build_geojson_instructions import process_split
    except Exception as exc:
        logger.warning("Skip auto-build: failed to import Tools.build_geojson_instructions (%s).", exc)
        return config

    logger.info("Auto-building Stage-3 GeoJSON instruction dataset: %s -> %s", train_root, output_root)
    process_split(
        data_root=train_root,
        output_dir=output_root,
        split="train",
        copy_images=bool(getattr(config, "geojson_copy_images", False)),
    )

    if bool(getattr(config, "geojson_build_val", False)) and val_root.exists():
        process_split(
            data_root=val_root,
            output_dir=output_root,
            split="val",
            copy_images=bool(getattr(config, "geojson_copy_images", False)),
        )

    generated_train_json = output_root / "GF_geojson_train.json"
    if not generated_train_json.exists():
        logger.warning(
            "Auto-build finished but generated train json not found: %s. Keep data-path=%s",
            generated_train_json,
            getattr(config, "data_path", None),
        )
        return config

    current_data_path = str(getattr(config, "data_path", ""))
    raw_train_default = os.path.join(str(raw_root), "train")
    raw_root_text = str(raw_root)
    current_norm = _norm_path(current_data_path)
    raw_train_norm = _norm_path(raw_train_default)
    raw_root_norm = _norm_path(raw_root_text)
    should_switch = (not current_data_path) or (current_norm in {raw_train_norm, raw_root_norm})

    if should_switch:
        config.data_path = str(output_root)
        logger.info("Auto-build finished. data-path switched to generated dataset root: %s", config.data_path)
    else:
        logger.info(
            "Auto-build finished, but keep user data-path unchanged: %s",
            current_data_path,
        )
    return config


def ensure_stage3_data_ready(config: ml_collections.config_dict.ConfigDict):
    stage_value = int(getattr(config, "stage", 3))
    if stage_value < 3:
        return config

    data_path = Path(str(getattr(config, "data_path", "")))
    if not data_path.exists():
        logger.warning("Configured data-path does not exist: %s", data_path)
        return config

    has_instruction_json = any(data_path.glob("*.json"))
    if has_instruction_json:
        return config

    has_geojson = any(data_path.glob("*.geojson"))
    output_root = Path(str(getattr(config, "geojson_output_root", "stage3_data")))
    generated_candidates = [
        output_root / "GF_geojson_train.json",
        output_root / "GF2_geojson_train.json",
    ]
    existing_generated = next((path for path in generated_candidates if path.exists()), None)
    if existing_generated is not None:
        logger.warning(
            "data-path=%s contains no instruction json; switch to generated Stage-3 dataset root: %s",
            data_path,
            output_root,
        )
        config.data_path = str(output_root)
        return config

    if _is_gf2_root(data_path):
        raise RuntimeError(
            "Stage-3 expects generated instruction json under the GF dataset root, "
            f"but data-path={data_path} points to raw GF tiles or a raw GF collection directory. "
            "Use --auto-build-geojson-data True or set --data-path to the generated stage3 directory."
        )

    if has_geojson:
        raise RuntimeError(
            "Stage-3 expects instruction json (e.g. GF_geojson_train.json), "
            f"but data-path={data_path} only contains raw .geojson files. "
            "Use --auto-build-geojson-data True or set --data-path to the generated dataset root."
        )

    logger.warning(
        "data-path=%s has no instruction json. Training may have zero samples; "
        "please check dataset layout.",
        data_path,
    )
    return config


if __name__ == "__main__":
    config = ml_collections.config_dict.ConfigDict(parse_option())
    if str(getattr(config, "accelerator", "")).lower() == "npu":
        patch_vector_norm_for_npu()
        patch_deepspeed_zero_grad_norm_for_npu()
    apply_npu_stability_overrides(config)
    config = apply_stage3_defaults(config)
    config = maybe_build_geojson_data(config)
    config = ensure_stage3_data_ready(config)
    config = setup_environment(config)
    save_config(config)
    config = init_wandb(config)
    train_model(config)
