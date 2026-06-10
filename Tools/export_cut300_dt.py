import argparse
import importlib.machinery
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import ml_collections
import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools.coco import COCO
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPImageProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _mock_optional_modules_if_missing() -> None:
    if importlib.util.find_spec("deepspeed") is not None:
        pass
    elif "deepspeed" not in sys.modules:
        deepspeed_mod = types.ModuleType("deepspeed")
        deepspeed_mod.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)
        ds_utils_mod = types.ModuleType("deepspeed.utils")
        ds_utils_mod.__spec__ = importlib.machinery.ModuleSpec("deepspeed.utils", loader=None)
        zero_mod = types.ModuleType("deepspeed.utils.zero_to_fp32")
        zero_mod.__spec__ = importlib.machinery.ModuleSpec("deepspeed.utils.zero_to_fp32", loader=None)

        def _not_available(*args, **kwargs):
            raise RuntimeError(
                "deepspeed is not installed in this environment. "
                "If you pass a DeepSpeed checkpoint directory, please install deepspeed first."
            )

        zero_mod.get_fp32_state_dict_from_zero_checkpoint = _not_available
        zero_mod.load_state_dict_from_zero_checkpoint = _not_available

        sys.modules["deepspeed"] = deepspeed_mod
        sys.modules["deepspeed.utils"] = ds_utils_mod
        sys.modules["deepspeed.utils.zero_to_fp32"] = zero_mod

    if importlib.util.find_spec("torch_npu") is None and "torch_npu" not in sys.modules:
        torch_npu_mod = types.ModuleType("torch_npu")
        torch_npu_mod.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
        sys.modules["torch_npu"] = torch_npu_mod


_mock_optional_modules_if_missing()

from Models.coastgpt import CoastGPT
from Models.utils import type_dict

logger = logging.getLogger("export_dt")


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    text = str(v).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool from: {v}")


def set_nested_value(dct: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = dct
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def parse_opts_into_dict(base: Dict[str, Any], opts: Optional[List[str]]) -> None:
    if not opts:
        return
    if len(opts) % 2 != 0:
        raise ValueError("--opts must be KEY VALUE pairs.")
    for i in range(0, len(opts), 2):
        key = opts[i]
        raw_val = opts[i + 1]
        try:
            val = yaml.safe_load(raw_val)
        except Exception:
            val = raw_val
        set_nested_value(base, key, val)


def parse_option() -> ml_collections.ConfigDict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="Configs/step2_dual.yaml")
    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument(
        "--ann-file",
        type=str,
        default="cut300/cut_300/cut_300/val/annotation.json",
        help="COCO style annotation json for cut300 val split.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="",
        help="Directory containing validation images. If empty, auto-discover near ann-file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Checkpoint path (.pt/.pth) or DeepSpeed checkpoint directory.",
    )
    parser.add_argument(
        "--output-dt",
        type=str,
        default="output/cut300_dt.json",
        help="Output dt-file json path.",
    )
    parser.add_argument(
        "--accelerator",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "mps", "npu"],
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=float, default=16.0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--use-semantic-routing", type=str2bool, default=False)
    parser.add_argument("--task-text", type=str, default="extract aquaculture region")
    parser.add_argument("--element-text", type=str, default="aquaculture")
    parser.add_argument("--validate-loadres", type=str2bool, default=True)
    args = parser.parse_args()

    cfg_dict: Dict[str, Any] = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(str(config_path), "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if isinstance(loaded, dict):
            cfg_dict.update(loaded)

    # CLI overrides.
    cfg_dict.update(
        {
            "ann_file": args.ann_file,
            "image_root": args.image_root,
            "model_path": args.model_path,
            "output_dt": args.output_dt,
            "accelerator": args.accelerator,
            "score_threshold": args.score_threshold,
            "min_area": args.min_area,
            "max_images": args.max_images,
            "category_id": args.category_id,
            "use_semantic_routing": args.use_semantic_routing,
            "task_text": args.task_text,
            "element_text": args.element_text,
            "validate_loadres": args.validate_loadres,
        }
    )
    parse_opts_into_dict(cfg_dict, args.opts)
    return ml_collections.config_dict.ConfigDict(cfg_dict)


def resolve_device(config: ml_collections.ConfigDict) -> torch.device:
    accelerator = str(getattr(config, "accelerator", "gpu")).lower()
    if accelerator == "gpu":
        return torch.device("cuda")
    if accelerator == "npu":
        try:
            import torch_npu  # noqa: F401
        except Exception as exc:
            raise RuntimeError("accelerator=npu requested, but torch_npu is unavailable.") from exc
        return torch.device("npu")
    return torch.device(accelerator)


def strip_prefix_if_present(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state.keys()):
        return state
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}


def load_state_into_model(model: CoastGPT, state: Dict[str, torch.Tensor]) -> None:
    state = strip_prefix_if_present(state, "module.")
    msg = model.load_state_dict(state, strict=False)
    logger.info(
        "Loaded state_dict strict=False. missing=%d unexpected=%d",
        len(msg.missing_keys),
        len(msg.unexpected_keys),
    )


def load_checkpoint(model: CoastGPT, ckpt_path: Path) -> None:
    if ckpt_path.is_dir():
        try:
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        except Exception as exc:
            raise RuntimeError(
                "Checkpoint path is a directory, but DeepSpeed zero_to_fp32 is unavailable."
            ) from exc
        logger.info("Loading DeepSpeed checkpoint directory: %s", ckpt_path)
        state = get_fp32_state_dict_from_zero_checkpoint(str(ckpt_path))
        load_state_into_model(model, state)
        return

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
        logger.info("Loading checkpoint format: {'model': state_dict}")
        load_state_into_model(model, ckpt["model"])
        return
    if isinstance(ckpt, dict) and isinstance(ckpt.get("module"), dict):
        logger.info("Loading checkpoint format: {'module': state_dict}")
        load_state_into_model(model, ckpt["module"])
        return
    if isinstance(ckpt, dict) and "vision_ckpt" in ckpt:
        logger.info("Loading structured checkpoint format: {'vision_ckpt','other_ckpt'}")
        msg_vision = model.vision.load_state_dict(ckpt["vision_ckpt"], strict=False)
        logger.info(
            "Vision load strict=False. missing=%d unexpected=%d",
            len(msg_vision.missing_keys),
            len(msg_vision.unexpected_keys),
        )
        other = ckpt.get("other_ckpt", {})
        mm = other.get("multimodal_projection", {})
        if isinstance(mm, dict) and len(mm) > 0:
            msg_mm = model.multimodal.projection.load_state_dict(mm, strict=False)
            logger.info(
                "Multimodal projection load strict=False. missing=%d unexpected=%d",
                len(msg_mm.missing_keys),
                len(msg_mm.unexpected_keys),
            )
        phy = other.get("physics", {})
        if isinstance(phy, dict) and len(phy) > 0 and hasattr(model, "physics"):
            msg_phy = model.physics.load_state_dict(phy, strict=False)
            logger.info(
                "Physics head load strict=False. missing=%d unexpected=%d",
                len(msg_phy.missing_keys),
                len(msg_phy.unexpected_keys),
            )
        seg = other.get("seg_head", {})
        if isinstance(seg, dict) and len(seg) > 0 and hasattr(model, "seg_head"):
            msg_seg = model.seg_head.load_state_dict(seg, strict=False)
            logger.info(
                "Seg head load strict=False. missing=%d unexpected=%d",
                len(msg_seg.missing_keys),
                len(msg_seg.unexpected_keys),
            )
        else:
            logger.warning(
                "No seg_head weights found in structured checkpoint. "
                "If this is FINAL.pt from current training pipeline, seg head may be absent."
            )
        return
    if isinstance(ckpt, dict):
        logger.info("Loading flat state_dict checkpoint.")
        load_state_into_model(model, ckpt)
        return
    raise RuntimeError(f"Unsupported checkpoint object type: {type(ckpt)}")


def discover_image_root(ann_file: Path, image_root_arg: str) -> Path:
    if image_root_arg:
        p = Path(image_root_arg)
        if not p.exists():
            raise FileNotFoundError(f"--image-root does not exist: {p}")
        return p
    base = ann_file.parent
    candidates = [base / "img", base / "image", base / "images_", base / "images"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Cannot auto-discover image root near {ann_file}. Please pass --image-root explicitly."
    )


def build_vlp_transform_local(config: ml_collections.ConfigDict):
    if str(config.rgb_vision.arch).startswith("vit"):
        return CLIPImageProcessor.from_pretrained(config.rgb_vision.vit_name)
    crop_pct = 224 / 256
    size = int(config.transform.input_size[0] / crop_pct)
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(config.transform.input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )


def build_image_tensor(
    config: ml_collections.ConfigDict,
    vision_processor,
    image_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, int, int]:
    image = Image.open(str(image_path)).convert("RGB")
    if str(config.rgb_vision.arch).startswith("vit"):
        tensor = vision_processor(image, return_tensors="pt").pixel_values
    else:
        tensor = vision_processor(image).unsqueeze(0)
    return tensor.to(device).to(dtype), image.size[1], image.size[0]


def build_semantic_embeddings(
    model: CoastGPT,
    device: torch.device,
    task_text: str,
    element_text: str,
    task_max_len: int,
    element_max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokenizer = model.language.tokenizer
    emb_layer = model.language.get_text_encoder().get_input_embeddings()
    task_tokens = tokenizer(
        [task_text],
        padding="max_length",
        truncation=True,
        max_length=task_max_len,
        return_tensors="pt",
    )
    element_tokens = tokenizer(
        [element_text],
        padding="max_length",
        truncation=True,
        max_length=element_max_len,
        return_tensors="pt",
    )
    task_ids = task_tokens.input_ids.to(device)
    task_mask = task_tokens.attention_mask.to(device)
    element_ids = element_tokens.input_ids.to(device)
    element_mask = element_tokens.attention_mask.to(device)

    task_embs = emb_layer(task_ids)
    element_embs = emb_layer(element_ids)
    task_embs = task_embs * task_mask.unsqueeze(-1).to(task_embs.dtype)
    element_embs = element_embs * element_mask.unsqueeze(-1).to(element_embs.dtype)
    return task_embs, element_embs, task_mask, element_mask


def contour_to_segmentation(cnt: np.ndarray) -> Optional[List[float]]:
    if cnt is None or len(cnt) < 3:
        return None
    cnt = cnt.reshape(-1, 2)
    if cnt.shape[0] < 3:
        return None
    epsilon = 0.005 * cv2.arcLength(cnt.astype(np.float32), True)
    approx = cv2.approxPolyDP(cnt.astype(np.float32), epsilon, True).reshape(-1, 2)
    pts = approx if approx.shape[0] >= 3 else cnt
    seg = pts.reshape(-1).astype(float).tolist()
    if len(seg) < 6:
        return None
    return seg


def masks_to_dt(
    image_id: int,
    probs: np.ndarray,
    binary_mask: np.ndarray,
    category_id: int,
    min_area: float,
) -> List[Dict]:
    dt_list: List[Dict] = []
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < float(min_area):
            continue
        seg = contour_to_segmentation(cnt)
        if seg is None:
            continue
        x, y, w, h = cv2.boundingRect(cnt.astype(np.int32))
        comp_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
        cv2.drawContours(comp_mask, [cnt.astype(np.int32)], contourIdx=-1, color=1, thickness=-1)
        if comp_mask.sum() == 0:
            continue
        score = float(probs[comp_mask > 0].mean())
        dt_list.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": [float(x), float(y), float(w), float(h)],
                "segmentation": [seg],
                "score": score,
            }
        )
    return dt_list


def main(config: ml_collections.ConfigDict) -> None:
    config.adjust_norm = False
    ann_file = Path(str(config.ann_file))
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    image_root = discover_image_root(ann_file, str(getattr(config, "image_root", "")))
    output_dt = Path(str(config.output_dt))
    output_dt.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config)
    dtype = type_dict.get(str(getattr(config, "dtype", "float16")).lower(), torch.float16)

    logger.info("Building model with config. accelerator=%s dtype=%s", config.accelerator, dtype)
    model = CoastGPT(config)
    load_checkpoint(model, Path(str(config.model_path)))
    model.to(dtype=dtype)
    model.to(device)
    model.eval()

    if not getattr(model, "seg_enabled", False) or not hasattr(model, "seg_head"):
        raise RuntimeError(
            "Current model config has no segmentation head enabled. "
            "Please use a config/checkpoint with aquaculture_seg.enabled=true."
        )

    vision_processor = build_vlp_transform_local(config)

    with open(str(ann_file), "r", encoding="utf-8") as f:
        ann = json.load(f)
    images = ann.get("images", [])
    if int(getattr(config, "max_images", 0)) > 0:
        images = images[: int(config.max_images)]

    use_semantic = bool(getattr(config, "use_semantic_routing", False))
    task_embs = element_embs = task_mask = element_mask = None
    if use_semantic:
        task_embs, element_embs, task_mask, element_mask = build_semantic_embeddings(
            model=model,
            device=device,
            task_text=str(getattr(config, "task_text", "extract aquaculture region")),
            element_text=str(getattr(config, "element_text", "aquaculture")),
            task_max_len=int(getattr(config, "task_text_max_len", 16)),
            element_max_len=int(getattr(config, "element_text_max_len", 16)),
        )

    dt: List[Dict] = []
    score_thr = float(getattr(config, "score_threshold", 0.5))
    min_area = float(getattr(config, "min_area", 16.0))
    category_id = int(getattr(config, "category_id", 1))

    logger.info("Exporting dt-file. images=%d score_threshold=%.3f min_area=%.1f", len(images), score_thr, min_area)
    with torch.inference_mode():
        for item in tqdm(images, desc="Infer+Export"):
            image_id = int(item["id"])
            file_name = str(item["file_name"])
            image_path = image_root / file_name
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            image_tensor, h, w = build_image_tensor(
                config=config,
                vision_processor=vision_processor,
                image_path=image_path,
                device=device,
                dtype=dtype,
            )
            _, fused_spatial, _ = model.vision.encode_with_spatial(image_tensor)
            seg_out = model.seg_head(
                fused_spatial,
                input_size=(h, w),
                task_text_embs=task_embs,
                element_text_embs=element_embs,
                task_text_mask=task_mask,
                element_text_mask=element_mask,
            )
            logits = seg_out["logits"][0]  # [C,H,W]
            probs = torch.softmax(logits.float(), dim=0)[1].detach().cpu().numpy()
            binary_mask = (probs >= score_thr).astype(np.uint8)
            dt.extend(
                masks_to_dt(
                    image_id=image_id,
                    probs=probs,
                    binary_mask=binary_mask,
                    category_id=category_id,
                    min_area=min_area,
                )
            )

    with open(str(output_dt), "w", encoding="utf-8") as f:
        json.dump(dt, f, ensure_ascii=False)
    logger.info("Saved dt-file: %s (instances=%d)", output_dt, len(dt))

    if bool(getattr(config, "validate_loadres", True)) and len(dt) > 0:
        coco_gt = COCO(str(ann_file))
        coco_gt.loadRes(str(output_dt))
        logger.info("COCO.loadRes validation passed.")
    elif len(dt) == 0:
        logger.warning("dt-file is empty; COCO.loadRes validation skipped.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = parse_option()
    main(cfg)
