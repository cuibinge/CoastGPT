import json as _json
import math
import os
import re as _re
from io import BytesIO
from typing import Optional, Tuple
from pathlib import Path

import ml_collections
import requests
import torch
from PIL import Image
from transformers import TextStreamer

from Dataset.build_transform import build_vlp_transform
from Dataset.conversation import SeparatorStyle, default_conversation
from Models import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    tokenizer_image_token,
)
from Models.coastgpt import CoastGPT
from Models.utils import KeywordsStoppingCriteria, type_dict
from Trainer.utils.config_parser import ConfigArgumentParser
from Trainer.utils.misc import str2bool


def _ensure_npu_runtime() -> None:
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "NPU runtime is unavailable. Please install/enable torch_npu before using --accelerator npu."
        ) from exc


def _to_dtype_name(dtype_name: str) -> torch.dtype:
    key = str(dtype_name).lower()
    if key not in type_dict:
        return torch.float16
    return type_dict[key]


def _normalize_npu_config(config: ml_collections.ConfigDict) -> None:
    if str(getattr(config, "accelerator", "")).lower() != "npu":
        return

    force_safe = bool(getattr(config, "force_safe_npu", True))
    if not force_safe:
        return

    bits = int(getattr(config, "bits", 16))
    if bits in (4, 8):
        print(
            f"[Inference] NPU safe mode: bits={bits} is not reliable in this stack, "
            f"fallback to bits=16."
        )
        config.bits = 16

    # Keep configured dtype for language model stability; vision-only bf16
    # incompatibilities are handled inside the vision encoder path.


def _normalize_inference_runtime(config: ml_collections.ConfigDict) -> None:
    # Inference should always run in eval-style checkpoint load path.
    # `stage>2` makes TextLoRA load as trainable in CoastGPT.custom_load_state_dict,
    # which is fragile for standalone generation scripts.
    try:
        stage_val = int(getattr(config, "stage", 0))
    except Exception:
        stage_val = 0
    if stage_val != 0:
        print(
            f"[Inference] force stage=0 for evaluation-style checkpoint load (was stage={stage_val})."
        )
        config.stage = 0

    merge_text_lora = getattr(config, "merge_text_lora", None)
    if merge_text_lora is None:
        if str(getattr(config, "accelerator", "")).lower() == "npu":
            config.merge_text_lora = False
            print("[Inference] disable TextLoRA merge on NPU for load stability.")
        else:
            config.merge_text_lora = True

    # Align fp16/bf16 flags with explicit dtype for stable model init on NPU.
    if str(getattr(config, "accelerator", "")).lower() != "npu":
        return
    dtype_key = str(getattr(config, "dtype", "float16")).lower()
    if dtype_key == "float16":
        if bool(getattr(config, "bf16", False)) or not bool(getattr(config, "fp16", False)):
            print("[Inference] align precision with dtype=float16: set fp16=True, bf16=False.")
        config.fp16 = True
        config.bf16 = False
    elif dtype_key in {"bfloat16", "bf16"}:
        if bool(getattr(config, "fp16", False)) or not bool(getattr(config, "bf16", False)):
            print("[Inference] align precision with dtype=bfloat16: set fp16=False, bf16=True.")
        config.fp16 = False
        config.bf16 = True


def _resolve_device(config: ml_collections.ConfigDict) -> torch.device:
    accelerator = str(getattr(config, "accelerator", "npu")).lower()
    if accelerator == "gpu":
        return torch.device("cuda")
    if accelerator == "npu":
        _ensure_npu_runtime()
        # torch_npu 2.1 + Ascend 8.0.RC2 may incorrectly route ``torch.device("npu")``
        # (no index) through the CUDA fallback inside its patched ``Module._apply``.
        # Always use an explicit ``npu:0`` (or whatever local rank we have) to keep
        # tensor moves on the NPU code path.
        try:
            import torch_npu  # noqa: F401
            local_idx = int(os.environ.get("LOCAL_RANK", "0"))
            torch_npu.npu.set_device(local_idx)
            return torch.device(f"npu:{local_idx}")
        except Exception:
            return torch.device("npu:0")
    return torch.device(accelerator)


def _load_image(image_file: str) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = None
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_file)
    return _resize_large_image_for_inference(image)


def _resize_large_image_for_inference(
    image: Image.Image,
    *,
    max_pixels: Optional[int] = None,
    max_side: Optional[int] = None,
) -> Image.Image:
    image = image.convert("RGB")
    max_pixels = int(max_pixels or os.environ.get("COASTGPT_MAX_INFERENCE_PIXELS", 16_000_000))
    max_side = int(max_side or os.environ.get("COASTGPT_MAX_INFERENCE_SIDE", 4096))
    width, height = image.size
    scales = [1.0]
    if max_pixels > 0 and width * height > max_pixels:
        scales.append(math.sqrt(max_pixels / float(width * height)))
    if max_side > 0 and max(width, height) > max_side:
        scales.append(max_side / float(max(width, height)))
    scale = min(scales)
    if scale >= 1.0:
        return image
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    print(f"[Inference] resize large image for inference: {width}x{height} -> {new_size[0]}x{new_size[1]}")
    return image.resize(new_size, Image.Resampling.BICUBIC)


def _normalize_user_instruction(text: str) -> str:
    prompt = str(text or "").strip()
    if not prompt:
        return prompt
    prompt_lower = prompt.lower()
    if "[det]" not in prompt_lower:
        return prompt

    geojson_markers = (
        "geojson",
        "featurecollection",
        "feature collection",
        "return json only",
        "return geojson",
    )
    if any(marker in prompt_lower for marker in geojson_markers):
        return prompt

    return (
        f"{prompt} Output the extracted feature information as a GeoJSON "
        "FeatureCollection. Return JSON only."
    )


def _fix_tokenizer_ids(tokenizer, model: CoastGPT) -> None:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    unk_id = getattr(tokenizer, "unk_token_id", None)

    if pad_id is None and eos_id is not None:
        tokenizer.pad_token_id = eos_id
        pad_id = eos_id

    if pad_id is not None and unk_id is not None and eos_id is not None and pad_id == unk_id:
        tokenizer.pad_token_id = eos_id
        pad_id = eos_id
        print(
            "[Inference] pad_token_id equals unk_token_id; "
            "use eos_token_id as pad_token_id for safer generation."
        )

    try:
        model.language.tokenizer.pad_token_id = tokenizer.pad_token_id
        model.language.get_text_encoder().config.pad_token_id = tokenizer.pad_token_id
        if eos_id is not None:
            model.language.get_text_encoder().config.eos_token_id = eos_id
    except Exception:
        pass

    print(
        f"[Inference] tokenizer ids: pad={getattr(tokenizer, 'pad_token_id', None)}, "
        f"eos={getattr(tokenizer, 'eos_token_id', None)}, "
        f"unk={getattr(tokenizer, 'unk_token_id', None)}"
    )


def _build_image_tensor(config: ml_collections.ConfigDict, vision_processor, image_file: Optional[str], device, dtype):
    if image_file is None:
        return None
    image = _load_image(image_file)
    if str(config.rgb_vision.arch).startswith("vit"):
        tensor = vision_processor(image, return_tensors="pt").pixel_values
        return tensor.to(device).to(dtype)
    tensor = vision_processor(image).unsqueeze(0)
    return tensor.to(device).to(dtype)


def _parse_option() -> ml_collections.ConfigDict:
    parser = ConfigArgumentParser()
    parser.add_argument("--opts", default=None, nargs="+")

    parser.add_argument("--image-file", type=str, default="../GeoJsonData/GF1/Size_128/Image_TrueColor/海水养殖区_GF1_PMS2_E119.4_N34.9_20170210_浅海区_R004C021_128_True_WFQ.jpg")
    parser.add_argument("--model-path", type=str, default="./output/stage3/mixed_v3/checkpoints/FINAL.pt")
    parser.add_argument("--seed", type=int, default=322)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Max new tokens per generation turn")
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--do-sample", type=str2bool, default=True)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stream-output", type=str2bool, default=False)
    parser.add_argument("--force-safe-npu", type=str2bool, default=True)
    parser.add_argument("--skip-text-lora", type=str2bool, default=False)
    parser.add_argument("--merge-text-lora", type=str2bool, default=None)
    parser.add_argument("--diag-on-unk", type=str2bool, default=True)
    parser.add_argument("--diag-topk", type=int, default=8)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--save-predictions", type=str, default=None)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--coord-transform-path", type=str, default=None,
                        help="Path to coord_transform_train.json for coordinate inverse transform")

    parser.add_argument(
        "--accelerator",
        default="npu",
        type=str,
        choices=["cpu", "npu", "gpu", "mps"],
    )
    # parser.add_argument("--use-checkpoint", default=False, type=str2bool)

    config = parser.parse_args(wandb=True)
    return ml_collections.config_dict.ConfigDict(config)


def _build_generation_kwargs(config, tokenizer, stopping_criteria):
    do_sample = bool(getattr(config, "do_sample", True))
    temperature = float(getattr(config, "temperature", 0.4))
    top_k = int(getattr(config, "top_k", 0))
    top_p = float(getattr(config, "top_p", 1.0))
    max_new_tokens = int(getattr(config, "max_new_tokens", 512))
    min_new_tokens = max(1, int(getattr(config, "min_new_tokens", 1)))

    if str(config.accelerator).lower() == "npu":
        if do_sample:
            print(
                "[Inference] NPU sampling path may trigger AICPU errors. "
                "Fallback to greedy decoding."
            )
        do_sample = False
        temperature = 1.0
        top_k = 50
        top_p = 1.0
        if min_new_tokens < 4:
            min_new_tokens = 4

    kwargs = dict(
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        use_cache=True,
        remove_invalid_values=True,
        renormalize_logits=True,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
    )

    if do_sample:
        kwargs["top_k"] = top_k
        kwargs["top_p"] = top_p
    else:
        kwargs["top_k"] = 50
        kwargs["top_p"] = 1.0
        kwargs["temperature"] = 1.0

    # On NPU we skip custom stopping criteria for stability.
    if str(config.accelerator).lower() != "npu":
        kwargs["stopping_criteria"] = [stopping_criteria]

    return kwargs


def _postprocess_geojson(text: str) -> str:
    """Fix common GeoJSON generation issues: missing wrapper & truncation."""
    stripped = text.strip()
    if not stripped:
        return text

    # --- 1. Wrap missing FeatureCollection envelope & first Feature ---
    has_fc_header = '"FeatureCollection"' in stripped[:200]
    looks_like_geojson = (
        '"Feature"' in stripped
        or '"Polygon"' in stripped
        or '"coordinates"' in stripped
    )

    if not has_fc_header and looks_like_geojson:
        if stripped.startswith('['):
            # Model started mid-stream at a coordinate ring.
            # Detect if there's a complete Feature boundary later.
            first_feat_pos = stripped.find('"type": "Feature"')
            if first_feat_pos == -1:
                first_feat_pos = stripped.find('"type":"Feature"')

            if first_feat_pos > 0:
                # There IS a Feature object further in. The text before it
                # is the tail of the first (incomplete) Feature: coords + closing
                # brackets + properties. We need to prepend the first Feature's
                # opening structure.
                #
                # Pattern: the model outputs starting from inside coordinates:
                #   [x,y], [x,y], ... ]]] }, "properties": {...} }, { "type": "Feature", ...
                # We prepend: {"type":"FeatureCollection","features":[{"type":"Feature",
                #              "geometry":{"type":"Polygon","coordinates":[[
                stripped = (
                    '{"type": "FeatureCollection", "features": '
                    '[{"type": "Feature", "geometry": '
                    '{"type": "Polygon", "coordinates": [['
                    + stripped
                )
                print("[Inference][geojson] prepended FeatureCollection + first Feature Polygon header")
            else:
                # No Feature objects found — bare coordinate data only.
                stripped = (
                    '{"type": "FeatureCollection", "features": '
                    '[{"type": "Feature", "geometry": '
                    '{"type": "Polygon", "coordinates": [' + stripped
                )
        elif stripped.startswith('{"type":') or stripped.startswith('{"type" :'):
            stripped = '{"type": "FeatureCollection", "features": [' + stripped
        elif stripped[:1].isdigit() or (stripped[:1] in ('-', '+') and stripped[1:2].isdigit()):
            # Model dropped all opening brackets and started directly at a coordinate value.
            # Reconstruct: FeatureCollection -> first Feature -> Polygon -> coordinates[[[<value>
            stripped = (
                '{"type": "FeatureCollection", "features": '
                '[{"type": "Feature", "geometry": '
                '{"type": "Polygon", "coordinates": [[['
                + stripped
            )
            print("[Inference][geojson] prepended full FC+Feature+Polygon header (digit-start case)")

    # --- 2. Repair truncated JSON ---
    open_braces = stripped.count('{') - stripped.count('}')
    open_brackets = stripped.count('[') - stripped.count(']')

    if open_braces > 0 or open_brackets > 0:
        repair = stripped.rstrip()
        if repair.endswith(','):
            repair = repair[:-1]
        # Strip back to last complete value
        while repair and repair[-1] not in ('}', ']', '"', '0', '1', '2', '3',
                                             '4', '5', '6', '7', '8', '9',
                                             'e', 'l', 'u'):
            repair = repair[:-1]
        if repair.endswith(','):
            repair = repair[:-1]
        open_braces = repair.count('{') - repair.count('}')
        open_brackets = repair.count('[') - repair.count(']')
        repair += ']' * max(0, open_brackets)
        repair += '}' * max(0, open_braces)
        stripped = repair

    # --- 3. Validate & compact ---
    try:
        obj = _json.loads(stripped)
        if isinstance(obj, dict) and obj.get('type') == 'FeatureCollection':
            feats = obj.get('features', [])
            print(f"[Inference][geojson] valid FeatureCollection with {len(feats)} features")
            # Compact output: one line per feature for readability
            return _json.dumps(obj, ensure_ascii=False, indent=2)
        elif isinstance(obj, dict) and obj.get('type') == 'Feature':
            obj = {'type': 'FeatureCollection', 'features': [obj]}
            print("[Inference][geojson] wrapped single Feature into FeatureCollection")
            return _json.dumps(obj, ensure_ascii=False, indent=2)
        return stripped
    except _json.JSONDecodeError as exc:
        print(f"[Inference][geojson] JSON repair incomplete: {exc}")
        # Attempt a more aggressive fix: find the last complete Feature and cut there.
        last_good = stripped.rfind('"DLMC"')
        if last_good > 0:
            # Find the closing of this properties block
            close_prop = stripped.find('}', last_good)
            if close_prop > 0:
                close_feat = stripped.find('}', close_prop + 1)
                if close_feat > 0:
                    candidate = stripped[:close_feat + 1]
                    # Close remaining structure
                    ob = candidate.count('{') - candidate.count('}')
                    obr = candidate.count('[') - candidate.count(']')
                    candidate += ']' * max(0, obr)
                    candidate += '}' * max(0, ob)
                    try:
                        obj2 = _json.loads(candidate)
                        if isinstance(obj2, dict) and obj2.get('type') == 'FeatureCollection':
                            feats = obj2.get('features', [])
                            print(f"[Inference][geojson] aggressive repair succeeded: {len(feats)} features")
                            return _json.dumps(obj2, ensure_ascii=False, indent=2)
                    except _json.JSONDecodeError:
                        pass
        return stripped

MAX_MULTITURN_ROUNDS = 20


def _has_continue_marker(text):
    """Check if generated text ends with the CONTINUE marker."""
    return text.rstrip().endswith("<CONTINUE>")


def _strip_continue_marker(text):
    """Remove CONTINUE marker from end of text."""
    if text.rstrip().endswith("<CONTINUE>"):
        return text.rstrip()[:-len("<CONTINUE>")].rstrip()
    return text


def _merge_feature_collections(json_strings):
    """Merge multiple FeatureCollection JSON strings into one.

    Concatenates all features arrays. Preserves CRS from first collection if present.
    """
    all_features = []
    crs = None
    for js in json_strings:
        try:
            obj = _json.loads(js)
        except _json.JSONDecodeError:
            continue
        feats = obj.get("features", [])
        if isinstance(feats, list):
            all_features.extend(feats)
        if crs is None and "crs" in obj:
            crs = obj["crs"]
    merged = {"type": "FeatureCollection", "features": all_features}
    if crs is not None:
        merged["crs"] = crs
    return _json.dumps(merged, ensure_ascii=False, separators=(",", ":"))


def _inverse_transform_coordinates(geojson_str, tile_transform):
    """Convert normalized [0,1] coordinates back to EPSG:4326 lon/lat.

    tile_transform fields:
        x_min, y_max: top-left corner in real coordinates
        pixel_width, pixel_height: resolution per pixel
        image_size: [width, height] in pixels
    """
    x_min = float(tile_transform["x_min"])
    y_max = float(tile_transform["y_max"])
    pixel_w = float(tile_transform["pixel_width"])
    pixel_h = float(tile_transform["pixel_height"])
    img_w = int(tile_transform["image_size"][0])
    img_h = int(tile_transform["image_size"][1])
    geo_w = pixel_w * img_w
    geo_h = pixel_h * img_h

    obj = _json.loads(geojson_str)

    def _transform_ring(ring):
        return [[x_min + pt[0] * geo_w, y_max - pt[1] * geo_h] for pt in ring]

    def _transform_geometry(geom):
        if geom["type"] == "Polygon":
            geom["coordinates"] = [
                _transform_ring(ring) for ring in geom["coordinates"]
            ]
        elif geom["type"] == "MultiPolygon":
            geom["coordinates"] = [
                [_transform_ring(ring) for ring in polygon]
                for polygon in geom["coordinates"]
            ]

    for feature in obj.get("features", []):
        geometry = feature.get("geometry")
        if isinstance(geometry, dict):
            _transform_geometry(geometry)

    # Insert CRS declaration for ArcGIS compatibility
    obj["crs"] = {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:EPSG::4326"},
    }
    return _json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _load_coord_transform(transform_path, image_key):
    """Load tile transform for a specific image from coord_transform_train.json."""
    path = Path(transform_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        all_transforms = _json.load(f)
    return all_transforms.get(image_key)


def _decode_new_tokens(tokenizer, output_ids: torch.Tensor, prompt_len: int, stop_str: str) -> Tuple[str, str, torch.Tensor]:
    seq = output_ids[0]
    if seq.shape[0] > prompt_len:
        new_tokens = seq[prompt_len:]
    else:
        new_tokens = seq
    raw_outputs = tokenizer.decode(new_tokens, skip_special_tokens=False)
    outputs = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    while outputs.startswith(stop_str):
        outputs = outputs[len(stop_str):].lstrip()
    outputs = outputs.split("<s>")[-1].strip()
    return outputs, raw_outputs, new_tokens


def _calc_unk_stats(new_tokens: torch.Tensor, tokenizer) -> Tuple[bool, float]:
    if new_tokens.numel() == 0:
        return False, 0.0
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is None:
        return False, 0.0
    unk_mask = new_tokens == int(unk_id)
    unk_ratio = float(unk_mask.float().mean().item())
    all_unk = bool(torch.all(unk_mask).item())
    return all_unk, unk_ratio


def _run_logits_probe(
    model: CoastGPT,
    input_ids: torch.Tensor,
    image_tensor: Optional[torch.Tensor],
    base_gen_kwargs: dict,
    tokenizer,
    topk: int = 8,
) -> None:
    base_probe_kwargs = dict(base_gen_kwargs)
    base_probe_kwargs.pop("streamer", None)
    base_probe_kwargs.pop("stopping_criteria", None)
    base_probe_kwargs.update(
        dict(
            do_sample=False,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
    )

    def _summarize_logits(tag: str, logits_tensor: torch.Tensor) -> None:
        logits = logits_tensor.float()
        finite_mask = torch.isfinite(logits)
        nan_count = int(torch.isnan(logits).sum().item())
        posinf_count = int(torch.isposinf(logits).sum().item())
        neginf_count = int(torch.isneginf(logits).sum().item())
        total_count = int(logits.numel())
        finite_count = int(finite_mask.sum().item())

        if finite_count > 0:
            finite_vals = logits[finite_mask]
            min_val = float(finite_vals.min().item())
            max_val = float(finite_vals.max().item())
            mean_abs = float(finite_vals.abs().mean().item())
            std_val = float(finite_vals.std(unbiased=False).item())
        else:
            min_val = float("nan")
            max_val = float("nan")
            mean_abs = float("nan")
            std_val = float("nan")

        print(
            f"[Inference][diag][{tag}] logits stats:",
            {
                "shape": tuple(logits.shape),
                "finite": finite_count,
                "total": total_count,
                "nan": nan_count,
                "+inf": posinf_count,
                "-inf": neginf_count,
                "min": min_val,
                "max": max_val,
                "mean_abs": mean_abs,
                "std": std_val,
            },
        )

        row = logits[0]
        safe_row = torch.nan_to_num(row, nan=-1e9, posinf=1e9, neginf=-1e9)
        vocab_size = int(safe_row.shape[0])
        k = max(1, min(int(topk), vocab_size))
        top_vals, top_ids = torch.topk(safe_row, k=k, dim=-1)
        probs = torch.softmax(safe_row, dim=-1)

        top_items = []
        for token_id, logit_val in zip(top_ids.tolist(), top_vals.tolist()):
            prob_val = float(probs[int(token_id)].item())
            try:
                token_str = tokenizer.convert_ids_to_tokens(int(token_id))
            except Exception:
                token_str = None
            if token_str is None:
                try:
                    token_str = tokenizer.decode([int(token_id)], skip_special_tokens=False)
                except Exception:
                    token_str = "<decode_error>"
            top_items.append(
                {
                    "id": int(token_id),
                    "token": str(token_str).replace("\n", "\\n"),
                    "logit": float(logit_val),
                    "prob": prob_val,
                }
            )
        print(f"[Inference][diag][{tag}] top-{k} next-token:", top_items)

        unk_id = getattr(tokenizer, "unk_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        watch_ids = {"unk": unk_id, "eos": eos_id, "pad": pad_id}
        watch_stats = {}
        for name, token_id in watch_ids.items():
            if token_id is None:
                continue
            tid = int(token_id)
            if tid < 0 or tid >= vocab_size:
                watch_stats[name] = {"id": tid, "status": "out_of_vocab"}
                continue
            watch_stats[name] = {
                "id": tid,
                "logit": float(safe_row[tid].item()),
                "prob": float(probs[tid].item()),
            }
        print(f"[Inference][diag][{tag}] special-token logits:", watch_stats)

    def _probe_once(tag: str, overrides: dict) -> None:
        probe_kwargs = dict(base_probe_kwargs)
        probe_kwargs.update(overrides)
        try:
            with torch.inference_mode():
                probe_out = model.generate(
                    input_ids=input_ids,
                    images=image_tensor,
                    **probe_kwargs,
                )
        except Exception as exc:
            print(f"[Inference][diag][{tag}] logits probe failed: {exc}")
            return

        scores = getattr(probe_out, "scores", None)
        if not scores:
            print(f"[Inference][diag][{tag}] logits probe returned no scores.")
            return
        logits = scores[0]
        if logits is None or logits.numel() == 0:
            print(f"[Inference][diag][{tag}] logits probe returned empty logits.")
            return
        _summarize_logits(tag, logits)

    # Pass-1: raw-like score path (turn off invalid-value cleanup and renormalization).
    # Useful to detect whether the model natively emits NaN/Inf or flat logits.
    _probe_once(
        "raw",
        dict(
            remove_invalid_values=False,
            renormalize_logits=False,
            min_new_tokens=0,
        ),
    )
    # Pass-2: safe score path mirrors real generation defaults.
    _probe_once(
        "safe",
        dict(
            remove_invalid_values=True,
            renormalize_logits=True,
            min_new_tokens=1,
        ),
    )


def _load_checkpoint(model: CoastGPT, model_path: str, skip_text_lora: bool = False):
    ckpt = torch.load(model_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint object type: {type(ckpt)}")

    # Structured stage-2 format: {'vision_ckpt': ..., 'other_ckpt': ...}
    if "vision_ckpt" in ckpt or "other_ckpt" in ckpt:
        print("[Inference] detected structured checkpoint (vision_ckpt/other_ckpt).")
        if hasattr(model, "custom_load_state_dict") and not skip_text_lora:
            msg = model.custom_load_state_dict(model_path)
            print("[Inference] loaded structured checkpoint via custom_load_state_dict.")
            return msg
        if skip_text_lora:
            print("[Inference] skip TextLoRA load by request; use fallback structured loader.")

        # Fallback path (should rarely happen in this repo).
        if "vision_ckpt" in ckpt and hasattr(model, "vision"):
            model.vision.load_state_dict(ckpt["vision_ckpt"], strict=False)
        other = ckpt.get("other_ckpt", {})
        if isinstance(other, dict):
            mm = other.get("multimodal_projection", None)
            if isinstance(mm, dict) and hasattr(model, "multimodal"):
                model.multimodal.projection.load_state_dict(mm, strict=False)

            def _report_load_result(module_name, incompatible):
                print(
                    f"[Inference] restored {module_name}: Missing: "
                    f"{getattr(incompatible, 'missing_keys', [])}. Unexpected: "
                    f"{getattr(incompatible, 'unexpected_keys', [])}"
                )

            if hasattr(model, "_restore_embed_tokens_from_ckpt"):
                model._restore_embed_tokens_from_ckpt(ckpt, _report_load_result)
            else:
                emb = other.get("embed_tokens", None)
                lm = other.get("lm_head", None)
                try:
                    text_encoder = model.language.get_text_encoder()
                    if isinstance(emb, dict):
                        text_encoder.get_input_embeddings().load_state_dict(emb, strict=False)
                    if isinstance(lm, dict) and text_encoder.get_output_embeddings() is not None:
                        text_encoder.get_output_embeddings().load_state_dict(lm, strict=False)
                except Exception:
                    pass
        return None

    # Generic flat formats.
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif "module" in ckpt and isinstance(ckpt["module"], dict):
        state = ckpt["module"]
    else:
        state = ckpt

    # Guard against accidental nested dict load (silent no-op risk).
    nested_keys = [k for k, v in state.items() if isinstance(v, dict)]
    if nested_keys:
        sample = nested_keys[:5]
        raise RuntimeError(
            "Checkpoint appears nested and is not a flat state_dict. "
            f"Nested keys sample: {sample}"
        )

    return model.load_state_dict(state, strict=False)


def main(config: ml_collections.ConfigDict):
    _normalize_npu_config(config)
    _normalize_inference_runtime(config)
    print(
        f"[Inference] effective config: accelerator={config.accelerator}, "
        f"bits={getattr(config, 'bits', 'NA')}, dtype={getattr(config, 'dtype', 'NA')}, "
        f"fp16={getattr(config, 'fp16', 'NA')}, bf16={getattr(config, 'bf16', 'NA')}"
    )

    torch.manual_seed(int(getattr(config, "seed", 322)))
    device = _resolve_device(config)
    dtype = _to_dtype_name(getattr(config, "dtype", "float16"))

    model = CoastGPT(config)
    vision_processor = (
        model.get_image_processor()
        if bool(getattr(config, "hf_model", False))
        else build_vlp_transform(config, is_train=False)
    )
    model.to(dtype)

    if config.model_path is not None:
        msg = _load_checkpoint(
            model,
            config.model_path,
            skip_text_lora=bool(getattr(config, "skip_text_lora", False)),
        )
        print(msg)

    tokenizer = model.language.tokenizer
    _fix_tokenizer_ids(tokenizer, model)

    # === Sanity probe: are the trained embedding rows actually loaded? ===
    try:
        text_encoder = model.language.get_text_encoder()
        emb = text_encoder.get_input_embeddings().weight
        lmh = text_encoder.get_output_embeddings().weight
        # 426 == '{' under LLaMA-2 tokenizer (first GeoJSON token in training).
        probe_ids = []
        for tok_str in ("{", "Feature", "Collection"):
            try:
                ids = tokenizer.encode(tok_str, add_special_tokens=False)
                if ids:
                    probe_ids.append((tok_str, ids[0]))
            except Exception:
                pass
        for tok_str, tok_id in probe_ids:
            print(
                f"[Inference][probe] embed[{tok_id}({tok_str!r}), :5]="
                f"{emb[tok_id, :5].detach().float().cpu().tolist()}"
            )
            print(
                f"[Inference][probe] lm_head[{tok_id}({tok_str!r}), :5]="
                f"{lmh[tok_id, :5].detach().float().cpu().tolist()}"
            )
        print(
            f"[Inference][probe] vocab_size={emb.shape[0]} "
            f"tokenizer_len={len(tokenizer)}"
        )
    except Exception as exc:
        print(f"[Inference][probe] embedding probe failed: {exc}")

    model.to(device)
    model.eval()

    image_tensor = _build_image_tensor(config, vision_processor, config.image_file, device, dtype)

    conv = default_conversation.copy()
    roles = conv.roles
    image_consumed = False

    print("[Inference] interactive chat is ready. Type an empty line to exit.")
    while True:
        try:
            print(f"{roles[0]}: ", end="", flush=True)
            inp = input()
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")
        inp = _normalize_user_instruction(inp)

        if not image_consumed and image_tensor is not None:
            if bool(getattr(config, "tune_im_start", False)):
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            image_consumed = True

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # === Prompt alignment diagnostic ===
        prompt_tail = prompt[-120:] if len(prompt) > 120 else prompt
        print(f"[Inference][prompt] tail repr: {prompt_tail!r}")

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        # Print last 8 token ids + decoded strings for alignment check
        tail_ids = input_ids[0, -8:].tolist()
        tail_strs = [tokenizer.decode([tid]) for tid in tail_ids]
        print(f"[Inference][prompt] last 8 token ids: {tail_ids}")
        print(f"[Inference][prompt] last 8 tokens:    {tail_strs}")
        print(f"[Inference][prompt] total prompt tokens: {input_ids.shape[1]}")

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        streamer = None
        if bool(getattr(config, "stream_output", False)) and str(config.accelerator).lower() != "npu":
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # --- Multi-turn GeoJSON generation ---
        all_geojson_parts = []
        multiturn_round = 0

        while multiturn_round < MAX_MULTITURN_ROUNDS:
            multiturn_round += 1

            # Re-tokenize prompt each turn (it grows with conversation history)
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria(
                [stop_str], tokenizer, input_ids
            )

            gen_kwargs = _build_generation_kwargs(config, tokenizer, stopping_criteria)
            if streamer is not None:
                gen_kwargs["streamer"] = streamer

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    images=image_tensor,
                    **gen_kwargs,
                )

            outputs, raw_outputs, new_tokens = _decode_new_tokens(
                tokenizer, output_ids, int(input_ids.shape[1]), stop_str
            )
            all_unk, unk_ratio = _calc_unk_stats(new_tokens, tokenizer)

            if new_tokens.numel() > 0:
                first_ids = new_tokens[:min(8, len(new_tokens))].tolist()
                first_strs = [tokenizer.decode([tid]) for tid in first_ids]
                print(
                    f"[Inference][gen round {multiturn_round}] "
                    f"new_tokens={new_tokens.shape[0]}, first={first_strs}"
                )

            if outputs == "":
                print(f"[Inference][debug round {multiturn_round}] raw decoded: {raw_outputs!r}")
                if all_unk:
                    print(f"[Inference][debug round {multiturn_round}] token collapse: all <unk> (ratio={unk_ratio:.4f})")
                    if bool(getattr(config, "diag_on_unk", True)):
                        _run_logits_probe(
                            model=model,
                            input_ids=input_ids,
                            image_tensor=image_tensor,
                            base_gen_kwargs=gen_kwargs,
                            tokenizer=tokenizer,
                            topk=int(getattr(config, "diag_topk", 8)),
                        )
                outputs = "[Empty response: generation finished without visible text]"
                all_geojson_parts.append(outputs)
                break
            elif bool(getattr(config, "diag_on_unk", True)) and all_unk:
                print(f"[Inference][debug round {multiturn_round}] token collapse: <unk> ratio={unk_ratio:.4f}")
                _run_logits_probe(
                    model=model,
                    input_ids=input_ids,
                    image_tensor=image_tensor,
                    base_gen_kwargs=gen_kwargs,
                    tokenizer=tokenizer,
                    topk=int(getattr(config, "diag_topk", 8)),
                )

            # Post-process GeoJSON part
            is_geojson = any(
                marker in outputs
                for marker in (
                    '"Feature"', '"FeatureCollection"', '"Polygon"', '"MultiPolygon"',
                    '"coordinates"', '"geometry"', '"properties"', '"DLMC"', '"name"'
                )
            )
            if is_geojson:
                outputs = _postprocess_geojson(outputs)

            conv.messages[-1][-1] = outputs
            if streamer is None:
                print(outputs)

            # Check for CONTINUE marker
            if _has_continue_marker(outputs):
                clean_output = _strip_continue_marker(outputs)
                all_geojson_parts.append(clean_output)
                continue_prompt = (
                    "Continue."
                    if multiturn_round > 1
                    else "Continue generating the remaining features."
                )
                print(f"\n[Inference] CONTINUE detected, auto-starting round {multiturn_round + 1}...")
                conv.append_message(conv.roles[0], continue_prompt)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
            else:
                all_geojson_parts.append(outputs)
                break

        # --- Merge features and inverse transform coordinates ---
        if len(all_geojson_parts) > 1 and any(
            '"FeatureCollection"' in p for p in all_geojson_parts
        ):
            merged_geojson = _merge_feature_collections(all_geojson_parts)
            print(f"\n[Inference] Merged {len(all_geojson_parts)} parts -> {len(merged_geojson)} chars")
        elif len(all_geojson_parts) == 1:
            merged_geojson = all_geojson_parts[0]
        else:
            merged_geojson = None

        if merged_geojson is not None:
            coord_transform_path = getattr(config, "coord_transform_path", None)
            if coord_transform_path and config.image_file:
                img_name = Path(config.image_file).name
                tile_transform = _load_coord_transform(
                    coord_transform_path, img_name
                )
            else:
                tile_transform = None

            if tile_transform is not None:
                merged_geojson = _inverse_transform_coordinates(
                    merged_geojson, tile_transform
                )
                print("[Inference] Applied inverse coordinate transform + CRS (EPSG:4326)")
            else:
                print("[Inference] No tile_transform found; coordinates remain normalized, no CRS")

            outputs = merged_geojson
            conv.messages[-1][-1] = merged_geojson
            if streamer is None:
                print(f"\n[Final GeoJSON]:\n{merged_geojson[:500]}...")
                if len(merged_geojson) > 500:
                    print(f"... ({len(merged_geojson)} chars total)")
        else:
            # Non-geojson: outputs already has the final value
            conv.messages[-1][-1] = outputs
        # Optionally save predictions.json compatible with Tools.decode_loc_geojson
        if getattr(config, "save_predictions", None):
            import json as __json
            save_path = str(getattr(config, "save_predictions"))
            sample_id = getattr(config, "sample_id", None)
            if not sample_id:
                # Fallback: try to extract a name-like id from the prompt tail or image filename
                sample_id = None
            try:
                payload = {}
                # If file exists, try to merge
                try:
                    with open(save_path, "r", encoding="utf-8") as fh:
                        existing = __json.load(fh)
                    if isinstance(existing, dict):
                        payload = existing
                except Exception:
                    payload = {}

                if isinstance(payload, dict) and "predictions" in payload and isinstance(payload["predictions"], list):
                    # list format: append one record
                    rec = {"sample_id": sample_id or "sample_0", "pred": outputs}
                    payload["predictions"].append(rec)
                elif isinstance(payload, dict) and payload:
                    # dict mapping format: set/overwrite
                    payload[str(sample_id or "sample_0")] = outputs
                else:
                    # create new list-format by default
                    payload = {"predictions": [{"sample_id": sample_id or "sample_0", "pred": outputs}]}

                with open(save_path, "w", encoding="utf-8") as fh:
                    __json.dump(payload, fh, ensure_ascii=False, indent=2)
                print(f"[Inference][save] predictions written to {save_path}")
            except Exception as exc:
                print(f"[Inference][save] failed to write predictions: {exc}")

        if bool(getattr(config, "debug", False)):
            print(
                "\n[debug]",
                {
                    "prompt_len": int(input_ids.shape[1]),
                    "new_tokens": int(new_tokens.shape[0]),
                    "first_tokens": new_tokens[:16].tolist(),
                },
                "\n",
            )


if __name__ == "__main__":
    cfg = _parse_option()
    cfg.adjust_norm = False
    main(cfg)
