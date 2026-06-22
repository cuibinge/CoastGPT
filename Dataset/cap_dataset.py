import copy
import io
import json
import logging
import os
import random
import re
import numpy as np
from dataclasses import dataclass
from multiprocessing import Value
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchvision.transforms as T
import transformers
import webdataset as wds
from PIL import Image, UnidentifiedImageError
from torch.utils.data import get_worker_info
from transformers import CLIPImageProcessor
from webdataset.filters import _shuffle
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

from Models import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from . import conversation as conversation_lib
from .constants import ELEMENT2ID, TASK2ID
from .multiband_source import load_multiband_tensor, stack_optional_multiband
from utils.geojson_coordinate_utils import repair_mojibake_in_obj
try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000
logger = logging.getLogger("train")
_TIFF_FALLBACK_WARNED = set()


def valid_path(path: Union[Path, str]) -> bool:
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        return False
    return True


def _normalize_remote_sensing_channel(channel: np.ndarray) -> np.ndarray:
    """Normalize one raster band to uint8 with robust percentile scaling."""
    channel = channel.astype(np.float32, copy=False)
    invalid_mask = ~np.isfinite(channel) | (channel <= -1e9)
    channel = channel.copy()
    channel[invalid_mask] = np.nan

    valid = channel[~np.isnan(channel)]
    if valid.size == 0:
        return np.zeros(channel.shape, dtype=np.uint8)

    lo = float(np.percentile(valid, 2.0))
    hi = float(np.percentile(valid, 98.0))
    if hi <= lo:
        lo = float(valid.min())
        hi = float(valid.max())
    if hi <= lo:
        return np.zeros(channel.shape, dtype=np.uint8)

    scaled = (channel - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0)
    return (scaled * 255.0).astype(np.uint8)


def _load_tiff_as_rgb(path: Union[Path, str]) -> Image.Image:
    try:
        import tifffile
    except ImportError as exc:
        raise UnidentifiedImageError(
            f"TIFF image requires tifffile fallback, but tifffile is unavailable: {path}"
        ) from exc

    raster = tifffile.imread(str(path))
    raster = np.asarray(raster)
    raster = np.squeeze(raster)

    if raster.ndim == 2:
        raster = np.repeat(raster[..., None], 3, axis=-1)
    elif raster.ndim == 3 and raster.shape[0] <= 8 and raster.shape[-1] > 8:
        raster = np.moveaxis(raster, 0, -1)

    if raster.ndim != 3:
        raise UnidentifiedImageError(f"Unsupported TIFF raster shape {raster.shape} for {path}")

    channels = raster.shape[-1]
    if channels >= 4:
        # GF2 multi-spectral order is typically Blue, Green, Red, NIR.
        rgb = raster[..., [2, 1, 0]]
    elif channels == 3:
        rgb = raster[..., :3]
    elif channels == 2:
        rgb = np.stack([raster[..., 0], raster[..., 1], raster[..., 1]], axis=-1)
    elif channels == 1:
        rgb = np.repeat(raster, 3, axis=-1)
    else:
        raise UnidentifiedImageError(f"Unsupported TIFF channel count {channels} for {path}")

    rgb_uint8 = np.stack(
        [_normalize_remote_sensing_channel(rgb[..., i]) for i in range(3)],
        axis=-1,
    )
    return Image.fromarray(rgb_uint8, mode="RGB")


def load_image_as_rgb(path: Union[Path, str]) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except UnidentifiedImageError:
        suffix = str(path).lower()
        if suffix.endswith(".tif") or suffix.endswith(".tiff"):
            warn_key = "__tiff_fallback__"
            if warn_key not in _TIFF_FALLBACK_WARNED:
                logger.warning("PIL failed to read TIFF, falling back to tifffile: %s", path)
                _TIFF_FALLBACK_WARNED.add(warn_key)
            return _load_tiff_as_rgb(path)
        raise


def pre_caption(caption, max_words=50):
    if isinstance(caption, Dict) or isinstance(caption, List):
        return caption
    else:
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > max_words:
            caption = " ".join(caption_words[:max_words])

    return caption


class CaptionDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root: Union[Path, str] = ".data/rsicd",
            transform: T.Compose = None,
            return_multiband: bool = False,
            multiband_channels: int = 4,
    ):
        if isinstance(root, str):
            root = Path(root)

        self.root = root
        self.transform = transform
        self.return_multiband = bool(return_multiband)
        self.multiband_channels = int(multiband_channels)
        self.img_dir = list(self.root.glob("*_Image"))
        self.json_dir = []
        for i in self.img_dir:
            if "captions" in i.name:
                self.json_dir.append(i.parent / "OSCapAnn" / (i.name.split("_Image")[0] + ".json"))
            elif "OSM" not in i.name:
                self.json_dir.append(i.parent / (i.name.split("_Image")[0] + ".json"))
            else:
                may_be_exist = i.parent / "OSMCapAnn"
                if may_be_exist.exists():
                    self.json_dir.append(may_be_exist)
                else:
                    self.json_dir.append(i.parent / (i.name.split("_Image")[0] + ".json"))

        self.img_list = []
        self.cap_list = []
        self.load_dataset()
        self.post_process()

    def post_process(self):
        pass

    def load_physics(self, idx: int):
        """
        加载与图像空间对齐的物理真值 (TSM) 与有效域掩码 (Mask)。
        这部分数据必须在制作 CoastBench 时提前由定量遥感算法生成。
        """
        img_path = Path(self.img_list[idx])

        # 假设物理真值存储在与图像同级的 TSM 文件夹或具有特定后缀
        # 实际路径逻辑请根据你的 CoastBench 存储规范严格修改
        tsm_path = img_path.parent / (img_path.stem + "_tsm.npy")
        mask_path = img_path.parent / (img_path.stem + "_mask.npy")

        if tsm_path.exists() and mask_path.exists():
            # 加载并转换为张量，通常需要下采样到特征图的尺度 (如 F32 对应的尺度)
            # 或者在这里保持原图尺寸，在损失函数计算前进行 F.interpolate
            tsm_tensor = torch.from_numpy(np.load(tsm_path)).float()
            mask_tensor = torch.from_numpy(np.load(mask_path)).float()
            return tsm_tensor, mask_tensor
        else:
            return None, None

    def load_dataset(self):
        for i in range(len(self.img_dir)):
            if "OSM" not in self.img_dir[i].stem:
                with open(self.json_dir[i], "rb") as f:
                    data = json.load(f)

            if "captions" in self.img_dir[i].stem:
                for item in data["data"]:
                    name = item["name"]
                    img_path = self.img_dir[i] / (name + ".png")
                    if valid_path(img_path):
                        self.img_list.append(img_path)
                        self.cap_list.append(item["cap"])
            elif "TextRS" in self.img_dir[i].stem:
                text_rs = data["TextRS"]
                for j in range(len(text_rs)):
                    img_path = self.img_dir[i] / (text_rs[j]["image"] + ".png")
                    if valid_path(img_path):
                        self.img_list.append(self.img_dir[i] / (text_rs[j]["image"] + ".png"))
                        self.cap_list.append(text_rs[j]["annotation"]["caption"][0])
            elif "UAVICD" in self.img_dir[i].stem:
                for j in range(len(data["images"])):
                    img_path = self.img_dir[i] / data["images"][j]["SubFolder"] / data["images"][j]["ImageName"]
                    if valid_path(img_path):
                        self.img_list.append(
                            self.img_dir[i] / data["images"][j]["SubFolder"] / data["images"][j]["ImageName"]
                        )
                        self.cap_list.append(data["images"][j]["Caption"])
            elif "NWPU" in self.img_dir[i].stem:
                for sub_folder in data.keys():
                    sub_data = data[sub_folder]
                    for j in range(len(sub_data)):
                        img_path = self.img_dir[i] / sub_folder / sub_data[j]["filename"]
                        if valid_path(img_path):
                            self.img_list.append(self.img_dir[i] / sub_folder / sub_data[j]["filename"])
                            self.cap_list.append(sub_data[j]["raw"])
            elif "OSM" in self.img_dir[i].stem:
                for json_file in self.json_dir[i].iterdir():
                    if json_file.is_file() and json_file.suffix == ".json":
                        with open(json_file, "r") as f:
                            data = json.load(f)["data"]

                        for item in data:
                            name = item["name"]
                            country, city = item["info"]["location"]
                            img_path = self.img_dir[i] / country / city / (name + ".jpg")
                            if valid_path(img_path):
                                self.img_list.append(img_path)
                                self.cap_list.append(item["cap"])
            elif "LLAVA" in self.img_dir[i].stem:
                data = data["data"]
                for item in data:
                    image_path = self.img_dir[i] / item["name"]
                    if valid_path(image_path):
                        self.img_list.append(image_path)
                        self.cap_list.append(item["conv"])
            else:
                for item in data["data"]:
                    name = item["name"]
                    for it in item["features"]:
                        properties = it["properties"]
                        img_path = self.img_dir[i] / (name + ".png")
                        if valid_path(img_path):
                            self.img_list.append(img_path)
                            self.cap_list.append(properties["caption1"])

    def __len__(self) -> int:
        return len(self.cap_list)

    def load_image(self, idx: int):
        x = load_image_as_rgb(self.img_list[idx])
        if self.transform is not None:
            if isinstance(self.transform, CLIPImageProcessor):
                x = self.transform(x, return_tensors="pt").pixel_values.squeeze()
            else:
                x = self.transform(x)

        return x

    def load_multiband(self, idx: int, reference: Optional[torch.Tensor] = None):
        output_size = tuple(reference.shape[-2:]) if torch.is_tensor(reference) else None
        return load_multiband_tensor(
            self.img_list[idx],
            output_size=output_size,
            max_channels=self.multiband_channels,
        )

    def __getitem__(self, idx: int) -> Dict:
        captions = self.cap_list[idx]
        if not isinstance(captions, list):
            captions = pre_caption(captions)

        x = self.load_image(idx)
        tsm, mask = self.load_physics(idx)  # 新增物理数据加载
        out = dict(rgb=x, text=captions, tsm=tsm, mask=mask)
        if self.return_multiband:
            out["multiband"] = self.load_multiband(idx, reference=x)
        return out


class VGEvalDataset(CaptionDataset):
    def __init__(
            self,
            root: Union[Path, str] = ".data/rsicd",
            target: Union[Path, str] = None,
            transform: T.Compose = None,
            tokenizer: transformers.PreTrainedTokenizer = None,
            **kwargs,
    ):
        prompt_type = kwargs.pop("prompt_type", "llava_llama_2")
        conversation_lib.default_conversation = conversation_lib.conv_templates[prompt_type]

        if isinstance(root, str):
            root = Path(root)

        if isinstance(target, str):
            target = Path(target)

        self.transform = transform
        self.img_dir = root
        self.json_dir = target

        self.img_list = []
        self.prompt_list = []
        self.target_list = []
        self.tokenizer = tokenizer
        self.tune_im_start = kwargs.pop("tune_im_start", False)
        self.load_dataset()
        self.post_process()

    def load_dataset(self):
        with open(self.json_dir, "rb") as f:
            data = json.load(f)["data"]

        dataset_name = self.json_dir.stem

        for item in data:
            if dataset_name.endswith("RSVG_test"):
                img_path = self.img_dir / item["img"]
                item["conv"] = dict(Question=item["question"], Answer=None)
            elif dataset_name.endswith("DIOR_test"):
                img_path = self.img_dir / (item["img"] + ".jpg")
                item["conv"] = dict(Question=item["question"], Answer=None)
            else:
                img_path = self.img_dir / item["name"]

            if valid_path(img_path):
                self.img_list.append(img_path)
                self.prompt_list.append(item["conv"])
                self.target_list.append(item["answer"])

    def post_process(self):
        for i, item in enumerate(self.prompt_list):
            if not isinstance(item, list):
                item = [item]
            first_conv = item[0]
            first_conv["Question"] = "<image>" + first_conv["Question"]
            item[0] = first_conv
            self.prompt_list[i] = item

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, idx: int) -> Dict:
        prompt = self.prompt_list[idx]

        img = self.load_image(idx)
        file_name = self.img_list[idx].name

        prompt = preprocess_multimodal(prompt, tune_im_start=self.tune_im_start)
        prompt = preprocess(prompt, self.tokenizer, has_image=True)
        prompt = prompt["input_ids"][0]

        target = self.target_list[idx]
        return img, prompt, target, file_name


class CapEvalDataset(CaptionDataset):
    def __init__(
            self,
            root: Union[Path, str] = ".data/rsicd",
            target: Union[Path, str] = None,
            transform: T.Compose = None,
    ):
        if isinstance(root, str):
            root = Path(root)

        if isinstance(target, str):
            target = Path(target)

        self.transform = transform
        self.img_dir = root
        self.json_dir = target

        self.img_list = []
        self.cap_list = []
        self.load_dataset()
        self.post_process()

        self.raw_transform = T.Compose([T.PILToTensor()])

    def load_dataset(self):
        with open(self.json_dir, "rb") as f:
            data = json.load(f)
        if "TextRS" in self.img_dir.stem:
            text_rs = data["TextRS"]
            for j in range(len(text_rs)):
                img_path = self.img_dir / (text_rs[j]["image"] + ".png")
                if valid_path(img_path):
                    self.img_list.append(self.img_dir / (text_rs[j]["image"] + ".png"))
                    self.cap_list.append(text_rs[j]["annotation"]["caption"][0])
        elif "UAVICD" in self.img_dir.stem:
            for j in range(len(data["images"])):
                img_path = self.img_dir / data["images"][j]["SubFolder"] / data["images"][j]["ImageName"]
                if valid_path(img_path):
                    self.img_list.append(
                        self.img_dir / data["images"][j]["SubFolder"] / data["images"][j]["ImageName"]
                    )
                    self.cap_list.append(data["images"][j]["Caption"])
        elif "NWPU" in self.img_dir.stem:
            for sub_folder in data.keys():
                sub_data = data[sub_folder]
                for j in range(len(sub_data)):
                    img_path = self.img_dir / sub_folder / sub_data[j]["filename"]
                    if valid_path(img_path):
                        self.img_list.append(self.img_dir / sub_folder / sub_data[j]["filename"])
                        self.cap_list.append(sub_data[j]["raw"])
        else:
            for j in range(len(data["images"])):
                img_path = self.img_dir / data["images"][j]["filename"]
                if valid_path(img_path):
                    self.img_list.append(self.img_dir / data["images"][j]["filename"])
                    self.cap_list.append(data["images"][j]["sentences"][0]["raw"])

    def __getitem__(self, idx: int) -> Dict:
        super_result = super().__getitem__(idx)
        file_name = self.img_list[idx].name
        raw_image = load_image_as_rgb(self.img_list[idx])
        raw_image = self.raw_transform(raw_image)
        super_result["filename"] = file_name
        super_result["raw_image"] = raw_image
        return super_result


class CaptionDatasetVQA(CaptionDataset):
    QUESTION_TEMPLACES = [
        "Describe the image concisely.\n<image>",
        "Provide a brief description of the given image.\n<image>",
        "Offer a succinct explanation of the picture presented.\n<image>",
        "Summarize the visual content of the image.\n<image>",
        "Give a short and clear explanation of the subsequent image.\n<image>",
        "Share a concise interpretation of the image provided.\n<image>",
        "Present a compact description of the photo’s key features.\n<image>",
        "Relay a brief, clear account of the picture shown.\n<image>",
        "Render a clear and concise summary of the photo.\n<image>",
        "Write a terse but informative summary of the picture.\n<image>",
        "Create a compact narrative representing the image presented.\n<image>",
    ]

    QUESTION2_TEMPLATE = [
        "Following this description output class token.",
        "Based on the above description, output class token",
    ]

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.tune_im_start = kwargs.pop("tune_im_start", False)
        prompt_type = kwargs.pop("prompt_type", "llava_llama_2")
        conversation_lib.default_conversation = conversation_lib.conv_templates[prompt_type]
        self.tokenizer = tokenizer

        super().__init__(**kwargs)

    def post_process(self):
        for idx, caption in enumerate(self.cap_list):
            if isinstance(caption, List):
                caption = caption[0]
                if isinstance(caption, Dict) and "<image>" in caption["Question"]:
                    if "Answer" not in caption.keys():
                        caption["Answer"] = caption["value"]
                        del caption["value"]
                        self.cap_list[idx] = [caption]
                    continue
            conv_cap_1 = dict()
            conv_cap_1["Question"] = random.choice(self.QUESTION_TEMPLACES)
            conv_cap_1["Answer"] = pre_caption(caption)

            self.cap_list[idx] = [conv_cap_1]

    def __getitem__(self, idx: int) -> Dict:
        out_dict = super().__getitem__(idx)
        out_dict["text"] = preprocess_multimodal(out_dict["text"], tune_im_start=self.tune_im_start)
        out_dict["text"] = preprocess(out_dict["text"], self.tokenizer, has_image=True)
        out_dict["text"] = dict(
            input_ids=out_dict["text"]["input_ids"][0],
            labels=out_dict["text"]["labels"][0],
        )

        return out_dict


class InstructDataset(CaptionDataset):
    def __init__(
            self,
            tokenizer: transformers.PreTrainedTokenizer,
            crop_size: int = 224,
            **kwargs,
    ):
        self.stage = int(kwargs.pop("stage", 2))
        self.geojson_priority = bool(kwargs.pop("geojson_priority", self.stage >= 3))
        # Number of prompt variants to keep for Stage-3 GeoJSON samples that
        # were not pre-baked by the build script. Earlier code emitted three
        # near-duplicate prompts sharing the same answer, which wastes training
        # compute. Override via dataset config when more diversity is desired.
        self.geojson_prompt_variants = int(kwargs.pop("geojson_prompt_variants", 1))
        # Drop samples whose target answer (after tokenisation) exceeds this
        # many tokens. Prevents truncated-answer training that systematically
        # biases the model toward the start of long FeatureCollections. 0
        # disables the filter.
        self.geojson_max_answer_tokens = int(kwargs.pop("geojson_max_answer_tokens", 0))
        self.repair_mojibake = bool(kwargs.pop("repair_mojibake", True))
        self.tune_im_start = kwargs.pop("tune_im_start", False)
        prompt_type = kwargs.pop("prompt_type", "llava_llama_2")
        conversation_lib.default_conversation = conversation_lib.conv_templates[prompt_type]
        self.tokenizer = tokenizer
        self.crop_size = crop_size
        self._geojson_filtered_oversize = 0

        super().__init__(**kwargs)

    @staticmethod
    def _resolve_image_path(img_dir: Path, item: Dict):
        if "name" in item:
            raw_name = str(item["name"])
        else:
            raw_name = item.get("filename", "")
            if isinstance(raw_name, list):
                raw_name = raw_name[0] if raw_name else ""
            raw_name = str(raw_name)
        if not raw_name:
            return None

        lower_name = raw_name.lower()
        has_img_ext = lower_name.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        candidates = [img_dir / raw_name]
        if not has_img_ext:
            for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                candidates.append(img_dir / f"{raw_name}{ext}")

        for candidate in candidates:
            if valid_path(candidate):
                return candidate
        return None

    @staticmethod
    def _pick_caption_from_feature(feature: Dict) -> str:
        props = feature.get("properties", {}) if isinstance(feature, dict) else {}
        for key in ("caption1", "caption2", "caption3"):
            value = props.get(key, "")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _infer_element_hint(text: str) -> str:
        text_lower = text.lower()
        for kw in (
            "coastline",
            "shoreline",
            "tidal flat",
            "mudflat",
            "mariculture",
            "aquaculture",
            "mangrove",
            "wind turbine",
        ):
            if kw in text_lower:
                return kw
        return "coastline"

    def _build_auto_conv_for_feature(self, feature: Dict, sample_idx: int):
        answer = self._pick_caption_from_feature(feature)
        if not answer:
            return None
        element_hint = self._infer_element_hint(answer)
        if sample_idx % 2 == 0:
            question = (
                f"Describe the image and summarize key {element_hint} related coastal elements."
            )
        else:
            question = (
                f"Extract the {element_hint} related targets from the image and give a concise summary."
            )
        return [{"Question": question, "Answer": answer}]

    def post_process(self):
        new_cap_list = []
        new_img_list = []
        for i, item in enumerate(self.cap_list):
            if not isinstance(item, list):
                item = [item]
            if len(item) == 0:
                continue
            first_conv = item[0]

            if DEFAULT_IMAGE_TOKEN not in first_conv["Question"]:
                first_conv["Question"] = "<image>" + first_conv["Question"]
                item[0] = first_conv

            # remove other DEFAULT_IMAGE_TOKEN
            length = len(item)
            for j in range(1, length):
                if DEFAULT_IMAGE_TOKEN in item[j]["Question"]:
                    item[j]["Question"] = item[j]["Question"].replace(DEFAULT_IMAGE_TOKEN, "")
                if DEFAULT_IMAGE_TOKEN in item[j]["Answer"]:
                    item[j]["Answer"] = item[j]["Answer"].replace(DEFAULT_IMAGE_TOKEN, "")

            new_cap_list.append(item)
            new_img_list.append(self.img_list[i])

        self.cap_list = new_cap_list
        self.img_list = new_img_list

    def load_physics(self, idx: int):
        """
        加载与图像空间对齐的物理真值 (TSM) 与有效域掩码 (Mask)。
        这部分数据必须在制作 CoastBench 时提前由定量遥感算法生成。
        """
        img_path = Path(self.img_list[idx])

        # 假设物理真值存储在与图像同级的 TSM 文件夹或具有特定后缀
        # 实际路径逻辑请根据你的 CoastBench 存储规范严格修改
        tsm_path = img_path.parent / (img_path.stem + "_tsm.npy")
        mask_path = img_path.parent / (img_path.stem + "_mask.npy")

        if tsm_path.exists() and mask_path.exists():
            # 加载并转换为张量，通常需要下采样到特征图的尺度 (如 F32 对应的尺度)
            # 或者在这里保持原图尺寸，在损失函数计算前进行 F.interpolate
            tsm_tensor = torch.from_numpy(np.load(tsm_path)).float()
            mask_tensor = torch.from_numpy(np.load(mask_path)).float()
            return tsm_tensor, mask_tensor
        else:
            return None, None

    def load_dataset(self):
        for i in range(len(self.img_dir)):
            with open(self.json_dir[i], "rb") as f:
                data = json.load(f)

            if isinstance(data, Dict) and "data" in data.keys():
                data = data["data"]

            dataset_name = self.json_dir[i].stem
            for item in data:
                conv_data = item.get("conv", None)
                img_path = None
                if dataset_name.endswith("RSVG"):
                    img_path = self.img_dir[i] / item["img"]
                    conv_data = dict(
                        Question=item["question"],
                        Answer=item["answer"],
                    )
                elif dataset_name.endswith("DIOR"):
                    img_path = self.img_dir[i] / (item["img"] + ".jpg")
                    conv_data = dict(
                        Question=item["question"],
                        Answer=item["answer"],
                    )
                elif "METERML" in dataset_name:
                    img_path = self.img_dir[i] / item["name"] / "naip.png"
                elif "OSM" in dataset_name:
                    img_path = self.img_dir[i] / (item["filename"] + ".jpg")
                else:
                    img_path = self._resolve_image_path(self.img_dir[i], item)
                    if conv_data is None:
                        features = item.get("features", [])
                        if isinstance(features, list) and len(features) > 0:
                            first_feature = features[0]
                            if (
                                self.stage >= 3
                                and self.geojson_priority
                                and self._is_geojson_feature(first_feature)
                            ):
                                conv_data = self._build_stage3_geojson_convs(item)
                            else:
                                conv_data = self._build_auto_conv_for_feature(
                                    first_feature, len(self.cap_list)
                                )

                if img_path is not None and valid_path(img_path) and conv_data is not None:
                    self.img_list.append(img_path)
                    if isinstance(conv_data, List) and len(conv_data) > 10:
                        conv = random.sample(conv_data, 10)
                        self.cap_list.append(conv)
                    else:
                        self.cap_list.append(conv_data)

    def load_image(self, idx: int):
        if idx >= len(self.img_list):
            x = torch.zeros(3, self.crop_size, self.crop_size)
        else:
            x = super().load_image(idx)
        return x

    def __getitem__(self, idx: int) -> Dict:
        out_dict = super().__getitem__(idx)

        # 文本特征的 Tokenize 预处理
        out_dict["text"] = preprocess_multimodal(out_dict["text"], tune_im_start=self.tune_im_start)
        out_dict["text"] = preprocess(out_dict["text"], self.tokenizer, has_image=True)
        out_dict["text"] = dict(
            input_ids=out_dict["text"]["input_ids"][0],
            labels=out_dict["text"]["labels"][0],
        )

        # 边界条件防御与物理张量透传
        if idx >= len(self.img_list):
            out_dict["valid_image"] = False
            # 对于纯文本或无效图像样本，物理先验必须严格置空
            out_dict["tsm"] = None
            out_dict["mask"] = None
        else:
            out_dict["valid_image"] = True
            # 获取物理数据并显式挂载到输出字典中
            tsm, mask = self.load_physics(idx)
            out_dict["tsm"] = tsm
            out_dict["mask"] = mask

        return out_dict


class InstructDatasetWithTaskId(InstructDataset):
    WEIGHT_DICT = {
        "GF_geojson_train": 15.0,
        "GF_landclass_train": 0.5,
        "NWPUDetail": 0.25,
        "NWPU": 0.25,
        "RSVG_DIOR": 1.0,
        "RSVG": 1.0,
        "HR": 1.0,
        "METERML": 1.0,
        "LR": 1.0,
        "RSICD": 1.0,
        "RSITMDDetail": 1.0,
        "RSITMD": 1.0,
        "UCM": 1.0,
        "fMoW": 1.0,
        "coord_transform": 1.0,
        "GF_geojson_manifest": 1.0,
        "geosignal": 15.0,
    }

    # 任务关键词映射到统一任务名（再由 TASK2ID 转为 ID）
    TASK_KEYWORDS = {
        "场景分类": [
            "classify", "classification", "分类", "识别", "recognize", "distinguish",
            "category", "label", "predict", "identify", "what type", "which class"
        ],
        "视觉问答": [
            "question", "answer", "问答", "vqa", "qa", "why", "how", "what", "where", "when"
        ],
        "视觉定位": [
            "locate", "location", "定位", "position", "where is", "bbox", "bounding box", "坐标"
        ],
        "描述": [
            "describe", "description", "描述", "explain", "caption", "summarize",
            "what do you see", "describe the", "tell me about", "visual content", "scene"
        ],
        "要素提取": [
            "extract", "extraction", "要素提取", "segment", "segmentation", "mask",
            "detect", "detection", "object", "target", "feature", "element"
        ]
    }

    # 统一要素关键词映射到 ELEMENT2ID 的标准 key
    ELEMENT_KEYWORDS = {
        "网箱养殖区": ["网箱", "cage", "cage-culture", "cage farming", "aquaculture cage"],
        "筏式养殖区": ["筏式", "raft", "raft-culture", "raft farming"],
        "赤潮": ["赤潮", "red tide", "algal bloom"],
        "浒苔": ["浒苔", "green tide", "ulva", "macroalgae"],
        "海岸线": ["海岸线", "coastline", "shoreline"],
        "风力发电机": ["风力发电机", "wind turbine", "windmill"],
        "海上钻井平台": ["海上钻井平台", "offshore platform", "oil rig"],
        "滩涂": ["滩涂", "tidal flat", "mudflat"],
        "红树林湿地": ["红树林", "mangrove", "mangrove wetland"],
        "土地覆盖": ["土地覆盖", "land cover", "land-use", "land use", "lc", "lulc"],
    }

    PHYSICAL_FIELD_ALIASES = {
        "sensor": [
            "sensor",
            "sensor_id",
            "platform",
            "satellite",
            "satellite_id",
            "instrument",
            "sat",
            "source",
        ],
        "gsd": [
            "gsd",
            "ground_sample_distance",
            "ground_sample_distance_m",
            "ground_sampling_distance",
            "spatial_resolution",
            "resolution",
            "pixel_size",
        ],
        "band": ["band", "bands", "channel", "channels", "spectral", "spectrum", "modality"],
        "time": [
            "time",
            "timestamp",
            "date",
            "acquisition_time",
            "acquisition_date",
            "datetime",
            "datetime_local",
            "temporal_info",
            "solar_term",
            "part_of_day",
        ],
    }

    def __init__(self, **kwargs):
        self.sample_weight = []
        self.task_ids = []       # 存储每个样本的任务ID
        self.category_ids = []   # 存储每个样本的地物类别ID（复用为 element_id）
        self.task_texts = []     # 存储每个样本任务文本
        self.element_texts = []  # 存储每个样本要素文本
        self.sample_phys_meta = []
        super().__init__(**kwargs)

    @staticmethod
    def _to_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            parts = [str(v).strip() for v in value if str(v).strip()]
            return ", ".join(parts)
        if isinstance(value, dict):
            parts = [f"{k}:{v}" for k, v in value.items() if v is not None and str(v).strip()]
            return ", ".join(parts)
        return str(value).strip()

    def _deep_lookup(self, data, aliases: List[str]) -> str:
        alias_set = {a.lower() for a in aliases}
        queue = [data]
        while queue:
            cur = queue.pop(0)
            if isinstance(cur, dict):
                for k, v in cur.items():
                    k_lower = str(k).lower()
                    if k_lower in alias_set:
                        text = self._to_text(v)
                        if text:
                            return text
                    if isinstance(v, (dict, list, tuple)):
                        queue.append(v)
            elif isinstance(cur, (list, tuple)):
                for v in cur:
                    if isinstance(v, (dict, list, tuple)):
                        queue.append(v)
        return ""

    @staticmethod
    def _guess_sensor_from_path(img_path: Path) -> str:
        path_lower = str(img_path).lower()
        rules = [
            ("sentinel", "Sentinel"),
            ("landsat", "Landsat"),
            ("gaofen", "GF"),
            ("worldview", "WorldView"),
            ("planet", "Planet"),
            ("jl1", "JL-1"),
            ("jilin", "JL-1"),
        ]
        for key, value in rules:
            if key in path_lower:
                return value
        return ""

    def _extract_physical_meta(self, item: Dict, dataset_name: str, img_path: Path) -> Dict[str, str]:
        meta = {"dataset": dataset_name}
        for field, aliases in self.PHYSICAL_FIELD_ALIASES.items():
            meta[field] = self._deep_lookup(item, aliases)
        if not meta["sensor"]:
            meta["sensor"] = self._guess_sensor_from_path(img_path)
        return meta

    @staticmethod
    def _resolve_image_path(img_dir: Path, item: Dict):
        if "name" in item:
            raw_name = str(item["name"])
        else:
            raw_name = item.get("filename", "")
            if isinstance(raw_name, list):
                raw_name = raw_name[0] if raw_name else ""
            raw_name = str(raw_name)
        if not raw_name:
            return None

        lower_name = raw_name.lower()
        has_img_ext = lower_name.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        candidates = [img_dir / raw_name]
        if not has_img_ext:
            for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                candidates.append(img_dir / f"{raw_name}{ext}")

        for candidate in candidates:
            if valid_path(candidate):
                return candidate
        return None

    @staticmethod
    def _pick_caption_from_feature(feature: Dict) -> str:
        props = feature.get("properties", {}) if isinstance(feature, dict) else {}
        for key in ("caption1", "caption2", "caption3"):
            value = props.get(key, "")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _infer_element_hint(self, text: str) -> str:
        text_lower = text.lower()
        for _, keywords in self.ELEMENT_KEYWORDS.items():
            for keyword in keywords:
                kw = str(keyword).strip()
                if not kw:
                    continue
                if kw.lower() in text_lower and any(ch.isalpha() for ch in kw):
                    return kw
        return "coastline"

    @staticmethod
    def _geojson_task_name() -> str:
        for task_name in TASK2ID.keys():
            if "geojson" in str(task_name).lower():
                return task_name
        return next(iter(TASK2ID.keys()))

    @staticmethod
    def _with_det_geojson_tag(prompt: str) -> str:
        prompt = str(prompt or "").strip()
        if not prompt:
            return "[DET]"
        if prompt.lower().startswith("[det]"):
            return prompt
        return f"[DET] {prompt}"

    @staticmethod
    def _is_geojson_feature(feature: Dict) -> bool:
        if not isinstance(feature, dict):
            return False
        geometry = feature.get("geometry", {})
        if not isinstance(geometry, dict):
            return False
        geo_type = str(geometry.get("type", "")).lower()
        return geo_type in {"polygon", "multipolygon"}

    @staticmethod
    def _collect_geojson_features(item: Dict) -> List[Dict]:
        if not isinstance(item, dict):
            return []
        if isinstance(item.get("features"), list):
            features = []
            for feature in item["features"]:
                if not isinstance(feature, dict):
                    continue
                geometry = feature.get("geometry", {})
                if not isinstance(geometry, dict) or "type" not in geometry:
                    continue
                properties = feature.get("properties", {})
                if not isinstance(properties, dict):
                    properties = {}
                features.append(
                    {
                        "type": "Feature",
                        "geometry": geometry,
                        "properties": properties,
                    }
                )
            return features

        geometry = item.get("geometry", {})
        properties = item.get("properties", {})
        if not isinstance(geometry, dict) or "type" not in geometry:
            return []
        if not isinstance(properties, dict):
            properties = {}
        return [
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": properties,
            }
        ]

    @classmethod
    def _build_geojson_feature_answer(cls, feature: Dict) -> str:
        features = cls._collect_geojson_features(feature)
        if not features:
            return ""
        return json.dumps(
            {
                "type": "FeatureCollection",
                "features": features,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _build_stage3_geojson_convs(self, feature: Dict) -> List[Dict]:
        full_feature_answer = self._build_geojson_feature_answer(feature)
        if not full_feature_answer:
            return []
        caption_source = feature
        if isinstance(feature, dict) and isinstance(feature.get("features"), list) and len(feature["features"]) > 0:
            caption_source = feature["features"][0]
        caption_hint = self._pick_caption_from_feature(caption_source)

        candidate_convs = [
            {
                "Question": self._with_det_geojson_tag(
                    "Extract the target features from this remote sensing image and output a valid GeoJSON FeatureCollection. Return JSON only."
                ),
                "Answer": full_feature_answer,
            },
            {
                "Question": self._with_det_geojson_tag(
                    "Generate an editable GeoJSON FeatureCollection for ArcGIS from this image. Return JSON only."
                ),
                "Answer": full_feature_answer,
            },
            {
                "Question": self._with_det_geojson_tag(
                    "Output the extracted feature information for this image in GeoJSON FeatureCollection format. Return JSON only."
                ),
                "Answer": full_feature_answer,
            },
        ]
        # Cap the number of near-duplicate prompts emitted per sample. Default
        # is 1: a single prompt avoids spending most of the training budget
        # memorising the same answer three times in a row.
        n = max(1, min(int(getattr(self, "geojson_prompt_variants", 1)), len(candidate_convs)))
        convs = candidate_convs[:n]
        if caption_hint and n >= len(candidate_convs):
            # Caption-conditioned prompt only adds value when more than one
            # variant is requested explicitly.
            convs.append(
                {
                    "Question": self._with_det_geojson_tag(
                        "Based on the following scene description, extract the target features and output GeoJSON FeatureCollection. Return JSON only.\n"
                        f"{caption_hint}"
                    ),
                    "Answer": full_feature_answer,
                }
            )
        return convs

    @staticmethod
    def _is_geojson_query_text(text: str) -> bool:
        text_lower = str(text).lower()
        geojson_keywords = (
            "[geojson]",
            "geojson",
            "featurecollection",
            "feature collection",
            "feature object",
            "polygon json",
            "geometry json",
            "output json boundary",
            "return geojson",
        )
        return any(keyword in text_lower for keyword in geojson_keywords)

    @staticmethod
    def _looks_like_geojson_answer(text: str) -> bool:
        text = str(text or "").strip().lower()
        return (
            text.startswith("{")
            and "featurecollection" in text
            and "\"features\"" in text
        )

    def _collapse_redundant_geojson_turns(self, item: List[Dict], sample_idx: int) -> List[Dict]:
        if not isinstance(item, list) or len(item) <= 1:
            return item

        questions = [str(conv.get("Question", "")) for conv in item if isinstance(conv, dict)]
        answers = [str(conv.get("Answer", "")) for conv in item if isinstance(conv, dict)]
        if len(questions) != len(item) or len(answers) != len(item):
            return item
        if not answers:
            return item

        first_answer = answers[0].strip()
        if not first_answer or not self._looks_like_geojson_answer(first_answer):
            return item
        if any(answer.strip() != first_answer for answer in answers[1:]):
            return item
        if not all(
            ("[det]" in question.lower()) or self._is_geojson_query_text(question)
            for question in questions
        ):
            return item

        keep_idx = int(sample_idx) % len(item)
        return [item[keep_idx]]

    def _build_auto_conv_for_feature(self, feature: Dict, sample_idx: int):
        if self.stage >= 3 and self.geojson_priority and self._is_geojson_feature(feature):
            geojson_convs = self._build_stage3_geojson_convs(feature)
            if geojson_convs:
                return geojson_convs

        answer = self._pick_caption_from_feature(feature)
        if not answer:
            return None
        element_hint = self._infer_element_hint(answer)
        if sample_idx % 2 == 0:
            question = (
                f"Describe the image and summarize key {element_hint} related coastal elements."
            )
        else:
            question = (
                f"Extract the {element_hint} related targets from the image and give a concise summary."
            )
        return [{"Question": question, "Answer": answer}]

    @staticmethod
    def _normalize_time_str(time_str: str) -> str:
        if not time_str:
            return ""
        return time_str.replace("T", " ").replace("Z", "").strip()

    def _build_physical_prompt(self, meta: Dict[str, str]) -> str:
        if meta is None:
            return ""
        parts = []
        if meta.get("dataset"):
            parts.append(f"[Dataset: {meta['dataset']}]")
        if meta.get("sensor"):
            parts.append(f"[Sensor: {meta['sensor']}]")
        if meta.get("gsd"):
            parts.append(f"[GSD: {meta['gsd']}]")
        if meta.get("band"):
            parts.append(f"[Band: {meta['band']}]")
        norm_time = self._normalize_time_str(meta.get("time", ""))
        if norm_time:
            parts.append(f"[Time: {norm_time}]")
        return " ".join(parts)

    @staticmethod
    def _default_task_id() -> int:
        return TASK2ID.get("描述", 0)

    @staticmethod
    def _default_element_id() -> int:
        return ELEMENT2ID.get("无", 0)

    @staticmethod
    def _default_task_text() -> str:
        return "描述"

    @staticmethod
    def _default_element_text() -> str:
        return "无"

    @staticmethod
    def _task_name_from_id(task_id: int) -> str:
        for name, idx in TASK2ID.items():
            if idx == task_id:
                return name
        return next(iter(TASK2ID.keys()))

    @staticmethod
    def _element_name_from_id(element_id: int) -> str:
        for name, idx in ELEMENT2ID.items():
            if idx == element_id:
                return name
        return next(iter(ELEMENT2ID.keys()))

    @staticmethod
    def _extract_free_element_text(text: str) -> str:
        if not isinstance(text, str) or not text:
            return ""
        text_lower = re.sub(r"\[[a-z0-9_]+\]", " ", text.lower())
        patterns = [
            r"(?:find|locate|detect|identify|segment|extract)\s+(?:a|an|the)?\s*([a-z][a-z0-9 -]{2,64})",
            r"(?:about|of|for)\s+(?:the|a|an)?\s*([a-z][a-z0-9 -]{2,64})",
        ]
        stop_terms = {
            "image",
            "scene",
            "picture",
            "photo",
            "target",
            "object",
            "area",
            "region",
            "class",
            "category",
            "dimensions",
            "following object",
        }
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if not match:
                continue
            candidate = match.group(1).strip(" .,;:!?\"'()[]{}")
            candidate = re.split(r",|\.|;|\?|!|\band\b|\bwith\b|\bthat\b|\bwhich\b", candidate)[0].strip()
            words = [w for w in candidate.split() if w]
            if not words:
                continue
            candidate = " ".join(words[:4])
            if candidate in stop_terms or len(candidate) < 3:
                continue
            return candidate
        return ""

    @staticmethod
    def _first_answer_text(conv_data) -> str:
        if isinstance(conv_data, dict):
            return str(conv_data.get("Answer", conv_data.get("answer", "")))
        if isinstance(conv_data, list) and conv_data:
            first = conv_data[0]
            if isinstance(first, dict):
                return str(first.get("Answer", first.get("answer", "")))
        return ""

    def _is_geojson_priority_sample(self, item, conv_data) -> bool:
        if not (self.stage >= 3 and self.geojson_priority):
            return False
        if isinstance(item, dict):
            if str(item.get("coord_encoding", "")).lower() in {"normalized", "absolute"}:
                return True
        first_answer = self._first_answer_text(conv_data)
        return self._looks_like_geojson_answer(first_answer)

    def detect_task_text_from_text(self, text: str) -> str:
        text_lower = str(text).lower()
        geojson_task_name = self._geojson_task_name()
        if self.stage >= 3 and self.geojson_priority and "[det]" in text_lower:
            return geojson_task_name
        if self._is_geojson_query_text(text_lower):
            return geojson_task_name
        if "[gj]" in text_lower:
            return geojson_task_name

        tag_to_task_id = [
            ("[cls]", 0),
            ("[vqa]", 1),
            ("[qa]", 1),
            ("[vg]", 2),
            ("[loc]", 2),
            ("[cap]", 3),
            ("[caption]", 3),
            ("[det]", 4),
            ("[seg]", 4),
        ]
        for tag, task_id in tag_to_task_id:
            if tag in text_lower:
                return self._task_name_from_id(task_id)
        if "rural" in text_lower and "urban" in text_lower:
            return "场景分类"
        for task_name in TASK2ID.keys():
            if task_name.lower() in text_lower:
                return task_name
        for task_type, keywords in self.TASK_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return task_type
        return self._default_task_text()

    def detect_element_text_from_text(self, text: str) -> str:
        text_lower = str(text).lower()
        if "[cls]" in text_lower:
            # Stage-2 classification samples usually contain broad land-cover classes.
            return self._element_name_from_id(10)
        if "rural" in text_lower and "urban" in text_lower:
            return "土地覆盖" if "土地覆盖" in ELEMENT2ID else self._element_name_from_id(10)
        for element_name in ELEMENT2ID.keys():
            if element_name == "无":
                continue
            if element_name.lower() in text_lower:
                return element_name
        for category, keywords in self.ELEMENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category
        free_element = self._extract_free_element_text(text)
        if free_element:
            return free_element
        return self._default_element_text()

    def detect_task_from_text(self, text: str) -> int:
        """
        根据输入文本检测任务类型
        Args:
            text: 输入文本
        Returns:
            task_id: 任务ID
        """
        task_text = self.detect_task_text_from_text(text)
        return TASK2ID.get(task_text, self._default_task_id())

    def detect_category_from_text(self, text: str) -> int:
        """
        根据输入文本检测地物类别
        Args:
            text: 输入文本
        Returns:
            category_id: 地物类别ID
        """
        element_text = self.detect_element_text_from_text(text)
        return ELEMENT2ID.get(element_text, self._default_element_id())

    def post_process(self):
        for i, item in enumerate(self.cap_list):
            if not isinstance(item, list):
                item = [item]
            if len(item) == 0:
                continue

            item = self._collapse_redundant_geojson_turns(item, sample_idx=i)

            first_conv = item[0]
            first_question = str(first_conv.get("Question", ""))
            if DEFAULT_IMAGE_TOKEN not in first_question:
                first_conv["Question"] = DEFAULT_IMAGE_TOKEN + first_question
            item[0] = first_conv

            # Keep exactly one image token per sample.
            for j in range(1, len(item)):
                if "Question" in item[j]:
                    item[j]["Question"] = str(item[j]["Question"]).replace(DEFAULT_IMAGE_TOKEN, "")
                if "Answer" in item[j]:
                    item[j]["Answer"] = str(item[j]["Answer"]).replace(DEFAULT_IMAGE_TOKEN, "")

            self.cap_list[i] = item

        self.txt_json_dir = []
        for file in self.root.glob("*text.json"):
            if file not in self.json_dir:
                self.txt_json_dir.append(file)

        for dir in self.txt_json_dir:
            if "geosignal" in dir.stem:
                with open(dir, "rb") as f:
                    data = json.load(f)
                for item in data:
                    question = item.get("instruction", "") + item.get("input", "")
                    conv = [
                        {
                            "Question": question,
                            "Answer": item["output"],
                        }
                    ]
                    self.cap_list.append(conv)
                    self.sample_weight.append(self.WEIGHT_DICT["geosignal"])
                    task_text = self.detect_task_text_from_text(question)
                    detect_text = f"{question} {item.get('output', '')}"
                    element_text = self.detect_element_text_from_text(detect_text)
                    self.task_texts.append(task_text)
                    self.element_texts.append(element_text)
                    self.task_ids.append(TASK2ID.get(task_text, self._default_task_id()))
                    self.category_ids.append(ELEMENT2ID.get(element_text, self._default_element_id()))
                    self.sample_phys_meta.append(
                        {
                            "dataset": "geosignal",
                            "sensor": "",
                            "gsd": "",
                            "band": "",
                            "time": "",
                        }
                    )

    def _is_answer_within_token_budget(self, answer_text: str) -> bool:
        """Return True iff the tokenized answer fits inside the configured
        per-sample budget. Always True when no tokenizer or budget is set.
        """
        budget = int(getattr(self, "geojson_max_answer_tokens", 0) or 0)
        if budget <= 0 or self.tokenizer is None:
            return True
        try:
            ids = self.tokenizer(answer_text, add_special_tokens=False).get("input_ids", [])
        except Exception:
            return True
        return len(ids) <= budget

    def load_dataset(self):
        for i in range(len(self.img_dir)):
            with open(self.json_dir[i], "rb") as f:
                data = json.load(f)

            if isinstance(data, Dict) and "data" in data.keys():
                data = data["data"]

            if getattr(self, "repair_mojibake", False):
                data = repair_mojibake_in_obj(data)

            dataset_name = self.json_dir[i].stem
            for item in data:
                conv_data = item.get("conv", None)
                img_path = None

                if dataset_name.endswith("RSVG"):
                    img_path = self.img_dir[i] / item["img"]
                    conv_data = dict(
                        Question=item["question"],
                        Answer=item["answer"],
                    )
                elif dataset_name.endswith("DIOR"):
                    img_path = self.img_dir[i] / (item["img"] + ".jpg")
                    conv_data = dict(
                        Question=item["question"],
                        Answer=item["answer"],
                    )
                elif "METERML" in dataset_name:
                    img_path = self.img_dir[i] / item["name"] / "naip.png"
                elif "OSM" in dataset_name:
                    img_path = self.img_dir[i] / (item["filename"] + ".jpg")
                else:
                    img_path = self._resolve_image_path(self.img_dir[i], item)
                    if conv_data is None:
                        features = item.get("features", [])
                        if isinstance(features, list) and len(features) > 0:
                            conv_data = self._build_auto_conv_for_feature(
                                features[0], len(self.cap_list)
                            )

                if img_path is not None and valid_path(img_path) and conv_data is not None:
                    # Drop samples whose Stage-3 GeoJSON answer would be
                    # truncated by ``model_max_length``. Training on truncated
                    # answers teaches the model to stop mid-FeatureCollection.
                    if self._is_geojson_priority_sample(item, conv_data):
                        first_answer = self._first_answer_text(conv_data)
                        if first_answer and not self._is_answer_within_token_budget(first_answer):
                            self._geojson_filtered_oversize += 1
                            continue

                    self.img_list.append(img_path)
                    if isinstance(conv_data, List) and len(conv_data) > 10:
                        # Keep the canonical first turn. Stage2 BEN-style evals use
                        # the first QA turn per image, so pure random truncation
                        # under-trains the exact question used by target metrics.
                        first_turn = conv_data[0]
                        sampled_rest = random.sample(conv_data[1:], 9)
                        conv = [first_turn] + sampled_rest
                        self.cap_list.append(conv)
                    else:
                        self.cap_list.append(conv_data)

                    # 检测任务ID和地物类别ID
                    conv_items = self.cap_list[-1]
                    if isinstance(conv_items, Dict):
                        conv_items = [conv_items]
                    for conv_item in conv_items:
                        if "Question" in conv_item:
                            question = str(conv_item.get("Question", ""))
                            answer = str(conv_item.get("Answer", conv_item.get("answer", "")))
                            detect_text = f"{question} {answer}".strip()
                            task_text = self.detect_task_text_from_text(question)
                            element_text = self.detect_element_text_from_text(detect_text)
                            self.task_texts.append(task_text)
                            self.element_texts.append(element_text)
                            self.task_ids.append(TASK2ID.get(task_text, self._default_task_id()))
                            self.category_ids.append(ELEMENT2ID.get(element_text, self._default_element_id()))
                            break
                    else:
                        # 如果没有Question，默认使用"描述"任务和"无"类别
                        self.task_texts.append(self._default_task_text())
                        self.element_texts.append(self._default_element_text())
                        self.task_ids.append(self._default_task_id())
                        self.category_ids.append(self._default_element_id())

                    meta_source = item
                    features = item.get("features", [])
                    if isinstance(features, list) and len(features) > 0:
                        first_feature = features[0]
                        if isinstance(first_feature, dict):
                            props = first_feature.get("properties", {})
                            if isinstance(props, dict):
                                meta_source = dict(item)
                                meta_source["properties"] = props
                    self.sample_phys_meta.append(
                        self._extract_physical_meta(meta_source, dataset_name, img_path)
                    )

                    base_weight = 0.5
                    for name, weight in self.WEIGHT_DICT.items():
                        if name in dataset_name:
                            base_weight = float(weight)
                            break

                    if (
                        self.stage >= 3
                        and self.geojson_priority
                        and len(self.task_texts) > 0
                        and self.task_texts[-1] == self._geojson_task_name()
                    ):
                        base_weight = max(base_weight, 1.5)

                    self.sample_weight.append(base_weight)

    def __getitem__(self, idx: int) -> Dict:
        out_dict = super().__getitem__(idx)

        # 添加task_id和category_id
        if idx < len(self.task_ids):
            out_dict["task_id"] = self.task_ids[idx]
        else:
            out_dict["task_id"] = self._default_task_id()

        if idx < len(self.category_ids):
            out_dict["category_id"] = self.category_ids[idx]
        else:
            out_dict["category_id"] = self._default_element_id()

        out_dict["task_text"] = self.task_texts[idx] if idx < len(self.task_texts) else self._default_task_text()
        out_dict["element_text"] = self.element_texts[idx] if idx < len(self.element_texts) else self._default_element_text()

        meta = self.sample_phys_meta[idx] if idx < len(self.sample_phys_meta) else None
        out_dict["physical_prompt"] = self._build_physical_prompt(meta)

        return out_dict


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logger.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


def byte_decode(x):
    return x.decode("utf-8")


def RS5MDataset(
        root: Union[Path, str] = ".data/rsicd",
        transform: T.Compose = None,
        tokenizer: transformers.PreTrainedTokenizer = None,
        **kwargs,
):
    tune_im_start = kwargs.pop("tune_im_start", False)
    prompt_type = kwargs.pop("prompt_type", "llava_llama_2")
    conversation_lib.default_conversation = conversation_lib.conv_templates[prompt_type]
    url = os.path.join(root, "{pub11,rs3}-train-{0000..0031}.tar")

    QUESTION_TEMPLACES = [
        "Describe the image concisely.\n<image>",
        "Provide a brief description of the given image.\n<image>",
        "Offer a succinct explanation of the picture presented.\n<image>",
        "Summarize the visual content of the image.\n<image>",
        "Give a short and clear explanation of the subsequent image.\n<image>",
        "Share a concise interpretation of the image provided.\n<image>",
        "Present a compact description of the photo’s key features.\n<image>",
        "Relay a brief, clear account of the picture shown.\n<image>",
        "Render a clear and concise summary of the photo.\n<image>",
        "Write a terse but informative summary of the picture.\n<image>",
        "Create a compact narrative representing the image presented.\n<image>",
    ]

    def get_text(x):
        x = byte_decode(x)
        conv_cap = dict()
        conv_cap["Question"] = random.choice(QUESTION_TEMPLACES)
        conv_cap["Answer"] = pre_caption(x)

        conv_cap = preprocess_multimodal([conv_cap], tune_im_start=tune_im_start)
        conv_cap = preprocess(conv_cap, tokenizer, has_image=True)
        conv_cap = dict(input_ids=conv_cap["input_ids"][0], labels=conv_cap["labels"][0])
        return conv_cap

    def my_decoder(key, value):
        if key.endswith(".img_content"):
            assert isinstance(value, bytes)
            value = Image.open(io.BytesIO(value))
            if transform is not None:
                if isinstance(transform, CLIPImageProcessor):
                    value = transform(value, return_tensors="pt").pixel_values.squeeze()
                else:
                    value = transform(value)
        elif key.endswith(".img_name"):
            value = byte_decode(value)
        elif key.endswith(".caption"):
            value = get_text(value)
        return value

    def convert_format(sample_tuple):
        rgb, text = sample_tuple["img_content"], sample_tuple["caption"]
        return dict(rgb=rgb, text=text)

    shared_epoch = SharedEpoch(epoch=0)  # create a shared epoch store to sync epoch to dataloader worker proc
    pipeline = [wds.SimpleShardList(url)]
    pipeline.extend(
        [
            detshuffle2(
                bufsize=_SHARD_SHUFFLE_SIZE,
                initial=_SHARD_SHUFFLE_INITIAL,
                seed=322,
                epoch=shared_epoch,
            ),
            wds.split_by_node,
            wds.split_by_worker,
        ]
    )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.decode(my_decoder),
            wds.map(convert_format),
        ]
    )

    return pipeline, url


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    physical_prompt_max_len: int = 64
    task_text_max_len: int = 16
    element_text_max_len: int = 16

    def _resolve_max_len(self, target_len: int) -> int:
        tokenizer_max_len = getattr(self.tokenizer, "model_max_length", target_len)
        if not isinstance(tokenizer_max_len, int) or tokenizer_max_len <= 0:
            tokenizer_max_len = target_len
        tokenizer_max_len = min(tokenizer_max_len, 4096)
        return max(1, min(int(target_len), int(tokenizer_max_len)))

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance["text"][key] for instance in instances] for key in ("input_ids", "labels")
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "rgb" in instances[0]:
            images = [instance["rgb"] for instance in instances]
            if not isinstance(images[0], Image.Image) and all(
                    x is not None and x.shape == images[0].shape for x in images
            ):
                batch["rgb"] = torch.stack(images)
            else:
                batch["rgb"] = images

        if any("multiband" in instance for instance in instances):
            multiband, valid_multiband = stack_optional_multiband(
                [instance.get("multiband", None) for instance in instances]
            )
            batch["multiband"] = multiband
            batch["valid_multiband"] = valid_multiband

        if "valid_image" in instances[0]:
            batch["valid_image"] = torch.tensor([instance["valid_image"] for instance in instances])

        if "tsm" in instances[0]:
            tsm_items = [instance.get("tsm", None) for instance in instances]
            mask_items = [instance.get("mask", None) for instance in instances]
            ref_tsm = next((x for x in tsm_items if torch.is_tensor(x)), None)
            if ref_tsm is not None:
                batch_tsm = []
                batch_mask = []
                valid_physics = []
                for tsm_i, mask_i in zip(tsm_items, mask_items):
                    if torch.is_tensor(tsm_i):
                        cur_tsm = tsm_i
                        cur_mask = mask_i if torch.is_tensor(mask_i) else torch.ones_like(tsm_i)
                        valid_physics.append(True)
                    else:
                        cur_tsm = torch.zeros_like(ref_tsm)
                        cur_mask = torch.zeros_like(ref_tsm)
                        valid_physics.append(False)
                    batch_tsm.append(cur_tsm)
                    batch_mask.append(cur_mask)
                batch["tsm"] = torch.stack(batch_tsm).unsqueeze(1)   # [B,1,H,W]
                batch["mask"] = torch.stack(batch_mask).unsqueeze(1)  # [B,1,H,W]
                batch["valid_physics"] = torch.tensor(valid_physics, dtype=torch.bool)
            else:
                batch["tsm"] = None
                batch["mask"] = None
                batch["valid_physics"] = torch.zeros(len(instances), dtype=torch.bool)
        else:
            batch["tsm"] = None
            batch["mask"] = None
            batch["valid_physics"] = torch.zeros(len(instances), dtype=torch.bool)

        # 添加task_ids和category_ids
        if "task_id" in instances[0]:
            task_ids = [instance["task_id"] for instance in instances]
            batch["task_ids"] = torch.tensor(task_ids, dtype=torch.long)

        if "category_id" in instances[0]:
            category_ids = [instance["category_id"] for instance in instances]
            batch["category_ids"] = torch.tensor(category_ids, dtype=torch.long)

        # 物理提示文本 -> token ids（在 collate 中统一 pad）
        if "physical_prompt" in instances[0]:
            phys_texts = [instance.get("physical_prompt", "") for instance in instances]
            phys_max_len = self._resolve_max_len(self.physical_prompt_max_len)
            phys_tokens = self.tokenizer(
                phys_texts,
                padding="max_length",
                truncation=True,
                max_length=phys_max_len,
                return_tensors="pt",
            )
            batch["physical_prompt_ids"] = phys_tokens.input_ids
            batch["physical_prompt_attention_mask"] = phys_tokens.attention_mask
        else:
            batch["physical_prompt_ids"] = None
            batch["physical_prompt_attention_mask"] = None

        if "task_text" in instances[0]:
            task_texts = [instance.get("task_text", "描述") for instance in instances]
            task_tokens = self.tokenizer(
                task_texts,
                padding="max_length",
                truncation=True,
                max_length=self._resolve_max_len(self.task_text_max_len),
                return_tensors="pt",
            )
            batch["task_text_ids"] = task_tokens.input_ids
            batch["task_text_attention_mask"] = task_tokens.attention_mask
        else:
            batch["task_text_ids"] = None
            batch["task_text_attention_mask"] = None

        if "element_text" in instances[0]:
            element_texts = [instance.get("element_text", "无") for instance in instances]
            element_tokens = self.tokenizer(
                element_texts,
                padding="max_length",
                truncation=True,
                max_length=self._resolve_max_len(self.element_text_max_len),
                return_tensors="pt",
            )
            batch["element_text_ids"] = element_tokens.input_ids
            batch["element_text_attention_mask"] = element_tokens.attention_mask
        else:
            batch["element_text_ids"] = None
            batch["element_text_attention_mask"] = None

        return batch


@dataclass
class DataCollatorForVGSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Tuple]) -> Tuple:
        input_ids = tuple([instance[1] for instance in instances])
        lengths = [len(ids) for ids in input_ids]
        max_length = max(lengths)

        def left_pad_sequences(sequences, desired_length, padding_value):
            """
            Pad each sequence in a tuple to the desired length with the specified padding value on the left.

            :param sequences: A tuple of sequences (e.g., lists, tuples).
            :param desired_length: The length to which each sequence will be padded.
            :param padding_value: The value used for padding.
            :return: A new tuple with padded sequences.
            """
            padded_sequences = tuple(
                [padding_value] * (desired_length - len(seq)) + list(seq) for seq in sequences
            )
            return padded_sequences

        input_ids = left_pad_sequences(input_ids, max_length, self.tokenizer.pad_token_id)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        images = [instance[0] for instance in instances]
        if not isinstance(images[0], Image.Image) and all(
                x is not None and x.shape == images[0].shape for x in images
        ):
            images = torch.stack(images)
        else:
            images = images

        targets = [instance[2] for instance in instances]
        filename = [instance[3] for instance in instances]

        return images, input_ids, targets, filename, attention_mask


def preprocess_multimodal(
        sources: List[Dict[str, str]],
        tune_im_start: bool = False,
) -> List[Dict[str, str]]:
    if not isinstance(sources, list):
        sources = [sources]
    for idx, source in enumerate(sources):
        for key, value in source.items():
            if value is not None and DEFAULT_IMAGE_TOKEN in value:
                value = value.replace(DEFAULT_IMAGE_TOKEN, "").strip()
                value = DEFAULT_IMAGE_TOKEN + "\n" + value
                value = value.strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    value = value.replace(
                        DEFAULT_IMAGE_TOKEN,
                        "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                    )
                replace_token = DEFAULT_IMAGE_TOKEN
                if tune_im_start:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                value = value.replace(DEFAULT_IMAGE_TOKEN, replace_token)
                source[key] = value
        sources[idx] = source

    return sources


def preprocess_llama_2(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"Question": conv.roles[0], "Answer": conv.roles[1], "value": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        for j, key in enumerate(source):
            role = roles[key]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, source[key])
    conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.shape[0])

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer_image_token(rou, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
        sources: Sequence[Dict],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source["Question"]
        source["Question"] = DEFAULT_IMAGE_TOKEN
        conversation = source["Question"] + source["Answer"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source["Question"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"Question": conv.roles[0], "Answer": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        for j, key in enumerate(source):
            role = roles[key]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, source[key])
    conversations.append(conv.get_prompt())

    # Tokenize conversations··
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.shape[0])

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                # print(
                # f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                # f" (ignored)"
                # )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
) -> Dict:
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    raise ValueError(f"Unsupported separator style: {conversation_lib.default_conversation.sep_style}")


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    max_len = getattr(tokenizer, "model_max_length", None)
    use_truncation = isinstance(max_len, int) and 0 < max_len < 10**6
    tokenize_kwargs = {}
    if use_truncation:
        tokenize_kwargs.update(
            dict(
                truncation=True,
                max_length=int(max_len),
                verbose=False,
            )
        )
    prompt_chunks = [tokenizer(chunk, **tokenize_kwargs).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if use_truncation and len(input_ids) > int(max_len):
        input_ids = input_ids[: int(max_len)]

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids
