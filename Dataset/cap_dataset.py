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
from typing import Dict, List, Sequence, Tuple, Union

import torch
import torchvision.transforms as T
import transformers
import webdataset as wds
from PIL import Image
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
import torch_npu

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000
logger = logging.getLogger("train")


def valid_path(path: Union[Path, str]) -> bool:
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        return False
    return True


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
    ):
        if isinstance(root, str):
            root = Path(root)

        self.root = root
        self.transform = transform
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
        x = Image.open(self.img_list[idx]).convert("RGB")
        if self.transform is not None:
            if isinstance(self.transform, CLIPImageProcessor):
                x = self.transform(x, return_tensors="pt").pixel_values.squeeze()
            else:
                x = self.transform(x)

        return x

    def __getitem__(self, idx: int) -> Dict:
        captions = self.cap_list[idx]
        if not isinstance(captions, list):
            captions = pre_caption(captions)

        x = self.load_image(idx)
        tsm, mask = self.load_physics(idx)  # 新增物理数据加载
        return dict(rgb=x, text=captions, tsm=tsm, mask=mask)


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
        raw_image = Image.open(self.img_list[idx]).convert("RGB")
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
        self.tune_im_start = kwargs.pop("tune_im_start", False)
        prompt_type = kwargs.pop("prompt_type", "llava_llama_2")
        conversation_lib.default_conversation = conversation_lib.conv_templates[prompt_type]
        self.tokenizer = tokenizer
        self.crop_size = crop_size

        super().__init__(**kwargs)

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
                if dataset_name.endswith("RSVG"):
                    img_path = self.img_dir[i] / item["img"]
                    item["conv"] = dict(Question=item["question"], Answer=item["answer"])
                elif dataset_name.endswith("DIOR"):
                    img_path = self.img_dir[i] / (item["img"] + ".jpg")
                    item["conv"] = dict(Question=item["question"], Answer=item["answer"])
                elif "METERML" in dataset_name:
                    img_path = self.img_dir[i] / item["name"] / "naip.png"
                elif "OSM" in dataset_name:
                    img_path = self.img_dir[i] / (item["filename"] + ".jpg")
                else:
                    if "name" in item.keys():
                        img_path = self.img_dir[i] / item["name"]
                    else:
                        file_name = item["filename"]
                        if isinstance(file_name, list):
                            file_name = file_name[0]
                        img_path = self.img_dir[i] / file_name

                if valid_path(img_path):
                    self.img_list.append(img_path)
                    if isinstance(item["conv"], List) and len(item["conv"]) > 10:
                        conv = random.sample(item["conv"], 10)
                        self.cap_list.append(conv)
                    else:
                        self.cap_list.append(item["conv"])

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
        "OSM": 0.6,
        "LLAVA": 1.0,
        "geosignal": 0.50,
        "RSITMD": 0.6,
        "NWPU": 0.6,
        "DOTA": 0.9,
        "FAST": 1.0,
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
        "sensor": ["sensor", "platform", "satellite", "instrument", "sat", "source"],
        "gsd": ["gsd", "ground_sample_distance", "ground_sampling_distance", "spatial_resolution", "resolution", "pixel_size"],
        "band": ["band", "bands", "channel", "channels", "spectral", "spectrum", "modality"],
        "time": ["time", "timestamp", "date", "acquisition_time", "acquisition_date", "datetime"],
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

    def detect_task_text_from_text(self, text: str) -> str:
        text_lower = text.lower()
        for task_name in TASK2ID.keys():
            if task_name.lower() in text_lower:
                return task_name
        for task_type, keywords in self.TASK_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return task_type
        return self._default_task_text()

    def detect_element_text_from_text(self, text: str) -> str:
        text_lower = text.lower()
        for element_name in ELEMENT2ID.keys():
            if element_name == "无":
                continue
            if element_name.lower() in text_lower:
                return element_name
        for category, keywords in self.ELEMENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category
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
            first_conv = item[0]
            if DEFAULT_IMAGE_TOKEN in first_conv["Question"]:
                continue
            first_conv["Question"] = "<image>" + first_conv["Question"]
            item[0] = first_conv
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
                    element_text = self.detect_element_text_from_text(question)
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

    def load_dataset(self):
        for i in range(len(self.img_dir)):
            with open(self.json_dir[i], "rb") as f:
                data = json.load(f)

            if isinstance(data, Dict) and "data" in data.keys():
                data = data["data"]

            dataset_name = self.json_dir[i].stem
            for item in data:
                if dataset_name.endswith("RSVG"):
                    img_path = self.img_dir[i] / item["img"]
                    item["conv"] = dict(Question=item["question"], Answer=item["answer"])
                elif dataset_name.endswith("DIOR"):
                    img_path = self.img_dir[i] / (item["img"] + ".jpg")
                    item["conv"] = dict(Question=item["question"], Answer=item["answer"])
                elif "METERML" in dataset_name:
                    img_path = self.img_dir[i] / item["name"] / "naip.png"
                elif "OSM" in dataset_name:
                    img_path = self.img_dir[i] / (item["filename"] + ".jpg")
                else:
                    if "name" in item.keys():
                        img_path = self.img_dir[i] / item["name"]
                    else:
                        file_name = item["filename"]
                        if isinstance(file_name, list):
                            file_name = file_name[0]
                        img_path = self.img_dir[i] / file_name

                if valid_path(img_path):
                    self.img_list.append(img_path)
                    if isinstance(item["conv"], List) and len(item["conv"]) > 10:
                        conv = random.sample(item["conv"], 10)
                        self.cap_list.append(conv)
                    else:
                        self.cap_list.append(item["conv"])

                    # 检测任务ID和地物类别ID
                    conv_items = self.cap_list[-1]
                    if isinstance(conv_items, Dict):
                        conv_items = [conv_items]
                    for conv_item in conv_items:
                        if "Question" in conv_item:
                            question = conv_item["Question"]
                            task_text = self.detect_task_text_from_text(question)
                            element_text = self.detect_element_text_from_text(question)
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

                    self.sample_phys_meta.append(self._extract_physical_meta(item, dataset_name, img_path))

                    process_flag = False
                    for name, weight in self.WEIGHT_DICT.items():
                        if name in dataset_name:
                            self.sample_weight.append(weight)
                            process_flag = True
                            break

                    if not process_flag:
                        self.sample_weight.append(0.5)

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
    rrent_sample = None
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
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

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
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

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
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids
