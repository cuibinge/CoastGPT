import json
import os
from dataclasses import dataclass
from glob import glob
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
import transformers
from PIL import Image
from transformers import CLIPImageProcessor

from . import conversation as conversation_lib
from .cap_dataset import preprocess, preprocess_multimodal


class Compose(T.Compose):
    """
    自定义的 Compose 类，用于处理输入为列表的情况。
    该类继承自 torchvision.transforms.Compose，
    可以对列表中的每个元素依次应用指定的变换。
    """

    def __init__(self, transforms: Sequence[Callable]):
        # 初始化变换列表
        self.transforms = transforms

    def __call__(self, x: Union[Any, Sequence]):
        # 如果输入是一个序列（如列表）
        if isinstance(x, Sequence):
            # 对序列中的每个元素依次应用变换
            for t in self.transforms:
                x = [t(i) for i in x]
        else:
            # 如果输入不是序列，直接应用变换
            for t in self.transforms:
                x = t(x)
        return x


class ToTensor(object):
    """
    自定义的 ToTensor 操作，不会进行最小 - 最大归一化。
    可以处理输入为 uint16 类型的图像，并根据需要调整维度顺序。
    """

    def __init__(self, permute_dims: bool = True):
        # 是否调整维度顺序的标志
        self.permute_dims = permute_dims

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        # 如果图像数据类型为 uint16，转换为 int32
        if x.dtype == "uint16":
            x = x.astype("int32")

        # 如果输入是 numpy 数组，转换为 torch.Tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # 如果输入是二维数组
        if x.ndim == 2:
            if self.permute_dims:
                # 添加一个维度，转换为 (H, W, 1)
                x = x[:, :, None]
            else:
                # 添加一个维度，转换为 (1, H, W)
                x = x[None, :, :]

        # 处理四通道图片
        if x.ndim == 3 and x.shape[2] == 4:
            if self.permute_dims:
                # 调整维度顺序为 (C, H, W)
                x = x.permute((2, 0, 1)).contiguous()

        # 将 (H, W, C) 转换为 (C, H, W)
        if self.permute_dims:
            if x.ndim == 4:
                # 四维数据，调整维度顺序为 (N, C, H, W)
                x = x.permute((0, 3, 1, 2)).contiguous()
            elif x.ndim == 3 and x.shape[2] != 4:
                # 三维数据，调整维度顺序为 (C, H, W)
                x = x.permute((2, 0, 1)).contiguous()

        return x


def sort(x):
    """
    对文件路径进行排序的辅助函数。
    提取文件名中的数字部分，用于排序。
    """
    x = os.path.basename(x)
    x = os.path.splitext(x)[0]
    return int(x)


class RSVQA(torch.utils.data.Dataset):
    """
    基础的 RSVQA 数据集类。
    用于处理遥感图像的视觉问答任务。
    """

    # 数据集的划分，包括训练集、验证集和测试集
    splits = ["train", "val", "test"]
    # 数据前缀
    prefix = ""

    def __init__(
        self,
        root: str = "",
        image_root: str = None,
        split: str = "train",
        image_transform: Compose = Compose([ToTensor()]),
        text_transform: Compose = Compose(
            [],
        ),
        token_prefix: str = "",
        tokenizer: Callable = None,
        **kwargs,
    ):
        # 确保划分类型在允许的范围内
        assert split in self.splits
        # 获取提示类型，默认为 "llava_llama_2"
        prompt_type = kwargs.pop("prompt_type", "llava_llama_2")
        # 设置默认的对话模板
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            prompt_type
        ]

        # 数据集根目录
        self.root = root
        # 数据集划分类型
        self.split = split
        # 图像根目录
        self.image_root = image_root
        # 图像变换
        self.image_transform = image_transform
        # 文本变换
        self.text_transform = text_transform
        # 拼接图像根目录
        self.image_root = os.path.join(root, self.image_root)
        # 标记前缀
        self.token_prefix = token_prefix
        # 是否调整图像起始标记的标志
        self.tune_im_start = kwargs.pop("tune_im_start", False)
        # 分词器
        self.tokenizer = tokenizer

        # 加载数据文件
        (
            self.ids,
            self.paths,
            self.images,
            self.questions,
            self.answers,
        ) = self.load_files(self.root, self.image_root, self.split, self.prefix)

        # 数据后处理
        self.post_process()

    @staticmethod
    def load_files(
        root: str, image_root: str, split: str, prefix: str
    ) -> Tuple[List[int], List[str], List[Dict], List[Dict], List[Dict]]:
        """
        静态方法，用于加载指定划分的问题、答案和图像信息。
        :param root: 数据集根目录
        :param image_root: 图像根目录
        :param split: 数据集划分类型
        :param prefix: 数据前缀
        :return: 图像 ID 列表、图像路径列表、图像信息列表、问题信息列表、答案信息列表
        """
        # 获取所有 .tif 图像文件的路径
        paths = glob(os.path.join(image_root, "*.tif"))
        # 对图像路径进行排序
        paths = sorted(paths, key=sort)
        # 加载问题信息
        with open(os.path.join(root, f"{prefix}_split_{split}_questions.json")) as f:
            questions = json.load(f)["questions"]
        # 加载答案信息
        with open(os.path.join(root, f"{prefix}_split_{split}_answers.json")) as f:
            answers = json.load(f)["answers"]
        # 加载图像信息
        with open(os.path.join(root, f"{prefix}_split_{split}_images.json")) as f:
            images = json.load(f)["images"]
        # 过滤出有效的图像 ID
        ids = [x["id"] for x in images if x["active"]]
        return ids, paths, images, questions, answers

    def post_process(self):
        """
        数据后处理方法，过滤掉问题类型为 "count" 和 "area" 的问题。
        """
        # 忽略的问题类型
        neglect_question_type = ["count", "area"]

        # 存储有效的问题 ID
        new_questions = []
        # 存储对应的图像 ID
        new_ids = []
        for id in self.ids:
            # 获取当前图像的问题 ID 列表
            questions_ids = self.images[id]["questions_ids"]
            # 过滤出有效的问题 ID
            valid_questions_ids = [
                i
                for i in questions_ids
                if self.questions[i]["type"].lower() not in neglect_question_type
            ]
            # 扩展有效的问题 ID 列表
            new_questions.extend(valid_questions_ids)
            # 扩展对应的图像 ID 列表
            new_ids.extend([id] * len(valid_questions_ids))

        # 更新问题 ID 列表
        self.questions_ids = new_questions
        # 更新图像 ID 列表
        self.ids = new_ids

    def __len__(self) -> int:
        """
        返回数据集的长度。
        """
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict:
        """
        根据索引获取数据样本。
        :param idx: 索引
        :return: 包含图像、问题、答案、问题类型等信息的字典
        """
        # 获取当前样本的图像 ID
        id = self.ids[idx]
        # 打开图像文件并转换为 numpy 数组
        x = np.array(Image.open(os.path.join(self.image_root, f"{id}.tif")))
        if isinstance(self.image_transform, CLIPImageProcessor):
            # 如果使用 CLIPImageProcessor 进行图像变换
            x = self.image_transform(x, return_tensors="pt").pixel_values.squeeze()
        else:
            # 否则使用自定义的图像变换
            x = self.image_transform(x)
        # 获取当前问题信息
        questions = self.questions[self.questions_ids[idx]]
        # 获取当前答案信息
        answers = self.answers[questions["answers_ids"][0]]["answer"]
        # 获取当前问题类型
        types = questions["type"]
        # 获取当前问题文本
        questions = questions["question"]
        # 对问题文本进行文本变换
        questions = self.text_transform(questions)
        # 对答案文本进行文本变换
        answers = self.text_transform(answers)

        # 添加标记前缀
        questions = self.token_prefix + questions

        # 构建问题和答案的字典
        item = dict(Question=questions, Answer=None)
        # 对问题进行多模态预处理
        questions = preprocess_multimodal(item, tune_im_start=self.tune_im_start)
        # 对问题进行预处理
        questions = preprocess(questions, self.tokenizer, has_image=True)
        # 获取输入 ID
        questions = questions["input_ids"][0]

        # 构建输出字典
        output = dict(
            x=x,
            question=questions,
            answer=answers,
            type=types,
            questions_idx=self.questions_ids[idx],
        )
        return output


class RSVQALR(RSVQA):
    """
    RSVQALR 数据集类，继承自 RSVQA。
    特定于 LR 前缀的数据集。
    """
    # 数据前缀
    prefix = "LR"

    def __init__(self, root: str = ".data/RSVQA_LR", *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(root, *args, **kwargs)


class RSVQAHR(RSVQA):
    """
    RSVQAHR 数据集类，继承自 RSVQA。
    特定于 USGS 前缀的数据集。
    """
    # 数据前缀
    prefix = "USGS"

    def __init__(self, root: str = ".data/RSVQA_HR", *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(root, *args, **kwargs)


class RSVQAxBEN(RSVQA):
    """
    RSVQAxBEN 数据集类，继承自 RSVQA。
    特定于 RSVQAxBEN 前缀的数据集。
    """
    # 数据前缀
    prefix = "RSVQAxBEN"

    def __init__(self, root: str = ".data/rsvqaxben", *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(root, *args, **kwargs)


@dataclass
class DataCollatorForVQASupervisedDataset(object):
    """
    用于监督微调的数据整理器类。
    将多个样本整理成一个批次。
    """
    # 分词器
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Tuple]) -> Tuple:
        """
        处理多个样本，将其整理成一个批次。
        :param instances: 样本序列
        :return: 包含图像、问题、注意力掩码、目标答案、问题类型、问题索引的字典
        """
        # 获取所有样本的问题输入 ID
        input_ids = tuple([instance["question"] for instance in instances])
        # 获取每个输入 ID 的长度
        lengths = [len(ids) for ids in input_ids]
        # 获取最大长度
        max_length = max(lengths)

        def left_pad_sequences(sequences, desired_length, padding_value):
            """
            对序列进行左填充，使其长度达到指定长度。
            :param sequences: 序列元组
            :param desired_length: 期望的长度
            :param padding_value: 填充值
            :return: 填充后的序列元组
            """
            padded_sequences = tuple(
                [padding_value] * (desired_length - len(seq)) + list(seq)
                for seq in sequences
            )
            return padded_sequences

        # 对输入 ID 进行左填充
        input_ids = left_pad_sequences(
            input_ids, max_length, self.tokenizer.pad_token_id
        )
        # 将填充后的输入 ID 转换为 torch.Tensor
        input_ids = torch.tensor(input_ids)
        # 截取到模型最大长度
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        # 生成注意力掩码
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # 获取所有样本的图像
        images = [instance["x"] for instance in instances]
        if not isinstance(images[0], Image.Image) and all(
            x is not None and x.shape == images[0].shape for x in images
        ):
            # 如果图像可以堆叠，将其堆叠成一个张量
            images = torch.stack(images)
        else:
            # 否则保持图像列表
            images = images

        # 获取所有样本的目标答案
        targets = [instance["answer"] for instance in instances]
        # 获取所有样本的问题类型
        type = [instance["type"] for instance in instances]
        # 获取所有样本的问题索引
        questions_idx = [instance["questions_idx"] for instance in instances]

        # 构建输出字典
        out = dict(
            images=images,
            questions=input_ids,
            attn_mask=attention_mask,
            targets=targets,
            types=type,
            questions_idx=questions_idx,
        )

        return out