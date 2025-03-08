import logging
import os
import pathlib
from typing import Dict, List, Tuple

import ml_collections
import torch
import torch.nn as nn
from deepspeed.utils.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint,
    load_state_dict_from_zero_checkpoint,
)
from peft import PeftModel
from vision_model import VisionModel  # 自定义视觉模型模块
from language_model import LanguageModel  # 自定义语言模型模块


# 定义 CoastGPT 类，继承自 PyTorch 的 nn.Module
class CoastGPT(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        """
        初始化 CoastGPT 模型。

        参数:
            config: 包含模型参数的配置字典
        """
        super(CoastGPT, self).__init__()
        self.stage = config.stage  # 从配置中存储训练/推理阶段

        # 初始化视觉和语言组件
        self.vision = VisionModel(config)  # 视觉处理模块
        self.language = LanguageModel(config)  # 语言处理模块

    def forward(self, data: Dict):
        """
        模型的前向传播。

        参数:
            data: 包含输入数据（图像、文本等）的字典

        返回:
            结合视觉和语言处理的模型输出
        """
        # 通过视觉模型处理图像
        image_embedding = self.vision(data)
        # 通过语言模型处理组合输入
        output = self.language(data, image_embedding=image_embedding)

        return output

    def encode_image(self, image, pool):
        """
        将输入图像编码为嵌入向量。

        参数:
            image: 输入图像张量
            pool: 布尔值，指示是否对嵌入向量进行池化

        返回:
            图像嵌入向量（池化或未池化）
        """
        # 从视觉模型获取原始图像嵌入
        image_embedding = self.vision.encode(image)
        if pool:
            # 如果请求池化，返回平均池化的嵌入向量
            return image_embedding.mean(dim=1)
        else:
            # 如果不池化，返回完整嵌入向量
            return image_embedding

    def generate(
            self,
            input_ids: torch.Tensor,
            images: torch.Tensor = None,
            do_sample: bool = True,
            temperature: float = 0.2,
            max_new_tokens: int = 1024,
            streamer=None,
            use_cache: bool = True,
            stopping_criteria=None,
            **kwargs,
    ):
        """
        生成文本输出。

        参数:
            input_ids: 输入的 token ID 张量
            images: 可选的输入图像张量，默认为 None
            do_sample: 是否使用采样生成，默认为 True
            temperature: 控制生成随机性的温度参数，默认为 0.2
            max_new_tokens: 最大生成 token 数，默认为 1024
            streamer: 可选的流式输出对象，默认为 None
            use_cache: 是否使用缓存加速生成，默认为 True
            stopping_criteria: 可选的停止条件，默认为 None
            **kwargs: 其他可选参数

        返回:
            生成的文本输出
        """
        if images is not None:
            # 如果提供了图像，编码为嵌入向量（不池化）
            image_embedding = self.encode_image(images, pool=False)
        else:
            image_embedding = None
        # 调用语言模型的生成方法
        return self.language.generate(
            input_ids=input_ids,
            image_embedding=image_embedding,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=use_cache,
            stopping_criteria=stopping_criteria,
            **kwargs,
        )