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
from .vision_model import VisionModel  # 自定义视觉模型模块
from .language_model import LanguageModel  # 自定义语言模型模块
# from .embedding_model import EmbeddingModel
from .embedding_model_r1 import EmbeddingModel


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
        self.multimodal = EmbeddingModel(config)  # 多模态嵌入模块

    def forward(self, data: Dict):
        """
        模型的前向传播。

        参数:
            data: 包含输入数据（图像、文本等）的字典

        返回:
            结合视觉和语言处理的模型输出
        """
        out = dict()
        total_loss = 0.0

        # 通过视觉模型处理图像
        image_embedding = self.vision(data)

        # 多模态嵌入处理
        multimodal_embedding = self.multimodal(data, image_embedding=image_embedding)

        # 通过语言模型处理组合输入
        output = self.language(data, multimodal_embedding=multimodal_embedding)

        text_loss = output
        total_loss += text_loss
        out.update({"text_loss": text_loss})
        out.update({"total_loss": total_loss})

        return out

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
        image_embedding = self.multimodal.encode_test(image_embedding)
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

    def load_vision_encoder(self, path: str):
        """
        从检查点文件加载视觉编码器。

        Args:
            path (str): 检查点文件的路径。
        """
        ckpt = torch.load(path, map_location="cpu")  # 将检查点加载到 CPU 内存
        if "model" in ckpt:
            ckpt = ckpt["model"]  # 如果检查点包含 "model" 键，则提取该部分
        # 将状态字典加载到视觉编码器中，strict=False 表示允许部分键不匹配
        self.rgb.encoder.load_state_dict(ckpt, strict=False)

    def custom_load_state_dict(self, state_dict_path, strict=False):
        """
        从指定路径加载模型的状态字典。

        如果路径是目录，则从零检查点加载；
        如果是文件，则加载检查点并提取视觉和文本部分。

        Args:
            state_dict_path (str): 状态字典的路径（可以是文件或目录）。
            strict (bool, optional): 是否严格要求状态字典的键与模型的键完全匹配。默认值为 False。
        """
        # return None
        # if os.path.isdir(state_dict_path):
        #     # 从零检查点目录加载状态字典（可能是 DeepSpeed 等框架的特性）
        #     self = load_state_dict_from_zero_checkpoint(self, state_dict_path)
        #     if isinstance(self.language.text_encoder, PeftModel):
        #         # 如果文本编码器是 PeftModel，则合并并卸载它
        #         self.language.text_encoder = self.language.text_encoder.merge_and_unload()
        #     return None

        # 从文件加载检查点
        ckpt = torch.load(state_dict_path, map_location="cpu")

        # # 获取模块（module）字典
        # module = ckpt.get('module', {})
        #
        # # 遍历字典中的每个键并修改
        # modified_module = {}
        # for key, value in module.items():
        #     # 替换键中的prefix
        #     if key.startswith('rgb.'):
        #         new_key = key.replace('rgb', 'vision', 1)
        #     elif key.startswith('rgb_pooler.'):
        #         new_key = key.replace('rgb_pooler', 'multimodal.projection', 1)
        #     elif key.startswith('text.'):
        #         new_key = key.replace('text', 'language', 1)
        #     else:
        #         new_key = key
        #
        #     modified_module[new_key] = value
        #
        # # 更新checkpoint中的module部分
        # ckpt['module'] = modified_module
        #
        # # 保存修改后的checkpoint
        # torch.save(ckpt, '/root/shared-nvme/CoastGPT/Checkpoint/test2/checkpoints/iter_1299/test.pt')


        if any(key.startswith('module') for key in ckpt.keys()):
            # filtered_state_dict = {k: v for k, v in ckpt["module"].items() if k.startswith("multimodal.")}
            filtered_state_dict = {k: v for k, v in ckpt["module"].items() if
                                   k.startswith("multimodal.") or k.startswith("vision.")}

            msg = self.load_state_dict(filtered_state_dict, strict=False)
            print(f"After loading: Missing: {msg.missing_keys}. Unexpected: {msg.unexpected_keys}")
        else:
            vision_ckpt = ckpt["rgb_ckpt"]
            multimodal_ckpt = ckpt["other_ckpt"]["rgb_pooler"]

            msg = self.vision.load_state_dict(vision_ckpt)
            print(f"After loading vision: Missing: {msg.missing_keys}. Unexpected: {msg.unexpected_keys}")
            msg = self.multimodal.projection.load_state_dict(multimodal_ckpt)
            print(f"After loading multimodal: Missing: {msg.missing_keys}. Unexpected: {msg.unexpected_keys}")

        text_path = pathlib.Path(state_dict_path).parent / "TextLoRA"  # 构造 TextLoRA 目录路径
        if text_path.exists():
            # 如果 TextLoRA 目录存在，则加载文本 LoRA
            self.language.text_encoder = PeftModel.from_pretrained(
                self.language.text_encoder,
                text_path,
                is_trainable=self.stage > 2,  # 仅在 stage > 2 时设置为可训练
                torch_dtype=torch.float16,  # 使用 float16 数据类型
            )

            if self.stage == 0:  # Eval 模式
                # 在评估模式下合并并卸载 PeftModel
                self.language.text_encoder = self.language.text_encoder.merge_and_unload()
        return None

        # if "model" in ckpt.keys():
        #     ckpt = ckpt["model"]  # 提取 "model" 部分（如果存在）
        # text_path = pathlib.Path(state_dict_path).parent / "TextLoRA"  # 构造 TextLoRA 目录路径
        #
        # # 从检查点加载视觉部分
        # self.vision.load_state_dict(ckpt["rgb_ckpt"], strict=strict)
        # del ckpt  # 删除检查点以释放内存
        #
        # if text_path.exists():
        #     # 如果 TextLoRA 目录存在，则加载文本 LoRA
        #     self.language.text_encoder = PeftModel.from_pretrained(
        #         self.language.text_encoder,
        #         text_path,
        #         is_trainable=self.stage > 2,  # 仅在 stage > 2 时设置为可训练
        #         torch_dtype=torch.float16,  # 使用 float16 数据类型
        #     )
        #
        #     if self.stage == 0:  # Eval 模式
        #         # 在评估模式下合并并卸载 PeftModel
        #         self.language.text_encoder = self.language.text_encoder.merge_and_unload()
        #
        # return None

    def prepare_for_training(
            self,
            freeze_vision: bool = False,
            freeze_text: bool = False,
            tune_multimodal: bool = False,
            model_path: str = None,
            tune_im_start: bool = False,
            compute_dtype: torch.dtype = torch.float32,
    ):
        """
        准备模型进行训练，设置梯度和数据类型。

        Args:
            freeze_vision (bool, optional): 是否冻结视觉参数。默认值为 False。
            freeze_text (bool, optional): 是否冻结文本参数。默认值为 False。
            tune_multimodal (bool, optional): 是否冻结多模态参数。默认值为 False。
            model_path (str, optional): 加载模型的路径。默认值为 None。
            tune_im_start (bool, optional): 在冻结文本时是否调整输入嵌入。默认值为 False。
            compute_dtype (torch.dtype, optional): 计算使用的数据类型。默认值为 torch.float32。
        """
        self.train()  # 将模型设置为训练模式

        # 设置视觉参数的 requires_grad 属性并转换数据类型
        for param in self.vision.parameters():
            if freeze_vision:
                param.requires_grad = False  # 冻结参数，不计算梯度
            else:
                param.requires_grad = True  # 解冻参数，计算梯度
            param.data = param.data.to(dtype=compute_dtype)  # 转换为指定数据类型

        # 将视觉缓冲区转换为计算数据类型（排除索引和 ID 相关的缓冲区）
        for name, buffer in self.vision.named_buffers():
            if "index" not in name and "id" not in name:
                buffer.data = buffer.data.to(dtype=compute_dtype)

        if freeze_text:
            self.language.eval()  # 将文本编码器设置为评估模式
            # 冻结输入和输出嵌入的参数
            for p in self.language.get_text_encoder().get_input_embeddings().parameters():
                p.requires_grad = False
            for p in self.language.get_text_encoder().get_output_embeddings().parameters():
                p.requires_grad = False
        else:
            # 即使不冻结文本，仍然冻结输入和输出嵌入（可能是特定设计选择）
            for p in self.language.get_text_encoder().get_input_embeddings().parameters():
                p.requires_grad = False
            for p in self.language.get_text_encoder().get_output_embeddings().parameters():
                p.requires_grad = False

        # 多模态相关参数是否训练
        for param in self.multimodal.parameters():
            if tune_multimodal:
                param.requires_grad = True
            else:
                param.requires_grad = False
            param.data = param.data.to(dtype=compute_dtype)

        if tune_im_start and freeze_text:
            # 如果 tune_im_start 为 True 且文本被冻结，则解冻输入嵌入
            for p in self.language.get_text_encoder().get_input_embeddings().parameters():
                p.requires_grad = True
            # 输出嵌入保持冻结状态（已在前面设置，此处无需重复）

        if model_path is not None:
            # 如果提供了模型路径，则加载模型
            self.custom_load_state_dict(model_path)
