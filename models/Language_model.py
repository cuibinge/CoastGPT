import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import ml_collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers import BatchEncoding

from ..Dataset import conversation as conversation_lib
from . import (
    DEFAULT_IM_END_TOKEN,  # 默认图像结束标记
    DEFAULT_IM_START_TOKEN,  # 默认图像开始标记
    DEFAULT_IMAGE_PATCH_TOKEN,  # 默认图像补丁标记
    DEFAULT_IMAGE_TOKEN,  # 默认图像标记
    IGNORE_INDEX,  # 忽略索引值
    IMAGE_TOKEN_INDEX,  # 图像标记索引
)

logger = logging.getLogger("train")
type_dict = {
    "float32": torch.float32,  # 32位浮点类型
    "float16": torch.float16,  # 16位浮点类型
    "bfloat16": torch.bfloat16,  # bfloat16类型
}


# 自定义Llama因果语言模型
class CustomLlamaForCausalLM(LlamaForCausalLM):
    _keep_in_fp32_modules = ["embed_tokens", "lm_head"]  # 需要保持FP32精度的模块

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    # 准备生成任务的输入
    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]  # 如果有过去的键值对，只取最后一个输入ID

        # 如果提供了inputs_embeds，仅在第一步生成中使用
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


# 语言模型类
class LanguageModel(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        """
        大型语言模型作为图像标题的文本编码器
        实现高度基于Huggingface transformers库

        编码器：LLaMA-v2-7B
        """
        super(LanguageModel, self).__init__()

        self.embedding_dim = config.text.hidden_size  # 嵌入维度
        self.tune_pooler = config.tune_rgb_pooler  # 是否调整RGB池化器
        self.tune_im_start = config.tune_im_start  # 是否调整图像开始标记
        self.tune_im_patch = config.tune_im_patch  # 是否调整图像补丁标记
        self.num_query = config.rgb_vision.attn_pooler.num_query  # 查询数量

        compute_dtype = type_dict[config.dtype]  # 计算数据类型
        bnb_model_from_pretrained_args = {}

        # 设置设备（支持分布式训练或单GPU）
        if getattr(config, "is_distribute", False):
            device = torch.device(getattr(config, "local_rank", 0))
        elif (
                "CUDA_VISABLE_DEVICES" in os.environ.keys()
                and len(os.environ["CUDA_VISABLE_DEVICES"].split(",")) == 1
        ):
            device = torch.device("cuda:" + os.environ["CUDA_VISABLE_DEVICES"])
        else:
            device = torch.device("cuda")

        # 如果使用4位或8位量化
        if config.bits in [4, 8]:
            from transformers import BitsAndBytesConfig
            bnb_model_from_pretrained_args.update(
                dict(
                    device_map={"": device},
                    load_in_4bit=config.bits == 4,  # 4位加载
                    load_in_8bit=config.bits == 8,  # 8位加载
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=config.bits == 4,
                        load_in_8bit=config.bits == 8,
                        llm_int8_threshold=6.0,  # LLM INT8阈值
                        llm_int8_has_fp16_weight=False,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_use_double_quant=config.double_quant,  # 双重量化
                        bnb_4bit_quant_type=config.quant_type,  # 量化类型
                    ),
                )
            )
        else:
            bnb_model_from_pretrained_args.update(
                dict(device_map={"": device}, torch_dtype=compute_dtype)
            )

        # 从预训练模型加载文本编码器
        self.text_encoder = CustomLlamaForCausalLM.from_pretrained(
            config.text.path, **bnb_model_from_pretrained_args
        )

        self.tokenizer = self.init_tokenizer(config.text.path)  # 初始化分词器

        # 为量化模型准备训练
        if config.bits in [4, 8]:
            from peft import prepare_model_for_kbit_training
            self.text_encoder.config.torch_dtype = (
                torch.float32 if config.fp16 else (torch.bfloat16 if config.bf16 else torch.float32)
            )
            self.text_encoder = prepare_model_for_kbit_training(
                self.text_encoder, use_gradient_checkpointing=config.use_checkpoint
            )

        # 如果启用LoRA（低秩适配）
        if config.lora.enable:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=config.lora.lora_r,  # LoRA秩
                lora_alpha=config.lora.lora_alpha,  # LoRA缩放因子
                target_modules=find_all_linear_names(self.text_encoder),  # 目标线性层
                lora_dropout=config.lora.lora_dropout,  # 丢弃率
                bias=config.lora.lora_bias,  # 偏置设置
                task_type="CAUSAL_LM",  # 任务类型
            )
            if config.bits == 16:
                if config.bf16:
                    self.text_encoder.to(torch.bfloat16)
                if config.fp16:
                    self.text_encoder.to(torch.float16)
            logger.info("Adding LoRA adapters...")
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)

        # 启用梯度检查点
        if getattr(config, "use_checkpoint", False):
            self.text_encoder.gradient_checkpointing_enable()
            if hasattr(self.get_text_encoder(), "enable_input_require_grads"):
                self.get_text_encoder().enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                self.get_text_encoder().get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

    # 获取文本编码器实例
    def get_text_encoder(self):
        text_encoder = self.text_encoder
        while not isinstance(text_encoder, CustomLlamaForCausalLM):
            text_encoder = text_encoder.model
        return text_encoder

    # 初始化分词器
    def init_tokenizer(self, tokenizer_name: str):
        """
        为LLaMA初始化分词器
        """
        tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_name)
        tokenizer.pad_token_id = tokenizer.unk_token_id  # 设置填充标记

        if self.tune_im_patch:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.get_text_encoder().resize_token_embeddings(len(tokenizer))

        if self.tune_im_start:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.get_text_encoder().resize_token_embeddings(len(tokenizer))

            # 初始化新添加的token嵌入
            if num_new_tokens > 0:
                input_embeddings = self.get_text_encoder().get_input_embeddings().weight.data
                output_embeddings = self.get_text_encoder().get_output_embeddings().weight.data
                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

        return tokenizer

    # 获取模态输入
    def get_modal_input(self, x: Dict[str, Union[str, torch.Tensor]]):
        return dict(
            input_ids=x["input_ids"],
            labels=x["labels"],
            attention_mask=x["attention_mask"],
        )

    # 前向传播
    def forward(
            self,
            x: Dict[str, Union[str, torch.Tensor, BatchEncoding]],
            image_embedding: torch.Tensor = None,
            **kwargs,
    ):
        modal_input = self.get_modal_input(x)
        return self.decode(**self.encode(modal_input), image_embedding=image_embedding, **kwargs)

    # 生成文本
    def generate(
            self,
            image_embedding: torch.Tensor = None,
            prompt: Optional[Union[str, Dict]] = "Describe the image in a sentence.\n<image>",  # 默认提示
            input_ids: Optional[torch.LongTensor] = None,
            do_sample: bool = True,  # 是否使用采样
            temperature: float = 0.2,  # 温度参数
            max_new_tokens: int = 1024,  # 最大新token数
            streamer=None,
            use_cache: bool = True,  # 是否使用缓存
            stopping_criteria=None,  # 停止条件
            attention_mask=None,
            **kwargs,
    ):
        conv = conversation_lib.default_conversation.copy()

        if input_ids is None:
            """图像标题生成"""
            if DEFAULT_IMAGE_TOKEN in prompt:
                value = prompt.replace(DEFAULT_IMAGE_TOKEN, "").strip()
                value = DEFAULT_IMAGE_TOKEN + "\n" + value
                value = value.strip()
                replace_token = DEFAULT_IMAGE_TOKEN
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                value = value.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                conv.append_message(conv.roles[0], value)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).expand(image_embedding.size(0), -1)
                input_ids = input_ids.to(image_embedding.device)
                input_ids = self.prepare_inputs_for_multimodal(
                    input_ids=input_ids,
                    attention_mask=None,
                    labels=None,
                    image_embedding=image_embedding,
                )
                outputs = self.text_encoder.generate(inputs_embeds=input_ids[1])
                outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return outputs


# 处理图像token的分词函数
def tokenizer_image_token(
        prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
            len(prompt_chunks) > 0
            and len(prompt_chunks[0]) > 0
            and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    return input_ids


# 查找所有线性层名称
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            lora_module_names.add(name)

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)