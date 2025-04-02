import ml_collections  # 导入ml_collections库，用于管理配置
import torch  # 导入PyTorch核心库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from transformers import CLIPVisionModel  # 从transformers库导入CLIP视觉模型

# 视觉模型类，继承自nn.Module
class VisionModel(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        """
        初始化视觉模型

        参数:
            config (ml_collections.ConfigDict): 配置对象，包含模型参数和设置
        """
        super(VisionModel, self).__init__()  # 调用父类nn.Module的初始化方法

        self.embedding_dim = config.vision.embedding_dim  # 设置嵌入维度，从配置中获取
        # 从预训练模型加载CLIP视觉编码器，模型名称由配置中的vit_name指定
        self.encoder = CLIPVisionModel.from_pretrained(config.vit_name)

        self.extract_stage = [
            self.encoder.config.num_hidden_layers // 3 - 1,
            self.encoder.config.num_hidden_layers // 3 * 2 - 1,
            self.encoder.config.num_hidden_layers - 2,
        ]

    def encode(self, x: torch.Tensor):
        """
        对输入图像进行编码

        参数:
            x (torch.Tensor): 输入图像张量，形状通常为 (B, C, H, W)，表示批量大小、通道数、高度和宽度

        返回:
            image_embeds (torch.Tensor): 图像嵌入张量，形状为 (B, S, D)，S为序列长度，D为嵌入维度
        """
        # 使用CLIP视觉编码器处理输入图像
        outputs = self.encoder(
            x,
            return_dict=True,         # 以字典形式返回输出
            output_hidden_states=True, # 返回所有隐藏状态
        )
        # 从输出中提取最后一层隐藏状态，去掉CLS token（第0个位置），仅保留补丁嵌入
        image_embeds = outputs.hidden_states[11][:, 1:, :]

        # image_embeds = []
        # for idx, stage in enumerate(self.extract_stage):
        #     current_hidden_states = outputs.hidden_states[stage][:, 1:, :]
        #     image_embeds.append(current_hidden_states)
        # image_embeds = torch.cat(image_embeds, dim=1)

        return image_embeds

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (dict): 输入字典，包含键"rgb"对应的图像张量

        返回:
            torch.Tensor: 编码后的图像嵌入
        """
        modal_input = x["rgb"]  # 从输入字典中提取RGB图像数据
        return self.encode(modal_input)  # 调用encode方法进行编码