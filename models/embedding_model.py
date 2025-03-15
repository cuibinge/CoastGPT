import ml_collections  # 导入ml_collections库，用于管理配置
import torch  # 导入PyTorch核心库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from typing import Dict, List, Optional, Tuple, Union

class EmbeddingModel(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        """
        初始化 EmbeddingModel 模型。

        参数:
            config: 包含模型参数的配置字典，包括 vision.embedding_dim 和 language.embedding_dim
        """
        super(EmbeddingModel, self).__init__()
        # 定义投影层
        self.projection = nn.Linear(config.vision.embedding_dim, config.language.embedding_dim)

    def forward(self, data: Dict, image_embedding):
        """
        前向传播，生成多模态嵌入。

        参数:
            data: 包含输入数据（图像、经纬度等）的字典
            image_embedding: 视觉模型输出的图像嵌入，形状为 (batch_size, vision_embedding_dim)

        返回:
            multimodal_embedding: 多模态嵌入，形状为 (batch_size, language_embedding_dim)
        """
        # 通过投影层将图像嵌入映射到语言嵌入维度
        projected_image_embedding = self.projection(image_embedding)
        return projected_image_embedding

    def encode_test(self, image_embedding):
        """
        纯图像编码测试。

        参数:
            image_embedding: 视觉模型输出的图像嵌入，形状为 (batch_size, vision_embedding_dim)

        返回:
            multimodal_embedding: 多模态嵌入，形状为 (batch_size, language_embedding_dim)
        """
        # 通过投影层将图像嵌入映射到语言嵌入维度
        projected_image_embedding = self.projection(image_embedding)
        return projected_image_embedding