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
        # 地理位置编码器
        self.geo_encoder = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 1024)
        )

        # 时间编码器
        self.time_encoder = nn.Sequential(
            nn.Linear(6, 256),  # 输入6维时间特征（年、月、日、时、分、秒）
            nn.ReLU(),
            nn.Linear(256, 1024)
        )
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
        # 地理位置编码
        lat, lon = data["lat"], data["lon"]
        lat_norm = (lat + 90) / 180  # 标准化
        lon_norm = (lon + 180) / 360
        geo_input = torch.tensor([lat_norm, lon_norm])  # [2]
        geo_embedding = self.geo_encoder(geo_input)
        geo_embedding = geo_embedding.unsqueeze(1)

        # 时间编码
        timestamp = data["timestamp"]  # 假设格式为 [year, month, day, hour, minute, second]
        # 标准化时间特征
        time_norm = torch.tensor([
            (timestamp[0] - 2000) / 100.0,  # 年份标准化
            timestamp[1] / 12.0,  # 月份
            timestamp[2] / 31.0,  # 日
            timestamp[3] / 24,  # 小时
            timestamp[4] / 60.0,  # 分钟
            timestamp[5] / 60.0  # 秒
        ])
        time_embedding = self.time_encoder(time_norm)
        time_embedding = time_embedding.unsqueeze(1)

        # 拼接所有嵌入
        multimodal_embedding = torch.cat([image_embedding, geo_embedding, time_embedding], dim=1)

        # 通过投影层将多模态嵌入映射到语言嵌入维度
        projected_multimodal_embedding = self.projection(multimodal_embedding)
        return projected_multimodal_embedding

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
