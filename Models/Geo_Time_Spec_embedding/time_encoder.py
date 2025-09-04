# -*- coding: utf-8 -*-
"""
多尺度时间编码器（PyTorch）
将时间周期性特征编码为向量，支持与主干网络融合（FiLM/通道拼接）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============== 基础工具：连续周期的多谐波编码（Fourier特征） ===============
def fourier_time_feats(t_values: torch.Tensor, period: float, harmonics: int = 3) -> torch.Tensor:
    """
    连续周期标量的多谐波编码（sin/cos组合）
    参数：
        t_values: (B,) —— 时间标量（如小时、积日、月龄）
        period: float —— 周期长度（如24=日内周期，365.2422=年周期）
        harmonics: int —— 谐波阶数（默认3，可调2~6）
    返回：
        (B, 2*K) —— 编码向量，顺序为[sin(1x), cos(1x), sin(2x), cos(2x), ...]
    """
    # 归一化为角度：theta = 2π * t / 周期
    theta = 2 * math.pi * (t_values / period).unsqueeze(-1)  # (B, 1)
    feats = []
    for k in range(1, harmonics + 1):
        feats.append(torch.sin(k * theta))
        feats.append(torch.cos(k * theta))
    return torch.cat(feats, dim=-1)  # (B, 2*K)


# =============== 离散周期编码：可学习Embedding ===============
class CyclicEmbedding(nn.Module):
    """
    离散周期的可学习嵌入（如24节气、4季节、12生肖）
    """
    def __init__(self, num_classes: int, emb_dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=num_classes, embedding_dim=emb_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        参数：indices (B,) —— 离散周期索引（如节气0~23，季节0~3）
        返回：(B, emb_dim) —— 嵌入向量
        """
        return self.emb(indices)  # (B, emb_dim)


# =============== 离散周期编码：圆上平滑One-Hot（von Mises分布） ===============
def von_mises_smooth_onehot(indices: torch.Tensor, num_classes: int, kappa: float = 2.0) -> torch.Tensor:
    """
    离散周期的非参数化平滑编码（基于von Mises分布，模拟圆上邻近类别相关性）
    参数：
        indices: (B,) —— 离散周期索引（0~num_classes-1）
        num_classes: int —— 类别总数（如24节气=24，4季节=4）
        kappa: float —— 平滑系数（越大越接近One-Hot，默认2.0）
    返回：
        (B, num_classes) —— 平滑编码向量（每行和为1）
    """
    B = indices.size(0)
    idx = indices.unsqueeze(-1)  # (B, 1)
    xs = torch.arange(num_classes, device=indices.device)  # (C,)
    # 计算角度差：Δθ = 2π * (类别索引 - 输入索引) / 总类别数
    delta = 2 * math.pi * (xs.view(1, -1) - idx) / num_classes  # (B, C)
    # von Mises核计算权重
    weights = torch.exp(kappa * torch.cos(delta))  # (B, C)
    # 归一化（避免数值溢出）
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
    return weights

