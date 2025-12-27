# -*- coding: utf-8 -*-
"""
多尺度时间编码器（PyTorch）
将时间周期性特征编码为向量，支持与主干网络融合（FiLM/通道拼接）
"""




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

