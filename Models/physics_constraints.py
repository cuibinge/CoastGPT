import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PhysicsGuidedLoss(nn.Module):
    """
    基于辐射传输方程（RTE）半分析模型的可微分物理约束模块。
    严格映射隐式视觉流形 Z_visual 至固有光学量（IOPs），
    从而对视觉编码器施加物理法则层面的梯度惩罚。
    """

    def __init__(self, visual_dim=768, num_bands=4):
        super().__init__()
        # 将高维视觉特征 Z_visual 降维并映射到物理光谱空间 (例如 Blue, Green, Red, NIR)
        self.latent_to_rrs = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.LayerNorm(256),
            nn.Linear(256, num_bands),
            nn.Softplus()  # Rrs 物理上严格非负
        )

        # 预置物理常数 (以 Nechad 2010 悬浮物 TSM 半分析模型为例)
        # 核心机理方程: TSM = (A * Rrs) / (1 - Rrs / C)
        self.A_tsm = nn.Parameter(torch.tensor(289.29), requires_grad=False)
        self.C_tsm = nn.Parameter(torch.tensor(0.1686), requires_grad=False)

    def compute_rte_tsm(self, rrs_red: torch.Tensor) -> torch.Tensor:
        """核心物理正演：将遥感反射率转化为总悬浮物浓度"""
        eps = 1e-6
        # 物理截断约束：Rrs/C 必须严格小于 1，保证分母正定且符合光学定律
        rrs_clamped = torch.clamp(rrs_red, max=(self.C_tsm.item() - eps))

        denominator = torch.clamp(1.0 - rrs_clamped / self.C_tsm, min=eps)
        tsm_pred = (self.A_tsm * rrs_clamped) / denominator

        return tsm_pred

    def forward(self, Z_visual: torch.Tensor, batch: Dict) -> torch.Tensor:
        """
        Args:
            Z_visual: [B, N, visual_dim] 异构双流编码器输出的解耦视觉表征
            batch: 包含理化真值 'tsm' 与有效域 'mask'
        """
        gt_tsm = batch.get('tsm')
        mask = batch.get('mask')

        if gt_tsm is None:
            # 缺失物理真值时，返回 0 梯度，避免训练中断
            return torch.tensor(0.0, device=Z_visual.device, requires_grad=True)

        # 1. 隐式流形解码为遥感反射率光谱
        pred_rrs = self.latent_to_rrs(Z_visual)  # [B, N, num_bands]

        # 提取对 TSM 敏感的 Red 波段 (假设索引为 2)
        rrs_red = pred_rrs[..., 2]

        # 2. 物理定律约束下的浓度反演
        pred_tsm = self.compute_rte_tsm(rrs_red)

        # 空间维度对齐 (展平真值矩阵以匹配序列长度 N)
        if gt_tsm.dim() == 4:
            gt_tsm = gt_tsm.flatten(2).transpose(1, 2)
        if mask is not None and mask.dim() == 4:
            mask = mask.flatten(2).transpose(1, 2)

        # 3. 构建物理先验正则化损失 (Physics-informed Regularization)
        loss_elements = F.mse_loss(pred_tsm, gt_tsm, reduction='none')

        if mask is not None:
            loss_phy = torch.sum(loss_elements * mask) / (torch.sum(mask) + 1e-6)
        else:
            loss_phy = torch.mean(loss_elements)

        return loss_phy


def compute_sar_sigma0_gt(batch: Dict, coeffs: Dict = None) -> Optional[torch.Tensor]:
    """
    SAR 经验模型后处理提取（保留辅助接口）。
    sigma0(dB) = A + B * log10(wind_speed) + C * (incidence_deg) + D_pol
    """
    if coeffs is None:
        return None
    wind = batch.get('wind_speed')
    inc = batch.get('incidence_deg')
    pol = batch.get('polarization')

    if wind is None or inc is None or pol is None:
        return None

    A, B, C = float(coeffs.get('A', 0.0)), float(coeffs.get('B', 0.0)), float(coeffs.get('C', 0.0))
    D = float(coeffs.get('D_VV', 0.0)) if pol == 'VV' else float(coeffs.get('D_VH', 0.0))

    # 张量清洗：保证风速的非负限制，阻断 log10 产生 NaN 导致的梯度污染
    wind = torch.clamp(torch.nan_to_num(wind, nan=1e-6, posinf=100.0, neginf=1e-6), min=1e-6)
    inc = torch.nan_to_num(inc, nan=0.0)

    sigma0_db = A + B * torch.log10(wind) + C * inc + D
    return sigma0_db