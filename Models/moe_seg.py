import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, d=1, groups=1):
        super().__init__()
        p = (k // 2) * d
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BandAttentionAdapter(nn.Module):
    """轻量级通道注意力 + 1x1 卷积，用于光谱强化与维度适配。
    - 通过 SE 式通道权重突出关键通道
    - 使用 1x1 卷积将特征压缩/投影到目标维度
    """
    def __init__(self, in_ch: int, target_dim: int):
        super().__init__()
        hid = max(16, in_ch // 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, hid, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hid, in_ch, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(in_ch, target_dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.act(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        x = x * w
        return self.proj(x)


class TextureRefineAdapter(nn.Module):
    """3x3 卷积堆叠的纹理细化模块，提取像素级纹理特征并适配维度。"""
    def __init__(self, in_ch: int, target_dim: int, depth: int = 2):
        super().__init__()
        mid = max(32, in_ch // 2)
        blocks = []
        ch = in_ch
        for _ in range(max(1, depth)):
            blocks.append(ConvBNAct(ch, mid, k=3, d=1))
            ch = mid
        self.block = nn.Sequential(*blocks)
        self.proj = nn.Conv2d(mid, target_dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return self.proj(x)


class SPPSelectorAdapter(nn.Module):
    """空间金字塔池化筛选与维度降适配。
    通过多尺度池化捕捉与目标相关的上下文，并用 1x1 卷积压缩到目标维度。
    """
    def __init__(self, in_ch: int, target_dim: int, levels=(1, 2, 4)):
        super().__init__()
        self.levels = tuple(levels)
        self.proj = nn.Conv2d(in_ch * (len(self.levels) + 1), target_dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        pyramids = [x]
        for l in self.levels:
            p = F.adaptive_avg_pool2d(x, output_size=l)
            p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
            pyramids.append(p)
        y = torch.cat(pyramids, dim=1)
        return self.proj(y)


class ModalityGate(nn.Module):
    """模态选择门：根据权重融合光学与 SAR 特征，或在无显式模态时旁路。
    modal_inputs: {"optical": Tensor[B,C,H,W], "sar": Tensor[B,C,H,W]} 可选
    weights: (optical_w, sar_w)
    将返回融合后的特征；若缺失 modal_inputs 则直接返回原始 x。
    """
    def __init__(self, optical_w: float = 0.5, sar_w: float = 0.5, cloudy_boost: float = 0.0):
        super().__init__()
        self.optical_w = float(optical_w)
        self.sar_w = float(sar_w)
        self.cloudy_boost = float(cloudy_boost)

    def forward(self, x: torch.Tensor, modal_inputs: Optional[Dict[str, torch.Tensor]] = None, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if modal_inputs is None:
            return x
        opt = modal_inputs.get("optical", None)
        sar = modal_inputs.get("sar", None)
        if opt is None or sar is None:
            return x
        ow, sw = self.optical_w, self.sar_w
        # 可根据上下文（如多云）动态提升 SAR 权重
        if context is not None and bool(context.get("cloudy", False)):
            sw = min(1.0, sw + self.cloudy_boost)
            # 归一化
            s = ow + sw
            ow, sw = ow / s, sw / s
        return ow * opt + sw * sar


class SpectralExpert(nn.Module):
    """强调通道混合的专家，适合光谱差异"""
    def __init__(self, in_ch, num_classes):
        super().__init__()
        mid = max(32, in_ch // 2)
        self.block = nn.Sequential(
            ConvBNAct(in_ch, mid, k=1),
            ConvBNAct(mid, mid, k=1),
        )
        self.head = nn.Conv2d(mid, num_classes, kernel_size=1)

    def forward(self, x):
        return self.head(self.block(x))


class TextureExpert(nn.Module):
    """强调纹理/局部模式的专家，使用空洞卷积"""
    def __init__(self, in_ch, num_classes):
        super().__init__()
        mid = max(32, in_ch // 2)
        self.block = nn.Sequential(
            ConvBNAct(in_ch, mid, k=3, d=1),
            ConvBNAct(mid, mid, k=3, d=2),
            ConvBNAct(mid, mid, k=3, d=3),
        )
        self.head = nn.Conv2d(mid, num_classes, kernel_size=1)

    def forward(self, x):
        return self.head(self.block(x))


class ShapeExpert(nn.Module):
    """强调形态/结构的专家，使用大核与深度可分离卷积"""
    def __init__(self, in_ch, num_classes):
        super().__init__()
        mid = max(32, in_ch // 2)
        self.block = nn.Sequential(
            # depthwise
            ConvBNAct(in_ch, in_ch, k=7, groups=in_ch),
            # pointwise
            ConvBNAct(in_ch, mid, k=1),
            ConvBNAct(mid, mid, k=5),
        )
        self.head = nn.Conv2d(mid, num_classes, kernel_size=1)

    def forward(self, x):
        return self.head(self.block(x))


class ContextExpert(nn.Module):
    """上下文专家：ASPP + 图像金字塔池化，扩大感受野以利用邻域与语义上下文"""
    def __init__(self, in_ch, num_classes):
        super().__init__()
        mid = max(32, in_ch // 2)
        br_ch = max(16, mid // 2)
        # ASPP 分支
        self.aspp1 = ConvBNAct(in_ch, br_ch, k=1, d=1)
        self.aspp2 = ConvBNAct(in_ch, br_ch, k=3, d=2)
        self.aspp3 = ConvBNAct(in_ch, br_ch, k=3, d=4)
        # 图像级池化分支（SPP 风格：自适应到 1x1，再上采样）
        self.img_pool_proj = nn.Conv2d(in_ch, br_ch, kernel_size=1, bias=False)
        self.img_pool_bn = nn.BatchNorm2d(br_ch)
        self.img_pool_act = nn.SiLU(inplace=True)
        # 融合与输出
        fused_ch = br_ch * 4
        self.fuse = ConvBNAct(fused_ch, mid, k=1)
        self.head = nn.Conv2d(mid, num_classes, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]
        b1 = self.aspp1(x)
        b2 = self.aspp2(x)
        b3 = self.aspp3(x)
        g = F.adaptive_avg_pool2d(x, output_size=1)
        g = self.img_pool_act(self.img_pool_bn(self.img_pool_proj(g)))
        g = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
        y = torch.cat([b1, b2, b3, g], dim=1)
        y = self.fuse(y)
        return self.head(y)


class GenericExpert(nn.Module):
    """通用专家：轻量多层卷积堆叠，让专家在训练中自发形成偏好。
    设计选择：不预设领域偏好，依靠门控与数据驱动学习。
    """
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        mid = max(32, in_ch // 2)
        self.block = nn.Sequential(
            ConvBNAct(in_ch, mid, k=3, d=1),
            ConvBNAct(mid, mid, k=3, d=2),
            ConvBNAct(mid, mid, k=3, d=3),
        )
        self.head = nn.Conv2d(mid, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(x))

class GatingNetwork(nn.Module):
    def __init__(self, in_ch, num_experts=3):
        super().__init__()
        hid = max(32, in_ch // 2)
        self.net = nn.Sequential(
            ConvBNAct(in_ch, hid, k=3),
            ConvBNAct(hid, hid, k=3),
            nn.Conv2d(hid, num_experts, kernel_size=1),
        )
        self.num_experts = num_experts

    def forward(self, x):
        logits = self.net(x)  # [B,E,H,W]
        weights = torch.softmax(logits, dim=1)
        return weights, logits


class AquacultureSegMOE(nn.Module):
    """
    MOE 分割头：门控网络根据特征选择专家，专家输出按权重融合为分割 logits。
    输入：融合后的空间特征（例如 DualVisionEncoder 的 fused_spatial），以及期望输出尺寸。
    输出：分割 logits（与输入图像尺寸一致）。
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        include_context: bool = False,
        adapter_cfg: Optional[Dict[str, Any]] = None,
        num_experts: Optional[int] = None,
        top_k: int = 0,
        use_generic_experts: bool = False,
        hard_topk_inference: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.top_k = int(top_k) if top_k is not None else 0
        self.hard_topk_inference = bool(hard_topk_inference)
        adapter_cfg = adapter_cfg or {}
        # 适配器配置
        ad_enable = bool(adapter_cfg.get("enable", False))
        # 各专家目标维度（可选）；未设置则保持 in_channels
        spec_dim = int(adapter_cfg.get("spectral", {}).get("target_dim", in_channels))
        tex_dim = int(adapter_cfg.get("texture", {}).get("target_dim", in_channels))
        spp_dim = int(adapter_cfg.get("spp", {}).get("project_dim", in_channels))
        # 模态门配置
        mg_cfg = adapter_cfg.get("modality_gate", {})
        self.modality_gate = None
        if ad_enable and bool(mg_cfg.get("enable", False)):
            self.modality_gate = ModalityGate(
                optical_w=float(mg_cfg.get("optical_weight", 0.5)),
                sar_w=float(mg_cfg.get("sar_weight", 0.5)),
                cloudy_boost=float(mg_cfg.get("cloudy_boost", 0.0)),
            )
        # SPP 过滤器（作为共享预筛选）
        self.spp_filter = None
        if ad_enable and bool(adapter_cfg.get("spp", {}).get("enable", False)):
            levels = adapter_cfg.get("spp", {}).get("levels", [1, 2, 4])
            self.spp_filter = SPPSelectorAdapter(in_channels, target_dim=spp_dim, levels=levels)
        # 专家专属适配器
        self.spectral_adapter = None
        if ad_enable and bool(adapter_cfg.get("spectral", {}).get("enable", False)):
            self.spectral_adapter = BandAttentionAdapter(in_channels if self.spp_filter is None else spp_dim, target_dim=spec_dim)
        self.texture_adapter = None
        if ad_enable and bool(adapter_cfg.get("texture", {}).get("enable", False)):
            depth = int(adapter_cfg.get("texture", {}).get("depth", 2))
            self.texture_adapter = TextureRefineAdapter(in_channels if self.spp_filter is None else spp_dim, target_dim=tex_dim, depth=depth)

        # 专家集
        # 注意：各专家的输入维度可根据适配器输出调整
        ch_in = spp_dim if self.spp_filter is not None else in_channels
        predefined_experts = []
        # 原有预设专家集合
        predefined_experts.append(SpectralExpert(spec_dim if (self.spectral_adapter is not None) else ch_in, num_classes))
        predefined_experts.append(TextureExpert(tex_dim if (self.texture_adapter is not None) else ch_in, num_classes))
        predefined_experts.append(ShapeExpert(ch_in, num_classes))
        if include_context:
            predefined_experts.append(ContextExpert(ch_in, num_classes))

        # 选择专家模式：通用专家 or 预设专家
        self.generic_mode = bool(use_generic_experts) or (num_experts is not None and num_experts != len(predefined_experts))
        if self.generic_mode:
            n = int(num_experts) if num_experts is not None else 16
            experts = [GenericExpert(ch_in, num_classes) for _ in range(n)]
            self.experts = nn.ModuleList(experts)
        else:
            self.experts = nn.ModuleList(predefined_experts)

        # 门控网络（专家数动态）
        self.gate = GatingNetwork(ch_in, num_experts=len(self.experts))

    def forward(self, x: torch.Tensor, input_size: Tuple[int, int], modal_inputs: Optional[Dict[str, torch.Tensor]] = None, context: Optional[Dict[str, Any]] = None):
        # x: [B,C,h,w]
        # 1) 模态门融合（若提供）
        if self.modality_gate is not None:
            x = self.modality_gate(x, modal_inputs=modal_inputs, context=context)
        # 2) 共享 SPP 预筛选（若启用）
        x_shared = self.spp_filter(x) if self.spp_filter is not None else x
        # 3) 门控权重（基于共享特征）
        weights, gate_logits = self.gate(x_shared)  # [B,E,h,w]
        # 推理时可选 Top-K 硬路由：选择前 k 个专家并归一化权重
        if (not self.training) and self.top_k and self.top_k > 0 and self.hard_topk_inference:
            E = weights.shape[1]
            k = min(self.top_k, E)
            topk_vals, topk_idx = torch.topk(weights, k, dim=1)
            mask = torch.zeros_like(weights)
            mask.scatter_(1, topk_idx, 1.0)
            weights = weights * mask
            # 防止全零，做归一化
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            weights = weights / denom
        # 4) 专家专属输入
        x_spec = self.spectral_adapter(x_shared) if self.spectral_adapter is not None else x_shared
        x_tex = self.texture_adapter(x_shared) if self.texture_adapter is not None else x_shared
        # 5) 专家前向
        if self.generic_mode:
            expert_inputs = [x_shared for _ in range(len(self.experts))]
        else:
            expert_inputs = [x_spec, x_tex, x_shared]
            if len(self.experts) == 4:
                expert_inputs.append(x_shared)
        expert_logits = [expert(inp) for expert, inp in zip(self.experts, expert_inputs)]  # [(B,C,h,w)]
        # 对每个专家的输出做加权融合
        fused = 0.0
        for i, el in enumerate(expert_logits):
            wi = weights[:, i : i + 1]  # [B,1,h,w]
            fused = fused + wi * el
        # 上采样到输入尺寸
        B, C, h, w = fused.shape
        H, W = input_size
        if (h, w) != (H, W):
            fused = F.interpolate(fused, size=(H, W), mode="bilinear", align_corners=False)
        out = {"logits": fused, "gate_logits": gate_logits}
        # 训练阶段：附加路由正则，帮助负载均衡与稳定（可选）
        if self.training:
            # 熵正则：鼓励分布不过于尖锐或塌缩
            w_clamped = weights.clamp_min(1e-6)
            entropy = -(w_clamped * w_clamped.log()).mean()
            # z-loss：限制路由 logits 的幅度，避免数值爆炸
            zloss = (gate_logits ** 2).mean()
            out["router_aux"] = {"entropy": entropy, "zloss": zloss}
        return out


def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255, eps: float = 1e-6):
    """计算多类 Dice Loss，支持 ignore_index。"""
    # 仅在有效像素上计算
    valid_mask = (target != ignore_index)
    if valid_mask.sum() == 0:
        return logits.new_tensor(0.0)
    # 概率使用 float32 提升数值稳定性
    probs = torch.softmax(logits.float(), dim=1)
    target = target.clamp(min=0)
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    # 掩码应用
    probs = probs * valid_mask[:, None].float()
    target_onehot = target_onehot * valid_mask[:, None].float()
    # 类别平均 Dice
    intersect = (probs * target_onehot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))
    dice = (2 * intersect + eps) / (denom + eps)
    loss = 1.0 - dice.mean()
    return loss
