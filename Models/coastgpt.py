import logging
import os
import pathlib
from typing import Dict, List, Tuple

import ml_collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.utils.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint,
    load_state_dict_from_zero_checkpoint,
)
from peft import PeftModel
from .vision_model import VisionModel as SingleVisionModel  # 自定义视觉模型模块
from .dual_vision_encoder import DualVisionEncoder  # 双编码器视觉模块
from .language_model import LanguageModel  # 自定义语言模型模块
# from .embedding_model import EmbeddingModel
from .embedding_model_r1 import EmbeddingModel
from . import physics_decoder as phys
# 物理解码与损失函数（兼容部分环境缺失符号）
PhysicsDecoder = phys.PhysicsDecoder
pixelwise_mse = getattr(phys, "pixelwise_mse", lambda pred, gt: F.mse_loss(pred, gt))

def _edge_preserve_default(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # 简单梯度一致性正则（不依赖 Sobel），在缺失 edge_preserve_loss 时回退使用
    dh_p = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    dw_p = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dh_g = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])
    dw_g = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
    return F.l1_loss(dh_p, dh_g) + F.l1_loss(dw_p, dw_g)

edge_preserve_loss = getattr(phys, "edge_preserve_loss", _edge_preserve_default)

def _tv_default(x: torch.Tensor) -> torch.Tensor:
    # 总变分回退实现
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw

total_variation_loss = getattr(phys, "total_variation_loss", _tv_default)

def _consistency_default(phy_full, phy_s1, phy_s2, phy_s3) -> torch.Tensor:
    # 多尺度一致性回退实现
    H, W = phy_full.shape[-2], phy_full.shape[-1]
    s1_up = F.interpolate(phy_s1, size=(H, W), mode="bilinear", align_corners=False)
    s2_up = F.interpolate(phy_s2, size=(H, W), mode="bilinear", align_corners=False)
    s3_up = F.interpolate(phy_s3, size=(H, W), mode="bilinear", align_corners=False)
    return (F.mse_loss(phy_full, s1_up) + F.mse_loss(phy_full, s2_up) + F.mse_loss(phy_full, s3_up)) / 3.0

consistency_loss_multiscale = getattr(phys, "consistency_loss_multiscale", _consistency_default)

def _sam_default(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # 光谱角映射回退实现
    B, C, H, W = pred.shape
    p = pred.permute(0, 2, 3, 1).reshape(-1, C)
    g = gt.permute(0, 2, 3, 1).reshape(-1, C)
    dot = (p * g).sum(dim=1)
    pn = torch.norm(p, dim=1)
    gn = torch.norm(g, dim=1)
    cos = torch.clamp(dot / (pn * gn + eps), min=-1.0, max=1.0)
    ang = torch.acos(cos)
    return ang.mean()

spectral_loss_sam = getattr(phys, "spectral_loss_sam", _sam_default)
from .physics_constraints import compute_rte_rrs_gt, compute_sar_sigma0_gt
from .moe_seg import AquacultureSegMOE, dice_loss
# 在不支持 NPU 的环境下忽略 torch_npu 导入错误
try:
    import torch_npu  # noqa: F401
    HAS_TORCH_NPU = True
except Exception:
    HAS_TORCH_NPU = False

logger = logging.getLogger("train")

# 定义 CoastGPT 类，继承自 PyTorch 的 nn.Module
class CoastGPT(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        """
        初始化 CoastGPT 模型。

        参数:
            config: 包含模型参数的配置字典
        """
        super(CoastGPT, self).__init__()
        # 保存配置以便在 forward 中访问（例如自动生成物理 GT 的开关和参数）
        self.config = config
        self.stage = config.stage  # 从配置中存储训练/推理阶段

        # 初始化视觉和语言组件
        if getattr(config, "rgb_vision", ml_collections.ConfigDict()).get("arch", "vit_large") == "dual":
            self.vision = DualVisionEncoder(config)  # 双编码器视觉处理模块
        else:
            self.vision = SingleVisionModel(config)  # 单编码器视觉处理模块
        self.language = LanguageModel(config)  # 语言处理模块
        self.multimodal = EmbeddingModel(config)  # 多模态嵌入模块

        # 物理解码器（可选）
        phy_cfg = getattr(config, "physics", ml_collections.ConfigDict())
        self.physics_enabled = bool(phy_cfg.get("enabled", False))
        if self.physics_enabled:
            out_channels = int(phy_cfg.get("out_channels", 1))
            # 使用视觉融合后的通道维度作为解码器通道
            dec_channels = int(getattr(self.vision, "embedding_dim", 256))
            self.physics = PhysicsDecoder(out_channels=out_channels, dec_channels=dec_channels)
            # 损失权重配置
            self.physics_loss_weight = float(phy_cfg.get("loss_weight", 1.0))
            self.physics_edge_weight = float(phy_cfg.get("edge_weight", 0.0))
            self.physics_tv_weight = float(phy_cfg.get("tv_weight", 0.0))
            self.physics_consistency_weight = float(phy_cfg.get("consistency_weight", 0.0))
            self.physics_spectral_weight = float(phy_cfg.get("spectral_weight", 0.0))

        # 养殖区分割（MOE）分支（可选）
        seg_cfg = getattr(config, "aquaculture_seg", ml_collections.ConfigDict())
        self.seg_enabled = bool(seg_cfg.get("enabled", False))
        if self.seg_enabled:
            num_classes = int(seg_cfg.get("num_classes", 2))
            dec_channels = int(getattr(self.vision, "embedding_dim", 256))
            include_context = bool(seg_cfg.get("include_context_expert", True))
            adapter_cfg = seg_cfg.get("adapters", ml_collections.ConfigDict())
            # 额外的 MOE 配置：与训练保持一致
            num_experts = seg_cfg.get("num_experts", None)
            use_generic_experts = bool(seg_cfg.get("use_generic_experts", False))
            top_k = int(seg_cfg.get("top_k", 0))
            hard_topk_inference = bool(seg_cfg.get("hard_topk_inference", True))
            # 传递适配器与 MOE 配置
            self.seg_head = AquacultureSegMOE(
                in_channels=dec_channels,
                num_classes=num_classes,
                include_context=include_context,
                adapter_cfg=adapter_cfg.to_dict() if hasattr(adapter_cfg, "to_dict") else adapter_cfg,
                num_experts=num_experts,
                top_k=top_k,
                use_generic_experts=use_generic_experts,
                hard_topk_inference=hard_topk_inference,
                text_embed_dim=int(getattr(config.text, "hidden_size", dec_channels)),
            )
            # 损失权重
            self.seg_loss_weight = float(seg_cfg.get("loss_weight", 1.0))
            self.seg_ce_weight = float(seg_cfg.get("ce_weight", 1.0))
            self.seg_dice_weight = float(seg_cfg.get("dice_weight", 1.0))
            self.seg_ignore_index = int(seg_cfg.get("ignore_index", 255))
            self.seg_log_gate_stats = bool(seg_cfg.get("log_gate_stats", True))
            # 路由器辅助损失系数（entropy、zloss）
            router_aux_cfg = seg_cfg.get("router_aux_loss_coef", {})
            self.seg_router_entropy_coef = float(router_aux_cfg.get("entropy", 0.0))
            self.seg_router_zloss_coef = float(router_aux_cfg.get("zloss", 0.0))

    def forward(self, data: Dict):
        """
        模型的前向传播。

        参数:
            data: 包含输入数据（图像、文本等）的字典

        返回:
            结合视觉和语言处理的模型输出
        """
        out = dict()
        total_loss = None

        # 物理提示文本 -> 连续嵌入（若存在）
        physical_prompt_embs = None
        task_text_embs = None
        element_text_embs = None
        emb_layer = None
        try:
            if hasattr(self.language, "get_text_encoder"):
                emb_layer = self.language.get_text_encoder().get_input_embeddings()
            elif hasattr(self.language, "text_encoder"):
                emb_layer = self.language.text_encoder.get_input_embeddings()
            elif hasattr(self.language, "model"):
                emb_layer = self.language.model.get_input_embeddings()
        except Exception as exc:
            emb_layer = None
            if not getattr(self, "_warned_semantic_emb_layer", False):
                logger.warning(
                    "Failed to get semantic embedding layer for task/element prompts: %s", exc
                )
                self._warned_semantic_emb_layer = True
        if (
            emb_layer is None
            and not getattr(self, "_warned_semantic_emb_layer_none", False)
            and any(data.get(k, None) is not None for k in ("task_text_ids", "element_text_ids", "physical_prompt_ids"))
        ):
            logger.warning(
                "Semantic prompt ids exist in batch, but embedding layer is None; route supervision may be disabled."
            )
            self._warned_semantic_emb_layer_none = True
        if emb_layer is not None:
            if "physical_prompt_ids" in data and data["physical_prompt_ids"] is not None:
                physical_prompt_embs = emb_layer(data["physical_prompt_ids"])
                if "physical_prompt_attention_mask" in data and data["physical_prompt_attention_mask"] is not None:
                    phy_mask = data["physical_prompt_attention_mask"].to(physical_prompt_embs.device)
                    physical_prompt_embs = physical_prompt_embs * phy_mask.unsqueeze(-1).to(physical_prompt_embs.dtype)
                data["physical_prompt_embs"] = physical_prompt_embs
            if "task_text_ids" in data and data["task_text_ids"] is not None:
                task_text_embs = emb_layer(data["task_text_ids"])
                if "task_text_attention_mask" in data and data["task_text_attention_mask"] is not None:
                    task_mask = data["task_text_attention_mask"].to(task_text_embs.device)
                    task_text_embs = task_text_embs * task_mask.unsqueeze(-1).to(task_text_embs.dtype)
                data["task_text_embs"] = task_text_embs
            if "element_text_ids" in data and data["element_text_ids"] is not None:
                element_text_embs = emb_layer(data["element_text_ids"])
                if "element_text_attention_mask" in data and data["element_text_attention_mask"] is not None:
                    element_mask = data["element_text_attention_mask"].to(element_text_embs.device)
                    element_text_embs = element_text_embs * element_mask.unsqueeze(-1).to(element_text_embs.dtype)
                data["element_text_embs"] = element_text_embs

        # 通过视觉模型处理图像
        if isinstance(self.vision, DualVisionEncoder):
            image_seq, fused_spatial, pyramid_raw = self.vision.encode_with_spatial(
                data["rgb"], physical_prompt_embs=physical_prompt_embs
            )
            if hasattr(self.vision, "get_alignment_stats"):
                for k, v in self.vision.get_alignment_stats().items():
                    out[f"vision_{k}"] = v
        else:
            image_seq = self.vision(data)
            fused_spatial, pyramid_raw = None, None

        # 多模态嵌入处理
        multimodal_embedding = self.multimodal(data, image_embedding=image_seq)

        # 通过语言模型处理组合输入
        output = self.language(data, multimodal_embedding=multimodal_embedding)

        text_loss = output
        if not torch.is_tensor(text_loss):
            raise RuntimeError(f"language model must return a tensor loss, got {type(text_loss)}")
        total_loss = text_loss
        out.update({"text_loss": text_loss})

        if hasattr(self.multimodal, "get_aux_loss"):
            # 获取辅助损失
            mm_aux_loss = self.multimodal.get_aux_loss()
            
            # 定义辅助损失的权重（建议在 0.01 左右，防止其梯度压过主任务 text_loss）
            # 最好从 config 中读取，这里做个 fallback 默认给 0.01
            mm_aux_weight = getattr(self.config, "mm_moe_aux_weight", 0.01) 
            
            if mm_aux_loss is not None:
                # 累加到总损失中，注意对齐 dtype 防止混合精度报错
                total_loss += (mm_aux_weight * mm_aux_loss).to(total_loss.dtype)
                
                # 更新到 out 字典，便于日志监控
                out.update({
                    "mm_moe_aux_loss_raw": mm_aux_loss,
                    "mm_moe_aux_loss_weighted": mm_aux_weight * mm_aux_loss
                })
            
            # 🌟 强烈建议：把门控状态也记录下来，这是调试 MoE 是否坍塌的唯一“眼睛”
            if hasattr(self.multimodal, "get_gate_stats"):
                gate_stats = self.multimodal.get_gate_stats()
                # 记录每个专家的实际负载率
                if "load" in gate_stats and gate_stats["load"] is not None:
                    load_tensor = gate_stats["load"]
                    for e_idx, load_val in enumerate(load_tensor):
                        out[f"mm_moe_expert_{e_idx}_load"] = load_val
                # 记录路由分布熵
                if "entropy" in gate_stats:
                    out["mm_moe_gate_entropy"] = gate_stats["entropy"]
                if "zloss" in gate_stats:
                    out["mm_moe_gate_zloss"] = gate_stats["zloss"]
                if "invalid_gate_ratio" in gate_stats:
                    out["mm_moe_invalid_gate_ratio"] = gate_stats["invalid_gate_ratio"]
                if "task_route_loss" in gate_stats:
                    out["mm_moe_task_route_loss"] = gate_stats["task_route_loss"]
                if "element_route_loss" in gate_stats:
                    out["mm_moe_element_route_loss"] = gate_stats["element_route_loss"]
                if "task_route_kl" in gate_stats:
                    out["mm_moe_task_route_kl"] = gate_stats["task_route_kl"]
                if "element_route_kl" in gate_stats:
                    out["mm_moe_element_route_kl"] = gate_stats["element_route_kl"]
                if "task_route_effect" in gate_stats:
                    out["mm_moe_task_route_effect"] = gate_stats["task_route_effect"]
                if "element_route_effect" in gate_stats:
                    out["mm_moe_element_route_effect"] = gate_stats["element_route_effect"]
                if "task_element_orth" in gate_stats:
                    out["mm_moe_task_element_orth"] = gate_stats["task_element_orth"]
                if "task_branch_mass" in gate_stats:
                    out["mm_moe_task_branch_mass"] = gate_stats["task_branch_mass"]
                if "element_branch_mass" in gate_stats:
                    out["mm_moe_element_branch_mass"] = gate_stats["element_branch_mass"]

        # 物理约束与逐像素监督
        if self.physics_enabled and fused_spatial is not None:
            H, W = data["rgb"].shape[-2], data["rgb"].shape[-1]

            # 为避免半精度下的数值不稳定导致 NaN，物理解码与损失在 fp32 中计算
            try:
                dev_type = fused_spatial.device.type
            except Exception:
                dev_type = "cuda"

            # 在物理分支禁用 autocast，显式使用 float32 计算
            from torch import autocast as _autocast
            with _autocast(device_type=dev_type, enabled=False):
                fused_spatial_fp32 = fused_spatial.float()
                # 清洗上游产生的 NaN/Inf，避免在物理解码器中扩散
                fused_spatial_fp32 = torch.nan_to_num(
                    fused_spatial_fp32, nan=0.0, posinf=0.0, neginf=0.0
                )
                # 确保物理解码器参数与输入 dtype/device 一致（显式转为 float32）
                try:
                    self.physics.to(device=fused_spatial_fp32.device, dtype=torch.float32)
                except Exception:
                    # 兼容部分环境不支持 module.to(dtype=...) 的情况
                    for p in self.physics.parameters():
                        p.data = p.data.to(dtype=torch.float32, device=fused_spatial_fp32.device)
                    for name, buffer in self.physics.named_buffers():
                        buffer.data = buffer.data.to(dtype=torch.float32, device=fused_spatial_fp32.device)
                phy_pred = self.physics(
                    fused_spatial=fused_spatial_fp32,
                    pyramid_raw=pyramid_raw,
                    input_size=(H, W),
                )
            # 如果物理解码输出仍出现非有限值，记录并回退为零以避免损失为 NaN
            phy_maps = [
                phy_pred.get("phy_full"),
                phy_pred.get("phy_s1"),
                phy_pred.get("phy_s2"),
                phy_pred.get("phy_s3"),
            ]
            has_nan = any([
                (m is not None and not torch.isfinite(m).all()) for m in phy_maps
            ])
            if has_nan:
                logger.warning("Physics decoder produced non-finite values; sanitizing outputs.")
                for k in ["phy_full", "phy_s1", "phy_s2", "phy_s3"]:
                    if k in phy_pred and phy_pred[k] is not None:
                        phy_pred[k] = torch.nan_to_num(phy_pred[k], nan=0.0, posinf=0.0, neginf=0.0)
                out.update({"phy_has_nan": torch.tensor(1.0, device=fused_spatial.device)})
            else:
                out.update({"phy_has_nan": torch.tensor(0.0, device=fused_spatial.device)})
            # 训练日志仅接受标量；不将大尺寸的预测图加入返回，避免日志崩溃

            # 先计算与GT无关的正则项
            tv_w = self.physics_tv_weight
            cons_w = self.physics_consistency_weight
            reg_tv = (
                total_variation_loss(phy_pred["phy_full"]) if tv_w > 0 else fused_spatial.new_zeros((), dtype=torch.float32)
            )
            reg_cons = (
                consistency_loss_multiscale(phy_pred["phy_full"], phy_pred["phy_s1"], phy_pred["phy_s2"], phy_pred["phy_s3"]) if cons_w > 0 else fused_spatial.new_zeros((), dtype=torch.float32)
            )

            # 监督：支持多尺度与全分辨率
            phy_gt = None
            if "phy_gt" in data:
                phy_gt = data["phy_gt"]
            elif "phy_gt_full" in data:
                phy_gt = data["phy_gt_full"]

            # 若无显式 phy_gt，且启用自动物理真值生成，则尝试约束推导
            if phy_gt is None and bool(getattr(self, "physics_enabled", False)):
                auto_gt_cfg = bool(self.config.physics.get("auto_gt_from_constraints", False))
                if auto_gt_cfg:
                    rte_coeffs = self.config.physics.get("rte_tsm_coeffs", None)
                    phy_gt = compute_rte_rrs_gt(data, rte_coeffs)
                    if phy_gt is None:
                        coeffs = self.config.physics.get("sar_sigma0_coeffs", None)
                        phy_gt = compute_sar_sigma0_gt(data, coeffs)

            if phy_gt is not None:
                # 对齐 GT 尺寸到各尺度
                gt_full = F.interpolate(phy_gt, size=(H, W), mode="bilinear", align_corners=False) if phy_gt.shape[-2:] != (H, W) else phy_gt
                l_full = pixelwise_mse(phy_pred["phy_full"], gt_full)

                # 多尺度 GT
                s1_sz = phy_pred["phy_s1"].shape[-2:]
                s2_sz = phy_pred["phy_s2"].shape[-2:]
                s3_sz = phy_pred["phy_s3"].shape[-2:]
                gt_s1 = F.interpolate(gt_full, size=s1_sz, mode="bilinear", align_corners=False)
                gt_s2 = F.interpolate(gt_full, size=s2_sz, mode="bilinear", align_corners=False)
                gt_s3 = F.interpolate(gt_full, size=s3_sz, mode="bilinear", align_corners=False)

                l_s1 = pixelwise_mse(phy_pred["phy_s1"], gt_s1)
                l_s2 = pixelwise_mse(phy_pred["phy_s2"], gt_s2)
                l_s3 = pixelwise_mse(phy_pred["phy_s3"], gt_s3)

                # 边缘保持项（仅在全分辨率上）
                edge_w = self.physics_edge_weight
                l_edge = edge_preserve_loss(phy_pred["phy_full"], gt_full) if edge_w > 0 else gt_full.new_zeros((), dtype=torch.float32)

                # 光谱损失（需要多通道GT）
                spec_w = self.physics_spectral_weight
                spec_l = (
                    spectral_loss_sam(phy_pred["phy_full"], gt_full) if spec_w > 0 and gt_full.shape[1] > 1 else gt_full.new_zeros((), dtype=torch.float32)
                )

                # 权重汇总
                # 基础权重（可从配置进一步扩展）
                w_full = 1.0
                w_s1 = 0.5
                w_s2 = 0.35
                w_s3 = 0.25
                physics_weight = self.physics_loss_weight

                phy_loss = (
                    w_full * l_full + w_s1 * l_s1 + w_s2 * l_s2 + w_s3 * l_s3
                ) + edge_w * l_edge + tv_w * reg_tv + cons_w * reg_cons + spec_w * spec_l
                # 保持与文本损失相同的数据类型，避免混合精度导致的类型不一致
                total_loss += (physics_weight * phy_loss).to(text_loss.dtype)
                out.update({
                    "phy_loss": phy_loss,
                    "phy_tv_loss": reg_tv,
                    "phy_consistency_loss": reg_cons,
                    "phy_spectral_loss": spec_l,
                })
            else:
                # 无GT时仅加入正则项
                physics_weight = self.physics_loss_weight
                phy_loss = tv_w * reg_tv + cons_w * reg_cons
                total_loss += (physics_weight * phy_loss).to(text_loss.dtype)
                out.update({
                    "phy_loss": phy_loss,
                    "phy_tv_loss": reg_tv,
                    "phy_consistency_loss": reg_cons,
                })

        # 养殖区分割（MOE）
        has_seg_supervision = ("seg_mask" in data and data["seg_mask"] is not None)
        run_seg_branch = self.seg_enabled and fused_spatial is not None and (has_seg_supervision or not self.training)
        if run_seg_branch:
            H, W = data["rgb"].shape[-2], data["rgb"].shape[-1]
            # 可选模态输入：若数据集提供光学/SAR 空间特征，可传入供模态门使用
            modal_inputs = data.get("modal_inputs", None)
            context_flags = {"cloudy": bool(data.get("cloudy", False))}

            # 纯文本语义路由输入
            task_text_embs = data.get("task_text_embs", None)
            element_text_embs = data.get("element_text_embs", None)
            task_text_mask = data.get("task_text_attention_mask", None)
            element_text_mask = data.get("element_text_attention_mask", None)

            seg_out = self.seg_head(
                fused_spatial,
                input_size=(H, W),
                modal_inputs=modal_inputs,
                context=context_flags,
                task_text_embs=task_text_embs,
                element_text_embs=element_text_embs,
                task_text_mask=task_text_mask,
                element_text_mask=element_text_mask,
            )
            seg_logits = seg_out["logits"]
            if not self.training:
                out.update({"seg_logits": seg_logits})
            if "task_bias_norm" in seg_out:
                out["seg_task_bias_norm"] = seg_out["task_bias_norm"]
            if "element_bias_norm" in seg_out:
                out["seg_element_bias_norm"] = seg_out["element_bias_norm"]
            # 门控统计信息（空间加权平均）
            if self.seg_log_gate_stats and "gate_logits" in seg_out:
                gate_logits = seg_out["gate_logits"]  # [B,E,h,w]
                gate_w = torch.softmax(gate_logits.float(), dim=1)
                # 每个专家的平均权重
                gate_mean = gate_w.mean(dim=(0, 2, 3))  # [E]
                # Top-1 专家占比
                top_idx = torch.argmax(gate_w, dim=1)  # [B,h,w]
                E = gate_w.shape[1]
                top_ratio = []
                total = float(top_idx.numel())
                for e in range(E):
                    frac = (top_idx == e).float().sum() / total
                    top_ratio.append(frac)
                # 展开为标量键，便于日志与可视化
                for e in range(E):
                    out[f"seg_gate_mean_e{e}"] = gate_mean[e]
                    out[f"seg_gate_top1_ratio_e{e}"] = top_ratio[e]
                # 路由分布的平均熵（标量）
                eps = 1e-8
                seg_gate_entropy = (-(gate_w * (gate_w + eps).log()).sum(dim=1)).mean()
                out["seg_gate_entropy"] = seg_gate_entropy
            if has_seg_supervision:
                target = data["seg_mask"]  # [B,H,W], int64
                # CrossEntropy
                ce = F.cross_entropy(seg_logits, target, ignore_index=self.seg_ignore_index)
                # Dice（在 float32 概率上计算）
                dl = dice_loss(seg_logits, target, num_classes=self.seg_head.num_classes, ignore_index=self.seg_ignore_index)
                seg_loss = self.seg_ce_weight * ce + self.seg_dice_weight * dl
                total_loss += (self.seg_loss_weight * seg_loss).to(text_loss.dtype)
                out.update({"seg_loss": seg_loss, "seg_ce_loss": ce, "seg_dice_loss": dl})
            # 路由器辅助损失（仅在训练时返回并纳入总损失）
            if self.training and has_seg_supervision and ("router_aux" in seg_out):
                aux = seg_out["router_aux"]
                aux_entropy = aux.get("entropy", None)
                aux_zloss = aux.get("zloss", None)
                aux_loss = seg_logits.new_zeros(())
                if aux_entropy is not None and self.seg_router_entropy_coef > 0:
                    aux_loss = aux_loss + self.seg_router_entropy_coef * aux_entropy
                    out["seg_router_entropy"] = aux_entropy
                if aux_zloss is not None and self.seg_router_zloss_coef > 0:
                    aux_loss = aux_loss + self.seg_router_zloss_coef * aux_zloss
                    out["seg_router_zloss"] = aux_zloss
                if aux_loss is not None:
                    total_loss += aux_loss.to(text_loss.dtype)
                    out["seg_router_aux_loss"] = aux_loss
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

    def custom_save_checkpoint(self, file_name: str):
        fp32_ckpt = get_fp32_state_dict_from_zero_checkpoint(file_name)
        vision_ckpt = get_rgb_maybe_zero_3(fp32_ckpt.items())
        other_ckpt = get_other_maybe_zero_3(fp32_ckpt.items())

        if self.stage >= 2:
            file_name = pathlib.Path(file_name)
            if file_name.is_file():
                loar_output_path = file_name.parent / "TextLoRA"
            else:
                loar_output_path = file_name / "TextLoRA"
            self.language.text_encoder.save_pretrained(str(loar_output_path))

        return dict(vision_ckpt=vision_ckpt, other_ckpt=other_ckpt)

    def custom_export_fp32_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        本地导出 FP32 权重，不依赖 DeepSpeed 已写入的 Zero 检查点。

        用于 NPU/HCCL 环境下的容错保存路径：直接从当前模块聚合 ZeRO-3 参数并转 CPU。
        """
        # 直接遍历当前模型参数；maybe_zero_3 在 ZeRO-3 下会自动聚合
        named_params = list(self.named_parameters())
        # 视觉分支参数（剥离前缀并转为 CPU Tensor）
        vision_ckpt = get_rgb_maybe_zero_3(named_params)
        # 其他关键模块参数（统一转为 CPU Tensor）
        other_ckpt = get_other_maybe_zero_3(named_params)
        return dict(vision_ckpt=vision_ckpt, other_ckpt=other_ckpt)
    # def custom_save_checkpoint(self, file_name: str):
    #     # 转换 ZeRO 检查点至 FP32 完整权重
    #     fp32_ckpt = get_fp32_state_dict_from_zero_checkpoint(file_name)
    #
    #     # 提取视觉和多模态参数（保留前缀）
    #     vision_ckpt = {k: v for k, v in fp32_ckpt.items() if k.startswith("vision.")}
    #     multimodal_ckpt = {k: v for k, v in fp32_ckpt.items() if k.startswith("multimodal.")}
    #
    #     # 保存 LoRA 适配器（如果需要）
    #     if self.stage >= 2:
    #         file_name = pathlib.Path(file_name)
    #         if file_name.is_file():
    #             loar_output_path = file_name.parent / "TextLoRA"
    #         else:
    #             loar_output_path = file_name / "TextLoRA"
    #         self.language.text_encoder.save_pretrained(str(loar_output_path))
    #     # if self.stage >= 2:
    #     #     output_dir = pathlib.Path(file_name).parent / "TextLoRA"
    #     # self.language.text_encoder.save_pretrained(str(output_dir))
    #
    #     # 返回完整参数名结构的字典
    #     return dict(vision_ckpt=vision_ckpt, other_ckpt=other_ckpt)
    #     # return {
    #     #     "vision": vision_ckpt,
    #     #     "multimodal": multimodal_ckpt,
    #     #     "language": {k: v for k, v in fp32_ckpt.items() if k.startswith("language.")}
    #     # }

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
        self.vision.encoder.load_state_dict(ckpt, strict=False)

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


        def _report_load_result(module_name, incompatible):
            print(
                f"After loading {module_name}: Missing: {incompatible.missing_keys}. "
                f"Unexpected: {incompatible.unexpected_keys}"
            )
            if not strict and incompatible.missing_keys:
                legacy_gate_keys = [
                    key for key in incompatible.missing_keys
                    if key.startswith("task_gate") or key.startswith("element_gate")
                ]
                if legacy_gate_keys:
                    print(
                        "Detected a legacy multimodal checkpoint without two-stage routing gates; "
                        "the missing gate parameters will keep their current initialization."
                    )

        if any(key.startswith('module') for key in ckpt.keys()):
            # filtered_state_dict = {k: v for k, v in ckpt["module"].items() if k.startswith("multimodal.")}
            filtered_state_dict = {k: v for k, v in ckpt["module"].items() if
                                   k.startswith("multimodal.") or k.startswith("vision.")}
            msg = self.load_state_dict(filtered_state_dict, strict=False)
            _report_load_result("model", msg)

        elif any(key.startswith('vision_ckpt') for key in ckpt.keys()) :
            vision_ckpt = ckpt["vision_ckpt"]
            multimodal_ckpt = ckpt["other_ckpt"]["multimodal_projection"]
            msg = self.vision.load_state_dict(vision_ckpt, strict=strict)
            _report_load_result("vision", msg)
            msg = self.multimodal.projection.load_state_dict(multimodal_ckpt, strict=strict)
            _report_load_result("multimodal", msg)

        elif any(key.startswith('rgb_ckpt') for key in ckpt.keys()) :
            vision_ckpt = ckpt["rgb_ckpt"]
            multimodal_ckpt = ckpt["other_ckpt"]["rgb_pooler"]
            msg = self.vision.load_state_dict(vision_ckpt, strict=strict)
            _report_load_result("vision", msg)
            msg = self.multimodal.projection.load_state_dict(multimodal_ckpt, strict=strict)
            _report_load_result("multimodal", msg)
        # if "vision" in ckpt:
        #     self.vision.load_state_dict(ckpt["vision"], strict=strict)
        # if "multimodal" in ckpt:
        #     self.multimodal.load_state_dict(ckpt["multimodal"], strict=strict)
        # if "language" in ckpt:
        #     self.language.load_state_dict(ckpt["language"], strict=strict)

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

        text_encoder = self.language.get_text_encoder()
        if freeze_text:
            self.language.eval()  # 将文本编码器设置为评估模式
            # Stage-1 需要冻结完整语言分支，而不只是词嵌入层
            for p in self.language.parameters():
                p.requires_grad = False
            for p in text_encoder.parameters():
                p.requires_grad = False
        else:
            self.language.train()
            # 即使训练文本分支，也保持 token 输入/输出嵌入冻结，避免词表漂移
            for p in text_encoder.get_input_embeddings().parameters():
                p.requires_grad = False
            for p in text_encoder.get_output_embeddings().parameters():
                p.requires_grad = False

        # 多模态相关参数是否训练
        for param in self.multimodal.parameters():
            if tune_multimodal:
                param.requires_grad = True
            else:
                param.requires_grad = False
            param.data = param.data.to(dtype=compute_dtype)

        # 物理分支：为避免 AMP 下 dtype 不一致导致的算子报错/NaN，强制使用 float32
        if getattr(self, "physics_enabled", False) and hasattr(self, "physics"):
            for p in self.physics.parameters():
                # 物理分支通常需要训练以优化约束映射
                p.requires_grad = True
                p.data = p.data.to(dtype=torch.float32)
            for name, buffer in self.physics.named_buffers():
                buffer.data = buffer.data.to(dtype=torch.float32)

        # 分割分支：采用 compute_dtype 参与 AMP 训练
        if getattr(self, "seg_enabled", False) and hasattr(self, "seg_head"):
            for p in self.seg_head.parameters():
                p.requires_grad = True
                p.data = p.data.to(dtype=compute_dtype)
            for name, buffer in self.seg_head.named_buffers():
                buffer.data = buffer.data.to(dtype=compute_dtype)

        if tune_im_start and freeze_text:
            # 如果 tune_im_start 为 True 且文本被冻结，则解冻输入嵌入
            for p in text_encoder.get_input_embeddings().parameters():
                p.requires_grad = True
            # 输出嵌入保持冻结状态（已在前面设置，此处无需重复）

        if model_path is not None:
            # 如果提供了模型路径，则加载模型
            self.custom_load_state_dict(model_path)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logger.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_other_maybe_zero_3(named_params):
    # 定义需要处理的键名
    names = ["multimodal.projection", "embed_tokens", "physics"]
    multimodal_projection = dict()
    text_proj = dict()
    embed_tokens = dict()
    lm_head = dict()
    physics_ckpt = dict()

    # 将输入的 named_params 转换为列表，方便遍历
    params = list(named_params)
    # 初始化 to_return 字典，键名需要与 names 列表中的键名一致
    to_return = dict(
        multimodal_projection=multimodal_projection,
        embed_tokens=embed_tokens,
        text_proj=text_proj,
        lm_head=lm_head,
        physics=physics_ckpt,
    )
    # 遍历参数
    for k, v in params:
        for name in names:
            if name in k:
                # 使用 name 作为键名，而不是其他变量名
                # 这里需要将 name 映射到 to_return 的键名
                if name == "multimodal.projection":
                    to_return["multimodal_projection"][k.split(name + ".")[-1]] = maybe_zero_3(v, ignore_status=True, name=k)
                elif name == "embed_tokens":
                    to_return["embed_tokens"][k.split(name + ".")[-1]] = maybe_zero_3(v, ignore_status=True, name=k)
                elif name == "physics":
                    to_return["physics"][k.split(name + ".")[-1]] = maybe_zero_3(v, ignore_status=True, name=k)

    return to_return
# def get_other_maybe_zero_3(named_params):
#     names = ["multimodal.projection", "embed_tokens"]
#     to_return = {"multimodal_projection": {}, "embed_tokens": {}}
#     for k, v in named_params:
#         if any(name in k for name in names):
#             if "multimodal.projection" in k:
#                 to_return["multimodal_projection"][k] = v  # 保留完整键名
#             elif "embed_tokens" in k:
#                 to_return["embed_tokens"][k] = v           # 保留完整键名
#     return to_return


def get_rgb_maybe_zero_3(named_params):
    to_return = {k[len("vision.") :]: t for k, t in named_params if "vision." in k}
    # to_return = {k: t for k, t in named_params if k.startswith("vision.")}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return
