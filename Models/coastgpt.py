import logging
import os
import pathlib
from typing import Dict, List

import ml_collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.utils.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint,
)
from peft import PeftModel
from .wavelet_adapter import MultiBandDirectAdapter, apply_multiband_adapter
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

        wavelet_cfg = getattr(config, "wavelet_adapter", ml_collections.ConfigDict())
        self.wavelet_adapter_enabled = bool(wavelet_cfg.get("enabled", False))
        self.wavelet_adapter_mode = str(wavelet_cfg.get("mode", "learnable_direct"))
        self.wavelet_adapter_normalize_output = bool(wavelet_cfg.get("normalize_output", True))
        wavelet_output_mean = wavelet_cfg.get("output_mean", [0.485, 0.456, 0.406])
        wavelet_output_std = wavelet_cfg.get("output_std", [0.229, 0.224, 0.225])
        self.register_buffer(
            "wavelet_output_mean",
            torch.tensor(wavelet_output_mean, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "wavelet_output_std",
            torch.tensor(wavelet_output_std, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.wavelet_adapter = None
        if self.wavelet_adapter_enabled:
            if self.wavelet_adapter_mode != "learnable_direct":
                raise ValueError(f"unsupported wavelet_adapter.mode: {self.wavelet_adapter_mode}")
            in_channels = int(wavelet_cfg.get("in_channels", wavelet_cfg.get("multiband_channels", 4)))
            target_channels = int(wavelet_cfg.get("target_channels", 3))
            init = str(wavelet_cfg.get("init", "rgb_from_bgrn"))
            self.wavelet_adapter = MultiBandDirectAdapter(
                in_channels=in_channels,
                target_channels=target_channels,
                output_size=None,
                init=init,
            )

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

    def _maybe_apply_wavelet_adapter(self, data: Dict, out: Dict) -> None:
        if not getattr(self, "wavelet_adapter_enabled", False) or self.wavelet_adapter is None:
            return
        multiband = data.get("multiband", None)
        if multiband is None:
            return

        reference_rgb = data.get("rgb", None)
        adapted_rgb, meta = apply_multiband_adapter(
            self.wavelet_adapter,
            multiband,
            reference_rgb=reference_rgb if torch.is_tensor(reference_rgb) else None,
        )
        if getattr(self, "wavelet_adapter_normalize_output", True):
            mean = self.wavelet_output_mean.to(device=adapted_rgb.device, dtype=adapted_rgb.dtype)
            std = self.wavelet_output_std.to(device=adapted_rgb.device, dtype=adapted_rgb.dtype)
            adapted_rgb = (adapted_rgb - mean) / std
        if torch.is_tensor(reference_rgb):
            valid_multiband = data.get("valid_multiband", None)
            if torch.is_tensor(valid_multiband):
                valid = valid_multiband.to(device=adapted_rgb.device, dtype=torch.bool).view(-1, 1, 1, 1)
                reference = reference_rgb.to(device=adapted_rgb.device, dtype=adapted_rgb.dtype)
                adapted_rgb = torch.where(valid, adapted_rgb, reference)
                out["wavelet_valid_multiband_ratio"] = valid_multiband.float().mean().to(adapted_rgb.device)
        data["rgb"] = adapted_rgb
        if "extra_band_weight_l1" in meta:
            out["wavelet_extra_band_weight_l1"] = self.wavelet_adapter.extra_band_weight_l1()

    def _add_prefix_first_token_loss(
        self,
        data: Dict,
        multimodal_embedding: torch.Tensor,
        total_loss: torch.Tensor,
        out: Dict,
    ) -> torch.Tensor:
        prefix_cfg = getattr(self.config, "prefix_first_token_loss", ml_collections.ConfigDict())
        prefix_enabled = bool(prefix_cfg.get("enabled", False))
        prefix_weight = float(prefix_cfg.get("weight", 0.0) or 0.0)
        if not prefix_enabled or prefix_weight <= 0:
            return total_loss

        prefix_loss, prefix_stats = self.language.prefix_first_token_loss(
            data,
            image_embedding=multimodal_embedding,
        )
        for stat_name, stat_value in prefix_stats.items():
            out[stat_name] = stat_value
        if prefix_loss is None:
            return total_loss

        prefix_loss_weighted = (prefix_weight * prefix_loss).to(total_loss.dtype)
        out.update({
            "prefix_first_token_loss": prefix_loss,
            "prefix_first_token_loss_weighted": prefix_loss_weighted,
        })
        return total_loss + prefix_loss_weighted

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

        self._maybe_apply_wavelet_adapter(data, out)

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

        total_loss = self._add_prefix_first_token_loss(
            data=data,
            multimodal_embedding=multimodal_embedding,
            total_loss=total_loss,
            out=out,
        )

        if hasattr(self.multimodal, "get_aux_loss"):
            # 获取辅助损失
            mm_aux_loss = self.multimodal.get_aux_loss()
            
            # 定义辅助损失的权重（建议在 0.01 左右，防止其梯度压过主任务 text_loss）
            # 最好从 config 中读取，这里做个 fallback 默认给 0.01
            mm_aux_weight = getattr(self.config, "mm_moe_aux_weight", 0.01) 
            
            if mm_aux_loss is not None:
                # 累加到总损失中，避免原地修改 text_loss 本身，保证日志口径清晰
                mm_aux_loss_weighted = (mm_aux_weight * mm_aux_loss).to(total_loss.dtype)
                total_loss = total_loss + mm_aux_loss_weighted
                
                # 更新到 out 字典，便于日志监控
                out.update({
                    "mm_moe_aux_loss_raw": mm_aux_loss,
                    "mm_moe_aux_loss_weighted": mm_aux_loss_weighted
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
                total_loss = total_loss + (physics_weight * phy_loss).to(text_loss.dtype)
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
                total_loss = total_loss + (physics_weight * phy_loss).to(text_loss.dtype)
                out.update({
                    "phy_loss": phy_loss,
                    "phy_tv_loss": reg_tv,
                    "phy_consistency_loss": reg_cons,
                })
        out.update({"non_text_loss": total_loss - text_loss})
        out.update({"total_loss": total_loss})

        return out

    def _embed_semantic_text_ids(self, input_ids, attention_mask=None):
        if input_ids is None or not torch.is_tensor(input_ids):
            return None
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
            if not getattr(self, "_warned_generate_semantic_emb_layer", False):
                logger.warning(
                    "Failed to get semantic embedding layer for generation: %s", exc
                )
                self._warned_generate_semantic_emb_layer = True
        if emb_layer is None:
            return None

        input_ids = input_ids.to(device=emb_layer.weight.device)
        embs = emb_layer(input_ids)
        if attention_mask is not None and torch.is_tensor(attention_mask):
            mask = attention_mask.to(device=embs.device)
            embs = embs * mask.unsqueeze(-1).to(embs.dtype)
        return embs

    def encode_image(
            self,
            image,
            pool,
            physical_prompt_embs=None,
            task_text_embs=None,
            element_text_embs=None,
            physical_prompt_attention_mask=None,
            task_text_attention_mask=None,
            element_text_attention_mask=None,
    ):
        """
        将输入图像编码为嵌入向量。

        参数:
            image: 输入图像张量
            pool: 布尔值，指示是否对嵌入向量进行池化

        返回:
            图像嵌入向量（池化或未池化）
        """
        # 从视觉模型获取原始图像嵌入。DualVisionEncoder 训练时使用
        # encode_with_spatial + physical_prompt_embs，生成时必须对齐。
        if isinstance(self.vision, DualVisionEncoder):
            image_embedding, _, _ = self.vision.encode_with_spatial(
                image,
                physical_prompt_embs=physical_prompt_embs,
            )
        else:
            image_embedding = self.vision.encode(image)
        image_embedding = self.multimodal.encode_test(
            image_embedding,
            physical_prompts=physical_prompt_embs,
            task_text_embs=task_text_embs,
            element_text_embs=element_text_embs,
            physical_prompt_mask=physical_prompt_attention_mask,
            task_text_mask=task_text_attention_mask,
            element_text_mask=element_text_attention_mask,
        )
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
            multiband: torch.Tensor = None,
            valid_multiband: torch.Tensor = None,
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
            physical_prompt_ids = kwargs.pop("physical_prompt_ids", None)
            physical_prompt_attention_mask = kwargs.pop("physical_prompt_attention_mask", None)
            task_text_ids = kwargs.pop("task_text_ids", None)
            task_text_attention_mask = kwargs.pop("task_text_attention_mask", None)
            element_text_ids = kwargs.pop("element_text_ids", None)
            element_text_attention_mask = kwargs.pop("element_text_attention_mask", None)
            physical_prompt_embs = self._embed_semantic_text_ids(
                physical_prompt_ids,
                attention_mask=physical_prompt_attention_mask,
            )
            task_text_embs = self._embed_semantic_text_ids(
                task_text_ids,
                attention_mask=task_text_attention_mask,
            )
            element_text_embs = self._embed_semantic_text_ids(
                element_text_ids,
                attention_mask=element_text_attention_mask,
            )
            if multiband is not None:
                wavelet_data = {"rgb": images, "multiband": multiband}
                if valid_multiband is not None:
                    wavelet_data["valid_multiband"] = valid_multiband
                wavelet_out = {}
                self._maybe_apply_wavelet_adapter(wavelet_data, wavelet_out)
                images = wavelet_data.get("rgb", images)
            # 如果提供了图像，编码为嵌入向量（不池化）
            image_embedding = self.encode_image(
                images,
                pool=False,
                physical_prompt_embs=physical_prompt_embs,
                task_text_embs=task_text_embs,
                element_text_embs=element_text_embs,
                physical_prompt_attention_mask=physical_prompt_attention_mask,
                task_text_attention_mask=task_text_attention_mask,
                element_text_attention_mask=element_text_attention_mask,
            )
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

    def _save_text_lora_checkpoint(self, file_name: str) -> None:
        if self.stage < 2:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
        file_path = pathlib.Path(file_name)
        if file_path.is_file():
            loar_output_path = file_path.parent / "TextLoRA"
        else:
            loar_output_path = file_path / "TextLoRA"
        self.language.text_encoder.save_pretrained(str(loar_output_path))

    def custom_save_checkpoint(self, file_name: str):
        # Only attempt ZeRO consolidation if a "latest" pointer is present AND
        # points to a global_step* dir that actually exists. Otherwise stale
        # global_step dirs from previous runs would be silently merged in,
        # producing a checkpoint that mixes old embed_tokens / lm_head with
        # current LoRA adapters, which is a very hard-to-debug failure mode.
        latest_pointer = os.path.join(file_name, "latest")
        zero_ok = False
        if os.path.isfile(latest_pointer):
            try:
                with open(latest_pointer, "r") as fh:
                    target_tag = fh.read().strip()
            except Exception:
                target_tag = ""
            if target_tag and os.path.isdir(os.path.join(file_name, target_tag)):
                zero_ok = True

        if zero_ok:
            try:
                fp32_ckpt = get_fp32_state_dict_from_zero_checkpoint(file_name)
                vision_ckpt = get_rgb_maybe_zero_3(fp32_ckpt.items())
                other_ckpt = get_other_maybe_zero_3(fp32_ckpt.items())
                state_dict = dict(vision_ckpt=vision_ckpt, other_ckpt=other_ckpt)
            except Exception as exc:
                logger.warning(
                    "ZeRO consolidation failed for %s (%s); falling back to local FP32 export.",
                    file_name,
                    exc,
                )
                state_dict = self.custom_export_fp32_state()
        else:
            logger.info(
                "No valid DeepSpeed 'latest' pointer at %s; using live model.named_parameters() "
                "for FP32 export to avoid loading stale ZeRO shards from previous runs.",
                file_name,
            )
            state_dict = self.custom_export_fp32_state()

        self._save_text_lora_checkpoint(file_name)
        return state_dict

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

        # Authoritative grab of embed_tokens and lm_head straight from the live
        # PEFT-wrapped text encoder. Without this, the pattern-match in
        # get_other_maybe_zero_3 plus HF's _keep_in_fp32_modules behaviour can
        # silently capture an initial reference instead of the trained tensor.
        # Also captures lm_head, which the legacy names list missed entirely.
        try:
            text_encoder = self.language.get_text_encoder()
            input_emb = text_encoder.get_input_embeddings()
            # Pick a probe row index that's safely in-range for both legacy
            # vocab=33000 (loc-token) layouts and the new vocab=32000 layouts.
            def _probe_row(weight: torch.Tensor) -> int:
                return min(31999, weight.shape[0] - 1)

            if input_emb is not None and hasattr(input_emb, "weight"):
                live_in = maybe_zero_3(input_emb.weight, ignore_status=True, name="live_embed_tokens")
                other_ckpt.setdefault("embed_tokens", {})["weight"] = live_in
                probe = _probe_row(input_emb.weight)
                logger.info(
                    "[save] live embed_tokens[%d, :5]=%s shape=%s",
                    probe,
                    input_emb.weight.data[probe, :5].detach().cpu().float().tolist(),
                    tuple(input_emb.weight.shape),
                )
            output_emb = text_encoder.get_output_embeddings()
            if output_emb is not None and hasattr(output_emb, "weight"):
                live_out = maybe_zero_3(output_emb.weight, ignore_status=True, name="live_lm_head")
                other_ckpt.setdefault("lm_head", {})["weight"] = live_out
                probe = _probe_row(output_emb.weight)
                logger.info(
                    "[save] live lm_head[%d, :5]=%s shape=%s",
                    probe,
                    output_emb.weight.data[probe, :5].detach().cpu().float().tolist(),
                    tuple(output_emb.weight.shape),
                )
        except Exception as exc:
            logger.warning("Live embed/lm_head capture failed: %s", exc)

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

    def _restore_embed_tokens_from_ckpt(self, ckpt, report) -> None:
        """Best-effort restore of text embedding rows saved in older checkpoints.

        Looks for ``ckpt["other_ckpt"]["embed_tokens"]`` (and optional
        ``lm_head``) and copies them onto the current text encoder. Silently
        no-ops when the keys are absent. When an older checkpoint has a larger
        vocabulary than the current tokenizer, restore only the aligned rows so
        current non-bin-token inference can still use the trained text weights.
        """
        other = ckpt.get("other_ckpt") if isinstance(ckpt, dict) else None
        if not isinstance(other, dict):
            return

        def _restore_layer_weight(layer, ckpt_weight, module_name: str) -> None:
            target_shape = tuple(layer.weight.shape)
            ckpt_shape = tuple(ckpt_weight.shape)
            if ckpt_shape == target_shape:
                msg = layer.load_state_dict({"weight": ckpt_weight}, strict=False)
                report(module_name, msg)
                return
            if len(ckpt_shape) == 2 and len(target_shape) == 2 and ckpt_shape[1] == target_shape[1]:
                rows = min(ckpt_shape[0], target_shape[0])
                with torch.no_grad():
                    layer.weight[:rows].copy_(
                        ckpt_weight[:rows].to(device=layer.weight.device, dtype=layer.weight.dtype)
                    )
                print(
                    f"[Inference] {module_name} shape mismatch: "
                    f"ckpt={ckpt_shape} vs model={target_shape}; "
                    f"restored first {rows} aligned rows without tokenizer resize."
                )
                return
            print(
                f"[Inference] {module_name} shape mismatch: "
                f"ckpt={ckpt_shape} vs model={target_shape}; keeping current init."
            )

        emb = other.get("embed_tokens")
        if isinstance(emb, dict) and "weight" in emb:
            try:
                text_encoder = self.language.get_text_encoder()
                input_layer = text_encoder.get_input_embeddings()
                ckpt_weight = emb["weight"]
                _restore_layer_weight(input_layer, ckpt_weight, "embed_tokens")
            except Exception as exc:
                print(f"[Inference] failed to restore embed_tokens: {exc}")
        lm = other.get("lm_head")
        if isinstance(lm, dict) and "weight" in lm:
            try:
                text_encoder = self.language.get_text_encoder()
                output_layer = text_encoder.get_output_embeddings()
                if output_layer is not None:
                    ckpt_weight = lm["weight"]
                    _restore_layer_weight(output_layer, ckpt_weight, "lm_head")
            except Exception as exc:
                print(f"[Inference] failed to restore lm_head: {exc}")

    @staticmethod
    def _unwrap_text_lora_base_encoder(text_encoder):
        while isinstance(text_encoder, PeftModel):
            text_encoder = text_encoder.model
        return text_encoder

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
                                   k.startswith("multimodal.") or k.startswith("vision.") or k.startswith("wavelet_adapter.")}
            msg = self.load_state_dict(filtered_state_dict, strict=False)
            _report_load_result("model", msg)

        elif any(key.startswith('vision_ckpt') for key in ckpt.keys()) :
            vision_ckpt = ckpt["vision_ckpt"]
            other_ckpt = ckpt.get("other_ckpt", {})
            multimodal_ckpt = other_ckpt["multimodal_projection"]
            msg = self.vision.load_state_dict(vision_ckpt, strict=strict)
            _report_load_result("vision", msg)
            msg = self.multimodal.projection.load_state_dict(multimodal_ckpt, strict=strict)
            _report_load_result("multimodal", msg)
            wavelet_ckpt = other_ckpt.get("wavelet_adapter")
            if self.wavelet_adapter is not None and isinstance(wavelet_ckpt, dict):
                msg = self.wavelet_adapter.load_state_dict(wavelet_ckpt, strict=False)
                _report_load_result("wavelet_adapter", msg)
            # Restore text-encoder input embeddings for compatibility with
            # older checkpoints that saved these tensors separately.
            self._restore_embed_tokens_from_ckpt(ckpt, _report_load_result)

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
            print(f"[Inference] loading TextLoRA from: {text_path}")
            lora_cfg = getattr(self.config, "lora", None)
            train_text_lora = bool(getattr(lora_cfg, "enable", False)) and self.stage >= 2
            base_text_encoder = self._unwrap_text_lora_base_encoder(self.language.text_encoder)
            if base_text_encoder is not self.language.text_encoder:
                print("[Inference] unwrap existing TextLoRA adapter before checkpoint load.")
            # 如果 TextLoRA 目录存在，则加载文本 LoRA
            self.language.text_encoder = PeftModel.from_pretrained(
                base_text_encoder,
                text_path,
                is_trainable=train_text_lora,
                torch_dtype=torch.float16,  # 使用 float16 数据类型
            )
            print("[Inference] TextLoRA load finished.")

            merge_text_lora = bool(getattr(self.config, "merge_text_lora", self.stage == 0))
            if self.stage == 0 and merge_text_lora:  # Eval 模式
                print("[Inference] merging TextLoRA into base model...")
                # 在评估模式下合并并卸载 PeftModel
                self.language.text_encoder = self.language.text_encoder.merge_and_unload()
                print("[Inference] TextLoRA merge finished.")
            elif self.stage == 0:
                print("[Inference] keep TextLoRA as adapter module without merge.")
        else:
            print(f"[Inference] TextLoRA directory not found, skip LoRA load: {text_path}")
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
            # Freeze input/output embeddings to avoid vocabulary drift on the
            # original LLaMA tokens while LoRA and multimodal layers train.
            for p in text_encoder.get_input_embeddings().parameters():
                p.requires_grad = False
            for p in text_encoder.get_output_embeddings().parameters():
                p.requires_grad = False
            for name, p in text_encoder.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True
                    p.data = p.data.to(dtype=compute_dtype)

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

        if getattr(self, "wavelet_adapter_enabled", False) and self.wavelet_adapter is not None:
            self.wavelet_adapter.train()
            for p in self.wavelet_adapter.parameters():
                p.requires_grad = True
                p.data = p.data.to(dtype=torch.float32)

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
    names = ["multimodal.projection", "embed_tokens", "physics", "wavelet_adapter"]
    multimodal_projection = dict()
    text_proj = dict()
    embed_tokens = dict()
    lm_head = dict()
    physics_ckpt = dict()
    wavelet_adapter_ckpt = dict()

    # 将输入的 named_params 转换为列表，方便遍历
    params = list(named_params)
    # 初始化 to_return 字典，键名需要与 names 列表中的键名一致
    to_return = dict(
        multimodal_projection=multimodal_projection,
        embed_tokens=embed_tokens,
        text_proj=text_proj,
        lm_head=lm_head,
        physics=physics_ckpt,
        wavelet_adapter=wavelet_adapter_ckpt,
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
                elif name == "wavelet_adapter":
                    to_return["wavelet_adapter"][k.split(name + ".")[-1]] = maybe_zero_3(
                        v, ignore_status=True, name=k
                    )

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
