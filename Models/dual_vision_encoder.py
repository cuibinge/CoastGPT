import math
import os
import re
from typing import Dict, Optional, Tuple

import ml_collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

try:
    import open_clip
except Exception as e:
    open_clip = None
    _open_clip_import_error = e



class CrossFrequencyAttention(nn.Module):
   
    def __init__(self, vit_dim, cnn_dim, num_heads=8):
        super().__init__()
        self.q_proj = nn.Linear(vit_dim, vit_dim)
        self.k_proj = nn.Linear(cnn_dim, vit_dim)
        self.v_proj = nn.Linear(cnn_dim, vit_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim=vit_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(vit_dim)
        self.norm2 = nn.LayerNorm(vit_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(vit_dim, vit_dim * 4),
            nn.GELU(),
            nn.Linear(vit_dim * 4, vit_dim)
        )

    def forward(self, z_vit, z_cnn):
        """
        Args:
            z_vit: [B, N_vit, C_vit] 浣庨璇箟瀹忚娴佸舰
            z_cnn: [B, N_cnn, C_cnn] 楂橀绾圭悊寰娴佸舰
        """
        Q = self.q_proj(z_vit)
        K = self.k_proj(z_cnn)
        V = self.v_proj(z_cnn)

        # 娉ㄦ剰鍔涜绠楋細Q (瀹忚) 妫€绱?K (寰)锛屾彁鍙?V (灞€閮ㄧ壒寰?
        attn_out, _ = self.attn(query=Q, key=K, value=V)
        
        # 娈嬪樊杩炴帴锛氫繚鐣欏師濮嬩綆棰戞嫇鎵戠殑鍚屾椂娉ㄥ叆楂橀缁嗚妭
        z_fused = self.norm1(z_vit + attn_out)
        out = self.norm2(z_fused + self.ffn(z_fused))
        return out
class SpatialScaleFusion(nn.Module):
    """
    对应论文描述中的：
    Gi = Conv1x1(Reshape(Gi))
    Li = Conv1x1(Li)
    Zraw = Gi ⊕ Li
    Zi = DW-Conv3x3(GELU(Zraw))

    输入：
        vit_grid: [B, C_vit, H_g, W_g]
        cnn_feat: [B, C_i, H_i, W_i]

    输出：
        z_i: [B, out_dim, H_i, W_i]
    """

    def __init__(self, vit_dim: int, cnn_dim: int, align_dim: int, out_dim: int):
        super().__init__()

        self.vit_align = nn.Conv2d(vit_dim, align_dim, kernel_size=1)
        self.cnn_align = nn.Conv2d(cnn_dim, align_dim, kernel_size=1)

        self.spatial_smooth = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                align_dim * 2,
                align_dim * 2,
                kernel_size=3,
                padding=1,
                groups=align_dim * 2,
                bias=False,
            ),
            nn.Conv2d(
                align_dim * 2,
                out_dim,
                kernel_size=1,
                bias=True,
            ),
            nn.GroupNorm(1, out_dim),
            nn.GELU(),
        )

    def forward(self, vit_grid: torch.Tensor, cnn_feat: torch.Tensor) -> torch.Tensor:
        if vit_grid.dtype != cnn_feat.dtype:
            vit_grid = vit_grid.to(dtype=cnn_feat.dtype)

        vit_resized = F.interpolate(
            vit_grid,
            size=cnn_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        g_i = self.vit_align(vit_resized)
        l_i = self.cnn_align(cnn_feat)

        z_raw = torch.cat([g_i, l_i], dim=1)
        z_i = self.spatial_smooth(z_raw)

        return z_i
class HeterogeneousFusionBlock(nn.Module):
    """
    实现新的混合多尺度视觉 token 构建逻辑：

    1. Stage1 / Stage3 / Stage4：
       通过 ViT Query 做 Cross-Attention，压缩到 196 个 token。
       再与原始 ViT token 在特征维拼接：
       [B, 196, 768] × 4 -> [B, 196, 3072]

    2. Stage2：
       保留高分辨率 28×28 = 784 个 token。
       每个 token 从 C8 投影到 3072 维：
       [B, 784, C8] -> [B, 784, 3072]

    3. 最后序列维拼接：
       [B, 196, 3072] + [B, 784, 3072]
       -> [B, 980, 3072]
    """

    def __init__(
        self,
        cnn_dims,
        vit_dim,
        semantic_grid_size=(14, 14),
        detail_grid_size=(28, 28),
    ):
        super().__init__()

        if len(cnn_dims) < 4:
            raise ValueError("cnn_dims must provide 4 stage dimensions")

        self.vit_dim = int(vit_dim)
        self.out_dim = int(vit_dim) * 4

        self.semantic_grid_size = tuple(semantic_grid_size) if semantic_grid_size is not None else None
        self.detail_grid_size = tuple(detail_grid_size) if detail_grid_size is not None else None

        c4_dim, c8_dim, c16_dim, c32_dim = [int(v) for v in cnn_dims[:4]]

        # Stage1 / Stage3 / Stage4 先投影到 C_vit
        self.pre_align_f4 = nn.Sequential(
            nn.Linear(c4_dim, vit_dim),
            nn.LayerNorm(vit_dim),
        )
        self.pre_align_f16 = nn.Sequential(
            nn.Linear(c16_dim, vit_dim),
            nn.LayerNorm(vit_dim),
        )
        self.pre_align_f32 = nn.Sequential(
            nn.Linear(c32_dim, vit_dim),
            nn.LayerNorm(vit_dim),
        )

        # ViT token 作为 Query，CNN token 作为 Key / Value
        self.align_f4 = CrossFrequencyAttention(vit_dim=vit_dim, cnn_dim=vit_dim)
        self.align_f16 = CrossFrequencyAttention(vit_dim=vit_dim, cnn_dim=vit_dim)
        self.align_f32 = CrossFrequencyAttention(vit_dim=vit_dim, cnn_dim=vit_dim)

        # ViT + Stage1 + Stage3 + Stage4 四路拼接
        self.semantic_logits = nn.Parameter(torch.zeros(4))

        self.semantic_ln = nn.LayerNorm(self.out_dim)
        self.semantic_mlp = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

        # Stage2 高分辨率细节分支：C8 -> 4*C_vit
        self.detail_proj = nn.Sequential(
            nn.Linear(c8_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

        self.detail_ln = nn.LayerNorm(self.out_dim)

        self._last_fusion_weights = None
        self._last_token_info = {}

    @staticmethod
    def _add_2d_sincos_pos_embed(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        x: [B, H*W, C]
        """
        device, dtype = x.device, x.dtype
        c = x.shape[-1]

        quarter = max(c // 4, 1)

        yy, xx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )

        omega = torch.arange(quarter, device=device, dtype=dtype)
        omega = 1.0 / (10000 ** (omega / quarter))

        yy = yy.reshape(-1, 1).to(dtype) * omega
        xx = xx.reshape(-1, 1).to(dtype) * omega

        pos = torch.cat(
            [
                torch.sin(yy),
                torch.cos(yy),
                torch.sin(xx),
                torch.cos(xx),
            ],
            dim=1,
        )

        if pos.shape[1] < c:
            pad = torch.zeros(
                pos.shape[0],
                c - pos.shape[1],
                device=device,
                dtype=dtype,
            )
            pos = torch.cat([pos, pad], dim=1)

        pos = pos[:, :c]

        return x + pos.unsqueeze(0)

    @staticmethod
    def _resize_feature_map(
        feat: torch.Tensor,
        target_size: Optional[Tuple[int, int]],
    ) -> torch.Tensor:
        if target_size is None:
            return feat

        target_size = tuple(target_size)
        if feat.shape[-2:] == target_size:
            return feat

        return F.interpolate(
            feat,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    def _feat_to_tokens(
        self,
        feat: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        [B, C, H, W] -> [B, H*W, C]
        """
        feat = self._resize_feature_map(feat, target_size)

        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        tokens = self._add_2d_sincos_pos_embed(tokens, h=h, w=w)

        return tokens

    def forward(
        self,
        vit_tokens: torch.Tensor,
        cnn_feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            vit_tokens:
                [B, 196, C_vit]

            cnn_feats:
                c4, c8, c16, c32
                c4  -> Stage1
                c8  -> Stage2，高分辨率细节分支
                c16 -> Stage3
                c32 -> Stage4

        Returns:
            visual_tokens:
                [B, 980, 4*C_vit]
        """

        c4, c8, c16, c32 = cnn_feats

        # ==================================================
        # A. Stage1 / Stage3 / Stage4：交叉注意力语义融合
        # ==================================================

        c4_tok = self._feat_to_tokens(c4)
        c16_tok = self._feat_to_tokens(c16)
        c32_tok = self._feat_to_tokens(c32)

        flat_c4 = self.pre_align_f4(c4_tok)
        flat_c16 = self.pre_align_f16(c16_tok)
        flat_c32 = self.pre_align_f32(c32_tok)

        F4 = self.align_f4(vit_tokens, flat_c4)
        F16 = self.align_f16(vit_tokens, flat_c16)
        F32 = self.align_f32(vit_tokens, flat_c32)

        weights = torch.softmax(self.semantic_logits, dim=0)

        semantic_tokens = torch.cat(
            [
                vit_tokens * weights[0],
                F4 * weights[1],
                F16 * weights[2],
                F32 * weights[3],
            ],
            dim=-1,
        )

        semantic_tokens = semantic_tokens + self.semantic_mlp(
            self.semantic_ln(semantic_tokens)
        )

        # semantic_tokens: [B, 196, 3072]，当 C_vit=768

        # ==================================================
        # B. Stage2：保留 28×28 高分辨率细节 token
        # ==================================================

        detail_tokens = self._feat_to_tokens(
            c8,
            target_size=self.detail_grid_size,
        )

        detail_tokens = self.detail_proj(detail_tokens)
        detail_tokens = self.detail_ln(detail_tokens)

        # detail_tokens: [B, 784, 3072]，当 detail_grid_size=(28,28)

        # ==================================================
        # C. 序列维拼接
        # ==================================================

        visual_tokens = torch.cat(
            [semantic_tokens, detail_tokens],
            dim=1,
        )

        self._last_fusion_weights = weights.detach()
        self._last_token_info = {
            "semantic_tokens": int(semantic_tokens.shape[1]),
            "detail_tokens": int(detail_tokens.shape[1]),
            "total_tokens": int(visual_tokens.shape[1]),
            "channel_dim": int(visual_tokens.shape[-1]),
        }

        return visual_tokens
class PhysicalGuidedAlign(nn.Module):
    """Use physical prompt tokens to re-align fused visual tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        max_phys_tokens: int = 64,
        physical_in_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_phys_tokens = max_phys_tokens
        self.physical_in_dim = int(physical_in_dim) if physical_in_dim is not None else int(dim)
        if self.physical_in_dim == self.dim:
            self.phys_proj = nn.Identity()
        else:
            self.phys_proj = nn.Linear(self.physical_in_dim, self.dim)
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self._last_stats = {}

    def _project_phys(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.physical_in_dim:
            raise ValueError(
                f"physical token dim {x.size(-1)} mismatches expected {self.physical_in_dim}. "
                "Set rgb_vision.physical_prompt_dim to match physical prompt embedding dim."
            )
        return self.phys_proj(x)

    def forward(self, visual_tokens: torch.Tensor, physical_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        if physical_tokens is None:
            return visual_tokens
        if physical_tokens.numel() == 0:
            return visual_tokens

        physical_tokens = self._project_phys(physical_tokens)
        if physical_tokens.size(1) > self.max_phys_tokens:
            physical_tokens = physical_tokens[:, : self.max_phys_tokens, :]

        q = self.q_norm(physical_tokens)
        kv = self.kv_norm(visual_tokens)
        attn_out, attn_w = self.attn(query=q, key=kv, value=kv)

        phys_summary = attn_out.mean(dim=1, keepdim=True).expand(-1, visual_tokens.size(1), -1)
        gate = torch.sigmoid(self.gate(torch.cat([visual_tokens, phys_summary], dim=-1)))
        out = visual_tokens + gate * phys_summary

        with torch.no_grad():
            entropy = -(attn_w.clamp_min(1e-8) * attn_w.clamp_min(1e-8).log()).sum(dim=-1).mean()
            self._last_stats = {
                "phys_align_entropy": entropy.detach(),
                "phys_align_gate_mean": gate.mean().detach(),
            }
        return out

class DualVisionEncoder(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__()

        # Use the configured alignment dimension for fused visual tokens
        # Older code referenced config.vision.embedding_dim, but configs define
        # rgb_vision and a top-level alignment_dim instead.
        self.embedding_dim = getattr(config, "alignment_dim", 768)
        rgb_cfg = getattr(config, "rgb_vision", ml_collections.ConfigDict())

        self.input_size = tuple(rgb_cfg.get("input_size", [224, 224]))
        self.freeze_global = rgb_cfg.get("freeze_global", True)
        self.freeze_local = rgb_cfg.get("freeze_local", True)

        self.global_name = rgb_cfg.get("global_encoder_name", "dinov3_vitl14")
        self.local_name = rgb_cfg.get("local_encoder_name", "convnext_large")
        self.local_pretrained = rgb_cfg.get("local_pretrained", "laion2b_s34b_b79k")
        self.global_ckpt_path = rgb_cfg.get("global_ckpt_path", None)
        self.local_source = rgb_cfg.get("local_source", "openclip")  # 'openclip' or 'dino'

        # Global encoder: DINOv3 ViT variants via timm, using provided checkpoint path
        self.global_encoder = None
        self._use_transformers_clip_vit = False
        if self.global_ckpt_path is None or (isinstance(self.global_ckpt_path, str) and len(self.global_ckpt_path) == 0):
            raise RuntimeError("DINOv3 ViT requires 'rgb_vision.global_ckpt_path' to point to pretrained weights")
        try:
            import timm
            timm_arch = self._map_global_timm_arch(self.global_name, self.global_ckpt_path)
            self.global_encoder = timm.create_model(timm_arch, pretrained=False)
            state = torch.load(self.global_ckpt_path, map_location='cpu')
            if isinstance(state, dict) and 'model' in state:
                state = state['model']
            # Enable RoPE if available in checkpoint
            self._enable_vit_rope(state)
            missing = self.global_encoder.load_state_dict(state, strict=False)
            self._log_state_dict_mismatch(
                model_name="DINOv3 ViT",
                incompatible=missing,
                expected_missing_exact={"pos_embed", "head.weight", "head.bias"},
                expected_unexpected_prefixes=("storage_tokens", "mask_token", "rope_embed", "local_cls_norm"),
                expected_unexpected_regex=(
                    r"^blocks\.\d+\.attn\.qkv\.bias_mask$",
                    r"^blocks\.\d+\.ls1\.gamma$",
                    r"^blocks\.\d+\.ls2\.gamma$",
                ),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DINOv3 ViT via timm: {e}")

        # Local encoder: OpenCLIP ConvNeXt-L (default) or DINO ConvNeXt (timm)
        if self.local_source == "openclip":
            if open_clip is None:
                raise RuntimeError(f"open_clip_torch is required for OpenCLIP ConvNeXt-L: {_open_clip_import_error}")
            try:
                self.local_clip_model, self.local_preprocess = open_clip.create_model_and_transforms(
                    self.local_name, pretrained=self.local_pretrained
                )
                self.local_encoder = self.local_clip_model.visual
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenCLIP ConvNeXt-L: {e}")
        elif self.local_source == "dino":
            try:
                import timm
            except Exception as e:
                raise RuntimeError(f"timm is required for DINO ConvNeXt local encoder: {e}")
            if self.local_name not in ["convnext_base", "convnext_large"]:
                warnings.warn("For DINO ConvNeXt checkpoints, set local_encoder_name to 'convnext_base' or 'convnext_large'. Using convnext_base by default.")
                self.local_name = "convnext_base"
            self.local_encoder = timm.create_model(self.local_name, pretrained=False, features_only=True, out_indices=(0,1,2,3))
            local_ckpt = rgb_cfg.get("local_ckpt_path", None)
            if not local_ckpt:
                raise RuntimeError("rgb_vision.local_ckpt_path must be set to your DINOv3 ConvNeXt checkpoint path")
            raw_state = torch.load(local_ckpt, map_location='cpu')
            state = raw_state['model'] if isinstance(raw_state, dict) and 'model' in raw_state else raw_state
            # Remap common DINO ConvNeXt key patterns to timm ConvNeXt naming to maximize match
            state = self._remap_dino_convnext_to_timm(state)
            missing = self.local_encoder.load_state_dict(state, strict=False)
            self._log_state_dict_mismatch(
                model_name="DINOv3 ConvNeXt",
                incompatible=missing,
            )
        else:
            raise RuntimeError(f"Unknown local_source '{self.local_source}'. Expected 'openclip' or 'dino'.")

        # Freeze encoders
        if self.freeze_global:
            for p in self.global_encoder.parameters():
                p.requires_grad = False
        if self.freeze_local:
            for p in self.local_encoder.parameters():
                p.requires_grad = False

        # Projection heads (lazy init)
        self.local_proj = None
        self.fusion_proj = None

        self.physical_align_heads = int(rgb_cfg.get("physical_align_heads", 8))
        self.physical_align_max_tokens = int(rgb_cfg.get("physical_align_max_tokens", 64))
        # 强制语义分支为 14×14 = 196 tokens
        self.semantic_grid_size = tuple(rgb_cfg.get("semantic_grid_size", [14, 14]))
        
        # 强制 Stage2 高分辨率分支为 28×28 = 784 tokens
        self.detail_grid_size = tuple(rgb_cfg.get("detail_grid_size", [28, 28]))
        
        # False: 最终输出 [B, 980, 3072]
        # True:  最终输出 [B, 980, 768]
        self.output_project_to_embedding = bool(
            rgb_cfg.get("output_project_to_embedding", False)
        )
        self.physical_prompt_dim = int(
            rgb_cfg.get(
                "physical_prompt_dim",
                getattr(getattr(config, "text", ml_collections.ConfigDict()), "hidden_size", self.embedding_dim * 4),
            )
        )
        self.vit_dim_cfg = int(rgb_cfg.get("vit_dim", 0)) or None
        cnn_dims_cfg = rgb_cfg.get("cnn_dims", None)
        self.cnn_dims_cfg = [int(v) for v in cnn_dims_cfg] if cnn_dims_cfg else None

        self.hetero_fusion = None
        self.physical_guided_align = None
        self.final_proj = None
        self._fusion_vit_dim = None
        self._fusion_cnn_dims = None

        vit_dim_guess = self.vit_dim_cfg or self._infer_global_dim()
        cnn_dims_guess = self.cnn_dims_cfg or self._infer_local_dims()
        if vit_dim_guess is not None and cnn_dims_guess is not None and len(cnn_dims_guess) >= 4:
            self._build_alignment_modules(
                vit_dim=vit_dim_guess,
                cnn_dims=cnn_dims_guess[:4],
                device=next(self.global_encoder.parameters()).device,
                dtype=next(self.global_encoder.parameters()).dtype,
            )

    @staticmethod
    def _is_main_rank() -> bool:
        try:
            return int(os.environ.get("RANK", "0")) == 0
        except Exception:
            return True

    @staticmethod
    def _filter_mismatch_keys(
        keys,
        expected_exact=(),
        expected_prefixes=(),
        expected_regex=(),
    ):
        expected_exact = set(expected_exact or [])
        prefixes = tuple(expected_prefixes or ())
        regexes = [re.compile(p) for p in (expected_regex or ())]
        kept = []
        for k in keys or []:
            if k in expected_exact:
                continue
            if prefixes and any(str(k).startswith(p) for p in prefixes):
                continue
            if regexes and any(r.match(str(k)) for r in regexes):
                continue
            kept.append(k)
        return kept

    def _log_state_dict_mismatch(
        self,
        model_name: str,
        incompatible,
        expected_missing_exact=(),
        expected_missing_prefixes=(),
        expected_missing_regex=(),
        expected_unexpected_exact=(),
        expected_unexpected_prefixes=(),
        expected_unexpected_regex=(),
    ) -> None:
        if not isinstance(incompatible, torch.nn.modules.module._IncompatibleKeys):
            return
        if not self._is_main_rank():
            return

        missing = self._filter_mismatch_keys(
            incompatible.missing_keys,
            expected_exact=expected_missing_exact,
            expected_prefixes=expected_missing_prefixes,
            expected_regex=expected_missing_regex,
        )
        unexpected = self._filter_mismatch_keys(
            incompatible.unexpected_keys,
            expected_exact=expected_unexpected_exact,
            expected_prefixes=expected_unexpected_prefixes,
            expected_regex=expected_unexpected_regex,
        )

        if not missing and not unexpected:
            return

        # Keep logs compact in distributed training.
        max_show = 8
        msg = (
            f"{model_name} weight mismatch after load_state_dict(strict=False). "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if missing:
            msg += f", missing_sample={missing[:max_show]}"
        if unexpected:
            msg += f", unexpected_sample={unexpected[:max_show]}"
        warnings.warn(msg)

    def _add_2d_sincos_pos_embed(self, x: torch.Tensor, h: int, w: int):
        """x: [B, H*W, C] add 2D sine-cos position."""
        device, dtype = x.device, x.dtype
        c = x.shape[-1]
        y, x_coord = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        omega = torch.arange(c // 4, device=device, dtype=dtype) / (c // 4)
        omega = 1. / (10000 ** (omega / (c // 4)))
        y = y.reshape(-1, 1) * omega
        x_coord = x_coord.reshape(-1, 1) * omega
        pos = torch.cat([torch.sin(y), torch.cos(y), torch.sin(x_coord), torch.cos(x_coord)], dim=1)  # [H*W, C]
        if pos.shape[1] < c:
            pad = torch.zeros((pos.shape[0], c - pos.shape[1]), device=device, dtype=dtype)
            pos = torch.cat([pos, pad], dim=1)
        pos = pos[:, :c]
        x = x + pos.unsqueeze(0)
        return x

    def _map_dino_name(self, name: str) -> str:
        mapping = {
            'dinov2_vitl14': 'dinov2_vitl14',
            'dinov2_vitb14': 'dinov2_vitb14',
            'dinov2_vitb8': 'dinov2_vitb8',
        }
        return mapping.get(name, 'dinov2_vitl14')

    def _map_global_timm_arch(self, global_name: str, ckpt_path: str) -> str:
        # Determine timm ViT architecture based on name or checkpoint filename
        name_lower = (global_name or '').lower()
        ckpt_lower = (ckpt_path or '').lower()
        # Patch size hint
        patch = 14 if ('14' in name_lower or 'patch14' in name_lower or '14' in ckpt_lower) else 16
        # Width hint
        if 'vitg' in name_lower or 'giant' in name_lower or '7b' in name_lower or 'vit7b' in ckpt_lower:
            base = 'vit_giant'
        elif 'vith' in name_lower or 'huge' in name_lower:
            base = 'vit_huge'
        elif 'vitl' in name_lower or 'large' in name_lower:
            base = 'vit_large'
        else:
            # default to large to maximize availability
            base = 'vit_large'
        candidate = f"{base}_patch{patch}_224"
        # Some timm builds may not have giant variants; fallback to large
        try:
            import timm
            if candidate not in timm.list_models('*vit*') and base != 'vit_large':
                warnings.warn(f"timm arch '{candidate}' not found; falling back to vit_large_patch{patch}_224")
                candidate = f"vit_large_patch{patch}_224"
        except Exception:
            pass
        return candidate

    def _remap_dino_convnext_to_timm(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        remapped = {}
        for k, v in state.items():
            nk = k
            # stem/downsample to timm stem
            nk = nk.replace('downsample_layers.0.0', 'stem_0')
            nk = nk.replace('downsample_layers.0.1', 'stem_1')
            # downsample layers in later stages to per-stage downsample
            nk = nk.replace('downsample_layers.1.0', 'stages_1.downsample.0')
            nk = nk.replace('downsample_layers.1.1', 'stages_1.downsample.1')
            nk = nk.replace('downsample_layers.2.0', 'stages_2.downsample.0')
            nk = nk.replace('downsample_layers.2.1', 'stages_2.downsample.1')
            nk = nk.replace('downsample_layers.3.0', 'stages_3.downsample.0')
            nk = nk.replace('downsample_layers.3.1', 'stages_3.downsample.1')
            # stages indexing: 'stages.0' -> 'stages_0'
            nk = nk.replace('stages.', 'stages_')
            # ensure 'blocks' level exists: 'stages_0.0.' -> 'stages_0.blocks.0.'
            for i in range(4):
                nk = nk.replace(f'stages_{i}.0.', f'stages_{i}.blocks.0.')
                nk = nk.replace(f'stages_{i}.1.', f'stages_{i}.blocks.1.')
                nk = nk.replace(f'stages_{i}.2.', f'stages_{i}.blocks.2.')
                nk = nk.replace(f'stages_{i}.3.', f'stages_{i}.blocks.3.')
                nk = nk.replace(f'stages_{i}.4.', f'stages_{i}.blocks.4.')
                nk = nk.replace(f'stages_{i}.5.', f'stages_{i}.blocks.5.')
                nk = nk.replace(f'stages_{i}.6.', f'stages_{i}.blocks.6.')
                nk = nk.replace(f'stages_{i}.7.', f'stages_{i}.blocks.7.')
                nk = nk.replace(f'stages_{i}.8.', f'stages_{i}.blocks.8.')
                nk = nk.replace(f'stages_{i}.9.', f'stages_{i}.blocks.9.')
                nk = nk.replace(f'stages_{i}.10.', f'stages_{i}.blocks.10.')
                nk = nk.replace(f'stages_{i}.11.', f'stages_{i}.blocks.11.')
                nk = nk.replace(f'stages_{i}.12.', f'stages_{i}.blocks.12.')
                nk = nk.replace(f'stages_{i}.13.', f'stages_{i}.blocks.13.')
                nk = nk.replace(f'stages_{i}.14.', f'stages_{i}.blocks.14.')
                nk = nk.replace(f'stages_{i}.15.', f'stages_{i}.blocks.15.')
                nk = nk.replace(f'stages_{i}.16.', f'stages_{i}.blocks.16.')
                nk = nk.replace(f'stages_{i}.17.', f'stages_{i}.blocks.17.')
                nk = nk.replace(f'stages_{i}.18.', f'stages_{i}.blocks.18.')
                nk = nk.replace(f'stages_{i}.19.', f'stages_{i}.blocks.19.')
                nk = nk.replace(f'stages_{i}.20.', f'stages_{i}.blocks.20.')
                nk = nk.replace(f'stages_{i}.21.', f'stages_{i}.blocks.21.')
                nk = nk.replace(f'stages_{i}.22.', f'stages_{i}.blocks.22.')
                nk = nk.replace(f'stages_{i}.23.', f'stages_{i}.blocks.23.')
                nk = nk.replace(f'stages_{i}.24.', f'stages_{i}.blocks.24.')
                nk = nk.replace(f'stages_{i}.25.', f'stages_{i}.blocks.25.')
                nk = nk.replace(f'stages_{i}.26.', f'stages_{i}.blocks.26.')
            # conv depthwise naming
            nk = nk.replace('.dwconv.', '.conv_dw.')
            # MLP pointwise conv naming
            nk = nk.replace('.pwconv1.', '.mlp.fc1.')
            nk = nk.replace('.pwconv2.', '.mlp.fc2.')
            remapped[nk] = v
        return remapped

    def _enable_vit_rope(self, state: Dict[str, torch.Tensor]):
        # Enable Rotary Position Embedding in timm ViT attention blocks if rope periods are available
        periods = None
        if isinstance(state, dict):
            periods = state.get('rope_embed.periods', None)
        if periods is None:
            warnings.warn("RoPE periods not found in checkpoint; skipping RoPE integration.")
            return

        try:
            patch_size = getattr(self.global_encoder, 'patch_embed', None)
            if patch_size is not None and hasattr(self.global_encoder.patch_embed, 'patch_size'):
                ps = self.global_encoder.patch_embed.patch_size
                patch = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
            else:
                patch = 16
            h, w = self.input_size
            gh, gw = h // patch, w // patch
            seq_len = gh * gw + (1 if hasattr(self.global_encoder, 'cls_token') else 0)

            # zero out absolute pos_embed if present (we use RoPE)
            if hasattr(self.global_encoder, 'pos_embed') and isinstance(self.global_encoder.pos_embed, torch.nn.Parameter):
                with torch.no_grad():
                    pe=self.global_encoder.pos_embed
                    if pe.data.shape[1] >= seq_len:
                        pe.data.zero_()

            periods = periods.detach().float()

            def apply_rope(xq: torch.Tensor, xk: torch.Tensor) -> (torch.Tensor, torch.Tensor):
                # xq, xk: [B, heads, N, head_dim]
                head_dim = xq.shape[-1]
                pair_dim = head_dim // 2
                device = xq.device
                pos = torch.arange(seq_len, device=device).float()  # [N]
                m = periods.numel()
                angles = pos[:, None] / periods[None, :].to(device)  # [N, m]
                cos = torch.cos(angles)
                sin = torch.sin(angles)
                # Tile to pair_dim
                if m < pair_dim:
                    repeat = (pair_dim + m - 1) // m
                    cos = cos.repeat(1, repeat)[:, :pair_dim]
                    sin = sin.repeat(1, repeat)[:, :pair_dim]
                else:
                    cos = cos[:, :pair_dim]
                    sin = sin[:, :pair_dim]
                cos = cos[None, None, :, :].to(xq.dtype)
                sin = sin[None, None, :, :].to(xq.dtype)
                def rope_on(x: torch.Tensor) -> torch.Tensor:
                    x_main = x[..., : 2 * pair_dim]
                    x_tail = x[..., 2 * pair_dim:]
                    x_main = x_main.view(*x_main.shape[:-1], pair_dim, 2)
                    x1 = x_main[..., 0]
                    x2 = x_main[..., 1]
                    out1 = x1 * cos - x2 * sin
                    out2 = x1 * sin + x2 * cos
                    out = torch.stack([out1, out2], dim=-1).view(*x_main.shape[:-2], 2 * pair_dim)
                    return torch.cat([out, x_tail], dim=-1)
                return rope_on(xq), rope_on(xk)

            # Monkey-patch attention forward to inject RoPE
            if hasattr(self.global_encoder, 'blocks'):
                for blk in self.global_encoder.blocks:
                    attn = getattr(blk, 'attn', None)
                    if attn is None or not hasattr(attn, 'forward'):
                        continue
                    def rope_forward(x, _attn=attn):
                        B, N, C = x.shape
                        qkv = _attn.qkv(x).reshape(B, N, 3, _attn.num_heads, C // _attn.num_heads).permute(2, 0, 3, 1, 4)
                        q, k, v = qkv[0], qkv[1], qkv[2]
                        q, k = apply_rope(q, k)
                        attn_scores = (q @ k.transpose(-2, -1)) * _attn.scale
                        attn_probs = attn_scores.softmax(dim=-1)
                        attn_probs = _attn.attn_drop(attn_probs) if hasattr(_attn, 'attn_drop') else attn_probs
                        out = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
                        out = _attn.proj(out)
                        out = _attn.proj_drop(out) if hasattr(_attn, 'proj_drop') else out
                        return out
                    attn.forward = rope_forward
        except Exception as e:
            warnings.warn(f"Failed to enable RoPE: {e}")

    def _infer_global_dim(self) -> Optional[int]:
        for attr in ("num_features", "embed_dim", "hidden_dim"):
            value = getattr(self.global_encoder, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    def _infer_local_dims(self) -> Optional[list]:
        if self.local_source == "dino":
            feature_info = getattr(self.local_encoder, "feature_info", None)
            if feature_info is not None and hasattr(feature_info, "channels"):
                channels = feature_info.channels()
                if channels:
                    return [int(c) for c in channels[:4]]
        local_dim = getattr(self.local_encoder, "num_features", None)
        if isinstance(local_dim, int) and local_dim > 0:
            return [local_dim, local_dim, local_dim, local_dim]
        return None

    def _build_alignment_modules(
        self,
        vit_dim: int,
        cnn_dims: list,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        vit_dim = int(vit_dim)
        cnn_dims = [int(c) for c in cnn_dims[:4]]
    
        if len(cnn_dims) < 4:
            raise ValueError("cnn_dims must provide 4 stage dimensions")
    
        fused_dim = vit_dim * 4
    
        self.hetero_fusion = HeterogeneousFusionBlock(
            cnn_dims=cnn_dims,
            vit_dim=vit_dim,
            semantic_grid_size=self.semantic_grid_size,
            detail_grid_size=self.detail_grid_size,
        ).to(device=device, dtype=dtype)
    
        self.physical_guided_align = PhysicalGuidedAlign(
            dim=fused_dim,
            num_heads=self.physical_align_heads,
            max_phys_tokens=self.physical_align_max_tokens,
            physical_in_dim=self.physical_prompt_dim,
        ).to(device=device, dtype=dtype)
    
        if self.output_project_to_embedding:
            self.final_proj = nn.Linear(
                fused_dim,
                self.embedding_dim,
            ).to(device=device, dtype=dtype)
        else:
            self.final_proj = nn.Identity().to(device=device, dtype=dtype)
    
        self._fusion_vit_dim = vit_dim
        self._fusion_cnn_dims = cnn_dims

    def _ensure_alignment_modules(self, z_vit: torch.Tensor, l_feats_raw) -> None:
        inferred_vit_dim = z_vit.shape[-1]
        inferred_cnn_dims = [int(f.shape[1]) for f in l_feats_raw[:4]]
        target_vit_dim = self.vit_dim_cfg or inferred_vit_dim
        target_cnn_dims = self.cnn_dims_cfg or inferred_cnn_dims

        if self.vit_dim_cfg is not None and self.vit_dim_cfg != inferred_vit_dim:
            warnings.warn(
                f"Configured rgb_vision.vit_dim={self.vit_dim_cfg} mismatches backbone output {inferred_vit_dim}; "
                f"falling back to inferred dim."
            )
            target_vit_dim = inferred_vit_dim

        if self.cnn_dims_cfg is not None and list(self.cnn_dims_cfg[:4]) != list(inferred_cnn_dims[:4]):
            warnings.warn(
                f"Configured rgb_vision.cnn_dims={self.cnn_dims_cfg[:4]} mismatches backbone output {inferred_cnn_dims[:4]}; "
                f"falling back to inferred dims."
            )
            target_cnn_dims = inferred_cnn_dims

        device = z_vit.device
        dtype = z_vit.dtype
        need_rebuild = self.hetero_fusion is None or self.final_proj is None or self.physical_guided_align is None
        if not need_rebuild:
            if self._fusion_vit_dim != target_vit_dim or self._fusion_cnn_dims != target_cnn_dims[:4]:
                warnings.warn(
                    f"Rebuilding fusion modules due to dim change: "
                    f"vit {self._fusion_vit_dim}->{target_vit_dim}, "
                    f"cnn {self._fusion_cnn_dims}->{target_cnn_dims[:4]}"
                )
                need_rebuild = True

        if need_rebuild:
            self._build_alignment_modules(target_vit_dim, target_cnn_dims, device=device, dtype=dtype)
            return

        self.hetero_fusion = self.hetero_fusion.to(device=device, dtype=dtype)
        self.physical_guided_align = self.physical_guided_align.to(device=device, dtype=dtype)
        self.final_proj = self.final_proj.to(device=device, dtype=dtype)

    def _ensure_fusion_proj(self, in_channels: int) -> None:
        if self.fusion_proj is None:
            self.fusion_proj = nn.Conv2d(in_channels=in_channels, out_channels=self.embedding_dim, kernel_size=1)

    def _build_local_proj(self, local_feats: torch.Tensor) -> None:
        if self.local_proj is None:
            in_channels = local_feats.shape[1]
            self.local_proj = nn.Conv2d(in_channels=in_channels, out_channels=1024, kernel_size=1)
        # Align projection to input device & dtype (AMP/NPU may use fp16)
        self.local_proj = self.local_proj.to(device=local_feats.device, dtype=local_feats.dtype)

    def _get_global_grid(self, pixel_values: torch.Tensor) -> torch.Tensor:
        def _to_grid(t: torch.Tensor):
            if not isinstance(t, torch.Tensor):
                return None
            if t.dim() == 4:
                return t
            if t.dim() == 3:
                b, n, c = t.shape
                s1 = int(math.sqrt(max(n - 1, 1)))
                if s1 * s1 == n - 1:
                    t = t[:, 1:, :]
                    n -= 1
                s = int(math.sqrt(max(n, 1)))
                if s * s == n:
                    return t.reshape(b, s, s, c).permute(0, 3, 1, 2).contiguous()
                return t.transpose(1, 2).unsqueeze(-1).contiguous()
            if t.dim() == 2:
                return t.unsqueeze(-1).unsqueeze(-1)
            return None

        try:
            enc_dtype = next(self.global_encoder.parameters()).dtype
        except Exception:
            enc_dtype = pixel_values.dtype
        x_in = pixel_values.to(enc_dtype) if pixel_values.dtype != enc_dtype else pixel_values

        last_error = None
        try:
            feats = self.global_encoder.forward_features(x_in)
            if isinstance(feats, dict):
                for k in [
                    "x_norm_patchtokens",
                    "x_norm",
                    "last_hidden_state",
                    "x_prenorm",
                    "x_cls_token",
                    "pooled",
                    "features",
                ]:
                    if k in feats:
                        g = _to_grid(feats[k])
                        if g is not None:
                            return g
                for v in feats.values():
                    g = _to_grid(v)
                    if g is not None:
                        return g
            elif isinstance(feats, (tuple, list)):
                for v in feats:
                    g = _to_grid(v)
                    if g is not None:
                        return g
            else:
                g = _to_grid(feats)
                if g is not None:
                    return g
        except Exception as e:
            last_error = e

        try:
            feats = self.global_encoder(x_in)
            if isinstance(feats, dict):
                for v in feats.values():
                    g = _to_grid(v)
                    if g is not None:
                        return g
            elif isinstance(feats, (tuple, list)):
                for v in feats:
                    g = _to_grid(v)
                    if g is not None:
                        return g
            else:
                g = _to_grid(feats)
                if g is not None:
                    return g
        except Exception as e:
            last_error = e

        try:
            x = x_in
            if hasattr(self.global_encoder, 'patch_embed'):
                x = self.global_encoder.patch_embed(x)
            if isinstance(x, torch.Tensor) and x.dim() == 4:
                x = x.flatten(2).transpose(1, 2)
            elif not (isinstance(x, torch.Tensor) and x.dim() == 3):
                raise RuntimeError("manual ViT token path failed")

            added_cls = False
            if hasattr(self.global_encoder, 'cls_token') and isinstance(self.global_encoder.cls_token, torch.nn.Parameter):
                cls = self.global_encoder.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls, x), dim=1)
                added_cls = True
            if hasattr(self.global_encoder, 'pos_embed'):
                pe = self.global_encoder.pos_embed
                if isinstance(pe, (torch.nn.Parameter, torch.Tensor)) and pe.shape[1] == x.shape[1]:
                    x = x + pe
            if hasattr(self.global_encoder, 'pos_drop'):
                x = self.global_encoder.pos_drop(x)
            if hasattr(self.global_encoder, 'blocks'):
                for blk in self.global_encoder.blocks:
                    x = blk(x)
            norm_layer = getattr(self.global_encoder, 'norm', None)
            if norm_layer is None:
                norm_layer = getattr(self.global_encoder, 'fc_norm', None)
            if norm_layer is not None:
                x = norm_layer(x)
            if added_cls and x.dim() == 3:
                x = x[:, 1:, :]
            g = _to_grid(x)
            if g is not None:
                return g
        except Exception as e:
            last_error = e

        warnings.warn(
            f"Falling back to pooled global grid because DINOv3 global extraction failed: {last_error}"
        )
        return x_in.mean(dim=(-2, -1), keepdim=True)

    def _local_forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Try known APIs to get feature maps from OpenCLIP ConvNeXt visual
        for method in ["forward_features", "trunk_forward_features", "trunk.forward_features"]:
            try:
                if method == "forward_features" and hasattr(self.local_encoder, method):
                    return getattr(self.local_encoder, method)(x)
                if method == "trunk_forward_features" and hasattr(self.local_encoder, method):
                    return getattr(self.local_encoder, method)(x)
                if method == "trunk.forward_features" and hasattr(self.local_encoder, "trunk"):
                    return getattr(self.local_encoder.trunk, "forward_features")(x)
            except Exception:
                continue
        # Last resort: use visual(x) to get pooled [B, D]; reshape to [B, D, 1, 1]
        pooled = self.local_encoder(x)
        if isinstance(pooled, torch.Tensor) and pooled.dim() == 2:
            return pooled.unsqueeze(-1).unsqueeze(-1)
        raise RuntimeError("Failed to obtain feature map from OpenCLIP ConvNeXt-L")

    def _get_local_pyramid(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.local_source == "openclip":
            # Multi-resolution inputs: H/4, H/8, H/16 relative to input_size
            h, w = self.input_size
            sizes = [(h // 4, w // 4), (h // 8, w // 8), (h // 16, w // 16)]
            feat_maps = []
            for size in sizes:
                x = F.interpolate(pixel_values, size=size, mode="bilinear", align_corners=False)
                f = self._local_forward_features(x)
                if f.dim() == 2:
                    f = f.unsqueeze(-1).unsqueeze(-1)
                feat_maps.append(f)
            # Upsample lower-res to highest-res
            target_h, target_w = feat_maps[0].shape[-2], feat_maps[0].shape[-1]
            upsampled = [feat_maps[0]]
            for f in feat_maps[1:]:
                upsampled.append(F.interpolate(f, size=(target_h, target_w), mode="bilinear", align_corners=False))
            local_concat = torch.cat(upsampled, dim=1)
            return local_concat
        else:
            # DINO ConvNeXt via timm (features_only): single input size, get stages and upsample to highest
            feats = self.local_encoder(pixel_values)
            high = feats[0]
            target_h, target_w = high.shape[-2], high.shape[-1]
            upsampled = [high]
            for f in feats[1:]:
                upsampled.append(F.interpolate(f, size=(target_h, target_w), mode="bilinear", align_corners=False))
            local_concat = torch.cat(upsampled, dim=1)
            return local_concat

    def _get_local_pyramid_raw(self, pixel_values: torch.Tensor):
        """Return raw multi-scale feature maps without upsampling for decoder taps."""
        if self.local_source == "openclip":
            h, w = self.input_size
            sizes = [(h // 4, w // 4), (h // 8, w // 8), (h // 16, w // 16)]
            feat_maps = []
            for size in sizes:
                x = F.interpolate(pixel_values, size=size, mode="bilinear", align_corners=False)
                f = self._local_forward_features(x)
                if f.dim() == 2:
                    f = f.unsqueeze(-1).unsqueeze(-1)
                feat_maps.append(f)
            return feat_maps  # s1, s2, s3
        else:
            feats = self.local_encoder(pixel_values)
            return feats  # timm features_only list
    def encode(self, x: torch.Tensor):
        return self.encode_with_spatial(x)[0]
    def forward(self, x: Dict[str, torch.Tensor]):
        pixel_values = x["rgb"]
        return self.encode(pixel_values)

    def encode_with_spatial(
        self,
        x: torch.Tensor,
        physical_prompt_embs: Optional[torch.Tensor] = None,
    ):
        # Ascend NPU: bilinear upsample does not support bfloat16.
        if x.device.type == "npu" and x.dtype == torch.bfloat16:
            x = x.to(torch.float16)
        pixel_values = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)

        global_values = pixel_values
        local_values = pixel_values
        try:
            g_dtype = next(self.global_encoder.parameters()).dtype
            if global_values.dtype != g_dtype:
                global_values = global_values.to(g_dtype)
        except Exception:
            pass
        try:
            l_dtype = next(self.local_encoder.parameters()).dtype
            if local_values.dtype != l_dtype:
                local_values = local_values.to(l_dtype)
        except Exception:
            pass

        g_grid = self._get_global_grid(global_values)  # [B, C, H, W]

        # 强制 ViT 语义分支为 14×14，因此 z_vit 是 196 个 token
        if self.semantic_grid_size is not None:
            g_grid = F.interpolate(
                g_grid,
                size=self.semantic_grid_size,
                mode="bilinear",
                align_corners=False,
            )
        
        z_vit = g_grid.flatten(2).transpose(1, 2).contiguous()  # [B, 196, C_vit]

        try:
            l_feats_raw = self._get_local_pyramid_raw(local_values)
        except Exception:
            l_feats_raw = self._get_local_pyramid_raw(local_values.to(torch.float16))
        l_feats_raw = [
            f.to(dtype=z_vit.dtype) if torch.is_tensor(f) and f.dtype != z_vit.dtype else f
            for f in l_feats_raw
        ]
        if len(l_feats_raw) == 3:
            l_feats_raw = list(l_feats_raw) + [F.avg_pool2d(l_feats_raw[-1], kernel_size=2, stride=2)]

        self._ensure_alignment_modules(z_vit, l_feats_raw)

        # 新版 HeterogeneousFusionBlock 内部完成：
        # 1. Stage1 / Stage3 / Stage4 的 Cross-Attention
        # 2. ViT + Stage1 + Stage3 + Stage4 的特征维拼接
        # 3. Stage2 的 28×28 高分辨率 token 保留
        # 4. 语义 token 与高分辨率 token 的序列维拼接
        fused = self.hetero_fusion(
            vit_tokens=z_vit,
            cnn_feats=tuple(l_feats_raw[:4]),
        )
        
        # fused: [B, 980, 3072]，当 C_vit=768
        fused = self.physical_guided_align(fused, physical_prompt_embs)
        
        # 若 output_project_to_embedding=False:
        # seq: [B, 980, 3072]
        #
        # 若 output_project_to_embedding=True:
        # seq: [B, 980, 768]
        seq = self.final_proj(fused)
        
        return seq, g_grid, tuple(l_feats_raw[:4])
    def get_alignment_stats(self) -> Dict[str, torch.Tensor]:
        stats = {}
    
        if self.hetero_fusion is not None and hasattr(self.hetero_fusion, "_last_fusion_weights"):
            w = self.hetero_fusion._last_fusion_weights
            if w is not None:
                for i, v in enumerate(w):
                    stats[f"fusion_scale_{i}_weight"] = v
    
        if self.hetero_fusion is not None and hasattr(self.hetero_fusion, "_last_token_info"):
            for k, v in self.hetero_fusion._last_token_info.items():
                stats[f"fusion_{k}"] = torch.tensor(v)
    
        if self.physical_guided_align is not None and hasattr(self.physical_guided_align, "_last_stats"):
            stats.update(self.physical_guided_align._last_stats)
    
        return stats



