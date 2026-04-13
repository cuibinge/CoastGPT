from collections import OrderedDict
from typing import Callable, List, Optional, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import warnings
try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None


class TextProjHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        if hidden_dim != in_dim:
            self.in_proj = nn.Linear(in_dim, hidden_dim)
            self.in_norm = nn.LayerNorm(hidden_dim)
        else:
            self.in_proj = self.in_norm = None
        self.proj = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor):
        if self.in_proj is not None:
            x = self.in_norm(self.in_proj(x))
        return self.proj(x)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        in_dim = kwargs.pop("in_dim", 4096)
        hidden_dim = kwargs.pop("hidden_dim", 768)
        out_dim = kwargs.pop("out_dim", 768)
        model = cls(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        ckpt = torch.load(path)
        if "text_projection.weight" in ckpt.keys():
            new_ckpt = {"weight": ckpt["text_projection.weight"]}
            model.proj.load_state_dict(new_ckpt)
            del new_ckpt
        del ckpt
        return model


class RgbProjHead(TextProjHead):
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        in_dim = kwargs.pop("in_dim", 4096)
        hidden_dim = kwargs.pop("hidden_dim", 1024)
        out_dim = kwargs.pop("out_dim", 768)
        model = cls(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        ckpt = torch.load(path)
        if "visual_projection.weight" in ckpt.keys():
            new_ckpt = {"weight": ckpt["visual_projection.weight"]}
            model.proj.load_state_dict(new_ckpt)
            del new_ckpt
        del ckpt
        return model


class LinearProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: int = 2):
        super().__init__()
        self.layers = [nn.Linear(in_channels, out_channels)]
        for _ in range(layers):
            self.layers.append(nn.GELU())
            self.layers.append(nn.Linear(out_channels, out_channels))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        org_dtype = x.dtype
        org_module_dtype = self.layers[0].weight.dtype

        self.layers.to(torch.float32)
        x = x.to(torch.float32)

        x = self.layers(x)
        x.to(org_dtype)
        self.layers.to(org_module_dtype)
        return x


class AttnPooler(nn.Module):
    """
    Attention Pooler (Expert 序列压缩专家)
    """

    def __init__(
            self,
            num_query: int,
            num_layers: int,
            num_attention_heads: int,
            encoder_hidden_size: int,
            hidden_size: int,
            output_size: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            checkpoint: bool = False,
            stage_num: Union[List, int] = [64, 48, 32],
            split_part: List = [256, 256, 256],
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.num_query = num_query
        self.stage_num = stage_num
        self.split_part = split_part

        self.query = nn.Parameter(torch.zeros(1, num_query, hidden_size))
        nn.init.trunc_normal_(self.query, std=0.01, mean=0.0)

        if encoder_hidden_size != hidden_size:
            self.in_proj = nn.Linear(encoder_hidden_size, hidden_size)
        else:
            self.in_proj = None

        self.layers = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    d_model=hidden_size,
                    n_head=num_attention_heads,
                    is_cross_attention=True,
                    ls_init_value=1e-1,
                    norm_layer=norm_layer,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_proj = nn.Linear(hidden_size, output_size)

    def forward(
            self,
            image_embs: torch.Tensor,
            physical_queries: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        image_embs: [B, L, C] key/value 序列 (通常为视觉+物理提示拼接后的 Z_tilde)
        physical_queries: [B, L_p, C] 可选的物理提示查询，与可学习查询一起参与注意力
        """
        if self.in_proj is not None:
            image_embs = self.in_proj(image_embs)
            if physical_queries is not None:
                physical_queries = self.in_proj(physical_queries)

        query_tokens = self.query.expand(image_embs.size(0), -1, -1)

        if isinstance(self.stage_num, int):
            stage1_query, stage2_query, stage3_query = torch.split(
                query_tokens, self.num_query // self.stage_num, dim=1
            )
            stage_query_sizes = [
                self.num_query // self.stage_num,
                self.num_query // self.stage_num,
                self.num_query // self.stage_num,
            ]
        else:
            stage_query_sizes = list(self.stage_num)
            if len(stage_query_sizes) != 3:
                raise ValueError(f"stage_num must have 3 parts, got: {stage_query_sizes}")
            if sum(stage_query_sizes) != self.num_query:
                # Keep training robust when num_query changes from default 144.
                even = self.num_query // 3
                stage_query_sizes = [even, even, self.num_query - 2 * even]
            stage1_query, stage2_query, stage3_query = torch.split(
                query_tokens, stage_query_sizes, dim=1
            )

        # 动态切分，兼容拼接了物理特征后长度 L 变长的情况
        L = image_embs.size(1)
        preset_sum = sum(self.split_part) if isinstance(self.split_part, (list, tuple)) else None
        if isinstance(self.split_part, (list, tuple)) and preset_sum == L:
            split_sizes = list(self.split_part)
        else:
            base = sum(stage_query_sizes)
            if base == 0:
                split_sizes = [L // 3, L // 3, L - 2 * (L // 3)]
            else:
                sizes = [int(round(L * s / base)) for s in stage_query_sizes]
                diff = L - sum(sizes)
                sizes[-1] += diff
                split_sizes = sizes

        stage1_image, stage2_image, stage3_image = torch.split(image_embs, split_sizes, dim=1)
        if physical_queries is not None:
            Lp = physical_queries.size(1)
            # 按照视觉分段比例切分物理查询；避免空段
            if L > 0:
                phy_sizes = [max(1, int(round(Lp * s / L))) for s in split_sizes]
                diff_phy = Lp - sum(phy_sizes)
                phy_sizes[-1] += diff_phy
            else:
                phy_sizes = [Lp // 3, Lp // 3, Lp - 2 * (Lp // 3)]
            stage1_phy, stage2_phy, stage3_phy = torch.split(physical_queries, phy_sizes, dim=1)
        else:
            stage1_phy = stage2_phy = stage3_phy = None

        all_tokens = []
        spatial_attns = []
        for sub_token, sub_image, sub_phy in zip(
                [stage1_query, stage2_query, stage3_query],
                [stage1_image, stage2_image, stage3_image],
                [stage1_phy, stage2_phy, stage3_phy],
        ):
            if sub_phy is not None:
                # 让物理提示也作为查询参与专家内部注意力
                sub_token = torch.cat([sub_token, sub_phy], dim=1)
            # key/value 仅由图像特征和物理序列组成，避免查询看到自身
            kv_parts = [sub_image]
            if sub_phy is not None:
                kv_parts.append(sub_phy)
            cat_embs = torch.cat(kv_parts, dim=1)
            cat_embs = cat_embs.permute(1, 0, 2)
            sub_token = sub_token.permute(1, 0, 2)

            for layer in self.layers:
                sub_token = layer(sub_token, cat_embs, cat_embs)

            if not self.training and hasattr(self.layers[-1], "_last_attn_weights"):
                attn = self.layers[-1]._last_attn_weights
                img_len = sub_image.size(1)
                img_attn = attn[:, :, :img_len].mean(dim=1)
                spatial_attns.append(img_attn)

            sub_token = sub_token.permute(1, 0, 2)
            all_tokens.append(sub_token)

        if not self.training and len(spatial_attns) == 3:
            full_spatial_attn = torch.cat(spatial_attns, dim=1)
            self._last_spatial_attn = full_spatial_attn[0].cpu()

        query_tokens = torch.cat(all_tokens, dim=1)
        out = self.out_proj(query_tokens)
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed


class PhysicalPromptEncoder(nn.Module):
    """
    将物理属性提示 (波段、GSD、时间等) 编码为可与视觉 token 拼接的序列。
    """

    def __init__(
            self,
            in_channels: int = 1,
            embed_dim: int = 768,
            pool_scales: Union[List[int], tuple] = (1, 2, 4),
    ):
        super().__init__()
        self.pool_scales = list(pool_scales) if pool_scales is not None else [1]
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
            self,
            tsm: Optional[torch.Tensor],
            mask: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        if tsm is None:
            return None

        # squeeze potential extra dimension from dataloader collate
        if tsm.dim() == 5:
            tsm = tsm.squeeze(1)
        if tsm.dim() == 3:
            tsm = tsm.unsqueeze(1)

        if mask is not None:
            if mask.dim() == 5:
                mask = mask.squeeze(1)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            tsm = tsm * mask

        prompts = []
        for s in self.pool_scales:
            pooled = F.adaptive_avg_pool2d(tsm, output_size=(s, s))
            tokens = self.proj(pooled).flatten(2).transpose(1, 2)
            prompts.append(self.norm(tokens))

        if len(prompts) == 0:
            return None
        return torch.cat(prompts, dim=1)


# =========================================================================
# 核心重构：任务与要素双驱动、物理先验复用的稀疏混合专家投影层 (T-MoE)
# =========================================================================
class MoEProjection(nn.Module):
    """
    对应论文 2.2 节：
    2.2.1 任务与要素双驱动的动态门控路由
    2.2.2 物理先验复用与联合特征投影
    """

    def __init__(
            self,
            num_experts: int,
            num_query: int,
            num_layers: int,
            num_attention_heads: int,
            encoder_hidden_size: int,
            hidden_size: int,
            output_size: int,
            task_dim: int = 256,
            text_embed_dim: Optional[int] = None,
            physical_prompt_embed_dims: Optional[List[int]] = None,
            include_physical_prompt: bool = True,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            checkpoint: bool = False,
            top_k: int = 2,
            routing_strategy: str = "joint",
            task_expert_ratio: float = 0.5,
            aux_balance_coef: float = 1.0,
            aux_entropy_coef: float = 1e-5,
            aux_zloss_coef: float = 1e-5,
            aux_task_route_coef: float = 0.05,
            aux_element_route_coef: float = 0.05,
            aux_task_element_orth_coef: float = 0.01,
            route_effect_margin: float = 0.05,
            route_supervision_temperature: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.checkpoint = checkpoint
        self.num_query = num_query
        self.output_size = output_size
        self.encoder_hidden_size = encoder_hidden_size
        self.include_physical_prompt = include_physical_prompt
        self.routing_strategy = str(routing_strategy).lower()
        self.task_expert_ratio = float(task_expert_ratio)
        self.text_embed_dim = int(text_embed_dim) if text_embed_dim is not None else int(encoder_hidden_size)

        # 1. 专家组：基于交叉注意力的序列压缩器 (AttnPooler)
        self.experts = nn.ModuleList([
            AttnPooler(
                num_query=num_query,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                encoder_hidden_size=encoder_hidden_size,
                hidden_size=hidden_size,
                output_size=output_size,
                norm_layer=norm_layer,
                checkpoint=checkpoint,
            ) for _ in range(num_experts)
        ])

        # 2. 任务与要素文本语义池化（彻底替代离散 ID 嵌入）
        self.task_text_query = nn.Parameter(torch.zeros(1, 1, encoder_hidden_size))
        self.element_text_query = nn.Parameter(torch.zeros(1, 1, encoder_hidden_size))
        nn.init.trunc_normal_(self.task_text_query, std=0.02)
        nn.init.trunc_normal_(self.element_text_query, std=0.02)
        if self.text_embed_dim != encoder_hidden_size:
            self.task_text_in_proj = nn.Linear(self.text_embed_dim, encoder_hidden_size)
            self.element_text_in_proj = nn.Linear(self.text_embed_dim, encoder_hidden_size)
        else:
            self.task_text_in_proj = nn.Identity()
            self.element_text_in_proj = nn.Identity()
        self.task_text_proj = nn.Linear(encoder_hidden_size, task_dim)
        self.element_text_proj = nn.Linear(encoder_hidden_size, task_dim)

        # 3. 双驱动门控网络 (全局视觉特征 + 任务特征 + 要素特征 + 物理先验)
        # 为门控对齐维度：将视觉全局和物理池化都投影到 task_dim
        self.gate_img_proj = nn.Linear(encoder_hidden_size, task_dim)
        combined_dim = task_dim + task_dim + task_dim
        if self.include_physical_prompt:
            combined_dim += task_dim
            self.physical_gate_proj = nn.Linear(encoder_hidden_size, task_dim)
            supported_phy_dims = set(physical_prompt_embed_dims or [])
            supported_phy_dims.add(encoder_hidden_size)
            self.supported_physical_dims = sorted({int(d) for d in supported_phy_dims if d is not None})
            self.physical_in_proj = nn.ModuleDict()
            for d in self.supported_physical_dims:
                if d != encoder_hidden_size:
                    self.physical_in_proj[str(d)] = nn.Linear(d, encoder_hidden_size)
            # 上下文感知的物理提示池化：单查询自注意力
            self.physical_pool_query = nn.Parameter(torch.zeros(1, 1, encoder_hidden_size))
            nn.init.trunc_normal_(self.physical_pool_query, std=0.02)
        else:
            self.physical_gate_proj = None
            self.physical_in_proj = None
            self.supported_physical_dims = []
            self.physical_pool_query = None
        self.gate_norm = nn.LayerNorm(combined_dim)
        self.gate = nn.Linear(combined_dim, num_experts)

        # 可选：两段式路由（前半专家偏任务，后半专家偏要素）
        self.task_expert_count = 0
        self.element_expert_count = 0
        self.task_gate = None
        self.task_gate_norm = None
        self.element_gate = None
        self.element_gate_norm = None
        if self.routing_strategy in ("two_stage", "task_then_element"):
            if num_experts < 2:
                warnings.warn("two_stage routing requires num_experts >= 2, fallback to joint routing.")
                self.routing_strategy = "joint"
            else:
                self.task_expert_count = max(1, min(num_experts - 1, int(round(num_experts * self.task_expert_ratio))))
                self.element_expert_count = num_experts - self.task_expert_count
                task_gate_dim = task_dim + task_dim + (task_dim if self.include_physical_prompt else 0)
                element_gate_dim = task_dim + task_dim + (task_dim if self.include_physical_prompt else 0)
                self.task_gate_norm = nn.LayerNorm(task_gate_dim)
                self.task_gate = nn.Linear(task_gate_dim, self.task_expert_count)
                self.element_gate_norm = nn.LayerNorm(element_gate_dim)
                self.element_gate = nn.Linear(element_gate_dim, self.element_expert_count)

        # 4. 稳定性输出增益和归一化
        self.output_gain = nn.Parameter(torch.ones(output_size) * 0.3)
        self.final_norm = RMSNorm(output_size)

        # 5. 辅助损失系数
        self.aux_balance_coef = float(aux_balance_coef)
        self.aux_entropy_coef = float(aux_entropy_coef)
        self.aux_zloss_coef = float(aux_zloss_coef)
        # 任务/要素语义路由监督：避免路由退化为纯视觉聚类
        self.aux_task_route_coef = float(aux_task_route_coef)
        self.aux_element_route_coef = float(aux_element_route_coef)
        self.aux_task_element_orth_coef = float(aux_task_element_orth_coef)
        self.route_effect_margin = float(route_effect_margin)
        self.route_supervision_temperature = float(route_supervision_temperature)
        self._gate_stats = {}
        self._aux_terms = {}
        self._last_routing = {}

    def _pool_semantic_text(
            self,
            text_embs: Optional[torch.Tensor],
            text_mask: Optional[torch.Tensor],
            in_proj: nn.Module,
            query: torch.Tensor,
            proj: nn.Linear,
            target_batch: int,
            dtype: torch.dtype,
            device: torch.device,
    ) -> torch.Tensor:
        if text_embs is None:
            return torch.zeros(target_batch, proj.out_features, device=device, dtype=dtype)
        if text_embs.size(0) != target_batch:
            raise ValueError(f"text_embs batch ({text_embs.size(0)}) != image batch ({target_batch})")
        text_embs = in_proj(text_embs).to(device=device, dtype=dtype)
        if text_mask is not None:
            text_mask = text_mask.to(device=device).bool()
        if text_embs.dim() == 2:
            pooled = text_embs
        else:
            q = query.expand(text_embs.size(0), -1, -1)  # [B,1,C]
            scores = torch.matmul(q, text_embs.transpose(1, 2)) / (text_embs.size(-1) ** 0.5)
            if text_mask is not None:
                if text_mask.dim() == 1:
                    text_mask = text_mask.unsqueeze(0).expand(text_embs.size(0), -1)
                if text_mask.dim() != 2:
                    raise ValueError("text_mask must be [B, L] for sequence text embeddings")
                if text_mask.size(1) != text_embs.size(1):
                    raise ValueError(
                        f"text_mask length ({text_mask.size(1)}) != text_embs length ({text_embs.size(1)})"
                    )
                scores = scores.masked_fill(~text_mask.unsqueeze(1), -1e4)
            weights = torch.softmax(scores, dim=-1)
            if text_mask is not None:
                weights = weights * text_mask.unsqueeze(1).to(weights.dtype)
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            pooled = torch.matmul(weights, text_embs).squeeze(1)  # [B,C]
        return proj(pooled.to(dtype=dtype, device=device))

    def _project_physical_prompts(self, physical_prompts: torch.Tensor) -> torch.Tensor:
        if physical_prompts.size(-1) == self.encoder_hidden_size:
            return physical_prompts
        key = str(int(physical_prompts.size(-1)))
        if self.physical_in_proj is None or key not in self.physical_in_proj:
            raise ValueError(
                f"Unsupported physical prompt dim {physical_prompts.size(-1)}; "
                f"supported dims: {self.supported_physical_dims}"
            )
        return self.physical_in_proj[key](physical_prompts)

    def forward(
            self,
            image_embs: torch.Tensor,
            physical_prompts: Optional[torch.Tensor] = None,
            task_text_embs: Optional[torch.Tensor] = None,
            element_text_embs: Optional[torch.Tensor] = None,
            physical_prompt_mask: Optional[torch.Tensor] = None,
            task_text_mask: Optional[torch.Tensor] = None,
            element_text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        image_embs: [B, L_v, C] 多尺度视觉特征序列 (Z_visual)
        physical_prompts: [B, L_p, C] 物理先验提示序列 (H_prior)
        task_text_embs: [B, L_t, C] 任务文本 embedding（Tokenizer + LLM Embedding）
        element_text_embs: [B, L_e, C] 要素文本 embedding（Tokenizer + LLM Embedding）
        *_mask: [B, L] 有效 token 掩码（用于避免 padding 稀释语义）
        """
        if isinstance(image_embs, (list, tuple)):
            image_embs = torch.cat(image_embs, dim=1)
        B, L_v, C = image_embs.shape
        has_task_text = task_text_embs is not None
        has_element_text = element_text_embs is not None

        # ==========================================
        # 1. 任务与要素双驱动的动态门控计算 (Sequence-level Routing)
        # ==========================================
        # 获取宏观全局图像上下文并对齐到 task_dim
        img_global = image_embs.mean(dim=1)  # [B, C]
        img_gate = self.gate_img_proj(img_global)  # [B, task_dim]

        # 从任务/要素文本 embedding 中提取语义上下文 h_task, h_element
        t_embs = self._pool_semantic_text(
            text_embs=task_text_embs,
            text_mask=task_text_mask,
            in_proj=self.task_text_in_proj,
            query=self.task_text_query,
            proj=self.task_text_proj,
            target_batch=B,
            dtype=image_embs.dtype,
            device=image_embs.device,
        )
        e_embs = self._pool_semantic_text(
            text_embs=element_text_embs,
            text_mask=element_text_mask,
            in_proj=self.element_text_in_proj,
            query=self.element_text_query,
            proj=self.element_text_proj,
            target_batch=B,
            dtype=image_embs.dtype,
            device=image_embs.device,
        )

        if self.include_physical_prompt:
            if physical_prompts is not None:
                physical_prompts = self._project_physical_prompts(physical_prompts).to(
                    device=image_embs.device,
                    dtype=image_embs.dtype,
                )
                # 上下文感知池化：单查询自注意力
                q = self.physical_pool_query.expand(physical_prompts.size(0), -1, -1)  # [B,1,C]
                attn_scores = torch.matmul(q, physical_prompts.transpose(1, 2)) / (physical_prompts.size(-1) ** 0.5)  # [B,1,L_p]
                if physical_prompt_mask is not None:
                    pm = physical_prompt_mask.to(device=image_embs.device).bool()
                    if pm.dim() == 1:
                        pm = pm.unsqueeze(0).expand(physical_prompts.size(0), -1)
                    if pm.dim() != 2:
                        raise ValueError("physical_prompt_mask must be [B, L_p]")
                    if pm.size(1) != physical_prompts.size(1):
                        raise ValueError(
                            f"physical_prompt_mask length ({pm.size(1)}) != physical_prompts length ({physical_prompts.size(1)})"
                        )
                    attn_scores = attn_scores.masked_fill(~pm.unsqueeze(1), -1e4)
                attn_weights = torch.softmax(attn_scores, dim=-1)
                if physical_prompt_mask is not None:
                    attn_weights = attn_weights * pm.unsqueeze(1).to(attn_weights.dtype)
                    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                phy_pooled = torch.matmul(attn_weights, physical_prompts).squeeze(1)  # [B, C]
                phy_gate = self.physical_gate_proj(phy_pooled)
                phy_query = physical_prompts
            else:
                phy_gate = torch.zeros_like(t_embs)
                phy_query = None
        else:
            phy_gate = None
            phy_query = None

        # 拼接用于联合门控评估 [z_bar_visual; h_task; h_element; h_phy]
        gate_parts = [img_gate, t_embs, e_embs]
        if self.include_physical_prompt and phy_gate is not None:
            gate_parts.append(phy_gate)
        gate_input = torch.cat(gate_parts, dim=-1).float()

        # 计算专家倾向性
        task_logits = None
        elem_logits = None
        if self.routing_strategy in ("two_stage", "task_then_element") and self.task_gate is not None:
            task_parts = [img_gate, t_embs]
            elem_parts = [img_gate, e_embs]
            if self.include_physical_prompt and phy_gate is not None:
                task_parts.append(phy_gate)
                elem_parts.append(phy_gate)
            task_input = torch.cat(task_parts, dim=-1).float()
            elem_input = torch.cat(elem_parts, dim=-1).float()

            task_logits = self.task_gate(self.task_gate_norm(task_input))
            elem_logits = self.element_gate(self.element_gate_norm(elem_input))
            gate_logits = torch.full(
                (B, self.num_experts),
                fill_value=-1e4,
                device=image_embs.device,
                dtype=task_logits.dtype,
            )
            gate_logits[:, :self.task_expert_count] = task_logits
            gate_logits[:, self.task_expert_count:] = elem_logits
        else:
            gate_logits = self.gate(self.gate_norm(gate_input))  # [B, num_experts]
        invalid_gate_ratio = (~torch.isfinite(gate_logits)).float().mean()
        gate_logits = torch.nan_to_num(gate_logits, nan=0.0, posinf=15.0, neginf=-15.0)
        gate_logits = torch.clamp(gate_logits, min=-15.0, max=15.0)
        gate_weights = torch.softmax(gate_logits, dim=-1)  # [B, num_experts]
        gate_weights = torch.nan_to_num(
            gate_weights,
            nan=1.0 / float(self.num_experts),
            posinf=1.0,
            neginf=0.0,
        )
        gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # 任务感知路由监督，抑制“只看视觉”的退化
        task_route_loss = torch.zeros((), device=image_embs.device, dtype=gate_weights.dtype)
        element_route_loss = torch.zeros((), device=image_embs.device, dtype=gate_weights.dtype)
        task_route_kl = torch.zeros((), device=image_embs.device, dtype=gate_weights.dtype)
        element_route_kl = torch.zeros((), device=image_embs.device, dtype=gate_weights.dtype)
        task_route_effect = torch.zeros((), device=image_embs.device, dtype=gate_weights.dtype)
        element_route_effect = torch.zeros((), device=image_embs.device, dtype=gate_weights.dtype)
        task_element_orth = torch.zeros((), device=image_embs.device, dtype=gate_weights.dtype)
        tau = max(self.route_supervision_temperature, 1e-4)

        if self.training:
            if self.routing_strategy in ("two_stage", "task_then_element") and self.task_gate is not None:
                if has_task_text and task_logits is not None:
                    task_text_only_parts = [torch.zeros_like(img_gate), t_embs]
                    if self.include_physical_prompt and phy_gate is not None:
                        task_text_only_parts.append(phy_gate)
                    task_text_only_input = torch.cat(task_text_only_parts, dim=-1).float()
                    task_text_only_logits = self.task_gate(self.task_gate_norm(task_text_only_input))
                    task_route_kl = F.kl_div(
                        F.log_softmax(task_logits / tau, dim=-1),
                        F.softmax(task_text_only_logits.detach() / tau, dim=-1),
                        reduction="batchmean",
                    ) * (tau * tau)
                    # Ensure task text has observable routing effect in two-stage mode.
                    task_no_text_parts = [img_gate, torch.zeros_like(t_embs)]
                    if self.include_physical_prompt and phy_gate is not None:
                        task_no_text_parts.append(phy_gate)
                    task_no_text_input = torch.cat(task_no_text_parts, dim=-1).float()
                    task_no_text_logits = self.task_gate(self.task_gate_norm(task_no_text_input))
                    task_text_effect = torch.mean(
                        torch.abs(task_logits - task_no_text_logits)
                    )
                    task_route_effect = F.relu(
                        gate_weights.new_tensor(self.route_effect_margin) - task_text_effect
                    )
                    task_route_loss = task_route_kl + task_route_effect

                if has_element_text and elem_logits is not None:
                    element_text_only_parts = [torch.zeros_like(img_gate), e_embs]
                    if self.include_physical_prompt and phy_gate is not None:
                        element_text_only_parts.append(phy_gate)
                    element_text_only_input = torch.cat(element_text_only_parts, dim=-1).float()
                    element_text_only_logits = self.element_gate(self.element_gate_norm(element_text_only_input))
                    element_route_kl = F.kl_div(
                        F.log_softmax(elem_logits / tau, dim=-1),
                        F.softmax(element_text_only_logits.detach() / tau, dim=-1),
                        reduction="batchmean",
                    ) * (tau * tau)
                    # Ensure element text has observable routing effect in two-stage mode.
                    element_no_text_parts = [img_gate, torch.zeros_like(e_embs)]
                    if self.include_physical_prompt and phy_gate is not None:
                        element_no_text_parts.append(phy_gate)
                    element_no_text_input = torch.cat(element_no_text_parts, dim=-1).float()
                    element_no_text_logits = self.element_gate(self.element_gate_norm(element_no_text_input))
                    element_text_effect = torch.mean(
                        torch.abs(elem_logits - element_no_text_logits)
                    )
                    element_route_effect = F.relu(
                        gate_weights.new_tensor(self.route_effect_margin) - element_text_effect
                    )
                    element_route_loss = element_route_kl + element_route_effect
            else:
                # joint 路由下，约束任务/要素文本对路由至少产生可观影响
                if has_task_text:
                    task_drop_parts = [img_gate, torch.zeros_like(t_embs), e_embs]
                    if self.include_physical_prompt and phy_gate is not None:
                        task_drop_parts.append(phy_gate)
                    task_drop_logits = self.gate(self.gate_norm(torch.cat(task_drop_parts, dim=-1).float()))
                    task_effect = torch.mean(torch.abs(gate_logits - task_drop_logits))
                    task_route_loss = F.relu(
                        gate_weights.new_tensor(self.route_effect_margin) - task_effect
                    )
                    task_route_effect = task_route_loss
                if has_element_text:
                    element_drop_parts = [img_gate, t_embs, torch.zeros_like(e_embs)]
                    if self.include_physical_prompt and phy_gate is not None:
                        element_drop_parts.append(phy_gate)
                    element_drop_logits = self.gate(self.gate_norm(torch.cat(element_drop_parts, dim=-1).float()))
                    element_effect = torch.mean(torch.abs(gate_logits - element_drop_logits))
                    element_route_loss = F.relu(
                        gate_weights.new_tensor(self.route_effect_margin) - element_effect
                    )
                    element_route_effect = element_route_loss

            if has_task_text and has_element_text:
                # 任务语义与要素语义尽量解耦
                task_element_orth = torch.mean(
                    torch.abs(F.cosine_similarity(t_embs.float(), e_embs.float(), dim=-1))
                ).to(gate_weights.dtype)

        # 选取 Top-K 专家
        top_weights, top_indices = torch.topk(gate_weights, self.top_k, dim=-1)  # [B, k]

        # 权重重归一化
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # 记录最近一次前向的路由细节（用于评测脚本可视化）
        with torch.no_grad():
            self._last_routing = {
                "routing_strategy": self.routing_strategy,
                "task_expert_count": int(self.task_expert_count),
                "element_expert_count": int(self.element_expert_count),
                "num_experts": int(self.num_experts),
                "top_k": int(self.top_k),
                "visual_token_length": int(L_v),
                "physical_token_length": int(physical_prompts.size(1)) if physical_prompts is not None else 0,
                "gate_logits": gate_logits.detach().float().cpu(),
                "gate_weights": gate_weights.detach().float().cpu(),
                "top_indices": top_indices.detach().long().cpu(),
                "top_weights": top_weights.detach().float().cpu(),
            }

        # 记录辅助损失状态
        if self.training:
            importance_soft = gate_weights.mean(dim=0)
            balance_loss = self.num_experts * torch.sum(importance_soft * importance_soft)
            entropy = -torch.mean(torch.sum(gate_weights * torch.log(gate_weights + 1e-6), dim=-1))
            zloss = torch.mean(torch.logsumexp(gate_logits.float(), dim=-1) ** 2)
            task_route_loss = torch.nan_to_num(task_route_loss, nan=0.0, posinf=0.0, neginf=0.0)
            element_route_loss = torch.nan_to_num(element_route_loss, nan=0.0, posinf=0.0, neginf=0.0)
            task_route_kl = torch.nan_to_num(task_route_kl, nan=0.0, posinf=0.0, neginf=0.0)
            element_route_kl = torch.nan_to_num(element_route_kl, nan=0.0, posinf=0.0, neginf=0.0)
            task_route_effect = torch.nan_to_num(task_route_effect, nan=0.0, posinf=0.0, neginf=0.0)
            element_route_effect = torch.nan_to_num(element_route_effect, nan=0.0, posinf=0.0, neginf=0.0)
            task_element_orth = torch.nan_to_num(task_element_orth, nan=0.0, posinf=0.0, neginf=0.0)
            entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
            zloss = torch.nan_to_num(zloss, nan=0.0, posinf=0.0, neginf=0.0)

            self._aux_terms = {
                "balance": balance_loss,
                "entropy": entropy,
                "zloss": zloss,
                "task_route": task_route_loss,
                "element_route": element_route_loss,
                "task_element_orth": task_element_orth,
            }

            with torch.no_grad():
                top1_indices = top_indices[:, 0]
                load = torch.zeros(self.num_experts, device=image_embs.device, dtype=torch.float32)
                for i in range(self.num_experts):
                    load[i] = (top1_indices == i).float().mean()
                self._gate_stats = {
                    "importance": importance_soft.detach(),
                    "load": load,
                    "entropy": entropy.detach(),
                    "zloss": zloss.detach(),
                    "invalid_gate_ratio": invalid_gate_ratio.detach(),
                    "task_route_loss": task_route_loss.detach(),
                    "element_route_loss": element_route_loss.detach(),
                    "task_route_kl": task_route_kl.detach(),
                    "element_route_kl": element_route_kl.detach(),
                    "task_route_effect": task_route_effect.detach(),
                    "element_route_effect": element_route_effect.detach(),
                    "task_element_orth": task_element_orth.detach(),
                }
            if self.routing_strategy in ("two_stage", "task_then_element") and self.task_expert_count > 0:
                task_mass = gate_weights[:, :self.task_expert_count].sum(dim=-1).mean()
                element_mass = gate_weights[:, self.task_expert_count:].sum(dim=-1).mean()
                self._gate_stats["task_branch_mass"] = task_mass
                self._gate_stats["element_branch_mass"] = element_mass
        else:
            self._aux_terms = {}

        # ==========================================
        # 2. 物理先验复用与联合特征构建 (Physical Prior Reuse)
        # ==========================================
        # 沿着序列维度拼接视觉特征与物理提示: Z_tilde = Concat(Z_visual, H_prior)
        if physical_prompts is not None:
            # physical_prompts shape expected to be [B, L_p, C]
            z_tilde = torch.cat([image_embs, physical_prompts], dim=1)  # [B, L_v + L_p, C]
        else:
            z_tilde = image_embs

        # ==========================================
        # 3. 专家分发与投影重采样 (Dispatch & Combine)
        # ==========================================
        for expert in self.experts:
            if hasattr(expert, "_last_spatial_attn"):
                expert._last_spatial_attn = None
        final_output = torch.zeros(B, self.num_query, self.output_size, device=image_embs.device,
                                   dtype=image_embs.dtype)

        for i in range(self.top_k):
            expert_idx = top_indices[:, i]  # [B]
            expert_weight = top_weights[:, i]  # [B]

            for e_id in range(self.num_experts):
                mask = (expert_idx == e_id)  # 布尔掩码 [B]
                if not mask.any():
                    continue

                # 提取指派给当前专家的 "物理联合序列 Z_tilde"
                selected_z_tilde = z_tilde[mask]  # [num_selected, L_v + L_p, C]

                # AttnPooler 处理包含物理属性的长序列
                expert_out = self.experts[e_id](selected_z_tilde, physical_queries=phy_query[mask] if phy_query is not None else None)  # [num_selected, num_query, output_size]
                # 专家内部可拼接物理查询，输出长度可能 > num_query；仅保留可学习查询对应的前 num_query 个 token
                if expert_out.size(1) != self.num_query:
                    expert_out = expert_out[:, : self.num_query, :]

                # 加权融合
                selected_weights = expert_weight[mask].unsqueeze(1).unsqueeze(2)  # [num_selected, 1, 1]
                weighted_out = expert_out * selected_weights

                final_output[mask] += weighted_out

        # ==========================================
        # 4. 特征对齐与约束输出
        # ==========================================
        final_output = self.final_norm(final_output) * self.output_gain

        return final_output

    def get_gate_stats(self) -> Dict[str, torch.Tensor]:
        return self._gate_stats

    def get_last_routing(self) -> Dict[str, object]:
        return self._last_routing

    def get_aux_loss(self) -> torch.Tensor:
        """计算路由辅助损失，确保任务感知与专家均衡。"""
        if not self.training or not self._aux_terms:
            return torch.tensor(0.0, device=self.gate.weight.device)

        balance_loss = self._aux_terms.get(
            "balance", torch.tensor(0.0, device=self.gate.weight.device)
        )
        entropy_loss = self._aux_terms.get(
            "entropy", torch.tensor(0.0, device=self.gate.weight.device)
        )
        zloss = self._aux_terms.get(
            "zloss", torch.tensor(0.0, device=self.gate.weight.device)
        )
        task_route_loss = self._aux_terms.get(
            "task_route", torch.tensor(0.0, device=self.gate.weight.device)
        )
        element_route_loss = self._aux_terms.get(
            "element_route", torch.tensor(0.0, device=self.gate.weight.device)
        )
        task_element_orth = self._aux_terms.get(
            "task_element_orth", torch.tensor(0.0, device=self.gate.weight.device)
        )

        total = (
            self.aux_balance_coef * balance_loss
            - self.aux_entropy_coef * entropy_loss
            + self.aux_zloss_coef * zloss
            + self.aux_task_route_coef * task_route_loss
            + self.aux_element_route_coef * element_route_loss
            + self.aux_task_element_orth_coef * task_element_orth
        )
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)

        return total.to(self.gate.weight.dtype)




class PatchDropout(nn.Module):
    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerNormFp32(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(
            x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps
        )
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        need_weights = not self.training
        attn_out, attn_weights = self.attn(
            q_x, k_x, v_x, need_weights=need_weights, attn_mask=attn_mask
        )

        if need_weights:
            self._last_attn_weights = attn_weights.detach()
        return attn_out

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = (
            self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        )
        v_x = (
            self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        )

        x = q_x + self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x
