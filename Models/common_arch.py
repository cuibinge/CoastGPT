from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import warnings
import math
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


class ImageConditionedPooler(nn.Module):
    """
    Image-Conditioned Pooler v3 — queries derived FROM image content, NOT free parameters.

    Pipeline:
      image_seq [B, L, C]
        │
        ├─→ ScoreNet: per-position → Q-head scores → softmax selection
        │   → content_queries [B, Q, C]  ← different for every image!
        │
        ├─→ Cross-Attention refinement (content_queries attend back to image_seq)
        │
        ├─→ Content residual (small scale, safety net)
        ├─→ Semantic Adapter
        └─→ out_proj → [B, Q, output_size]

    No learnable query tokens. Queries ARE the image content → cannot collapse.
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
            content_residual_scale: float = 1.0,       # kept for ckpt compat, ignored
            residual_scale_start: float = 0.1,
            residual_scale_end: float = 0.3,
            residual_scale_warmup_steps: int = 1000,
            semantic_adapter_scale_init: float = 0.05,
            stage0_global_residual_scale: float = 1.0,
            # ScoreNet config
            score_hidden_mult: float = 2.0,
            score_temperature: float = 0.05,  # 106 eff_pos, balanced specialization
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.num_query = num_query
        self.stage_num = stage_num
        self.split_part = split_part
        self.residual_scale_start = float(residual_scale_start)
        self.residual_scale_end = float(residual_scale_end)
        self.residual_scale_warmup_steps = max(1, int(residual_scale_warmup_steps))
        self.content_residual_scale = float(content_residual_scale)
        self.semantic_adapter_scale_init = float(semantic_adapter_scale_init)
        self.stage0_global_residual_scale = float(stage0_global_residual_scale)
        self.score_temperature = float(score_temperature)
        self._current_step = 0

        # ── ScoreNet: image content → query selection weights ──
        # For each spatial position, predict Q scores = how much this position
        # contributes to each of the Q query tokens.
        score_hidden = max(1, int(round(hidden_size * score_hidden_mult)))
        # 2D sinusoidal position embedding so ScoreNet knows WHICH region to attend
        self.register_buffer('_score_pos_embed', self._make_2d_sincos_embed(hidden_size), persistent=False)
        # Per-query learnable 2D position bias: query i prefers certain spatial regions
        # [Q, H, W] flattened to [L, Q]. Content modulates but bias ensures baseline diversity.
        self.query_pos_bias = nn.Parameter(torch.zeros(num_query, 32, 32))
        nn.init.trunc_normal_(self.query_pos_bias, std=0.5, mean=0.0)
        # Per-query learnable "seed" embedding: each query has a unique identity
        # that persists even when image content is uniform.
        # Added at small scale (0.1) after content-weighted aggregation.
        self.query_seed = nn.Parameter(torch.zeros(1, num_query, hidden_size))
        nn.init.trunc_normal_(self.query_seed, std=0.3, mean=0.0)
        self.score_net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, score_hidden),
            nn.GELU(),
            nn.Linear(score_hidden, num_query),
        )

        if encoder_hidden_size != hidden_size:
            self.in_proj = nn.Linear(encoder_hidden_size, hidden_size)
        else:
            self.in_proj = None

        # ── Cross-Attention layers for query refinement ──
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

        # ── Semantic adapter (after content residual) ──
        adapter_hidden = hidden_size * 2
        self.semantic_adapter = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, adapter_hidden),
            nn.GELU(),
            nn.Linear(adapter_hidden, hidden_size),
            # NO final LayerNorm — it kills cross-image variance
            # by normalizing every token to mean=0, std=1.
        )
        self.adapter_scale = nn.Parameter(torch.ones(hidden_size) * self.semantic_adapter_scale_init)

        # Stage0 trains mean(image_seq) -> LayerNorm -> Linear(1024, LLM).
        # Keep the same normalizer immediately before out_proj so copied Stage0
        # weights see a compatible input distribution.
        norm_cls = norm_layer or nn.LayerNorm
        self.pre_out_norm = norm_cls(hidden_size)

    @staticmethod
    def _make_2d_sincos_embed(dim: int, max_h: int = 32, max_w: int = 32) -> torch.Tensor:
        """2D sinusoidal position embedding [max_h*max_w, dim]."""
        assert dim % 4 == 0, f"dim must be divisible by 4, got {dim}"
        half = dim // 2
        y_embed = torch.zeros(max_h, half)
        x_embed = torch.zeros(max_w, half)
        pos_y = torch.arange(max_h, dtype=torch.float32).unsqueeze(1)
        pos_x = torch.arange(max_w, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, half // 2, dtype=torch.float32) * (-math.log(10000.0) / (half // 2)))
        y_embed[:, 0::2] = torch.sin(pos_y * div)
        y_embed[:, 1::2] = torch.cos(pos_y * div)
        x_embed[:, 0::2] = torch.sin(pos_x * div)
        x_embed[:, 1::2] = torch.cos(pos_x * div)
        embed = torch.zeros(max_h, max_w, dim)
        embed[:, :, :half] = y_embed.unsqueeze(1)
        embed[:, :, half:] = x_embed.unsqueeze(0)
        return embed.reshape(-1, dim)

    def _get_residual_scale(self) -> float:
        if self._current_step >= self.residual_scale_warmup_steps:
            return self.residual_scale_end
        progress = self._current_step / max(1, self.residual_scale_warmup_steps)
        return self.residual_scale_start + (
            self.residual_scale_end - self.residual_scale_start
        ) * progress

    def _build_content_queries(self, image_embs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build query tokens from image content via ScoreNet.

        Returns:
            queries: [B, Q, C] — content-derived query tokens
            weights: [B, Q, L] — attention weights over spatial positions (for diagnostics)
        """
        B, L, C = image_embs.shape
        # Add 2D position encoding so ScoreNet can differentiate regions
        pos = self._score_pos_embed[:L].to(device=image_embs.device, dtype=image_embs.dtype)
        scored_input = image_embs + pos.unsqueeze(0)  # [B, L, C]
        scores = self.score_net(scored_input)  # [B, L, Q]
        # Add per-query position bias: query i has learned spatial preference
        H = W = int(L ** 0.5) if abs((int(L**0.5))**2 - L) < 1e-4 else None
        if H is not None:
            bias = self.query_pos_bias[:, :H, :W].reshape(self.num_query, -1).T  # [L, Q]
        else:
            bias = F.adaptive_avg_pool1d(
                self.query_pos_bias.view(self.num_query, -1).T.unsqueeze(0).float(),
                output_size=L
            ).squeeze(0).to(dtype=scores.dtype)  # [L, Q]
        scores = scores + bias.unsqueeze(0)  # [B, L, Q]
        scores = scores / max(self.score_temperature, 0.01)
        weights = F.softmax(scores.transpose(1, 2), dim=-1)  # [B, Q, L]
        queries = torch.bmm(weights, image_embs)  # [B, Q, C]
        # Add per-query seed at meaningful scale. image_seq positions are
        # near-identical (cos≈0.98), so weighted sums will be similar
        # regardless of attention distribution. The seed gives each query
        # a learnable identity that content modulates around.
        queries = queries * 0.7 + self.query_seed * 0.3
        return queries, weights

    def _resample_content_tokens(self, image_embs: torch.Tensor, target_tokens: int) -> torch.Tensor:
        if target_tokens <= 0:
            raise ValueError(f"target_tokens must be positive, got {target_tokens}")
        if image_embs.size(1) == target_tokens:
            content = image_embs
        else:
            content = F.adaptive_avg_pool1d(
                image_embs.transpose(1, 2).float(),
                output_size=target_tokens,
            ).transpose(1, 2)
            content = content.to(dtype=image_embs.dtype)
        # LayerNorm on content kills cross-image variance.
        # Content tokens are already well-conditioned after adaptive_pool.
        # Just scale to reasonable magnitude.
        content = content / (content.float().square().mean(dim=-1, keepdim=True).sqrt() + 1e-6)
        return content.to(dtype=image_embs.dtype)

    def reset_parameters(self):
        """Reinitialize all trainable components."""
        nn.init.trunc_normal_(self.query_pos_bias, std=0.5, mean=0.0)
        nn.init.trunc_normal_(self.query_seed, std=0.02, mean=0.0)
        for mod in self.score_net:
            if hasattr(mod, 'reset_parameters'):
                mod.reset_parameters()

    def calibrate_score_net(self, image_embs: torch.Tensor):
        """Scale ScoreNet output so scores/temperature has RMS ≈ 2.0.

        Ensures softmax produces meaningful spatial selection regardless
        of initial weight scale. Call once after reset_parameters().
        """
        with torch.no_grad():
            raw_scores = self.score_net(image_embs)  # [B, L, Q]
            actual_rms = raw_scores.float().square().mean().sqrt().item()
            # Target: scores / temperature ≈ 2.0 → scores.rms ≈ 2.0 * temperature
            target_rms = 2.0 * max(self.score_temperature, 0.01)
            scale = target_rms / max(actual_rms, 1e-8)
            last_linear = self.score_net[3]  # Linear(2048→144)
            last_linear.weight.data.mul_(scale)
            last_linear.bias.data.mul_(scale)
        if self.in_proj is not None:
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.zeros_(self.in_proj.bias)
        for layer in self.layers:
            if hasattr(layer, 'attn'):
                if hasattr(layer.attn, 'in_proj_weight'):
                    nn.init.xavier_uniform_(layer.attn.in_proj_weight)
                if hasattr(layer.attn, 'in_proj_bias'):
                    nn.init.zeros_(layer.attn.in_proj_bias)
                if hasattr(layer.attn, 'out_proj'):
                    nn.init.xavier_uniform_(layer.attn.out_proj.weight)
                    nn.init.zeros_(layer.attn.out_proj.bias)
            for mod in layer.mlp:
                if hasattr(mod, 'reset_parameters'):
                    mod.reset_parameters()
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        for mod in self.semantic_adapter:
            if hasattr(mod, 'reset_parameters'):
                mod.reset_parameters()
        if hasattr(self.pre_out_norm, 'reset_parameters'):
            self.pre_out_norm.reset_parameters()
        nn.init.constant_(self.adapter_scale, self.semantic_adapter_scale_init)
        self._current_step = 0

    def forward(
            self,
            image_embs: torch.Tensor,
            physical_queries: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        image_embs: [B, L, C]
        physical_queries: optional [B, L_p, C]
        """
        if self.in_proj is not None:
            image_embs = self.in_proj(image_embs)
            if physical_queries is not None:
                physical_queries = self.in_proj(physical_queries)

        B, L, C = image_embs.shape
        global_content = image_embs.float().mean(dim=1).to(dtype=image_embs.dtype)

        # ── 1. Build content-derived queries via ScoreNet ──
        query_tokens, score_weights = self._build_content_queries(image_embs)  # [B, Q, C]

        # Apply physical query correction if present
        if physical_queries is not None:
            phy_scores = self.score_net(physical_queries)
            phy_scores = phy_scores / max(self.score_temperature, 0.01)
            phy_weights = F.softmax(phy_scores.transpose(1, 2), dim=-1)
            phy_queries = torch.bmm(phy_weights, physical_queries)
            query_tokens = query_tokens + phy_queries

        # ── 2. Cross-attention refinement: queries attend back to image_seq ──
        pre_attn_queries = query_tokens  # save for early token diversity loss
        kv = image_embs.permute(1, 0, 2)  # [L, B, C]
        q_in = query_tokens.permute(1, 0, 2)  # [Q, B, C]
        for layer in self.layers:
            q_in = layer(q_in, kv, kv)

        if not self.training and hasattr(self.layers[-1], "_last_attn_weights"):
            self._last_spatial_attn = self.layers[-1]._last_attn_weights.mean(dim=1)[0].cpu()

        query_tokens = q_in.permute(1, 0, 2)  # [B, Q, C]

        # ── 3. Content residual (small scale, safety net) ──
        current_scale = self._get_residual_scale()
        content_tokens = self._resample_content_tokens(image_embs, query_tokens.size(1))
        query_tokens = query_tokens + content_tokens * current_scale

        # ── 4. Semantic adapter ──
        # Adapter is a small residual. Replacing the content tokens here breaks
        # Stage0 transfer because the copied out_proj then receives random MLP
        # activations instead of normalized visual content.
        adapter_delta = self.semantic_adapter(query_tokens) * self.adapter_scale
        query_tokens = query_tokens + adapter_delta

        # ── 5. Output projection ──
        out = self.out_proj(self.pre_out_norm(query_tokens))
        if self.stage0_global_residual_scale != 0:
            global_out = self.out_proj(self.pre_out_norm(global_content)).unsqueeze(1)
            out = out + global_out * self.stage0_global_residual_scale

        if self.training:
            self._current_step += 1
            self._last_score_weights = score_weights.detach()
            self._last_content_queries = pre_attn_queries.detach()

        return out


# Alias for backward compatibility
AttnPooler = ImageConditionedPooler


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
            router_noise: float = 0.1,
            router_noise_end: float = 0.05,
            router_noise_warmup_steps: int = 5000,
            gate_temperature: float = 1.0,
            moe_warmup_steps: int = 0,
            force_balanced_topk: bool = False,
            visual_descriptor: str = "mean",
            visual_spatial_pool_sizes: Optional[List[int]] = None,
            visual_gate_hidden_mult: float = 1.0,
            # ── new v2 anti-collapse params ──
            score_temperature: float = 0.05,
            residual_scale_start: float = 0.1,
            residual_scale_end: float = 0.3,
            residual_scale_warmup_steps: int = 1000,
            semantic_adapter_scale_init: float = 0.05,
            stage0_global_residual_scale: float = 1.0,
            aux_variance_coef: float = 0.02,
            aux_token_diversity_coef: float = 0.01,
            attention_diversity_weight: float = 0.02,
            attention_diversity_margin: float = 0.90,
            hard_load_balance_coef: float = 0.05,
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
        self.router_noise = float(router_noise)
        self.router_noise_end = float(router_noise_end)
        self.router_noise_warmup_steps = int(router_noise_warmup_steps)
        self.gate_temperature = float(gate_temperature)
        self.moe_warmup_steps = int(moe_warmup_steps)
        self.force_balanced_topk = bool(force_balanced_topk)
        self.visual_descriptor = str(visual_descriptor).lower()
        raw_pool_sizes = visual_spatial_pool_sizes if visual_spatial_pool_sizes is not None else [1, 2, 4]
        if hasattr(raw_pool_sizes, "to_list"):
            raw_pool_sizes = raw_pool_sizes.to_list()
        self.visual_spatial_pool_sizes = [max(1, int(s)) for s in raw_pool_sizes]
        self.visual_gate_hidden_mult = float(visual_gate_hidden_mult)
        self.score_temperature = float(score_temperature)
        # v2 anti-collapse
        self.residual_scale_start = float(residual_scale_start)
        self.residual_scale_end = float(residual_scale_end)
        self.residual_scale_warmup_steps = int(residual_scale_warmup_steps)
        self.semantic_adapter_scale_init = float(semantic_adapter_scale_init)
        self.stage0_global_residual_scale = float(stage0_global_residual_scale)
        self.aux_variance_coef = float(aux_variance_coef)
        self.aux_token_diversity_coef = float(aux_token_diversity_coef)
        self.attention_diversity_weight = float(attention_diversity_weight)
        self.attention_diversity_margin = float(attention_diversity_margin)
        self.hard_load_balance_coef = float(hard_load_balance_coef)
        self._moe_step = 0
        self._element_labels = None  # stored for variance loss
        self.text_embed_dim = int(text_embed_dim) if text_embed_dim is not None else int(encoder_hidden_size)

        # 1. 专家组：基于交叉注意力的序列压缩器 (AttnPooler) v2
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
                score_temperature=self.score_temperature,
                residual_scale_start=self.residual_scale_start,
                residual_scale_end=self.residual_scale_end,
                residual_scale_warmup_steps=self.residual_scale_warmup_steps,
                semantic_adapter_scale_init=self.semantic_adapter_scale_init,
                stage0_global_residual_scale=self.stage0_global_residual_scale,
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
        # 视觉 gate 使用 mean/max/std/spatial pyramid 描述子，保留粗粒度空间结构。
        self.visual_descriptor_dim = self._visual_gate_descriptor_dim(encoder_hidden_size)
        visual_gate_hidden = max(task_dim, int(round(task_dim * self.visual_gate_hidden_mult)))
        self.gate_img_proj = nn.Sequential(
            nn.LayerNorm(self.visual_descriptor_dim),
            nn.Linear(self.visual_descriptor_dim, visual_gate_hidden),
            nn.GELU(),
            nn.Linear(visual_gate_hidden, task_dim),
        )
        supported_visual_dims = {
            int(encoder_hidden_size),
            int(hidden_size),
            int(output_size),
            int(self.text_embed_dim),
            int(encoder_hidden_size * 2),
            int(encoder_hidden_size * 3),
            int(encoder_hidden_size * 4),
        }
        self.supported_visual_dims = sorted(d for d in supported_visual_dims if d > 0)
        self.visual_in_proj = nn.ModuleDict()
        for d in self.supported_visual_dims:
            if d != encoder_hidden_size:
                self.visual_in_proj[str(d)] = nn.Linear(d, encoder_hidden_size)
        self._visual_dim_warned = set()
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
        nn.init.normal_(self.gate.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.gate.bias)

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
                nn.init.normal_(self.task_gate.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(self.task_gate.bias)
                self.element_gate_norm = nn.LayerNorm(element_gate_dim)
                self.element_gate = nn.Linear(element_gate_dim, self.element_expert_count)
                nn.init.normal_(self.element_gate.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(self.element_gate.bias)

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
        text_embs = in_proj(text_embs)
        text_embs = torch.nan_to_num(
            text_embs.to(device=device, dtype=torch.float32),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )
        text_embs = torch.clamp(text_embs, min=-1e4, max=1e4)
        if text_mask is not None:
            text_mask = text_mask.to(device=device).bool()
        if text_embs.dim() == 2:
            pooled = text_embs
        else:
            q = query.expand(text_embs.size(0), -1, -1).to(device=device, dtype=torch.float32)  # [B,1,C]
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
            scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)
            scores = torch.clamp(scores, min=-1e4, max=1e4)
            weights = torch.softmax(scores, dim=-1)
            weights = torch.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
            if text_mask is not None:
                weights = weights * text_mask.unsqueeze(1).to(weights.dtype)
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            pooled = torch.matmul(weights, text_embs).squeeze(1)  # [B,C]
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1e4, neginf=-1e4)
        pooled = torch.clamp(pooled, min=-1e4, max=1e4)
        pooled = pooled.to(device=device, dtype=proj.weight.dtype)
        out = proj(pooled)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out.to(dtype=dtype)

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

    def _project_image_embs(self, image_embs: torch.Tensor) -> torch.Tensor:
        if image_embs.size(-1) == self.encoder_hidden_size:
            return image_embs

        input_dim = int(image_embs.size(-1))
        key = str(input_dim)
        if key in self.visual_in_proj:
            if input_dim not in self._visual_dim_warned:
                warnings.warn(
                    f"Visual embedding dim {input_dim} does not match encoder_hidden_size "
                    f"{self.encoder_hidden_size}; using learned adapter visual_in_proj[{key}]."
                )
                self._visual_dim_warned.add(input_dim)
            return self.visual_in_proj[key](image_embs)

        if input_dim not in self._visual_dim_warned:
            warnings.warn(
                f"Visual embedding dim {input_dim} is unsupported; fallback to linear interpolation "
                f"towards encoder_hidden_size={self.encoder_hidden_size}. This usually indicates a "
                f"checkpoint/code mismatch between the vision encoder and multimodal projector."
            )
            self._visual_dim_warned.add(input_dim)

        orig_dtype = image_embs.dtype
        flat = image_embs.to(dtype=torch.float32).reshape(-1, 1, input_dim)
        resized = F.interpolate(flat, size=self.encoder_hidden_size, mode="linear", align_corners=False)
        return resized.reshape(*image_embs.shape[:-1], self.encoder_hidden_size).to(dtype=orig_dtype)

    def _visual_gate_descriptor_dim(self, channels: int) -> int:
        if self.visual_descriptor in ("mean", "global_mean"):
            return int(channels)
        if self.visual_descriptor in ("mean_max_std", "mean_max_std_spatial"):
            width = int(channels) * 3
            if self.visual_descriptor.endswith("spatial"):
                width += int(channels) * sum(s * s for s in self.visual_spatial_pool_sizes)
            return width
        raise ValueError(
            f"Unsupported visual_descriptor={self.visual_descriptor!r}; "
            "expected 'mean', 'mean_max_std', or 'mean_max_std_spatial'."
        )

    def _build_visual_gate_descriptor(self, image_embs: torch.Tensor) -> torch.Tensor:
        if image_embs.dim() != 3:
            raise ValueError(f"image_embs must be [B, L, C], got shape={tuple(image_embs.shape)}")

        x = torch.nan_to_num(image_embs.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        x = torch.clamp(x, min=-1e4, max=1e4)
        mean = x.mean(dim=1)
        if self.visual_descriptor in ("mean", "global_mean"):
            return mean

        parts = [mean, x.max(dim=1).values, x.std(dim=1, unbiased=False)]
        if self.visual_descriptor.endswith("spatial"):
            B, L, C = x.shape
            side = math.isqrt(int(L))
            if side * side == int(L):
                grid = x.transpose(1, 2).reshape(B, C, side, side)
                for pool_size in self.visual_spatial_pool_sizes:
                    pooled = F.adaptive_avg_pool2d(grid, output_size=(pool_size, pool_size))
                    parts.append(pooled.flatten(2).reshape(B, C * pool_size * pool_size))
            else:
                sequence = x.transpose(1, 2)
                for pool_size in self.visual_spatial_pool_sizes:
                    pooled = F.adaptive_avg_pool1d(sequence, output_size=pool_size * pool_size)
                    parts.append(pooled.reshape(B, C * pool_size * pool_size))

        descriptor = torch.cat(parts, dim=-1)
        expected_width = int(getattr(self, "visual_descriptor_dim", descriptor.size(-1)))
        if descriptor.size(-1) != expected_width:
            raise RuntimeError(
                f"visual descriptor width {descriptor.size(-1)} != expected {expected_width}"
            )
        return descriptor

    def _build_visual_gate_feature(self, image_embs: torch.Tensor) -> torch.Tensor:
        descriptor = self._build_visual_gate_descriptor(image_embs)
        proj_dtype = next(self.gate_img_proj.parameters()).dtype
        img_gate = self.gate_img_proj(descriptor.to(dtype=proj_dtype))
        img_gate = torch.nan_to_num(img_gate, nan=0.0, posinf=1e4, neginf=-1e4)
        img_gate = torch.clamp(img_gate, min=-1e4, max=1e4)
        return img_gate.to(dtype=image_embs.dtype)

    def _variance_loss(
            self,
            embeddings: torch.Tensor,
            element_labels: Optional[List[str]] = None,
            soft_margin: float = 0.85,
            hard_margin: float = 0.70,
    ) -> torch.Tensor:
        """Two-tier anti-collapse: soft margin for all pairs, hard margin for different-class pairs."""
        B = embeddings.size(0)
        if B < 2:
            return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
        if embeddings.dim() == 3:
            pooled = embeddings.float().mean(dim=1)  # [B, D]
        else:
            pooled = embeddings.float()
        norms = F.normalize(pooled, dim=-1, eps=1e-6)  # [B, D]
        cos_mat = torch.matmul(norms, norms.T)  # [B, B]
        mask = ~torch.eye(B, device=cos_mat.device, dtype=torch.bool)

        # ── A. Soft: all pairs, margin=0.85 (don't collapse to same direction) ──
        soft_excess = F.relu(cos_mat[mask] - soft_margin)
        soft_loss = soft_excess.mean()

        # ── B. Hard: only different-class pairs, margin=0.70 ──
        hard_loss = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
        if element_labels is not None and len(element_labels) == B:
            # Build same-class mask
            same_class = torch.zeros(B, B, dtype=torch.bool, device=cos_mat.device)
            for i in range(B):
                for j in range(i + 1, B):
                    if element_labels[i] and element_labels[j] and element_labels[i] == element_labels[j]:
                        same_class[i, j] = True
                        same_class[j, i] = True
            hard_mask = mask & ~same_class
            if hard_mask.any():
                hard_excess = F.relu(cos_mat[hard_mask] - hard_margin)
                hard_loss = hard_excess.mean()

        loss = 0.5 * soft_loss + 0.5 * hard_loss
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=embeddings.dtype)

    def _attention_diversity_loss(
            self, attn_weights: torch.Tensor, max_cos: float = 0.90
    ) -> torch.Tensor:
        """Penalize similar attention distributions across queries.

        attn_weights: [B, Q, L] — softmax attention over L spatial positions.
        Different queries should attend to different regions.
        """
        B, Q, L = attn_weights.shape
        if Q < 2:
            return torch.zeros((), device=attn_weights.device, dtype=attn_weights.dtype)
        # Mean over batch to get [Q, L]
        w_mean = attn_weights.float().mean(dim=0)  # [Q, L]
        w_norm = F.normalize(w_mean, dim=-1, eps=1e-6)  # [Q, L]
        intra_cos = torch.matmul(w_norm, w_norm.T)  # [Q, Q]
        mask = ~torch.eye(Q, device=intra_cos.device, dtype=torch.bool)
        excess = F.relu(intra_cos[mask] - max_cos)
        loss = excess.mean()
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=attn_weights.dtype)

    def _get_router_noise(self) -> float:
        """Dynamic router noise: decays from router_noise → router_noise_end."""
        if self._moe_step >= self.router_noise_warmup_steps:
            return self.router_noise_end
        progress = self._moe_step / max(1, self.router_noise_warmup_steps)
        return self.router_noise + (self.router_noise_end - self.router_noise) * progress

    def _hard_load_balance_loss(
            self,
            gate_weights: torch.Tensor,
            num_experts: int,
            top_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Balance soft importance and actual top-k expert coverage.

        The soft gate can look balanced while hard dispatch starves an expert after
        top-k selection. Use a straight-through top-k coverage term so the loss
        reflects the routed experts while gradients still flow through gate_weights.
        """
        soft_load = gate_weights.float().mean(dim=0)
        target = 1.0 / num_experts
        soft_loss = (soft_load - target).abs().mean()
        if top_indices is None:
            return soft_loss.to(dtype=gate_weights.dtype)

        hard_load = torch.zeros(num_experts, device=gate_weights.device, dtype=torch.float32)
        for i in range(num_experts):
            hard_load[i] = ((top_indices == i).any(dim=1)).float().mean()
        target = float(top_indices.size(-1)) / float(num_experts)
        hard_proxy = hard_load + (soft_load - soft_load.detach())
        relative_error = (hard_proxy - target) / max(target, 1e-6)
        hard_loss = 2.0 * relative_error.pow(2).mean()
        return (soft_loss + hard_loss).to(dtype=gate_weights.dtype)

    def _token_diversity_loss(self, proj_emb: torch.Tensor, max_sim: float = 0.5) -> torch.Tensor:
        """Penalize high cosine similarity between query tokens within each sample."""
        if proj_emb.size(1) < 2:
            return torch.zeros((), device=proj_emb.device, dtype=proj_emb.dtype)
        norms = F.normalize(proj_emb.float(), dim=-1)  # [B, Q, D]
        intra_cos = torch.matmul(norms, norms.transpose(1, 2))  # [B, Q, Q]
        mask = ~torch.eye(intra_cos.size(1), device=intra_cos.device, dtype=torch.bool)
        excess = F.relu(intra_cos[:, mask] - float(max_sim))
        loss = excess.mean()
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=proj_emb.dtype)

    def _balanced_topk_enabled(self) -> bool:
        return (
            self.force_balanced_topk
            and self.routing_strategy in ("two_stage", "task_then_element")
            and self.task_expert_count > 0
            and self.element_expert_count > 0
            and self.top_k >= 2
        )

    def _select_topk_balanced(self, gate_weights: torch.Tensor):
        if not self._balanced_topk_enabled():
            top_weights, top_indices = torch.topk(gate_weights, self.top_k, dim=-1)
            return top_weights, top_indices, False

        task_weights = gate_weights[:, :self.task_expert_count]
        element_weights = gate_weights[:, self.task_expert_count:]
        task_top_weights, task_top_indices = torch.topk(task_weights, 1, dim=-1)
        elem_top_weights, elem_top_indices = torch.topk(element_weights, 1, dim=-1)

        if self.training:
            if gate_weights.size(0) >= self.num_experts:
                task_top_indices = self._ensure_branch_coverage(task_weights, task_top_indices)
                elem_top_indices = self._ensure_branch_coverage(element_weights, elem_top_indices)
            else:
                rows = torch.arange(gate_weights.size(0), device=gate_weights.device)
                step = int(getattr(self, '_moe_step', 0))
                task_top_indices = ((rows + step) % self.task_expert_count).unsqueeze(-1)
                elem_top_indices = ((rows + step) % self.element_expert_count).unsqueeze(-1)
            task_top_weights = task_weights.gather(1, task_top_indices)
            elem_top_weights = element_weights.gather(1, elem_top_indices)

        elem_top_indices = elem_top_indices + self.task_expert_count

        top_weights = torch.cat([task_top_weights, elem_top_weights], dim=-1)
        top_indices = torch.cat([task_top_indices, elem_top_indices], dim=-1)

        if self.top_k > 2:
            selected_mask = torch.zeros_like(gate_weights, dtype=torch.bool)
            selected_mask.scatter_(1, top_indices, True)
            remaining_weights = gate_weights.masked_fill(selected_mask, -1.0)
            extra_weights, extra_indices = torch.topk(remaining_weights, self.top_k - 2, dim=-1)
            top_weights = torch.cat([top_weights, extra_weights], dim=-1)
            top_indices = torch.cat([top_indices, extra_indices], dim=-1)

        sort_order = torch.argsort(top_weights, dim=-1, descending=True)
        top_weights = torch.gather(top_weights, 1, sort_order)
        top_indices = torch.gather(top_indices, 1, sort_order)
        return top_weights, top_indices, True

    @staticmethod
    def _ensure_branch_coverage(
            branch_weights: torch.Tensor,
            selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Ensure each branch expert receives at least one routed sample."""
        if selected_indices.size(1) != 1:
            return selected_indices
        num_branch_experts = branch_weights.size(1)
        if num_branch_experts <= 1 or branch_weights.size(0) < num_branch_experts:
            return selected_indices

        selected = selected_indices.clone()
        for expert_idx in range(num_branch_experts):
            if (selected[:, 0] == expert_idx).any():
                continue
            order = torch.argsort(branch_weights[:, expert_idx], descending=True)
            for row in order:
                row_idx = int(row.item())
                current = int(selected[row_idx, 0].item())
                if (selected[:, 0] == current).sum() <= 1:
                    continue
                selected[row_idx, 0] = expert_idx
                break
        return selected

    def _select_warmup_topk(self, gate_weights: torch.Tensor, step_idx: int):
        B_dev = gate_weights.size(0)
        k = self.top_k
        device = gate_weights.device
        rows = torch.arange(B_dev, device=device)

        if self._balanced_topk_enabled():
            task_base = (rows + step_idx) % self.task_expert_count
            elem_base = self.task_expert_count + ((rows + step_idx) % self.element_expert_count)
            top_indices = torch.stack([task_base, elem_base], dim=-1)
            if k > 2:
                global_base = (rows + step_idx) % self.num_experts
                candidate_offsets = torch.arange(self.num_experts, device=device)
                candidates = (global_base.unsqueeze(-1) + candidate_offsets.unsqueeze(0)) % self.num_experts
                used = (candidates.unsqueeze(-1) == top_indices.unsqueeze(1)).any(dim=-1)
                extras = candidates.masked_select(~used).reshape(B_dev, self.num_experts - 2)[:, : k - 2]
                top_indices = torch.cat([top_indices, extras], dim=-1)
            top_weights = torch.ones_like(top_indices, dtype=gate_weights.dtype) / float(k)
            return top_weights, top_indices, True

        base = (rows + step_idx) % self.num_experts
        top_indices = base.unsqueeze(-1)
        if k > 1:
            top_indices = (base.unsqueeze(-1) + torch.arange(k, device=device)) % self.num_experts
        top_weights = torch.ones_like(top_indices, dtype=gate_weights.dtype) / float(k)
        return top_weights, top_indices, False

    def forward(
            self,
            image_embs: torch.Tensor,
            physical_prompts: Optional[torch.Tensor] = None,
            task_text_embs: Optional[torch.Tensor] = None,
            element_text_embs: Optional[torch.Tensor] = None,
            physical_prompt_mask: Optional[torch.Tensor] = None,
            task_text_mask: Optional[torch.Tensor] = None,
            element_text_mask: Optional[torch.Tensor] = None,
            element_text_labels: Optional[List[str]] = None,
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
        image_embs = self._project_image_embs(image_embs)

        # Lazy ScoreNet calibration: on first training step, scale weights
        # so scores/temperature has RMS ≈ 2 (meaningful softmax selection).
        if self.training and not getattr(self, '_score_nets_calibrated', True):
            for expert in self.experts:
                expert.calibrate_score_net(image_embs)
            self._score_nets_calibrated = True

        B, L_v, C = image_embs.shape
        has_task_text = task_text_embs is not None
        has_element_text = element_text_embs is not None

        # ==========================================
        # 1. 任务与要素双驱动的动态门控计算 (Sequence-level Routing)
        # ==========================================
        # 获取宏观全局 + 粗粒度空间图像上下文并对齐到 task_dim
        img_gate = self._build_visual_gate_feature(image_embs)  # [B, task_dim]
        # v3: two-tier anti-collapse (soft margin 0.85 all pairs + hard margin 0.70 diff-class)
        visual_gate_variance_loss = torch.zeros((), device=image_embs.device, dtype=image_embs.dtype)
        if self.training and self.aux_variance_coef > 0:
            visual_gate_variance_loss = self._variance_loss(img_gate.unsqueeze(1), element_labels=element_text_labels)

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
        gate_input = torch.cat(gate_parts, dim=-1)
        gate_input = torch.nan_to_num(gate_input, nan=0.0, posinf=1e4, neginf=-1e4)
        gate_input = torch.clamp(gate_input, min=-1e4, max=1e4)

        # 计算专家倾向性
        task_logits = None
        elem_logits = None
        if self.routing_strategy in ("two_stage", "task_then_element") and self.task_gate is not None:
            task_parts = [img_gate, t_embs]
            elem_parts = [img_gate, e_embs]
            if self.include_physical_prompt and phy_gate is not None:
                task_parts.append(phy_gate)
                elem_parts.append(phy_gate)
            task_input = torch.cat(task_parts, dim=-1).to(dtype=self.task_gate_norm.weight.dtype)
            elem_input = torch.cat(elem_parts, dim=-1).to(dtype=self.element_gate_norm.weight.dtype)
            task_input = torch.nan_to_num(task_input, nan=0.0, posinf=1e4, neginf=-1e4)
            elem_input = torch.nan_to_num(elem_input, nan=0.0, posinf=1e4, neginf=-1e4)
            task_input = torch.clamp(task_input, min=-1e4, max=1e4)
            elem_input = torch.clamp(elem_input, min=-1e4, max=1e4)

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
            gate_input = gate_input.to(dtype=self.gate_norm.weight.dtype)
            gate_logits = self.gate(self.gate_norm(gate_input))  # [B, num_experts]
        invalid_gate_ratio = (~torch.isfinite(gate_logits)).float().mean()
        gate_logits = torch.nan_to_num(gate_logits, nan=0.0, posinf=15.0, neginf=-15.0)
        gate_logits = torch.clamp(gate_logits, min=-15.0, max=15.0)
        # Router noise: decays from router_noise → router_noise_end over warmup
        if self.training:
            noise_std = self._get_router_noise()
            if noise_std > 0:
                gate_logits = gate_logits + torch.randn_like(gate_logits) * noise_std

        # Gate temperature for smoother routing (SOP Section 9)
        temperature = float(getattr(self, "gate_temperature", 1.0))
        gate_weights = torch.softmax(gate_logits / temperature, dim=-1)  # [B, num_experts]
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
                    task_text_only_input = torch.cat(task_text_only_parts, dim=-1).to(
                        dtype=self.task_gate_norm.weight.dtype
                    )
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
                    task_no_text_input = torch.cat(task_no_text_parts, dim=-1).to(
                        dtype=self.task_gate_norm.weight.dtype
                    )
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
                    element_text_only_input = torch.cat(element_text_only_parts, dim=-1).to(
                        dtype=self.element_gate_norm.weight.dtype
                    )
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
                    element_no_text_input = torch.cat(element_no_text_parts, dim=-1).to(
                        dtype=self.element_gate_norm.weight.dtype
                    )
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
                    task_drop_input = torch.cat(task_drop_parts, dim=-1).to(dtype=self.gate_norm.weight.dtype)
                    task_drop_logits = self.gate(self.gate_norm(task_drop_input))
                    task_effect = torch.mean(torch.abs(gate_logits - task_drop_logits))
                    task_route_loss = F.relu(
                        gate_weights.new_tensor(self.route_effect_margin) - task_effect
                    )
                    task_route_effect = task_route_loss
                if has_element_text:
                    element_drop_parts = [img_gate, t_embs, torch.zeros_like(e_embs)]
                    if self.include_physical_prompt and phy_gate is not None:
                        element_drop_parts.append(phy_gate)
                    element_drop_input = torch.cat(element_drop_parts, dim=-1).to(dtype=self.gate_norm.weight.dtype)
                    element_drop_logits = self.gate(self.gate_norm(element_drop_input))
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

        # 选取 Top-K 专家；two-stage 可强制每次至少包含任务/要素各一个专家。
        balanced_topk = False
        if self.training:
            warmup_steps = int(getattr(self, "moe_warmup_steps", 0))
            self._moe_step = getattr(self, "_moe_step", 0) + 1
            if self._moe_step <= warmup_steps:
                step_idx = self._moe_step % self.num_experts
                top_weights, top_indices, balanced_topk = self._select_warmup_topk(gate_weights, step_idx)
            else:
                top_weights, top_indices, balanced_topk = self._select_topk_balanced(gate_weights)
        else:
            top_weights, top_indices, balanced_topk = self._select_topk_balanced(gate_weights)

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
                "balanced_topk": bool(balanced_topk),
                "visual_descriptor": self.visual_descriptor,
                "visual_spatial_pool_sizes": list(self.visual_spatial_pool_sizes),
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
                "gate_variance": visual_gate_variance_loss,
                "proj_variance": torch.zeros((), device=image_embs.device, dtype=image_embs.dtype),
                "token_diversity": torch.zeros((), device=image_embs.device, dtype=image_embs.dtype),
            }

            with torch.no_grad():
                # top-1 load
                top1_indices = top_indices[:, 0]
                load = torch.zeros(self.num_experts, device=image_embs.device, dtype=torch.float32)
                for i in range(self.num_experts):
                    load[i] = (top1_indices == i).float().mean()
                # top-k load: fraction of samples where expert appears in ANY position
                load_topk = torch.zeros(self.num_experts, device=image_embs.device, dtype=torch.float32)
                for i in range(self.num_experts):
                    load_topk[i] = ((top_indices == i).any(dim=1)).float().mean()
                self._gate_stats = {
                    "importance": importance_soft.detach(),
                    "load": load,
                    "load_topk": load_topk,
                    "entropy": entropy.detach(),
                    "zloss": zloss.detach(),
                    "invalid_gate_ratio": invalid_gate_ratio.detach(),
                    "balanced_topk": torch.tensor(float(balanced_topk), device=image_embs.device),
                    "task_route_loss": task_route_loss.detach(),
                    "element_route_loss": element_route_loss.detach(),
                    "task_route_kl": task_route_kl.detach(),
                    "element_route_kl": element_route_kl.detach(),
                    "task_route_effect": task_route_effect.detach(),
                    "element_route_effect": element_route_effect.detach(),
                    "task_element_orth": task_element_orth.detach(),
                    "gate_variance_loss": visual_gate_variance_loss.detach(),
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
        final_output = torch.zeros(
            B,
            self.num_query,
            self.output_size,
            device=image_embs.device,
            dtype=image_embs.dtype,
        )
        dispatch_weights = torch.zeros(
            B,
            self.num_experts,
            device=image_embs.device,
            dtype=top_weights.dtype,
        )
        dispatch_weights.scatter_(1, top_indices, top_weights)

        # Keep every expert on the graph for every rank.
        # Sparse per-rank expert execution can leave different parameter subsets unused
        # across ranks, which is fragile with DeepSpeed ZeRO + HCCL on NPU.
        for e_id in range(self.num_experts):
            expert_out = self.experts[e_id](
                z_tilde,
                physical_queries=phy_query if phy_query is not None else None,
            )  # [B, num_query, output_size]
            if expert_out.size(1) != self.num_query:
                expert_out = expert_out[:, : self.num_query, :]

            expert_weight = dispatch_weights[:, e_id].unsqueeze(1).unsqueeze(2).to(expert_out.dtype)
            final_output = final_output + expert_out * expert_weight

        # ==========================================
        # 3.5 Attention diversity + hard load balance
        # ==========================================
        attn_div_loss = torch.zeros((), device=image_embs.device, dtype=image_embs.dtype)
        hard_load_loss = torch.zeros((), device=image_embs.device, dtype=image_embs.dtype)
        if self.training:
            if self.attention_diversity_weight > 0:
                attn_losses = []
                for e_id in range(self.num_experts):
                    sw = getattr(self.experts[e_id], '_last_score_weights', None)
                    if sw is not None and sw.size(1) >= 2:
                        attn_losses.append(self._attention_diversity_loss(
                            sw, max_cos=self.attention_diversity_margin
                        ))
                if attn_losses:
                    attn_div_loss = torch.stack(attn_losses).mean()
            if self.hard_load_balance_coef > 0:
                hard_load_loss = self._hard_load_balance_loss(
                    gate_weights, self.num_experts, top_indices
                )
            self._aux_terms['attn_diversity'] = attn_div_loss
            self._aux_terms['hard_load_balance'] = hard_load_loss
            self._gate_stats['attn_diversity_loss'] = attn_div_loss.detach()
            self._gate_stats['hard_load_balance_loss'] = hard_load_loss.detach()

        # ==========================================
        # 4. 特征对齐与约束输出
        # ==========================================
        final_output = self.final_norm(final_output) * self.output_gain
        if self.training and (self.aux_variance_coef > 0 or self.aux_token_diversity_coef > 0):
            proj_variance_loss = torch.zeros((), device=image_embs.device, dtype=image_embs.dtype)
            tok_div_loss = torch.zeros((), device=image_embs.device, dtype=image_embs.dtype)
            if self.aux_variance_coef > 0:
                proj_variance_loss = self._variance_loss(final_output, element_labels=element_text_labels)
            if self.aux_token_diversity_coef > 0:
                tok_div_loss = self._token_diversity_loss(final_output)
            self._aux_terms["proj_variance"] = proj_variance_loss
            self._aux_terms["token_diversity"] = tok_div_loss
            self._gate_stats["proj_variance_loss"] = proj_variance_loss.detach()
            self._gate_stats["token_diversity_loss"] = tok_div_loss.detach()

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
        gate_variance = self._aux_terms.get(
            "gate_variance", torch.tensor(0.0, device=self.gate.weight.device)
        )
        proj_variance = self._aux_terms.get(
            "proj_variance", torch.tensor(0.0, device=self.gate.weight.device)
        )
        token_diversity = self._aux_terms.get(
            "token_diversity", torch.tensor(0.0, device=self.gate.weight.device)
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

    def reset_gate_and_experts(self):
        """Reinitialize gate_img_proj, all experts (AttnPooler), and gates."""
        self._score_nets_calibrated = False  # will calibrate on first forward
        # Reinit gate_img_proj
        for mod in self.gate_img_proj:
            if hasattr(mod, 'reset_parameters'):
                mod.reset_parameters()
        # Reinit each expert
        for expert in self.experts:
            expert.reset_parameters()
        # Reinit task/element text projections
        nn.init.trunc_normal_(self.task_text_query, std=0.02)
        nn.init.trunc_normal_(self.element_text_query, std=0.02)
        if hasattr(self, 'task_text_in_proj') and not isinstance(self.task_text_in_proj, nn.Identity):
            if hasattr(self.task_text_in_proj, 'reset_parameters'):
                self.task_text_in_proj.reset_parameters()
        if hasattr(self, 'element_text_in_proj') and not isinstance(self.element_text_in_proj, nn.Identity):
            if hasattr(self.element_text_in_proj, 'reset_parameters'):
                self.element_text_in_proj.reset_parameters()
        if hasattr(self, 'task_text_proj'):
            self.task_text_proj.reset_parameters()
        if hasattr(self, 'element_text_proj'):
            self.element_text_proj.reset_parameters()
        # Reinit gate layers
        for gate_attr in ['gate', 'task_gate', 'element_gate']:
            g = getattr(self, gate_attr, None)
            if g is not None:
                nn.init.normal_(g.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(g.bias)
        # Reset step counter
        self._moe_step = 0
        # Reset output gain
        nn.init.constant_(self.output_gain, 0.3)




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
