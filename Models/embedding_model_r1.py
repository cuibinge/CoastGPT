import ml_collections
import torch
import torch.nn as nn
from typing import Dict, Optional

from .common_arch import LayerNorm, LayerNormFp32, PhysicalPromptEncoder
from .generic_projection import GenericConditionalMoEProjection
try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None


class EmbeddingModel(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__()

        if config.adjust_norm:
            norm_layer = LayerNormFp32 if config.dtype in ("float16", "bfloat16") else LayerNorm
        else:
            norm_layer = LayerNorm

        moe_cfg = getattr(config, "moe_proj", ml_collections.ConfigDict())
        phy_cfg = getattr(config, "physics", ml_collections.ConfigDict())
        text_embed_dim = int(getattr(config.text, "hidden_size", getattr(config, "alignment_dim", 768)))
        alignment_dim = int(getattr(config, "alignment_dim", 768))

        pool_scales = phy_cfg.get("prompt_pool_sizes", [1, 2, 4])
        if hasattr(pool_scales, "to_list"):
            pool_scales = pool_scales.to_list()
        self.use_physical_prompt = bool(phy_cfg.get("prompt_enabled", True))
        self.physical_prompt_encoder = PhysicalPromptEncoder(
            in_channels=int(phy_cfg.get("prompt_in_channels", 1)),
            embed_dim=getattr(config, "alignment_dim", 768),
            pool_scales=pool_scales,
        )

        self.projection = GenericConditionalMoEProjection(
            num_experts=int(moe_cfg.get("num_experts", 4)),
            num_query=config.rgb_vision.attn_pooler.num_query,
            num_layers=config.rgb_vision.attn_pooler.num_layers,
            num_attention_heads=config.rgb_vision.attn_pooler.num_attn_heads,
            encoder_hidden_size=getattr(config, "alignment_dim", 768),
            hidden_size=getattr(config, "alignment_dim", 768),
            output_size=config.text.hidden_size,
            condition_dim=int(moe_cfg.get("condition_dim", 256)),
            condition_embed_dim=text_embed_dim,
            norm_layer=norm_layer,
            checkpoint=getattr(config, "use_checkpoint", False),
            top_k=int(moe_cfg.get("top_k", 2)),
            aux_balance_coef=float(moe_cfg.get("aux_balance_coef", 1.0)),
            aux_entropy_coef=float(moe_cfg.get("aux_entropy_coef", 1e-5)),
            aux_zloss_coef=float(moe_cfg.get("aux_zloss_coef", 1e-5)),
            router_noise=float(moe_cfg.get("router_noise", 0.1)),
            gate_temperature=float(moe_cfg.get("gate_temperature", 1.0)),
            warmup_steps=int(moe_cfg.get("warmup_steps", moe_cfg.get("moe_warmup_steps", 0))),
        )

    def _build_physical_prompts(
            self,
            data: Dict,
            device: torch.device,
            dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        if "physical_prompt_embs" in data and data["physical_prompt_embs"] is not None:
            embs = data["physical_prompt_embs"]
            if torch.is_tensor(embs):
                return embs.to(device=device, dtype=dtype)
            return None

        if not self.use_physical_prompt:
            return None
        tsm = data.get("tsm", None)
        if tsm is None or not torch.is_tensor(tsm):
            return None

        mask = data.get("mask", None)
        tsm = tsm.to(device=device, dtype=dtype)
        if mask is not None and torch.is_tensor(mask):
            mask = mask.to(device=device, dtype=dtype)
        else:
            mask = None
        prompts = self.physical_prompt_encoder(tsm, mask)
        if prompts is not None and prompts.dtype != dtype:
            prompts = prompts.to(dtype=dtype)
        return prompts

    def forward(self, data: Dict, image_embedding: torch.Tensor):
        device = image_embedding.device
        physical_prompts = self._build_physical_prompts(data, device=device, dtype=image_embedding.dtype)
        physical_prompt_mask = data.get("physical_prompt_attention_mask", None)
        if physical_prompt_mask is not None and torch.is_tensor(physical_prompt_mask):
            physical_prompt_mask = physical_prompt_mask.to(device=device)
        else:
            physical_prompt_mask = None

        return self.projection(
            image_embs=image_embedding,
            condition_tokens=physical_prompts,
            condition_mask=physical_prompt_mask,
        )

    def encode_test(
            self,
            image_embedding: torch.Tensor,
            condition_tokens: Optional[torch.Tensor] = None,
    ):
        if condition_tokens is not None:
            condition_tokens = condition_tokens.to(image_embedding.device, dtype=image_embedding.dtype)

        return self.projection(
            image_embs=image_embedding,
            condition_tokens=condition_tokens,
        )

    def get_aux_loss(self) -> torch.Tensor:
        if hasattr(self.projection, "get_aux_loss"):
            return self.projection.get_aux_loss()
        return torch.tensor(0.0)

    def get_gate_stats(self) -> Dict[str, torch.Tensor]:
        if hasattr(self.projection, "get_gate_stats"):
            return self.projection.get_gate_stats()
        return {}

    def get_last_routing(self) -> Dict[str, object]:
        if hasattr(self.projection, "get_last_routing"):
            return self.projection.get_last_routing()
        return {}
