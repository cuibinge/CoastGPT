import ml_collections
import torch
import torch.nn as nn
from typing import Dict, Optional

from .common_arch import LayerNorm, LayerNormFp32, MoEProjection, PhysicalPromptEncoder
import torch_npu  # noqa: F401


class EmbeddingModel(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__()

        if config.adjust_norm:
            norm_layer = LayerNormFp32 if config.dtype in ("float16", "bfloat16") else LayerNorm
        else:
            norm_layer = LayerNorm

        moe_cfg = getattr(config, "moe_proj", ml_collections.ConfigDict())
        phy_cfg = getattr(config, "physics", ml_collections.ConfigDict())

        pool_scales = phy_cfg.get("prompt_pool_sizes", [1, 2, 4])
        if hasattr(pool_scales, "to_list"):
            pool_scales = pool_scales.to_list()
        self.use_physical_prompt = bool(phy_cfg.get("prompt_enabled", True))
        self.physical_prompt_encoder = PhysicalPromptEncoder(
            in_channels=int(phy_cfg.get("prompt_in_channels", 1)),
            embed_dim=getattr(config, "alignment_dim", 768),
            pool_scales=pool_scales,
        )

        self.projection = MoEProjection(
            num_experts=int(moe_cfg.get("num_experts", 4)),
            num_query=config.rgb_vision.attn_pooler.num_query,
            num_layers=config.rgb_vision.attn_pooler.num_layers,
            num_attention_heads=config.rgb_vision.attn_pooler.num_attn_heads,
            encoder_hidden_size=getattr(config, "alignment_dim", 768),
            hidden_size=getattr(config, "alignment_dim", 768),
            output_size=config.text.hidden_size,
            num_tasks=int(moe_cfg.get("num_tasks", 3)),
            num_elements=int(moe_cfg.get("num_elements", phy_cfg.get("num_elements", 10))),
            task_dim=int(moe_cfg.get("task_dim", 256)),
            include_physical_prompt=bool(moe_cfg.get("include_physical_prompt", True)),
            norm_layer=norm_layer,
            checkpoint=getattr(config, "use_checkpoint", False),
            top_k=int(moe_cfg.get("top_k", 2)),
        )

    def _build_physical_prompts(
            self,
            data: Dict,
            device: torch.device,
            dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
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
        batch_size = image_embedding.shape[0]
        device = image_embedding.device

        task_ids = data.get("task_ids", None)
        if task_ids is None:
            task_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            task_ids = task_ids.to(device)

        element_ids = data.get("category_ids", None)
        if element_ids is None:
            element_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            element_ids = element_ids.to(device)

        physical_prompts = self._build_physical_prompts(data, device=device, dtype=image_embedding.dtype)

        return self.projection(
            image_embs=image_embedding,
            physical_prompts=physical_prompts,
            task_ids=task_ids,
            element_ids=element_ids,
        )

    def encode_test(
            self,
            image_embedding: torch.Tensor,
            task_ids: Optional[torch.Tensor] = None,
            element_ids: Optional[torch.Tensor] = None,
    ):
        if task_ids is None:
            batch_size = image_embedding.shape[0]
            task_ids = torch.zeros(batch_size, dtype=torch.long, device=image_embedding.device)
        else:
            task_ids = task_ids.to(image_embedding.device)

        if element_ids is not None:
            element_ids = element_ids.to(image_embedding.device)

        return self.projection(
            image_embs=image_embedding,
            physical_prompts=None,
            task_ids=task_ids,
            element_ids=element_ids,
        )

    def get_aux_loss(self) -> torch.Tensor:
        if hasattr(self.projection, "get_aux_loss"):
            return self.projection.get_aux_loss()
        return torch.tensor(0.0)

    def get_gate_stats(self) -> Dict[str, torch.Tensor]:
        if hasattr(self.projection, "get_gate_stats"):
            return self.projection.get_gate_stats()
        return {}
