from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_arch import AttnPooler, RMSNorm


class GenericConditionalMoEProjection(nn.Module):
    """Generic sparse projection from visual tokens into language token space.

    The module is intentionally domain-neutral. Routing depends only on visual
    context plus optional generic conditioning tokens, not on task classes,
    object categories, or feature-specific identifiers.
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
        condition_dim: int = 256,
        condition_embed_dim: Optional[int] = None,
        norm_layer=None,
        checkpoint: bool = False,
        top_k: int = 2,
        aux_balance_coef: float = 1.0,
        aux_entropy_coef: float = 1e-5,
        aux_zloss_coef: float = 1e-5,
        router_noise: float = 0.0,
        gate_temperature: float = 1.0,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.router_noise = float(router_noise)
        self.gate_temperature = max(float(gate_temperature), 1e-6)
        self.warmup_steps = max(0, int(warmup_steps))
        self.register_buffer("step", torch.zeros((), dtype=torch.long), persistent=False)

        self.experts = nn.ModuleList(
            [
                AttnPooler(
                    num_query=num_query,
                    num_layers=num_layers,
                    num_attention_heads=num_attention_heads,
                    encoder_hidden_size=encoder_hidden_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    norm_layer=norm_layer,
                    checkpoint=checkpoint,
                )
                for _ in range(self.num_experts)
            ]
        )

        self.image_gate = nn.Linear(encoder_hidden_size, condition_dim)
        self.condition_embed_dim = int(condition_embed_dim or encoder_hidden_size)
        if self.condition_embed_dim != encoder_hidden_size:
            self.condition_in = nn.Linear(self.condition_embed_dim, encoder_hidden_size)
        else:
            self.condition_in = nn.Identity()
        self.condition_query = nn.Parameter(torch.zeros(1, 1, encoder_hidden_size))
        nn.init.trunc_normal_(self.condition_query, std=0.02)
        self.condition_gate = nn.Linear(encoder_hidden_size, condition_dim)

        gate_dim = condition_dim * 2
        self.gate_norm = nn.LayerNorm(gate_dim)
        self.gate = nn.Linear(gate_dim, self.num_experts)
        self.output_gain = nn.Parameter(torch.ones(output_size) * 0.3)
        self.final_norm = RMSNorm(output_size)

        self.aux_balance_coef = float(aux_balance_coef)
        self.aux_entropy_coef = float(aux_entropy_coef)
        self.aux_zloss_coef = float(aux_zloss_coef)
        self._aux_loss = None
        self._gate_stats: Dict[str, torch.Tensor] = {}
        self._last_routing: Dict[str, object] = {}

    def _pool_condition(
        self,
        condition_tokens: Optional[torch.Tensor],
        condition_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if condition_tokens is None:
            return torch.zeros(batch_size, self.condition_gate.out_features, device=device, dtype=dtype)

        x = self.condition_in(condition_tokens)
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        x = torch.clamp(x, min=-1e4, max=1e4)
        query = self.condition_query.expand(batch_size, -1, -1).to(device=device, dtype=x.dtype)
        scores = torch.matmul(query, x.transpose(1, 2)) / (x.size(-1) ** 0.5)
        if condition_mask is not None:
            mask = condition_mask.to(device=device).bool()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(batch_size, -1)
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e4)
        weights = torch.softmax(scores, dim=-1)
        if condition_mask is not None:
            weights = weights * mask.unsqueeze(1).to(weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.matmul(weights, x).squeeze(1).to(self.condition_gate.weight.dtype)
        return self.condition_gate(pooled).to(dtype=dtype)

    def _routing_loss(self, probs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        load = probs.mean(dim=0)
        balance = ((load - (1.0 / self.num_experts)) ** 2).sum()
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        zloss = torch.logsumexp(logits.float(), dim=-1).square().mean()
        self._gate_stats = {
            "load": load.detach(),
            "entropy": entropy.detach(),
            "zloss": zloss.detach(),
        }
        return (
            self.aux_balance_coef * balance
            - self.aux_entropy_coef * entropy
            + self.aux_zloss_coef * zloss
        )

    def forward(
        self,
        image_embs: torch.Tensor,
        condition_tokens: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(image_embs, (list, tuple)):
            image_embs = torch.cat(image_embs, dim=1)
        batch_size = image_embs.size(0)

        image_global = torch.nan_to_num(image_embs.float().mean(dim=1), nan=0.0, posinf=1e4, neginf=-1e4)
        image_part = self.image_gate(image_global.to(self.image_gate.weight.dtype)).to(image_embs.dtype)
        condition_part = self._pool_condition(
            condition_tokens=condition_tokens,
            condition_mask=condition_mask,
            batch_size=batch_size,
            device=image_embs.device,
            dtype=image_embs.dtype,
        )

        gate_input = torch.cat([image_part, condition_part], dim=-1)
        gate_input = self.gate_norm(gate_input.to(self.gate_norm.weight.dtype))
        logits = self.gate(gate_input).float()
        if self.training and self.router_noise > 0:
            logits = logits + torch.randn_like(logits) * self.router_noise
        logits = logits / self.gate_temperature
        probs = torch.softmax(logits, dim=-1)

        if self.training and self.warmup_steps > 0 and int(self.step.item()) < self.warmup_steps:
            probs = torch.full_like(probs, 1.0 / self.num_experts)
        if self.training:
            self.step += 1

        top_vals, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
        weights = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        expert_outputs = [expert(image_embs) for expert in self.experts]
        stacked = torch.stack(expert_outputs, dim=1)
        gather_idx = top_idx[:, :, None, None].expand(-1, -1, stacked.size(2), stacked.size(3))
        selected = torch.gather(stacked, dim=1, index=gather_idx)
        output = (selected * weights[:, :, None, None].to(selected.dtype)).sum(dim=1)
        output = self.final_norm(output * self.output_gain.to(output.dtype))

        self._aux_loss = self._routing_loss(probs, logits)
        self._last_routing = {
            "top_idx": top_idx.detach().cpu(),
            "top_weight": weights.detach().cpu(),
            "prob": probs.detach().cpu(),
        }
        return output

    def get_aux_loss(self) -> Optional[torch.Tensor]:
        return self._aux_loss

    def get_gate_stats(self) -> Dict[str, torch.Tensor]:
        return self._gate_stats

    def get_last_routing(self) -> Dict[str, object]:
        return self._last_routing

