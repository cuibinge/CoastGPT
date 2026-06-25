# CoastGPT MoE Gate Redesign Spec

## Goal
Improve CoastGPT MoE routing for Stage 4 by preventing task/element branch starvation and giving the router richer spatial evidence from the visual token sequence.

## Scope
This change modifies the sample-level `MoEProjection` router only. It does not wire FPN detection heads into the LLM path and does not implement region-level or token-level expert dispatch in this iteration.

## Design
Use the existing `two_stage` split of experts into task and element branches, but make expert selection branch-balanced when `force_balanced_topk` is enabled. For `top_k >= 2`, select at least one expert from the task branch and one from the element branch, then fill any remaining slots from the best unused global experts. This directly targets expert starvation noted in the total design.

Replace the old image gate descriptor `mean(image_tokens)` with a richer descriptor:

- mean over visual tokens
- max over visual tokens
- standard deviation over visual tokens
- spatial pyramid pooled visual descriptors using configurable pool sizes, default `[1, 2, 4]`

The descriptor is projected through `LayerNorm + MLP` to the existing `task_dim`, then reused by both task and element gate branches. Checkpoint compatibility is not required; retraining from Stage 1 is acceptable.

## Config
Add under `moe_proj`:

```yaml
force_balanced_topk: true
visual_descriptor: mean_max_std_spatial
visual_spatial_pool_sizes: [1, 2, 4]
visual_gate_hidden_mult: 1.0
```

These keys will be enabled in `Configs/step2_dual.yaml` and `Configs/step3_dual.yaml`.

## Testing
Add focused CPU tests for `MoEProjection`:

1. In two-stage routing, even if task logits dominate, selected `top_indices` must contain at least one task expert and one element expert.
2. The visual descriptor must distinguish token sequences with identical mean but different spatial distribution.
3. Forward output shape remains `[B, num_query, output_size]`.

## Non-Goals
Region-level/token-level MoE will be a later experiment. This patch records enough routing stats to support that next step, but does not change dispatch granularity.
