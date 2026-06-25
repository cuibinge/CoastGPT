# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoastGPT is a multimodal coastal remote sensing model that generates GeoJSON annotations from satellite imagery. It combines a dual vision encoder (DINOv3 ViT-L/16 global + ConvNeXt-Base local), a LLaMA-based language model, a task-aware Mixture-of-Experts (MoE) router, and a physics-constrained decoder. The model runs on Huawei Ascend NPUs with DeepSpeed distributed training.

## Training Pipeline (3 stages)

| Stage | Script | Config | Hardware | Key params |
|-------|--------|--------|----------|------------|
| 1 — Pretrain | `train_stage_one.py` | `Configs/train_dual.yaml` | 8 NPU | batch=8/GPU |
| 2 — MoE fine-tune | `train_stage_two.py` | `Configs/step2_dual.yaml` | 8 NPU | batch=4/GPU |
| 3 — GeoJSON gen | `train_stage_three.py` | `Configs/step3_dual.yaml` | 2 NPU | batch=2/GPU, accum=8 |

Stage 3 inherits core trainer logic from stage 2 (`from train_stage_two import ...`). Stages 1 and 2 share nearly identical structure; all three use `ConfigArgumentParser` (YAML config + CLI override).

## Environment

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate PyTorch-2.1.0
export HF_HOME=/home/ma-user/work/CoastGPT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

Python 3.10, torch 2.1.2, DeepSpeed, CANN-8.0.RC2. Package list in `requirement.txt`. HF models use offline cache at `hf_cache/`.

## Task & element taxonomy

Defined in `Dataset/constants.py`. The MoE router uses these for task-aware gating. Must stay in sync with `MoEProjection(num_tasks, num_elements)`.

**5 Task types:**
| ID | Task | Description |
|----|------|-------------|
| 0 | 场景分类 (Scene Classification) | Classify land use/land cover |
| 1 | 视觉问答 (Visual QA) | Answer questions about imagery |
| 2 | 视觉定位 (Visual Grounding) | Localize objects/regions |
| 3 | 描述 (Captioning) | Generate scene descriptions |
| 4 | 要素提取 (Element Extraction) | Extract GeoJSON features/annotations |

**9 Element types:**
| ID | Element |
|----|---------|
| 0 | 无 (None) |
| 1 | 网箱养殖区 (Cage aquaculture) |
| 2 | 筏式养殖区 (Raft aquaculture) |
| 3 | 赤潮 (Red tide) |
| 4 | 浒苔 (Ulva/Green tide) |
| 5 | 海岸线 (Shoreline) |
| 6 | 滩涂 (Tidal flat) |
| 7 | 红树林湿地 (Mangrove wetland) |
| 8 | 土地覆盖 (Land cover) |

## POC development stages (pre-main-pipeline)

Before the 3-stage training pipeline, individual components were validated via **Proof-of-Concept (POC)** scripts in `scripts/`. These are single-NPU training scripts that do NOT use DeepSpeed or `EpochBasedTrainer` — they load a frozen vision encoder, attach a task head, and train end-to-end with manual loops. They use `configs/` (not `Configs/`).

| POC | Script | Config dir | What it validates |
|-----|--------|-----------|-------------------|
| PoC-1 — Detection | `scripts/poc_stage_one_det.py` | `configs/poc_aqua_instance.yaml` | Mask R-CNN aquaculture detection head |
| PoC-2 — Semantic | `scripts/poc_stage_semantic.py` | `configs/poc2_landcover_semantic.yaml` | Landcover semantic segmentation head (CE + Dice) |
| PoC-3 — Edge | `scripts/poc_stage_edge_ds.py` | `configs/poc3_edge_a2_soft_multiscale.yaml` | Coastline edge detection head (DeepSpeed, 8 NPU) |
| PoC-4 — Fusion | `scripts/poc_stage_fusion.py` | `Configs/poc4_fusion.yaml` | LLM fallback + detection head fusion pipeline |

**PoC-3** is under active development in the `worktree-poc3-edge-head` worktree. Its edge head (`Models/edge_head.py`) provides `SingleScaleEdgeHead` (concat FPN P1-P4 at 56×56 → 1-channel logit) and `MultiScaleEdgeHead` (HED-style side outputs). The worktree has its own `Inference.py`, `Models/coastgpt.py`, and `Eval/` directory (`eval_cls.py`, `eval_vg.py`, `eval_vqa.py`).

**PoC-4** uses `Models/fusion_pipeline.py` (`FusionPipeline`) which orchestrates the full inference stack: prefix check → [CAP]/[VQA] early return → LLM parsing → gating → detection heads (known classes) → LLM fallback (unknown classes, with Layer1+Layer2 validation) → dedup (self-dedup + cross-source) → FeatureCollection. Predictors live in `Models/fusion_predictors.py`: `InstancePredictor` (Mask R-CNN), `SemanticPredictor` (landcover), `EdgePredictor` (coastline), `LLMPredictor` (GeoJSON generation), `LLMTextPredictor` (text for CAP/VQA).

## Key architectural decisions

- **NPU-native training only** — code conditionally imports `torch_npu`; there is no CUDA fallback path.
- **Stage 3 train imports stage 2** — `train_stage_three.py` extends `train_stage_two.py` for GeoJSON-specific data building and configuration normalization. Changes to stage 2 trainer affect stage 3.
- **Main model orchestrator** — `Models/coastgpt.py` (`CoastGPT` class) wires together the dual vision encoder, MoE projection, LLaMA language model, and physics decoder. It imports `embedding_model_r1.py` (active), not `embedding_model.py` (legacy). Also imports `vision_model.py` (active), not `vision_model1.py` (alternative).
- **Inference entry point** — `Inference.py` loads a consolidated checkpoint via `CoastGPT`, processes a single image through the full pipeline, and saves GeoJSON output. It forces `stage=0` (eval-style load), disables LoRA (`lora.enable=False` — loads merged weights directly, no PeftModel shell), and applies NPU safety overrides (forces bits=16, handles dtype fallbacks). Supports `--prompt`, `--image`, `--output`, `--model-path`, `--georef` CLI args.
- **MoE routing lives in `Models/common_arch.py`** — The `MoEProjection` class provides two-stage task-aware gating (task gate → element gate), task/element embeddings, physical prompt encoding (`PhysicalPromptEncoder`), and `AttnPooler`. Auxiliary losses: balance, entropy, task-route (KL + effect), element-route (KL + effect). `Models/moe_seg.py` holds expert adapters (BandAttentionAdapter, TextureRefineAdapter, SPPSelectorAdapter), per-modality experts (SpectralExpert, TextureExpert, ShapeExpert, ContextExpert), plus `ModalityGate` and z-loss regularization.
- **Dual vision encoder** — `Models/dual_vision_encoder.py` fuses DINOv3 ViT-L/16 global context with ConvNeXt-Base local patches via `CrossFrequencyAttention`. Requires pretrained `.pth` checkpoints in repo root (paths set in config via `rgb_vision.global_ckpt_path` / `rgb_vision.local_ckpt_path`): `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` (ViT-L/16) and `dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth` (ConvNeXt-Base).
- **Wavelet adapter (DWT fusion)** — `Models/wavelet_adapter.py` provides discrete wavelet transform (DWT) for heterogeneous sensor fusion. Aligns sensors with different GSDs (0.8m–10m) and band counts (SAR 2ch, multispectral 4ch) to a unified 3-channel representation in frequency domain, before the vision encoder. Configurable via `wavelet_adapter` config section.
- **Physics decoder** — `Models/physics_decoder.py` and `Models/physics_constraints.py` provide RTE/SAR/SAM losses with fallback implementations for missing NPU operators.
- **Dynamic Tanh** — `Models/dynamic_tanh.py` provides `DynamicTanh` and `convert_ln_to_dyt()` to replace LayerNorm. `Models/DyT.py` has a simpler standalone DyT implementation.
- **Embedding model variants** — `Models/embedding_model_r1.py` is the active embedding model used by `coastgpt.py`; `Models/embedding_model.py` is the legacy version.
- **Vision model variants** — `Models/vision_model.py` (used by `coastgpt.py`) and `Models/vision_model1.py` (alternative) both export `VisionModel`. `Models/models_vit.py` wraps timm's `VisionTransformer` for checkpoint loading.
- **Hyperspectral encoder** — `Models/hypimage_encoder.py` provides `HypImageEncoder`, a ViT-based encoder with 3D convolutions and spectral attention for hyperspectral imagery (e.g., GF-5 AHSI).
- **Edge detection head** — `Models/edge_head.py` provides `SingleScaleEdgeHead` (concat FPN P1-P4 at 56×56 → 1-channel logit) and `MultiScaleEdgeHead` (HED-style side outputs from each FPN level) for coastline boundary extraction. Part of PoC-3; actively developed in the `worktree-poc3-edge-head` worktree.
- **Coordinate encoding** — Uses normalized [0,1] float coordinates (no location-token vocabulary by default). Quantized location tokens available via `Tools/build_gf2_geojson_dataset.py --quantize-coords`.
- **FPN neck** — `Models/fpn_neck.py` provides a shared Feature Pyramid Network neck used by both task heads for multi-scale feature fusion.
- **Dual task heads** — `Models/semantic_head.py` (landcover segmentation via FPN fusion) and `Models/det_head.py` (aquaculture instance segmentation via Mask R-CNN + FPN) operate on the same vision encoder features.
- **Multi-band data source** — `Dataset/multiband_source.py` provides `MultibandGeojsonSource` that reads 4-band TIF imagery and merges NIR/channel data for the wavelet adapter pipeline.
- **Special tokens** — `Models/__init__.py` defines: `<image>` (image placeholder in text), `[SEG]` (segmentation prompt token, triggers semantic head routing), `<im_start>`/`<im_end>` (image boundary markers).
- **LoRA fine-tuning** — PEFT LoRA applied to LLaMA attention layers. Config in `lora:` config section. `TextLoRA/` contains adapter weights. For inference, LoRA is disabled and merged weights are loaded directly (no PeftModel wrapper). Merge command: `python scripts/merge_lora_checkpoint.py --base <base.pt> --lora <TextLoRA/> --output <merged.pt>`.
- **Trainer hook system** — `Trainer/trainer.py` (base) and `Trainer/EpochBasedTrainer.py` (main) use a plug-in hook architecture (`Trainer/hook/`): optimizer hooks (FP16, gradient accumulation), checkpoint hooks (epoch/iter-based), DeepSpeed hook, logger hook, eval hook, LR scheduler hook, EMA hook, CleanEmbedGrad hook, DINO loss warmup hook, MoCo warmup hook, KNN eval hook, param_flops hook, plot_rec hook. Hooks are composed at trainer init time.
- **Config system** — `Trainer/utils/config_parser.py` provides `ConfigArgumentParser`: YAML base config + CLI overrides via dot-path notation (e.g., `--rgb_vision.freeze True` sets `config.rgb_vision.freeze`). All training scripts use `-c <yaml>` plus `--batch-size`, `--epochs`, etc. overrides. Two config directories: **`Configs/`** (main 3-stage pipeline: `train_dual.yaml`, `step2_dual.yaml`, `step3_dual.yaml`, `inference.yaml`) and **`configs/`** (POC ablation configs: `poc_aqua_instance.yaml`, `poc2_landcover_semantic.yaml`, `poc3_edge_a2_soft_multiscale.yaml`, etc.). Stage 2 experimental configs in `Configs/` follow the naming pattern `step2_dual_bf16_prefix_aux_<experiment>_<lr>_<date>.yaml`.
- **Conversation template system** — `Dataset/conversation.py` defines `Conversation` dataclass with 5 separator styles (SINGLE, TWO, MPT, PLAIN, LLAMA_2). Handles `<image>` token placement, system messages, and role formatting for LLaMA-2 chat template. Used by the dataset to format question-answer pairs.
- **Dataset JSON format** — Each dataset directory contains `*_Image/` (images) + `*.json` manifests. JSON structure: `{"data": [{"name": "<tile_id>", "conv": [{"Question": "<image>...", "Answer": "{\"type\":\"Feature\",...}"}]}]}`. Images are matched by `name` field. The `MixedStage3Data_v2/` merged dataset combines multiple sources (GF geojson, landcover, RSVG, scene classification, etc.) via symlinks for stage 3 training.

## Directory map

```
CoastGPT/
  Models/           # CoastGPT, dual vision encoder, MoE (common_arch.py), physics decoder,
                    #   language model, FPN neck, semantic/det/edge heads, DWT wavelet adapter,
                    #   fusion_pipeline.py (PoC-4 orchestration), fusion_predictors.py
  Trainer/          # EpochBasedTrainer, hook/ (checkpoint, logger, optimizer, eval, DeepSpeed),
                    #   optimizer/, utils/ (ConfigArgumentParser, distribute, sampler)
  Dataset/          # cap_dataset.py (main dataset classes), build_loader.py, build_transform.py,
                    #   conversation.py (templates), constants.py (task/element taxonomy),
                    #   rasterize_geojson.py, landcover_dataset.py, multiband_source.py,
                    #   multiscale_sampler.py, aqua_poc_dataset.py, rsvqa.py, UCM.py, meterml.py
  Configs/          # Main pipeline YAML configs (train_dual.yaml, step2_dual.yaml, step3_dual.yaml,
                    #   inference.yaml) + ~20 step2 experimental configs
  configs/          # POC ablation YAML configs (poc_aqua_instance.yaml, poc2_landcover_semantic.yaml,
                    #   poc3_edge_a2_soft_multiscale.yaml, etc.)
  scripts/          # Shell wrappers + Python scripts: train/infer/eval/overfit/DWT/POC stages
  Tools/            # GeoJSON builders, batch eval, heatmaps, weight inspection, probes,
                    #   model_evaluate/ (metric calculators), data_prepare/ (dataset generation)
  tests/            # 25 standalone test scripts (run directly with python, no pytest needed)
  utils/            # geojson_builder.py, geojson_coordinate_utils.py, georef_transform.py,
                    #   mask_utils.py, semantic_bg_prior.py, semantic_overlay.py, vis_overlay.py,
                    #   geojson_dedup.py, geojson_validator.py
  Transformers/     # Vendored transformers (HF offline)
  Docs/             # Architecture diagrams, data documentation, sample images
  output/           # Checkpoints from each stage
  MixedStage3Data_v2/  # Symlinked merged dataset for stage 3
  TextLoRA/         # LoRA adapter weights for text model
```

## Common commands

**Inference:**
```bash
bash scripts/run_infer.sh <image_path> ["prompt"] [output.json] [sample_id]
```
`scripts/run_infer.sh` wraps `Inference.py`, which loads a consolidated checkpoint via `Models/coastgpt.py` → `CoastGPT` class, processes a single image through the full pipeline (vision encoder → MoE projection → LLM → GeoJSON decode), and saves the output. The consolidated checkpoint must be a single `.pt` file (not DeepSpeed shards).

**Inference (Python directly):**
```bash
python Inference.py --model-path <checkpoint.pt> --image <image_path> --prompt "Describe this scene" --output result.json
```

**Stage 1 training (8 NPU):**
```bash
bash scripts/train_stage_one.sh
```

**Stage 2 training (8 NPU):**
```bash
bash scripts/train_stage_two.sh
```

**Stage 3 training (2 NPU, typical):**
```bash
bash scripts/run_stage3.sh
```

**Stage 3 training (multi-scale, 2 NPU):**
```bash
bash scripts/run_stage3_multiscale.sh
```

**MoE diagnostic (100-step smoke test):**
```bash
bash scripts/run_moe_diag_100step.sh
```

**Direct DeepSpeed launch:**
```bash
deepspeed --num_nodes=1 --num_gpus=2 train_stage_three.py \
  -c Configs/step3_dual.yaml --batch-size 2 --workers 1 \
  --accumulation-steps 8 --epochs 4 --model-path ./FINAL.pt \
  --data-path ./MixedStage3Data_v2 --output ./output/stage3/mixed_v3 \
  --accelerator npu --enable-amp True --use-checkpoint --wandb False
```

**Merge stage 2 + stage 3 data:**
```bash
bash scripts/setup_mixed_data_v2.sh
```

**4-band overfit verification (smoke test for DWT/multiband pipeline):**
```bash
python scripts/overfit_4band_verify.py --in-chans 3 --device npu:1
python scripts/overfit_4band_verify.py --in-chans 4 --device npu:1
```

**DWT multi-sensor fusion verification:**
```bash
python scripts/verify_dwt_fusion.py
```

**Precache landcover targets (run before semantic training):**
```bash
python scripts/precache_landcover_targets.py
```

**Batch GeoJSON evaluation:**
```bash
python Tools/run_geojson_batch_eval.py --config Configs/step3_dual.yaml \
  --model-path <checkpoint.pt> --data-path <dataset_root> --output <results_dir>
```

**Stage 2 batch evaluation:**
```bash
python Tools/run_stage2_batch_eval.py --config Configs/step2_dual.yaml \
  --model-path <checkpoint.pt> --data-path <dataset_root> --output <results_dir>
```

**Semantic segmentation evaluation:**
```bash
python scripts/eval_semantic_full.py --config Configs/step2_dual.yaml \
  --model-path <checkpoint.pt> --data-path <dataset_root> --output <results_dir>
```

**POC evaluation (aquaculture/landcover):**
```bash
python scripts/eval_poc1.py --config Configs/poc_aqua_instance.yaml \
  --model-path <checkpoint.pt> --data-path <dataset_root> --output <results_dir>
```

**Wavelet adapter ablation:**
```bash
python scripts/run_wavelet_adapter_ablation.py --config Configs/step2_dual.yaml \
  --model-path <checkpoint.pt> --data-path <dataset_root> --output <results_dir>
```

**GeoJSON conversion (to Shapefile / ArcGIS):**
```bash
python Tools/geojson_to_shp.py <input.geojson> <output_dir>
python Tools/geojson_to_arcgis.py <input.geojson> <output_dir>
```

**Weight inspection:**
```bash
python Tools/inspect_weights.py <checkpoint.pt>
```

**LoRA merge (for inference):**
```bash
python scripts/merge_lora_checkpoint.py --base <base_model.pt> --lora <TextLoRA/> --output <merged.pt>
```
Merges LoRA adapter weights into the base LLaMA checkpoint. Inference (`Inference.py`) uses `lora.enable=False` to load merged checkpoints directly without the PeftModel wrapper.

## Testing

All tests live in `tests/` as standalone Python scripts (plain `unittest` or raw assert — no pytest). Run them directly on NPU:

```bash
# Single test file
python tests/test_dual_vision_encoder.py

# Core model tests (NPU required)
python tests/test_wavelet_adapter.py
python tests/test_multiband_source.py
python tests/test_semantic_overlay.py
python tests/test_semantic_bg_prior.py

# Stage 2 / LoRA tests
python tests/test_stage2_lora_trainable.py
python tests/test_stage2_teacher_forced_probe.py
python tests/test_stage2_generate_route_alignment.py
python tests/test_stage2_batch_eval.py

# Inference tests
python tests/test_inference_image_loading.py
python tests/test_inference_semantic_routes.py
python tests/test_geojson_multiband_inference.py

# Checkpoint, prefix loss, training config tests
python tests/test_checkpoint_text_weight_restore.py
python tests/test_prefix_first_token_loss.py
python tests/test_deepspeed_trainable_parameters.py
python tests/test_llama2_preprocess_labels.py
python tests/test_no_loc_tokens.py
python tests/test_wavelet_training_config.py

# Fusion pipeline tests
python tests/test_poc4_fusion.py
```

Tests import from the repo root — run them from the repo root or with `PYTHONPATH=.`. Some tests require NPU hardware; CPU-only tests will fail on `import torch_npu`.

## Checkpoint conventions

- Stage 1 output: `output/checkpoints/FINAL.pt`
- Stage 2 checkpoints: `output/stage2/checkpoints/iter_NNNN_consolidated.pt`
- Stage 3 final: `output/stage3/mixed_v3/checkpoints/FINAL.pt`
- Mid-training stage 3: `runs/` directory (diagnostic runs)

Consolidated checkpoints are created from DeepSpeed ZeRO shards via `get_fp32_state_dict_from_zero_checkpoint`.

## Git notes

The repo tracks model code and configs. Large `.pth` files and `kernel_meta/` are gitignored. The default branch is `main`. Active development branch is `CoastGPT_dual`. Collaborator branches: `remotes/origin/cuibinge`, `remotes/origin/CoastGPT-hgh`.

**Active worktrees:**
- `worktree-poc3-edge-head` — PoC-3 coastline edge head development. Has its own `Models/coastgpt.py`, `Models/coastGPT_NPU.py`, `Inference.py`, and `Eval/` directory. This is the next major feature: an edge-aware detection head for coastline boundary extraction.

## Tooling

- **MCP**: `.mcp.json` configures the CodeGraph MCP server for codebase-aware analysis (`/home/ma-user/.local/bin/codegraph serve --mcp`).
- **Claude settings**: `.claude/settings.json` has allowlisted permissions for `weixin_claude_bot/` read/write operations. `.claude/settings.local.json` may hold additional local overrides.
