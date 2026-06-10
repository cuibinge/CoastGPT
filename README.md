# CoastGPT

Multimodal coastal remote sensing foundation model — generates GeoJSON annotations from multi-sensor satellite imagery.

## Architecture

```
Satellite Imagery (GF1/2/6, SAR, Multispectral)
    │
    ▼
┌─────────────────────────────────────┐
│  WaveletFusion (DWT multi-sensor)   │  ← 异构传感器统一（频域对齐 + 光谱融合）
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Dual Vision Encoder                │
│  DINOv3 ViT-L/16 (global)           │  ← 双编码器：全局语义 + 局部细节
│  + ConvNeXt-Base (local)            │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Task-aware MoE Projection          │
│  Task/Element Dual-Driven Gating    │  ← 物理先验引导门控网络选择专家
└─────────────────────────────────────┘
    │
    ├──► LLaMA-2-7B → GeoJSON Text
    │
    └──► PhysicsDecoder (RTE/SAR)      ← 物理解码器
         AquacultureSegMOE (Seg Head)  ← 养殖区分割头
         LandcoverSemanticHead         ← 土地覆盖分割头
```

## Key Features

- **4-band multispectral support** — Blue / Green / Red / NIR from GF-1/2/6 PMS sensors
- **DWT wavelet fusion** — align heterogeneous sensors (SAR, multispectral, hyperspectral) and resolutions (0.8m–10m) in frequency domain before encoder
- **Task-aware MoE routing** — task + element + physical-prompt dual-driven gating selects experts dynamically
- **Physics-constrained decoder** — RTE (Nechad 2010) and SAR σ₀ losses constrain visual features with physical laws
- **End-to-end GeoJSON generation** — outputs ready-to-use GeoJSON with geometry + properties + CRS metadata
- **Huawei Ascend NPU native** — DeepSpeed ZeRO distributed training on 910B2 NPUs

## Training Pipeline

| Stage | Script | Hardware | Key Params | Output |
|-------|--------|----------|------------|--------|
| **1 — Pretrain** | `train_stage_one.py` | 8 NPU | batch=8/GPU | Vision-language alignment |
| **2 — MoE Fine-tune** | `train_stage_two.py` | 8 NPU | batch=4/GPU | Task-aware expert routing |
| **3 — GeoJSON Gen** | `train_stage_three.py` | 2 NPU | batch=2/GPU, accum=8 | End-to-end GeoJSON output |

Stage 3 inherits core trainer logic from stage 2.

## Environment

```bash
# Activate NPU environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate PyTorch-2.1.0

# Offline HF cache
export HF_HOME=./hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# NPU memory config
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

**Requirements:** Python 3.10, torch 2.1.2, DeepSpeed, CANN-8.0.RC2, Ascend 910B2 NPU.  
Full package list in `requirement.txt`.

## Installation

```bash
git clone https://github.com/cuibinge/CoastGPT.git
cd CoastGPT
conda create -n CoastGPT python=3.10
conda activate CoastGPT
pip install -r requirement.txt
```

## Quick Start

### Inference

```bash
bash scripts/run_infer.sh <image_path> ["prompt"] [output.json] [sample_id]
```

### Stage 3 Training (2 NPU)

```bash
bash scripts/run_stage3.sh
```

### MoE Diagnostic (100-step smoke test)

```bash
bash scripts/run_moe_diag_100step.sh
```

### Direct DeepSpeed Launch

```bash
deepspeed --num_nodes=1 --num_gpus=2 train_stage_three.py \
  -c Configs/step3_dual.yaml --batch-size 2 --workers 1 \
  --accumulation-steps 8 --epochs 4 --model-path ./FINAL.pt \
  --data-path ./MixedStage3Data_v2 --output ./output/stage3/mixed_v3 \
  --accelerator npu --enable-amp True --use-checkpoint --wandb False
```

### 4-band Overfit Verification

```bash
# 3ch baseline
python scripts/overfit_4band_verify.py --in-chans 3 --device npu:1

# 4ch (RGB+NIR) comparison
python scripts/overfit_4band_verify.py --in-chans 4 --device npu:1
```

### DWT Multi-Sensor Fusion Verification

```bash
python scripts/verify_dwt_fusion.py
```

## Directory Map

```
CoastGPT/
  Models/              # CoastGPT model, dual vision encoder, MoE, physics decoder, DWT fusion
  Trainer/             # EpochBasedTrainer, DeepSpeed hooks, optimizers, checkpointer
  Dataset/             # Data loaders, transforms, conversation templates
  Configs/             # YAML configs per training stage
  Tools/               # GeoJSON builders, evaluation, heatmaps, weight inspection
  scripts/             # Shell wrappers for train/infer, overfit verification, DWT verification
  utils/               # Georeferencing, coordinate encoding, semantic overlay utilities
  Docs/                # Design specs and data documentation
  output/              # Checkpoints and evaluation outputs
  MixedStage3Data_v2/  # Symlinked merged dataset for stage 3
```

## Datasets

Training data organized under `Stage3Data/`:

| Category | Sensors | Size | Description |
|----------|---------|------|-------------|
| 养殖区 (Aquaculture) | GF1/GF2/GF6 | ~1,800 tiles | 4-band TIF with GeoJSON labels |
| 土地分类 (Land Cover) | GF1 | ~33,000 tiles | 21-class land cover with per-class binary masks |
| 海岸线 (Shoreline) | GF1/GF2 | ~950 tiles | 6 shoreline types with GeoJSON labels |

**Image variants per tile:**
- `Image_Orig/` — 4-band TIF (Blue+Green+Red+NIR), full radiometric precision
- `Image_TrueColor/` — 3-channel RGB PNG
- `Image_FalseColor/` — 3-channel false-color PNG (NIR-R-G)

## Checkpoint Conventions

- Stage 1 output: `output/checkpoints/FINAL.pt`
- Stage 2 checkpoints: `output/stage2/checkpoints/iter_NNNN_consolidated.pt`
- Stage 3 final: `output/stage3/mixed_v3/checkpoints/FINAL.pt`
- Mid-training stage 3: `runs/` directory (diagnostic runs)

Consolidated checkpoints are created from DeepSpeed ZeRO shards via `get_fp32_state_dict_from_zero_checkpoint`.

## Pretrained Weights

Required checkpoint files in repo root:
- `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` — DINOv3 ViT-L/16 global encoder
- `dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth` — DINOv3 ConvNeXt-Base local encoder

## License

Internal research project. Contact the authors for usage.
