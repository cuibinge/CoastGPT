# CoastGPT

CoastGPT is a multimodal remote-sensing foundation model for image-grounded
language generation and structured output. The production architecture keeps
model components domain-neutral: task labels, feature categories, and output
schemas are represented as data or prompts, not as dedicated model branches.

## Architecture

```text
Remote-sensing imagery
    |
    v
Generic sensor preprocessing
    |
    v
Dual vision encoder
  - global visual stream
  - local visual stream
    |
    v
Generic conditional sparse projection
    |
    v
Language model decoder
    |
    v
Text or structured output
```

## Vector Object Objective

CoastGPT treats vector production as a generic language-grounded object
modeling problem. GeoJSON targets are parsed into domain-neutral geometry
statistics and supervised through the same language decoder. The auxiliary
objective increases the loss weight on generic GeoJSON structure and coordinate
tokens, without adding line-specific, polygon-specific, task-specific, or
feature-specific decoder branches.

The training batch exposes optional monitoring fields:

```text
vector_feature_count
vector_point_count
vector_bbox_area
vector_closure_error
```

These fields are used for logging and supervision weighting only. They do not
create category routers or specialized model heads.

## Interactive Tasks

CoastGPT keeps remote-sensing captioning, scene classification, visual
grounding, visual question answering, instruction following, and vector-object
extraction in one natural-language interaction loop. The model receives an
image plus a user instruction, then follows the requested response format.

The interactive request layer only normalizes output contracts such as text,
JSON, GeoJSON, WKT, or export-ready vector products. Feature names, categories,
and product scopes remain user-provided free text and training data, not
hard-coded model branches.

Vector-object extraction is open vocabulary. A user may request any feature
scope supported by the image evidence and training distribution. The model must
return only objects matching that request, keep non-requested regions
unassigned, and avoid forcing ambiguous or background regions into a fixed
known category.

Example inference controls:

```bash
python Inference.py \
  --image-file sample.tif \
  --model-path CheckPoints/model.pt \
  --accelerator gpu \
  --default-output-format auto
```

For product-style extraction demos, use:

```bash
python Inference.py \
  --image-file sample.tif \
  --model-path CheckPoints/model.pt \
  --accelerator gpu \
  --default-output-format geojson \
  --json-only true
```

## CoastGPT-Bench Preparation

CoastGPT-Bench can be converted into the generic instruction format used by
the training loader. The conversion keeps the usable subsets:

- visual grounding coordinates as generic GeoJSON LineString targets
- georeferenced image footprints as generic FeatureCollection targets
- visual question answering conversations
- image caption conversations
- scene classification samples when class folders are available

Prepare the dataset:

```bash
python Dataset/coastgpt_bench_builder.py \
  --source-root data/raw/CoastGPT-Bench \
  --output-root data/prepared/CoastBench \
  --download
```

For vector-only training data:

```bash
python Dataset/coastgpt_bench_builder.py \
  --source-root data/raw/CoastGPT-Bench \
  --output-root data/prepared/CoastBenchVector \
  --download \
  --no-vqa \
  --no-caption \
  --no-geojson \
  --no-classification
```

Launch Stage-3 training with the prepared dataset:

```bash
torchrun --nproc_per_node=2 train_stage_three.py \
  -c Configs/step3_dual.yaml \
  --accelerator gpu \
  --data-path data/prepared/CoastBench \
  --auto-build-geojson-data false \
  --batch-size 1 \
  --accumulation-steps 8 \
  --workers 4 \
  --output output/coast_bench_stage3
```

## Design Rules

- Model code must stay independent of specific feature categories.
- Model code must stay independent of specific task types.
- Vector extraction must follow the user-provided open vocabulary scope.
- Non-requested or uncertain regions must not be forced into a known category.
- Routing and projection modules may use visual context and generic condition
  tokens only.
- Specialized data preparation, evaluation, or export logic must live outside
  the core model path.
- Runtime behavior must be selected through the shared runtime abstraction,
  not by hard-coded CUDA or NPU branches.

## Hardware

The codebase supports both NVIDIA CUDA systems and Huawei Ascend systems.

Recommended NVIDIA 4090D path:

```bash
conda create -n coastgpt-cu121 python=3.10
conda activate coastgpt-cu121
pip install -r requirements-nvidia4090d.txt
```

Recommended launch style:

```bash
deepspeed --num_nodes=1 --num_gpus=1 train_stage_two.py \
  -c Configs/step2_dual.yaml \
  --accelerator gpu \
  --batch-size 1 \
  --accumulation-steps 8 \
  --workers 4 \
  --model-path CheckPoints/FINAL_epoch8_loc_off_lr_stuck.pt \
  --output output/stage2_cuda
```

Use `--accelerator auto` to select CUDA when available, then NPU, then CPU.

## Checkpoints

Place model checkpoints under:

```text
CheckPoints/
```

Large external weights should not be committed to Git.

## Repository Map

```text
Configs/      Runtime and training configs
Dataset/      Generic data loading and preprocessing
Models/       Core model components
Trainer/      Training loop, hooks, distributed utilities
utils/        Runtime, georeference, and output utilities
```

## Current Refactor Status

- Runtime device selection is centralized in `utils/runtime.py`.
- The language model no longer imports NPU runtime unconditionally.
- The main multimodal projection path uses a generic conditional MoE module.
- Vector-object supervision is implemented as a generic GeoJSON language
  objective.
- Legacy feature-specific experiment files were removed from the core path.
