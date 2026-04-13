# CoastGPT: Train-to-Test Guide

This document provides a full workflow from environment setup to training, inference, and evaluation.

## 1. Scope

Recommended scripts in this repo:

- Training stage 1: `train_stage_one.py`
- Training stage 2: `train_stage_two.py`
- Interactive inference: `Inference.py`
- Classification evaluation: `eval_cls.py`
- VQA evaluation: `eval_vqa.py`
- Benchmark helpers: `Tools/model_evaluate/*`

## 2. Environment Setup

### 2.1 Requirements

- Python 3.10
- Conda
- Git
- Device environment: NPU or GPU

### 2.2 Installation

```bash
git clone git@github.com:cuibinge/CoastGPT.git
cd CoastGPT
conda create -n coastgpt python=3.10 -y
conda activate coastgpt
pip install -r requirement.txt
```

Common mistake:

- Wrong: `pip install requirement`
- Correct: `pip install -r requirement.txt`

### 2.3 Optional health check

```bash
python Tools/smoke_test_semantic_routing.py
```

Expected key output:

- `SMOKE_TEST_PASS`

## 3. Data Layout

### 3.1 Stage-1 and Stage-2 training data root

The dataset loader scans `*_Image` folders and sibling annotation JSON files.

Example:

```text
<DATA_ROOT>/
  RSICD_Image/
    xxx.jpg
  RSICD.json

  RSVG_Image/
    yyy.jpg
  RSVG.json

  LR_Image/
    zzz.tif
  LR.json
```

For instruction data, each item should contain `conv`, for example:

```json
{
  "name": "example.jpg",
  "conv": [
    {
      "Question": "[CAP] Describe the image.",
      "Answer": "..."
    }
  ]
}
```

### 3.2 Classification evaluation data

For `eval_cls.py` (when `eval.dataset` is not UCM/METERML), use ImageFolder format:

```text
<CLS_DATA_ROOT>/
  class_a/
    a1.jpg
    a2.jpg
  class_b/
    b1.jpg
```

### 3.3 VQA evaluation data

`eval_vqa.py` expects:

- `--data-target`: annotation root
- `--data-path`: image directory (absolute path is recommended)
- `--data-type`: `HR` or `LR`

RSVQA annotation naming should follow loader rules, for example:

- `USGS_split_test_questions.json`
- `USGS_split_test_answers.json`
- `USGS_split_test_images.json`

Images are referenced as `<id>.tif`.

## 4. End-to-End Workflow

## 4.1 Stage-1 training (alignment)

```bash
deepspeed --num_nodes=1 --num_gpus=8 train_stage_one.py \
  -c Configs/train_dual.yaml \
  --data-path <STAGE1_DATA_ROOT> \
  --output <STAGE1_OUTPUT_DIR> \
  --batch-size 8 \
  --workers 4 \
  --accelerator npu \
  --enable-amp True \
  --use-checkpoint
```

Checkpoint output:

- `<STAGE1_OUTPUT_DIR>/checkpoints/FINAL.pt`

## 4.2 Stage-2 training (instruction tuning)

```bash
deepspeed --num_nodes=1 --num_gpus=8 train_stage_two.py \
  -c Configs/step2_dual.yaml \
  --data-path <STAGE2_DATA_ROOT> \
  --model-path <STAGE1_OUTPUT_DIR>/checkpoints/FINAL.pt \
  --output <STAGE2_OUTPUT_DIR> \
  --batch-size 4 \
  --workers 2 \
  --accelerator npu \
  --enable-amp True \
  --use-checkpoint
```

Important:

- `train_stage_two.sh` defines `MODEL_PATH` but does not pass it by default.
- Use the explicit command above (or add `--model-path` manually in your shell script).

Stage-2 final checkpoint:

- `<STAGE2_OUTPUT_DIR>/checkpoints/FINAL.pt`

## 4.3 Interactive inference

```bash
python Inference.py -c Configs/step2_dual.yaml \
  --model-path <STAGE2_OUTPUT_DIR>/checkpoints/FINAL.pt \
  --image-file Images/test.png \
  --accelerator gpu
```

You can type questions in terminal after startup.

## 4.4 Classification evaluation

```bash
python eval_cls.py -c Configs/step2_dual.yaml \
  --model-path <STAGE2_OUTPUT_DIR>/checkpoints/FINAL.pt \
  --data-path <CLS_DATA_ROOT> \
  --output output/eval_cls \
  --batch-size 8 \
  --workers 4 \
  --accelerator gpu \
  --enable-amp False \
  --opts eval.dataset AID
```

Main metrics are logged in output logs:

- `classification_report`
- `balanced_accuracy`

## 4.5 VQA evaluation

```bash
python eval_vqa.py -c Configs/step2_dual.yaml \
  --model-path <STAGE2_OUTPUT_DIR>/checkpoints/FINAL.pt \
  --data-target <VQA_ANNOTATION_ROOT> \
  --data-path <VQA_IMAGE_ROOT> \
  --data-type HR \
  --output output/eval_vqa \
  --batch-size 1 \
  --workers 2 \
  --accelerator gpu \
  --enable-amp False
```

Result file:

- `output/eval_vqa/eval_save_file.json`

Optional additional metric export:

```bash
python Tools/model_evaluate/VQAMetricsCalculator.py \
  --result-file output/eval_vqa/eval_save_file.json \
  --output-json output/eval_vqa/metrics.json
```

## 4.6 Visual grounding evaluation (optional)

```bash
python Tools/model_evaluate/main_vg_infer.py \
  -c Configs/inference.yaml \
  --model-path <STAGE2_OUTPUT_DIR>/checkpoints/FINAL.pt \
  --data-path <VG_IMAGE_ROOT> \
  --data-target <VG_JSON_PATH> \
  --output output/eval_vg \
  --batch-size 1 \
  --workers 2 \
  --accelerator gpu \
  --enable-amp False

python Tools/model_evaluate/VGMetricsCalculator.py \
  --json-file output/eval_vg/rsvg_result.json
```

## 5. Command Template (Copy and Fill)

```bash
# 1) stage 1
STAGE1_DATA_ROOT=""
STAGE1_OUTPUT_DIR=""

# 2) stage 2
STAGE2_DATA_ROOT=""
STAGE2_OUTPUT_DIR=""
STAGE1_CKPT="${STAGE1_OUTPUT_DIR}/checkpoints/FINAL.pt"

# 3) eval
CLS_DATA_ROOT=""
VQA_ANNOTATION_ROOT=""
VQA_IMAGE_ROOT=""
VG_IMAGE_ROOT=""
VG_JSON_PATH=""
```

## 6. Troubleshooting

- If install fails, verify you used `pip install -r requirement.txt`.
- If stage-2 training starts from scratch unexpectedly, check `--model-path` is passed.
- If you are on GPU, switch training args from `--accelerator npu` to `--accelerator gpu`.
- If evaluation with AMP has dtype/device warnings, set `--enable-amp False` first.
- If model/processor downloads are blocked, pre-download required HuggingFace weights in your environment.
- If VQA image path is not found, pass an absolute `--data-path`.
- If `Tools/model_evaluate/main_vg_infer.py` fails to load `/root/autodl-tmp/clip-vit-large-patch14`, replace it with `openai/clip-vit-large-patch14` in that script.

## 7. Recommended Progression

- Step 1: run smoke test
- Step 2: run stage-1 short debug (`--max-debug-iters`)
- Step 3: run full stage-1 and stage-2
- Step 4: run inference sanity check on a few images
- Step 5: run full evaluation and archive metrics JSON

