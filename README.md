# CoastGPT

CoastGPT is a multimodal remote-sensing project for coastal intelligence tasks, including captioning, VQA, visual grounding, and classification.

## Quick Start

### 1) Environment setup

```bash
git clone git@github.com:cuibinge/CoastGPT.git
cd CoastGPT
conda create -n coastgpt python=3.10 -y
conda activate coastgpt
pip install -r requirement.txt
```

Note: the correct install command is `pip install -r requirement.txt`.

### 2) Optional smoke test

```bash
python Tools/smoke_test_semantic_routing.py
```

If you see `SMOKE_TEST_PASS`, core data-collation and semantic-routing modules are functioning.

### 3) Stage-1 training (alignment)

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

### 4) Stage-2 training (instruction tuning)

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

If you are training on GPU instead of NPU, change `--accelerator npu` to `--accelerator gpu`.

Training checkpoints are saved to:

- `<OUTPUT_DIR>/checkpoints/`
- final file: `<OUTPUT_DIR>/checkpoints/FINAL.pt`

### 5) Quick inference after training

```bash
python Inference.py -c Configs/step2_dual.yaml \
  --model-path <STAGE2_OUTPUT_DIR>/checkpoints/FINAL.pt \
  --image-file Images/test.png \
  --accelerator gpu
```

### 6) Evaluation examples

Classification:

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

VQA:

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

## Full Documentation

For the full "train -> test" workflow, dataset layout, command templates, and troubleshooting, see:

- [Docs/README.md](Docs/README.md)

