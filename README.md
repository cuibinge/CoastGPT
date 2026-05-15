# CoastGPT

CoastGPT is a multimodal remote-sensing project for coastal intelligence tasks, including captioning, VQA, visual grounding, and classification.

## Model Download

Pretrained models are available on Hugging Face: [cuibinge/CoastGPT](https://huggingface.co/cuibinge/CoastGPT)

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

The legacy segmentation-routing smoke test has been removed. Use the Stage-3 dataset builder and training entrypoint below as the supported path.

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

### 5) Stage-3 training (GF2 -> GeoJSON -> ArcGIS)

Stage-3 now targets direct `GeoJSON FeatureCollection` generation for end-to-end localization and ArcGIS editing. This path removes the segmentation branch and coordinate-bin Geo-Tokens. Use the `[DET]` tag in prompts to request GeoJSON feature extraction.

```bash
deepspeed --num_nodes=1 --num_gpus=8 train_stage_three.py \
  -c Configs/step3_dual.yaml \
  --model-path <STAGE2_OUTPUT_DIR>/checkpoints/FINAL.pt \
  --output <STAGE3_OUTPUT_DIR> \
  --batch-size 4 \
  --workers 2 \
  --raw-data-root GF2 \
  --gf2-sizes 512 \
  --gf2-image-subdir Image_FalseColor \
  --auto-build-geojson-data True \
  --geojson-output-root GF2/stage3_geojson_512_falsecolor \
  --geojson-priority True \
  --accelerator npu \
  --enable-amp True \
  --use-checkpoint
```

If you only want to rebuild the GF2 Stage-3 dataset first:

```bash
python Tools/build_gf2_geojson_dataset.py \
  --gf2-root GF2 \
  --sizes 512 \
  --image-subdir Image_FalseColor \
  --compact-answer
```

### 6) Quick inference after training

```bash
python Inference.py -c Configs/step3_dual.yaml \
  --model-path <STAGE3_OUTPUT_DIR>/checkpoints/FINAL.pt \
  --image-file Images/test.png \
  --accelerator gpu
```

Use a prompt such as:

```text
[DET] Generate an editable GeoJSON FeatureCollection for ArcGIS from this image patch. Return JSON only.
```

If you want to normalize model output into an ArcGIS-ready file:

```bash
python Tools/geojson_to_arcgis.py --input output.txt --output output/result.geojson
```

For Stage-3 batch inference + automatic `predictions.jsonl` export + evaluation:

```bash
python Tools/run_geojson_batch_eval.py -c Configs/step3_dual.yaml \
  --model-path <STAGE3_OUTPUT_DIR>/checkpoints/FINAL.pt \
  --dataset-json <STAGE3_DATA_ROOT>/GF2_geojson_val.json \
  --output-dir output/geojson_eval \
  --accelerator gpu \
  --max-new-tokens 2048
```

This command writes:

- `output/geojson_eval/predictions.jsonl`
- `output/geojson_eval/eval_summary.json`
- `output/geojson_eval/normalized_geojson/` for ArcGIS-ready predictions that parsed successfully

If you already have `predictions.jsonl` and only want to re-run evaluation:

```bash
python Tools/run_geojson_batch_eval.py \
  --dataset-json <STAGE3_DATA_ROOT>/GF2_geojson_val.json \
  --output-dir output/geojson_eval \
  --skip-inference True
```

### 7) Evaluation examples

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

