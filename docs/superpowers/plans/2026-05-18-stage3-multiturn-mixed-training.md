# 阶段三 多轮对话 + 混合训练 实施计划

> **面向 agentic worker：** 必须子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 来逐任务实施此计划。步骤使用 checkbox（`- [ ]`）语法跟踪。

**目标：** 通过多轮对话拆分 GeoJSON 答案（CONTINUE 标记衔接）、混合阶段二数据、增加推理时多轮生成和坐标反算+CRS 输出，解决阶段三训练截断和灾难性遗忘两个问题。

**架构：** 修改 build_gf2_geojson_dataset.py，当 FeatureCollection 超出每轮 token 预算时，输出多轮对话列表而非单轮样本。非最后一轮末尾追加 ` <CONTINUE>`。推理时 Inference.py 循环：生成 → 检测 CONTINUE → 自动追加 "Continue." → 再次生成 → 合并 features → 坐标反算 → 插入 CRS。

**技术栈：** Python、PyTorch、DeepSpeed，基于现有 CoastGPT 代码（无新依赖）

---

### 任务 1：多轮训练数据构建

**涉及文件：**
- 修改：`Tools/build_gf2_geojson_dataset.py`

**背景：** 目前 `split_feature_collection_by_chars` 拆分 FeatureCollection 后，每个子集合变成独立样本（单轮）。需要将它们合并为多轮对话的单个样本。

- [ ] **步骤 1：添加 --multiturn 和 --max-chars-per-turn 参数**

在 `parse_args()` 函数中，`return parser.parse_args()` 之前添加：

```python
parser.add_argument(
    "--multiturn",
    action="store_true",
    help="启用多轮对话模式，非最后一轮末尾追加 <CONTINUE> 标记",
)
parser.add_argument(
    "--max-chars-per-turn",
    type=int,
    default=3000,
    help="多轮模式下每轮答案最大字符数",
)
```

- [ ] **步骤 2：添加 CONTINUE_MARKER 常量和多轮构建函数**

在 `DEFAULT_PROMPTS` 元组之后（约第 48 行）添加：

```python
CONTINUE_MARKER = " <CONTINUE>"
```

在 `build_conversations()` 之后、`link_or_copy()` 之前插入：

```python
def build_multiturn_conversations(
    sub_collections,
    target_name,
    first_prompt,
    compact_answer=True,
):
    """将拆分的子 FeatureCollection 列表构建为多轮对话。

    第一轮：完整任务 prompt + 第一组 features
    后续轮次：简短延续 prompt + 对应 features
    非最后一轮答案末尾追加 CONTINUE_MARKER。
    """
    convs = []
    for i, sub in enumerate(sub_collections):
        answer_text = dumps_feature_collection(sub, compact=compact_answer)
        if i == 0:
            question = first_prompt
        elif i == 1:
            question = "Continue generating the remaining features."
        else:
            question = "Continue."

        is_last = i == len(sub_collections) - 1
        if not is_last:
            answer_text += CONTINUE_MARKER

        convs.append({"Question": question, "Answer": answer_text})
    return convs
```

- [ ] **步骤 3：build_dataset 签名中添加 multiturn 和 max_chars_per_turn 参数**

在 `build_dataset()` 中（约第 735 行），在 `prompt_variants: int = 1` 之后添加：

```python
    multiturn: bool = True,
    max_chars_per_turn: int = 3000,
```

拆分逻辑处（约第 805 行），使用 max_chars_per_turn 作为多轮预算：

```python
            if split_by_answer_budget and max_answer_chars > 0:
                budget = max_chars_per_turn if multiturn else max_answer_chars
                sub_collections = split_feature_collection_by_chars(
                    feature_collection=feature_collection,
                    max_answer_chars=budget,
                    explode_multi_geometries=True,
                )
```

- [ ] **步骤 4：build_dataset 中添加多轮样本构建逻辑**

在 `build_dataset()` 中，将 `for part_idx, sub_collection in enumerate(sub_collections, start=1)` 循环体替换为判断多轮/单轮的逻辑：

- 若 `multiturn=True` 且 `len(sub_collections) > 1`：构建一个多轮对话样本，所有 sub_collection 合并到一个 `conv` 列表中
- 否则：保留原有的每个 sub_collection 独立样本的逻辑

多轮样本的 sample_record 包含 `"conv"`（多轮对话列表）、`"part_count"`、`"tile_transform"` 等字段。

- [ ] **步骤 5：保存 coord_transform_train.json**

在样本构建循环结束后（`out_json = output_dir / "GF_geojson_train.json"` 之前），遍历所有样本，提取有 `tile_transform` 的样本，按 image_name -> tile_transform 写入 `coord_transform_train.json`。

- [ ] **步骤 6：main() 中传递 CLI 参数**

在 `main()` 的 `build_dataset()` 调用中添加：

```python
        multiturn=bool(args.multiturn),
        max_chars_per_turn=int(args.max_chars_per_turn),
```

- [ ] **步骤 7：提交**

```bash
git add Tools/build_gf2_geojson_dataset.py
git commit -m "feat: GeoJSON 数据集构建器增加多轮对话支持"
```

---

### 任务 2：数据混合脚本

**涉及文件：**
- 新建：`Tools/merge_stage_data.sh`

- [ ] **步骤 1：编写合并脚本**

```bash
#!/bin/bash
# 合并阶段二和阶段三数据到统一目录，用于混合训练。
# 用法：bash Tools/merge_stage_data.sh <stage2目录> <stage3目录> <输出目录>

STAGE2_DIR="${1:-../Stage2Data}"
STAGE3_DIR="${2:-./output/stage3_geojson_multiturn}"
OUTPUT_DIR="${3:-./MixedStage3Data}"

mkdir -p "$OUTPUT_DIR"

echo "链接阶段二数据: $STAGE2_DIR ..."
for f in "$STAGE2_DIR"/*.json; do
    [ -e "$f" ] || continue
    dst="$OUTPUT_DIR/$(basename "$f")"
    [ -e "$dst" ] || ln -s "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$dst"
done
for d in "$STAGE2_DIR"/*_Image; do
    [ -e "$d" ] || continue
    dst="$OUTPUT_DIR/$(basename "$d")"
    [ -e "$dst" ] || ln -s "$(cd "$(dirname "$d")" && pwd)/$(basename "$d")" "$dst"
done

echo "链接阶段三多轮数据: $STAGE3_DIR ..."
for f in "$STAGE3_DIR"/*.json; do
    [ -e "$f" ] || continue
    dst="$OUTPUT_DIR/$(basename "$f")"
    [ -e "$dst" ] || ln -s "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$dst"
done
for d in "$STAGE3_DIR"/*_Image; do
    [ -e "$d" ] || continue
    dst="$OUTPUT_DIR/$(basename "$d")"
    [ -e "$dst" ] || ln -s "$(cd "$(dirname "$d")" && pwd)/$(basename "$d")" "$dst"
done

echo "合并完成，输出目录: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR/"
echo "训练时使用 --data-path $OUTPUT_DIR"
```

- [ ] **步骤 2：提交**

```bash
git add Tools/merge_stage_data.sh
git commit -m "feat: 添加阶段二 + 阶段三数据合并脚本"
```

---

### 任务 3：更新训练脚本

**涉及文件：**
- 修改：`train_stage_three.sh`

- [ ] **步骤 1：增加多轮数据构建 + 混合数据路径**

```bash
MODEL_PATH=./FINAL.pt
OUTPUT_PATH="./output/stage3/multiturn_mixed"
RAW_DATA_ROOT="../GeoJsonData"
GEOJSON_OUTPUT_ROOT="./output/stage3_geojson_multiturn"
CONFIG_PATH=./Configs/step3_dual.yaml
SCRIPT_PATH=./train_stage_three.py
MERGED_DATA_PATH="./MixedStage3Data"

# 步骤 0：构建多轮 GeoJSON 数据
python Tools/build_gf2_geojson_dataset.py \
    --gf2-root $RAW_DATA_ROOT \
    --output-dir $GEOJSON_OUTPUT_ROOT \
    --sizes 128 \
    --image-subdir Image_TrueColor \
    --compact-answer \
    --normalize-coords \
    --multiturn \
    --max-chars-per-turn 3000 \
    --prompt-variants 1

# 步骤 1：合并阶段二和阶段三数据
bash Tools/merge_stage_data.sh ../Stage2Data $GEOJSON_OUTPUT_ROOT $MERGED_DATA_PATH

# 步骤 2：混合训练
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0

deepspeed \
    --num_nodes=1 \
    --num_gpus=8 \
    $SCRIPT_PATH \
    -c \
    $CONFIG_PATH \
    --batch-size 4 \
    --workers 2 \
    --model-path $MODEL_PATH \
    --data-path $MERGED_DATA_PATH \
    --raw-data-root $RAW_DATA_ROOT \
    --auto-build-geojson-data False \
    --geojson-output-root $GEOJSON_OUTPUT_ROOT \
    --geojson-priority True \
    --output $OUTPUT_PATH \
    --accelerator "npu" \
    --enable-amp True \
    --use-checkpoint \
    --weight-sample
```

- [ ] **步骤 2：提交**

```bash
git add train_stage_three.sh
git commit -m "feat: 阶段三训练脚本增加多轮数据构建和混合训练"
```

---

### 任务 4：多轮推理 + CONTINUE 检测 + 坐标反算 + CRS

**涉及文件：**
- 修改：`Inference.py`

- [ ] **步骤 1：修改 max_new_tokens 默认值 256 → 512**

在 `_build_generation_kwargs()` 中（第 239 行）：

```python
max_new_tokens = int(getattr(config, "max_new_tokens", 512))
```

- [ ] **步骤 2：添加辅助函数**

在 `_postprocess_geojson` 函数之后添加：

```python
CONTINUE_MARKER = " <CONTINUE>"
MAX_MULTITURN_ROUNDS = 20


def _has_continue_marker(text):
    """检测文本末尾是否有 CONTINUE 标记"""
    return text.rstrip().endswith("<CONTINUE>")


def _strip_continue_marker(text):
    """移除文本末尾的 CONTINUE 标记"""
    if text.rstrip().endswith("<CONTINUE>"):
        return text.rstrip()[:-len("<CONTINUE>")].rstrip()
    return text


def _merge_feature_collections(json_strings):
    """合并多个 FeatureCollection JSON 字符串。

    将所有 features 数组合并到一个 FeatureCollection 中。
    保留第一个 EffectiveCollection 中的 CRS（如有）。
    """
    import json as _json
    all_features = []
    crs = None
    for js in json_strings:
        try:
            obj = _json.loads(js)
        except _json.JSONDecodeError:
            continue
        feats = obj.get("features", [])
        if isinstance(feats, list):
            all_features.extend(feats)
        if crs is None and "crs" in obj:
            crs = obj["crs"]
    merged = {"type": "FeatureCollection", "features": all_features}
    if crs is not None:
        merged["crs"] = crs
    return _json.dumps(merged, ensure_ascii=False, separators=(",", ":"))


def _inverse_transform_coordinates(geojson_str, tile_transform):
    """将归一化 [0,1] 坐标反算为 EPSG:4326 真实经纬度。

    tile_transform 字段说明：
        x_min, y_max: 瓦片左上角真实坐标
        pixel_width, pixel_height: 每个像素的地理分辨率
        image_size: [width, height] 像素尺寸
    反算后插入 CRS 声明。
    """
    import json as _json
    x_min = float(tile_transform["x_min"])
    y_max = float(tile_transform["y_max"])
    pixel_w = float(tile_transform["pixel_width"])
    pixel_h = float(tile_transform["pixel_height"])
    img_w = int(tile_transform["image_size"][0])
    img_h = int(tile_transform["image_size"][1])
    geo_w = pixel_w * img_w
    geo_h = pixel_h * img_h

    obj = _json.loads(geojson_str)

    def _transform_ring(ring):
        return [[x_min + pt[0] * geo_w, y_max - pt[1] * geo_h] for pt in ring]

    def _transform_geometry(geom):
        if geom["type"] == "Polygon":
            geom["coordinates"] = [
                _transform_ring(ring) for ring in geom["coordinates"]
            ]
        elif geom["type"] == "MultiPolygon":
            geom["coordinates"] = [
                [_transform_ring(ring) for ring in polygon]
                for polygon in geom["coordinates"]
            ]

    for feature in obj.get("features", []):
        geometry = feature.get("geometry")
        if isinstance(geometry, dict):
            _transform_geometry(geometry)

    # 插入 CRS 声明，ArcGIS 可直接识别并转为 shp
    obj["crs"] = {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:EPSG::4326"},
    }
    return _json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _load_coord_transform(transform_path, image_key):
    """从 coord_transform_train.json 加载指定图像的坐标变换参数"""
    import json as _json
    from pathlib import Path
    path = Path(transform_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        all_transforms = _json.load(f)
    return all_transforms.get(image_key)
```

- [ ] **步骤 3：在 main() 中添加多轮生成循环**

在 `main()` 中，将原来的单次 `model.generate()` 替换为多轮循环：

```python
        # --- 多轮 GeoJSON 生成 ---
        all_geojson_parts = []
        multiturn_round = 0

        while multiturn_round < MAX_MULTITURN_ROUNDS:
            multiturn_round += 1

            # 每次重新 tokenize（prompt 随轮次增长）
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria(
                [stop_str], tokenizer, input_ids
            )

            gen_kwargs = _build_generation_kwargs(config, tokenizer, stopping_criteria)
            if streamer is not None:
                gen_kwargs["streamer"] = streamer

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    images=image_tensor,
                    **gen_kwargs,
                )

            outputs, raw_outputs, new_tokens = _decode_new_tokens(
                tokenizer, output_ids, int(input_ids.shape[1]), stop_str
            )
            all_unk, unk_ratio = _calc_unk_stats(new_tokens, tokenizer)

            if new_tokens.numel() > 0:
                first_ids = new_tokens[:min(8, len(new_tokens))].tolist()
                first_strs = [tokenizer.decode([tid]) for tid in first_ids]
                print(
                    f"[Inference][gen 第{multiturn_round}轮] "
                    f"new_tokens={new_tokens.shape[0]}, first={first_strs}"
                )

            # 后处理 GeoJSON
            is_geojson = any(
                marker in outputs
                for marker in (
                    '"Feature"', '"Polygon"', '"coordinates"', '"FeatureCollection"'
                )
            )
            if is_geojson:
                outputs = _postprocess_geojson(outputs)

            conv.messages[-1][-1] = outputs
            if streamer is None:
                print(outputs)

            # 检测 CONTINUE 标记
            if _has_continue_marker(outputs):
                clean_output = _strip_continue_marker(outputs)
                all_geojson_parts.append(clean_output)
                continue_prompt = (
                    "Continue."
                    if multiturn_round > 1
                    else "Continue generating the remaining features."
                )
                print(f"\n[Inference] 检测到 CONTINUE 标记，自动进入第{multiturn_round + 1}轮...")
                conv.append_message(conv.roles[0], continue_prompt)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
            else:
                # 最后一轮，无 CONTINUE 标记
                all_geojson_parts.append(outputs)
                break

        # --- 合并 features + 坐标反算 + CRS ---
        if len(all_geojson_parts) > 1:
            merged_geojson = _merge_feature_collections(all_geojson_parts)
            print(f"\n[Inference] 合并 {len(all_geojson_parts)} 轮结果 -> {len(merged_geojson)} 字符")

            coord_transform_path = getattr(config, "coord_transform_path", None)
            if coord_transform_path and config.image_file:
                from pathlib import Path
                img_name = Path(config.image_file).name
                tile_transform = _load_coord_transform(
                    coord_transform_path, img_name
                )
            else:
                tile_transform = None

            if tile_transform is not None:
                merged_geojson = _inverse_transform_coordinates(
                    merged_geojson, tile_transform
                )
                print("[Inference] 已应用坐标反算 + CRS（EPSG:4326）")
            else:
                print("[Inference] 未找到 tile_transform，坐标保持归一化，无 CRS")

            conv.messages[-1][-1] = merged_geojson
            if streamer is None:
                print(merged_geojson[:500])
```

- [ ] **步骤 4：添加 --max-new-tokens 和 --coord-transform-path CLI 参数**

在 `parse_option()` 中，`return parser.parse_args(wandb=True)` 之前添加：

```python
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="每轮生成最大 token 数")
    parser.add_argument("--coord-transform-path", type=str, default=None,
                        help="coord_transform_train.json 路径，用于坐标反算")
```

- [ ] **步骤 5：提交**

```bash
git add Inference.py
git commit -m "feat: 推理增加多轮生成、CONTINUE检测、Feature合并和CRS输出"
```

---

### 任务 5：配置文件更新（加权采样）

**涉及文件：**
- 修改：`Configs/step3_dual.yaml`

- [ ] **步骤 1：启用加权采样**

修改：
```yaml
weight_sample: True
```

防止 GeoJSON 样本（~3,000）被阶段二样本（~98,000）淹没。

- [ ] **步骤 2：提交**

```bash
git add Configs/step3_dual.yaml


























































































































































































































































































































































































































































































































git commit -m "feat: 阶段三混合训练启用加权采样"
```

---

### 任务 6：验证

- [ ] **步骤 1：用多轮模式重建 GeoJSON 数据**

```bash
cd /home/ma-user/work/CoastGPT
python Tools/build_gf2_geojson_dataset.py \
    --gf2-root ../GeoJsonData \
    --output-dir ./output/stage3_geojson_multiturn \
    --sizes 128 \
    --image-subdir Image_TrueColor \
    --compact-answer \
    --normalize-coords \
    --multiturn \
    --max-chars-per-turn 3000 \
    --prompt-variants 1
```

- [ ] **步骤 2：验证数据质量**

检查多轮样本存在、CONTINUE 标记仅在非最后一轮出现、coord_transform_train.json 存在且条目正确。

- [ ] **步骤 3：合并并验证混合数据**

```bash
bash Tools/merge_stage_data.sh ../Stage2Data ./output/stage3_geojson_multiturn ./MixedStage3Data
ls ./MixedStage3Data/*.json | wc -l
```

- [ ] **步骤 4：训练 dry-run**

在 train_stage_three.sh 中临时添加 `--max-debug-iters 3`，运行验证数据加载正常无报错。
