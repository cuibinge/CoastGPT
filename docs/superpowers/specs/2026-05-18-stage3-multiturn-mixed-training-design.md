# Stage 3 多轮对话 + 混合训练 设计文档

## 问题

1. **GeoJSON 答案截断**：单张图最多 72 个 feature，归一化后 FeatureCollection 最长 10,846 字符（~3,000 tokens），超出 `max_text_tokens=3953` 后被 tokenizer 截断，模型学到不完整输出
2. **灾难性遗忘**：Stage 3 只用纯 GeoJSON 数据（3,057 样本）训练 8 epochs，Stage 2 的 VQA/Caption/VG 能力（98,683 样本）全部丢失
3. **推理截断**：`max_new_tokens` 默认 256，远小于 GeoJSON 所需 token 数
4. **缺少 CRS**：输出归一化坐标无法直接在 ArcGIS 中转 shp

## 约束

- 坐标精度不可降低（用于 ArcGIS 制图）
- `max_position_embeddings=4096` 不可扩展（NPU 显存限制）
- 不可新增 token（避免 embed_tokens 保存/加载风险）

## 方案：多轮对话 + 数据混合 + 坐标反算

### 1. 训练数据：多轮 GeoJSON

**格式**：单个 FeatureCollection 按 token 预算拆分为多轮，每轮是合法 JSON

```json
{
  "conv": [
    {
      "Question": "<image>\n[DET] Extract the target features from this remote sensing image and output a valid GeoJSON FeatureCollection. Return JSON only.",
      "Answer": "{\"type\":\"FeatureCollection\",\"features\":[feat_1,...,feat_10]} <CONTINUE>"
    },
    {
      "Question": "Continue generating the remaining features.",
      "Answer": "{\"type\":\"FeatureCollection\",\"features\":[feat_11,...,feat_20]} <CONTINUE>"
    },
    {
      "Question": "Continue.",
      "Answer": "{\"type\":\"FeatureCollection\",\"features\":[feat_21,...,feat_28]}"
    }
  ]
}
```

规则：
- 每轮答案 ≤ `MAX_ANSWER_TOKENS_PER_TURN`（默认 900），安全在 3953 内
- 首轮包含图像 + 完整任务描述（`[DET] Extract...`）
- 中间轮次用 `Continue.` 作为问题
- 非最终轮答案末尾追加 ` <CONTINUE>` 标记（注意前面有一个空格）
- 最终轮答案正常结束，无标记
- 单轮就能装下的样本（≤ 10 个 feature）不做拆分，无 `<CONTINUE>`

### 2. 坐标与坐标系

**训练时**：
- 坐标归一化到 [0,1]（已有逻辑，`normalize_coords=True`）
- 每张图计算 `geo_transform`（归一化→真实经纬度的仿射参数），保存到 `coord_transform_train.json`

**推理时**：
- 模型输出归一化坐标
- 读取对应图像的 `geo_transform`，反算为 EPSG:4326 经纬度
- 在 FeatureCollection 开头插入 CRS 声明：
  ```json
  {
    "type": "FeatureCollection",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},
    "features": [...]
  }
  ```

### 3. 数据混合

**方案**：文件系统级别合并

```
mkdir MixedStage3Data
# symlink Stage 2 数据
ln -s /home/ma-user/work/Stage2Data/*.json MixedStage3Data/
ln -s /home/ma-user/work/Stage2Data/*_Image MixedStage3Data/
# symlink Stage 3 GeoJSON 数据（多轮重建后）
ln -s /path/to/rebuild_stage3_data/*.json MixedStage3Data/
ln -s /path/to/rebuild_stage3_data/*_Image MixedStage3Data/
```

统计：
- Stage 2：98,683 样本（VQA/Caption/VG/分类）
- Stage 3：~3,000 样本（GeoJSON 多轮拆分后，含多轮膨胀会让实际对话轮数更多）

比例约 97:3，Stage 3 占比偏小。通过 `weight_sample=True` + 调整 `WEIGHT_DICT` 可提高 GeoJSON 采样权重。

### 4. 推理：多轮生成

```
输入: 图像 + "提取目标要素"
  ↓
第1轮: model.generate(max_new_tokens=512)
  → {"features":[...10个...]} <CONTINUE>
  ↓ 检测 <CONTINUE>
第2轮: 拼接对话历史 + "Continue." + model.generate()
  → {"features":[...10个...]} <CONTINUE>
  ↓ 检测 <CONTINUE>
  ...
第N轮: → {"features":[...8个...]}  ← 无 <CONTINUE>，结束
  ↓
拼接所有 features → 完整 FeatureCollection
  ↓
读取 geo_transform → 反算真实经纬度
  ↓
插入 CRS → 输出最终 GeoJSON
```

停止条件：
- 答案不包含 `<CONTINUE>`
- 达到最大轮次上限（`MAX_TURNS`=20）
- EOS token

### 5. 改动清单

| 文件 | 改动 |
|---|---|
| `Tools/build_gf2_geojson_dataset.py` | 新增 `--max-answer-tokens-per-turn`；多轮拆分逻辑；输出 `coord_transform_train.json` |
| `Dataset/cap_dataset.py` | 无需改动（已有 list-of-dict 多轮格式支持） |
| `Inference.py` | `max_new_tokens` 默认 256→512；新增多轮生成循环 + `<CONTINUE>` 检测 + 坐标反算 + CRS 插入 |
| `train_stage_three.sh` | `--data-path` 指向 MixedStage3Data；可选 `--weight-sample` |
| `Configs/step3_dual.yaml` | 可选：`weight_sample: true` + MoE 权重调整 |
| `Tools/merge_stage_data.sh` | 新增脚本，symlink Stage 2 + 3 到合并目录 |

### 6. `<CONTINUE>` 标记规范

- 标记字符串：` <CONTINUE>`（前面有一个空格，避免污染 JSON 内容）
- 位置：答案字符串末尾，在 JSON 闭合 `}` 之后
- 检测方式：`answer_text.rstrip().endswith("<CONTINUE>")`
- 仅 Stage 3 GeoJSON 任务使用，Stage 2 数据不受影响

### 7. geo_transform 格式

```json
{
  "image_name": {
    "x_min": 119.4,
    "y_max": 34.9,
    "pixel_width": 0.00027,
    "pixel_height": 0.00027,
    "image_size": [128, 128],
    "crs": "EPSG:4326"
  }
}
```

反算公式：
```
lon = x_min + norm_x * pixel_width * image_width
lat = y_max - norm_y * pixel_height * image_height
```
