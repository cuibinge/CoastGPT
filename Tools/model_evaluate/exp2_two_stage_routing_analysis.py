"""
实验2：两段式路由细粒度分析脚本

目标：
1) 统计“视觉定位”与“描述”两类任务下，task-expert 组和 element-expert 组的权重占比；
2) 对同一张图在两条指令下提取被激活专家的注意力图并可视化。
"""

import argparse
import copy
import importlib.machinery
import json
import math
import os
import random
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ml_collections
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def _mock_optional_modules() -> None:
    """Allow script execution on envs without training-only deps."""
    if "torch_npu" not in sys.modules:
        m = types.ModuleType("torch_npu")
        m.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
        sys.modules["torch_npu"] = m

    try:
        import deepspeed  # noqa: F401
        return
    except Exception:
        pass

    def _ensure(name: str) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        sys.modules[name] = mod
        return mod

    ds = _ensure("deepspeed")
    _ensure("deepspeed.ops")
    ops_adam = _ensure("deepspeed.ops.adam")
    _ensure("deepspeed.runtime")
    zero_mod = _ensure("deepspeed.runtime.zero")
    part_mod = _ensure("deepspeed.runtime.zero.partition_parameters")
    ds_utils = _ensure("deepspeed.utils")
    zero_to_fp32 = _ensure("deepspeed.utils.zero_to_fp32")

    class _DummyCtx:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _ZeroStatus:
        NOT_AVAILABLE = 0

    class _DummyOpt:
        def __init__(self, *args, **kwargs):
            pass

    zero_mod.GatheredParameters = _DummyCtx
    part_mod.ZeroParamStatus = _ZeroStatus
    ds.zero = zero_mod
    ops_adam.FusedAdam = _DummyOpt
    ops_adam.DeepSpeedCPUAdam = _DummyOpt

    def _dummy_get_fp32(*args, **kwargs):
        return {}

    def _dummy_load_zero(model, *args, **kwargs):
        return model

    zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint = _dummy_get_fp32
    zero_to_fp32.load_state_dict_from_zero_checkpoint = _dummy_load_zero
    ds_utils.zero_to_fp32 = zero_to_fp32


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_mock_optional_modules()

from Dataset.build_transform import build_vlp_transform
from Dataset.cap_dataset import DataCollatorForSupervisedDataset, InstructDatasetWithTaskId
from Models.coastgpt import CoastGPT


TYPE_DICT = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args():
    parser = argparse.ArgumentParser("Exp2 two-stage routing analysis")
    parser.add_argument("--config", type=str, default="Configs/train_dual.yaml")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="output/exp2_two_stage")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples-per-task", type=int, default=200)
    parser.add_argument("--seed", type=int, default=322)
    parser.add_argument("--loc-task-name", type=str, default="视觉定位")
    parser.add_argument("--desc-task-name", type=str, default="描述")
    parser.add_argument("--loc-instruction", type=str, default="检测所有船只")
    parser.add_argument("--desc-instruction", type=str, default="描述海岸线形态")
    parser.add_argument("--loc-element-text", type=str, default="船只")
    parser.add_argument("--desc-element-text", type=str, default="海岸线")
    parser.add_argument("--vis-index", type=int, default=-1)
    parser.add_argument("--vis-image-path", type=str, default=None)
    return parser.parse_args()


def _load_config(path: str, data_path_override: Optional[str] = None) -> ml_collections.ConfigDict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if data_path_override:
        cfg["data_path"] = data_path_override
    return ml_collections.ConfigDict(cfg)


def _recursive_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device=device)
    if isinstance(x, dict):
        return {k: _recursive_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_recursive_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_recursive_to_device(v, device) for v in x)
    return x


def _embed_text_fields(model: CoastGPT, data: Dict, dtype: torch.dtype) -> Dict:
    out = dict(data)
    emb_layer = model.language.model.get_input_embeddings()

    if out.get("physical_prompt_ids", None) is not None:
        p = emb_layer(out["physical_prompt_ids"])
        p_mask = out.get("physical_prompt_attention_mask", None)
        if p_mask is not None:
            p = p * p_mask.unsqueeze(-1).to(p.dtype)
        out["physical_prompt_embs"] = p.to(dtype=dtype)
    else:
        out["physical_prompt_embs"] = None

    if out.get("task_text_ids", None) is not None:
        t = emb_layer(out["task_text_ids"])
        t_mask = out.get("task_text_attention_mask", None)
        if t_mask is not None:
            t = t * t_mask.unsqueeze(-1).to(t.dtype)
        out["task_text_embs"] = t.to(dtype=dtype)
    else:
        out["task_text_embs"] = None

    if out.get("element_text_ids", None) is not None:
        e = emb_layer(out["element_text_ids"])
        e_mask = out.get("element_text_attention_mask", None)
        if e_mask is not None:
            e = e * e_mask.unsqueeze(-1).to(e.dtype)
        out["element_text_embs"] = e.to(dtype=dtype)
    else:
        out["element_text_embs"] = None

    return out


def _forward_routing_only(model: CoastGPT, batch: Dict, device: torch.device, dtype: torch.dtype):
    batch = _recursive_to_device(batch, device)
    batch = _embed_text_fields(model, batch, dtype=dtype)

    if hasattr(model.vision, "encode_with_spatial"):
        image_seq, _, _ = model.vision.encode_with_spatial(batch["rgb"], physical_prompt_embs=batch.get("physical_prompt_embs", None))
    else:
        image_seq = model.vision(batch)
    _ = model.multimodal(batch, image_embedding=image_seq)

    if hasattr(model.multimodal, "get_last_routing"):
        routing = model.multimodal.get_last_routing()
    else:
        routing = model.multimodal.projection.get_last_routing()
    return routing, image_seq.shape[1]


def _compute_group_mass(top_indices: torch.Tensor, top_weights: torch.Tensor, task_expert_count: int):
    task_mask = (top_indices < task_expert_count).to(top_weights.dtype)
    task_mass = (top_weights * task_mask).sum(dim=-1)
    element_mass = (top_weights * (1.0 - task_mask)).sum(dim=-1)
    task_pick_ratio = task_mask.float().mean(dim=-1)
    return task_mass, element_mass, task_pick_ratio


def _collect_indices_by_task(dataset: InstructDatasetWithTaskId, task_name: str) -> List[int]:
    results = []
    for i, t in enumerate(dataset.task_texts):
        if t == task_name or task_name in t:
            results.append(i)
    return results


def _reshape_token_map(attn_1d: torch.Tensor, token_len: int) -> torch.Tensor:
    if token_len > 0 and attn_1d.numel() >= token_len:
        attn_1d = attn_1d[:token_len]
    n = int(attn_1d.numel())
    if n <= 0:
        return torch.zeros(1, 1)
    s = int(round(math.sqrt(n)))
    if s * s != n:
        s = int(math.ceil(math.sqrt(n)))
        padded = torch.zeros(s * s, dtype=attn_1d.dtype)
        padded[:n] = attn_1d
        attn_1d = padded
    return attn_1d.view(s, s)


def _run_task_stats(
    model: CoastGPT,
    loader: DataLoader,
    max_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    sample_count = 0
    task_mass_vals = []
    element_mass_vals = []
    task_pick_vals = []

    projection = model.multimodal.projection
    for batch in tqdm(loader, desc="Routing stats", leave=False):
        with torch.no_grad():
            routing, _ = _forward_routing_only(model, batch, device, dtype)

        top_indices = routing["top_indices"]
        top_weights = routing["top_weights"]
        task_count = int(routing["task_expert_count"])

        task_mass, element_mass, task_pick_ratio = _compute_group_mass(top_indices, top_weights, task_count)
        task_mass_vals.extend(task_mass.tolist())
        element_mass_vals.extend(element_mass.tolist())
        task_pick_vals.extend(task_pick_ratio.tolist())

        sample_count += top_indices.shape[0]
        if sample_count >= max_samples:
            break

    if len(task_mass_vals) == 0:
        return {
            "num_samples": 0,
            "task_group_weight_mean": 0.0,
            "element_group_weight_mean": 0.0,
            "task_group_pick_ratio_mean": 0.0,
            "task_expert_count": int(projection.task_expert_count),
            "element_expert_count": int(projection.element_expert_count),
        }

    return {
        "num_samples": int(len(task_mass_vals)),
        "task_group_weight_mean": float(np.mean(task_mass_vals)),
        "element_group_weight_mean": float(np.mean(element_mass_vals)),
        "task_group_pick_ratio_mean": float(np.mean(task_pick_vals)),
        "task_expert_count": int(projection.task_expert_count),
        "element_expert_count": int(projection.element_expert_count),
    }


def _select_visual_index(
    dataset: InstructDatasetWithTaskId,
    loc_indices: List[int],
    vis_index: int,
    vis_image_path: Optional[str],
) -> int:
    if vis_index >= 0 and vis_index < len(dataset):
        return vis_index
    if vis_image_path:
        p = Path(vis_image_path).resolve()
        for i, ip in enumerate(dataset.img_list):
            if Path(ip).resolve() == p:
                return i
    if len(loc_indices) > 0:
        return loc_indices[0]
    return 0


def _run_single_prompt_attention(
    model: CoastGPT,
    collator: DataCollatorForSupervisedDataset,
    sample: Dict,
    task_text: str,
    element_text: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict:
    inst = copy.deepcopy(sample)
    inst["task_text"] = task_text
    inst["element_text"] = element_text
    batch = collator([inst])

    with torch.no_grad():
        routing, visual_token_len = _forward_routing_only(model, batch, device, dtype)

    top_indices = routing["top_indices"][0].tolist()
    top_weights = routing["top_weights"][0].tolist()
    task_expert_count = int(routing["task_expert_count"])

    expert_maps = []
    for eid, weight in zip(top_indices, top_weights):
        attn = getattr(model.multimodal.projection.experts[eid], "_last_spatial_attn", None)
        if attn is None:
            attn_2d = torch.zeros(1, 1)
        else:
            attn_2d = _reshape_token_map(attn.float().cpu().flatten(), token_len=visual_token_len)
        expert_maps.append(
            {
                "expert_id": int(eid),
                "weight": float(weight),
                "group": "task" if eid < task_expert_count else "element",
                "attn_2d": attn_2d,
            }
        )

    return {
        "task_text": task_text,
        "element_text": element_text,
        "top_experts": expert_maps,
    }


def _save_bar_chart(summary: Dict, output_path: Path):
    import matplotlib.pyplot as plt

    labels = [summary["loc_task_name"], summary["desc_task_name"]]
    task_vals = [
        summary["loc_stats"]["task_group_weight_mean"],
        summary["desc_stats"]["task_group_weight_mean"],
    ]
    elem_vals = [
        summary["loc_stats"]["element_group_weight_mean"],
        summary["desc_stats"]["element_group_weight_mean"],
    ]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, task_vals, width=width, label="Task Expert Group")
    plt.bar(x + width / 2, elem_vals, width=width, label="Element Expert Group")
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Average Top-2 Weight Mass")
    plt.title("Two-stage Routing Group Weights")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _to_uint8_image(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def _save_attention_figure(
    image_np: np.ndarray,
    vis_loc: Dict,
    vis_desc: Dict,
    out_path: Path,
):
    import matplotlib.pyplot as plt

    rows = [vis_loc, vis_desc]
    max_k = max(len(vis_loc["top_experts"]), len(vis_desc["top_experts"]))
    cols = max_k + 1
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 7))
    if cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for r, item in enumerate(rows):
        axes[r, 0].imshow(image_np)
        axes[r, 0].set_title(f"Prompt: {item['task_text']} / {item['element_text']}")
        axes[r, 0].axis("off")

        for c in range(1, cols):
            axes[r, c].axis("off")
            if c - 1 >= len(item["top_experts"]):
                continue
            ex = item["top_experts"][c - 1]
            heat = ex["attn_2d"]
            heat = heat - heat.min()
            heat = heat / (heat.max() + 1e-6)
            heat = torch.nn.functional.interpolate(
                heat[None, None],
                size=(image_np.shape[0], image_np.shape[1]),
                mode="bilinear",
                align_corners=False,
            )[0, 0].cpu().numpy()
            axes[r, c].imshow(image_np)
            axes[r, c].imshow(heat, cmap="jet", alpha=0.42)
            axes[r, c].set_title(
                f"Expert {ex['expert_id']} ({ex['group']})\nweight={ex['weight']:.3f}"
            )
            axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = _load_config(args.config, data_path_override=args.data_path)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = TYPE_DICT.get(str(getattr(config, "dtype", "float16")), torch.float16)
    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32

    print(f"[Info] Build model on {device}, dtype={dtype}")
    model = CoastGPT(config)
    if args.model_path:
        print(f"[Info] Loading checkpoint: {args.model_path}")
        model.custom_load_state_dict(args.model_path)
    model = model.to(device=device)
    model = model.eval()

    proj = model.multimodal.projection
    if str(getattr(proj, "routing_strategy", "")).lower() not in ("two_stage", "task_then_element"):
        print("[Warn] moe_proj.routing_strategy is not two_stage/task_then_element. Current setting may not match Exp2.")

    transform = build_vlp_transform(config, is_train=False, num_channels=3)
    dataset = InstructDatasetWithTaskId(
        root=config.data_path,
        tokenizer=model.language.tokenizer,
        transform=transform,
        crop_size=int(config.transform.input_size[0]),
    )
    collator = DataCollatorForSupervisedDataset(
        tokenizer=model.language.tokenizer,
        physical_prompt_max_len=int(getattr(config, "physical_prompt_max_len", 64)),
        task_text_max_len=int(getattr(config, "task_text_max_len", 16)),
        element_text_max_len=int(getattr(config, "element_text_max_len", 16)),
    )

    loc_indices = _collect_indices_by_task(dataset, args.loc_task_name)
    desc_indices = _collect_indices_by_task(dataset, args.desc_task_name)
    if len(loc_indices) == 0 or len(desc_indices) == 0:
        raise RuntimeError(
            f"Task subset empty. loc={len(loc_indices)} desc={len(desc_indices)}. "
            f"Check task names in dataset.task_texts."
        )

    loc_loader = DataLoader(
        Subset(dataset, loc_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collator,
    )
    desc_loader = DataLoader(
        Subset(dataset, desc_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collator,
    )

    print(f"[Info] loc samples={len(loc_indices)}, desc samples={len(desc_indices)}")
    loc_stats = _run_task_stats(model, loc_loader, args.max_samples_per_task, device, dtype)
    desc_stats = _run_task_stats(model, desc_loader, args.max_samples_per_task, device, dtype)

    vis_idx = _select_visual_index(dataset, loc_indices, args.vis_index, args.vis_image_path)
    vis_sample = dataset[vis_idx]
    vis_image_path = Path(dataset.img_list[vis_idx])
    image_np = _to_uint8_image(Image.open(vis_image_path))

    vis_loc = _run_single_prompt_attention(
        model=model,
        collator=collator,
        sample=vis_sample,
        task_text=args.loc_instruction,
        element_text=args.loc_element_text,
        device=device,
        dtype=dtype,
    )
    vis_desc = _run_single_prompt_attention(
        model=model,
        collator=collator,
        sample=vis_sample,
        task_text=args.desc_instruction,
        element_text=args.desc_element_text,
        device=device,
        dtype=dtype,
    )

    summary = {
        "loc_task_name": args.loc_task_name,
        "desc_task_name": args.desc_task_name,
        "loc_instruction": args.loc_instruction,
        "desc_instruction": args.desc_instruction,
        "loc_stats": loc_stats,
        "desc_stats": desc_stats,
        "routing_strategy": str(proj.routing_strategy),
        "num_experts": int(proj.num_experts),
        "task_expert_count": int(proj.task_expert_count),
        "element_expert_count": int(proj.element_expert_count),
        "vis_index": int(vis_idx),
        "vis_image_path": str(vis_image_path),
        "vis_loc_top_experts": [
            {
                "expert_id": e["expert_id"],
                "weight": e["weight"],
                "group": e["group"],
            }
            for e in vis_loc["top_experts"]
        ],
        "vis_desc_top_experts": [
            {
                "expert_id": e["expert_id"],
                "weight": e["weight"],
                "group": e["group"],
            }
            for e in vis_desc["top_experts"]
        ],
    }

    bar_path = output_dir / "exp2_group_weight_bar.png"
    heatmap_path = output_dir / "exp2_attention_heatmaps.png"
    json_path = output_dir / "exp2_summary.json"

    _save_bar_chart(summary, bar_path)
    _save_attention_figure(image_np=image_np, vis_loc=vis_loc, vis_desc=vis_desc, out_path=heatmap_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Done] summary: {json_path}")
    print(f"[Done] bar chart: {bar_path}")
    print(f"[Done] attention figure: {heatmap_path}")
    print(
        "[Result] "
        f"loc(task/elem)={loc_stats['task_group_weight_mean']:.4f}/{loc_stats['element_group_weight_mean']:.4f}, "
        f"desc(task/elem)={desc_stats['task_group_weight_mean']:.4f}/{desc_stats['element_group_weight_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
