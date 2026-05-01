import os
import argparse
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms

from ml_collections import ConfigDict

from Models.dual_vision_encoder import DualVisionEncoder
from Models.moe_seg import AquacultureSegMOE, dice_loss
import torch_npu


class SegFolderDataset(Dataset):
    """简单遥感分割数据集读取器
    结构约定：
      root/
        images/xxx.png|jpg
        masks/xxx.png  # 像素类别ID（灰度或调色板），忽略标签为 ignore_index
    支持可选的文件列表 txt：一行一个文件名（不含扩展名），用于子集采样。
    """

    def __init__(self, root: str, filelist: Optional[str] = None, image_suffix: str = None, mask_suffix: str = None, transform=None):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.msk_dir = self.root / "masks"
        self.transform = transform
        self.items = []
        names = None
        if filelist and Path(filelist).exists():
            with open(filelist, "r", encoding="utf-8") as f:
                names = [line.strip() for line in f if line.strip()]
        if names is None:
            # 按文件名匹配（取 images 下文件名去扩展名）
            names = [p.stem for p in sorted(self.img_dir.glob("*")) if p.is_file()]
        for n in names:
            img_path = self.img_dir / (n + (image_suffix or ""))
            if not img_path.exists():
                # 尝试常见后缀
                for suf in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                    p = self.img_dir / (n + suf)
                    if p.exists():
                        img_path = p
                        break
            msk_path = self.msk_dir / (n + (mask_suffix or ""))
            if not msk_path.exists():
                for suf in [".png", ".tif", ".jpg"]:
                    p = self.msk_dir / (n + suf)
                    if p.exists():
                        msk_path = p
                        break
            if img_path.exists() and msk_path.exists():
                self.items.append((img_path, msk_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, msk_path = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        # 以灰度模式读取掩膜，避免调色板/彩色误读
        mask = Image.open(msk_path).convert("L")
        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            # 容错：若仍为 3 通道，取第一个通道
            mask_np = mask_np[..., 0]
        # 安全映射：若仅包含 {0,255}，将 255 视作前景类 1
        uniq = np.unique(mask_np)
        if set(uniq.tolist()) == {0, 255}:
            mask_np = (mask_np == 255).astype(np.int64)
        else:
            # 保证为整型类别ID
            if mask_np.dtype != np.int64 and mask_np.dtype != np.int32 and mask_np.dtype != np.uint8:
                mask_np = mask_np.astype(np.int64)
        mask_t = torch.from_numpy(mask_np).long()
        # 图像转 Tensor，范围 [0,1]
        img_t = transforms.ToTensor()(img)
        if self.transform is not None:
            img_t = self.transform(img_t)
        return img_t, mask_t


def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255):
    return F.cross_entropy(logits.float(), target.long(), ignore_index=ignore_index)


def compute_seg_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ce_w: float, dice_w: float, ignore_index: int = 255):
    ce = cross_entropy_loss(logits, target, ignore_index=ignore_index)
    dl = dice_loss(logits, target, num_classes=num_classes, ignore_index=ignore_index)
    # 支持加权组合
    loss = ce_w * ce + dice_w * dl
    return loss, {"ce": ce.detach(), "dice": dl.detach()}


@torch.no_grad()
def evaluate_miou(model: nn.Module, head: nn.Module, loader: Iterable, ignore_index: int, num_classes: int, device: torch.device, input_size):
    model.eval()
    head.eval()
    correct = torch.zeros(num_classes, dtype=torch.float64, device=device)
    total = torch.zeros(num_classes, dtype=torch.float64, device=device)
    total_loss = 0.0
    total_batches = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # DualVisionEncoder 处理输入尺寸
        images_rs = F.interpolate(images, size=input_size, mode="bilinear", align_corners=False)
        _, fused_spatial, _ = model.encode_with_spatial(images_rs)
        logits = head(fused_spatial, input_size=targets.shape[-2:])["logits"]
        # 简单验证损失（与训练一致便于对齐）
        loss, _ = compute_seg_loss(logits, targets, num_classes=num_classes, ce_w=1.0, dice_w=1.0, ignore_index=ignore_index)
        total_loss += float(loss.detach().cpu())
        total_batches += 1
        preds = logits.argmax(dim=1)
        mask = (targets != ignore_index)
        for cls in range(num_classes):
            cls_mask = (targets == cls) & mask
            correct[cls] += (preds[cls_mask] == cls).sum()
            total[cls] += cls_mask.sum()
    iou = torch.where(total > 0, correct / total, torch.zeros_like(total))
    miou = float(iou.mean().cpu())
    avg_loss = total_loss / max(1, total_batches)
    return {"miou": miou, "avg_loss": avg_loss, "iou_per_class": iou.detach().cpu().tolist()}


def is_main_process() -> bool:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return int(os.environ.get("RANK", "0")) == 0


def load_train_config(yaml_path: str) -> ConfigDict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return ConfigDict(cfg)


def build_dual_vision(cfg: ConfigDict) -> DualVisionEncoder:
    # 仅提取视觉相关配置传入 DualVisionEncoder
    vision_cfg = ConfigDict()
    vision_cfg.alignment_dim = int(cfg.get("alignment_dim", 768))
    vision_cfg.rgb_vision = ConfigDict(cfg.get("rgb_vision", {}))
    return DualVisionEncoder(vision_cfg)


def _get_vit_patch_size(dual: DualVisionEncoder) -> int:
    try:
        ps = getattr(dual.global_encoder, "patch_embed", None)
        if ps is not None and hasattr(ps, "patch_size"):
            p = ps.patch_size
            return p[0] if isinstance(p, (tuple, list)) else int(p)
    except Exception:
        pass
    # 合理默认值
    return 16


def _adjust_size_to_patch(size: tuple, patch: int) -> tuple:
    h, w = int(size[0]), int(size[1])
    ah = max(patch, (h // patch) * patch)
    aw = max(patch, (w // patch) * patch)
    return (ah, aw)


def load_pretrained_vision_from_coastgpt(dual: DualVisionEncoder, ckpt_path: str):
    """从 CoastGPT 预训练权重中加载视觉编码器到 DualVisionEncoder。
    支持以下常见结构：
    - 顶层含 'module'，其中键以 'vision.' 前缀命名；
    - 顶层或 'model' 内含 'vision_ckpt' 或 'rgb_ckpt'；
    - 直接是视觉编码器的 state_dict。
    加载使用 strict=False 以兼容部分键不匹配。
    """
    try:
        obj = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"[Warn] Failed to load checkpoint: {ckpt_path}, error: {e}")
        return False

    state_dict = None
    if isinstance(obj, dict):
        # nested under 'module'
        if isinstance(obj.get("module"), dict):
            md = obj["module"]
            vis_prefix = "vision."
            if any(k.startswith(vis_prefix) for k in md.keys()):
                state_dict = {k[len(vis_prefix):]: v for k, v in md.items() if k.startswith(vis_prefix)}
        # nested under 'model'
        if state_dict is None and isinstance(obj.get("model"), dict):
            inner = obj["model"]
            if isinstance(inner.get("vision"), dict):
                state_dict = inner["vision"]
            elif isinstance(inner.get("vision_ckpt"), dict):
                state_dict = inner["vision_ckpt"]
            elif isinstance(inner.get("rgb_ckpt"), dict):
                state_dict = inner["rgb_ckpt"]
        # top-level keys
        if state_dict is None:
            if isinstance(obj.get("vision_ckpt"), dict):
                state_dict = obj["vision_ckpt"]
            elif isinstance(obj.get("rgb_ckpt"), dict):
                state_dict = obj["rgb_ckpt"]
            elif any(isinstance(v, torch.Tensor) for v in obj.values()):
                # looks like a plain state dict
                state_dict = obj
    else:
        state_dict = obj

    if state_dict is None:
        print(f"[Warn] No vision state_dict found in checkpoint: {ckpt_path}")
        return False

    msg = dual.load_state_dict(state_dict, strict=False)
    print(f"[Vision Load] from '{ckpt_path}' -> missing={msg.missing_keys}, unexpected={msg.unexpected_keys}")
    return True


def maybe_load_seg_head_from_ckpt(head: nn.Module, ckpt_path: str) -> bool:
    """尝试从 checkpoint 加载分割头权重（如果存在）。
    兼容几种可能的命名：'seg_head', 'moe_seg_head', 'aquaculture_seg_head' 或
    顶层 'module' 前缀 'seg_head.' / 'moe_seg.' / 'aquaculture_seg.'；
    同时兼容嵌套于 'model' / 'state_dict' 的完整模型权重（按前缀提取）。
    """
    try:
        obj = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"[Warn] Failed to open head checkpoint: {ckpt_path}, error: {e}")
        return False

    sd = None
    if isinstance(obj, dict):
        # direct named blocks
        for k in ["seg_head", "moe_seg_head", "aquaculture_seg_head", "head", "seg"]:
            if isinstance(obj.get(k), dict):
                sd = obj[k]
                break
        # module-prefixed (top-level)
        if sd is None and isinstance(obj.get("module"), dict):
            md = obj["module"]
            prefixes = ["seg_head.", "moe_seg.", "aquaculture_seg."]
            for p in prefixes:
                if any(key.startswith(p) for key in md.keys()):
                    sd = {key[len(p):]: md[key] for key in md.keys() if key.startswith(p)}
                    break
        # nested under common containers like 'model' or 'state_dict'
        if sd is None:
            for nest in ["model", "state_dict"]:
                if isinstance(obj.get(nest), dict):
                    md = obj[nest]
                    # Case A: md is itself a plain state_dict
                    if any(isinstance(v, torch.Tensor) for v in md.values()):
                        prefixes = ["seg_head.", "moe_seg.", "aquaculture_seg.", "head.", "seg."]
                        chosen = None
                        for p in prefixes:
                            if any(k.startswith(p) for k in md.keys()):
                                chosen = p
                                break
                        if chosen is not None:
                            sd = {k[len(chosen):]: v for k, v in md.items() if k.startswith(chosen)}
                        else:
                            # fallback: maybe it's already just the head state_dict
                            sd = md
                    else:
                        # Case B: nested named blocks inside md
                        for k in ["seg_head", "moe_seg_head", "aquaculture_seg_head", "head", "seg", "moe_seg"]:
                            if isinstance(md.get(k), dict):
                                sd = md[k]
                                break
        # plain state dict fallback
        if sd is None and any(isinstance(v, torch.Tensor) for v in obj.values()):
            sd = obj
    else:
        sd = obj

    if sd is None:
        return False
    msg = head.load_state_dict(sd, strict=False)
    print(f"[SegHead Load] from '{ckpt_path}' -> missing={msg.missing_keys}, unexpected={msg.unexpected_keys}")
    return True


def main():
    parser = argparse.ArgumentParser("Train AquacultureSeg MOE with DualVisionEncoder")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[2] / "Configs" / "train.yaml"))
    parser.add_argument("--train_root", type=str, default="../RSBuilding-main/data_dir/whubuilding_png/train", help="训练数据根目录，包含 images/ 与 masks/")
    parser.add_argument("--val_root", type=str, default="../RSBuilding-main/data_dir/whubuilding_png/val", help="验证数据根目录，包含 images/ 与 masks/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=str, default="checkpoints/seg_moe_dual")
    parser.add_argument("--coastgpt_ckpt", type=str, default="./FINAL.pt", help="CoastGPT 预训练权重路径（用于加载视觉编码器/可能的分割头）")
    parser.add_argument("--head_ckpt", type=str, default=None, help="分割头权重路径（如已有预训练分割头）")
    args = parser.parse_args()

    # 统一保存目录为绝对路径，并打印关键信息便于定位
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Config] cwd={os.getcwd()} out_dir={out_dir} train_root={os.path.abspath(args.train_root)} val_root={(os.path.abspath(args.val_root) if args.val_root else 'None')} coastgpt_ckpt={args.coastgpt_ckpt or 'None'}")

    # 校验训练/验证数据路径是否存在
    if not os.path.isdir(args.train_root):
        raise FileNotFoundError(f"train_root '{args.train_root}' 不存在。请通过 --train_root 指向包含 images/ 与 masks/ 的目录。")
    if args.val_root and not os.path.isdir(args.val_root):
        print(f"[Warn] val_root '{args.val_root}' 不存在，将跳过验证。")
        args.val_root = None
    cfg = load_train_config(args.config)
    device = torch.device("npu" if hasattr(torch, "npu") and torch.npu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 构造视觉编码器
    dual = build_dual_vision(cfg).to(device)
    # 如提供 CoastGPT 权重，先行加载视觉编码器的预训练权重
    if args.coastgpt_ckpt:
        load_pretrained_vision_from_coastgpt(dual, args.coastgpt_ckpt)
    # 分割头配置
    aq_cfg = ConfigDict(cfg.get("aquaculture_seg", {}))
    num_classes = int(aq_cfg.get("num_classes", 2))
    ignore_index = int(aq_cfg.get("ignore_index", 255))
    ce_w = float(aq_cfg.get("ce_weight", 1.0))
    dice_w = float(aq_cfg.get("dice_weight", 1.0))
    include_ctx = bool(aq_cfg.get("include_context_expert", True))
    adapters = aq_cfg.get("adapters", {})
    router_coef = aq_cfg.get("router_aux_loss_coef", {"entropy": 1e-4, "zloss": 1e-4})
    entropy_w = float(router_coef.get("entropy", 1e-4))
    zloss_w = float(router_coef.get("zloss", 1e-4))

    # 构造分割头（输入通道为 DualVisionEncoder.embedding_dim）
    in_ch = int(getattr(dual, "embedding_dim", int(cfg.get("alignment_dim", 768))))
    # 通用专家配置（关键修复：启用并传递参数）
    use_generic_experts = bool(aq_cfg.get("use_generic_experts", True))
    num_experts = int(aq_cfg.get("num_experts", 16))
    top_k = int(aq_cfg.get("top_k", 2))
    head = AquacultureSegMOE(
        in_channels=in_ch,
        num_classes=num_classes,
        include_context=include_ctx,
        adapter_cfg=adapters,
        use_generic_experts=use_generic_experts,
        num_experts=num_experts,
        top_k=top_k,
    ).to(device)
    # 如提供分割头权重，尝试部分加载
    if args.head_ckpt:
        maybe_load_seg_head_from_ckpt(head, args.head_ckpt)

    # 数据集与 DataLoader
    input_size = tuple(cfg.get("rgb_vision", {}).get("input_size", [224, 224]))
    # 校验并自适配到 ViT patch 的整数倍，避免 DINOv3 全局编码器无法生成网格特征
    patch = _get_vit_patch_size(dual)
    adj_size = _adjust_size_to_patch(input_size, patch)
    if adj_size != input_size:
        print(f"[InputSize] Adjusted from {input_size} to {adj_size} for ViT patch={patch}")
    input_size = adj_size
    # 视觉归一化（关键修复：添加 ImageNet 标准归一化）
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_ds = SegFolderDataset(root=args.train_root, transform=norm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = None
    if args.val_root:
        val_ds = SegFolderDataset(root=args.val_root, transform=norm)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

    # 优化器
    lr = float(cfg.get("lr", 2e-4))
    wd = float(cfg.get("wd", 0.0))
    optim = torch.optim.AdamW(list(head.parameters()), lr=lr, weight_decay=wd)
    # 如果你希望训练视觉编码器的某些部分，可将其参数也加入优化器（当前默认冻结 DualVisionEncoder）

    # 冻结与评估模式（关键修复：避免未加入优化器却处于训练态）
    for p in dual.parameters():
        p.requires_grad = False
    dual.eval()

    best_miou = -1.0
    for epoch in range(1, args.epochs + 1):
        head.train()
        running = {"loss": 0.0, "ce": 0.0, "dice": 0.0, "aux": 0.0}
        batches = 0
        for step, (images, targets) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # 在首个 batch 打印掩膜值分布，便于数据检查（仅主进程打印）
            if epoch == 1 and step == 1 and is_main_process():
                vals, counts = torch.unique(targets, return_counts=True)
                try:
                    vals_l = vals.detach().cpu().tolist()
                    counts_l = counts.detach().cpu().tolist()
                except Exception:
                    vals_l = vals.tolist()
                    counts_l = counts.tolist()
                total_pixels = int(targets.numel())
                ignore_pixels = int((targets == ignore_index).sum().item())
                print(f"[Mask Distribution] unique_values={vals_l} counts={counts_l} total_pixels={total_pixels} ignore_index={ignore_index} ignore_pixels={ignore_pixels}")

            optim.zero_grad(set_to_none=True)
            # DualVisionEncoder 输入按其配置尺寸缩放
            images_rs = F.interpolate(images, size=input_size, mode="bilinear", align_corners=False)
            with torch.no_grad():
                _, fused_spatial, _ = dual.encode_with_spatial(images_rs)
            out = head(fused_spatial, input_size=targets.shape[-2:])
            logits = out["logits"]
            seg_loss, parts = compute_seg_loss(logits, targets, num_classes=num_classes, ce_w=ce_w, dice_w=dice_w, ignore_index=ignore_index)
            aux_loss = 0.0
            if "router_aux" in out:
                aux = out["router_aux"]
                aux_loss = entropy_w * aux["entropy"] + zloss_w * aux["zloss"]
            total_loss = seg_loss + aux_loss
            if not torch.isfinite(total_loss):
                continue
            total_loss.backward()
            max_grad_norm = float(cfg.get("max_grad_norm", 0.0))
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=max_grad_norm)
            optim.step()

            running["loss"] += float(total_loss.detach().cpu())
            running["ce"] += float(parts["ce"].cpu())
            running["dice"] += float(parts["dice"].cpu())
            running["aux"] += float(aux_loss if isinstance(aux_loss, float) else aux_loss.detach().cpu())
            batches += 1
            if step % 10 == 0 and is_main_process():
                avg = {k: v / max(1, batches) for k, v in running.items()}
                print(f"[Epoch {epoch} Step {step}] loss={avg['loss']:.4f} ce={avg['ce']:.4f} dice={avg['dice']:.4f} aux={avg['aux']:.6f}")

        # 验证与保存
        if val_loader and is_main_process():
            metrics = evaluate_miou(dual, head, val_loader, ignore_index=ignore_index, num_classes=num_classes, device=device, input_size=input_size)
            print(f"[Epoch {epoch}] val_mIoU={metrics['miou']:.4f} val_loss={metrics['avg_loss']:.4f}")
            if metrics["miou"] > best_miou:
                best_miou = metrics["miou"]
                torch.save(head.state_dict(), os.path.join(out_dir, "best_miou_head.pt"))

        if is_main_process():
            torch.save(head.state_dict(), os.path.join(out_dir, f"seg_head_epoch{epoch}.pt"))

    if is_main_process():
        torch.save(head.state_dict(), os.path.join(out_dir, "seg_head_final.pt"))


if __name__ == "__main__":
    main()