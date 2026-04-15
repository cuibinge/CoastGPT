import os
# 🌟 显存优化：启用可扩展段
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
import json
import yaml
import torch
import logging
import argparse
from pathlib import Path
import ml_collections
from PIL import Image
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import get_cosine_schedule_with_warmup
import torch.distributed as dist
import torchvision.transforms as T
import deepspeed
import numpy as np

# 导入 CoastGPT 组件
from Models.coastgpt import CoastGPT
from Models import IMAGE_TOKEN_INDEX, IGNORE_INDEX, tokenizer_image_token
import torch_npu

# NPU 精度模式
torch_npu.npu.set_option({"ACL_PRECISION_MODE": "allow_fp32_to_fp16"})

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger = logging.getLogger("train")

# =========================================================
# 1. 数据集类
# =========================================================
class AutoScanDataset(Dataset):
    def __init__(self, data_root, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.data_list = [] 
        data_root = Path(data_root)
        self.meta_index = {}
        # 扫描 JSON
        json_dirs = [data_root / "OSCapAnn", data_root / "OSMCapAnn"]
        for json_dir in json_dirs:
            if not json_dir.exists(): continue
            for jf in sorted(list(json_dir.glob("*.json"))):
                try:
                    with open(jf, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                    items = content.get("data", content)
                    for item in items:
                        name, cap = item.get("name"), item.get("caption", item.get("cap", ""))
                        if name and cap: self.meta_index[name] = cap.strip()
                except: pass

        for img_path in data_root.rglob("*"):
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif"] and img_path.stem in self.meta_index:
                self.data_list.append({"path": str(img_path), "text": self.meta_index[img_path.stem]})
        
        # 使用 336 分辨率以平衡显存和精度
        self.transforms = T.Compose([
            T.Resize((336, 336)), 
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.prompt_template = "<|im_start|>user\n<image>\n请详细描述这张卫星图片的内容。<|im_end|>\n<|im_start|>assistant\n"

    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx):
        item = self.data_list[idx]
        try:
            image = Image.open(item['path']).convert('RGB')
            pixel_values = self.transforms(image)
        except: return self.__getitem__((idx + 1) % len(self))

        full_text = self.prompt_template + item['text'] + "<|im_end|>"
        # 🌟 核心：确保 <image> 映射为 -200
        input_ids = tokenizer_image_token(full_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        labels = input_ids.clone()
        return {"rgb": pixel_values, "input_ids": input_ids, "labels": labels}

def collate_fn(batch):
    if not batch: return None
    rgb = torch.stack([x['rgb'] for x in batch])
    input_ids = [x['input_ids'] for x in batch]
    labels = [x['labels'] for x in batch]
    max_l = max(len(t) for t in input_ids)
    padded_ids = torch.zeros((len(batch), max_l), dtype=torch.long)
    padded_labels = torch.full((len(batch), max_l), IGNORE_INDEX, dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(input_ids, labels)):
        padded_ids[i, :len(ids)] = ids
        padded_labels[i, :len(lab)] = lab
    return {"rgb": rgb, "input_ids": padded_ids, "labels": padded_labels}

# =========================================================
# 2. 训练逻辑
# =========================================================
def train_one_epoch(model_engine, dataloader, scheduler, epoch, is_main_process):
    model_engine.train()
    device = torch.device("npu", dist.get_rank())
    
    for step, batch in enumerate(dataloader):
        if batch is None: continue
        images = batch['rgb'].to(device).to(torch.float16)
        input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)

        # Token 检查
        if step == 0 and is_main_process:
            if IMAGE_TOKEN_INDEX in input_ids[0]:
                logger.info(f"✅ Token 映射检查通过：发现 IMAGE_TOKEN_INDEX ({IMAGE_TOKEN_INDEX})")
            else:
                logger.error("❌ Token 映射失败：未发现图片占位符索引！")

        outputs = model_engine({"rgb": images, "input_ids": input_ids, "labels": labels, "attention_mask": input_ids.ne(0)})
        loss = outputs.get("total_loss", outputs.get("text_loss"))
        
        model_engine.backward(loss)
        
        if step % 50 == 0 and is_main_process:
            gain_p = model_engine.module.multimodal.projection.output_gain
            g_norm = gain_p.grad.norm().item() if gain_p.grad is not None else 0.0
            logger.info(f"Step {step} | Loss: {loss.item():.4f} | Gain Grad Norm: {g_norm:.6f}")

        model_engine.step()
        scheduler.step()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./Configs/train.yaml")
    parser.add_argument("--data-path", dest="data_root", type=str, default="../PretrainData")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", dest="output_dir", type=str, default="./checkpoints")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--accelerator", type=str, default="npu")
    parser.add_argument("--enable-amp", type=str, default="True")
    
    # 🌟 修复点：改用 action="store_true" 处理开关参数
    parser.add_argument("--use-checkpoint", action="store_true", help="启用梯度检查点以节省显存")
    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    deepspeed.init_distributed()
    torch.npu.set_device(args.local_rank)
    is_main_process = (args.local_rank <= 0)

    with open(args.config, 'r') as f:
        config = ml_collections.ConfigDict(yaml.safe_load(f))
    
    model = CoastGPT(config)
    
    # Stage 1: 仅开启多模态层梯度
    for n, p in model.named_parameters(): p.requires_grad = False
    for p in model.multimodal.parameters(): p.requires_grad = True

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": 16,
        "fp16": {"enabled": True, "initial_scale_power": 12},
        "zero_optimization": {"stage": 2}
    }
    
    model_engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, optimizer=optimizer, config=ds_config)
    tokenizer = model.language.tokenizer
    dataset = AutoScanDataset(args.data_root, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=DistributedSampler(dataset, shuffle=True), 
                            num_workers=args.workers, collate_fn=collate_fn)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer.optimizer if hasattr(optimizer, 'optimizer') else optimizer, 
                                                num_warmup_steps=100, num_training_steps=len(dataloader)*config.epochs)

    for epoch in range(config.epochs):
        dataloader.sampler.set_epoch(epoch)
        train_one_epoch(model_engine, dataloader, scheduler, epoch, is_main_process)
        if is_main_process:
            save_p = Path(args.output_dir) / f"multimodal_epoch{epoch+1}.bin"
            torch.save(model_engine.module.multimodal.state_dict(), save_p)

if __name__ == "__main__":
    main()
