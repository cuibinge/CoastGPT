import os
import yaml
import torch
import torch_npu
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from peft import PeftModel
import ml_collections

# 从你的项目代码库中导入
from Models.coastgpt import CoastGPT

# ================= 🛠️ 1. 配置区域 =================
DEVICE = "npu:0"
BASE_MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
CONFIG_PATH = "./Configs/step2.yaml"  # 确保 YAML 中包含 bits: 16 和 tune_rgb_pooler: True

# 权重路径
LORA_PATH = "./checkpoints/stage2_results/epoch_0/lora"
PROJ_PATH = "./checkpoints/stage2_results/epoch_0/proj.bin"

# 测试图片与真实框 (GT Box)
TEST_IMAGE_PATH = "./Images/GF1_WFV1_E113.7_N21.8_20220407_L1A0006395069.jpg"
# 归一化坐标 [x1, y1, x2, y2], 0-1 之间。示例：[0.4, 0.4, 0.6, 0.6] 为中心区域
TEST_GT_BOX = [0.4, 0.4, 0.6, 0.6] 

# 用于 Hook 存储
visual_acts = {}

# ================= 🧠 2. 核心函数定义 =================

def get_activation_hook(name):
    """抓取投影层后的特征 Token"""
    def hook(model, input, output):
        visual_acts[name] = output.detach().float().cpu()
    return hook

def calculate_semantic_drift(heatmap, gt_box, grid_size):
    """计算感知高能区与真实框的 IoU"""
    # 1. 提取 Top 10% 响应区域
    threshold = np.percentile(heatmap, 90)
    binary_map = (heatmap >= threshold).astype(np.uint8)
    
    # 2. 生成 GT 掩码
    gt_map = np.zeros((grid_size, grid_size))
    gx1, gy1 = int(gt_box[0]*grid_size), int(gt_box[1]*grid_size)
    gx2, gy2 = int(gt_box[2]*grid_size), int(gt_box[3]*grid_size)
    gt_map[gy1:gy2, gx1:gx2] = 1
    
    # 3. 计算 IoU
    intersection = np.logical_and(binary_map, gt_map).sum()
    union = np.logical_or(binary_map, gt_map).sum()
    return intersection / (union + 1e-6)

def generate_heatmap(image_path, token_norms, grid_size):
    """生成叠加在原图上的热力图"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape

    heatmap = token_norms.reshape(grid_size, grid_size).numpy()
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    
    heatmap_resized = cv2.resize(heatmap, (W, H))
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(heatmap_color, 0.4, img, 0.6, 0)
    return superimposed_img, heatmap

# ================= 🚀 3. 加载与执行逻辑 =================

def load_model_with_hook():
    print("🚀 初始化 CoastGPT 架构...")
    with open(CONFIG_PATH, 'r') as f:
        config = ml_collections.ConfigDict(yaml.safe_load(f))

    model = CoastGPT(config)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[CLS]', '[VG]']})
    model.language.text_encoder.resize_token_embeddings(len(tokenizer))

    print("📦 挂载 LoRA 与 MoE 投影层...")
    # 使用方案 A 的 LoRA 挂载逻辑
    model.language.text_encoder = PeftModel.from_pretrained(model.language.text_encoder, LORA_PATH)
    
    proj_weights = torch.load(PROJ_PATH, map_location="cpu")
    model.multimodal.load_state_dict(proj_weights, strict=True)
    
    model.to(DEVICE).eval()

    # 注册钩子捕获最终视觉特征
    model.multimodal.projection.register_forward_hook(get_activation_hook("final_visual_tokens"))
    return model, tokenizer, config

def visualize_attention(model, tokenizer, config, image_path, prompt):
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    raw_image = Image.open(image_path).convert("RGB")
    image_tensor = processor(raw_image, return_tensors="pt").pixel_values.to(DEVICE)
    
    full_prompt = f"<image>\nTask:[CLS]{prompt}"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)

    print("✨ 执行前向传播捕获特征...")
    with torch.no_grad():
        # 使用 forward 绕过 generate 的参数校验与 class 限制
        _ = model(
            rgb=image_tensor, 
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )

    feats = visual_acts["final_visual_tokens"]
    token_norms = torch.norm(feats[0], dim=-1, p=2)

    # 动态确定 Grid Size
    num_total = len(token_norms)
    grid_size = int(num_total**0.5)
    print(f"📊 分析结果: 总 Token 数={num_total}, 识别网格={grid_size}x{grid_size}")

    # 分离全局与局部（基于 144 结构，此时 local 往往为 0）
    global_norms = token_norms[:grid_size*grid_size]
    local_norms = token_norms[grid_size*grid_size:]

    # 计算漂移
    heatmap_matrix = global_norms.reshape(grid_size, grid_size)
    iou_score = calculate_semantic_drift(heatmap_matrix.numpy(), TEST_GT_BOX, grid_size)

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(raw_image)
    axes[0].set_title("Original Image")
    
    heatmap_img, _ = generate_heatmap(image_path, global_norms, grid_size)
    axes[1].imshow(heatmap_img)
    axes[1].set_title(f"Attention Heatmap (IoU: {iou_score:.4f})")
    
    # 叠加绘制 GT Box 辅助观察
    W_img, H_img = raw_image.size
    rect = plt.Rectangle((TEST_GT_BOX[0]*W_img, TEST_GT_BOX[1]*H_img), 
                         (TEST_GT_BOX[2]-TEST_GT_BOX[0])*W_img, 
                         (TEST_GT_BOX[3]-TEST_GT_BOX[1])*H_img, 
                         fill=False, edgecolor='white', linestyle='--', linewidth=2)
    axes[1].add_patch(rect)

    avg_global = global_norms.mean().item()
    avg_local = local_norms.mean().item() if len(local_norms) > 0 else 0
    axes[2].bar(['Global', 'Local'], [avg_global, avg_local], color=['#3498db', '#e74c3c'])
    axes[2].set_title("Feature Contribution Intensity")

    plt.savefig("attention_drift_analysis.png")
    print(f"✅ 分析报告已保存至 attention_drift_analysis.png, IoU Score: {iou_score:.4f}")
    plt.show()

if __name__ == "__main__":
    torch.npu.empty_cache()
    model, tokenizer, config = load_model_with_hook()
    prompt = "请详细描述海岸线的曲折程度以及沿岸的微小人工设施。"
    visualize_attention(model, tokenizer, config, TEST_IMAGE_PATH, prompt)
