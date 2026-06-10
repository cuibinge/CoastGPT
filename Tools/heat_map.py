import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import yaml
import ml_collections
import os

# 导入你的模型组件 (请确保路径与你的实际项目结构匹配)
from Models.coastgpt import CoastGPT

# =====================================================================
# 1. 配置加载函数
# =====================================================================
def load_yaml_config(config_path):
    """读取 YAML 文件并转换为 ml_collections.ConfigDict"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
        
    return ml_collections.ConfigDict(config_dict)

# =====================================================================
# 2. 绘图函数 (将路由权重叠加到原图)
# =====================================================================
def visualize_moe_routing(image_path, routing_weights, num_experts=4, save_path="moe_routing1.png"):
    img = Image.open(image_path).convert('RGB')
    
    # 假设你的视觉模型输入是 224x224 (DINOv3 ViT)
    target_size = (224, 224) 
    img = img.resize(target_size)
    img_np = np.array(img) / 255.0

    # L 是空间 token 数量(例如 256), E 是专家数量
    L, E = routing_weights.shape
    grid_size = int(np.sqrt(L))
    
    # 创建画布 (1 行, E+1 列)
    fig, axes = plt.subplots(1, E + 1, figsize=(4 * (E + 1), 4))
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')

    for i in range(E):
        expert_weight = routing_weights[:, i].numpy()
        attention_matrix = expert_weight.reshape(grid_size, grid_size)
        
        # 上采样到 224x224
        heatmap = cv2.resize(attention_matrix, target_size, interpolation=cv2.INTER_CUBIC)
        
        # 伪彩色映射 (越红代表分配给该专家的概率越高)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = np.float32(heatmap_colored) / 255.0
        heatmap_colored = heatmap_colored[:, :, ::-1] # BGR 转 RGB
        
        # 将热力图叠加到原图上
        cam = heatmap_colored * 0.55 + img_np * 0.45
        cam = np.clip(cam, 0, 1)

        axes[i+1].imshow(cam)
        axes[i+1].set_title(f"Expert {i} Routing", fontsize=14)
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ MoE 路由分布热力图已成功保存至: {save_path}")

# =====================================================================
# 3. 核心推理主函数
# =====================================================================
def main():
    # 设定设备 (根据你的环境，这里可能需要改成 "npu" 如果你在华为昇腾上)
    device = torch.device("npu" if hasattr(torch, 'npu') and torch.npu.is_available() else "cpu")
    print(f"🚀 当前使用设备: {device}")

    # 1. 动态加载你的真实配置文件
    config_path = "./Configs/train.yaml" 
    print(f"📂 正在加载配置文件: {config_path}")
    config = load_yaml_config(config_path)

    # 2. 初始化模型
    print("⏳ 正在构建 CoastGPT 模型结构 (双编码器 DINOv3 架构)...")
    model = CoastGPT(config).to(device)
    model.eval() # 必须开启 eval 模式，触发缓存！

    # 3. 🌟 智能加载与重映射权重
    ckpt_path = "./FINAL.pt" 
    if os.path.exists(ckpt_path):
        print(f"📥 正在读取权重文件: {ckpt_path}")
        ckpt_dict_raw = torch.load(ckpt_path, map_location=device)
        
        if isinstance(ckpt_dict_raw, dict) and 'state_dict' in ckpt_dict_raw:
            ckpt_dict = ckpt_dict_raw['state_dict']
        elif isinstance(ckpt_dict_raw, dict) and 'model' in ckpt_dict_raw:
            ckpt_dict = ckpt_dict_raw['model']
        else:
            ckpt_dict = ckpt_dict_raw

        # =================================================
        # 🔨 新增：递归压平嵌套的字典 (拆开 other_ckpt 和 rgb_ckpt 箱子)
        # =================================================
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
            
        flat_ckpt_dict = flatten_dict(ckpt_dict)

        # =================================================
        # 🧠 智能翻译器：重映射旧名字到新名字
        # =================================================
        print("🔄 正在执行权重名称重映射 (翻译字典) ...")
        mapped_state_dict = {}
        
        for old_key, tensor_value in flat_ckpt_dict.items():
            # 1. 将旧版单体 rgb_pooler 强行装载到 MoE 的第 0 号专家 (experts.0) 中
            if "rgb_pooler." in old_key:
                # 把 ...rgb_pooler.xxx 替换成 multimodal.projection.experts.0.xxx
                tail = old_key.split("rgb_pooler.")[-1]
                new_key = f"multimodal.projection.experts.0.{tail}"
                mapped_state_dict[new_key] = tensor_value
            
            # 2. 映射语言模型的词典
            elif "embed_tokens.weight" in old_key:
                new_key = "language.text_encoder.model.embed_tokens.weight"
                mapped_state_dict[new_key] = tensor_value
                
            else:
                mapped_state_dict[old_key] = tensor_value

        print(f"⚙️ 正在向模型注入重映射后的权重 (strict=False) ...")
        # 捕获可能出现的 Size Mismatch 异常
        try:
            load_info = model.load_state_dict(mapped_state_dict, strict=False)
            
            # 🩺 真实体检报告
            all_model_keys = set(model.state_dict().keys())
            missing_keys = set(load_info.missing_keys)
            true_loaded_keys = {k for k in (all_model_keys - missing_keys) if "num_batches_tracked" not in k}
            
            print("\n" + "="*50)
            print("✅ 权重加载执行完毕！真实体检报告如下：")
            print(f"📊 模型总层数: {len(all_model_keys)}")
            print(f"🟢 真实成功加载的层数: {len(true_loaded_keys)}")
            if len(true_loaded_keys) > 0:
                print("   ↳ 恭喜！同声传译器 (Projection Experts) 已成功对接！")
            print("="*50 + "\n")
            
        except RuntimeError as e:
            print("\n❌ 发生致命错误: 权重尺寸不匹配 (Size Mismatch)!")
            print("💡 提示: 请务必在 yaml 配置文件中将 `alignment_dim` 修改为与权重文件一致的值 (如 1024)！")
            print(f"详细报错信息: {e}")
            return # 停止运行，避免生成假热力图
        
    else:
        print(f"⚠️ 警告: 未找到权重文件 {ckpt_path}。将使用随机初始化的权重。")

    # 4. 准备测试图像
    test_image_path = "dummy_test.jpg"
    if not os.path.exists(test_image_path):
        # 自动生成一张彩色噪声图作为兜底测试
        Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255)).save(test_image_path)
        
    img_pil = Image.open(test_image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(tuple(config.transform.input_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_pil).unsqueeze(0).to(device) # [1, 3, 224, 224]

    # 5. 执行前向传播
    print("🧠 正在提取特征并执行门控路由...")
    with torch.no_grad():
        data = {
            "rgb": img_tensor,
            "task_ids": torch.tensor([0], dtype=torch.long).to(device)
        }
        
        # 获取视觉特征并进行特征投影
        if getattr(model, "physics_enabled", False) and hasattr(model.vision, "encode_with_spatial"):
            image_seq, _, _ = model.vision.encode_with_spatial(data["rgb"])
        else:
            image_seq = model.vision(data)
            
        _ = model.multimodal.encode_test(image_seq, data["task_ids"])

    # 6. 提取缓存的路由权重并绘图
    try:
        # 这个是我们在代码里挂好的后门
        routing_weights = model.multimodal.projection._cached_routing_weights
        
        # 移除 batch 维度 (假设 bs=1)
        if len(routing_weights.shape) == 3:
            routing_weights = routing_weights.squeeze(0)
            
        print(f"📊 成功截获路由权重！形状: {routing_weights.shape} (Patch数 x 专家数)")
        
        # 默认使用 4 个专家绘图，你可以根据实际配置调整
        num_experts = routing_weights.shape[1] if len(routing_weights.shape) > 1 else 1
        
        visualize_moe_routing(
            image_path=test_image_path, 
            routing_weights=routing_weights.cpu(), 
            num_experts=num_experts, 
            save_path="coastgpt_expert_routing_result.png"
        )
    except AttributeError:
        print("❌ 提取失败：没有找到 _cached_routing_weights。")
        print("💡 请确认：CoastGPT 源码中对应的 forward 方法是否保存了路由权重！")

if __name__ == "__main__":
    main()