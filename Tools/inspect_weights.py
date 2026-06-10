import torch
from Models.coastgpt import CoastGPT
import yaml
import json
from types import SimpleNamespace
import ml_collections
# 为了让脚本能独立运行演示，这里用一个临时的假模型代替
import torch.nn as nn
import torch_npu

# 假设是 yaml 格式的配置文件
config_path = './Configs/train.yaml' # 替换为实际的文件名

with open(config_path, 'r', encoding='utf-8') as f:
    config_dict = yaml.safe_load(f)
config_obj = ml_collections.ConfigDict(config_dict)
model = CoastGPT(config_obj) # 替换成你的 model = CoastGPT()

# ==========================================
# 2. 加载 .ckpt 权重文件
# ==========================================
ckpt_path = './FINAL.pt'
print(f"正在加载权重文件: {ckpt_path} ...")
# 使用 map_location='cpu' 防止在没有 GPU 的机器上显存溢出
checkpoint = torch.load(ckpt_path, map_location='npu')

# 解析权重文件中的 state_dict
# 有些框架（如 PyTorch Lightning）会把权重包在 'state_dict' 或 'model' 字典里
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    ckpt_state_dict = checkpoint['state_dict']
elif isinstance(checkpoint, dict) and 'model' in checkpoint:
    ckpt_state_dict = checkpoint['model']
else:
    ckpt_state_dict = checkpoint # 如果存的直接就是纯权重字典

# ==========================================
# ==========================================
# 前 10 层直观对比检查
# ==========================================
model_state_dict = model.state_dict()
def get_shape_str(obj):
    """安全获取张量的形状，如果不是张量则返回类型"""
    if hasattr(obj, 'shape'):
        return str(list(obj.shape))
    elif isinstance(obj, torch.Tensor):
        return str(list(obj.size()))
    else:
        return f"非张量 (类型: {type(obj).__name__})"

def flatten_dict(d, parent_key='', sep='.'):
    """
    递归展开嵌套字典。
    例如把 {'other_ckpt': {'rgb_pooler': {'weight': tensor}}}
    展开为 'other_ckpt.rgb_pooler.weight': tensor_shape
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # 如果值还是个字典，继续往里挖
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            # 如果到底了（是张量或基础数据类型），记录形状
            items.append((new_key, get_shape_str(v)))
    return dict(items)

# ==========================================
# 3. 提取并处理所有的键
# ==========================================
print("正在解析模型结构...")
model_state_dict = model.state_dict()
# 模型本身的 state_dict 已经是扁平的了，直接提取
model_info = {k: get_shape_str(v) for k, v in model_state_dict.items()}

print("正在递归解析权重文件 (这可能需要几秒钟)...")
# 权重文件是嵌套的，使用 flatten_dict 把它彻底拍平
ckpt_info = flatten_dict(ckpt_state_dict)

# ==========================================
# 4. 将结果写入文本文件
# ==========================================
output_file = "all_layers_compare.txt"

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write(f"🔍 深度比对报告：模型结构 VS 权重文件\n")
    f.write("="*80 + "\n\n")

    # 写入代码实例化的模型层
    f.write(f"【Part 1: 代码实例化的模型 (Model)】\n")
    f.write(f"总层数: {len(model_info)}\n")
    f.write("-" * 50 + "\n")
    for i, (k, shape) in enumerate(model_info.items()):
        f.write(f"[{i+1}] {k}\n")
        f.write(f"    ↳ 形状: {shape}\n")
    
    f.write("\n\n" + "="*80 + "\n\n")

    # 写入权重文件层
    f.write(f"【Part 2: 准备加载的权重文件 (.ckpt)】\n")
    f.write(f"总层数: {len(ckpt_info)}\n")
    f.write("-" * 50 + "\n")
    # 为了方便查看，我们可以把解析出来的键排个序
    sorted_ckpt_keys = sorted(ckpt_info.keys())
    for i, k in enumerate(sorted_ckpt_keys):
        f.write(f"[{i+1}] {k}\n")
        f.write(f"    ↳ 形状: {ckpt_info[k]}\n")

print(f"\n✅ 成功！全量比对数据已保存至当前目录下的: {output_file}")
print(f"模型包含 {len(model_info)} 个层，权重文件包含 {len(ckpt_info)} 个层。")
print("请使用文本编辑器打开该文件进行搜索和对比。")