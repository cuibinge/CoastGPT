import os
import json
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# ----------- 1. 初始化 CLIP 模型和预处理器 -----------
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ----------- 2. 自动匹配图像文件路径（支持多扩展名） -----------
def find_image_path(image_folder, base_name):
    exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    for ext in exts:
        path = os.path.join(image_folder, base_name + ext)
        if os.path.exists(path):
            return path
    return None

# ----------- 3. 计算图像与多个文本句子的 CLIP 相似度 -----------
def clip_score(image_path, sentences):
    image = Image.open(image_path).convert("RGB")
    # 多文本句子 + 1 张图像送入模型
    inputs = processor(images=image, text=sentences, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds    # 图像嵌入
        text_embeds = outputs.text_embeds      # 文本嵌入

    # 对嵌入做单位向量归一化（余弦相似度标准流程）
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # 批量计算每个文本与图像之间的相似度
    similarity = torch.matmul(text_embeds, image_embeds.T).squeeze()  # shape: [num_sentences]
    return similarity.tolist()

# ----------- 4. 主流程：读取 JSON，处理图像和文本 -----------
def process_json(json_path, image_folder):
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    results = []

    for item in json_data["data"]:
        name = item["name"]
        caption = item.get("caption", "").strip()

        if not caption:
            print(f"⚠️ 跳过无 caption 的图像: {name}")
            continue

        # 匹配图像路径
        image_path = find_image_path(image_folder, name)
        if not image_path:
            print(f"❌ 未找到图像文件: {name}（尝试了多种扩展名）")
            continue

        # 拆分 caption 为多个句子（句号分隔）
        sentences = [s.strip() for s in caption.split(".") if s.strip()]
        sentences = sentences[:5]  # 只取前 5 个句子

        if not sentences:
            print(f"⚠️ 无有效句子: {name}")
            continue

        # 使用 CLIP 打分
        scores = clip_score(image_path, sentences)

        # 保存结果
        results.append({
            "name": name,
            "sentences": sentences,
            "scores": scores
        })

    return results

# ----------- 5. 程序入口：指定路径并运行主逻辑 -----------
if __name__ == "__main__":
    image_folder = r"C:\Users\me\Desktop\images"     # 图像目录路径
    json_path = "RS_Caption_Dataset.json"            # 注释 JSON 路径

    # 执行处理函数
    results = process_json(json_path, image_folder)

    # 输出每幅图像每句话的相似度结果
    for item in results:
        print(f"\n📷 {item['name']}")
        for i, (s, sc) in enumerate(zip(item["sentences"], item["scores"])):
            print(f"  [{i+1}] {s} --> 相似度: {sc:.4f}")
