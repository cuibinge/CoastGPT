import os
import torch
import clip
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


root_dir = r"C:/Users/zhhq9/Desktop/GF_cut"      # 图像目录
output_root = "filtered_images"      # 输出目录
csv_output_path = "filter_results.csv"

BRIGHTNESS_THRESH = 30
VARIANCE_THRESH = 50
SATURATION_THRESH = 20
SOBEL_THRESH = 10
OVEREXPOSED_THRESH = 245
OVEREXPOSED_RATIO = 0.8
CLIP_SCORE_THRESH = 0.25


os.makedirs(output_root, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

clip_prompts = [
    "a satellite image of a city",
    "a satellite image of a road",
    "a satellite image of farmland",
    "a satellite image of forest",
    "a satellite image of water",
    "empty land",
    "pure cloud"
]
text_tokens = clip.tokenize(clip_prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

def is_valid_image(img, img_path):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < BRIGHTNESS_THRESH:
        return False, brightness, 0, 0, 0, "", 0

    variance = np.var(gray)
    if variance < VARIANCE_THRESH:
        return False, brightness, variance, 0, 0, "", 0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1])
    if saturation < SATURATION_THRESH:
        return False, brightness, variance, saturation, 0, "", 0

    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0) + cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    edge_strength = np.mean(np.abs(sobel))
    if edge_strength < SOBEL_THRESH:
        return False, brightness, variance, saturation, edge_strength, "", 0

    overexposed_ratio = np.sum(gray > OVEREXPOSED_THRESH) / (gray.shape[0] * gray.shape[1])
    if overexposed_ratio > OVEREXPOSED_RATIO:
        return False, brightness, variance, saturation, edge_strength, "", 0

    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).squeeze(0)
        top_score, top_idx = similarities.max(0)
        top_label = clip_prompts[top_idx]

    if top_score.item() < CLIP_SCORE_THRESH:
        return False, brightness, variance, saturation, edge_strength, top_label, top_score.item()

    return True, brightness, variance, saturation, edge_strength, top_label, top_score.item()

results = []

for sub_dir in tqdm(os.listdir(root_dir), desc="遍历大图文件夹"):
    full_subdir_path = os.path.join(root_dir, sub_dir)
    if not os.path.isdir(full_subdir_path):
        continue

    output_subdir = os.path.join(output_root, sub_dir)
    os.makedirs(output_subdir, exist_ok=True)

    for file in os.listdir(full_subdir_path):
        if file.endswith(".png"):
            img_path = os.path.join(full_subdir_path, file)
            img = cv2.imread(img_path)
            if img is None or img.shape[:2] != (256, 256):
                continue

            keep, brightness, variance, saturation, edge, label, score = is_valid_image(img, img_path)
            results.append({
                "subdir": sub_dir,
                "filename": file,
                "brightness": brightness,
                "variance": variance,
                "saturation": saturation,
                "edge_strength": edge,
                "clip_label": label,
                "clip_score": score,
                "keep": keep
            })

            if keep:
                cv2.imwrite(os.path.join(output_subdir, file), img)

# 保存结果为 CSV
df = pd.DataFrame(results)
df.to_csv(csv_output_path, index=False)
print(f"筛选完成，结果已保存至 {csv_output_path}")
