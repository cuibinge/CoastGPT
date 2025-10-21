import os
import torch
import clip
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

# 参数配置
root_dir = r"D:\GF_cut-28"
output_root = r"D:\refiltered_images_v2"
csv_output_path = r"D:\refilter_results_v2.csv"

# 阈值配置
BRIGHTNESS_THRESH = 20
VARIANCE_THRESH = 30
SATURATION_THRESH = 10
SOBEL_THRESH = 5
OVEREXPOSED_THRESH = 245
OVEREXPOSED_RATIO = 0.85

STRUCTURE_ENTROPY_THRESH = 1.8
COLOR_ENTROPY_THRESH = 1.2
CLIP_STRICT_THRESH = 0.25
CLIP_BLACKLIST = {"pure cloud", "empty land", "a satellite image of water", "a satellite image of forest"}

clip_prompts = [
    "a satellite image of a city",
    "a satellite image of a road",
    "a satellite image of farmland",
    "a satellite image of forest",
    "a satellite image of water",
    "empty land",
    "pure cloud"
]

os.makedirs(output_root, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
text_tokens = clip.tokenize(clip_prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

def calculate_entropy(img):
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    hist = hist.ravel() / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def fast_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < BRIGHTNESS_THRESH:
        return False, brightness, 0, 0, 0, 0, 0

    variance = np.var(gray)
    if variance < VARIANCE_THRESH:
        return False, brightness, variance, 0, 0, 0, 0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1])
    if saturation < SATURATION_THRESH:
        return False, brightness, variance, saturation, 0, 0, 0

    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0) + cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    edge_strength = np.mean(np.abs(sobel))
    if edge_strength < SOBEL_THRESH:
        return False, brightness, variance, saturation, edge_strength, 0, 0

    overexposed_ratio = np.sum(gray > OVEREXPOSED_THRESH) / (gray.shape[0] * gray.shape[1])
    if overexposed_ratio > OVEREXPOSED_RATIO:
        return False, brightness, variance, saturation, edge_strength, 0, 0

    structure_entropy = calculate_entropy(gray)
    if structure_entropy < STRUCTURE_ENTROPY_THRESH:
        return False, brightness, variance, saturation, edge_strength, structure_entropy, 0

    hue_entropy = calculate_entropy(hsv[:, :, 0])
    if hue_entropy < COLOR_ENTROPY_THRESH:
        return False, brightness, variance, saturation, edge_strength, structure_entropy, hue_entropy

    return True, brightness, variance, saturation, edge_strength, structure_entropy, hue_entropy

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

            passed, brightness, variance, saturation, edge, se, he = fast_filter(img)
            label = ""
            score = 0.0
            keep = False

            if passed:
                # 仅对通过初筛图像执行CLIP
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    similarities = (image_features @ text_features.T).squeeze(0)
                    top_score, top_idx = similarities.max(0)
                    label = clip_prompts[top_idx]
                    score = top_score.item()

                # 放宽 clip 保留条件
                if score >= CLIP_STRICT_THRESH or label not in CLIP_BLACKLIST:
                    keep = True

            results.append({
                "subdir": sub_dir,
                "filename": file,
                "brightness": brightness,
                "variance": variance,
                "saturation": saturation,
                "edge_strength": edge,
                "structure_entropy": se,
                "hue_entropy": he,
                "clip_label": label,
                "clip_score": score,
                "keep": keep
            })

            if keep:
                cv2.imwrite(os.path.join(output_subdir, file), img)

df = pd.DataFrame(results)
df.to_csv(csv_output_path, index=False)
print(f"优化筛选完成，结果保存至 {csv_output_path}")
