import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image
from model import (VisualProjection,ViTFeatureExtractor,CNNFeatureExtractor,BiasTuningLinear,EarthGPTUnified,EarthGPT)

# 测试 ViTFeatureExtractor
vit_extractor = ViTFeatureExtractor()
dummy_img = torch.randn(1, 3, 224, 224)  # 假设输入为 224x224 RGB 图像
vit_features = vit_extractor(dummy_img)
assert vit_features.shape[1] == 12 * 384, f"Unexpected shape: {vit_features.shape}"
print("ViTFeatureExtractor test passed!")

# 测试 CNNFeatureExtractor
cnn_extractor = CNNFeatureExtractor()
cnn_features = cnn_extractor(dummy_img)
assert cnn_features.shape[0] == 1, f"Unexpected shape: {cnn_features.shape}"
print("CNNFeatureExtractor test passed!")

# 测试 VisualProjection
dummy_vit_feat = torch.randn(1, 12 * 384)
dummy_cnn_feat = torch.randn(1, 1000)
visual_proj = VisualProjection(12 * 384, 1000, 512)
projected_features = visual_proj(dummy_vit_feat, dummy_cnn_feat)
assert projected_features.shape == (1, 512), f"Unexpected shape: {projected_features.shape}"
print("VisualProjection test passed!")

# 测试 BiasTuningLinear
dummy_input = torch.randn(1, 512)
bias_linear = BiasTuningLinear(512, 256)
bias_output = bias_linear(dummy_input)
assert bias_output.shape == (1, 256), f"Unexpected shape: {bias_output.shape}"
print("BiasTuningLinear test passed!")

# 测试 EarthGPT
earth_gpt = EarthGPT(512)
dummy_visual_feat = torch.randn(1, 512)
dummy_text = "Describe the image"
loss = earth_gpt(dummy_visual_feat, dummy_text)
assert loss is not None, "Loss is None!"
print("EarthGPT test passed!")

# 测试 EarthGPTUnified
earth_gpt_unified = EarthGPTUnified(512)
dummy_visual_feat = torch.randn(1, 512)
dummy_text = "Analyze the satellite image"
loss = earth_gpt_unified(dummy_visual_feat, dummy_text)
assert loss is not None, "Loss is None!"
print("EarthGPTUnified test passed!")