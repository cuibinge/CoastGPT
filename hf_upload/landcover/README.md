# CoastGPT Landcover Semantic Head

CoastGPT 土地覆盖语义分割头 — a per-pixel semantic segmentation decoder operating on FPN feature pyramid from the DualVisionEncoder.

## Architecture

```
P1 [B,256,56,56]  ──┐
P2 [B,256,28,28]  ──┤ upsample to P1 size
P3 [B,256,14,14]  ──┤ → concat [B,1024,56,56]
P4 [B,256,7,7]    ──┘     │
                           ▼
                    Conv 256 + BN + ReLU
                           │
                    Conv 128 + BN + ReLU
                           │
                    Conv num_classes
                           │
                    bilinear upsample → [H, W]
```

## Usage

```python
import torch
from model import LandcoverSemanticHead
from huggingface_hub import hf_hub_download

# Load model
model = LandcoverSemanticHead(in_channels=256, num_classes=25)
state_dict = torch.load(
    hf_hub_download("cuibinge/coastgpt-landcover-semantic", "pytorch_model.bin"),
    map_location="cpu"
)
model.load_state_dict(state_dict["sem_head"])
model.eval()

# Requires FPN features from DualVisionEncoder:
# from Models.fpn_neck import FPNNeck
# fpn = FPNNeck(in_channels_list=[256, 512, 1024, 2048])
# fpn.load_state_dict(state_dict["fpn"])
# p1, p2, p3, p4 = fpn(encoder_features)

# Inference
with torch.no_grad():
    logits = model(p1, p2, p3, p4)  # [B, 25, 224, 224]
    pred = logits.argmax(dim=1)      # [B, 224, 224]
```

## Classes

25 land cover classes from GF-1 21-class DLMC mapping, including:
water, cropland, forest, grassland, buildings, roads, aquaculture ponds, bare land, coastal wetland, etc.

See `Dataset/landcover_label_map.py` for the full class mapping.

## Performance

| Metric | Value |
|--------|-------|
| Best mIoU | 71.4% |
| Image Size | 224×224 |
| Training Data | ~33,000 GF-1 4-band tiles |

## Dependencies

- PyTorch >= 2.0
- DualVisionEncoder from CoastGPT (DINOv3 ViT-L/16 + ConvNeXt-Base)
- FPN neck from `Models/fpn_neck.py`

## Citation

CoastGPT: Multimodal Coastal Remote Sensing Foundation Model.
