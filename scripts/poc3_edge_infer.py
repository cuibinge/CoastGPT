#!/usr/bin/env python3
"""PoC-3 Edge → GeoJSON inference pipeline using A1-relabel unified model."""

import argparse, json, sys
from pathlib import Path
import numpy as np, torch, yaml, tifffile
from ml_collections import ConfigDict
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from Models.dual_vision_encoder import DualVisionEncoder
from Models.fpn_neck import FPNNeck
from Models.edge_head import SingleScaleEdgeHead
from utils.edge_postprocess import heatmap_to_binary, binary_to_skeleton, skeleton_to_paths, simplify_path, paths_to_geojson
from utils.georef_transform import resize_georef


def load_model(config_path, checkpoint_path, device):
    with open(config_path) as f:
        cfg = ConfigDict(yaml.safe_load(f))
    vision = DualVisionEncoder(ConfigDict(cfg.get('model', {})))
    sd = torch.load(cfg.get('model.vision_checkpoint'), map_location='cpu')
    sd = sd.get('vision_ckpt', sd.get('model', sd))
    clean = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith('module.'): nk = nk[7:]
        if nk.startswith('vision.'): nk = nk[7:]
        clean[nk] = v
    vision.load_state_dict(clean, strict=False)
    vision = vision.to(device).eval()
    for p in vision.parameters(): p.requires_grad = False

    fpn = FPNNeck([128, 256, 512, 1024], 256, vit_in_channels=1024).to(device).eval()
    edge_head = SingleScaleEdgeHead(256, output_size=(224, 224)).to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    fpn.load_state_dict(ckpt['fpn'])
    edge_head.load_state_dict(ckpt['edge_head'])
    return vision, fpn, edge_head


def predict(vision, fpn, edge_head, image_tensor, device):
    with torch.no_grad():
        _, g_grid, pyr = vision.encode_with_spatial(image_tensor.to(device))
        p1, p2, p3, p4 = fpn(pyr[0], pyr[1], pyr[2], pyr[3], vit_feat=g_grid if fpn.has_vit else None)
        logits = edge_head(p1, p2, p3, p4)
    return torch.sigmoid(logits).cpu().numpy()


def heatmap_to_geojson(heatmap, georef, threshold=0.25, min_length=5, max_components=10,
                        simplify_epsilon=0.1, sample_id=''):
    binary = heatmap_to_binary(heatmap, threshold=threshold, min_area=4)
    skeleton = binary_to_skeleton(binary)
    paths = skeleton_to_paths(skeleton)
    # Keep original paths before simplification
    paths = [p for p in paths if len(p) >= 3]  # need at least 3 pts for a line
    if not paths:
        return {"type": "FeatureCollection", "features": []}
    paths = [simplify_path(p, epsilon=simplify_epsilon) for p in paths]
    paths = sorted(paths, key=len, reverse=True)[:max_components]
    paths = [p for p in paths if len(p) >= min_length]
    return paths_to_geojson(paths, georef, sample_id, class_name="海岸线")


def main():
    parser = argparse.ArgumentParser(description="PoC-3 Edge → GeoJSON Inference")
    parser.add_argument("--config", "-c", default="configs/poc3_edge_a1_focal_dice.yaml")
    parser.add_argument("--checkpoint", default="outputs/poc3_edge/a1_relabel_unified/checkpoints/epoch_010.pt")
    parser.add_argument("--image", required=True, help="Path to input TIF image")
    parser.add_argument("--geojson", help="Path to GeoJSON label (for georef)")
    parser.add_argument("--output", "-o", default="output.geojson")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--device", default="npu")
    args = parser.parse_args()

    if args.device == 'npu':
        import torch_npu
        device = torch.device('npu:0')
    else:
        device = torch.device('cpu')

    print(f"Loading model...")
    vision, fpn, edge_head = load_model(args.config, args.checkpoint, device)

    print(f"Loading image: {args.image}")
    arr = tifffile.imread(args.image)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]  # RGB channels only
    elif arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)  # grayscale → RGB
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.max() <= 1:
        arr = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr).resize((224, 224), Image.BILINEAR)
    img_tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    # Georef from GeoJSON label or default
    georef = {"source_crs": "EPSG:4326", "model_transform": [1e-5, 0, 0, 0, -1e-5, 0]}
    if args.geojson:
        with open(args.geojson) as f:
            gj = json.load(f)
        # Try to get tile bounds
        features = gj.get('features', [])
        if features:
            all_coords = []
            for feat in features:
                geom = feat.get('geometry', {})
                coords = geom.get('coordinates', [])
                if geom['type'] == 'LineString': all_coords.extend(coords)
                elif geom['type'] == 'MultiLineString':
                    for line in coords: all_coords.extend(line)
            if all_coords:
                lons = [c[0] for c in all_coords]; lats = [c[1] for c in all_coords]
                x_res = (max(lons) - min(lons)) / 224
                y_res = (max(lats) - min(lats)) / 224
                georef['model_transform'] = [x_res, 0, min(lons), 0, -y_res, max(lats)]
                print(f"Derived georef from GeoJSON bounds")

    print(f"Predicting...")
    heatmap = predict(vision, fpn, edge_head, img_tensor, device)[0, 0]

    print(f"Heatmap: max={heatmap.max():.3f}, fg@0.25={(heatmap>0.25).mean()*100:.1f}%")
    fc = heatmap_to_geojson(heatmap, georef, threshold=args.threshold, sample_id=Path(args.image).stem)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(fc['features'])} LineStrings → {args.output}")


if __name__ == "__main__":
    main()
