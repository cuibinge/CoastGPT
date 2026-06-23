"""Fusion Predictors — uniform interface for all detection heads and LLM.

Each predictor:
  - Receives: image tensor, prompt string, georef dict, classes list
  - Returns:  PredictorOutput (branch, List[GeoJSON Feature], metadata, raw)
  - Features are already WGS84 GeoJSON — FusionPipeline does NOT re-transform.

Predictors:
  InstancePredictor  — Mask R-CNN for aquaculture
  SemanticPredictor  — LandcoverSemanticHead for land cover
  EdgePredictor      — SingleScaleEdgeHead for coastline
  LLMPredictor       — CoastGPT for GeoJSON generation (unknown class fallback)
  LLMTextPredictor   — CoastGPT for text generation (CAP/VQA)
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from utils.georef_transform import pixel_to_wgs84
from utils.geojson_builder import (
    build_feature_collection,
    filter_sliver_features,
    polygon_pixel_to_geojson_feature,
)
from utils.mask_utils import filter_small_polygons, mask_to_polygon


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PredictorOutput:
    """Uniform output from every predictor."""
    branch: str
    features: List[dict]     # GeoJSON Feature dicts (WGS84 coords)
    metadata: dict = field(default_factory=dict)
    # metadata keys: confidence, num_features, timing_ms, ...
    raw: Optional[dict] = None  # raw model output for debug/visualization


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BasePredictor(ABC):
    """All predictors implement this interface."""

    @abstractmethod
    def predict(
        self,
        image: torch.Tensor,
        prompt: str,
        georef: dict,
        classes: List[str],
    ) -> PredictorOutput:
        """Run inference for the given classes on the given image."""
        ...

    @property
    @abstractmethod
    def branch(self) -> str:
        ...


# ---------------------------------------------------------------------------
# InstancePredictor — Mask R-CNN (PoC-1)
# ---------------------------------------------------------------------------

class InstancePredictor(BasePredictor):
    """Aquaculture instance segmentation via Mask R-CNN."""

    def __init__(
        self,
        vision_encoder,
        fpn_neck,
        mask_rcnn: torch.nn.Module,
        score_thresh: float = 0.5,
        mask_thresh: float = 0.5,
        min_area_px: float = 8.0,
        simplify_epsilon: float = 0.5,
        class_name: str = "海水养殖区",
        device: str = "npu:0",
    ):
        self._vision = vision_encoder
        self._fpn = fpn_neck
        self._model = mask_rcnn
        self._score_thresh = score_thresh
        self._mask_thresh = mask_thresh
        self._min_area_px = min_area_px
        self._simplify_epsilon = simplify_epsilon
        self._class_name = class_name
        self._device = torch.device(device)

    @property
    def branch(self) -> str:
        return "instance"

    def predict(
        self,
        image: torch.Tensor,
        prompt: str,
        georef: dict,
        classes: List[str],
    ) -> PredictorOutput:
        if self._class_name not in classes:
            return PredictorOutput(
                branch=self.branch,
                features=[],
                metadata={"skipped": True, "reason": "class not requested"},
            )

        t0 = time.time()
        self._model.eval()

        with torch.inference_mode():
            # image: [1, 3, 224, 224]
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self._device)
            outputs = self._model(image)

        features = []
        for output in outputs:
            scores = output["scores"].cpu().numpy()
            masks = output["masks"].cpu().numpy()  # [N, 1, H, W]

            for i in range(len(scores)):
                score = float(scores[i])
                if score < self._score_thresh:
                    continue
                mask = (masks[i, 0] > self._mask_thresh).astype(np.uint8)
                polygons = mask_to_polygon(mask, simplify_epsilon=self._simplify_epsilon)
                polygons = filter_small_polygons(polygons, self._min_area_px)
                for poly in polygons:
                    feat = polygon_pixel_to_geojson_feature(
                        poly,
                        georef,
                        class_name=self._class_name,
                        confidence=score,
                    )
                    features.append(feat)

        timing_ms = (time.time() - t0) * 1000.0
        return PredictorOutput(
            branch=self.branch,
            features=features,
            metadata={
                "num_features": len(features),
                "timing_ms": timing_ms,
                "score_thresh": self._score_thresh,
            },
        )


# ---------------------------------------------------------------------------
# SemanticPredictor — Landcover Semantic Head (PoC-2b)
# ---------------------------------------------------------------------------

class SemanticPredictor(BasePredictor):
    """Land cover semantic segmentation."""

    # Mapping from class index → class name (must match training label_map)
    _DEFAULT_CLASS_NAMES: Dict[int, str] = {}  # populated from label_map.json at init

    def __init__(
        self,
        vision_encoder,
        fpn_neck,
        semantic_head: torch.nn.Module,
        class_names: Dict[int, str],
        min_area_px: float = 50.0,
        simplify_epsilon: float = 1.0,
        device: str = "npu:0",
    ):
        self._vision = vision_encoder
        self._fpn = fpn_neck
        self._head = semantic_head
        self._class_names = class_names
        self._min_area_px = min_area_px
        self._simplify_epsilon = simplify_epsilon
        self._device = torch.device(device)

    @property
    def branch(self) -> str:
        return "semantic"

    def predict(
        self,
        image: torch.Tensor,
        prompt: str,
        georef: dict,
        classes: List[str],
    ) -> PredictorOutput:
        t0 = time.time()

        with torch.inference_mode():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self._device)

            # Get FPN features
            _, _, pyramid_raw = self._vision.encode_with_spatial(image)
            c4, c8, c16, c32 = pyramid_raw
            p1, p2, p3, p4 = self._fpn(c4, c8, c16, c32)

            # Semantic head forward
            logits = self._head(p1, p2, p3, p4)  # [B, num_classes, 224, 224]
            pred = logits.argmax(dim=1)[0].cpu().numpy()  # [224, 224]

        features = []
        for class_idx, class_name in self._class_names.items():
            if class_name not in classes:
                continue
            mask = (pred == class_idx).astype(np.uint8)
            if mask.sum() < self._min_area_px:
                continue
            polygons = mask_to_polygon(mask, simplify_epsilon=self._simplify_epsilon)
            polygons = filter_small_polygons(polygons, self._min_area_px)
            for poly in polygons:
                feat = polygon_pixel_to_geojson_feature(
                    poly, georef, class_name=class_name, confidence=1.0,
                )
                features.append(feat)

        timing_ms = (time.time() - t0) * 1000.0
        return PredictorOutput(
            branch=self.branch,
            features=features,
            metadata={"num_features": len(features), "timing_ms": timing_ms},
            raw={"logits": logits.cpu().numpy() if hasattr(logits, "cpu") else None},
        )


# ---------------------------------------------------------------------------
# EdgePredictor — Coastline Edge Head (PoC-3)
# ---------------------------------------------------------------------------

class EdgePredictor(BasePredictor):
    """Coastline edge detection."""

    def __init__(
        self,
        vision_encoder,
        fpn_neck,
        edge_head: torch.nn.Module,
        threshold: float = 0.6,
        min_area: int = 8,
        min_length: int = 10,
        max_components: int = 5,
        simplify_epsilon: float = 0.1,
        class_name: str = "海岸线",
        device: str = "npu:0",
    ):
        self._vision = vision_encoder
        self._fpn = fpn_neck
        self._head = edge_head
        self._threshold = threshold
        self._min_area = min_area
        self._min_length = min_length
        self._max_components = max_components
        self._simplify_epsilon = simplify_epsilon
        self._class_name = class_name
        self._device = torch.device(device)

    @property
    def branch(self) -> str:
        return "edge"

    def predict(
        self,
        image: torch.Tensor,
        prompt: str,
        georef: dict,
        classes: List[str],
    ) -> PredictorOutput:
        if self._class_name not in classes:
            return PredictorOutput(
                branch=self.branch,
                features=[],
                metadata={"skipped": True, "reason": "class not requested"},
            )

        t0 = time.time()

        with torch.inference_mode():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self._device)

            _, _, pyramid_raw = self._vision.encode_with_spatial(image)
            c4, c8, c16, c32 = pyramid_raw
            p1, p2, p3, p4 = self._fpn(c4, c8, c16, c32)

            logits = self._head(p1, p2, p3, p4)  # [B, 1, 224, 224]
            heatmap = torch.sigmoid(logits[0, 0]).cpu().numpy()

        from utils.edge_postprocess import postprocess_edge
        fc = postprocess_edge(
            heatmap=heatmap,
            georef=georef,
            threshold=self._threshold,
            min_area=self._min_area,
            min_length=self._min_length,
            max_components=self._max_components,
            simplify_epsilon=self._simplify_epsilon,
        )
        features = fc.get("features", [])

        timing_ms = (time.time() - t0) * 1000.0
        return PredictorOutput(
            branch=self.branch,
            features=features,
            metadata={"num_features": len(features), "timing_ms": timing_ms},
            raw={"heatmap": heatmap, "logits": logits.cpu().numpy()},
        )


# ---------------------------------------------------------------------------
# LLMPredictor — GeoJSON generation for unknown classes
# ---------------------------------------------------------------------------

class LLMPredictor(BasePredictor):
    """LLM-based GeoJSON generation for unknown classes.

    Used ONLY for LLM fallback path, not for the parser.
    """

    def __init__(
        self,
        coastgpt_model,
        tokenizer,
        config,
        max_new_tokens: int = 1024,
        device: str = "npu:0",
    ):
        self._model = coastgpt_model
        self._tokenizer = tokenizer
        self._config = config
        self._max_new_tokens = max_new_tokens
        self._device = torch.device(device)

    @property
    def branch(self) -> str:
        return "llm"

    def predict(
        self,
        image: torch.Tensor,
        prompt: str,
        georef: dict,
        classes: List[str],
    ) -> PredictorOutput:
        t0 = time.time()

        # Build generation prompt for unknown classes only
        class_list = "、".join(classes)
        full_prompt = (
            f"[DET] 请检测图中的{class_list}。"
            f"Output the extracted feature information as a GeoJSON "
            f"FeatureCollection. Return JSON only."
        )

        # This mirrors the Inference.py generation path
        from Models import (
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from Models.utils import tokenizer_image_token

        gen_prompt = DEFAULT_IMAGE_TOKEN + "\n" + full_prompt
        input_ids = tokenizer_image_token(
            gen_prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self._device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids=input_ids,
                images=image.to(self._device),
                do_sample=False,
                temperature=1.0,
                max_new_tokens=self._max_new_tokens,
                use_cache=True,
            )

        # Decode
        new_tokens = output_ids[0, input_ids.shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Parse GeoJSON from text
        import json as _json
        features = []
        try:
            # Try to extract JSON from text
            json_start = text.find("{")
            json_end = text.rfind("}")
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end + 1]
                obj = _json.loads(json_str)
                if obj.get("type") == "FeatureCollection":
                    features = obj.get("features", [])
                elif obj.get("type") == "Feature":
                    features = [obj]
        except _json.JSONDecodeError:
            pass

        timing_ms = (time.time() - t0) * 1000.0
        return PredictorOutput(
            branch=self.branch,
            features=features,
            metadata={"num_features": len(features), "timing_ms": timing_ms, "raw_text": text},
            raw={"text": text},
        )


# ---------------------------------------------------------------------------
# LLMTextPredictor — text generation for CAP/VQA
# ---------------------------------------------------------------------------

class LLMTextPredictor(BasePredictor):
    """LLM for text generation (CAP/VQA paths)."""

    def __init__(
        self,
        coastgpt_model,
        tokenizer,
        config,
        max_new_tokens: int = 512,
        device: str = "npu:0",
    ):
        self._model = coastgpt_model
        self._tokenizer = tokenizer
        self._config = config
        self._max_new_tokens = max_new_tokens
        self._device = torch.device(device)

    @property
    def branch(self) -> str:
        return "llm_text"

    def predict(
        self,
        image: torch.Tensor,
        prompt: str,
        georef: dict,
        classes: List[str],
    ) -> PredictorOutput:
        t0 = time.time()

        from Models import (
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from Models.utils import tokenizer_image_token

        gen_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        input_ids = tokenizer_image_token(
            gen_prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self._device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids=input_ids,
                images=image.to(self._device),
                do_sample=False,
                temperature=1.0,
                max_new_tokens=self._max_new_tokens,
                use_cache=True,
            )

        new_tokens = output_ids[0, input_ids.shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        timing_ms = (time.time() - t0) * 1000.0
        return PredictorOutput(
            branch=self.branch,
            features=[],  # text mode — no GeoJSON features
            metadata={"num_features": 0, "timing_ms": timing_ms},
            raw={"text": text},
        )
