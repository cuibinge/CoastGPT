"""FusionPipeline — orchestrates detection heads + LLM fallback with validation and dedup.

Main entry: FusionPipeline.run(image, prompt, georef) → FeatureCollection dict + diagnostics.

Flow:
  Step 1: Prefix check (rule layer)
  Step 2: [CAP]/[VQA] early return → llm_text predictor
  Step 3: LLM Parser → ParseResult
  Step 4: Gating → DispatchMap
  Step 5: Detection heads (known classes only)
  Step 6: LLM Fallback (unknown classes only, with Layer1 + Layer2 validation)
  Step 7: Dedup (LLM self-dedup + cross-source)
  Step 8: Build FeatureCollection + diagnostics
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from Models.fusion_predictors import BasePredictor
from utils.geojson_builder import build_feature_collection, validate_geojson, filter_sliver_features
from utils.geojson_dedup import dedup_within_source, dedup_cross_source
from utils.geojson_validator import validate_llm_fallback


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FusionConfig:
    """PoC-4 fusion pipeline configuration."""
    # Label map
    label_map_path: str = "Configs/label_map.json"

    # Parser
    parser_max_new_tokens: int = 128
    parser_confidence_threshold: float = 0.3
    parser_prompt_template: str = (
        "You are a task parser. Given a user prompt, extract:\n"
        "1. task_type: one of DET, CAP, VQA\n"
        "2. target_classes: list of class names the user wants to detect/describe\n"
        "3. raw_mentions: list of class-like phrases found in the prompt\n"
        "4. confidence: 0.0-1.0\n\n"
        "Respond ONLY with a JSON object. No explanation.\n\n"
        "Prompt: {user_prompt}\n\n"
        "JSON:"
    )

    # Gating
    empty_target_policy: str = "error"  # "error" | "all_heads" | "llm_only"

    # Validation
    sliver_min_area_deg: float = 1e-10
    sliver_min_length_deg: float = 1e-6
    sliver_min_points: int = 2
    llm_min_confidence: float = 0.3

    # Dedup
    dedup_internal_thresholds: dict = field(default_factory=lambda: {"area": 0.7, "line": 0.6, "point": 5e-6})
    dedup_cross_thresholds: dict = field(default_factory=lambda: {"area": 0.5, "line": 0.5, "point": 1e-5})


@dataclass
class ParseResult:
    """Output of LLM Parser."""
    task_type: str         # "DET" | "CAP" | "VQA"
    target_classes: List[str]
    raw_mentions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "llm"    # "llm" | "rule_fallback"


@dataclass
class DispatchMap:
    """Output of Gating."""
    known: Dict[str, List[str]] = field(default_factory=lambda: {
        "instance": [], "semantic": [], "edge": [],
    })
    unknown: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM Parser
# ---------------------------------------------------------------------------

class LLMParser:
    """LLM-based prompt parser for task type and target class extraction.

    Falls back to Rule Layer on parse failure or low confidence.
    """

    def __init__(
        self,
        coastgpt_model,
        tokenizer,
        config: FusionConfig,
        label_map: dict,
        device: str = "npu:0",
    ):
        self._model = coastgpt_model
        self._tokenizer = tokenizer
        self._config = config
        self._label_map = label_map
        self._device = torch.device(device)

    def parse(self, prompt: str, image: torch.Tensor) -> ParseResult:
        """Parse prompt to extract task_type and target_classes."""
        import json as _json

        # --- Attempt LLM parse ---
        try:
            result = self._llm_parse(prompt, image)
            if result is not None:
                return result
        except Exception:
            pass

        # --- Rule Layer fallback ---
        return self._rule_parse(prompt)

    def _llm_parse(self, prompt: str, image: torch.Tensor) -> Optional[ParseResult]:
        """Try LLM-based parsing. Returns None on failure."""
        from Models import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, tokenizer_image_token

        parse_prompt_text = self._config.parser_prompt_template.replace("{user_prompt}", prompt)
        full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + parse_prompt_text

        input_ids = tokenizer_image_token(
            full_prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self._device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids=input_ids,
                images=image.to(self._device),
                do_sample=False,
                temperature=1.0,
                max_new_tokens=self._config.parser_max_new_tokens,
                use_cache=True,
                remove_invalid_values=True,
                renormalize_logits=True,
            )

        new_tokens = output_ids[0, input_ids.shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Extract JSON
        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start < 0 or json_end <= json_start:
            return None

        try:
            obj = _json.loads(text[json_start:json_end + 1])
        except _json.JSONDecodeError:
            return None

        task_type = str(obj.get("task_type", "DET")).upper()
        if task_type not in ("DET", "CAP", "VQA"):
            task_type = "DET"

        target_classes = obj.get("target_classes", [])
        if not isinstance(target_classes, list):
            target_classes = []

        confidence = float(obj.get("confidence", 0.0))
        if confidence < self._config.parser_confidence_threshold:
            return None  # Trigger rule fallback

        return ParseResult(
            task_type=task_type,
            target_classes=target_classes,
            raw_mentions=obj.get("raw_mentions", []),
            confidence=confidence,
            source="llm",
        )

    def _rule_parse(self, prompt: str) -> ParseResult:
        """Conservative rule-based fallback: extract prefix + keyword match."""
        # Prefix extraction
        prompt_upper = prompt.upper()
        if "[CAP]" in prompt_upper:
            return ParseResult(task_type="CAP", target_classes=[], source="rule_fallback")
        if "[VQA]" in prompt_upper:
            return ParseResult(task_type="VQA", target_classes=[], source="rule_fallback")

        # [DET] — try keyword matching against label_map
        # substring matching against label_map class names
        all_class_names = list(self._label_map.get("branch_classes", {}).get("instance", {}).values())
        all_class_names += list(self._label_map.get("branch_classes", {}).get("semantic", {}).values())
        all_class_names += list(self._label_map.get("branch_classes", {}).get("edge", {}).values())
        all_class_names = [n for n in all_class_names if n and n != "background"]

        matched = []
        for name in all_class_names:
            if name in prompt:
                matched.append(name)

        return ParseResult(
            task_type="DET",
            target_classes=matched if matched else [],
            source="rule_fallback",
            confidence=0.5 if matched else 0.0,
        )


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

class Gating:
    """Deterministic class→branch routing based on label_map.json."""

    def __init__(self, label_map: dict):
        self._label_map = label_map
        # Build class_name → branch index
        self._class_to_branch: Dict[str, str] = {}
        branch_classes = label_map.get("branch_classes", {})
        for branch in ("instance", "semantic", "edge"):
            for class_id, class_name in branch_classes.get(branch, {}).items():
                if class_name and class_name != "background":
                    self._class_to_branch[class_name] = branch

    def route(self, target_classes: List[str]) -> DispatchMap:
        """Route target classes to branches."""
        dm = DispatchMap()
        for cls_name in target_classes:
            branch = self._class_to_branch.get(cls_name)
            if branch:
                dm.known[branch].append(cls_name)
            else:
                dm.unknown.append(cls_name)
        return dm


# ---------------------------------------------------------------------------
# FusionPipeline
# ---------------------------------------------------------------------------

class FusionPipeline:
    """Orchestrates detection heads + LLM fallback → validated, deduped GeoJSON."""

    def __init__(
        self,
        predictors: Dict[str, BasePredictor],
        config: FusionConfig,
        label_map: dict,
        coastgpt_model=None,
        tokenizer=None,
        device: str = "npu:0",
    ):
        self.predictors = predictors
        self.config = config
        self.label_map = label_map

        self.parser = LLMParser(
            coastgpt_model=coastgpt_model,
            tokenizer=tokenizer,
            config=config,
            label_map=label_map,
            device=device,
        )
        self.gating = Gating(label_map)

    def run(self, image: torch.Tensor, prompt: str, georef: dict) -> Tuple[dict, dict]:
        """Main entry point.

        Args:
            image: [1, 3, 224, 224] or [3, 224, 224] tensor.
            prompt: User text prompt.
            georef: Dict with source_crs, model_transform, tile_bounds_wgs84.

        Returns:
            (feature_collection: dict, diagnostics: dict)
        """
        diagnostics: Dict[str, Any] = {}

        # Step 1: Prefix check (rule layer, fast path)
        task_prefix = self._extract_prefix(prompt)
        diagnostics["prefix"] = task_prefix

        # Step 2: [CAP]/[VQA] early return — pure text path
        if task_prefix in ("CAP", "VQA"):
            llm_text = self.predictors.get("llm_text")
            if llm_text is None:
                diagnostics["early_return_error"] = "llm_text predictor not configured"
                return {"type": "FeatureCollection", "features": []}, diagnostics
            output = llm_text.predict(image, prompt, georef, [])
            diagnostics["early_return"] = task_prefix
            diagnostics["text_output"] = output.raw.get("text", "") if output.raw else ""
            return {"type": "FeatureCollection", "features": []}, diagnostics

        # Step 3: LLM Parser → ParseResult
        parse_result = self.parser.parse(prompt, image)
        diagnostics["parse"] = {
            "task_type": parse_result.task_type,
            "target_classes": parse_result.target_classes,
            "confidence": parse_result.confidence,
            "source": parse_result.source,
        }

        # Step 4: Gating → DispatchMap
        dispatch_map = self.gating.route(parse_result.target_classes)
        diagnostics["dispatch"] = {
            "known": {k: v for k, v in dispatch_map.known.items()},
            "unknown": dispatch_map.unknown,
        }

        # Handle empty target_classes
        if not dispatch_map.known and not dispatch_map.unknown:
            if self.config.empty_target_policy == "error":
                raise ValueError(
                    f"No target classes extracted from prompt: {prompt!r}. "
                    f"Parse result: {parse_result}"
                )
            elif self.config.empty_target_policy == "all_heads":
                # Run all heads blindly
                dispatch_map.known = {
                    "instance": ["海水养殖区"],
                    "semantic": list(self._get_semantic_class_names()),
                    "edge": ["海岸线"],
                }
            elif self.config.empty_target_policy == "llm_only":
                # Route everything to LLM fallback with original prompt
                dispatch_map.unknown = ["*"]

        # Step 5: Run detection heads (known classes only)
        det_features = []
        tile_bounds = georef.get("tile_bounds_wgs84")
        for branch in ("instance", "semantic", "edge"):
            classes = dispatch_map.known.get(branch, [])
            if not classes:
                continue
            predictor = self.predictors.get(branch)
            if predictor is None:
                diagnostics.setdefault("det_errors", {})[branch] = f"Predictor '{branch}' not configured"
                continue
            try:
                output = predictor.predict(image, prompt, georef, classes)
                # Layer 1 validation for detection head features
                fc = build_feature_collection(output.features)
                val_result = validate_geojson(fc, tile_bounds_wgs84=tile_bounds)
                if val_result.get("valid"):
                    det_features.extend(output.features)
                else:
                    diagnostics.setdefault("det_val_errors", {})[branch] = val_result.get("errors", [])
            except Exception as exc:
                diagnostics.setdefault("det_errors", {})[branch] = str(exc)

        diagnostics["det_feature_count"] = len(det_features)

        # Step 6: LLM Fallback (unknown classes only)
        llm_features = []
        if dispatch_map.unknown:
            try:
                llm_prompt = self._build_fallback_prompt(
                    prompt, dispatch_map.unknown
                )
                llm_predictor = self.predictors.get("llm")
                if llm_predictor is None:
                    raise RuntimeError("llm predictor not configured")
                output = llm_predictor.predict(
                    image, llm_prompt, georef, dispatch_map.unknown
                )
                # Layer 1: GeoJSON validation
                fc = build_feature_collection(output.features)
                val_l1 = validate_geojson(fc, tile_bounds_wgs84=tile_bounds)
                if val_l1.get("valid"):
                    # Layer 2: LLM fallback policy validation
                    valid_llm, report_l2 = validate_llm_fallback(
                        output.features,
                        unknown_class_whitelist=dispatch_map.unknown,
                        config={
                            "min_confidence": self.config.llm_min_confidence,
                            "known_classes": list(self._get_all_known_class_names()),
                        },
                    )
                    llm_features = valid_llm
                    diagnostics["llm_val"] = report_l2.stats
                else:
                    diagnostics["llm_val_l1_errors"] = val_l1.get("errors", [])
            except Exception as exc:
                diagnostics["llm_error"] = str(exc)

        diagnostics["llm_feature_count_before_dedup"] = len(llm_features)

        # Step 7: Dedup
        # 7a: LLM self-dedup
        llm_deduped, report_self = dedup_within_source(
            llm_features, thresholds=self.config.dedup_internal_thresholds
        )
        diagnostics["dedup_self"] = report_self.stats

        # 7b: Cross-source dedup (det vs LLM)
        final_features, report_cross = dedup_cross_source(
            det_features, llm_deduped, thresholds=self.config.dedup_cross_thresholds
        )
        diagnostics["dedup_cross"] = report_cross.stats

        # Sliver filter (final pass)
        final_features, sliver_removed = filter_sliver_features(
            final_features,
            min_area_deg=self.config.sliver_min_area_deg,
            min_length_deg=self.config.sliver_min_length_deg,
            min_points=self.config.sliver_min_points,
        )
        diagnostics["sliver_removed"] = len(sliver_removed)

        # Step 8: Build final FeatureCollection
        fc = build_feature_collection(final_features)
        diagnostics["final_feature_count"] = len(final_features)

        return fc, diagnostics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_prefix(prompt: str) -> str:
        prompt_upper = prompt.upper()
        if "[CAP]" in prompt_upper:
            return "CAP"
        if "[VQA]" in prompt_upper:
            return "VQA"
        if "[DET]" in prompt_upper:
            return "DET"
        return "DET"  # default

    @staticmethod
    def _build_fallback_prompt(original_prompt: str, unknown_classes: List[str]) -> str:
        if "*" in unknown_classes:
            # Wildcard: LLM generates freely from original prompt
            return original_prompt
        class_list = "、".join(unknown_classes)
        original_context = original_prompt[:200] + "..." if len(original_prompt) > 200 else original_prompt
        return (
            f"[DET] 请检测图中的{class_list}。"
            f" Original request: {original_context}. "
            f"Output the extracted feature information as a GeoJSON "
            f"FeatureCollection. Return JSON only."
        )

    def _get_semantic_class_names(self) -> List[str]:
        """Get all semantic branch class names from label_map."""
        classes = self.label_map.get("branch_classes", {}).get("semantic", {})
        return [n for n in classes.values() if n and n != "background"]

    def _get_all_known_class_names(self) -> List[str]:
        """Get all known class names across all branches."""
        names = []
        for branch in ("instance", "semantic", "edge"):
            classes = self.label_map.get("branch_classes", {}).get(branch, {})
            names.extend(n for n in classes.values() if n and n != "background")
        return names
