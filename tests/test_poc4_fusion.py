"""PoC-4 FusionPipeline smoke test — uses mock predictors, no NPU required."""
import json
import sys
from pathlib import Path

# Add repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from Models.fusion_pipeline import FusionConfig, FusionPipeline, Gating, DispatchMap
from Models.fusion_predictors import BasePredictor, PredictorOutput
from utils.geojson_builder import build_feature_collection, validate_geojson
from utils.geojson_dedup import dedup_within_source, dedup_cross_source
from utils.geojson_validator import validate_llm_fallback


# --- Mock predictors ---

class MockInstancePredictor(BasePredictor):
    @property
    def branch(self) -> str:
        return "instance"

    def predict(self, image, prompt, georef, classes):
        if "海水养殖区" not in classes:
            return PredictorOutput(branch=self.branch, features=[], metadata={"skipped": True})
        feat = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[119.3001, 35.0701], [119.3010, 35.0701],
                                 [119.3010, 35.0710], [119.3001, 35.0710],
                                 [119.3001, 35.0701]]]
            },
            "properties": {"class": "海水养殖区", "confidence": 0.95}
        }
        return PredictorOutput(branch=self.branch, features=[feat], metadata={"num_features": 1})


class MockEdgePredictor(BasePredictor):
    @property
    def branch(self) -> str:
        return "edge"

    def predict(self, image, prompt, georef, classes):
        if "海岸线" not in classes:
            return PredictorOutput(branch=self.branch, features=[], metadata={"skipped": True})
        feat = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[119.3005, 35.0705], [119.3015, 35.0715]]
            },
            "properties": {"class": "海岸线", "confidence": 0.85}
        }
        return PredictorOutput(branch=self.branch, features=[feat], metadata={"num_features": 1})


class MockLLMPredictor(BasePredictor):
    @property
    def branch(self) -> str:
        return "llm"

    def predict(self, image, prompt, georef, classes):
        features = []
        for cls_name in classes:
            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[119.3020, 35.0720], [119.3030, 35.0720],
                                     [119.3030, 35.0730], [119.3020, 35.0730],
                                     [119.3020, 35.0720]]]
                },
                "properties": {"class": cls_name, "confidence": 0.70}
            }
            features.append(feat)
        return PredictorOutput(branch=self.branch, features=features, metadata={"num_features": len(features)})


class MockLLMTextPredictor(BasePredictor):
    @property
    def branch(self) -> str:
        return "llm_text"

    def predict(self, image, prompt, georef, classes):
        return PredictorOutput(
            branch=self.branch, features=[],
            metadata={"num_features": 0},
            raw={"text": "这是一张遥感影像。"},
        )


class MockLLMParser:
    """Mock parser that returns pre-determined results."""
    def __init__(self, parse_result):
        self.parse_result = parse_result

    def parse(self, prompt, image):
        return self.parse_result


# --- Tests ---

def test_gating():
    """Test Gating routes classes correctly."""
    label_map = json.loads(Path("Configs/label_map.json").read_text())
    g = Gating(label_map)
    dm = g.route(["海水养殖区", "海岸线", "红树林"])
    assert dm.known["instance"] == ["海水养殖区"]
    assert dm.known["edge"] == ["海岸线"]
    assert dm.unknown == ["红树林"]
    print("test_gating: PASS")


def test_dedup_internal():
    """Test LLM self-dedup merges overlapping features."""
    f1 = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
        },
        "properties": {"class": "红树林", "confidence": 0.9}
    }
    f2 = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]]]
        },
        "properties": {"class": "红树林", "confidence": 0.5}
    }
    kept, report = dedup_within_source([f1, f2], thresholds={"area": 0.1, "line": 0.6, "point": 5e-6})
    assert report.stats["n_kept"] == 1, f"Expected 1 kept, got {report.stats['n_kept']}"
    print(f"test_dedup_internal: PASS (kept={report.stats['n_kept']})")


def test_dedup_cross_known_wins():
    """Test cross-source dedup: known feature survives conflict."""
    det_f = [{
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
        },
        "properties": {"class": "海水养殖区", "confidence": 0.9}
    }]
    llm_f = [{
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9], [0.1, 0.1]]]
        },
        "properties": {"class": "海水养殖区", "confidence": 0.7}
    }]
    merged, report = dedup_cross_source(det_f, llm_f, thresholds={"area": 0.3, "line": 0.5, "point": 1e-5})
    assert len(merged) == 1, f"Expected 1 feature (known wins), got {len(merged)}"
    assert merged[0]["properties"]["class"] == "海水养殖区"
    assert merged[0]["properties"]["confidence"] == 0.9, "Known feature should survive unchanged"
    assert report.stats["n_cross_removed"] == 1
    print("test_dedup_cross_known_wins: PASS")


def test_validator_whitelist():
    """Test Layer 2 validator catches whitelist violations."""
    f1 = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        "properties": {"class": "海岸线", "confidence": 0.9}
    }
    valid, report = validate_llm_fallback(
        [f1], ["红树林"],
        config={"known_classes": ["海岸线", "海水养殖区"], "min_confidence": 0.3}
    )
    assert len(valid) == 0, "Coastline from LLM should be rejected (known class)"
    assert report.stats["n_whitelist_violation"] == 1
    print("test_validator_whitelist: PASS")


def test_validator_low_confidence():
    """Test Layer 2 validator filters low confidence."""
    f1 = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        "properties": {"class": "红树林", "confidence": 0.1}
    }
    valid, report = validate_llm_fallback(
        [f1], ["红树林"],
        config={"known_classes": ["海岸线"], "min_confidence": 0.3}
    )
    assert len(valid) == 0, "Low confidence should be filtered"
    assert report.stats["n_confidence_drop"] == 1
    print("test_validator_low_confidence: PASS")


def test_fusion_pipeline_known_only():
    """Test FusionPipeline with known classes only (no LLM fallback)."""
    label_map = json.loads(Path("Configs/label_map.json").read_text())
    config = FusionConfig(label_map_path="Configs/label_map.json")

    from Models.fusion_pipeline import ParseResult
    mock_parse = ParseResult(
        task_type="DET",
        target_classes=["海水养殖区", "海岸线"],
        confidence=0.95,
        source="mock",
    )

    # Build pipeline with mock parser
    pipeline = FusionPipeline(
        predictors={
            "instance": MockInstancePredictor(),
            "semantic": MockEdgePredictor(),  # unused in this test
            "edge": MockEdgePredictor(),
            "llm": MockLLMPredictor(),
            "llm_text": MockLLMTextPredictor(),
        },
        config=config,
        label_map=label_map,
    )
    # Override parser with mock
    pipeline.parser = MockLLMParser(mock_parse)

    georef = {
        "source_crs": "EPSG:4326",
        "tile_bounds_wgs84": [119.30, 35.07, 119.31, 35.08],
    }
    image = torch.randn(3, 224, 224)
    fc, diag = pipeline.run(image, "[DET] 检测海岸线和养殖区", georef)

    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) >= 2, f"Expected at least 2 features, got {len(fc['features'])}"
    assert diag["parse"]["source"] == "mock"
    assert len(diag["dispatch"]["unknown"]) == 0, "No unknown classes expected"
    assert diag["llm_feature_count_before_dedup"] == 0, "LLM fallback should be skipped"
    print(f"test_fusion_pipeline_known_only: PASS ({len(fc['features'])} features)")
    # Verify JSON can be serialized
    json.dumps(fc)
    print("  JSON serialization: OK")


def test_fusion_pipeline_with_unknown():
    """Test FusionPipeline with unknown class triggering LLM fallback."""
    label_map = json.loads(Path("Configs/label_map.json").read_text())
    config = FusionConfig(label_map_path="Configs/label_map.json")

    from Models.fusion_pipeline import ParseResult
    mock_parse = ParseResult(
        task_type="DET",
        target_classes=["海水养殖区", "红树林"],
        confidence=0.90,
        source="mock",
    )

    pipeline = FusionPipeline(
        predictors={
            "instance": MockInstancePredictor(),
            "semantic": MockEdgePredictor(),
            "edge": MockEdgePredictor(),
            "llm": MockLLMPredictor(),
            "llm_text": MockLLMTextPredictor(),
        },
        config=config,
        label_map=label_map,
    )
    pipeline.parser = MockLLMParser(mock_parse)

    georef = {
        "source_crs": "EPSG:4326",
        "tile_bounds_wgs84": [119.30, 35.07, 119.31, 35.08],
    }
    image = torch.randn(3, 224, 224)
    fc, diag = pipeline.run(image, "[DET] 检测养殖区和红树林", georef)

    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) >= 1
    assert diag["dispatch"]["unknown"] == ["红树林"]
    assert diag["llm_feature_count_before_dedup"] >= 1, "LLM fallback should produce features"
    print(f"test_fusion_pipeline_with_unknown: PASS ({len(fc['features'])} features)")
    print(f"  Det: {diag['det_feature_count']}, LLM: {diag['llm_feature_count_before_dedup']}, Final: {diag['final_feature_count']}")


def test_cap_early_return():
    """Test that [CAP] prompts skip all detection heads."""
    label_map = json.loads(Path("Configs/label_map.json").read_text())
    config = FusionConfig(label_map_path="Configs/label_map.json")

    pipeline = FusionPipeline(
        predictors={
            "instance": MockInstancePredictor(),
            "semantic": MockEdgePredictor(),
            "edge": MockEdgePredictor(),
            "llm": MockLLMPredictor(),
            "llm_text": MockLLMTextPredictor(),
        },
        config=config,
        label_map=label_map,
    )

    image = torch.randn(3, 224, 224)
    fc, diag = pipeline.run(image, "[CAP] 描述这张图", {"source_crs": "EPSG:4326"})

    assert diag["early_return"] == "CAP"
    assert diag["text_output"] == "这是一张遥感影像。"
    print("test_cap_early_return: PASS")


if __name__ == "__main__":
    test_gating()
    test_dedup_internal()
    test_dedup_cross_known_wins()
    test_validator_whitelist()
    test_validator_low_confidence()
    test_fusion_pipeline_known_only()
    test_fusion_pipeline_with_unknown()
    test_cap_early_return()
    print("\n=== All PoC-4 integration tests passed! ===")
