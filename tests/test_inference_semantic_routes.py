import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Tools.run_geojson_batch_eval import _infer_semantic_route_texts


def test_lr_prompt_infers_classification_land_cover_route():
    task_text, element_text = _infer_semantic_route_texts("Is it a rural or an urban area")

    assert task_text == "场景分类"
    assert element_text == "土地覆盖"


def test_vg_prompt_infers_visual_grounding_route():
    task_text, element_text = _infer_semantic_route_texts("[VG] locate the airplane in the image")

    assert task_text == "视觉定位"
    assert element_text == "airplane"


if __name__ == "__main__":
    test_lr_prompt_infers_classification_land_cover_route()
    test_vg_prompt_infers_visual_grounding_route()
