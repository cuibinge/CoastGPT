import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Tools.probe_stage2_generate_scores import compare_label_logits, pick_label_from_logits


def test_pick_label_from_logits_uses_configured_label_token_ids():
    logits = torch.tensor([[0.0, 2.0, -1.0, 5.0]])
    label_token_ids = {"urban": 1, "rural": 3}

    picked = pick_label_from_logits(logits, label_token_ids)

    assert picked["pred_label"] == "rural"
    assert picked["scores"] == {"urban": 2.0, "rural": 5.0}


def test_compare_label_logits_reports_match_and_score_deltas():
    manual_logits = torch.tensor([[0.0, 3.0, -1.0, 2.0]])
    generate_logits = torch.tensor([[0.0, 2.5, -1.0, 2.1]])
    label_token_ids = {"urban": 1, "rural": 3}

    comparison = compare_label_logits(manual_logits, generate_logits, label_token_ids)

    assert comparison["manual_pred_label"] == "urban"
    assert comparison["generate_score_pred_label"] == "urban"
    assert comparison["pred_match"] is True
    assert comparison["label_score_abs_diff"] == {"urban": 0.5, "rural": 0.1}
