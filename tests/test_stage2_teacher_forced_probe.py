import sys
from pathlib import Path

import torch
from ml_collections import ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Tools.probe_stage2_teacher_forced_vs_generate import (
    IGNORE_INDEX,
    first_label_index,
    resolve_dataset_stage,
    summarize_label_pairs,
)


def test_first_label_index_returns_first_supervised_token():
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 42, 43]])

    assert first_label_index(labels) == 2


def test_first_label_index_returns_none_when_all_ignored():
    labels = torch.full((1, 4), IGNORE_INDEX)

    assert first_label_index(labels) is None


def test_summarize_label_pairs_reports_confusion_and_balanced_accuracy():
    summary = summarize_label_pairs(
        [
            ("urban", "urban"),
            ("urban", "rural"),
            ("rural", "rural"),
            ("rural", "rural"),
        ]
    )

    assert summary["total"] == 4
    assert summary["label_exact_rate_pct"] == 75.0
    assert summary["label_balanced_accuracy_pct"] == 75.0
    assert summary["label_recall_by_answer"] == {"rural": 100.0, "urban": 50.0}
    assert summary["label_confusion"] == {
        "rural": {"rural": 2},
        "urban": {"rural": 1, "urban": 1},
    }


def test_resolve_dataset_stage_keeps_stage2_before_inference_mutation():
    config = ConfigDict({"stage": 2})

    stage = resolve_dataset_stage(config)
    config.stage = 0

    assert stage == 2
