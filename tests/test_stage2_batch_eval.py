import importlib.util
import sys
from pathlib import Path


def _load_stage2_eval_module():
    script_path = Path(__file__).resolve().parents[1] / "Tools" / "run_stage2_batch_eval.py"
    spec = importlib.util.spec_from_file_location("run_stage2_batch_eval", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flatten_stage2_entries_expands_conv_list(tmp_path):
    import json

    module = _load_stage2_eval_module()

    dataset_path = tmp_path / "LR.json"
    image_root = tmp_path / "LR_Image"
    image_root.mkdir()
    (image_root / "0.tif").write_bytes(b"placeholder")
    dataset_path.write_text(
        json.dumps(
            {
                "train": [
                    {
                        "name": "0.tif",
                        "conv": [
                            {"Question": "Is it rural?", "Answer": "yes"},
                            {"Question": "How many buildings?", "Answer": "3"},
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    entries = module.flatten_stage2_entries(
        dataset_path=dataset_path,
        image_root=image_root,
        max_questions_per_image=1,
    )

    assert len(entries) == 1
    assert entries[0]["sample_id"] == "0.tif#0:0"
    assert entries[0]["image_path"] == image_root / "0.tif"
    assert entries[0]["question"] == "Is it rural?"
    assert entries[0]["answer"] == "yes"


def test_score_answer_reports_exact_and_contains():
    module = _load_stage2_eval_module()

    exact = module.score_answer("Yes.", "yes")
    contains = module.score_answer("The answer is urban area.", "urban")
    miss = module.score_answer("rural", "urban")

    assert exact["exact"] is True
    assert exact["contains"] is True
    assert contains["exact"] is False
    assert contains["contains"] is True
    assert miss["exact"] is False
    assert miss["contains"] is False


def test_score_answer_extracts_first_binary_label_for_classification():
    module = _load_stage2_eval_module()

    scores = module.score_answer("rural \n\nIs there a river nearby", "rural")
    miss = module.score_answer("urban \n\nA bridge is visible.", "rural")

    assert scores["label_prediction"] == "rural"
    assert scores["label_exact"] is True
    assert miss["label_prediction"] == "urban"
    assert miss["label_exact"] is False


def test_summarize_reports_label_exact_rate():
    module = _load_stage2_eval_module()

    summary = module.summarize(
        [
            {"exact": False, "contains": True, "label_exact": True, "all_unk": False, "unk_ratio": 0.0},
            {"exact": False, "contains": False, "label_exact": False, "all_unk": False, "unk_ratio": 0.0},
        ]
    )

    assert summary["label_exact_count"] == 1
    assert summary["label_exact_rate"] == 0.5


def test_summarize_reports_label_confusion_and_balanced_accuracy():
    module = _load_stage2_eval_module()

    summary = module.summarize(
        [
            {"label_answer": "rural", "label_prediction": "rural", "label_exact": True, "all_unk": False, "unk_ratio": 0.0},
            {"label_answer": "rural", "label_prediction": "urban", "label_exact": False, "all_unk": False, "unk_ratio": 0.0},
            {"label_answer": "urban", "label_prediction": "rural", "label_exact": False, "all_unk": False, "unk_ratio": 0.0},
        ]
    )

    assert summary["label_confusion"] == {
        "rural": {"rural": 1, "urban": 1},
        "urban": {"rural": 1},
    }
    assert summary["label_recall_by_answer"] == {
        "rural": 0.5,
        "urban": 0.0,
    }
    assert summary["label_balanced_accuracy"] == 0.25


def test_resolve_image_path_accepts_zero_padded_dior_ids(tmp_path):
    import json

    module = _load_stage2_eval_module()

    dataset_path = tmp_path / "RSVG_DIOR.json"
    image_root = tmp_path / "RSVG_DIOR_Image"
    image_root.mkdir()
    (image_root / "00003.jpg").write_bytes(b"placeholder")
    dataset_path.write_text(json.dumps({"data": [{"img": "3", "question": "[VG] object", "answer": "[0,0,1,1]"}]}))

    path = module._resolve_image_path({"img": "3"}, dataset_path, image_root)

    assert path == image_root / "00003.jpg"


def test_resolve_image_path_uses_naip_png_inside_meterml_directory(tmp_path):
    module = _load_stage2_eval_module()

    dataset_path = tmp_path / "METERML.json"
    image_root = tmp_path / "METERML_Image"
    sample_dir = image_root / "36.1_-75.2"
    sample_dir.mkdir(parents=True)
    (sample_dir / "naip.png").write_bytes(b"placeholder")

    path = module._resolve_image_path({"name": "36.1_-75.2"}, dataset_path, image_root)

    assert path == sample_dir / "naip.png"


def test_flatten_stage2_entries_keeps_duplicate_image_items_unique(tmp_path):
    import json

    module = _load_stage2_eval_module()

    dataset_path = tmp_path / "RSVG.json"
    image_root = tmp_path / "RSVG_Image"
    image_root.mkdir()
    (image_root / "same.jpg").write_bytes(b"placeholder")
    dataset_path.write_text(
        json.dumps(
            [
                {"img": "same.jpg", "question": "[VG] first", "answer": "[0,0,1,1]"},
                {"img": "same.jpg", "question": "[VG] second", "answer": "[0,0,1,1]"},
            ]
        ),
        encoding="utf-8",
    )

    entries = module.flatten_stage2_entries(dataset_path=dataset_path, image_root=image_root)

    assert [entry["sample_id"] for entry in entries] == ["same.jpg#0:0", "same.jpg#1:0"]


def test_score_answer_computes_vg_iou():
    module = _load_stage2_eval_module()

    scores = module.score_answer("[0.0, 0.0, 0.5, 0.5]", "[0,0,1,1]", task_type="vg")

    assert scores["task_type"] == "vg"
    assert scores["vg_iou"] == 0.25
    assert scores["vg_acc_at_025"] is True
    assert scores["vg_acc_at_05"] is False
    assert scores["vg_ap_at_05"] == 0.0


def test_infer_task_type_treats_urban_rural_prompt_as_cls():
    module = _load_stage2_eval_module()

    task_type = module.infer_task_type("Is it a rural or an urban area", "urban")

    assert task_type == "cls"


def test_score_answer_computes_caption_rouge_l_and_token_f1():
    module = _load_stage2_eval_module()

    scores = module.score_answer("green trees near river", "many green trees are near a river", task_type="cap")

    assert scores["task_type"] == "cap"
    assert round(scores["cap_token_f1"], 4) == 0.8
    assert round(scores["cap_rouge_l"], 4) == 0.6667
    assert "label_exact" not in scores


def test_rescore_existing_record_drops_stale_label_metrics_for_cap_vg():
    module = _load_stage2_eval_module()

    record = {
        "sample_id": "image#0",
        "prediction": "[0.0, 0.0, 0.5, 0.5]",
        "task_type": "vg",
        "label_answer": "stale",
        "label_prediction": "stale",
        "label_exact": True,
        "all_unk": False,
        "unk_ratio": 0.0,
    }

    rescored = module.rescore_existing_record(
        record,
        answer="[0,0,1,1]",
        label_choices=[],
        task_type="vg",
    )

    assert "label_answer" not in rescored
    assert "label_prediction" not in rescored
    assert "label_exact" not in rescored
    assert rescored["vg_iou"] == 0.25


def test_summarize_reports_metrics_by_task():
    module = _load_stage2_eval_module()

    summary = module.summarize(
        [
            {"task_type": "vg", "vg_iou": 0.75, "vg_acc_at_05": True, "all_unk": False, "unk_ratio": 0.0},
            {"task_type": "vg", "vg_iou": 0.25, "vg_acc_at_05": False, "all_unk": False, "unk_ratio": 0.0},
            {"task_type": "cap", "cap_token_f1": 0.5, "cap_rouge_l": 0.25, "all_unk": False, "unk_ratio": 0.0},
        ]
    )

    assert summary["by_task"]["vg"]["total"] == 2
    assert summary["by_task"]["vg"]["vg_mean_iou"] == 0.5
    assert summary["by_task"]["vg"]["vg_acc_at_025_rate"] == 1.0
    assert summary["by_task"]["vg"]["vg_acc_at_05_rate"] == 0.5
    assert summary["by_task"]["vg"]["vg_ap_at_05"] == 0.5
    assert summary["by_task"]["cap"]["total"] == 1
    assert summary["by_task"]["cap"]["cap_token_f1"] == 0.5
    assert summary["by_task"]["cap"]["cap_rouge_l"] == 0.25


def test_parse_args_allows_precision_override(tmp_path, monkeypatch):
    module = _load_stage2_eval_module()
    config_path = tmp_path / "stage2.yaml"
    config_path.write_text("dtype: float16\nfp16: true\nbf16: false\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_stage2_batch_eval.py",
            "-c",
            str(config_path),
            "--dataset-json",
            str(tmp_path / "LR.json"),
            "--output-dir",
            str(tmp_path / "out"),
            "--model-path",
            "checkpoint.pt",
            "--dtype",
            "bfloat16",
            "--fp16",
            "false",
            "--bf16",
            "true",
        ],
    )

    config = module.parse_args()

    assert config.dtype == "bfloat16"
    assert config.fp16 is False
    assert config.bf16 is True
