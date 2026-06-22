from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    ROOT / "configs" / "poc_aqua_instance.yaml",
    ROOT / "configs" / "poc_aqua_128.yaml",
    ROOT / "configs" / "poc_aqua_512.yaml",
    ROOT / "configs" / "poc_aqua_multisize.yaml",
]


def _anchor_counts_per_level(anchor_cfg):
    if anchor_cfg.get("mode") == "relative":
        size_levels = anchor_cfg["scales"]
    else:
        size_levels = anchor_cfg["sizes"]
    ratio_levels = anchor_cfg["aspect_ratios"]
    assert len(size_levels) == len(ratio_levels)
    return [
        len(size_level) * len(ratio_level)
        for size_level, ratio_level in zip(size_levels, ratio_levels)
    ]


def test_poc1_anchor_counts_match_rpn_head_assumption():
    offenders = {}
    for path in CONFIGS:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        counts = _anchor_counts_per_level(cfg["model"]["anchors"])
        if len(set(counts)) != 1:
            offenders[path.name] = counts
    assert offenders == {}
