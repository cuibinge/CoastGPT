import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.semantic_bg_prior import (
    build_pseudo_background_mask,
    scheduled_background_lambda,
    selective_pseudo_background_bce_loss,
)


def _logits_from_probs(probs):
    return torch.log(torch.tensor(probs, dtype=torch.float32)).permute(2, 0, 1).unsqueeze(0)


def test_pseudo_background_mask_selects_confident_ignore_background():
    target = torch.full((1, 3, 3), 255, dtype=torch.long)
    probs = torch.zeros(3, 3, 3)
    probs[..., 0] = 0.9
    probs[..., 1] = 0.05
    probs[..., 2] = 0.05
    logits = torch.log(probs).permute(2, 0, 1).unsqueeze(0)

    mask = build_pseudo_background_mask(
        logits,
        target,
        p_bg_threshold=0.7,
        entropy_threshold=0.5,
        protect_radius_px=0,
    )

    assert mask.shape == target.shape
    assert mask.all()


def test_pseudo_background_mask_protects_pixels_near_labeled_foreground():
    target = torch.full((1, 5, 5), 255, dtype=torch.long)
    target[0, 2, 2] = 4
    probs = torch.zeros(5, 5, 3)
    probs[..., 0] = 0.9
    probs[..., 1] = 0.05
    probs[..., 2] = 0.05
    logits = torch.log(probs).permute(2, 0, 1).unsqueeze(0)

    mask = build_pseudo_background_mask(
        logits,
        target,
        p_bg_threshold=0.7,
        entropy_threshold=0.5,
        protect_radius_px=1,
    )

    assert not mask[0, 1:4, 1:4].any()
    assert mask[0, 0, 0]
    assert mask[0, 4, 4]


def test_pseudo_background_mask_rejects_uncertain_ignore_pixels():
    target = torch.full((1, 2, 2), 255, dtype=torch.long)
    logits = _logits_from_probs(
        [
            [[0.4, 0.3, 0.3], [0.9, 0.05, 0.05]],
            [[0.4, 0.3, 0.3], [0.9, 0.05, 0.05]],
        ]
    )

    mask = build_pseudo_background_mask(
        logits,
        target,
        p_bg_threshold=0.7,
        entropy_threshold=0.5,
        protect_radius_px=0,
    )

    assert not mask[0, 0, 0]
    assert mask[0, 0, 1]
    assert not mask[0, 1, 0]
    assert mask[0, 1, 1]


def test_selective_pseudo_background_bce_reports_candidate_stats():
    target = torch.full((1, 5, 5), 255, dtype=torch.long)
    target[0, 2, 2] = 2
    probs = torch.zeros(5, 5, 3)
    probs[..., 0] = 0.9
    probs[..., 1] = 0.05
    probs[..., 2] = 0.05
    logits = torch.log(probs).permute(2, 0, 1).unsqueeze(0).requires_grad_(True)

    loss, stats = selective_pseudo_background_bce_loss(
        logits,
        target,
        p_bg_threshold=0.7,
        entropy_threshold=0.5,
        protect_radius_px=1,
        max_ignore_ratio=1.0,
    )

    assert loss.item() > 0
    assert stats["pseudo_bg_candidate_pixel_count"] == 16
    assert stats["pseudo_bg_active_tile_ratio"] == 1.0


def test_scheduled_background_lambda_warms_up_then_ramps():
    assert scheduled_background_lambda(0.2, epoch=2, warmup_epochs=2, ramp_epochs=3) == 0.0
    assert scheduled_background_lambda(0.2, epoch=3, warmup_epochs=2, ramp_epochs=3) == 0.2 / 3
    assert scheduled_background_lambda(0.2, epoch=5, warmup_epochs=2, ramp_epochs=3) == 0.2
    assert scheduled_background_lambda(0.2, epoch=10, warmup_epochs=2, ramp_epochs=3) == 0.2
