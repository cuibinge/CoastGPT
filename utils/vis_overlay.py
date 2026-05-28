"""
Pixel-space overlay visualization for GT and predicted instance masks/boxes.

Used by the PoC-1 aquaculture instance detection pipeline to render
bounding boxes and segmentation masks on satellite image patches (224x224).
All coordinates are in model pixel space — no WGS84 conversion is performed here.
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_boxes(draw, boxes, color, width=2):
    """Draw bounding boxes on PIL ImageDraw."""
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def _draw_masks(image, masks, color, alpha=0.4):
    """Overlay masks on numpy image with alpha blending. Modifies image in-place."""
    for mask in masks:
        mask_bool = mask > 0.5
        if mask_bool.any():
            overlay = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
            image[mask_bool] = (
                image[mask_bool] * (1 - alpha) + overlay * alpha
            ).astype(np.uint8)


# ---------------------------------------------------------------------------
# Internal: single-view overlay helper
# ---------------------------------------------------------------------------

def _overlay_single(
    image_tensor,
    boxes,
    masks,
    output_path,
    mask_color,
    box_color,
    scores=None,
    label="",
):
    """Single-view overlay: masks first (numpy), then boxes + text (PIL).

    Parameters
    ----------
    image_tensor : np.ndarray
        Image in [C, H, W] float32 [0,1] or [H, W, C] uint8.
    boxes : np.ndarray or tensor-like
        [N, 4] boxes in (x1, y1, x2, y2) pixel coordinates.
    masks : np.ndarray or tensor-like
        [N, H, W] binary masks.
    output_path : str
        File path to save the rendered PNG.
    mask_color : tuple of int
        RGB tuple for mask overlay, e.g. (0, 255, 0).
    box_color : str or tuple
        Colour for box outlines / text (passed directly to PIL).
    scores : np.ndarray or tensor-like, optional
        [N,] confidence scores to display above each box.
    label : str
        Log label printed after saving.
    """
    # Convert [C,H,W] float → [H,W,C] uint8 if needed
    if image_tensor.ndim == 3 and image_tensor.shape[0] == 3:
        img = (image_tensor.transpose(1, 2, 0) * 255).astype(np.uint8)
    else:
        img = image_tensor.copy()

    # Handle tensor inputs (may have .cpu() method)
    boxes_np = None
    masks_np = None
    scores_np = None
    if boxes is not None and len(boxes) > 0:
        boxes_np = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.array(boxes)
    if masks is not None and len(masks) > 0:
        masks_np = masks.cpu().numpy() if hasattr(masks, "cpu") else np.array(masks)

    # Draw masks on numpy array FIRST (under boxes)
    if masks_np is not None:
        _draw_masks(img, masks_np[:5], mask_color, alpha=0.3)

    # Create PIL image and draw boxes + text
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    if boxes_np is not None:
        _draw_boxes(draw, boxes_np, box_color)
        if scores is not None:
            scores_np = scores.cpu().numpy() if hasattr(scores, "cpu") else np.array(scores)
            for box, score in zip(boxes_np, scores_np):
                draw.text(
                    (box[0], max(0, box[1] - 8)),
                    f"{score:.2f}",
                    fill=box_color,
                )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img_pil.save(output_path)
    print(f"  [{label}] Saved overlay to {output_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def overlay_gt_pixel(image_tensor, target, output_path, mask_color=(0, 255, 0), box_color="green"):
    """Overlay GT masks (green) and boxes on the original image, save to output_path.

    Parameters
    ----------
    image_tensor : np.ndarray
        Image in [C, H, W] float32 [0,1] or [H, W, C] uint8.
    target : dict
        Target dictionary with keys ``"masks"`` and ``"boxes"``.
    output_path : str
        Save path for the rendered PNG.
    mask_color : tuple
        RGB colour for the mask overlay.
    box_color : str or tuple
        Colour for box outlines.
    """
    _overlay_single(
        image_tensor=image_tensor,
        boxes=target["boxes"],
        masks=target["masks"],
        output_path=output_path,
        mask_color=mask_color,
        box_color=box_color,
        label="GT",
    )


def overlay_pred_pixel(
    image_tensor, boxes, masks, scores, output_path, mask_color=(255, 0, 0), box_color="red"
):
    """Overlay predicted masks (red) and boxes with score labels.

    Parameters
    ----------
    image_tensor : np.ndarray
        Image in [C, H, W] float32 [0,1] or [H, W, C] uint8.
    boxes : np.ndarray or tensor-like
        [N, 4] predicted boxes in pixel coordinates.
    masks : np.ndarray or tensor-like
        [N, H, W] predicted binary masks.
    scores : np.ndarray or tensor-like
        [N,] confidence scores.
    output_path : str
        Save path for the rendered PNG.
    mask_color : tuple
        RGB colour for the mask overlay.
    box_color : str or tuple
        Colour for box outlines and score text.
    """
    _overlay_single(
        image_tensor=image_tensor,
        boxes=boxes,
        masks=masks,
        output_path=output_path,
        mask_color=mask_color,
        box_color=box_color,
        scores=scores,
        label="Pred",
    )


def overlay_gt_pred_pixel(
    image_tensor, target, pred_boxes, pred_masks, pred_scores, output_path
):
    """Side-by-side comparison: GT (green) vs Pred (red) on a single image.

    Mask draw order: GT masks first (green), then pred masks (red)
    so pred masks appear on top.  Boxes: GT green, Pred red.

    Parameters
    ----------
    image_tensor : np.ndarray
        Image in [C, H, W] float32 [0,1] or [H, W, C] uint8.
    target : dict
        GT dictionary with ``"masks"`` and ``"boxes"``.
    pred_boxes : np.ndarray or tensor-like
        [N, 4] predicted boxes.
    pred_masks : np.ndarray or tensor-like
        [N, H, W] predicted masks.
    pred_scores : np.ndarray or tensor-like
        [N,] confidence scores.
    output_path : str
        Save path for the rendered PNG.
    """
    # Convert [C,H,W] float → [H,W,C] uint8 if needed
    if image_tensor.ndim == 3 and image_tensor.shape[0] == 3:
        img = (image_tensor.transpose(1, 2, 0) * 255).astype(np.uint8)
    else:
        img = image_tensor.copy()

    # Helper for tensor→numpy conversion
    def _to_np(x):
        if x is None or len(x) == 0:
            return None
        return x.cpu().numpy() if hasattr(x, "cpu") else np.array(x)

    gt_boxes_np = _to_np(target.get("boxes"))
    gt_masks_np = _to_np(target.get("masks"))
    pred_boxes_np = _to_np(pred_boxes)
    pred_masks_np = _to_np(pred_masks)
    pred_scores_np = _to_np(pred_scores)

    # Draw GT masks first (green), then pred masks (red) on numpy array
    if gt_masks_np is not None:
        _draw_masks(img, gt_masks_np[:5], (0, 255, 0), alpha=0.3)
    if pred_masks_np is not None:
        _draw_masks(img, pred_masks_np[:5], (255, 0, 0), alpha=0.3)

    # Create PIL image and draw boxes + text
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    # GT boxes in green
    if gt_boxes_np is not None:
        _draw_boxes(draw, gt_boxes_np, "green")

    # Pred boxes in red with scores
    if pred_boxes_np is not None:
        _draw_boxes(draw, pred_boxes_np, "red")
        if pred_scores_np is not None:
            for box, score in zip(pred_boxes_np, pred_scores_np):
                draw.text(
                    (box[0], max(0, box[1] - 8)),
                    f"{score:.2f}",
                    fill="red",
                )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img_pil.save(output_path)
    print(f"  [GT+Pred] Saved overlay to {output_path}")


def save_overlay_grid(images, targets, pred_outputs, output_dir, sample_ids=None):
    """Generate overlay images for a batch of samples.

    For each sample saves ``{sample_id}_gt.png``, and if *pred_outputs* is
    provided also ``{sample_id}_pred.png`` and ``{sample_id}_gt_pred.png``.

    Parameters
    ----------
    images : list of np.ndarray
        List of images as [H, W, C] uint8 numpy arrays.
    targets : list of dict
        List of GT target dicts, each containing ``"boxes"`` and ``"masks"``.
    pred_outputs : list of dict or None
        List of prediction dicts.  Each dict should have keys ``"boxes"``,
        ``"masks"``, ``"scores"``.  If ``None``, only GT overlays are saved.
    output_dir : str
        Output directory path.
    sample_ids : list of str, optional
        Sample identifiers.  Defaults to ``sample_0000``, ``sample_0001``, ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sample_ids is None:
        sample_ids = [f"sample_{i:04d}" for i in range(len(images))]

    for idx, (img, target, sid) in enumerate(zip(images, targets, sample_ids)):
        # GT overlay
        overlay_gt_pixel(img, target, str(output_dir / f"{sid}_gt.png"))

        if pred_outputs is not None:
            pred = pred_outputs[idx]
            # Pred overlay
            overlay_pred_pixel(
                img,
                pred["boxes"],
                pred["masks"],
                pred["scores"],
                str(output_dir / f"{sid}_pred.png"),
            )
            # GT+Pred overlay
            overlay_gt_pred_pixel(
                img,
                target,
                pred["boxes"],
                pred["masks"],
                pred["scores"],
                str(output_dir / f"{sid}_gt_pred.png"),
            )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    # Create synthetic image: 224x224 RGB gradient (C, H, W float32)
    img = np.zeros((3, 224, 224), dtype=np.float32)
    img[0, :, :] = np.linspace(0, 1, 224).reshape(1, -1)  # R gradient
    img[1, 50:180, 40:160] = 0.8  # G block
    img[2, :, :] = 0.3  # B constant

    # Synthetic target
    target = {
        "boxes": np.array([[40, 50, 160, 180]], dtype=np.float32),
        "masks": np.zeros((1, 224, 224), dtype=np.uint8),
    }
    target["masks"][0, 50:180, 40:160] = 1

    # Synthetic predictions
    pred_boxes = np.array([[45, 55, 155, 175]], dtype=np.float32)
    pred_masks = np.zeros((1, 224, 224), dtype=np.uint8)
    pred_masks[0, 55:175, 45:155] = 1
    pred_scores = np.array([0.92], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test GT overlay
        overlay_gt_pixel(img, target, f"{tmpdir}/test_gt.png")
        assert Path(f"{tmpdir}/test_gt.png").exists()

        # Test pred overlay
        overlay_pred_pixel(img, pred_boxes, pred_masks, pred_scores, f"{tmpdir}/test_pred.png")
        assert Path(f"{tmpdir}/test_pred.png").exists()

        # Test GT+Pred overlay
        overlay_gt_pred_pixel(
            img, target, pred_boxes, pred_masks, pred_scores, f"{tmpdir}/test_gt_pred.png"
        )
        assert Path(f"{tmpdir}/test_gt_pred.png").exists()

        # Test save_overlay_grid (no pred_outputs)
        save_overlay_grid([img.copy()], [target], None, f"{tmpdir}/grid_test")
        assert Path(f"{tmpdir}/grid_test/sample_0000_gt.png").exists()

        # Test save_overlay_grid (with pred_outputs)
        save_overlay_grid(
            [img.copy()],
            [target],
            [{"boxes": pred_boxes, "masks": pred_masks, "scores": pred_scores}],
            f"{tmpdir}/grid_test2",
        )
        assert Path(f"{tmpdir}/grid_test2/sample_0000_gt.png").exists()
        assert Path(f"{tmpdir}/grid_test2/sample_0000_pred.png").exists()
        assert Path(f"{tmpdir}/grid_test2/sample_0000_gt_pred.png").exists()

        print("All vis_overlay tests passed.")
