"""Multi-scale batch sampler for DDP training.

Supports two modes:
  1. Weighted random: samples a resolution from a pool per training step.
  2. Bucket by original size: groups samples by their native resolution;
     each batch contains only samples of the same original size. DDP-safe.
"""

import torch
import torch.distributed as dist
from typing import List, Optional, Tuple


class MultiScaleBatchSampler:
    """Samples one resolution per training step, broadcast to all DDP ranks.

    Args:
        sizes: List of (H, W) candidate resolutions.
        weights: Sampling weights for each size. Defaults to uniform.
        enabled: If False, always returns sizes[0] (legacy fixed-size mode).
        bucket_by_original: If True, ``sample()`` returns -1 as a signal that
            the dataloader should use each sample's ``original_size`` directly,
            rather than picking a size from the pool. The ``current_size``
            property is not meaningful in this mode.
    """

    def __init__(
        self,
        sizes: List[Tuple[int, int]],
        weights: Optional[List[float]] = None,
        enabled: bool = True,
        bucket_by_original: bool = False,
    ):
        if not sizes:
            raise ValueError("sizes must not be empty")
        self.sizes = sizes
        self.weights = weights if weights is not None else [1.0 / len(sizes)] * len(sizes)

        if len(self.weights) != len(self.sizes):
            raise ValueError(
                f"weights length ({len(self.weights)}) != sizes length ({len(self.sizes)})"
            )

        self.enabled = enabled
        self.bucket_by_original = bucket_by_original
        self._current_size = sizes[0] if not bucket_by_original else (-1, -1)

    @property
    def current_size(self) -> Tuple[int, int]:
        """Current step's target size, or (-1, -1) in bucket-by-original mode."""
        return self._current_size

    def sample(self) -> Tuple[int, int]:
        """Sample a size for the current global step.

        In bucket-by-original mode, returns (-1, -1) to signal the dataloader
        should use each sample's native original_size.

        Must be called on ALL ranks. Only rank 0 samples; the result is
        broadcast to all other ranks for DDP consistency.
        """
        if not self.enabled:
            self._current_size = self.sizes[0]
            return self._current_size

        if self.bucket_by_original:
            self._current_size = (-1, -1)
            return self._current_size

        if dist.is_initialized():
            if dist.get_rank() == 0:
                idx = torch.multinomial(
                    torch.tensor(self.weights, dtype=torch.float),
                    num_samples=1,
                ).item()
                size_tensor = torch.tensor(
                    [self.sizes[idx][0], self.sizes[idx][1]], dtype=torch.long
                )
            else:
                size_tensor = torch.zeros(2, dtype=torch.long)

            dist.broadcast(size_tensor, src=0)
            self._current_size = (int(size_tensor[0].item()), int(size_tensor[1].item()))
        else:
            idx = torch.multinomial(
                torch.tensor(self.weights, dtype=torch.float),
                num_samples=1,
            ).item()
            self._current_size = self.sizes[idx]

        return self._current_size


def build_multiscale_sampler(cfg) -> MultiScaleBatchSampler:
    """Build a MultiScaleBatchSampler from a config's multi_scale block.

    If multi_scale is not present or disabled, returns a fixed-size sampler.
    """
    ms_cfg = getattr(cfg, 'multi_scale', None)
    if ms_cfg is None or not ms_cfg.get('enabled', False):
        # Fixed-size mode: use default_input_size from config
        default_size = getattr(
            getattr(cfg, 'rgb_vision', None), 'default_input_size', None
        ) or getattr(getattr(cfg, 'rgb_vision', None), 'input_size', [224, 224])
        return MultiScaleBatchSampler(
            sizes=[tuple(default_size)],
            enabled=False,
        )

    sizes = [tuple(s) for s in ms_cfg['sizes']]
    weights = ms_cfg.get('weights')
    bucket_by_original = ms_cfg.get('bucket_by_original_size', False)
    return MultiScaleBatchSampler(
        sizes=sizes,
        weights=weights,
        enabled=True,
        bucket_by_original=bucket_by_original,
    )
