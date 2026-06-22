from collections import deque

import numpy as np
import torch


class HistoryBuffer:
    """The class tracks a series of values and provides access to the smoothed
    value over a window or the global average / sum of the series.

    Example::

        >>> his_buf = HistoryBuffer()
        >>> his_buf.update(0.1)
        >>> his_buf.update(0.2)
        >>> his_buf.avg
        0.15
    """

    def __init__(self, window_size: int = 20) -> None:
        self._history = deque(maxlen=window_size)
        self._count: int = 0
        self._sum: float = 0.0

    @staticmethod
    def _to_float(value) -> float:
        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.numel() != 1:
                tensor = tensor.float().mean()
            return float(tensor.float().cpu().item())
        if isinstance(value, np.ndarray):
            if value.size != 1:
                value = value.astype(np.float32).mean()
            return float(value.item())
        return float(value)

    def update(self, value: float) -> None:
        value = self._to_float(value)
        self._history.append(value)
        self._count += 1
        self._sum += value

    @property
    def latest(self) -> float:
        return self._history[-1]

    @property
    def avg(self) -> float:
        return np.mean(self._history)

    @property
    def global_avg(self) -> float:
        return self._sum / self._count

    @property
    def global_sum(self) -> float:
        return self._sum
