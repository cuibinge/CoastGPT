from collections import OrderedDict
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)