import math
from typing import Callable, Dict, Tuple, Union

import ml_collections
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import CLIPVisionModel
from transformers import BatchEncoding


class VisionModel(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        super(VisionModel, self).__init__(config)

        self.embedding_dim = config.embedding_dim
        self.encoder = CLIPVisionModel.from_pretrained(config.vit_name)

    def encode(self, x: torch.Tensor):
        outputs = self.encoder(
            x,
            return_dict=True,
            output_hidden_states=True,
        )
        image_embeds = outputs.hidden_states[-1][:, 1:, :]
        return image_embeds

    def forward(self, x):
        modal_input = x["rgb"]
        return self.encode(modal_input)
