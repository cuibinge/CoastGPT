import logging
import os
import pathlib
from typing import Dict, List, Tuple

import ml_collections
import torch
import torch.nn as nn
from deepspeed.utils.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint,
    load_state_dict_from_zero_checkpoint,
)
from peft import PeftModel
from Vision_model import VisionModel
from Language_model import LanguageModel


class CoastGPT(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        super(CoastGPT, self).__init__()
        self.stage = config.stage

        self.vision = VisionModel(config)
        self.language = LanguageModel(config)

    def forward(self, data: Dict):

        image_embedding = self.vision(data)
        output = self.language(data, image_embedding = image_embedding)

        return output

    def encode_image(self, image, pool):
        image_embedding = self.vision.encode(image)
        if pool:
            return image_embedding.mean(dim=1)
        else:
            return image_embedding

    def generate(
            self,
            input_ids: torch.Tensor,
            images: torch.Tensor = None,
            do_sample: bool = True,
            temperature: float = 0.2,
            max_new_tokens: int = 1024,
            streamer=None,
            use_cache: bool = True,
            stopping_criteria=None,
            **kwargs,
    ):

        if images is not None:
            image_embedding = self.encode_image(images, pool=False)
        else:
            image_embedding = None
        return self.language.generate(
            input_ids=input_ids,
            image_embedding=image_embedding,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=use_cache,
            stopping_criteria=stopping_criteria,
            **kwargs,
        )
