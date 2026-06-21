import ml_collections
import torch
import torch.nn as nn
from transformers import CLIPVisionModel
try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

class VisionModel(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        """
        genericgenericgenericāgeneric?

        generic:
            config (ml_collections.ConfigDict): genericāgeneric
        """
        super(VisionModel, self).__init__()

        self.embedding_dim = config.vision.embedding_dim
        self.encoder = CLIPVisionModel.from_pretrained(config.vit_name)

        self.extract_stage = [
            self.encoder.config.num_hidden_layers // 3 - 1,
            self.encoder.config.num_hidden_layers // 3 * 2 - 1,
            self.encoder.config.num_hidden_layers - 2,
        ]

    def encode(self, x: torch.Tensor):
        """
        genericュgeneric?

        generic:
            x (torch.Tensor): generic€generic?(B, C, H, W)generic〃genericぇgeneric€generic€generic€generic﹀generic

        generic:
            image_embeds (torch.Tensor): generic (B, S, D)generic︼genericDgenericョgeneric?
        """
        outputs = self.encoder(
            x,
            return_dict=True,
            output_hidden_states=True,
        )
        # image_embeds = outputs.hidden_states[11][:, 1:, :]

        image_embeds = []
        for idx, stage in enumerate(self.extract_stage):
            current_hidden_states = outputs.hidden_states[stage][:, 1:, :]
            image_embeds.append(current_hidden_states)
        image_embeds = torch.cat(image_embeds, dim=1)

        return image_embeds

    def forward(self, x):
        """
        generic

        generic:
            x (dict): genericgeneric"rgb"generic?

        generic:
            torch.Tensor: generic
        """
        modal_input = x["rgb"]
        return self.encode(modal_input)
