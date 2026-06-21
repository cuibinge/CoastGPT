import ml_collections
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from .common_arch import AttnPooler, LayerNorm, LayerNormFp32
try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None


class EmbeddingModel(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        """
        genericgeneric?EmbeddingModel generic″generic?

        generic:
            config: generic″genericgeneric vision.embedding_dim generic?language.embedding_dim
        """
        self.vision_embedding_dim = config.vision.embedding_dim
        self.language_embedding_dim = config.language.embedding_dim

        super(EmbeddingModel, self).__init__()
        self.geo_encoder = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, self.language_embedding_dim)
        )

        self.time_encoder = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
            nn.Linear(256, self.language_embedding_dim)
        )
        # self.projection = nn.Linear(config.vision.embedding_dim, config.language.embedding_dim)

        if config.adjust_norm:
            norm_layer = (
                LayerNormFp32 if config.dtype in ("float16", "bfloat16") else LayerNorm
            )
        else:
            norm_layer = LayerNorm

        self.projection = AttnPooler(
            num_query=config.rgb_vision.attn_pooler.num_query,
            num_layers=config.rgb_vision.attn_pooler.num_layers,
            num_attention_heads=config.rgb_vision.attn_pooler.num_attn_heads,
            encoder_hidden_size=config.vision.embedding_dim,
            hidden_size=config.vision.embedding_dim,
            output_size=config.text.hidden_size,
            norm_layer=norm_layer,
            checkpoint=getattr(config, "use_checkpoint", False),
        )

    def forward(self, data: Dict, image_embedding):
        """
        genericgeneric℃€genericャ€?

        generic:
            data: generic€genericgeneric?
            image_embedding: genericgeneric″genericワgeneric㈢generic?(batch_size, vision_embedding_dim)

        generic:
            multimodal_embedding: genericāgenericワgeneric㈢generic?(batch_size, language_embedding_dim)
        """
        lat, lon = data["lat"], data["lon"]
        lat_norm = (lat + 90) / 180
        lon_norm = (lon + 180) / 360
        geo_input = torch.tensor([lat_norm, lon_norm])  # [2]
        geo_embedding = self.geo_encoder(geo_input)
        geo_embedding = geo_embedding.unsqueeze(1)

        timestamp = data["timestamp"]
        time_norm = torch.tensor([
            (timestamp[0] - 2000) / 100.0,
            timestamp[1] / 12.0,
            timestamp[2] / 31.0,
            timestamp[3] / 24,
            timestamp[4] / 60.0,
            timestamp[5] / 60.0
        ])
        time_embedding = self.time_encoder(time_norm)
        time_embedding = time_embedding.unsqueeze(1)

        image_embedding = self.projection(image_embedding)

        multimodal_embedding = torch.cat([image_embedding, geo_embedding, time_embedding], dim=1)
        return multimodal_embedding

    def encode_test(self, image_embedding):
        """
        genericgeneric€?

        generic:
            image_embedding: genericgeneric″genericワgeneric㈢generic?(batch_size, vision_embedding_dim)

        generic:
            multimodal_embedding: genericāgenericワgeneric㈢generic?(batch_size, language_embedding_dim)
        """
        projected_image_embedding = self.projection(image_embedding)
        return projected_image_embedding
