import ml_collections
import torch
import torch.nn as nn
from transformers import CLIPVisionModel
from Models.GeoLangBindtest2.WavelenDynamicEncoder import WavelengthDynamicEncoder, ModalityAwareAggregation
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
        
        self.use_wavelength_encoder = config.vision.get('use_wavelength_encoder', True)
        if self.use_wavelength_encoder:
            hidden_size = self.encoder.config.hidden_size
            self.wavelength_encoder = WavelengthDynamicEncoder(
                in_channels=hidden_size,
                out_channels=hidden_size,
                latent_dim=config.vision.get('wavelength_latent_dim', 64)
            )
            
            self.modality_aggregation = ModalityAwareAggregation(embed_dim=hidden_size)
            
            self.modality_adapters = nn.ModuleDict({
                'rgb': nn.Identity(),
                'infrared': nn.Conv2d(3, 3, kernel_size=1),
                'radar': nn.Conv2d(3, 3, kernel_size=1),
                'multispectral': nn.Conv2d(10, 3, kernel_size=1)
            })
            
            self.default_wavelengths = {
                'rgb': torch.tensor([[0.45], [0.54], [0.65]]),
                'infrared': torch.tensor([[1.4], [3.0], [5.0]]),
                'radar': torch.tensor([[0.8], [3.0], [10.0]]),
                'multispectral': None
            }

    def encode(self, x: torch.Tensor, modality: str = 'rgb', wavelengths=None):
        """
        genericュgeneric?

        generic:
            x (torch.Tensor): generic€generic?(B, C, H, W)
            modality (str): generic℃€generic€?rgb', 'infrared', 'radar', 'multispectral'
            wavelengths (torch.Tensor, optional): generic㈤generic℃generic (C, 1)

        generic:
            image_embeds (torch.Tensor): generic (B, S, D)
        """
        if self.use_wavelength_encoder:
            if wavelengths is None and modality in self.default_wavelengths:
                wavelengths = self.default_wavelengths[modality]
                if wavelengths is not None:
                    wavelengths = wavelengths.to(x.device)
            
            if modality in self.modality_adapters:
                x = self.modality_adapters[modality](x)
        
        outputs = self.encoder(
            x,
            return_dict=True,
            output_hidden_states=True,
        )
        image_embeds = outputs.hidden_states[11][:, 1:, :]

        if self.use_wavelength_encoder and wavelengths is not None:
            batch_size, seq_len, hidden_dim = image_embeds.shape
            h = w = int(seq_len ** 0.5)
            
            reshaped_embeds = image_embeds.transpose(1, 2).reshape(batch_size, hidden_dim, h, w)
            
            encoded_embeds = self.wavelength_encoder(reshaped_embeds, wavelengths)
            
            image_embeds = encoded_embeds.reshape(batch_size, hidden_dim, seq_len).transpose(1, 2)
            
            pooled_embeds = image_embeds.mean(dim=1)  # [B, D]
            enhanced_embeds = self.modality_aggregation(pooled_embeds, wavelengths)
            
            enhanced_scale = enhanced_embeds.unsqueeze(1) / pooled_embeds.unsqueeze(1)
            image_embeds = image_embeds * enhanced_scale.unsqueeze(1)

        return image_embeds

    def forward(self, x):
        """
        generic

        generic:
            x (dict): genericgenericgeneric℃generic
                - generic』generic€genericāgeneric(generic?rgb", "infrared"generic?generic?
                - generic€generic?modality"genericgenericāgeneric?
                - generic€generic?wavelengths"genericgeneric?

        generic:
            torch.Tensor: generic
        """
        modality = x.get("modality", "rgb")
        
        if modality in x:
            modal_input = x[modality]
        else:
            modal_input = x.get("rgb", None)
            if modal_input is None:
                for key in ["infrared", "radar", "multispectral"]:
                    if key in x:
                        modal_input = x[key]
                        modality = key
                        break
                        
        wavelengths = x.get("wavelengths", None)
        
        return self.encode(modal_input, modality, wavelengths)
