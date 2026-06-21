import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Sequence

class PhysicsGuidedLoss(nn.Module):
    """
    generic▼genericTEgeneric″genericgeneric╃generic︽generic″generic?
    genericユgenericgeneric Z_visual generic﹂genericOPsgeneric
    generic€genericgenericgenericㄦgeneric㈢genericgeneric╃generic?
    """
    def __init__(self, visual_dim=768, num_bands=4):
        super().__init__()
        self.latent_to_rrs = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.LayerNorm(256),
            nn.Linear(256, num_bands),
            nn.Softplus()
        )
        
        self.A_tsm = nn.Parameter(torch.tensor(289.29), requires_grad=False)
        self.C_tsm = nn.Parameter(torch.tensor(0.1686), requires_grad=False)

    def compute_rte_tsm(self, rrs_red: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        rrs_clamped = torch.clamp(rrs_red, max=(self.C_tsm.item() - eps))
        
        denominator = torch.clamp(1.0 - rrs_clamped / self.C_tsm, min=eps)
        tsm_pred = (self.A_tsm * rrs_clamped) / denominator
        
        return tsm_pred

    def forward(self, Z_visual: torch.Tensor, batch: Dict) -> torch.Tensor:
        """
        Args:
            Z_visual: [B, N, visual_dim] genericㄨgenericｈ€﹁generic〃generic?
            batch: generic€?'tsm' generic 'mask'
        """
        gt_tsm = batch.get('tsm')
        mask = batch.get('mask')
        
        if gt_tsm is None:
            return torch.tensor(0.0, device=Z_visual.device, requires_grad=True)

        pred_rrs = self.latent_to_rrs(Z_visual) # [B, N, num_bands]
        
        rrs_red = pred_rrs[..., 2]
        
        pred_tsm = self.compute_rte_tsm(rrs_red)
        
        if gt_tsm.dim() == 4:
            gt_tsm = gt_tsm.flatten(2).transpose(1, 2)
        if mask is not None and mask.dim() == 4:
            mask = mask.flatten(2).transpose(1, 2)

        loss_elements = F.mse_loss(pred_tsm, gt_tsm, reduction='none')
        
        if mask is not None:
            loss_phy = torch.sum(loss_elements * mask) / (torch.sum(mask) + 1e-6)
        else:
            loss_phy = torch.mean(loss_elements)
            
        return loss_phy


def _extract_tensor(batch: Dict, keys: Sequence[str]) -> Optional[torch.Tensor]:
    for key in keys:
        value = batch.get(key, None)
        if torch.is_tensor(value):
            return value
    return None


def _to_bchw(x: torch.Tensor) -> Optional[torch.Tensor]:
    if x.dim() == 4:
        return x
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)
    return None


def compute_rte_rrs_gt(batch: Dict, coeffs: Dict = None) -> Optional[torch.Tensor]:
    """
    Build physical GT from water-leaving reflectance with a Nechad-style semi-analytical model.

    TSM = (A * Rrs_red) / (1 - Rrs_red / C)
    """
    coeffs = coeffs or {}
    A_tsm = float(coeffs.get("A_tsm", 289.29))
    C_tsm = float(coeffs.get("C_tsm", 0.1686))
    red_band_index = int(coeffs.get("red_band_index", 2))
    eps = 1e-6

    # Prefer explicitly provided red-band reflectance.
    rrs_red = _extract_tensor(batch, ["rrs_red", "Rrs_red"])

    # Fallback: derive red band from multi-band reflectance tensor.
    if rrs_red is None:
        rrs_full = _extract_tensor(
            batch,
            ["rrs", "Rrs", "rrs_map", "Rrs_map", "rrs_full", "Rrs_full", "reflectance", "Reflectance"],
        )
        if rrs_full is None:
            return None

        if rrs_full.dim() == 4:
            if rrs_full.size(1) > 1:
                red_band_index = min(max(red_band_index, 0), rrs_full.size(1) - 1)
                rrs_red = rrs_full[:, red_band_index : red_band_index + 1, :, :]
            else:
                rrs_red = rrs_full
        elif rrs_full.dim() == 3:
            # [B,H,W] style input
            rrs_red = rrs_full.unsqueeze(1)
        elif rrs_full.dim() == 2:
            # Single map [H,W]
            rrs_red = rrs_full.unsqueeze(0).unsqueeze(0)
        else:
            return None
    else:
        rrs_red = _to_bchw(rrs_red)
        if rrs_red is None:
            return None

    rrs_red = torch.nan_to_num(rrs_red.float(), nan=0.0, posinf=0.0, neginf=0.0)
    rrs_red = torch.clamp(rrs_red, min=0.0, max=(C_tsm - eps))
    denominator = torch.clamp(1.0 - rrs_red / C_tsm, min=eps)
    tsm_pred = (A_tsm * rrs_red) / denominator
    return tsm_pred


def compute_sar_sigma0_gt(batch: Dict, coeffs: Dict = None) -> Optional[torch.Tensor]:
    """
    SAR generic″genericgenericュgeneric€?
    sigma0(dB) = A + B * log10(wind_speed) + C * (incidence_deg) + D_pol
    """
    if coeffs is None:
        return None
    wind = batch.get('wind_speed')
    inc = batch.get('incidence_deg')
    pol = batch.get('polarization')
    
    if wind is None or inc is None or pol is None:
        return None
        
    A, B, C = float(coeffs.get('A', 0.0)), float(coeffs.get('B', 0.0)), float(coeffs.get('C', 0.0))
    D = float(coeffs.get('D_VV', 0.0)) if pol == 'VV' else float(coeffs.get('D_VH', 0.0))
    
    wind = torch.clamp(torch.nan_to_num(wind, nan=1e-6, posinf=100.0, neginf=1e-6), min=1e-6)
    inc = torch.nan_to_num(inc, nan=0.0)
    
    sigma0_db = A + B * torch.log10(wind) + C * inc + D
    return sigma0_db
