from typing import Dict, Optional
import torch


def compute_rte_rrs_gt(batch: Dict) -> Optional[torch.Tensor]:
    """
    Placeholder for radiative transfer equation based remote sensing reflectance (Rrs) per-pixel.
    Expected inputs in batch (examples):
      - 'chl': chlorophyll concentration map [B,1,H,W]
      - 'tsm': total suspended matter map [B,1,H,W]
      - 'cdom': colored dissolved organic matter map [B,1,H,W]
      - 'wavelengths': sensor wavelength bands (list or tensor)
    Return a tensor [B,C,H,W] aligned with image where C=out_channels.
    """
    # Users should implement with their RTE or in-situ inversion pipeline.
    return None


def compute_sar_sigma0_gt(batch: Dict, coeffs: Dict = None) -> Optional[torch.Tensor]:
    """
    Example empirical SAR backscatter sigma0 approximation:
      sigma0(dB) = A + B * log10(wind_speed) + C * (incidence_deg) + D_pol
    Inputs in batch (examples):
      - 'wind_speed': [B,1,H,W]
      - 'incidence_deg': [B,1,H,W]
      - 'polarization': string per sample (e.g., 'VV', 'VH') mapped to D_pol
    coeffs may contain keys 'A','B','C','D_VV','D_VH'.
    Returns sigma0 map in linear or dB depending on downstream scaling.
    """
    if coeffs is None:
        return None
    wind = batch.get('wind_speed')
    inc = batch.get('incidence_deg')
    pol = batch.get('polarization')
    if wind is None or inc is None or pol is None:
        return None
    A = float(coeffs.get('A', 0.0))
    B = float(coeffs.get('B', 0.0))
    C = float(coeffs.get('C', 0.0))
    D_vv = float(coeffs.get('D_VV', 0.0))
    D_vh = float(coeffs.get('D_VH', 0.0))
    D = D_vv if pol == 'VV' else D_vh
    # 清洗与约束：风速需非负且有限，避免 log10 产生 NaN
    wind = torch.nan_to_num(wind, nan=0.0, posinf=0.0, neginf=0.0)
    wind = torch.clamp(wind, min=1e-6)
    inc = torch.nan_to_num(inc, nan=0.0, posinf=0.0, neginf=0.0)
    sigma0_db = A + B * torch.log10(wind) + C * inc + D
    return sigma0_db