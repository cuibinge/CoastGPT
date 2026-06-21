import logging
import os
import pathlib
from typing import Dict, List

import ml_collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.utils.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint,
)
from peft import PeftModel
from .vision_model import VisionModel as SingleVisionModel
from .dual_vision_encoder import DualVisionEncoder
from .language_model import LanguageModel
from .embedding_model_r1 import EmbeddingModel
from . import physics_decoder as phys

PhysicsDecoder = phys.PhysicsDecoder
pixelwise_mse = getattr(phys, "pixelwise_mse", lambda pred, gt: F.mse_loss(pred, gt))

def _edge_preserve_default(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    dh_p = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    dw_p = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dh_g = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])
    dw_g = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
    return F.l1_loss(dh_p, dh_g) + F.l1_loss(dw_p, dw_g)

edge_preserve_loss = getattr(phys, "edge_preserve_loss", _edge_preserve_default)

def _tv_default(x: torch.Tensor) -> torch.Tensor:
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw

total_variation_loss = getattr(phys, "total_variation_loss", _tv_default)

def _consistency_default(phy_full, phy_s1, phy_s2, phy_s3) -> torch.Tensor:
    H, W = phy_full.shape[-2], phy_full.shape[-1]
    s1_up = F.interpolate(phy_s1, size=(H, W), mode="bilinear", align_corners=False)
    s2_up = F.interpolate(phy_s2, size=(H, W), mode="bilinear", align_corners=False)
    s3_up = F.interpolate(phy_s3, size=(H, W), mode="bilinear", align_corners=False)
    return (F.mse_loss(phy_full, s1_up) + F.mse_loss(phy_full, s2_up) + F.mse_loss(phy_full, s3_up)) / 3.0

consistency_loss_multiscale = getattr(phys, "consistency_loss_multiscale", _consistency_default)

def _sam_default(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, C, H, W = pred.shape
    p = pred.permute(0, 2, 3, 1).reshape(-1, C)
    g = gt.permute(0, 2, 3, 1).reshape(-1, C)
    dot = (p * g).sum(dim=1)
    pn = torch.norm(p, dim=1)
    gn = torch.norm(g, dim=1)
    cos = torch.clamp(dot / (pn * gn + eps), min=-1.0, max=1.0)
    ang = torch.acos(cos)
    return ang.mean()

spectral_loss_sam = getattr(phys, "spectral_loss_sam", _sam_default)
from .physics_constraints import compute_rte_rrs_gt, compute_sar_sigma0_gt
try:
    import torch_npu  # noqa: F401
    HAS_TORCH_NPU = True
except Exception:
    HAS_TORCH_NPU = False

logger = logging.getLogger("train")


class CoastGPT(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        """Initialize the generic multimodal model."""
        super(CoastGPT, self).__init__()
        self.config = config
        self.stage = config.stage

        if getattr(config, "rgb_vision", ml_collections.ConfigDict()).get("arch", "vit_large") == "dual":
            self.vision = DualVisionEncoder(config)
        else:
            self.vision = SingleVisionModel(config)
        self.language = LanguageModel(config)
        self.multimodal = EmbeddingModel(config)

        phy_cfg = getattr(config, "physics", ml_collections.ConfigDict())
        self.physics_enabled = bool(phy_cfg.get("enabled", False))
        if self.physics_enabled:
            out_channels = int(phy_cfg.get("out_channels", 1))
            dec_channels = int(getattr(self.vision, "embedding_dim", 256))
            self.physics = PhysicsDecoder(out_channels=out_channels, dec_channels=dec_channels)
            self.physics_loss_weight = float(phy_cfg.get("loss_weight", 1.0))
            self.physics_edge_weight = float(phy_cfg.get("edge_weight", 0.0))
            self.physics_tv_weight = float(phy_cfg.get("tv_weight", 0.0))
            self.physics_consistency_weight = float(phy_cfg.get("consistency_weight", 0.0))
            self.physics_spectral_weight = float(phy_cfg.get("spectral_weight", 0.0))
    def forward(self, data: Dict):
        """Run a multimodal training forward pass."""
        out = dict()
        total_loss = None

        # Convert optional generic condition prompt tokens to continuous embeddings.
        physical_prompt_embs = None
        emb_layer = None
        try:
            if hasattr(self.language, "get_text_encoder"):
                emb_layer = self.language.get_text_encoder().get_input_embeddings()
            elif hasattr(self.language, "text_encoder"):
                emb_layer = self.language.text_encoder.get_input_embeddings()
            elif hasattr(self.language, "model"):
                emb_layer = self.language.model.get_input_embeddings()
        except Exception as exc:
            emb_layer = None
            if not getattr(self, "_warned_semantic_emb_layer", False):
                logger.warning(
                    "Failed to get the text embedding layer for condition prompts: %s", exc
                )
                self._warned_semantic_emb_layer = True
        if (
            emb_layer is None
            and not getattr(self, "_warned_semantic_emb_layer_none", False)
            and data.get("physical_prompt_ids", None) is not None
        ):
            logger.warning(
                "Condition prompt ids exist in batch, but the embedding layer is unavailable."
            )
            self._warned_semantic_emb_layer_none = True
        if emb_layer is not None:
            if "physical_prompt_ids" in data and data["physical_prompt_ids"] is not None:
                physical_prompt_embs = emb_layer(data["physical_prompt_ids"])
                if "physical_prompt_attention_mask" in data and data["physical_prompt_attention_mask"] is not None:
                    phy_mask = data["physical_prompt_attention_mask"].to(physical_prompt_embs.device)
                    physical_prompt_embs = physical_prompt_embs * phy_mask.unsqueeze(-1).to(physical_prompt_embs.dtype)
                data["physical_prompt_embs"] = physical_prompt_embs

        if isinstance(self.vision, DualVisionEncoder):
            image_seq, fused_spatial, pyramid_raw = self.vision.encode_with_spatial(
                data["rgb"], physical_prompt_embs=physical_prompt_embs
            )
            if hasattr(self.vision, "get_alignment_stats"):
                for k, v in self.vision.get_alignment_stats().items():
                    out[f"vision_{k}"] = v
        else:
            image_seq = self.vision(data)
            fused_spatial, pyramid_raw = None, None

        multimodal_embedding = self.multimodal(data, image_embedding=image_seq)

        output = self.language(
            data,
            multimodal_embedding=multimodal_embedding,
            vector_objective_mask=data.get("vector_objective_mask", None),
        )

        text_loss = output
        if not torch.is_tensor(text_loss):
            raise RuntimeError(f"language model must return a tensor loss, got {type(text_loss)}")
        total_loss = text_loss
        out.update({"text_loss": text_loss})

        if hasattr(self.language, "get_vector_loss"):
            vector_loss = self.language.get_vector_loss()
            if vector_loss is not None:
                vector_cfg = getattr(self.config, "vector_objective", ml_collections.ConfigDict())
                vector_weight = float(vector_cfg.get("loss_weight", 0.0))
                vector_loss_weighted = (vector_weight * vector_loss).to(total_loss.dtype)
                total_loss = total_loss + vector_loss_weighted
                out.update(
                    {
                        "vector_loss_raw": vector_loss,
                        "vector_loss_weighted": vector_loss_weighted,
                    }
                )
        for key in ("vector_feature_count", "vector_point_count", "vector_bbox_area", "vector_closure_error"):
            value = data.get(key, None)
            if torch.is_tensor(value):
                out[key] = value.float().mean().to(text_loss.device)

        if hasattr(self.multimodal, "get_aux_loss"):
            mm_aux_loss = self.multimodal.get_aux_loss()
            
            mm_aux_weight = getattr(self.config, "mm_moe_aux_weight", 0.01) 
            
            if mm_aux_loss is not None:
                mm_aux_loss_weighted = (mm_aux_weight * mm_aux_loss).to(total_loss.dtype)
                total_loss = total_loss + mm_aux_loss_weighted
                
                out.update({
                    "mm_moe_aux_loss_raw": mm_aux_loss,
                    "mm_moe_aux_loss_weighted": mm_aux_loss_weighted
                })
            
            if hasattr(self.multimodal, "get_gate_stats"):
                gate_stats = self.multimodal.get_gate_stats()
                if "load" in gate_stats and gate_stats["load"] is not None:
                    load_tensor = gate_stats["load"]
                    for e_idx, load_val in enumerate(load_tensor):
                        out[f"mm_moe_expert_{e_idx}_load"] = load_val
                if "entropy" in gate_stats:
                    out["mm_moe_gate_entropy"] = gate_stats["entropy"]
                if "zloss" in gate_stats:
                    out["mm_moe_gate_zloss"] = gate_stats["zloss"]
                if "invalid_gate_ratio" in gate_stats:
                    out["mm_moe_invalid_gate_ratio"] = gate_stats["invalid_gate_ratio"]

        if self.physics_enabled and fused_spatial is not None:
            H, W = data["rgb"].shape[-2], data["rgb"].shape[-1]

            try:
                dev_type = fused_spatial.device.type
            except Exception:
                dev_type = "cuda"

            from torch import autocast as _autocast
            with _autocast(device_type=dev_type, enabled=False):
                fused_spatial_fp32 = fused_spatial.float()
                fused_spatial_fp32 = torch.nan_to_num(
                    fused_spatial_fp32, nan=0.0, posinf=0.0, neginf=0.0
                )
                try:
                    self.physics.to(device=fused_spatial_fp32.device, dtype=torch.float32)
                except Exception:
                    for p in self.physics.parameters():
                        p.data = p.data.to(dtype=torch.float32, device=fused_spatial_fp32.device)
                    for name, buffer in self.physics.named_buffers():
                        buffer.data = buffer.data.to(dtype=torch.float32, device=fused_spatial_fp32.device)
                phy_pred = self.physics(
                    fused_spatial=fused_spatial_fp32,
                    pyramid_raw=pyramid_raw,
                    input_size=(H, W),
                )
            phy_maps = [
                phy_pred.get("phy_full"),
                phy_pred.get("phy_s1"),
                phy_pred.get("phy_s2"),
                phy_pred.get("phy_s3"),
            ]
            has_nan = any([
                (m is not None and not torch.isfinite(m).all()) for m in phy_maps
            ])
            if has_nan:
                logger.warning("Physics decoder produced non-finite values; sanitizing outputs.")
                for k in ["phy_full", "phy_s1", "phy_s2", "phy_s3"]:
                    if k in phy_pred and phy_pred[k] is not None:
                        phy_pred[k] = torch.nan_to_num(phy_pred[k], nan=0.0, posinf=0.0, neginf=0.0)
                out.update({"phy_has_nan": torch.tensor(1.0, device=fused_spatial.device)})
            else:
                out.update({"phy_has_nan": torch.tensor(0.0, device=fused_spatial.device)})

            tv_w = self.physics_tv_weight
            cons_w = self.physics_consistency_weight
            reg_tv = (
                total_variation_loss(phy_pred["phy_full"]) if tv_w > 0 else fused_spatial.new_zeros((), dtype=torch.float32)
            )
            reg_cons = (
                consistency_loss_multiscale(phy_pred["phy_full"], phy_pred["phy_s1"], phy_pred["phy_s2"], phy_pred["phy_s3"]) if cons_w > 0 else fused_spatial.new_zeros((), dtype=torch.float32)
            )

            phy_gt = None
            if "phy_gt" in data:
                phy_gt = data["phy_gt"]
            elif "phy_gt_full" in data:
                phy_gt = data["phy_gt_full"]

            if phy_gt is None and bool(getattr(self, "physics_enabled", False)):
                auto_gt_cfg = bool(self.config.physics.get("auto_gt_from_constraints", False))
                if auto_gt_cfg:
                    rte_coeffs = self.config.physics.get("rte_tsm_coeffs", None)
                    phy_gt = compute_rte_rrs_gt(data, rte_coeffs)
                    if phy_gt is None:
                        coeffs = self.config.physics.get("sar_sigma0_coeffs", None)
                        phy_gt = compute_sar_sigma0_gt(data, coeffs)

            if phy_gt is not None:
                gt_full = F.interpolate(phy_gt, size=(H, W), mode="bilinear", align_corners=False) if phy_gt.shape[-2:] != (H, W) else phy_gt
                l_full = pixelwise_mse(phy_pred["phy_full"], gt_full)

                s1_sz = phy_pred["phy_s1"].shape[-2:]
                s2_sz = phy_pred["phy_s2"].shape[-2:]
                s3_sz = phy_pred["phy_s3"].shape[-2:]
                gt_s1 = F.interpolate(gt_full, size=s1_sz, mode="bilinear", align_corners=False)
                gt_s2 = F.interpolate(gt_full, size=s2_sz, mode="bilinear", align_corners=False)
                gt_s3 = F.interpolate(gt_full, size=s3_sz, mode="bilinear", align_corners=False)

                l_s1 = pixelwise_mse(phy_pred["phy_s1"], gt_s1)
                l_s2 = pixelwise_mse(phy_pred["phy_s2"], gt_s2)
                l_s3 = pixelwise_mse(phy_pred["phy_s3"], gt_s3)

                edge_w = self.physics_edge_weight
                l_edge = edge_preserve_loss(phy_pred["phy_full"], gt_full) if edge_w > 0 else gt_full.new_zeros((), dtype=torch.float32)

                spec_w = self.physics_spectral_weight
                spec_l = (
                    spectral_loss_sam(phy_pred["phy_full"], gt_full) if spec_w > 0 and gt_full.shape[1] > 1 else gt_full.new_zeros((), dtype=torch.float32)
                )

                w_full = 1.0
                w_s1 = 0.5
                w_s2 = 0.35
                w_s3 = 0.25
                physics_weight = self.physics_loss_weight

                phy_loss = (
                    w_full * l_full + w_s1 * l_s1 + w_s2 * l_s2 + w_s3 * l_s3
                ) + edge_w * l_edge + tv_w * reg_tv + cons_w * reg_cons + spec_w * spec_l
                total_loss = total_loss + (physics_weight * phy_loss).to(text_loss.dtype)
                out.update({
                    "phy_loss": phy_loss,
                    "phy_tv_loss": reg_tv,
                    "phy_consistency_loss": reg_cons,
                    "phy_spectral_loss": spec_l,
                })
            else:
                physics_weight = self.physics_loss_weight
                phy_loss = tv_w * reg_tv + cons_w * reg_cons
                total_loss = total_loss + (physics_weight * phy_loss).to(text_loss.dtype)
                out.update({
                    "phy_loss": phy_loss,
                    "phy_tv_loss": reg_tv,
                    "phy_consistency_loss": reg_cons,
                })
        out.update({"non_text_loss": total_loss - text_loss})
        out.update({"total_loss": total_loss})

        return out

    def encode_image(self, image, pool):
        """


        """
        image_embedding = self.vision.encode(image)
        image_embedding = self.multimodal.encode_test(image_embedding)
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
        """


        """
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

    def _save_text_lora_checkpoint(self, file_name: str) -> None:
        if self.stage < 2:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
        file_path = pathlib.Path(file_name)
        if file_path.is_file():
            loar_output_path = file_path.parent / "TextLoRA"
        else:
            loar_output_path = file_path / "TextLoRA"
        self.language.text_encoder.save_pretrained(str(loar_output_path))

    def custom_save_checkpoint(self, file_name: str):
        # Only attempt ZeRO consolidation if a "latest" pointer is present AND
        # points to a global_step* dir that actually exists. Otherwise stale
        # global_step dirs from previous runs would be silently merged in,
        # producing a checkpoint that mixes old embed_tokens / lm_head with
        # current LoRA adapters, which is a very hard-to-debug failure mode.
        latest_pointer = os.path.join(file_name, "latest")
        zero_ok = False
        if os.path.isfile(latest_pointer):
            try:
                with open(latest_pointer, "r") as fh:
                    target_tag = fh.read().strip()
            except Exception:
                target_tag = ""
            if target_tag and os.path.isdir(os.path.join(file_name, target_tag)):
                zero_ok = True

        if zero_ok:
            try:
                fp32_ckpt = get_fp32_state_dict_from_zero_checkpoint(file_name)
                vision_ckpt = get_rgb_maybe_zero_3(fp32_ckpt.items())
                other_ckpt = get_other_maybe_zero_3(fp32_ckpt.items())
                state_dict = dict(vision_ckpt=vision_ckpt, other_ckpt=other_ckpt)
            except Exception as exc:
                logger.warning(
                    "ZeRO consolidation failed for %s (%s); falling back to local FP32 export.",
                    file_name,
                    exc,
                )
                state_dict = self.custom_export_fp32_state()
        else:
            logger.info(
                "No valid DeepSpeed 'latest' pointer at %s; using live model.named_parameters() "
                "for FP32 export to avoid loading stale ZeRO shards from previous runs.",
                file_name,
            )
            state_dict = self.custom_export_fp32_state()

        self._save_text_lora_checkpoint(file_name)
        return state_dict

    def custom_export_fp32_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """

        """
        named_params = list(self.named_parameters())
        vision_ckpt = get_rgb_maybe_zero_3(named_params)
        other_ckpt = get_other_maybe_zero_3(named_params)

        # Authoritative grab of embed_tokens and lm_head straight from the live
        # PEFT-wrapped text encoder. Without this, the pattern-match in
        # get_other_maybe_zero_3 plus HF's _keep_in_fp32_modules behaviour can
        # silently capture an initial reference instead of the trained tensor.
        # Also captures lm_head, which the legacy names list missed entirely.
        try:
            text_encoder = self.language.get_text_encoder()
            input_emb = text_encoder.get_input_embeddings()
            # Pick a probe row index that's safely in-range for both legacy
            # vocab=33000 (loc-token) layouts and the new vocab=32000 layouts.
            def _probe_row(weight: torch.Tensor) -> int:
                return min(31999, weight.shape[0] - 1)

            if input_emb is not None and hasattr(input_emb, "weight"):
                live_in = maybe_zero_3(input_emb.weight, ignore_status=True, name="live_embed_tokens")
                other_ckpt.setdefault("embed_tokens", {})["weight"] = live_in
                probe = _probe_row(input_emb.weight)
                logger.info(
                    "[save] live embed_tokens[%d, :5]=%s shape=%s",
                    probe,
                    input_emb.weight.data[probe, :5].detach().cpu().float().tolist(),
                    tuple(input_emb.weight.shape),
                )
            output_emb = text_encoder.get_output_embeddings()
            if output_emb is not None and hasattr(output_emb, "weight"):
                live_out = maybe_zero_3(output_emb.weight, ignore_status=True, name="live_lm_head")
                other_ckpt.setdefault("lm_head", {})["weight"] = live_out
                probe = _probe_row(output_emb.weight)
                logger.info(
                    "[save] live lm_head[%d, :5]=%s shape=%s",
                    probe,
                    output_emb.weight.data[probe, :5].detach().cpu().float().tolist(),
                    tuple(output_emb.weight.shape),
                )
        except Exception as exc:
            logger.warning("Live embed/lm_head capture failed: %s", exc)

        return dict(vision_ckpt=vision_ckpt, other_ckpt=other_ckpt)
    # def custom_save_checkpoint(self, file_name: str):
    #     fp32_ckpt = get_fp32_state_dict_from_zero_checkpoint(file_name)
    #
    #     vision_ckpt = {k: v for k, v in fp32_ckpt.items() if k.startswith("vision.")}
    #     multimodal_ckpt = {k: v for k, v in fp32_ckpt.items() if k.startswith("multimodal.")}
    #
    #     if self.stage >= 2:
    #         file_name = pathlib.Path(file_name)
    #         if file_name.is_file():
    #             loar_output_path = file_name.parent / "TextLoRA"
    #         else:
    #             loar_output_path = file_name / "TextLoRA"
    #         self.language.text_encoder.save_pretrained(str(loar_output_path))
    #     # if self.stage >= 2:
    #     #     output_dir = pathlib.Path(file_name).parent / "TextLoRA"
    #     # self.language.text_encoder.save_pretrained(str(output_dir))
    #
    #     return dict(vision_ckpt=vision_ckpt, other_ckpt=other_ckpt)
    #     # return {
    #     #     "vision": vision_ckpt,
    #     #     "multimodal": multimodal_ckpt,
    #     #     "language": {k: v for k, v in fp32_ckpt.items() if k.startswith("language.")}
    #     # }

    def load_vision_encoder(self, path: str):
        """

        Args:
        """
        ckpt = torch.load(path, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        self.vision.encoder.load_state_dict(ckpt, strict=False)

    def _restore_embed_tokens_from_ckpt(self, ckpt, report) -> None:
        """Best-effort restore of text embedding rows saved in older checkpoints.

        Looks for ``ckpt["other_ckpt"]["embed_tokens"]`` (and optional
        ``lm_head``) and copies them onto the current text encoder. Silently
        no-ops when the keys are absent or shapes mismatch so it stays safe on
        legacy checkpoints.
        """
        other = ckpt.get("other_ckpt") if isinstance(ckpt, dict) else None
        if not isinstance(other, dict):
            return
        emb = other.get("embed_tokens")
        if isinstance(emb, dict) and "weight" in emb:
            try:
                text_encoder = self.language.get_text_encoder()
                input_layer = text_encoder.get_input_embeddings()
                ckpt_weight = emb["weight"]
                target_shape = tuple(input_layer.weight.shape)
                if tuple(ckpt_weight.shape) == target_shape:
                    msg = input_layer.load_state_dict({"weight": ckpt_weight}, strict=False)
                    report("embed_tokens", msg)
                else:
                    print(
                        f"[Inference] embed_tokens shape mismatch: "
                        f"ckpt={tuple(ckpt_weight.shape)} vs model={target_shape}; "
                        f"keeping current init."
                    )
            except Exception as exc:
                print(f"[Inference] failed to restore embed_tokens: {exc}")
        lm = other.get("lm_head")
        if isinstance(lm, dict) and "weight" in lm:
            try:
                text_encoder = self.language.get_text_encoder()
                output_layer = text_encoder.get_output_embeddings()
                if output_layer is not None:
                    ckpt_weight = lm["weight"]
                    target_shape = tuple(output_layer.weight.shape)
                    if tuple(ckpt_weight.shape) == target_shape:
                        msg = output_layer.load_state_dict({"weight": ckpt_weight}, strict=False)
                        report("lm_head", msg)
            except Exception as exc:
                print(f"[Inference] failed to restore lm_head: {exc}")

    def custom_load_state_dict(self, state_dict_path, strict=False):
        """


        Args:
        """
        # return None
        # if os.path.isdir(state_dict_path):
        #     if isinstance(self.language.text_encoder, PeftModel):
        #         self.language.text_encoder = self.language.text_encoder.merge_and_unload()
        #     return None

        ckpt = torch.load(state_dict_path, map_location="cpu")

        # module = ckpt.get('module', {})
        #
        # modified_module = {}
        # for key, value in module.items():
        #     if key.startswith('rgb.'):
        #         new_key = key.replace('rgb', 'vision', 1)
        #     elif key.startswith('rgb_pooler.'):
        #         new_key = key.replace('rgb_pooler', 'multimodal.projection', 1)
        #     elif key.startswith('text.'):
        #         new_key = key.replace('text', 'language', 1)
        #     else:
        #         new_key = key
        #
        #     modified_module[new_key] = value
        #
        # ckpt['module'] = modified_module
        #
        # torch.save(ckpt, '/root/shared-nvme/CoastGPT/Checkpoint/test2/checkpoints/iter_1299/test.pt')


        def _report_load_result(module_name, incompatible):
            print(
                f"After loading {module_name}: Missing: {incompatible.missing_keys}. "
                f"Unexpected: {incompatible.unexpected_keys}"
            )
            if not strict and incompatible.missing_keys:
                legacy_gate_keys = [
                    key for key in incompatible.missing_keys
                    if key.startswith("task_gate") or key.startswith("element_gate")
                ]
                if legacy_gate_keys:
                    print(
                        "Detected a legacy multimodal checkpoint without two-stage routing gates; "
                        "the missing gate parameters will keep their current initialization."
                    )

        if any(key.startswith('module') for key in ckpt.keys()):
            # filtered_state_dict = {k: v for k, v in ckpt["module"].items() if k.startswith("multimodal.")}
            filtered_state_dict = {k: v for k, v in ckpt["module"].items() if
                                   k.startswith("multimodal.") or k.startswith("vision.")}
            msg = self.load_state_dict(filtered_state_dict, strict=False)
            _report_load_result("model", msg)

        elif any(key.startswith('vision_ckpt') for key in ckpt.keys()) :
            vision_ckpt = ckpt["vision_ckpt"]
            multimodal_ckpt = ckpt["other_ckpt"]["multimodal_projection"]
            msg = self.vision.load_state_dict(vision_ckpt, strict=strict)
            _report_load_result("vision", msg)
            msg = self.multimodal.projection.load_state_dict(multimodal_ckpt, strict=strict)
            _report_load_result("multimodal", msg)
            # Restore text-encoder input embeddings for compatibility with
            # older checkpoints that saved these tensors separately.
            self._restore_embed_tokens_from_ckpt(ckpt, _report_load_result)

        elif any(key.startswith('rgb_ckpt') for key in ckpt.keys()) :
            vision_ckpt = ckpt["rgb_ckpt"]
            multimodal_ckpt = ckpt["other_ckpt"]["rgb_pooler"]
            msg = self.vision.load_state_dict(vision_ckpt, strict=strict)
            _report_load_result("vision", msg)
            msg = self.multimodal.projection.load_state_dict(multimodal_ckpt, strict=strict)
            _report_load_result("multimodal", msg)
        # if "vision" in ckpt:
        #     self.vision.load_state_dict(ckpt["vision"], strict=strict)
        # if "multimodal" in ckpt:
        #     self.multimodal.load_state_dict(ckpt["multimodal"], strict=strict)
        # if "language" in ckpt:
        #     self.language.load_state_dict(ckpt["language"], strict=strict)

        text_path = pathlib.Path(state_dict_path).parent / "TextLoRA"
        if text_path.exists():
            print(f"[Inference] loading TextLoRA from: {text_path}")
            self.language.text_encoder = PeftModel.from_pretrained(
                self.language.text_encoder,
                text_path,
                is_trainable=self.stage > 2,
                torch_dtype=torch.float16,
            )
            print("[Inference] TextLoRA load finished.")

            merge_text_lora = bool(getattr(self.config, "merge_text_lora", self.stage == 0))
            if self.stage == 0 and merge_text_lora:
                print("[Inference] merging TextLoRA into base model...")
                self.language.text_encoder = self.language.text_encoder.merge_and_unload()
                print("[Inference] TextLoRA merge finished.")
            elif self.stage == 0:
                print("[Inference] keep TextLoRA as adapter module without merge.")
        else:
            print(f"[Inference] TextLoRA directory not found, skip LoRA load: {text_path}")
        return None

        # if "model" in ckpt.keys():
        #
        # self.vision.load_state_dict(ckpt["rgb_ckpt"], strict=strict)
        #
        # if text_path.exists():
        #     self.language.text_encoder = PeftModel.from_pretrained(
        #         self.language.text_encoder,
        #         text_path,
        #     )
        #
        #         self.language.text_encoder = self.language.text_encoder.merge_and_unload()
        #
        # return None

    def prepare_for_training(
            self,
            freeze_vision: bool = False,
            freeze_text: bool = False,
            tune_multimodal: bool = False,
            model_path: str = None,
            tune_im_start: bool = False,
            compute_dtype: torch.dtype = torch.float32,
    ):
        """

        Args:
        """
        self.train()

        for param in self.vision.parameters():
            if freeze_vision:
                param.requires_grad = False
            else:
                param.requires_grad = True
            param.data = param.data.to(dtype=compute_dtype)

        for name, buffer in self.vision.named_buffers():
            if "index" not in name and "id" not in name:
                buffer.data = buffer.data.to(dtype=compute_dtype)

        text_encoder = self.language.get_text_encoder()
        if freeze_text:
            self.language.eval()
            for p in self.language.parameters():
                p.requires_grad = False
            for p in text_encoder.parameters():
                p.requires_grad = False
        else:
            self.language.train()
            # Freeze input/output embeddings to avoid vocabulary drift on the
            # original LLaMA tokens while LoRA and multimodal layers train.
            for p in text_encoder.get_input_embeddings().parameters():
                p.requires_grad = False
            for p in text_encoder.get_output_embeddings().parameters():
                p.requires_grad = False

        for param in self.multimodal.parameters():
            if tune_multimodal:
                param.requires_grad = True
            else:
                param.requires_grad = False
            param.data = param.data.to(dtype=compute_dtype)

        if getattr(self, "physics_enabled", False) and hasattr(self, "physics"):
            for p in self.physics.parameters():
                p.requires_grad = True
                p.data = p.data.to(dtype=torch.float32)
            for name, buffer in self.physics.named_buffers():
                buffer.data = buffer.data.to(dtype=torch.float32)

        if tune_im_start and freeze_text:
            for p in text_encoder.get_input_embeddings().parameters():
                p.requires_grad = True

        if model_path is not None:
            self.custom_load_state_dict(model_path)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logger.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_other_maybe_zero_3(named_params):
    names = ["multimodal.projection", "embed_tokens", "physics"]
    multimodal_projection = dict()
    text_proj = dict()
    embed_tokens = dict()
    lm_head = dict()
    physics_ckpt = dict()

    params = list(named_params)
    to_return = dict(
        multimodal_projection=multimodal_projection,
        embed_tokens=embed_tokens,
        text_proj=text_proj,
        lm_head=lm_head,
        physics=physics_ckpt,
    )
    for k, v in params:
        for name in names:
            if name in k:
                if name == "multimodal.projection":
                    to_return["multimodal_projection"][k.split(name + ".")[-1]] = maybe_zero_3(v, ignore_status=True, name=k)
                elif name == "embed_tokens":
                    to_return["embed_tokens"][k.split(name + ".")[-1]] = maybe_zero_3(v, ignore_status=True, name=k)
                elif name == "physics":
                    to_return["physics"][k.split(name + ".")[-1]] = maybe_zero_3(v, ignore_status=True, name=k)

    return to_return
# def get_other_maybe_zero_3(named_params):
#     names = ["multimodal.projection", "embed_tokens"]
#     to_return = {"multimodal_projection": {}, "embed_tokens": {}}
#     for k, v in named_params:
#         if any(name in k for name in names):
#             if "multimodal.projection" in k:
#             elif "embed_tokens" in k:
#     return to_return


def get_rgb_maybe_zero_3(named_params):
    to_return = {k[len("vision.") :]: t for k, t in named_params if "vision." in k}
    # to_return = {k: t for k, t in named_params if k.startswith("vision.")}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return
