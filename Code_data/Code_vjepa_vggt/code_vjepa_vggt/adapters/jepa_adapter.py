from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from code_vjepa_vggt.utils.paths import ensure_upstream_paths

ensure_upstream_paths()

from app.vjepa_2_1.utils import init_video_model  # type: ignore


@dataclass
class JEPAAdapterOutput:
    patch_tokens: torch.Tensor
    input_hw: tuple[int, int]
    token_grid_hw: tuple[int, int]
    token_grid_t: int


class JEPAPatchAdapter(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        crop_size: int = 384,
        num_frames: int = 8,
        patch_size: int = 16,
        tubelet_size: int = 2,
        model_name: str = "vit_giant_xformers",
        pred_embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.device_obj = torch.device(device)
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        encoder, _predictor = init_video_model(
            device=self.device_obj,
            patch_size=patch_size,
            max_num_frames=num_frames,
            tubelet_size=tubelet_size,
            model_name=model_name,
            crop_size=crop_size,
            pred_depth=6,
            pred_num_heads=None,
            pred_embed_dim=pred_embed_dim,
            uniform_power=False,
            use_mask_tokens=False,
            num_mask_tokens=2,
            zero_init_mask_tokens=True,
            use_sdpa=False,
            use_rope=True,
            use_silu=False,
            use_pred_silu=False,
            wide_silu=True,
            is_causal=False,
            pred_is_causal=False,
            use_activation_checkpointing=False,
            return_all_tokens=False,
            chop_last_n_tokens=0,
            init_type="default",
            img_temporal_dim_size=None,
            n_registers=0,
            n_registers_predictor=0,
            has_cls_first=False,
            interpolate_rope=False,
            modality_embedding=False,
        )
        state = torch.load(ckpt_path, map_location="cpu")
        encoder.load_state_dict(self._select_encoder_state(state), strict=False)
        self.encoder = encoder.eval().requires_grad_(False).to(self.device_obj)

    @staticmethod
    def _select_encoder_state(state: dict[str, Any]) -> dict[str, torch.Tensor]:
        if "target_encoder" in state:
            return state["target_encoder"]
        if "encoder" in state:
            return state["encoder"]
        return state

    def forward(self, video_bcthw: torch.Tensor) -> JEPAAdapterOutput:
        batch, _, frames, _, _ = video_bcthw.shape
        resized = F.interpolate(
            video_bcthw.permute(0, 2, 1, 3, 4).reshape(-1, 3, video_bcthw.shape[-2], video_bcthw.shape[-1]),
            size=(self.crop_size, self.crop_size),
            mode="bilinear",
            align_corners=False,
        ).view(batch, frames, 3, self.crop_size, self.crop_size).permute(0, 2, 1, 3, 4)

        with torch.no_grad():
            feats_nested = self.encoder([resized], masks=None, gram_mode=False, training_mode=False)
            feats = feats_nested[0]

        token_t = max(1, frames // self.tubelet_size)
        token_h = self.crop_size // self.patch_size
        token_w = self.crop_size // self.patch_size
        expected = token_t * token_h * token_w
        if feats.shape[1] != expected:
            raise RuntimeError(f"unexpected JEPA token count: got {feats.shape[1]}, expected {expected}")

        patch_tokens = feats.view(batch, token_t, token_h, token_w, feats.shape[-1])
        return JEPAAdapterOutput(
            patch_tokens=patch_tokens,
            input_hw=(self.crop_size, self.crop_size),
            token_grid_hw=(token_h, token_w),
            token_grid_t=token_t,
        )
