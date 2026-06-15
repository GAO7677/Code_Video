from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from code_vjepa_vggt.utils.paths import ensure_upstream_paths

ensure_upstream_paths()

import app.vjepa_2_1.models.vision_transformer as video_vit  # type: ignore
from app.vjepa_2_1.wrappers import MultiSeqWrapper  # type: ignore


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
        use_activation_checkpointing: bool = False,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.device_obj = torch.device(device)
        self.trainable = bool(trainable)
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        resolved_model_name = model_name
        encoder_kwargs = dict(
            img_size=crop_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            uniform_power=False,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
            use_activation_checkpointing=bool(use_activation_checkpointing),
            is_causal=False,
            init_type="default",
            img_temporal_dim_size=None,
            n_registers=0,
            has_cls_first=False,
            interpolate_rope=False,
            modality_embedding=False,
        )
        if resolved_model_name == "vit_giant_xformers":
            resolved_model_name = "vit_giant_xformers_rope"
        else:
            encoder_kwargs["use_rope"] = True

        encoder_backbone = video_vit.__dict__[resolved_model_name](**encoder_kwargs)
        encoder = MultiSeqWrapper(encoder_backbone).to(self.device_obj)
        state = torch.load(ckpt_path, map_location="cpu")
        encoder.load_state_dict(self._select_encoder_state(state), strict=False)
        if self.trainable:
            self.encoder = encoder.train().to(self.device_obj)
        else:
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

        with torch.set_grad_enabled(self.trainable and torch.is_grad_enabled()):
            feats_nested = self.encoder(
                [resized],
                masks=None,
                gram_mode=False,
                training_mode=self.trainable,
            )
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
