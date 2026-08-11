"""Frozen V-JEPA2.1 ViT-L/16 video adapter for xSSC."""

import importlib
import os
from pathlib import Path
import sys

from einops import rearrange
import torch
import torch.nn as nn


DEFAULT_VJEPA2_ROOT = Path("/home/gaoya/Code_Video/vjepa2-main")
DEFAULT_VJEPA2_CHECKPOINT = Path(
    "/data/gaoya/agent-data/weights/"
    "vjepa2_1_vitl_dist_vitG_384_ema_encoder.pt"
)


class VJEPA21VideoViT(nn.Module):
    """Return native two-frame tubelet features as ``[B,T/2,C,H,W]``."""

    def __init__(
        self,
        model_name="vjepa2_1_vit_large_384",
        checkpoint=DEFAULT_VJEPA2_CHECKPOINT,
        source_root=DEFAULT_VJEPA2_ROOT,
        in_size=256,
        patch_size=16,
        tubelet_size=2,
        temporal_mode="noncausal",
    ):
        super().__init__()
        if model_name != "vjepa2_1_vit_large_384":
            raise ValueError(f"Unsupported V-JEPA2.1 model: {model_name}")
        if int(in_size) != 256 or int(patch_size) != 16 or int(tubelet_size) != 2:
            raise ValueError(
                "This controlled experiment requires reference input=256, patch=16, "
                f"tubelet=2; got {in_size}, {patch_size}, {tubelet_size}"
            )
        if temporal_mode not in {"noncausal", "prefix_causal"}:
            raise ValueError(
                "temporal_mode must be 'noncausal' or 'prefix_causal', "
                f"got {temporal_mode!r}"
            )

        source_root = Path(
            os.environ.get("VJEPA2_ROOT", source_root)
        ).expanduser().resolve()
        checkpoint = Path(
            os.environ.get("VJEPA2_CHECKPOINT", checkpoint)
        ).expanduser().resolve()
        if not (source_root / "src/hub/backbones.py").is_file():
            raise FileNotFoundError(f"V-JEPA2 source repository not found: {source_root}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"V-JEPA2.1 checkpoint not found: {checkpoint}")

        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        backbones = importlib.import_module("src.hub.backbones")
        module_path = Path(backbones.__file__).resolve()
        if source_root != module_path.parent and source_root not in module_path.parents:
            raise ImportError(
                f"Imported src.hub.backbones from {module_path}, expected it below {source_root}"
            )

        encoder, predictor = backbones.vjepa2_1_vit_large_384(pretrained=False)
        del predictor
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if "ema_encoder" not in payload:
            raise KeyError(f"Checkpoint has no ema_encoder: {checkpoint}")
        state = {
            key.removeprefix("module.").removeprefix("backbone."): value
            for key, value in payload["ema_encoder"].items()
        }
        load_result = encoder.load_state_dict(state, strict=True)
        del state, payload

        encoder.requires_grad_(False)
        encoder.eval()
        self.model = encoder
        self.in_size = int(in_size)
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.temporal_mode = temporal_mode
        self.reference_grid_size = self.in_size // self.patch_size
        self.embed_dim = int(encoder.embed_dim)
        self.checkpoint = str(checkpoint)
        self.source_root = str(source_root)
        self.load_report = str(load_result)

        if self.reference_grid_size != 16 or self.embed_dim != 1024:
            raise RuntimeError(
                "Unexpected V-JEPA2.1 ViT-L geometry: "
                f"reference_grid={self.reference_grid_size}, dim={self.embed_dim}"
            )

    def train(self, mode=True):
        """Keep the frozen encoder in evaluation mode when the xSSC head trains."""
        super().train(False)
        self.model.eval()
        return self

    def _encode_video(self, input):
        temporal_tokens = input.shape[1] // self.tubelet_size
        grid_height = input.shape[-2] // self.patch_size
        grid_width = input.shape[-1] // self.patch_size
        video = rearrange(input, "b t c h w -> b c t h w")
        tokens = self.model(video)
        expected_tokens = temporal_tokens * grid_height * grid_width
        if tokens.shape[1:] != (expected_tokens, self.embed_dim):
            raise RuntimeError(
                "Unexpected V-JEPA2.1 output shape: "
                f"{tuple(tokens.shape)} != [B,{expected_tokens},{self.embed_dim}]"
            )
        return rearrange(
            tokens,
            "b (t h w) c -> b t c h w",
            t=temporal_tokens,
            h=grid_height,
            w=grid_width,
        )

    def forward(self, input):
        if input.ndim != 5 or input.shape[2] != 3:
            raise ValueError(
                f"V-JEPA2.1 video input must be [B,T,3,H,W], got {tuple(input.shape)}"
            )
        height, width = input.shape[-2:]
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"V-JEPA2.1 input is smaller than one patch: {(height, width)}"
            )
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                "V-JEPA2.1 rectangular input dimensions must be divisible by "
                f"patch_size={self.patch_size}, got {(height, width)}"
            )
        if input.shape[1] % self.tubelet_size:
            raise ValueError(
                "V-JEPA2.1 video input must contain a whole number of tubelets; "
                f"got {input.shape[1]} frames for tubelet_size={self.tubelet_size}"
            )
        temporal_tokens = input.shape[1] // self.tubelet_size
        if temporal_tokens < 1:
            raise ValueError("V-JEPA2.1 video input requires at least two frames")

        self.model.eval()
        if self.temporal_mode == "noncausal":
            return self._encode_video(input)

        # Prefix encoding is the reference implementation of block-temporal
        # causality: a tubelet can attend bidirectionally within its spatial
        # tokens and to all earlier tubelets, but never to a future tubelet.
        outputs = []
        for end in range(self.tubelet_size, input.shape[1] + 1, self.tubelet_size):
            outputs.append(self._encode_video(input[:, :end])[:, -1:])
        return torch.cat(outputs, dim=1)
