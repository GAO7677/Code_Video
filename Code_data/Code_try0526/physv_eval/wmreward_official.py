from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any

from .paths import VJEPA2_ROOT, WMREWARD_CKPT, WMREWARD_ROOT


class WMRewardRunner:
    def __init__(
        self,
        *,
        device: str = "cuda",
        autocast_dtype: str = "bfloat16",
        window_size: int = 16,
        context_frames: int = 8,
        stride: int = 2,
    ) -> None:
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.window_size = window_size
        self.context_frames = context_frames
        self.stride = stride
        self._torch = None
        self._resize = None
        self._compute_loss = None
        self._get_video = None
        self._models: tuple[Any, Any, Any] | None = None

    def _lazy_imports(self) -> None:
        if self._torch is not None:
            return
        os.environ.setdefault("PYTHONNOUSERSITE", "1")
        sys.path.insert(0, str(VJEPA2_ROOT))
        sys.path.insert(0, str(VJEPA2_ROOT / "src"))
        sys.path.insert(0, str(WMREWARD_ROOT))

        import torch
        from torchvision.transforms.functional import resize
        from models.predictor import vit_predictor
        from models.vision_transformer import vit_giant_xformers_rope
        from utils import compute_vjepa_loss_sliding_window, get_video

        self._torch = torch
        self._resize = resize
        self._compute_loss = compute_vjepa_loss_sliding_window
        self._get_video = get_video
        self._vit_predictor = vit_predictor
        self._vit_giant = vit_giant_xformers_rope

    def _model_dtype(self) -> Any | None:
        if self.autocast_dtype == "bfloat16":
            return self._torch.bfloat16
        if self.autocast_dtype == "float16":
            return self._torch.float16
        return None

    def _load_models(self) -> tuple[Any, Any, Any]:
        if self._models is not None:
            return self._models
        self._lazy_imports()
        torch = self._torch
        encoder = self._vit_giant(
            img_size=(384, 384),
            num_frames=16,
            patch_size=16,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
        )
        predictor = self._vit_predictor(
            img_size=(384, 384),
            patch_size=16,
            num_frames=16,
            tubelet_size=2,
            use_mask_tokens=True,
            embed_dim=encoder.embed_dim,
            predictor_embed_dim=384,
            depth=12,
            num_heads=12,
            num_mask_tokens=10,
            uniform_power=False,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
        )
        checkpoint = torch.load(str(WMREWARD_CKPT), map_location="cpu")

        def clean(state_dict: dict[str, Any]) -> dict[str, Any]:
            return {
                key.replace("module.", "").replace("backbone.", ""): value
                for key, value in state_dict.items()
            }

        encoder.load_state_dict(clean(checkpoint["target_encoder"]), strict=False)
        predictor.load_state_dict(clean(checkpoint["predictor"]), strict=False)
        target_encoder = copy.deepcopy(encoder)

        dtype = self._model_dtype()
        encoder = encoder.eval().to(self.device)
        target_encoder = target_encoder.eval().to(self.device)
        predictor = predictor.eval().to(self.device)
        if dtype is not None and str(self.device).startswith("cuda"):
            encoder = encoder.to(dtype=dtype)
            target_encoder = target_encoder.to(dtype=dtype)
            predictor = predictor.to(dtype=dtype)

        self._models = (encoder, target_encoder, predictor)
        return self._models

    def _load_video_tensor(self, video_path: Path) -> Any:
        torch = self._torch
        video_np = self._get_video(str(video_path), max_frames=49)
        video_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float()
        video_tensor = self._resize(video_tensor.permute(1, 0, 2, 3), [384, 384]).permute(1, 0, 2, 3)
        return ((video_tensor / 127.5) - 1.0).unsqueeze(0).to(self.device)

    def score(self, video_path: Path) -> dict[str, Any]:
        self._lazy_imports()
        torch = self._torch
        encoder, target_encoder, predictor = self._load_models()
        video_tensor = self._load_video_tensor(video_path)

        contexts: list[Any] = [torch.inference_mode()]
        dtype = self._model_dtype()
        if dtype is not None and str(self.device).startswith("cuda"):
            contexts.append(torch.autocast(device_type="cuda", dtype=dtype))

        with contexts[0]:
            if len(contexts) == 2:
                with contexts[1]:
                    loss = self._compute_loss(
                        video_tensor=video_tensor,
                        encoder=encoder,
                        target_encoder=target_encoder,
                        predictor=predictor,
                        img_size=384,
                        window_size=self.window_size,
                        loss_exp=2,
                        masking_mode="causal",
                        context_frames=self.context_frames,
                        is_vae_output=True,
                        seed=42,
                        stride=self.stride,
                        mode="mean",
                    )
            else:
                loss = self._compute_loss(
                    video_tensor=video_tensor,
                    encoder=encoder,
                    target_encoder=target_encoder,
                    predictor=predictor,
                    img_size=384,
                    window_size=self.window_size,
                    loss_exp=2,
                    masking_mode="causal",
                    context_frames=self.context_frames,
                    is_vae_output=True,
                    seed=42,
                    stride=self.stride,
                    mode="mean",
                )

        surprise = float(loss.item())
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        return {
            "surprise": surprise,
            "similarity": 1.0 - surprise,
            "window_size": self.window_size,
            "context_frames": self.context_frames,
            "stride": self.stride,
        }
