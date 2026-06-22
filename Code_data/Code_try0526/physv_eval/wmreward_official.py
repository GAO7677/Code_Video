from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .paths import VPHY_PYTHON, WMREWARD_ROOT, WMREWARD_VJEPA2_ROOT


FALLBACK_WMREWARD_ROOTS = [
    Path("/home/gaoya/Code_Video/WMReward-main"),
    WMREWARD_ROOT,
]
FALLBACK_VJEPA2_ROOTS = [
    WMREWARD_VJEPA2_ROOT,
    Path("/home/gaoya/Code_Video/vjepa2-main"),
]


class WMRewardRunner:
    """Batch-friendly runner numerically aligned with official compute_wmreward.py defaults.

    This reuses the upstream script helpers directly and keeps the same default
    model selection and scoring parameters as:

        python compute_wmreward.py --video_path <video>

    The only optimization is caching the loaded VJEPA model in-process so many
    videos can be scored without reloading weights every time.
    """

    def __init__(
        self,
        *,
        cuda_visible_devices: str | None = None,
        model_name: str = "vitg384",
        window_size: int = 16,
        context_frames: int = 8,
        stride: int = 8,
        seed: int = 42,
    ) -> None:
        self.cuda_visible_devices = cuda_visible_devices
        self.model_name = model_name
        self.window_size = window_size
        self.context_frames = context_frames
        self.stride = stride
        self.seed = seed

        self._torch = None
        self._load_vjepa_models = None
        self._load_video_as_tensor = None
        self._compute_loss = None
        self._load_vjepa_models_local = None
        self._models: tuple[Any, Any, Any, int] | None = None
        self._device = None
        self._wmreward_root = None
        self._vjepa2_root = None

    def _resolve_roots(self) -> tuple[Path, Path]:
        if self._wmreward_root is not None and self._vjepa2_root is not None:
            return self._wmreward_root, self._vjepa2_root

        wmreward_root = next(
            (root for root in FALLBACK_WMREWARD_ROOTS if (root / "compute_wmreward.py").is_file() and (root / "utils.py").is_file()),
            None,
        )
        if wmreward_root is None:
            raise FileNotFoundError("WMReward root not found")

        vjepa2_root = next(
            (root for root in FALLBACK_VJEPA2_ROOTS if (root / "src").is_dir()),
            None,
        )
        if vjepa2_root is None:
            raise FileNotFoundError("VJEPA2 root for WMReward not found")

        self._wmreward_root = wmreward_root
        self._vjepa2_root = vjepa2_root
        return wmreward_root, vjepa2_root

    def _lazy_imports(self) -> None:
        if self._torch is not None:
            return

        if self.cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
        os.environ.setdefault("PYTHONNOUSERSITE", "1")

        import torch
        self._torch = torch

        try:
            import decord  # noqa: F401
        except Exception:
            vphy_site_packages = VPHY_PYTHON.parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
            if vphy_site_packages.is_dir():
                site_path = str(vphy_site_packages)
                if site_path not in sys.path:
                    sys.path.append(site_path)
            import decord  # noqa: F401

        wmreward_root, vjepa2_root = self._resolve_roots()
        vjepa2_root_str = str(vjepa2_root)
        vjepa2_src = str(vjepa2_root / "src")
        for path in [str(wmreward_root), vjepa2_root_str, vjepa2_src]:
            if path not in sys.path:
                sys.path.insert(0, path)

        from compute_wmreward import load_video_as_tensor
        from utils import compute_vjepa_loss_sliding_window, load_vjepa_model_source

        self._load_vjepa_models = None
        self._load_vjepa_models_local = load_vjepa_model_source
        self._load_video_as_tensor = load_video_as_tensor
        self._compute_loss = compute_vjepa_loss_sliding_window
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_models_once(self) -> tuple[Any, Any, Any, int]:
        if self._models is not None:
            return self._models

        self._lazy_imports()
        encoder, target_encoder, predictor, img_size = self._load_vjepa_models_local(self.model_name)
        encoder = encoder.to(self._device).eval()
        target_encoder = target_encoder.to(self._device).eval()
        predictor = predictor.to(self._device).eval()
        self._models = (encoder, target_encoder, predictor, img_size)
        return self._models

    def score(self, video_path: Path) -> dict[str, Any]:
        encoder, target_encoder, predictor, img_size = self._load_models_once()
        video_tensor = self._load_video_as_tensor(str(video_path), max_frames=49, img_size=img_size)
        video_tensor = video_tensor.to(self._device)

        with self._torch.no_grad():
            loss = self._compute_loss(
                video_tensor=video_tensor,
                encoder=encoder,
                target_encoder=target_encoder,
                predictor=predictor,
                img_size=img_size,
                window_size=self.window_size,
                loss_exp=2,
                masking_mode="causal",
                context_frames=self.context_frames,
                is_vae_output=True,
                seed=self.seed,
                stride=self.stride,
                mode="mean",
            )

        surprise = float(loss.item())
        return {
            "surprise": surprise,
            "similarity": 1.0 - surprise,
            "method": "official compute_wmreward.py batched-equivalent",
            "model": self.model_name,
            "img_size": img_size,
            "window_size": self.window_size,
            "context_frames": self.context_frames,
            "stride": self.stride,
            "seed": self.seed,
        }
