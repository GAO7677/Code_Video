from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

from .utils import require_torch

torch = require_torch()


DEFAULT_WAN_REPO_ROOT = Path("/home/gaoya/Code_Video/Wan2.2-main")


def resolve_wan_repo_root(wan_repo_root: str | Path | None = None) -> Path:
    root = Path(wan_repo_root) if wan_repo_root is not None else DEFAULT_WAN_REPO_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Wan repo root does not exist: {root}")
    return root


def ensure_wan_importable(wan_repo_root: str | Path | None = None) -> Path:
    root = resolve_wan_repo_root(wan_repo_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_wan_modules(wan_repo_root: str | Path | None = None) -> dict[str, Any]:
    ensure_wan_importable(wan_repo_root)

    original_current_device = getattr(torch.cuda, "current_device", None)
    patched_current_device = False
    if original_current_device is not None:
        try:
            original_current_device()
        except Exception:
            warnings.warn(
                "Wan upstream imports touched torch.cuda.current_device() during module import. "
                "Applying a temporary CPU-safe compatibility patch in load_wan_modules(); "
                "this should be treated as an environment/upstream compatibility workaround, not normal flow.",
                RuntimeWarning,
                stacklevel=2,
            )
            torch.cuda.current_device = lambda: 0
            patched_current_device = True
    try:
        from wan_.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
        from wan_.image2video import WanI2V
        from wan_.textimage2video import WanTI2V
        from wan_.modules.t5 import T5EncoderModel
        from wan_.modules.vae2_1 import Wan2_1_VAE
        from wan_.modules.vae2_2 import Wan2_2_VAE
    finally:
        if patched_current_device and original_current_device is not None:
            torch.cuda.current_device = original_current_device

    return {
        "WAN_CONFIGS": WAN_CONFIGS,
        "SIZE_CONFIGS": SIZE_CONFIGS,
        "MAX_AREA_CONFIGS": MAX_AREA_CONFIGS,
        "SUPPORTED_SIZES": SUPPORTED_SIZES,
        "WanI2V": WanI2V,
        "WanTI2V": WanTI2V,
        "T5EncoderModel": T5EncoderModel,
        "Wan2_1_VAE": Wan2_1_VAE,
        "Wan2_2_VAE": Wan2_2_VAE,
    }
