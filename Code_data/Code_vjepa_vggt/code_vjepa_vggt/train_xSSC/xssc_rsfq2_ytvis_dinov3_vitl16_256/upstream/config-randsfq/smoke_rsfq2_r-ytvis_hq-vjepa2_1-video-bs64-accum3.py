"""Two-GPU integration config for the formal batch-64 schedule."""

from pathlib import Path as _Path

from object_centric_bench.util import importlib_cfg as _importlib_cfg


_base = _importlib_cfg(
    _Path(__file__).with_name(
        "rsfq2_r-ytvis_hq-vjepa2_1_vitl16_256-video-slot512.py"
    ),
    name="_xssc_vjepa_bs64_smoke_base",
)
globals().update(_base)

val_subset_size = 2
checkpoint_interval = 1
checkpoint_keep_steps = [1]

del _base, _Path, _importlib_cfg

