"""Resume the 10-frame YTVIS run at step 14k with global grad clip 2.0."""

from pathlib import Path as _Path

from object_centric_bench.util import importlib_cfg as _importlib_cfg


_base = _importlib_cfg(
    _Path(__file__).with_name(
        "rsfq2_r-ytvis_hq-vjepa2_1_vitl16-ar10f-slot512-transfer10000-bs64.py"
    ),
    name="_vjepa_10f_ar_transfer10000_bs64_clip2_base",
)
globals().update(_base)

# Keep the source variant name unchanged so the full step-14000 training state
# (model, Adam moments, scaler, RNG, and sampler epoch) can be resumed strictly.
# The config filename, save directory, and W&B project identify this fork.
gradient_clip_norm = 2.0
gclip["max_norm"] = gradient_clip_norm
fork_source_step = 14000
fork_description = "YTVIS 10f AR continuation from step 14000; global grad clip 0.05 -> 2.0"

del _base, _Path, _importlib_cfg
