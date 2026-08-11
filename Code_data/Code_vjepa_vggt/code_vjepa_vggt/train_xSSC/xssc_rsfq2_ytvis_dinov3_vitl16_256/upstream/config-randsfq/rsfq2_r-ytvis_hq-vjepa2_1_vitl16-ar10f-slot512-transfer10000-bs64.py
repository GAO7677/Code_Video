"""Batch-64/GPU variant of the 10-frame aspect-ratio YTVIS continuation."""

from pathlib import Path as _Path

from object_centric_bench.util import importlib_cfg as _importlib_cfg


_base = _importlib_cfg(
    _Path(__file__).with_name(
        "rsfq2_r-ytvis_hq-vjepa2_1_vitl16-ar10f-slot512-transfer10000.py"
    ),
    name="_vjepa_10f_ar_transfer10000_base",
)
globals().update(_base)

variant_name = (
    "vjepa2_1_vitl16_video_ytvis_hq_10f_ar_slot512_transfer10000_bs64"
)
batch_size_t = 64
gradient_accumulation_steps = 3
effective_global_batch_size = (
    batch_size_t * expected_world_size * gradient_accumulation_steps
)

del _base, _Path, _importlib_cfg
