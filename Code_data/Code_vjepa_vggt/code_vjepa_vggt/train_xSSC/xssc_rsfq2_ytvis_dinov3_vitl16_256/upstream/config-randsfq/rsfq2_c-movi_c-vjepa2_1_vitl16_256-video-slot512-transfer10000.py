"""Noncausal V-JEPA2.1 xSSC transfer from YTVIS step-10000 to MOVi-C."""

from pathlib import Path as _Path

from object_centric_bench.util import importlib_cfg as _importlib_cfg


_base = _importlib_cfg(
    _Path(__file__).with_name(
        "rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-slot512-transfer15000.py"
    ),
    name="_xssc_vjepa_noncausal_movic_transfer15000_base",
)
globals().update(_base)

variant_name = (
    "vjepa2_1_vitl16_video_256_movi_c_slot512_transfer10000_native_tubelet"
)
start_step = 10000
transfer_expected_source_step = start_step

# Match the completed YTVIS run and the DINOv3 MOVi-C reference: effective
# global batch = 64 samples/GPU * 2 GPUs * 3 accumulated microbatches = 384.
batch_size_t = 64
gradient_accumulation_steps = 3
effective_global_batch_size = (
    batch_size_t * expected_world_size * gradient_accumulation_steps
)

# Keep every requested 1k checkpoint for downstream comparisons.
checkpoint_keep_steps = list(
    range(start_step + checkpoint_interval, total_step + 1, checkpoint_interval)
)

del _base, _Path, _importlib_cfg
