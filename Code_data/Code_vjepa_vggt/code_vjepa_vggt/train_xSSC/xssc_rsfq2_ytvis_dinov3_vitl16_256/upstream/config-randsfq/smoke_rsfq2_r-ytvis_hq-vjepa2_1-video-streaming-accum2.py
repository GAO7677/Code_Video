"""Two-GPU streaming gradient-accumulation integration config."""

from pathlib import Path as _Path

from object_centric_bench.util import importlib_cfg as _importlib_cfg


_base = _importlib_cfg(
    _Path(__file__).with_name(
        "smoke_rsfq2_r-ytvis_hq-vjepa2_1-video-noncausal.py"
    ),
    name="_xssc_vjepa_streaming_accum_smoke_base",
)
globals().update(_base)

gradient_accumulation_steps = 2
effective_global_batch_size = (
    batch_size_t * expected_world_size * gradient_accumulation_steps
)

del _base, _Path, _importlib_cfg

