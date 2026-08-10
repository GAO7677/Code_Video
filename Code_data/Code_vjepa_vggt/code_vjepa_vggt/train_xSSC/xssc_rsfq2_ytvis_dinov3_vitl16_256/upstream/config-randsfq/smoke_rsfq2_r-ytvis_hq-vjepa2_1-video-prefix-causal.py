"""Two-GPU one-step integration config for prefix-causal V-JEPA xSSC."""

from pathlib import Path as _Path

from object_centric_bench.util import importlib_cfg as _importlib_cfg


_base = _importlib_cfg(
    _Path(__file__).with_name(
        "rsfq2_r-ytvis_hq-vjepa2_1_vitl16_256-video-slot512-prefix-causal.py"
    ),
    name="_xssc_vjepa_prefix_causal_smoke_base",
)
globals().update(_base)

batch_size_t = 1
gradient_accumulation_steps = 1
effective_global_batch_size = batch_size_t * expected_world_size
num_work = 1
val_subset_size = 2
checkpoint_interval = 1
checkpoint_keep_steps = [1, 2]

del _base, _Path, _importlib_cfg
