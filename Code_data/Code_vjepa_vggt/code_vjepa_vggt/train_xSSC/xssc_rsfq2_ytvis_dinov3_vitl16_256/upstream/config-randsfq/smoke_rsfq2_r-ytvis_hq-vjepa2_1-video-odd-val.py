"""Two-GPU integration config for non-divisible validation cardinality."""

from pathlib import Path as _Path

from object_centric_bench.util import importlib_cfg as _importlib_cfg


_base = _importlib_cfg(
    _Path(__file__).with_name(
        "smoke_rsfq2_r-ytvis_hq-vjepa2_1-video-noncausal.py"
    ),
    name="_xssc_vjepa_odd_val_smoke_base",
)
globals().update(_base)

val_subset_size = 3

del _base, _Path, _importlib_cfg

