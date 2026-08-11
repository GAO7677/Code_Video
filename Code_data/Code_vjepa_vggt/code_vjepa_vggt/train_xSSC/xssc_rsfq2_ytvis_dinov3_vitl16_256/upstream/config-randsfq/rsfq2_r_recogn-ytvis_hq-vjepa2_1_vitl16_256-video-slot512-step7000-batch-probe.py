"""One-step, one-validation-sample batch-size probe for the step-7000 pilot."""

import os as _os
from pathlib import Path as _Path

from object_centric_bench.util import importlib_cfg as _importlib_cfg


_base = _importlib_cfg(
    _Path(__file__).with_name(
        "rsfq2_r_recogn-ytvis_hq-vjepa2_1_vitl16_256-video-"
        "slot512-step7000-pilot.py"
    ),
    name="_vjepa_step7000_downstream_pilot",
)
globals().update(_base)

batch_size_t = int(_os.environ["XSSC_PROBE_BATCH_SIZE"])
batch_size_v = 1
num_work = min(16, max(4, batch_size_t // 16))
val_subset_size = 1
variant_name = f"{variant_name}_batch_probe_bs{batch_size_t}"
checkpoint_key = r"^m\.recogn\..*"

del _base, _importlib_cfg, _os, _Path
