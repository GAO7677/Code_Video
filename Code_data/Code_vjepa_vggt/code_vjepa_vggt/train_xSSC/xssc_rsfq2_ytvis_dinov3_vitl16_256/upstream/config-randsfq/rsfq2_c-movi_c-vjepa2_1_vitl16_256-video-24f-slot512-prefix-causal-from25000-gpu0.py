"""GPU-0 Stage-1 causal-adaptation branch from MOVi-C step 25k.

The per-GPU microbatch is selected by the launch-time capacity probe.  The
effective batch is held at 384 so this branch remains comparable with the
source noncausal run despite changing from two GPUs to one.
"""

import os as _os
from pathlib import Path as _Path

from object_centric_bench.learn import CbLinearCosineRestart
from object_centric_bench.util import importlib_cfg as _importlib_cfg


_base = _importlib_cfg(
    _Path(__file__).with_name(
        "rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-24f-slot512-"
        "prefix-causal-stage1.py"
    ),
    name="_xssc_vjepa_movic_24f_prefix_causal_stage1_base",
)
globals().update(_base)

variant_name = (
    "vjepa2_1_vitl16_video_256_movi_c_24f_slot512_"
    "prefix_causal_from25000_gpu0"
)

# This is a model-only fork.  Causal adaptation receives a fresh Adam state
# and a new learning-rate phase; the source checkpoint and run stay untouched.
start_step = 25000
total_step = 35000
max_step = total_step
transfer_expected_source_variant = source_variant_name
transfer_expected_source_step = start_step

gpu_ids = [0]
expected_world_size = 1
_effective_batch = 384
batch_size_t = int(_os.environ.get("STAGE1_BATCH_SIZE_T", "8"))
if batch_size_t <= 0 or _effective_batch % batch_size_t:
    raise ValueError(
        "STAGE1_BATCH_SIZE_T must be a positive divisor of 384, got "
        f"{batch_size_t}"
    )
gradient_accumulation_steps = _effective_batch // batch_size_t
drop_incomplete_accumulation = True
effective_global_batch_size = (
    batch_size_t * expected_world_size * gradient_accumulation_steps
)

checkpoint_interval = 1000
checkpoint_keep_steps = list(
    range(start_step + checkpoint_interval, total_step + 1, checkpoint_interval)
)
val_interval = 500

phase_total_steps = total_step - start_step
phase_warmup_steps = int(phase_total_steps * warmup_fraction)
before_step[1] = dict(
    type=CbLinearCosineRestart,
    assigns=["optimiz.param_groups[0]['lr']=value"],
    start_step=start_step,
    nlin=phase_warmup_steps,
    ntotal=phase_total_steps,
    vstart=0,
    vbase=lr,
    vfinal=lr * final_lr_ratio,
)

stage1_branch = (
    "model-only fork from noncausal MOVi-C step-025000; 24-frame repeated-"
    "prefix causal adaptation on GPU0 with effective batch 384"
)

del _base, _effective_batch, _importlib_cfg, _os, _Path
