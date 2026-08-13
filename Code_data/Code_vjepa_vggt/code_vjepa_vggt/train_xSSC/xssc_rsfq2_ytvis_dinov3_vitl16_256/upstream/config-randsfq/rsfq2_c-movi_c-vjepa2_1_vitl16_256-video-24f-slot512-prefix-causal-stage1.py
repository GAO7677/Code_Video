"""Stage-1 causal adaptation on full 24-frame MOVi-C trajectories."""

from pathlib import Path as _Path

from object_centric_bench.learn import CbLinearCosineRestart
from object_centric_bench.util import importlib_cfg as _importlib_cfg


_base = _importlib_cfg(
    _Path(__file__).with_name(
        "rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-10f-slot512-"
        "transfer16000-clip2.py"
    ),
    name="_xssc_vjepa_movic_stage1_noncausal_base",
)
globals().update(_base)

variant_name = (
    "vjepa2_1_vitl16_video_256_movi_c_24f_slot512_"
    "prefix_causal_stage1_adapt50000"
)
source_variant_name = (
    "vjepa2_1_vitl16_video_256_movi_c_10f_slot512_"
    "transfer16000_clip2_noncausal"
)
temporal_mode = "prefix_causal"
raw_clip_frames = 24
xssc_steps = raw_clip_frames // 2
label_frame_indices = list(range(1, raw_clip_frames, 2))
train_clip_frames = raw_clip_frames

# Keep the already-trained five-state transition window.  The corrected
# RSFQTransit implementation applies this as a rolling window on longer clips.
transition_dt = 5
model["encode_backbone"]["temporal_mode"] = temporal_mode
model["transit"]["dt"] = transition_dt
dataset_t["transform0"]["size"] = raw_clip_frames

# The official validation split is used for selection.  Test remains untouched
# until the frozen Stage-1 audit.
dataset_v["split"] = "validation"
val_subset_size = 250

# Model-only branch from the completed noncausal MOVi-C representation.  The
# external frozen V-JEPA weights are restored independently.
start_step = 50000
total_step = 60000
max_step = total_step
transfer_expected_source_variant = source_variant_name
transfer_expected_source_step = start_step
transfer_load_exclude = [r"^m\.encode_backbone\..*"]
transfer_allowed_missing = [r"^m\.encode_backbone\..*"]
transfer_partial_row_patterns = []

# Conservative executable default.  A capacity smoke test may raise the
# microbatch while keeping the effective global batch fixed at 384.
expected_world_size = 2
gpu_ids = [5, 6]
batch_size_t = 8
gradient_accumulation_steps = 24
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

stage1_scope = (
    "24-frame repeated-prefix causal V-JEPA adaptation; decoder is a training "
    "objective only and is excluded from Stage-1 trajectory extraction"
)

del _base, _Path, _importlib_cfg
