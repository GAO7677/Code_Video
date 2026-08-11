"""Continue the 10-frame noncausal V-JEPA xSSC run on MOVi-C."""

from pathlib import Path as _Path

from object_centric_bench.learn import CbLinearCosineRestart
from object_centric_bench.util import importlib_cfg as _importlib_cfg


_base = _importlib_cfg(
    _Path(__file__).with_name(
        "rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-slot512-transfer10000.py"
    ),
    name="_xssc_vjepa_movic_6f_transfer10000_base",
)
globals().update(_base)

variant_name = (
    "vjepa2_1_vitl16_video_256_movi_c_10f_slot512_transfer20000_noncausal"
)
source_variant_name = (
    "vjepa2_1_vitl16_video_ytvis_hq_10f_ar_slot512_transfer10000_bs64"
)
raw_clip_frames = 10
xssc_steps = raw_clip_frames // 2
label_frame_indices = [1, 3, 5, 7, 9]
train_clip_frames = raw_clip_frames
transition_dt = xssc_steps

# Keep global optimizer-step numbering across the YTVIS -> MOVi-C transfer.
start_step = 20000
total_step = 50000
max_step = total_step
transfer_expected_source_variant = source_variant_name
transfer_expected_source_step = start_step
transfer_load_exclude = [r"^m\.initializ\..*"]
transfer_allowed_missing = [r"^m\.encode_backbone\..*", r"^m\.initializ\..*"]
transfer_partial_row_patterns = []

# Match the source run: 64 samples/GPU * 2 GPUs * 3 accumulation = 384.
gpu_ids = [5, 6]
expected_world_size = len(gpu_ids)
batch_size_t = 64
gradient_accumulation_steps = 3
drop_incomplete_accumulation = True
effective_global_batch_size = (
    batch_size_t * expected_world_size * gradient_accumulation_steps
)

checkpoint_interval = 1000
checkpoint_keep_steps = list(
    range(start_step + checkpoint_interval, total_step + 1, checkpoint_interval)
)
val_interval = 500

# MOVi-C is a new domain and its bbox-conditioned initializer starts fresh.
# Restart the LR over this 30k-step phase instead of inheriting the near-zero
# terminal LR from the completed YTVIS phase.
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

# The TFRecord reader slices encoded RGB/masks first, then derives bboxes from
# the transformed tubelet-aligned masks. Thus video, segment, and bbox all have
# five temporal steps when they reach the model.
dataset_t["transform0"]["size"] = raw_clip_frames
model["transit"]["dt"] = transition_dt
model["decode"]["posit_embed"]["spatial_shape"] = [16, 16]

del _base, _Path, _importlib_cfg
