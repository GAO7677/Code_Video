"""Continue noncausal V-JEPA xSSC with 10 frames and aspect-ratio buckets."""

from pathlib import Path as _Path

import torch.nn.functional as ptnf

from object_centric_bench.datum import (
    Lambda,
    Normalize,
    RandomFlip,
    ResizeToAspectRatioBucket,
    StridedRandomSliceSequence,
)
from object_centric_bench.learn import CbLinearCosineRestart
from object_centric_bench.util import Compose, importlib_cfg as _importlib_cfg
from object_centric_bench.util_model import interpolat_argmax_attent


_base = _importlib_cfg(
    _Path(__file__).with_name(
        "rsfq2_r-ytvis_hq-vjepa2_1_vitl16_256-video-slot512.py"
    ),
    name="_xssc_vjepa_ytvis_6f_base",
)
globals().update(_base)

variant_name = "vjepa2_1_vitl16_video_ytvis_hq_10f_ar_slot512_transfer10000"
source_variant_name = "vjepa2_1_vitl16_video_256_ytvis_hq_slot512_native_tubelet"
raw_clip_frames = 10
xssc_steps = raw_clip_frames // 2
label_frame_indices = [1, 3, 5, 7, 9]
train_clip_frames = raw_clip_frames
transition_dt = xssc_steps

# All non-square buckets contain 252 ViT patches; the square bucket contains
# 256. This preserves the original aspect ratio without changing token budget.
aspect_ratio_buckets = [
    [336, 192],
    [256, 256],
    [224, 288],
    [192, 336],
    [144, 448],
]

start_step = 10000
total_step = 20000
max_step = total_step
gpu_ids = [5, 6]
expected_world_size = len(gpu_ids)
batch_size_t = 16
gradient_accumulation_steps = 12
drop_incomplete_accumulation = True
effective_global_batch_size = (
    batch_size_t * expected_world_size * gradient_accumulation_steps
)
checkpoint_interval = 1000
checkpoint_keep_steps = list(
    range(start_step + checkpoint_interval, total_step + 1, checkpoint_interval)
)
val_interval = 500

# This is a model-weight transfer, not an optimizer resume. The three learned
# transition rows are retained and the two new rows keep their initialization.
transfer_load_exclude = []
transfer_allowed_missing = [r"^m\.encode_backbone\..*"]
transfer_expected_source_variant = source_variant_name
transfer_expected_source_step = start_step
transfer_partial_row_patterns = [r"^m\.transit\.te\.weight$"]

# Restart LR over the new 10k-step phase while retaining global optimizer-step
# numbering in checkpoints and W&B.
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


def _pad_train_clip(video, segment):
    missing = raw_clip_frames - len(video)
    if missing <= 0:
        return video, segment
    return (
        list(video) + [video[-1]] * missing,
        list(segment) + [segment[-1]] * missing,
    )


def _pad_validation_even(video, segment):
    if len(video) % 2 == 0:
        return video, segment
    return list(video) + [video[-1]], list(segment) + [segment[-1]]


dataset_t["transform0"] = dict(
    type=Compose,
    transforms=[
        dict(
            type=StridedRandomSliceSequence,
            keys=["video", "segment"],
            size=raw_clip_frames,
        ),
        dict(
            type=Lambda,
            ikeys=[["video"], ["segment"]],
            okeys=[["video"], ["segment"]],
            func=_pad_train_clip,
        ),
    ],
)
dataset_t["ts"] = None
dataset_v["transform0"] = dict(
    type=Lambda,
    ikeys=[["video"], ["segment"]],
    okeys=[["video"], ["segment"]],
    func=_pad_validation_even,
)

transform_t = [
    dict(
        type=ResizeToAspectRatioBucket,
        keys=["video"],
        buckets=aspect_ratio_buckets,
        interp="bilinear",
    ),
    dict(
        type=ResizeToAspectRatioBucket,
        keys=["segment"],
        buckets=aspect_ratio_buckets,
        interp="nearest-exact",
        c=0,
    ),
    dict(type=RandomFlip, keys=["video", "segment"], dims=[-1], p=train_flip_prob),
    dict(type=Normalize, keys=["video"], mean=[IMAGENET_MEAN], std=[IMAGENET_STD]),
    dict(type=Lambda, ikeys=[["segment"]], func=lambda value: value[1::2]),
]
transform_v = [
    dict(
        type=ResizeToAspectRatioBucket,
        keys=["video"],
        buckets=aspect_ratio_buckets,
        interp="bilinear",
    ),
    dict(
        type=ResizeToAspectRatioBucket,
        keys=["segment"],
        buckets=aspect_ratio_buckets,
        interp="nearest-exact",
        c=0,
    ),
    dict(type=Normalize, keys=["video"], mean=[IMAGENET_MEAN], std=[IMAGENET_STD]),
    dict(type=Lambda, ikeys=[["segment"]], func=lambda value: value[1::2]),
]
dataset_t["transform"] = dict(type=Compose, transforms=transform_t)
dataset_v["transform"] = dict(type=Compose, transforms=transform_v)

model["transit"]["dt"] = transition_dt
model["decode"]["posit_embed"]["spatial_shape"] = [16, 16]

# Match predicted masks to the current bucket rather than the old fixed 256².
after_forward[0] = dict(
    type=Lambda,
    ikeys=[["output.attentd"], ["batch.segment"]],
    func=lambda attention, segment: ptnf.one_hot(
        interpolat_argmax_attent(
            attention.detach(), size=segment.shape[-3:-1]
        ).long()
    ).bool(),
    okeys=[["output.segment"]],
)

del _base, _Path, _importlib_cfg
