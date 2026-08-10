"""YTVIS-HQ xSSC with a frozen V-JEPA2.1 ViT-L video backbone."""

from pathlib import Path as _Path

from object_centric_bench.model import RandSFQ2VJEPAVideo, VJEPA21VideoViT
from object_centric_bench.util import importlib_cfg as _importlib_cfg


_base = _importlib_cfg(
    _Path(__file__).with_name("rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512.py"),
    name="_xssc_dinov3_ytvis_base",
)
globals().update(_base)

variant_name = "vjepa2_1_vitl16_video_256_ytvis_hq_slot512_native_tubelet"
temporal_mode = "noncausal"
tubelet_label_policy = "second_frame"
vjepa2_root = "/home/gaoya/Code_Video/vjepa2-main"
vjepa2_checkpoint = (
    "/data/gaoya/agent-data/weights/"
    "vjepa2_1_vitl_dist_vitG_384_ema_encoder.pt"
)
raw_clip_frames = 6
xssc_steps = raw_clip_frames // 2
label_frame_indices = [1, 3, 5]
train_clip_frames = raw_clip_frames
transition_dt = xssc_steps

start_step = 0
total_step = 10000
max_step = total_step
gpu_ids = [5, 6]
expected_world_size = len(gpu_ids)
batch_size_t = 64
gradient_accumulation_steps = 3
drop_incomplete_accumulation = True
effective_global_batch_size = (
    batch_size_t * expected_world_size * gradient_accumulation_steps
)
checkpoint_key = r"^(?!m\.encode_backbone\.).*"
checkpoint_allowed_missing = [r"^m\.encode_backbone\..*"]
checkpoint_interval = 1000
checkpoint_keep_steps = list(range(checkpoint_interval, total_step + 1, checkpoint_interval))
val_interval = 500
deterministic_warn_only = False
deterministic_sdp_math = True

# The inherited schedule was materialized for 50k steps during base-config
# import. Rebuild its horizon for this 10k run while keeping the same warmup
# fraction, peak LR, and final LR ratio.
before_step[1]["nlin"] = int(total_step * warmup_fraction)
before_step[1]["ntotal"] = total_step


def _pad_encoded_clip(video, segment):
    """Repeat the last encoded frame for the two five-frame YTVIS videos."""
    missing = raw_clip_frames - len(video)
    if missing <= 0:
        return video, segment
    return (
        list(video) + [video[-1]] * missing,
        list(segment) + [segment[-1]] * missing,
    )


def _pad_encoded_even(video, segment):
    """Repeat the last frame so every validation frame belongs to a tubelet."""
    if len(video) % 2 == 0:
        return video, segment
    return list(video) + [video[-1]], list(segment) + [segment[-1]]


temporal_slice = dataset_t["transform0"]
temporal_slice["size"] = raw_clip_frames
dataset_t["transform0"] = dict(
    type=Compose,
    transforms=[
        temporal_slice,
        dict(
            type=Lambda,
            ikeys=[["video"], ["segment"]],
            okeys=[["video"], ["segment"]],
            func=_pad_encoded_clip,
        ),
    ],
)
# All local YTVIS-HQ clips are at most 36 frames, so the inherited ts=30
# branch scans every LMDB value but repeats each key exactly once.
dataset_t["ts"] = None
dataset_v["transform0"] = dict(
    type=Lambda,
    ikeys=[["video"], ["segment"]],
    okeys=[["video"], ["segment"]],
    func=_pad_encoded_even,
)
transform_t.append(
    dict(type=Lambda, ikeys=[["segment"]], func=lambda value: value[1::2])
)
transform_v.append(
    dict(type=Lambda, ikeys=[["segment"]], func=lambda value: value[1::2])
)

model["type"] = RandSFQ2VJEPAVideo
model["encode_backbone"] = dict(
    type=VJEPA21VideoViT,
    model_name="vjepa2_1_vit_large_384",
    checkpoint=vjepa2_checkpoint,
    source_root=vjepa2_root,
    in_size=resolut0[0],
    patch_size=16,
    tubelet_size=2,
    temporal_mode=temporal_mode,
)
model["transit"]["dt"] = transition_dt

del _base, _Path, _importlib_cfg
