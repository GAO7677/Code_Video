"""Frozen V-JEPA2.1 xSSC step-7000 recognition and box-regression pilot."""

from pathlib import Path as _Path

import numpy as _np

from object_centric_bench.learn import ClipGradNorm
from object_centric_bench.model import MLP
from object_centric_bench.util import importlib_cfg as _importlib_cfg


_official = _importlib_cfg(
    _Path(__file__).with_name("rsfq2_r_recogn-ytvis_hq.py"),
    name="_official_xssc_recognition",
)
globals().update(_official)
_vjepa = _importlib_cfg(
    _Path(__file__).with_name(
        "rsfq2_r-ytvis_hq-vjepa2_1_vitl16_256-video-slot512.py"
    ),
    name="_vjepa_xssc_noncausal_step7000",
)

# Runtime controls. The launcher reads these values from this file, so future
# reruns only need edits here.
gpu_ids = [7]
expected_world_size = len(gpu_ids)
seed = 42
data_dir = _Path("/data/gaoya/dataset")
save_dir = _Path(
    "/data/gaoya/agent-data/checkpoints/"
    "xssc_vjepa2_1_downstream_recognition_step7000_pilot"
)
wandb_project = "xssc_vjepa2_1_downstream"
wandb_mode = "online"
source_checkpoint = _Path(
    "/data/gaoya/agent-data/checkpoints/"
    "xssc_vjepa2_1_video_noncausal_ytvis_hq_bs64_steps10000/"
    "rsfq2_r-ytvis_hq-vjepa2_1_vitl16_256-video-slot512/42/"
    "step-007000.pth"
)

# Downstream optimization controls. The official protocol uses 5,000 steps,
# CE + L1 losses, and a validation interval of 125 steps.
variant_name = "vjepa2_1_vitl16_video_slot512_step7000_recognition_pilot"
source_variant_name = _vjepa["variant_name"]
source_optimizer_step = 7000
start_step = 0
total_step = 5000
max_step = total_step
batch_size_t = 16
batch_size_v = 1
gradient_accumulation_steps = 1
drop_incomplete_accumulation = False
val_interval = 125
checkpoint_interval = 1000
checkpoint_keep_steps = list(range(checkpoint_interval, total_step + 1, checkpoint_interval))
num_work = 4
lr = 1e-3
gradient_clip_norm = 1.0
recognition_dropout = 0.1
match_iou_threshold = 0.1

# V-JEPA/xSSC geometry and deterministic DDP settings.
raw_clip_frames = 6
label_frame_indices = [1, 3, 5]
emb_dim = _vjepa["emb_dim"]
vfm_dim = _vjepa["vfm_dim"]
amp_dtype = _vjepa["amp_dtype"]
distributed_backend = _vjepa["distributed_backend"]
distributed_timeout_minutes = _vjepa["distributed_timeout_minutes"]
train_sampler_drop_last = False
train_loader_drop_last = False
cudnn_benchmark = False
cudnn_deterministic = True
use_deterministic_algorithms = True
deterministic_warn_only = False
deterministic_sdp_math = True


def _repeat_last(sequence, count):
    if count <= 0:
        return sequence
    if isinstance(sequence, _np.ndarray):
        return _np.concatenate(
            [sequence, _np.repeat(sequence[-1:], count, axis=0)], axis=0
        )
    return list(sequence) + [sequence[-1]] * count


def _pad_six(video, segment, clazz):
    missing = raw_clip_frames - len(video)
    return tuple(_repeat_last(value, missing) for value in (video, segment, clazz))


def _pad_even(video, segment, clazz):
    missing = len(video) % 2
    return tuple(_repeat_last(value, missing) for value in (video, segment, clazz))


# Preserve the official recognition dataset/augmentation pipeline, changing
# only temporal alignment for V-JEPA's native two-frame tubelets.
dataset_t["transform0"]["size"] = raw_clip_frames
dataset_t["transform0"] = dict(
    type=Compose,
    transforms=[
        dataset_t["transform0"],
        dict(
            type=Lambda,
            ikeys=[["video"], ["segment"], ["clazz"]],
            okeys=[["video"], ["segment"], ["clazz"]],
            func=_pad_six,
        ),
    ],
)
dataset_t["ts"] = None
dataset_v["transform0"] = dict(
    type=Lambda,
    ikeys=[["video"], ["segment"], ["clazz"]],
    okeys=[["video"], ["segment"], ["clazz"]],
    func=_pad_even,
)
transform_t.append(
    dict(
        type=Lambda,
        ikeys=[["segment", "clazz"]],
        func=lambda value: value[1::2],
    )
)
transform_v.append(
    dict(
        type=Lambda,
        ikeys=[["segment", "clazz"]],
        func=lambda value: value[1::2],
    )
)

# Reuse the official ObjDiscovRecogn head while replacing its frozen discovery
# backbone with the trained V-JEPA xSSC model.
discov = _vjepa["model"]
recogn = dict(
    type=MLP,
    in_dim=emb_dim,
    dims=[emb_dim * 2, ncls + cbox],
    ln=None,
    dropout=recognition_dropout,
)
model["discov"] = discov
model["recogn"] = recogn
model["thresh_iou"] = match_iou_threshold

# Strictly map source m.* keys into downstream m.discov.*. The frozen V-JEPA
# encoder is reconstructed from its external checkpoint and the MLP is new.
ckpt_map = []
transfer_load_exclude = []
transfer_prefix_map = [["m.discov.", "m."]]
transfer_allowed_missing = [
    r"^m\.discov\.encode_backbone\..*",
    r"^m\.recogn\..*",
]
transfer_expected_source_variant = source_variant_name
transfer_expected_source_step = source_optimizer_step
freez = [r"^m\.discov\..*"]

# Save xSSC + downstream head for exact resume, excluding only the external
# frozen V-JEPA encoder.
checkpoint_key = r"^(?!m\.discov\.encode_backbone\.).*"
checkpoint_allowed_missing = [r"^m\.discov\.encode_backbone\..*"]
gclip = dict(type=ClipGradNorm, max_norm=gradient_clip_norm)
before_step[1]["nlin"] = total_step // 20
before_step[1]["ntotal"] = total_step
before_step[1]["vbase"] = lr
before_step[1]["vfinal"] = lr / 1e3

# The DDP trainer owns checkpointing; retain only the official metric callbacks.
callback_v = [
    callback
    for callback in callback_v
    if callback["type"].__name__ != "SaveModel"
]

del _official, _vjepa, _Path, _importlib_cfg
