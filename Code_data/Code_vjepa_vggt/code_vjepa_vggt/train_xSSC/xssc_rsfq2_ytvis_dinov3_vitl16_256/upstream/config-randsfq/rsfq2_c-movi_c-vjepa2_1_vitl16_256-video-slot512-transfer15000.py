"""MOVi-C transfer xSSC with a frozen V-JEPA2.1 ViT-L video backbone."""

from pathlib import Path as _Path

from object_centric_bench.model import RandSFQ2VJEPAVideo, VJEPA21VideoViT
from object_centric_bench.util import importlib_cfg as _importlib_cfg


_base = _importlib_cfg(
    _Path(__file__).with_name(
        "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000.py"
    ),
    name="_xssc_dinov3_movic_base",
)
globals().update(_base)

variant_name = "vjepa2_1_vitl16_video_256_movi_c_slot512_transfer15000_native_tubelet"
source_variant_name = "vjepa2_1_vitl16_video_256_ytvis_hq_slot512_native_tubelet"
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

gpu_ids = [5, 6]
expected_world_size = len(gpu_ids)
batch_size_t = 16
gradient_accumulation_steps = 12
drop_incomplete_accumulation = True
effective_global_batch_size = (
    batch_size_t * expected_world_size * gradient_accumulation_steps
)
checkpoint_allowed_missing = [r"^m\.encode_backbone\..*"]
checkpoint_keep_steps = [15000, 50000]
deterministic_warn_only = False
deterministic_sdp_math = True

# YTVIS and MOVi-C both use three native tubelet steps. Only the dataset-specific
# initializer changes, so the learned transition time embedding can transfer.
transfer_load_exclude = [r"^m\.initializ\..*"]
transfer_allowed_missing = [r"^m\.encode_backbone\..*", r"^m\.initializ\..*"]
transfer_expected_source_variant = source_variant_name
transfer_expected_source_step = 15000


def _pad_encoded_even(video, segment):
    if len(video) % 2 == 0:
        return video, segment
    return list(video) + [video[-1]], list(segment) + [segment[-1]]

dataset_t["transform0"]["size"] = raw_clip_frames
dataset_t["index_cache_dir"] = (
    "/data/gaoya/agent-data/cache/movi_tfrecord_indices"
)
dataset_v["index_cache_dir"] = dataset_t["index_cache_dir"]
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
