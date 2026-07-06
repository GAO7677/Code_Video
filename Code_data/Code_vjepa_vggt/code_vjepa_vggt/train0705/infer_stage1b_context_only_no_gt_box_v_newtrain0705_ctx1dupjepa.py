from __future__ import annotations

# Run command example:
'''
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/infer_stage1b_context_only_no_gt_box_v_newtrain0705_ctx1dupjepa.py \
  --checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-002500/checkpoint.safetensors \
  --context-video /data/gaoya/agent-data/outputs/train0705_ctx1_smoke_20260706/input/physicIQ_0002_ctx01f.mp4 \
  --prompt "Two pillows on a table and two grabber tools hanging above them from which a brown tennis ball and an orange block are suspended. The grabber tools let go of the ball and block. Static shot with no camera movement." \
  --output-dir /data/gaoya/AAA_test_video/0623/test/ti2v/train0705_kubric_test5_compare \
  --context-frames 1 \
  --num-frames  \
  --sampling-steps 40 \
  --initialize-model-on-cpu

'''


import torch

from code_vjepa_vggt.train0705 import (
    infer_stage1b_context_only_no_gt_box_v_newtrain0705 as base,
)
from code_vjepa_vggt.utils.vggt_cache import load_vggt_cache

_ORIG_BUILD_MODEL_ARGS = base._build_model_args


def _prepare_jepa_context_video(context_video: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
    if int(context_video.shape[2]) != 1:
        return context_video, {
            "duplicated_for_jepa": False,
            "input_context_frames": int(context_video.shape[2]),
            "jepa_context_frames": int(context_video.shape[2]),
        }
    duplicated = torch.cat([context_video, context_video], dim=2)
    return duplicated, {
        "duplicated_for_jepa": True,
        "input_context_frames": 1,
        "jepa_context_frames": int(duplicated.shape[2]),
    }


def _build_object_context(
    model,
    *,
    context_video_single: torch.Tensor,
    prompt: str,
    video_path: str,
):
    pipe = model.pipe
    device = torch.device(pipe.device)
    context_video = context_video_single.unsqueeze(0).to(device=device, dtype=pipe.torch_dtype)
    image_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))

    sample = {
        "context_video": context_video_single,
        "num_context_frames": int(context_video_single.shape[1]),
        "caption": prompt,
        "video_path": video_path,
    }
    query_points_prior, query_frame_ids, object_valid_mask, box_prior_xyxy = (
        model._build_object_query_priors(sample, image_hw=image_hw)
    )
    query_points_prior = query_points_prior.to(device=device, dtype=pipe.torch_dtype)
    query_frame_ids = query_frame_ids.to(device=device, dtype=pipe.torch_dtype)
    object_valid_mask = object_valid_mask.to(device=device, dtype=pipe.torch_dtype)
    box_prior_xyxy = box_prior_xyxy.to(device=device, dtype=pipe.torch_dtype)

    frames_bthwc_01 = (
        (context_video.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0
    ).clamp(0.0, 1.0)
    cotracker_out = model._run_cotracker(
        frames_bthwc_01,
        query_points_prior=query_points_prior,
        query_frame_ids=query_frame_ids,
        query_image_hw=image_hw,
    )

    if model.vggt_cache_root:
        vggt_out = load_vggt_cache(sample, model.vggt_cache_root, allow_missing=False)
        if vggt_out is None:
            raise RuntimeError(f"VGGT cache missing for {video_path}")
    else:
        vggt_out = model.vggt_adapter(
            frames_bthwc_01,
            query_points_prior=query_points_prior,
            query_image_hw=image_hw,
        )

    tracks_grouped, visibility_grouped, confidence_grouped = model._group_tracks_to_objects(
        cotracker_out.tracks,
        cotracker_out.visibility,
        cotracker_out.confidence,
        max_objects=model.aux_max_objects,
        points_per_object=model.object_num_queries,
    )
    jepa_context_video, jepa_debug = _prepare_jepa_context_video(context_video)
    jepa_out = model._run_jepa(jepa_context_video)
    clean_prefix_latents = base._encode_context_latents(pipe, context_video_single)
    object_out = model.object_pooler(
        jepa_patch_tokens=jepa_out.patch_tokens,
        context_latents=clean_prefix_latents,
        tracks=tracks_grouped,
        visibility=visibility_grouped,
        confidence=confidence_grouped,
        track_image_hw=image_hw,
        object_valid_mask=object_valid_mask,
        box_prior_xyxy=box_prior_xyxy,
        vggt_world_points=getattr(vggt_out, "world_points", None),
        vggt_world_points_conf=getattr(vggt_out, "world_points_conf", None),
        vggt_depth=getattr(vggt_out, "depth", None),
        vggt_depth_conf=getattr(vggt_out, "depth_conf", None),
        vggt_dense_patch_tokens=getattr(vggt_out, "dense_patch_tokens", None),
        vggt_patch_grid_hw=getattr(vggt_out, "patch_grid_hw", None),
        vggt_geometry_image_hw=getattr(vggt_out, "input_hw", None)
        if getattr(vggt_out, "input_hw", None) is not None
        else getattr(vggt_out, "image_hw", None),
        frame_valid_mask=None,
    )
    object_context = model.object_adapter(
        object_out.object_latent_tokens,
        object_valid_mask=object_valid_mask,
    )
    debug = {
        "query_points_shape": list(query_points_prior.shape),
        "query_frame_ids_shape": list(query_frame_ids.shape),
        "object_valid_mask_shape": list(object_valid_mask.shape),
        "object_valid_count": float(object_valid_mask.sum().item()),
        "box_prior_shape": list(box_prior_xyxy.shape),
        "tracks_shape": list(cotracker_out.tracks.shape),
        "object_latent_tokens_shape": list(object_out.object_latent_tokens.shape),
        "object_context_shape": list(object_context.shape),
        "clean_prefix_latents_shape": list(clean_prefix_latents.shape),
        "jepa_patch_tokens_shape": list(jepa_out.patch_tokens.shape),
        "jepa_ctx_fix": jepa_debug,
    }
    return object_context, debug


def _build_model_args(args):
    model_args = _ORIG_BUILD_MODEL_ARGS(args)
    if int(args.context_frames) == 1:
        model_args.fixed_num_context_frames = 2
    return model_args


def main() -> None:
    base._build_object_context = _build_object_context
    base._build_model_args = _build_model_args
    base.main()


if __name__ == "__main__":
    main()
