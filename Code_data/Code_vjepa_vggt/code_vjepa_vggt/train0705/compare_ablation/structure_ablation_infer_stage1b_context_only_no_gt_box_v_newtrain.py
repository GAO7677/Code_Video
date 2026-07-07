from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from code_vjepa_vggt.train0705 import infer_stage1b_context_only_no_gt_box_v_newtrain0705 as base_infer
from code_vjepa_vggt.train0705.compare_ablation import (
    structure_ablation_train_stage1b_context_only_no_gt_box_v_newtrain as sablation,
)
import code_vjepa_vggt.train_v_newtrain as tvn
from code_vjepa_vggt.utils.vggt_cache import load_vggt_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--structure-ablation-type",
        default="none",
        choices=("none", "wo_cotracker", "wo_jepa", "wo_vggt"),
    )
    known, remaining = parser.parse_known_args()

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *remaining]
        args = base_infer.parse_args()
    finally:
        sys.argv = original_argv

    args.structure_ablation_type = str(known.structure_ablation_type)
    return args


def _build_model_args(args: argparse.Namespace) -> argparse.Namespace:
    model_args = base_infer._build_model_args(args)
    model_args.structure_ablation_type = str(args.structure_ablation_type)
    return model_args


def _build_runtime_model(args: argparse.Namespace):
    base_infer.apply_vjepa_preset_if_requested(args)
    model_args = _build_model_args(args)
    accelerator = SimpleNamespace(device=torch.device(args.device))
    model = sablation.build_model(model_args, accelerator)

    if model_args.enable_object_branch:
        stage1a_info = tvn._load_filtered_checkpoint_into_model(
            model,
            model_args.stage1a_init_from,
            include_prefixes=("object_pooler.", "object_aux_heads."),
        )
        stage1b_info = tvn._load_filtered_checkpoint_into_model(
            model,
            args.checkpoint,
            include_prefixes=("object_adapter.",),
            include_substrings=(
                "object_embedding",
                ".object_cross_attn.",
                ".object_gate",
                ".norm4.",
            ),
        )
    else:
        stage1a_info = {"skipped": True, "reason": "disable_object_branch"}
        stage1b_info = {"skipped": True, "reason": "disable_object_branch"}

    target_device = torch.device(args.device)
    model.to(target_device)
    model.pipe.to(device=target_device, dtype=model.pipe.torch_dtype)
    for aux_name in ("cotracker_adapter", "jepa_adapter", "vggt_adapter"):
        aux_module = getattr(model, aux_name, None)
        if aux_module is not None and hasattr(aux_module, "device_obj"):
            aux_module.device_obj = target_device
    model.eval()
    base_infer.configure_runtime_pipe_vjepa(model.pipe, args)
    return model, model_args, {
        "stage1a_info": stage1a_info,
        "stage1b_info": stage1b_info,
    }


def _build_object_context(
    model: sablation.StructureAblationContextOnlyNoGTBoxWanModule,
    *,
    context_video_single: torch.Tensor,
    prompt: str,
    video_path: str,
):
    if not bool(getattr(model, "enable_object_branch", False)):
        return None, {"enabled": False}

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

    if bool(getattr(model, "disable_cotracker", False)):
        tracks_grouped, visibility_grouped, confidence_grouped = model._build_static_anchor_tracks(
            query_points_prior,
            src_frames=int(frames_bthwc_01.shape[1]),
        )
        track_debug_shape = list(tracks_grouped.shape)
    else:
        cotracker_out = model._run_cotracker(
            frames_bthwc_01,
            query_points_prior=query_points_prior,
            query_frame_ids=query_frame_ids,
            query_image_hw=image_hw,
        )
        tracks_grouped, visibility_grouped, confidence_grouped = model._group_tracks_to_objects(
            cotracker_out.tracks,
            cotracker_out.visibility,
            cotracker_out.confidence,
            max_objects=model.aux_max_objects,
            points_per_object=model.object_num_queries,
        )
        track_debug_shape = list(cotracker_out.tracks.shape)

    vggt_out = None
    if not bool(getattr(model, "disable_vggt", False)):
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

    jepa_patch_tokens = None
    if not bool(getattr(model, "disable_jepa", False)):
        jepa_out = model._run_jepa(context_video)
        jepa_patch_tokens = jepa_out.patch_tokens

    clean_prefix_latents = base_infer._encode_context_latents(pipe, context_video_single)
    object_out = model.object_pooler(
        jepa_patch_tokens=jepa_patch_tokens,
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
        "structure_ablation_type": str(model.structure_ablation_type),
        "query_points_shape": list(query_points_prior.shape),
        "query_frame_ids_shape": list(query_frame_ids.shape),
        "object_valid_mask_shape": list(object_valid_mask.shape),
        "object_valid_count": float(object_valid_mask.sum().item()),
        "box_prior_shape": list(box_prior_xyxy.shape),
        "tracks_shape": track_debug_shape,
        "object_latent_tokens_shape": list(object_out.object_latent_tokens.shape),
        "object_context_shape": list(object_context.shape),
        "clean_prefix_latents_shape": list(clean_prefix_latents.shape),
        "disable_cotracker": bool(getattr(model, "disable_cotracker", False)),
        "disable_jepa": bool(getattr(model, "disable_jepa", False)),
        "disable_vggt": bool(getattr(model, "disable_vggt", False)),
    }
    return object_context, debug


def main() -> None:
    args = parse_args()
    base_infer.apply_vjepa_preset_if_requested(args)
    args.device = base_infer._resolve_launch_device()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, model_args, load_info = _build_runtime_model(args)
    pipe = model.pipe
    pipe.dit.eval()

    context_video_path = Path(args.context_video).expanduser().resolve()
    frames, frame_indices = base_infer._load_context_video(
        video_path=context_video_path,
        target_context_frames=int(args.context_frames),
    )
    context_video_single = base_infer.preprocess_video_rgb_uint8(
        frames,
        (int(args.height), int(args.width)),
    )
    context_pil = base_infer._tensor_video_to_pil_list(context_video_single)

    with torch.no_grad():
        object_context_raw, object_debug = _build_object_context(
            model,
            context_video_single=context_video_single,
            prompt=str(args.prompt),
            video_path=str(context_video_path),
        )
        object_context, ablation_debug = base_infer._apply_object_context_ablation(
            object_context_raw,
            mode=str(args.object_context_ablation),
            random_seed=args.object_context_random_seed,
            random_scale=float(args.object_context_random_scale),
        )
        object_debug["object_context_ablation"] = ablation_debug
        pipe_kwargs = dict(
            prompt=str(args.prompt),
            negative_prompt="",
            context_video=context_pil,
            seed=int(args.seed),
            tiled=True,
            height=int(args.height),
            width=int(args.width),
            num_frames=int(args.num_frames),
            num_inference_steps=int(args.sampling_steps),
            cfg_scale=float(args.cfg_scale),
        )
        if bool(getattr(model, "enable_object_branch", False)):
            pipe_kwargs["object_context"] = object_context
        video = pipe(**pipe_kwargs)

    checkpoint_path = Path(tvn._resolve_checkpoint_file(args.checkpoint)).resolve()
    checkpoint_tag = checkpoint_path.parent.name
    output_video = output_dir / f"{checkpoint_tag}.mp4"
    base_infer.save_video(video, str(output_video), fps=int(args.fps), quality=int(args.quality))

    result = {
        "structure_ablation_type": str(args.structure_ablation_type),
        "checkpoint": str(checkpoint_path),
        "output_video": str(output_video),
        "context_video": str(context_video_path),
        "prompt": str(args.prompt),
        "frame_indices": frame_indices.tolist(),
        "model_device": str(args.device),
        "model_args": {
            "height": int(model_args.height),
            "width": int(model_args.width),
            "num_frames": int(model_args.num_frames),
            "context_frames": int(model_args.fixed_num_context_frames),
            "enable_object_branch": bool(model_args.enable_object_branch),
            "lora_checkpoint": str(model_args.lora_checkpoint),
            "stage1a_init_from": str(model_args.stage1a_init_from),
        },
        "load_info": base_infer._summarize_load_info(load_info),
        "object_debug": object_debug,
        "vjepa": base_infer.summarize_vjepa_args(args),
    }
    (output_dir / f"{checkpoint_tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
