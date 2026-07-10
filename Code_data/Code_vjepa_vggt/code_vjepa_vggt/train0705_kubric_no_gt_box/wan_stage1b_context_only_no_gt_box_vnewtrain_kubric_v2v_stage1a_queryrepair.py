from __future__ import annotations

"""
Batch v2v inference for Kubric stage1b no-GT-box, with the Stage1A object-token
path explicitly upgraded to the train0710querypoints GT-mask query-repair scheme.

This keeps the existing batch v2v workflow intact, but changes the object-context
builder so that:

1. runtime model args force-enable `grounding_gt_mask_query_repair`
2. inference passes the real sampled source-frame indices into the repair step
3. per-case result json includes query-repair debug payloads

Example:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v_stage1a_queryrepair.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-005500 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name train_stage1b_kubric0708_step5500_repair \
  --output-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_queryrepair \
  --num-inference-steps 40 \
  --num-frames 49
"""

import json
from pathlib import Path
from typing import Any

import torch

from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    train_stage1b_context_only_no_gt_box_v_newtrain_kubric as trainmod,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as base,
)
from code_vjepa_vggt.train0710querypoints.gt_mask_query_repair import (
    repair_grouped_queries_with_gt_masks,
)
from code_vjepa_vggt.utils.vggt_cache import load_vggt_cache


_ORIG_BUILD_MODEL_ARGS = base.infer0705._build_model_args


def _build_model_args_with_query_repair(args):
    model_args = _ORIG_BUILD_MODEL_ARGS(args)
    model_args.grounding_gt_mask_query_repair = bool(
        getattr(args, "grounding_gt_mask_query_repair", True)
    )
    model_args.grounding_gt_mask_oversample_factor = int(
        getattr(args, "grounding_gt_mask_oversample_factor", 4)
    )
    model_args.grounding_gt_mask_min_visible_ratio = float(
        getattr(args, "grounding_gt_mask_min_visible_ratio", 0.60)
    )
    model_args.grounding_gt_mask_min_in_mask_ratio = float(
        getattr(args, "grounding_gt_mask_min_in_mask_ratio", 0.60)
    )
    model_args.grounding_gt_mask_color_tolerance = int(
        getattr(args, "grounding_gt_mask_color_tolerance", 18)
    )
    return model_args


def _build_object_context_with_query_repair(
    model,
    *,
    context_video_single: torch.Tensor,
    prompt: str,
    video_path: str,
    sampled_source_frame_indices: list[int] | None = None,
):
    if not bool(getattr(model, "enable_object_branch", False)):
        return None, {"enabled": False}

    pipe = model.pipe
    device = torch.device(pipe.device)
    context_video = context_video_single.unsqueeze(0).to(device=device, dtype=pipe.torch_dtype)
    image_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))
    num_context_frames = int(context_video_single.shape[1])

    if sampled_source_frame_indices is None:
        sampled_source_frame_indices = list(range(num_context_frames))
    else:
        sampled_source_frame_indices = [int(v) for v in sampled_source_frame_indices]

    sample = {
        "context_video": context_video_single,
        "num_context_frames": num_context_frames,
        "caption": prompt,
        "video_path": video_path,
        "context_frame_indices": torch.arange(num_context_frames, dtype=torch.long),
        "metadata": {
            "sampled_frame_indices": sampled_source_frame_indices,
            "source_video_path": str(video_path),
        },
    }

    valid_frames = max(num_context_frames, 1)
    frames_tchw_01 = (
        ((context_video_single[:, :valid_frames].permute(1, 0, 2, 3).float() + 1.0) / 2.0)
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
    )
    grounding_sample = model.viewer_grounding.build_sample(
        frames_tchw_01=frames_tchw_01,
        caption=str(prompt),
        image_hw=image_hw,
    )

    repair_debug: dict[str, Any] = {"applied": False, "reason": "disabled"}
    query_repair_enabled = bool(getattr(model.gt_mask_query_repair, "enabled", False))
    if query_repair_enabled:
        repaired_queries_px, repair_debug = repair_grouped_queries_with_gt_masks(
            sample=sample,
            image_hw=image_hw,
            frames_bthwc_01=torch.from_numpy(frames_tchw_01).permute(0, 2, 3, 1).unsqueeze(0).float(),
            grouped_queries_px=grounding_sample.grouped_queries_px,
            object_valid_mask=grounding_sample.object_valid_mask,
            object_tracks=getattr(grounding_sample, "object_tracks", []),
            prompt_frame_idx=int(getattr(grounding_sample, "prompt_frame_idx", 0)),
            points_per_object=int(model.object_num_queries),
            run_cotracker=model._run_cotracker,
            config=model.gt_mask_query_repair,
        )
        grounding_sample.grouped_queries_px = repaired_queries_px
        if isinstance(getattr(grounding_sample, "debug", None), dict):
            grounding_sample.debug["gt_mask_query_repair"] = repair_debug

    grouped_queries = torch.from_numpy(grounding_sample.grouped_queries_px).float()
    object_valid_mask = torch.from_numpy(grounding_sample.object_valid_mask).float()
    context_boxes_norm = torch.from_numpy(grounding_sample.context_boxes_norm).float()
    prompt_frame_idx = int(getattr(grounding_sample, "prompt_frame_idx", 0))

    query_points_prior = grouped_queries.view(1, int(model.total_object_queries), 2)
    box_priors = []
    frame_ids = []
    for object_idx in range(int(model.aux_max_objects)):
        is_valid = bool(object_valid_mask[object_idx].item() > 0.5)
        first_valid_frame = 0
        box = None
        if is_valid:
            for frame_idx in range(min(valid_frames, int(context_boxes_norm.shape[0]))):
                candidate = context_boxes_norm[frame_idx, object_idx]
                if bool(
                    (candidate[2] - candidate[0] > 1.0e-6)
                    and (candidate[3] - candidate[1] > 1.0e-6)
                ):
                    first_valid_frame = frame_idx
                    box = candidate
                    break
        if box is None:
            box = torch.tensor(trainmod._DUMMY_BOX_XYXY, dtype=torch.float32)
            first_valid_frame = prompt_frame_idx if is_valid else 0
        box_priors.append(box.to(dtype=torch.float32))
        frame_ids.extend([float(first_valid_frame)] * int(model.object_num_queries))

    query_frame_ids = torch.tensor(frame_ids, dtype=torch.float32).view(
        1, int(model.total_object_queries), 1
    )
    box_prior_xyxy = torch.stack(box_priors, dim=0).view(1, int(model.aux_max_objects), 4)

    query_points_prior = query_points_prior.to(device=device, dtype=pipe.torch_dtype)
    query_frame_ids = query_frame_ids.to(device=device, dtype=pipe.torch_dtype)
    object_valid_mask = object_valid_mask.view(1, int(model.aux_max_objects)).to(
        device=device,
        dtype=pipe.torch_dtype,
    )
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
        vggt_out = model._run_vggt(
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
    clean_prefix_latents = base.infer0705._encode_context_latents(pipe, context_video_single)
    jepa_context_video, jepa_debug = trainmod.prepare_jepa_context_video(
        context_video,
        latent_frames=int(clean_prefix_latents.shape[2]),
        tubelet_size=int(getattr(model, "_jepa_tubelet_size", 2)),
    )
    jepa_out = model._run_jepa(jepa_context_video)
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
        "query_repair_enabled": query_repair_enabled,
        "query_repair_debug": repair_debug,
        "sampled_source_frame_indices": sampled_source_frame_indices,
    }
    return object_context, debug


def _run_single_case_in_process_with_query_repair(**kwargs):
    model = kwargs["model"]
    checkpoint_dir = kwargs["checkpoint_dir"]
    input_json_path = kwargs["input_json_path"]
    source_video = kwargs["source_video"]
    input_caption = kwargs["input_caption"]
    output_video = kwargs["output_video"]
    num_frames = kwargs["num_frames"]
    context_frames = kwargs["context_frames"]
    sampling_mode = kwargs["sampling_mode"]
    sampling_steps = kwargs["sampling_steps"]
    negative_prompt = kwargs["negative_prompt"]
    fps = kwargs["fps"]
    seed = kwargs["seed"]
    cfg_scale = kwargs["cfg_scale"]
    height = kwargs["height"]
    width = kwargs["width"]
    input_cover_crop_height = kwargs["input_cover_crop_height"]
    input_cover_crop_width = kwargs["input_cover_crop_width"]
    quality = kwargs["quality"]
    lora_checkpoint = kwargs["lora_checkpoint"]
    stage1a_init_from = kwargs["stage1a_init_from"]
    vjepa_summary = kwargs["vjepa_summary"]

    logs: list[str] = []
    logs.append(f"[case] input_json={input_json_path}")
    logs.append(f"[case] source_video={source_video}")
    logs.append(f"[case] input_caption={input_caption}")
    logs.append(f"[case] checkpoint_dir={checkpoint_dir}")

    context_video_path = Path(source_video).expanduser().resolve()
    frames, frame_indices = base._load_context_video_for_mode(
        video_path=context_video_path,
        target_context_frames=int(context_frames),
        sampling_mode=sampling_mode,
    )
    effective_context_frames = int(frames.shape[0])
    context_video_single = base.preprocess_video_rgb_uint8(
        frames,
        (int(height), int(width)),
        resize_mode="cover_crop",
        cover_crop_hw=(int(input_cover_crop_height), int(input_cover_crop_width)),
    )
    context_pil = base.infer0705._tensor_video_to_pil_list(context_video_single)
    object_context, object_debug = _build_object_context_with_query_repair(
        model=model,
        context_video_single=context_video_single,
        prompt=str(input_caption),
        video_path=str(context_video_path),
        sampled_source_frame_indices=frame_indices.tolist(),
    )
    object_context, ablation_debug = base.infer0705._apply_object_context_ablation(
        object_context,
        mode=str(getattr(model, "_object_context_ablation_mode", "none")),
        random_seed=getattr(model, "_object_context_random_seed", None),
        random_scale=float(getattr(model, "_object_context_random_scale", 1.0)),
        slot_count=int(getattr(model, "aux_max_objects", 0)),
        keep_slot_ids=getattr(model, "_object_context_keep_slot_ids", None),
    )
    object_debug["object_context_ablation"] = ablation_debug
    object_debug["object_context_stats"] = base._tensor_numeric_stats(object_context)

    pipe = model.pipe
    numeric_trace_root = getattr(model, "_dump_numeric_trace_root", None)
    if numeric_trace_root is not None:
        pipe._numeric_trace_enabled = True
        pipe._numeric_trace_path = str(Path(numeric_trace_root) / f"{output_video.stem}_numeric_trace.json")
    else:
        pipe._numeric_trace_enabled = False
        pipe._numeric_trace_path = None
    pipe.dit.eval()
    with torch.no_grad():
        pipe_kwargs = dict(
            prompt=str(input_caption),
            context_video=context_pil,
            seed=int(seed),
            tiled=True,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            num_inference_steps=int(sampling_steps),
            cfg_scale=float(cfg_scale),
        )
        if negative_prompt is not None:
            pipe_kwargs["negative_prompt"] = str(negative_prompt)
        if bool(getattr(model, "enable_object_branch", False)):
            pipe_kwargs["object_context"] = object_context
        dump_pipe_inputs_root = getattr(model, "_dump_pipe_inputs_root", None)
        if dump_pipe_inputs_root is not None:
            base._dump_pipe_inputs(
                dump_root=Path(dump_pipe_inputs_root),
                sample_stem=output_video.stem,
                context_pil=context_pil,
                prompt=str(input_caption),
                negative_prompt=negative_prompt,
                pipe_kwargs=pipe_kwargs,
                source_video=str(source_video),
                frame_indices=frame_indices,
            )
        video = pipe(**pipe_kwargs)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    context_sheet_path = output_video.with_name(
        f"{output_video.stem}_input_ctx{effective_context_frames:02d}.jpg"
    )
    base._save_context_contact_sheet(context_pil=context_pil, output_path=context_sheet_path)
    base.save_video(video, str(output_video), fps=int(fps), quality=int(quality))

    result = {
        "input_json": str(input_json_path),
        "input_video": str(context_sheet_path),
        "source_video": str(source_video),
        "input_caption": str(input_caption),
        "output_video": str(output_video),
        "seed": int(seed),
        "step": int(sampling_steps),
        "guidance": float(cfg_scale),
        "negative_prompt": negative_prompt,
        "ckpt": str(checkpoint_dir),
        "frame_indices": frame_indices.tolist(),
        "requested_context_frames": int(context_frames),
        "effective_context_frames": effective_context_frames,
        "sampling_mode": str(sampling_mode),
        "model_device": str(model.pipe.device),
        "object_context_ablation": {
            "mode": str(getattr(model, "_object_context_ablation_mode", "none")),
            "random_seed": getattr(model, "_object_context_random_seed", None),
            "random_scale": float(getattr(model, "_object_context_random_scale", 1.0)),
            "keep_slot_ids": getattr(model, "_object_context_keep_slot_ids", None),
        },
        "stage1a_scheme": "train0710querypoints_gt_mask_query_repair",
        "query_repair": object_debug.get("query_repair_debug"),
        "model_args": {
            "height": int(height),
            "width": int(width),
            "num_frames": int(num_frames),
            "context_frames": effective_context_frames,
            "num_inference_steps": int(sampling_steps),
            "cfg_scale": float(cfg_scale),
            "negative_prompt": negative_prompt,
            "input_resize_mode": "cover_crop",
            "input_cover_crop_height": int(input_cover_crop_height),
            "input_cover_crop_width": int(input_cover_crop_width),
            "enable_object_branch": bool(getattr(model, "enable_object_branch", False)),
            "lora_checkpoint": str(lora_checkpoint),
            "stage1a_init_from": str(stage1a_init_from),
            "grounding_gt_mask_query_repair": True,
            "grounding_gt_mask_oversample_factor": int(getattr(model.gt_mask_query_repair, "oversample_factor", 4)),
            "grounding_gt_mask_min_visible_ratio": float(getattr(model.gt_mask_query_repair, "min_visible_ratio", 0.60)),
            "grounding_gt_mask_min_in_mask_ratio": float(getattr(model.gt_mask_query_repair, "min_in_mask_ratio", 0.60)),
            "grounding_gt_mask_color_tolerance": int(getattr(model.gt_mask_query_repair, "color_tolerance", 18)),
        },
        "vjepa": vjepa_summary,
        "object_debug": object_debug,
    }
    return result, logs


def _install_query_repair_runtime_hooks() -> None:
    base.infer0705.t0705 = trainmod
    base.infer0705._build_model_args = _build_model_args_with_query_repair
    base.infer0705._build_object_context = _build_object_context_with_query_repair


def main() -> None:
    base._install_kubric_runtime_hooks = _install_query_repair_runtime_hooks
    base._run_single_case_in_process = _run_single_case_in_process_with_query_repair
    base.main()


if __name__ == "__main__":
    main()
