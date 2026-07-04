from __future__ import annotations

"""
Stage1B context-only no-GT-box inference for the train0705 DiffSynth-native run.

This is based on the official DiffSynth example:
  examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py

Example run command:
  env PYTHONPATH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main" \
  CUDA_VISIBLE_DEVICES="7" \
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/infer_stage1b_context_only_no_gt_box_v_newtrain0705.py \
    --checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-001000 \
    --context-video /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/F5_drop_support/sample_001460/source_video/context_video_8f.mp4 \
    --prompt "f5 sample 001460 industrial rigid body simulation sphere box" \
    --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/inference_review/step-001000 \
    --sampling-steps 12

The script keeps the inference core on top of `diffsynth`, but reconstructs the
same object-conditioning path used during train0705:

  context video -> viewer grounding pseudo boxes -> CoTracker / VGGT / JEPA ->
  ObjectTubeProjector -> ObjectConditionAdapter -> Wan object branch

It loads three weight sources exactly like training:
  1. Wan 2.2 base from --wan-root
  2. Frozen base LoRA from --lora-checkpoint
  3. Frozen Stage1A object_pooler / object_aux_heads from --stage1a-init-from
  4. Stage1B trainable object_adapter + DiT object-branch weights from --checkpoint
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _read_cli_arg_value(argv: list[str], names: tuple[str, ...], default: str | None = None) -> str | None:
    for name in names:
        if name not in argv:
            continue
        index = argv.index(name)
        if index + 1 < len(argv):
            return argv[index + 1]
    return default


_DEFAULT_DIFFSYNTH_ROOT = "/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main"
_SELECTED_DIFFSYNTH_ROOT = _read_cli_arg_value(
    sys.argv,
    ("--diffsynth-root", "--diffsynth_root"),
    os.environ.get("DIFFSYNTH_ROOT", _DEFAULT_DIFFSYNTH_ROOT),
)
if _SELECTED_DIFFSYNTH_ROOT:
    os.environ["DIFFSYNTH_ROOT"] = _SELECTED_DIFFSYNTH_ROOT
    if _SELECTED_DIFFSYNTH_ROOT not in sys.path:
        sys.path.insert(0, _SELECTED_DIFFSYNTH_ROOT)

from diffsynth.utils.data import save_video

import code_vjepa_vggt.train_v_newtrain as tvn
from code_vjepa_vggt.context_wan_v_newtrain import ContextAwareWanVideoPipeline
from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset
from code_vjepa_vggt.train0705 import train_stage1b_context_only_no_gt_box_v_newtrain as t0705
from code_vjepa_vggt.utils.vggt_cache import load_vggt_cache
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, read_video_prefix


def _summarize_string_list(values, *, sample_limit: int = 8):
    values = list(values or [])
    return {
        "count": len(values),
        "sample": values[:sample_limit],
    }


def _summarize_load_info(load_info: dict) -> dict:
    summarized = {}
    for name, info in load_info.items():
        summarized[name] = {
            "loaded_count": int(info.get("loaded_count", 0)),
            "selected_source_keys": int(info.get("selected_source_keys", 0)),
            "missing_keys": _summarize_string_list(info.get("missing_keys", [])),
            "unexpected_keys": _summarize_string_list(info.get("unexpected_keys", [])),
            "skipped_shape_mismatch_count": len(info.get("skipped_shape_mismatch", [])),
            "skipped_shape_mismatch_sample": list(info.get("skipped_shape_mismatch", []))[:4],
        }
    return summarized


def _resolve_launch_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


def _tensor_video_to_pil_list(context_video_single: torch.Tensor):
    from PIL import Image

    frames = context_video_single.detach().cpu().permute(1, 2, 3, 0)
    frames = ((frames + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
    return [Image.fromarray(frame) for frame in frames]


def _build_model_args(args: argparse.Namespace) -> argparse.Namespace:
    parser = t0705.build_parser()
    model_args = parser.parse_args([])

    model_args.diffsynth_root = str(args.diffsynth_root)
    model_args.wan_root = str(args.wan_root)
    model_args.height = int(args.height)
    model_args.width = int(args.width)
    model_args.num_frames = int(args.num_frames)
    model_args.fixed_num_context_frames = int(args.context_frames)
    model_args.max_train_steps = 1
    model_args.num_epochs = 1
    model_args.output_path = str(args.output_dir)
    model_args.dataset_type = "wan_ti2v"
    model_args.report_to = "none"

    model_args.lora_base_model = "dit"
    model_args.lora_target_modules = "q,k,v,o,ffn.0,ffn.2"
    model_args.lora_rank = int(args.lora_rank)
    model_args.lora_alpha = int(args.lora_alpha)
    model_args.lora_checkpoint = str(args.lora_checkpoint)
    model_args.extra_inputs = "input_image"

    model_args.enable_object_branch = True
    model_args.freeze_non_object_trainables = True
    model_args.train_object_adapter = True
    model_args.train_object_dit_branch = True
    model_args.train_object_pooler = False
    model_args.train_object_aux_heads = False

    model_args.object_num_queries = int(args.object_num_queries)
    model_args.aux_max_objects = int(args.aux_max_objects)
    model_args.object_pooler_latent_dim = int(args.object_pooler_latent_dim)
    model_args.cond_proj_dim = int(args.cond_proj_dim)
    model_args.jepa_window_radius = int(args.jepa_window_radius)
    model_args.latent_window_radius = int(args.latent_window_radius)
    model_args.object_gate_init = float(args.object_gate_init)

    model_args.lambda_main = 1.0
    model_args.lambda_track_aux = 0.0
    model_args.lambda_box_aux = 0.0
    model_args.lambda_depth_aux = 0.0
    model_args.lambda_object_context_reg = 0.0

    model_args.stage1a_init_from = str(args.stage1a_init_from)
    model_args.jepa_ckpt_path = str(args.jepa_ckpt_path)
    model_args.jepa_input_size = int(args.jepa_input_size)
    model_args.jepa_patch_size = int(args.jepa_patch_size)
    model_args.jepa_tubelet_size = int(args.jepa_tubelet_size)
    model_args.cotracker_checkpoint = str(args.cotracker_checkpoint)
    model_args.cotracker_input_h = int(args.cotracker_input_h)
    model_args.cotracker_input_w = int(args.cotracker_input_w)
    model_args.cotracker_window_len = int(args.cotracker_window_len)
    model_args.vggt_model_path = str(args.vggt_model_path)
    model_args.vggt_input_h = int(args.vggt_input_h)
    model_args.vggt_input_w = int(args.vggt_input_w)
    model_args.vggt_cache_root = None if args.vggt_cache_root is None else str(args.vggt_cache_root)

    model_args.grounding_device = None if args.grounding_device is None else str(args.grounding_device)
    model_args.sam2_segment_len = int(args.sam2_segment_len)
    model_args.grounding_proposal_source = str(args.grounding_proposal_source)
    model_args.grounding_motion_score_ratio = float(args.grounding_motion_score_ratio)
    model_args.grounding_text_prompt = str(args.grounding_text_prompt)
    model_args.grounding_extra_prompt_terms = str(args.grounding_extra_prompt_terms)
    model_args.grounding_disable_caption_terms = bool(args.grounding_disable_caption_terms)
    model_args.grounding_gdino_box_threshold = float(args.grounding_gdino_box_threshold)
    model_args.grounding_gdino_text_threshold = float(args.grounding_gdino_text_threshold)
    model_args.grounding_prompt_frame_mode = str(args.grounding_prompt_frame_mode)
    model_args.grounding_track_dedupe_iou_threshold = float(args.grounding_track_dedupe_iou_threshold)
    model_args.grounding_container_suppress_ratio_threshold = float(
        args.grounding_container_suppress_ratio_threshold
    )
    model_args.grounding_container_suppress_min_contained = int(
        args.grounding_container_suppress_min_contained
    )
    model_args.grounding_container_suppress_min_area_ratio = float(
        args.grounding_container_suppress_min_area_ratio
    )
    model_args.grounding_container_suppress_small_iou_threshold = float(
        args.grounding_container_suppress_small_iou_threshold
    )

    return tvn.prepare_args(model_args)


def _build_runtime_model(args: argparse.Namespace):
    model_args = _build_model_args(args)
    accelerator = SimpleNamespace(device=torch.device(args.device))
    model = t0705.build_model(model_args, accelerator)

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
    model.to(torch.device(args.device))
    model.eval()
    return model, model_args, {
        "stage1a_info": stage1a_info,
        "stage1b_info": stage1b_info,
    }


def _encode_context_latents(
    pipe: ContextAwareWanVideoPipeline,
    context_video_single: torch.Tensor,
) -> torch.Tensor:
    context_pil = _tensor_video_to_pil_list(context_video_single)
    preprocessed = pipe.preprocess_video(context_pil)
    return pipe.vae.encode(
        preprocessed,
        device=pipe.device,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ).to(dtype=pipe.torch_dtype, device=pipe.device)


def _build_object_context(
    model: t0705.ContextOnlyNoGTBoxWanModule,
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
    jepa_out = model._run_jepa(context_video)
    clean_prefix_latents = _encode_context_latents(pipe, context_video_single)
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
    }
    return object_context, debug


def _load_context_video(
    *,
    video_path: Path,
    target_context_frames: int,
):
    frames, frame_indices = read_video_prefix(video_path, target_context_frames)
    if int(frames.shape[0]) < int(target_context_frames):
        raise RuntimeError(
            f"context video {video_path} only provides {int(frames.shape[0])} frames, "
            f"smaller than required num_context_frames={int(target_context_frames)}"
        )
    if int(frames.shape[0]) > int(target_context_frames):
        frames = frames[:target_context_frames]
        frame_indices = frame_indices[:target_context_frames]
    return frames, frame_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run train0705 stage1b context-only no-GT-box inference on a single context-video case. "
            "Built on DiffSynth Wan2.2 inference with the same object-conditioning path as training."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir or checkpoint.safetensors")
    parser.add_argument("--context-video", required=True, help="Context video mp4")
    parser.add_argument("--prompt", required=True, help="Prompt / caption for generation")
    parser.add_argument("--output-dir", required=True, help="Directory for video + json outputs")
    parser.add_argument("--wan-root", default="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
    parser.add_argument(
        "--diffsynth-root",
        default="/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main",
    )
    parser.add_argument(
        "--lora-checkpoint",
        default="/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors",
    )
    parser.add_argument(
        "--stage1a-init-from",
        default="/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt",
    )
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--object-num-queries", type=int, default=8)
    parser.add_argument("--aux-max-objects", type=int, default=4)
    parser.add_argument("--object-pooler-latent-dim", type=int, default=16)
    parser.add_argument("--cond-proj-dim", type=int, default=4096)
    parser.add_argument("--jepa-window-radius", type=int, default=1)
    parser.add_argument("--latent-window-radius", type=int, default=1)
    parser.add_argument("--object-gate-init", type=float, default=0.1)
    parser.add_argument(
        "--jepa-ckpt-path",
        default="/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth",
    )
    parser.add_argument("--jepa-input-size", type=int, default=384)
    parser.add_argument("--jepa-patch-size", type=int, default=16)
    parser.add_argument("--jepa-tubelet-size", type=int, default=2)
    parser.add_argument(
        "--cotracker-checkpoint",
        default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth",
    )
    parser.add_argument("--cotracker-input-h", type=int, default=384)
    parser.add_argument("--cotracker-input-w", type=int, default=512)
    parser.add_argument("--cotracker-window-len", type=int, default=60)
    parser.add_argument("--vggt-model-path", default="/data/gaoya/ckpt/facebook-VGGT-1B")
    parser.add_argument("--vggt-input-h", type=int, default=420)
    parser.add_argument("--vggt-input-w", type=int, default=728)
    parser.add_argument("--vggt-cache-root", default=None)
    parser.add_argument("--grounding-device", default=None)
    parser.add_argument("--sam2-segment-len", type=int, default=8)
    parser.add_argument("--grounding-proposal-source", default="gdino_only")
    parser.add_argument("--grounding-motion-score-ratio", type=float, default=0.15)
    parser.add_argument(
        "--grounding-text-prompt",
        default="box . cube . block . cylinder . capsule . sphere . ball .",
    )
    parser.add_argument("--grounding-extra-prompt-terms", default="")
    parser.add_argument("--grounding-disable-caption-terms", action="store_true", default=True)
    parser.add_argument("--grounding-gdino-box-threshold", type=float, default=0.20)
    parser.add_argument("--grounding-gdino-text-threshold", type=float, default=0.15)
    parser.add_argument("--grounding-prompt-frame-mode", default="first")
    parser.add_argument("--grounding-track-dedupe-iou-threshold", type=float, default=0.75)
    parser.add_argument("--grounding-container-suppress-ratio-threshold", type=float, default=0.95)
    parser.add_argument("--grounding-container-suppress-min-contained", type=int, default=2)
    parser.add_argument("--grounding-container-suppress-min-area-ratio", type=float, default=1.5)
    parser.add_argument("--grounding-container-suppress-small-iou-threshold", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.device = _resolve_launch_device()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, model_args, load_info = _build_runtime_model(args)
    pipe = model.pipe
    pipe.dit.eval()

    context_video_path = Path(args.context_video).expanduser().resolve()
    frames, frame_indices = _load_context_video(
        video_path=context_video_path,
        target_context_frames=int(args.context_frames),
    )
    context_video_single = preprocess_video_rgb_uint8(
        frames,
        (int(args.height), int(args.width)),
    )
    context_pil = _tensor_video_to_pil_list(context_video_single)

    with torch.no_grad():
        object_context, object_debug = _build_object_context(
            model,
            context_video_single=context_video_single,
            prompt=str(args.prompt),
            video_path=str(context_video_path),
        )
        video = pipe(
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
            object_context=object_context,
        )

    checkpoint_path = Path(tvn._resolve_checkpoint_file(args.checkpoint)).resolve()
    checkpoint_tag = checkpoint_path.parent.name
    output_video = output_dir / f"{checkpoint_tag}.mp4"
    save_video(video, str(output_video), fps=int(args.fps), quality=int(args.quality))

    result = {
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
            "lora_checkpoint": str(model_args.lora_checkpoint),
            "stage1a_init_from": str(model_args.stage1a_init_from),
        },
        "load_info": _summarize_load_info(load_info),
        "object_debug": object_debug,
    }
    (output_dir / f"{checkpoint_tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
