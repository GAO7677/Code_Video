from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from code_vjepa_vggt import batch_infer_v_newtrain_from_jsonl as core
from code_vjepa_vggt.train0705 import infer_stage1b_context_only_no_gt_box_v_newtrain0705 as infer0705
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, read_video_prefix, read_video_uniform
from diffsynth.utils.data import save_video


DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
DEFAULT_BASE_LORA = Path(
    "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/"
    "raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors"
)
DEFAULT_STAGE1A = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
    "pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build object_context from two checkpoints separately, then cross-feed them "
            "into the other checkpoint's Wan denoising path for diagnosis."
        )
    )
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--diffsynth-root", type=Path, default=DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument("--lora-checkpoint", type=Path, default=DEFAULT_BASE_LORA)
    parser.add_argument("--stage1a-init-from", type=Path, default=DEFAULT_STAGE1A)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--input-cover-crop-height", type=int, default=480)
    parser.add_argument("--input-cover-crop-width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--direction", choices=["both", "a_to_b", "b_to_a"], default="both")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--disable-object-branch", action="store_true")
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
    parser.add_argument("--aux-device", default=None)
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
    parser.add_argument("--initialize-model-on-cpu", action="store_true")
    infer0705.add_vjepa_cli_args(parser)
    return parser.parse_args()


def _resolve_runtime_device(device_arg: str) -> str:
    if str(device_arg).strip() and str(device_arg).strip().lower() != "cuda":
        return str(device_arg).strip()
    return infer0705._resolve_launch_device()


def _load_case(args: argparse.Namespace) -> tuple[Path, str, np.ndarray, np.ndarray, torch.Tensor]:
    payload = core._load_input_json(args.input_json)
    input_video = Path(core._resolve_input_video(payload, args.input_json)).expanduser().resolve()
    input_caption = core._ensure_str_field(payload, "input_caption", args.input_json)
    if str(args.sampling_mode) == "uniform":
        frames, frame_indices = read_video_uniform(input_video, int(args.context_frames))
    else:
        frames, frame_indices = read_video_prefix(input_video, int(args.context_frames))
    if int(frames.shape[0]) > int(args.context_frames):
        frames = frames[: int(args.context_frames)]
        frame_indices = frame_indices[: int(args.context_frames)]
    context_video_single = preprocess_video_rgb_uint8(
        frames,
        (int(args.height), int(args.width)),
        resize_mode="cover_crop",
        cover_crop_hw=(int(args.input_cover_crop_height), int(args.input_cover_crop_width)),
    )
    return input_video, input_caption, frames, frame_indices, context_video_single


def _make_runtime_args(args: argparse.Namespace, checkpoint_dir: Path, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=str(checkpoint_dir),
        context_video="",
        prompt="",
        output_dir=str(output_dir),
        wan_root=str(args.wan_root),
        diffsynth_root=str(args.diffsynth_root),
        lora_checkpoint=str(args.lora_checkpoint),
        stage1a_init_from=str(args.stage1a_init_from),
        num_frames=int(args.num_frames),
        context_frames=int(args.context_frames),
        sampling_steps=int(args.sampling_steps),
        height=int(args.height),
        width=int(args.width),
        fps=int(args.fps),
        seed=int(args.seed),
        cfg_scale=float(args.cfg_scale),
        quality=int(args.quality),
        lora_rank=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        disable_object_branch=bool(args.disable_object_branch),
        object_num_queries=int(args.object_num_queries),
        aux_max_objects=int(args.aux_max_objects),
        object_pooler_latent_dim=int(args.object_pooler_latent_dim),
        cond_proj_dim=int(args.cond_proj_dim),
        jepa_window_radius=int(args.jepa_window_radius),
        latent_window_radius=int(args.latent_window_radius),
        object_gate_init=float(args.object_gate_init),
        jepa_ckpt_path=str(args.jepa_ckpt_path),
        jepa_input_size=int(args.jepa_input_size),
        jepa_patch_size=int(args.jepa_patch_size),
        jepa_tubelet_size=int(args.jepa_tubelet_size),
        cotracker_checkpoint=str(args.cotracker_checkpoint),
        cotracker_input_h=int(args.cotracker_input_h),
        cotracker_input_w=int(args.cotracker_input_w),
        cotracker_window_len=int(args.cotracker_window_len),
        vggt_model_path=str(args.vggt_model_path),
        vggt_input_h=int(args.vggt_input_h),
        vggt_input_w=int(args.vggt_input_w),
        vggt_cache_root=args.vggt_cache_root,
        aux_device=args.aux_device,
        grounding_device=args.grounding_device,
        sam2_segment_len=int(args.sam2_segment_len),
        grounding_proposal_source=str(args.grounding_proposal_source),
        grounding_motion_score_ratio=float(args.grounding_motion_score_ratio),
        grounding_text_prompt=str(args.grounding_text_prompt),
        grounding_extra_prompt_terms=str(args.grounding_extra_prompt_terms),
        grounding_disable_caption_terms=bool(args.grounding_disable_caption_terms),
        grounding_gdino_box_threshold=float(args.grounding_gdino_box_threshold),
        grounding_gdino_text_threshold=float(args.grounding_gdino_text_threshold),
        grounding_prompt_frame_mode=str(args.grounding_prompt_frame_mode),
        grounding_track_dedupe_iou_threshold=float(args.grounding_track_dedupe_iou_threshold),
        grounding_container_suppress_ratio_threshold=float(args.grounding_container_suppress_ratio_threshold),
        grounding_container_suppress_min_contained=int(args.grounding_container_suppress_min_contained),
        grounding_container_suppress_min_area_ratio=float(args.grounding_container_suppress_min_area_ratio),
        grounding_container_suppress_small_iou_threshold=float(args.grounding_container_suppress_small_iou_threshold),
        device=str(args.device),
        initialize_model_on_cpu=bool(args.initialize_model_on_cpu),
        enable_vjepa_guidance=bool(getattr(args, "enable_vjepa_guidance", False)),
        vjepa_preset=getattr(args, "vjepa_preset", None),
        vjepa_device=getattr(args, "vjepa_device", None),
        vjepa_model=getattr(args, "vjepa_model", "vith"),
        vjepa_ckpt=getattr(args, "vjepa_ckpt", None),
        vjepa_guidance_mode=getattr(args, "vjepa_guidance_mode", "context_anchored"),
        vjepa_motion_mask_mode=getattr(args, "vjepa_motion_mask_mode", "temporal_union_except_first"),
        vjepa_guidance_steps=int(getattr(args, "vjepa_guidance_steps", 12)),
        vjepa_min_step_percent=float(getattr(args, "vjepa_min_step_percent", 0.35)),
        vjepa_max_step_percent=float(getattr(args, "vjepa_max_step_percent", 0.80)),
        vjepa_target_step_indices=getattr(args, "vjepa_target_step_indices", None),
        vjepa_target_timesteps=getattr(args, "vjepa_target_timesteps", None),
        vjepa_latent_step_size=float(getattr(args, "vjepa_latent_step_size", 0.20)),
        vjepa_inner_k=int(getattr(args, "vjepa_inner_k", 1)),
        vjepa_backtracking=bool(getattr(args, "vjepa_backtracking", False)),
        vjepa_backtracking_taps=getattr(args, "vjepa_backtracking_taps", None),
        vjepa_line_search_taps=getattr(args, "vjepa_line_search_taps", None),
        vjepa_preview_downsample_factor=int(getattr(args, "vjepa_preview_downsample_factor", 4)),
        vjepa_preview_frame_stride=int(getattr(args, "vjepa_preview_frame_stride", 1)),
        vjepa_window_size=int(getattr(args, "vjepa_window_size", 24)),
        vjepa_context_frames=int(getattr(args, "vjepa_context_frames", 8)),
        vjepa_stride=int(getattr(args, "vjepa_stride", 4)),
        vjepa_reduction=str(getattr(args, "vjepa_reduction", "mean")),
        vjepa_grad_norm_mode=str(getattr(args, "vjepa_grad_norm_mode", "rms")),
        vjepa_max_grad_norm=getattr(args, "vjepa_max_grad_norm", 10.0),
        vjepa_max_correction_ratio=getattr(args, "vjepa_max_correction_ratio", 0.05),
        vjepa_stay_close_max_video_l1=getattr(args, "vjepa_stay_close_max_video_l1", 0.03),
        vjepa_artifact_guard_mode=str(getattr(args, "vjepa_artifact_guard_mode", "video_l1_backoff")),
        vjepa_use_spectral_guidance=bool(getattr(args, "vjepa_use_spectral_guidance", False)),
        vjepa_spectral_source=str(getattr(args, "vjepa_spectral_source", "temporal_lowpass_residual")),
        vjepa_spectral_lowpass_ratio=float(getattr(args, "vjepa_spectral_lowpass_ratio", 0.18)),
        vjepa_spectral_normalize_percentile=float(getattr(args, "vjepa_spectral_normalize_percentile", 95.0)),
        vjepa_spectral_weight_floor=float(getattr(args, "vjepa_spectral_weight_floor", 0.25)),
        vjepa_spectral_weight_scale=float(getattr(args, "vjepa_spectral_weight_scale", 1.0)),
        vjepa_spectral_mask_dilation=int(getattr(args, "vjepa_spectral_mask_dilation", 0)),
    )


def _tensor_stats(tensor: torch.Tensor) -> dict[str, object]:
    tensor_fp32 = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "mean": float(tensor_fp32.mean().item()),
        "std": float(tensor_fp32.std(unbiased=False).item()),
        "abs_mean": float(tensor_fp32.abs().mean().item()),
        "abs_max": float(tensor_fp32.abs().max().item()),
    }


def _build_object_context_for_checkpoint(
    *,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    context_video_single: torch.Tensor,
    prompt: str,
    video_path: Path,
) -> tuple[torch.Tensor, dict[str, object]]:
    runtime_args = _make_runtime_args(args, checkpoint_dir, args.output_root / "_tmp_runtime")
    model, _, load_info = infer0705._build_runtime_model(runtime_args)
    try:
        with torch.no_grad():
            object_context, object_debug = infer0705._build_object_context(
                model,
                context_video_single=context_video_single,
                prompt=prompt,
                video_path=str(video_path),
            )
        if object_context is None:
            raise RuntimeError(f"object_context is None for checkpoint {checkpoint_dir}")
        object_context_cpu = object_context.detach().cpu()
        debug = {
            "checkpoint": str(checkpoint_dir),
            "load_info": infer0705._summarize_load_info(load_info),
            "object_debug": object_debug,
            "object_context_stats": _tensor_stats(object_context_cpu),
        }
        return object_context_cpu, debug
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_denoise_with_external_object_context(
    *,
    args: argparse.Namespace,
    denoise_checkpoint_dir: Path,
    source_checkpoint_dir: Path,
    object_context_cpu: torch.Tensor,
    context_video_single: torch.Tensor,
    context_video_path: Path,
    input_caption: str,
    frame_indices: np.ndarray,
    output_dir: Path,
) -> dict[str, object]:
    runtime_args = _make_runtime_args(args, denoise_checkpoint_dir, output_dir)
    model, model_args, load_info = infer0705._build_runtime_model(runtime_args)
    try:
        pipe = model.pipe
        pipe.dit.eval()
        context_pil = infer0705._tensor_video_to_pil_list(context_video_single)
        object_context = object_context_cpu.to(device=pipe.device, dtype=pipe.torch_dtype)
        with torch.no_grad():
            video = pipe(
                prompt=str(input_caption),
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

        output_dir.mkdir(parents=True, exist_ok=True)
        sample_stem = args.input_json.stem
        mix_tag = f"obj_from_{source_checkpoint_dir.name}_denoise_{denoise_checkpoint_dir.name}"
        output_video = output_dir / f"{sample_stem}_{mix_tag}.mp4"
        save_video(video, str(output_video), fps=int(args.fps), quality=int(args.quality))
        result = {
            "input_json": str(args.input_json),
            "input_video": str(context_video_path),
            "input_caption": str(input_caption),
            "output_video": str(output_video),
            "object_context_source_checkpoint": str(source_checkpoint_dir),
            "denoise_checkpoint": str(denoise_checkpoint_dir),
            "seed": int(args.seed),
            "sampling_steps": int(args.sampling_steps),
            "cfg_scale": float(args.cfg_scale),
            "frame_indices": frame_indices.tolist(),
            "sampling_mode": str(args.sampling_mode),
            "model_device": str(model.pipe.device),
            "load_info": infer0705._summarize_load_info(load_info),
            "external_object_context_stats": _tensor_stats(object_context_cpu),
            "model_args": {
                "height": int(model_args.height),
                "width": int(model_args.width),
                "num_frames": int(model_args.num_frames),
                "context_frames": int(args.context_frames),
                "enable_object_branch": bool(model_args.enable_object_branch),
                "lora_checkpoint": str(model_args.lora_checkpoint),
                "stage1a_init_from": str(model_args.stage1a_init_from),
            },
        }
        return result
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    infer0705.apply_vjepa_preset_if_requested(args)
    args.device = _resolve_runtime_device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt_a = args.checkpoint_a.expanduser().resolve()
    ckpt_b = args.checkpoint_b.expanduser().resolve()
    context_video_path, input_caption, _, frame_indices, context_video_single = _load_case(args)

    object_context_a, debug_a = _build_object_context_for_checkpoint(
        args=args,
        checkpoint_dir=ckpt_a,
        context_video_single=context_video_single,
        prompt=input_caption,
        video_path=context_video_path,
    )
    object_context_b, debug_b = _build_object_context_for_checkpoint(
        args=args,
        checkpoint_dir=ckpt_b,
        context_video_single=context_video_single,
        prompt=input_caption,
        video_path=context_video_path,
    )

    a_fp32 = object_context_a.detach().float().reshape(1, -1)
    b_fp32 = object_context_b.detach().float().reshape(1, -1)
    cosine = torch.nn.functional.cosine_similarity(a_fp32, b_fp32, dim=1).item()
    mse = torch.mean((a_fp32 - b_fp32) ** 2).item()

    cross_runs: dict[str, dict[str, object]] = {}
    if str(args.direction) in {"both", "a_to_b"}:
        cross_a_to_b_dir = output_root / f"{ckpt_a.name}_to_{ckpt_b.name}"
        cross_runs["object_from_a_denoise_with_b"] = _run_denoise_with_external_object_context(
            args=args,
            denoise_checkpoint_dir=ckpt_b,
            source_checkpoint_dir=ckpt_a,
            object_context_cpu=object_context_a,
            context_video_single=context_video_single,
            context_video_path=context_video_path,
            input_caption=input_caption,
            frame_indices=frame_indices,
            output_dir=cross_a_to_b_dir,
        )
    if str(args.direction) in {"both", "b_to_a"}:
        cross_b_to_a_dir = output_root / f"{ckpt_b.name}_to_{ckpt_a.name}"
        cross_runs["object_from_b_denoise_with_a"] = _run_denoise_with_external_object_context(
            args=args,
            denoise_checkpoint_dir=ckpt_a,
            source_checkpoint_dir=ckpt_b,
            object_context_cpu=object_context_b,
            context_video_single=context_video_single,
            context_video_path=context_video_path,
            input_caption=input_caption,
            frame_indices=frame_indices,
            output_dir=cross_b_to_a_dir,
        )

    summary = {
        "input_json": str(args.input_json),
        "input_video": str(context_video_path),
        "input_caption": str(input_caption),
        "checkpoint_a": str(ckpt_a),
        "checkpoint_b": str(ckpt_b),
        "context_video_shape": list(context_video_single.shape),
        "frame_indices": frame_indices.tolist(),
        "sampling_mode": str(args.sampling_mode),
        "sampling_steps": int(args.sampling_steps),
        "cfg_scale": float(args.cfg_scale),
        "seed": int(args.seed),
        "object_context_similarity": {
            "cosine": float(cosine),
            "mse": float(mse),
        },
        "object_context_from_a": debug_a,
        "object_context_from_b": debug_b,
        "cross_runs": cross_runs,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "cross_outputs": summary["cross_runs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
