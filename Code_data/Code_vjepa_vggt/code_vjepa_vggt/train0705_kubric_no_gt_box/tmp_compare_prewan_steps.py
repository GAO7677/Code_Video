from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from code_vjepa_vggt import batch_infer_v_newtrain_from_jsonl as core
from code_vjepa_vggt.train0705 import infer_stage1b_context_only_no_gt_box_v_newtrain0705 as infer0705
from code_vjepa_vggt.utils.vggt_cache import load_vggt_cache
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, read_video_prefix, read_video_uniform


DEFAULT_INPUT_JSON = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/"
    "physicIQ_026_Solid_Mechanics_0005_perspective-center_trimmed-ball-behind-rotating-paper.json"
)
DEFAULT_CHECKPOINT_A = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
    "train_stage1b_kubric0708/checkpoints/step-003500"
)
DEFAULT_CHECKPOINT_B = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
    "train_stage1b_kubric0708/checkpoints/step-005000"
)
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/query_prior_compare_20260710")
DEFAULT_CASE_NAME = "physicIQ_026_prewan_step003500_vs_step005000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare every pre-WAN stage output for two stage1b checkpoints on one case. "
            "Starts at query points and ends at the final object_context fed into Wan."
        )
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--checkpoint-a", type=Path, default=DEFAULT_CHECKPOINT_A)
    parser.add_argument("--checkpoint-b", type=Path, default=DEFAULT_CHECKPOINT_B)
    parser.add_argument("--label-a", default="step-003500")
    parser.add_argument("--label-b", default="step-005000")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case-name", default=DEFAULT_CASE_NAME)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--context-frames", type=int, default=20)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aux-device", default=None)
    parser.add_argument("--grounding-device", default=None)
    parser.add_argument("--vggt-cache-root", default=None)
    parser.add_argument("--initialize-model-on-cpu", action="store_true")
    parser.add_argument("--wan-root", default="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
    parser.add_argument("--diffsynth-root", default="/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
    parser.add_argument(
        "--lora-checkpoint",
        default="/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors",
    )
    parser.add_argument(
        "--stage1a-init-from",
        default="/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt",
    )
    return parser.parse_args()


def _resolve_runtime_device(device_arg: str) -> str:
    if str(device_arg).strip() and str(device_arg).strip().lower() != "cuda":
        return str(device_arg).strip()
    return infer0705._resolve_launch_device()


def _build_infer_args(
    args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
    output_dir: Path,
) -> argparse.Namespace:
    old_argv = list(sys.argv)
    try:
        sys.argv = [
            old_argv[0],
            "--checkpoint",
            str(checkpoint_dir),
            "--context-video",
            "/tmp/dummy_context.mp4",
            "--prompt",
            "dummy",
            "--output-dir",
            str(output_dir),
        ]
        infer_args = infer0705.parse_args()
    finally:
        sys.argv = old_argv

    infer_args.checkpoint = str(checkpoint_dir)
    infer_args.output_dir = str(output_dir)
    infer_args.device = _resolve_runtime_device(args.device)
    infer_args.height = int(args.height)
    infer_args.width = int(args.width)
    infer_args.context_frames = int(args.context_frames)
    infer_args.fps = int(args.fps)
    infer_args.seed = int(args.seed)
    infer_args.wan_root = str(args.wan_root)
    infer_args.diffsynth_root = str(args.diffsynth_root)
    infer_args.lora_checkpoint = str(args.lora_checkpoint)
    infer_args.stage1a_init_from = str(args.stage1a_init_from)
    infer_args.grounding_device = args.grounding_device
    infer_args.aux_device = args.aux_device
    infer_args.vggt_cache_root = args.vggt_cache_root
    infer_args.initialize_model_on_cpu = bool(args.initialize_model_on_cpu)
    infer_args.object_context_ablation = "none"
    return infer_args


def _load_context_video(
    *,
    source_video: Path,
    sampling_mode: str,
    context_frames: int,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_mode == "uniform":
        frames, frame_indices = read_video_uniform(source_video, int(context_frames))
    else:
        frames, frame_indices = read_video_prefix(source_video, int(context_frames))
    context_video_single = preprocess_video_rgb_uint8(
        frames,
        (int(height), int(width)),
        resize_mode="cover_crop",
        cover_crop_hw=(int(height), int(width)),
    )
    return context_video_single, frame_indices


def _tensor_hash(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    if not cpu.is_contiguous():
        cpu = cpu.contiguous()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
    cpu = tensor.detach().cpu().contiguous()
    fp32 = cpu.float()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "numel": int(cpu.numel()),
        "mean": float(fp32.mean().item()),
        "std": float(fp32.std().item()) if cpu.numel() > 1 else 0.0,
        "abs_mean": float(fp32.abs().mean().item()),
        "abs_max": float(fp32.abs().max().item()),
        "min": float(fp32.min().item()),
        "max": float(fp32.max().item()),
        "sha256": _tensor_hash(cpu),
    }


def _compare_optional_tensor(name: str, a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, object]:
    if a is None and b is None:
        return {
            "name": name,
            "same_shape": True,
            "exact_equal": True,
            "summary_a": None,
            "summary_b": None,
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
        }
    if a is None or b is None:
        return {
            "name": name,
            "same_shape": False,
            "exact_equal": False,
            "summary_a": None if a is None else _tensor_summary(a),
            "summary_b": None if b is None else _tensor_summary(b),
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }
    return _compare_tensor(name, a, b)


def _compare_tensor(name: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, object]:
    a_cpu = a.detach().cpu().contiguous()
    b_cpu = b.detach().cpu().contiguous()
    same_shape = list(a_cpu.shape) == list(b_cpu.shape)
    exact_equal = False
    max_abs_diff = None
    mean_abs_diff = None
    if same_shape:
        if a_cpu.dtype == b_cpu.dtype and torch.equal(a_cpu, b_cpu):
            exact_equal = True
            max_abs_diff = 0.0
            mean_abs_diff = 0.0
        else:
            diff = (a_cpu.float() - b_cpu.float()).abs()
            max_abs_diff = float(diff.max().item())
            mean_abs_diff = float(diff.mean().item())
    return {
        "name": name,
        "same_shape": same_shape,
        "exact_equal": exact_equal,
        "summary_a": _tensor_summary(a_cpu),
        "summary_b": _tensor_summary(b_cpu),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def _maybe_detach_cpu(value):
    if value is None:
        return None
    return value.detach().cpu()


def _extract_prewan_tensors(
    model,
    *,
    context_video_single: torch.Tensor,
    prompt: str,
    video_path: str,
) -> dict[str, torch.Tensor | list[str] | dict[str, object]]:
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

    query_points_prior, query_frame_ids, object_valid_mask, box_prior_xyxy = model._build_object_query_priors(
        sample,
        image_hw=image_hw,
    )
    query_points_prior = query_points_prior.to(device=device, dtype=pipe.torch_dtype)
    query_frame_ids = query_frame_ids.to(device=device, dtype=pipe.torch_dtype)
    object_valid_mask = object_valid_mask.to(device=device, dtype=pipe.torch_dtype)
    box_prior_xyxy = box_prior_xyxy.to(device=device, dtype=pipe.torch_dtype)

    frames_bthwc_01 = ((context_video.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0.0, 1.0)
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

    if model.vggt_cache_root:
        vggt_out = load_vggt_cache(sample, model.vggt_cache_root, allow_missing=False)
        if vggt_out is None:
            raise RuntimeError(f"VGGT cache missing for {video_path}")
    elif getattr(model, "vggt_adapter", None) is not None:
        vggt_device = getattr(model.vggt_adapter, "device_obj", device)
        vggt_out = model.vggt_adapter(
            frames_bthwc_01.to(vggt_device),
            query_points_prior=query_points_prior.to(vggt_device),
            query_image_hw=image_hw,
        )
        for attr_name in (
            "query_points",
            "tracks",
            "visibility",
            "confidence",
            "pose_enc",
            "depth",
            "depth_conf",
            "world_points",
            "world_points_conf",
            "dense_patch_tokens",
        ):
            attr_value = getattr(vggt_out, attr_name, None)
            if isinstance(attr_value, torch.Tensor):
                setattr(vggt_out, attr_name, attr_value.to(device))
    else:
        vggt_out = SimpleNamespace(
            world_points=None,
            world_points_conf=None,
            depth=None,
            depth_conf=None,
            dense_patch_tokens=None,
            patch_grid_hw=None,
            input_hw=None,
            image_hw=None,
        )

    jepa_out = model._run_jepa(context_video)
    context_latents = infer0705._encode_context_latents(pipe, context_video_single)
    object_out = model.object_pooler(
        jepa_patch_tokens=jepa_out.patch_tokens,
        context_latents=context_latents,
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

    grounding_sample = getattr(model, "_cached_viewer_grounding_sample", None)
    object_phrases = []
    if grounding_sample is not None and getattr(grounding_sample, "object_tracks", None) is not None:
        object_phrases = [str(track.phrase) for track in grounding_sample.object_tracks]

    return {
        "object_phrases": object_phrases,
        "query_points_prior": query_points_prior.detach().cpu(),
        "query_frame_ids": query_frame_ids.detach().cpu(),
        "object_valid_mask": object_valid_mask.detach().cpu(),
        "box_prior_xyxy": box_prior_xyxy.detach().cpu(),
        "cotracker_tracks": cotracker_out.tracks.detach().cpu(),
        "cotracker_visibility": cotracker_out.visibility.detach().cpu(),
        "cotracker_confidence": cotracker_out.confidence.detach().cpu(),
        "tracks_grouped": tracks_grouped.detach().cpu(),
        "visibility_grouped": visibility_grouped.detach().cpu(),
        "confidence_grouped": confidence_grouped.detach().cpu(),
        "vggt_world_points": _maybe_detach_cpu(getattr(vggt_out, "world_points", None)),
        "vggt_world_points_conf": _maybe_detach_cpu(getattr(vggt_out, "world_points_conf", None)),
        "vggt_depth": _maybe_detach_cpu(getattr(vggt_out, "depth", None)),
        "vggt_depth_conf": _maybe_detach_cpu(getattr(vggt_out, "depth_conf", None)),
        "vggt_dense_patch_tokens": _maybe_detach_cpu(getattr(vggt_out, "dense_patch_tokens", None)),
        "jepa_patch_tokens": jepa_out.patch_tokens.detach().cpu(),
        "context_latents": context_latents.detach().cpu(),
        "object_latent_tokens": object_out.object_latent_tokens.detach().cpu(),
        "object_context": object_context.detach().cpu(),
        "final_object_context_to_wan": object_context.detach().cpu(),
    }


def _collect_checkpoint_outputs(
    args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
    label: str,
    output_dir: Path,
    context_video_single: torch.Tensor,
    prompt: str,
    source_video: Path,
) -> dict[str, object]:
    infer_args = _build_infer_args(args, checkpoint_dir=checkpoint_dir, output_dir=output_dir)
    torch.manual_seed(int(infer_args.seed))
    model, model_args, load_info = infer0705._build_runtime_model(infer_args)
    _ = model_args
    try:
        tensors = _extract_prewan_tensors(
            model,
            context_video_single=context_video_single,
            prompt=prompt,
            video_path=str(source_video),
        )
        return {
            "label": label,
            "checkpoint": str(checkpoint_dir),
            "load_info": infer0705._summarize_load_info(load_info),
            "object_phrases": list(tensors.pop("object_phrases")),
            "tensors": tensors,
        }
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    output_dir = (args.output_root / args.case_name).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = core._load_input_json(args.input_json)
    source_video = Path(str(payload["source_video"])).expanduser().resolve()
    prompt = core._ensure_str_field(payload, "input_caption", args.input_json)
    context_video_single, frame_indices = _load_context_video(
        source_video=source_video,
        sampling_mode=str(args.sampling_mode),
        context_frames=int(args.context_frames),
        height=int(args.height),
        width=int(args.width),
    )

    output_a = _collect_checkpoint_outputs(
        args,
        checkpoint_dir=args.checkpoint_a.expanduser().resolve(),
        label=str(args.label_a),
        output_dir=output_dir / str(args.label_a),
        context_video_single=context_video_single,
        prompt=prompt,
        source_video=source_video,
    )
    output_b = _collect_checkpoint_outputs(
        args,
        checkpoint_dir=args.checkpoint_b.expanduser().resolve(),
        label=str(args.label_b),
        output_dir=output_dir / str(args.label_b),
        context_video_single=context_video_single,
        prompt=prompt,
        source_video=source_video,
    )

    tensor_names = [
        "query_points_prior",
        "query_frame_ids",
        "object_valid_mask",
        "box_prior_xyxy",
        "cotracker_tracks",
        "cotracker_visibility",
        "cotracker_confidence",
        "tracks_grouped",
        "visibility_grouped",
        "confidence_grouped",
        "vggt_world_points",
        "vggt_world_points_conf",
        "vggt_depth",
        "vggt_depth_conf",
        "vggt_dense_patch_tokens",
        "jepa_patch_tokens",
        "context_latents",
        "object_latent_tokens",
        "object_context",
        "final_object_context_to_wan",
    ]
    comparisons = []
    for name in tensor_names:
        comparisons.append(_compare_optional_tensor(name, output_a["tensors"][name], output_b["tensors"][name]))

    report = {
        "input_json": str(args.input_json),
        "source_video": str(source_video),
        "prompt": prompt,
        "frame_indices": frame_indices.tolist(),
        "sampling_mode": str(args.sampling_mode),
        "context_frames": int(context_video_single.shape[1]),
        "checkpoint_a": {
            "label": output_a["label"],
            "path": output_a["checkpoint"],
            "object_phrases": output_a["object_phrases"],
            "load_info": output_a["load_info"],
        },
        "checkpoint_b": {
            "label": output_b["label"],
            "path": output_b["checkpoint"],
            "object_phrases": output_b["object_phrases"],
            "load_info": output_b["load_info"],
        },
        "comparisons": comparisons,
    }
    report_path = output_dir / "compare_prewan_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "output_dir": str(output_dir),
        "report": str(report_path),
        "comparison_count": len(comparisons),
        "all_exact_equal": bool(all(bool(item["exact_equal"]) for item in comparisons)),
        "different_steps": [item["name"] for item in comparisons if not bool(item["exact_equal"])],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
