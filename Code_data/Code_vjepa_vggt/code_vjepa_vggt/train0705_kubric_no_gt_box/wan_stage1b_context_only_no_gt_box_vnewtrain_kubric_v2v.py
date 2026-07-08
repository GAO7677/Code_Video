
'''
Run command example (single GPU):
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name train_stage1b_kubric0708_step1000 \
  --output-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708 \
  --num-inference-steps 40 \
  --num-frames 49

Run command example (two GPU):
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=0,1 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name train_stage1b_kubric0708_step1000_f98 \
  --output-root /data/gaoya/AAA_test_video/tmp \
  --inference-devices cuda:0,cuda:1 \
  --num-inference-steps 40 \
  --output-num-frames 98

'''
from __future__ import annotations

"""
Batch v2v inference for the Kubric stage1b context-only no-GT-box model.

This follows the same txt/json-list driven workflow as the train0705 batch
script, but swaps the runtime model and object-conditioning path to the Kubric
no-GT-box variant. It also supports an optional two-device inference layout:

  --inference-devices cuda:0,cuda:1
      First device: Wan inference / DiT / VAE / object adapter
      Second device: auxiliary object stack device

Typical single-GPU run:
  PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
  CUDA_VISIBLE_DEVICES=2 \
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.py \
    --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
    --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
    --model-name train_stage1b_kubric0708_step1000 \
    --output-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708 \
    --num-inference-steps 40 \
    --num-frames 49

Typical two-GPU run:
  PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
  CUDA_VISIBLE_DEVICES=2,3 \
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.py \
    --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
    --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
    --model-name train_stage1b_kubric0708_step1000 \
    --output-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708 \
    --inference-devices cuda:0,cuda:1 \
    --num-inference-steps 40 \
    --output-num-frames 49

Default output root:
  /data/gaoya/AAA_test_video/0623/test/v2v/<model-name>/<step-name>
"""

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _read_cli_arg_value(argv: list[str], names: tuple[str, ...], default: str | None = None) -> str | None:
    for name in names:
        if name not in argv:
            continue
        index = argv.index(name)
        if index + 1 < len(argv):
            return argv[index + 1]
    return default


_DEFAULT_DIFFSYNTH_ROOT_STR = "/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main"
_SELECTED_DIFFSYNTH_ROOT = _read_cli_arg_value(
    sys.argv,
    ("--diffsynth-root", "--diffsynth_root"),
    os.environ.get("DIFFSYNTH_ROOT", _DEFAULT_DIFFSYNTH_ROOT_STR),
)
if _SELECTED_DIFFSYNTH_ROOT:
    os.environ["DIFFSYNTH_ROOT"] = _SELECTED_DIFFSYNTH_ROOT
    if _SELECTED_DIFFSYNTH_ROOT not in sys.path:
        sys.path.insert(0, _SELECTED_DIFFSYNTH_ROOT)

from code_vjepa_vggt import batch_infer_v_newtrain_from_jsonl as core
from code_vjepa_vggt.AAAinfer.utils.named_paths import resolve_output_root
from code_vjepa_vggt.train0705 import infer_stage1b_context_only_no_gt_box_v_newtrain0705 as infer0705
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    infer_stage1b_context_only_no_gt_box_v_newtrain_kubric as kubric_infer,
)
from code_vjepa_vggt.utils.video_io import (
    preprocess_video_rgb_uint8,
    read_video_prefix,
    read_video_uniform,
)
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


def _install_kubric_runtime_hooks() -> None:
    infer0705.t0705 = kubric_infer.trainmod
    infer0705._build_object_context = kubric_infer._build_object_context
    infer0705._build_model_args = kubric_infer._build_model_args


def _normalize_ckpt_method_name(name: str) -> str:
    normalized = re.sub(r"^[A-Za-z]+\d+_", "", name, count=1)
    return normalized or name


def _build_method_name_from_checkpoint_dir(
    checkpoint_dir: Path,
    *,
    context_frames: int,
    num_frames: int,
) -> str:
    step_name = checkpoint_dir.name
    checkpoint_parent = checkpoint_dir.parent
    suffix = f"_ctx{int(context_frames):02d}_{int(num_frames):02d}f"
    if checkpoint_parent.name == "checkpoints" and checkpoint_parent.parent.name:
        method_root = _normalize_ckpt_method_name(checkpoint_parent.parent.name)
        return f"{method_root}_{step_name}{suffix}"
    if checkpoint_parent.name:
        method_root = _normalize_ckpt_method_name(checkpoint_parent.name)
        return f"{method_root}_{step_name}{suffix}"
    return f"{step_name}{suffix}"


def _normalize_device_token(raw_value: str) -> str:
    value = str(raw_value).strip()
    if not value:
        raise ValueError("empty device token")
    if value.lower() == "cpu":
        return "cpu"
    if value.isdigit():
        return f"cuda:{int(value)}"
    if value.startswith("cuda:"):
        suffix = value.split(":", 1)[1].strip()
        if suffix.isdigit():
            return f"cuda:{int(suffix)}"
    raise ValueError(
        f"unsupported device token: {raw_value!r}; expected forms like '0', '1', 'cuda:0', 'cuda:1'"
    )


def _parse_two_gpu_devices(raw_value: str) -> tuple[str, str]:
    parts = [part.strip() for part in str(raw_value).split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"--inference-devices expects exactly two devices, got {raw_value!r}")
    return _normalize_device_token(parts[0]), _normalize_device_token(parts[1])


def _resolve_runtime_device(device_arg: str) -> str:
    if str(device_arg).strip() and str(device_arg).strip().lower() != "cuda":
        return str(device_arg).strip()
    return infer0705._resolve_launch_device()


def _resolve_runtime_devices(cli_args: argparse.Namespace) -> tuple[str, str | None]:
    main_device: str | None = None
    aux_device = None if cli_args.aux_device is None else str(cli_args.aux_device).strip() or None

    if cli_args.inference_devices:
        inferred_main, inferred_aux = _parse_two_gpu_devices(str(cli_args.inference_devices))
        main_device = inferred_main
        if aux_device is None:
            aux_device = inferred_aux

    if main_device is None:
        main_device = _resolve_runtime_device(str(cli_args.device))

    return main_device, aux_device


def _load_context_video_for_mode(
    *,
    video_path: Path,
    target_context_frames: int,
    sampling_mode: str,
):
    if sampling_mode == "uniform":
        frames, frame_indices = read_video_uniform(video_path, target_context_frames)
    else:
        frames, frame_indices = read_video_prefix(video_path, target_context_frames)
    if int(frames.shape[0]) <= 0:
        raise RuntimeError(f"context video {video_path} does not provide any readable frames")
    if int(frames.shape[0]) > int(target_context_frames):
        frames = frames[:target_context_frames]
        frame_indices = frame_indices[:target_context_frames]
    return frames, frame_indices


def _resolve_source_video(payload: dict[str, object], json_path: Path) -> str:
    source_video = payload.get("source_video")
    if isinstance(source_video, str) and source_video.strip():
        return source_video.strip()
    return core._resolve_input_video(payload, json_path)


def _save_context_contact_sheet(
    *,
    context_pil: list[Image.Image],
    output_path: Path,
) -> None:
    if not context_pil:
        raise RuntimeError("context_pil is empty; cannot save contact sheet")
    widths = [int(image.width) for image in context_pil]
    heights = [int(image.height) for image in context_pil]
    canvas = Image.new("RGB", (sum(widths), max(heights)))
    cursor_x = 0
    for image in context_pil:
        canvas.paste(image, (cursor_x, 0))
        cursor_x += int(image.width)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=95)


def _has_complete_existing_output(output_video: Path, output_json: Path) -> bool:
    if not output_video.exists() or not output_json.exists():
        return False
    try:
        with output_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return False

    input_video = payload.get("input_video")
    if not isinstance(input_video, str) or not input_video.strip():
        return False

    input_video_path = Path(input_video).expanduser()
    if not input_video_path.is_absolute():
        input_video_path = (output_json.parent / input_video_path).resolve()
    else:
        input_video_path = input_video_path.resolve()

    return input_video_path.exists()


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "cuda out of memory" in message or "out of memory" in message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run the Kubric stage1b context-only no-GT-box inference pipeline "
            "over a txt file containing one input json path per line."
        )
    )
    parser.add_argument("--weights-root", type=Path, required=True, help="step-* dir containing checkpoint.safetensors")
    parser.add_argument("--input-json-list-path", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--aux-device", type=str, default=None)
    parser.add_argument(
        "--inference-devices",
        type=str,
        default=None,
        help="Optional two-device layout like cuda:0,cuda:1. First is main inference device, second is aux device.",
    )
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--diffsynth-root", type=Path, default=DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument("--lora-checkpoint", type=Path, default=DEFAULT_BASE_LORA)
    parser.add_argument("--stage1a-init-from", type=Path, default=DEFAULT_STAGE1A)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument(
        "--input-cover-crop-height",
        type=int,
        default=480,
        help="Resize input video proportionally to cover this height before center cropping.",
    )
    parser.add_argument(
        "--input-cover-crop-width",
        type=int,
        default=832,
        help="Resize input video proportionally to cover this width before center cropping.",
    )
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument(
        "--output-num-frames",
        type=int,
        default=None,
        help="Alias of --num-frames. If set, overrides --num-frames.",
    )
    parser.add_argument("--context-frames", type=int, default=20)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=5)
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
    parser.add_argument("--jepa-ckpt-path", default="/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth")
    parser.add_argument("--jepa-input-size", type=int, default=384)
    parser.add_argument("--jepa-patch-size", type=int, default=16)
    parser.add_argument("--jepa-tubelet-size", type=int, default=2)
    parser.add_argument("--cotracker-checkpoint", default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")
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
    parser.add_argument("--initialize-model-on-cpu", action="store_true")
    infer0705.add_vjepa_cli_args(parser)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--object-context-ablation",
        choices=["none", "zero", "random"],
        default="none",
        help="Replace the final object_context fed into Wan DiT for ablation.",
    )
    parser.add_argument(
        "--object-context-random-seed",
        type=int,
        default=None,
        help="Optional RNG seed used when --object-context-ablation=random.",
    )
    parser.add_argument(
        "--object-context-random-scale",
        type=float,
        default=1.0,
        help="Std multiplier used when --object-context-ablation=random.",
    )
    return parser.parse_args()


def _build_runtime_args(cli_args: argparse.Namespace, checkpoint_dir: Path, output_dir: Path) -> argparse.Namespace:
    runtime_kwargs = dict(
        checkpoint=str(checkpoint_dir),
        context_video="",
        prompt="",
        output_dir=str(output_dir),
        wan_root=str(cli_args.wan_root),
        diffsynth_root=str(cli_args.diffsynth_root),
        lora_checkpoint=str(cli_args.lora_checkpoint),
        stage1a_init_from=str(cli_args.stage1a_init_from),
        num_frames=int(cli_args.num_frames),
        context_frames=int(cli_args.context_frames),
        sampling_steps=int(cli_args.num_inference_steps),
        height=int(cli_args.height),
        width=int(cli_args.width),
        fps=int(cli_args.fps),
        seed=int(cli_args.seed),
        cfg_scale=float(cli_args.cfg_scale),
        quality=int(cli_args.quality),
        lora_rank=int(cli_args.lora_rank),
        lora_alpha=int(cli_args.lora_alpha),
        disable_object_branch=bool(cli_args.disable_object_branch),
        object_num_queries=int(cli_args.object_num_queries),
        aux_max_objects=int(cli_args.aux_max_objects),
        object_pooler_latent_dim=int(cli_args.object_pooler_latent_dim),
        cond_proj_dim=int(cli_args.cond_proj_dim),
        jepa_window_radius=int(cli_args.jepa_window_radius),
        latent_window_radius=int(cli_args.latent_window_radius),
        object_gate_init=float(cli_args.object_gate_init),
        jepa_ckpt_path=str(cli_args.jepa_ckpt_path),
        jepa_input_size=int(cli_args.jepa_input_size),
        jepa_patch_size=int(cli_args.jepa_patch_size),
        jepa_tubelet_size=int(cli_args.jepa_tubelet_size),
        cotracker_checkpoint=str(cli_args.cotracker_checkpoint),
        cotracker_input_h=int(cli_args.cotracker_input_h),
        cotracker_input_w=int(cli_args.cotracker_input_w),
        cotracker_window_len=int(cli_args.cotracker_window_len),
        vggt_model_path=str(cli_args.vggt_model_path),
        vggt_input_h=int(cli_args.vggt_input_h),
        vggt_input_w=int(cli_args.vggt_input_w),
        vggt_cache_root=cli_args.vggt_cache_root,
        grounding_device=cli_args.grounding_device,
        sam2_segment_len=int(cli_args.sam2_segment_len),
        grounding_proposal_source=str(cli_args.grounding_proposal_source),
        grounding_motion_score_ratio=float(cli_args.grounding_motion_score_ratio),
        grounding_text_prompt=str(cli_args.grounding_text_prompt),
        grounding_extra_prompt_terms=str(cli_args.grounding_extra_prompt_terms),
        grounding_disable_caption_terms=bool(cli_args.grounding_disable_caption_terms),
        grounding_gdino_box_threshold=float(cli_args.grounding_gdino_box_threshold),
        grounding_gdino_text_threshold=float(cli_args.grounding_gdino_text_threshold),
        grounding_prompt_frame_mode=str(cli_args.grounding_prompt_frame_mode),
        grounding_track_dedupe_iou_threshold=float(cli_args.grounding_track_dedupe_iou_threshold),
        grounding_container_suppress_ratio_threshold=float(cli_args.grounding_container_suppress_ratio_threshold),
        grounding_container_suppress_min_contained=int(cli_args.grounding_container_suppress_min_contained),
        grounding_container_suppress_min_area_ratio=float(cli_args.grounding_container_suppress_min_area_ratio),
        grounding_container_suppress_small_iou_threshold=float(cli_args.grounding_container_suppress_small_iou_threshold),
        device=str(cli_args.device),
        aux_device=cli_args.aux_device,
        initialize_model_on_cpu=bool(cli_args.initialize_model_on_cpu),
    )
    for name in infer0705._VJEPA_RUNTIME_ARG_NAMES:
        runtime_kwargs[name] = getattr(cli_args, name)
    return argparse.Namespace(**runtime_kwargs)


def _run_single_case_in_process(
    *,
    model,
    checkpoint_dir: Path,
    input_json_path: Path,
    source_video: str,
    input_caption: str,
    output_dir: Path,
    output_video: Path,
    num_frames: int,
    context_frames: int,
    sampling_mode: str,
    sampling_steps: int,
    fps: int,
    seed: int,
    cfg_scale: float,
    height: int,
    width: int,
    input_cover_crop_height: int,
    input_cover_crop_width: int,
    quality: int,
    load_info: dict[str, object],
    lora_checkpoint: str,
    stage1a_init_from: str,
    vjepa_summary: dict | None,
) -> tuple[dict[str, object], list[str]]:
    logs: list[str] = []
    logs.append(f"[case] input_json={input_json_path}")
    logs.append(f"[case] source_video={source_video}")
    logs.append(f"[case] input_caption={input_caption}")
    logs.append(f"[case] checkpoint_dir={checkpoint_dir}")

    context_video_path = Path(source_video).expanduser().resolve()
    frames, frame_indices = _load_context_video_for_mode(
        video_path=context_video_path,
        target_context_frames=int(context_frames),
        sampling_mode=sampling_mode,
    )
    effective_context_frames = int(frames.shape[0])
    context_video_single = preprocess_video_rgb_uint8(
        frames,
        (int(height), int(width)),
        resize_mode="cover_crop",
        cover_crop_hw=(int(input_cover_crop_height), int(input_cover_crop_width)),
    )
    context_pil = infer0705._tensor_video_to_pil_list(context_video_single)
    object_context, object_debug = infer0705._build_object_context(
        model=model,
        context_video_single=context_video_single,
        prompt=str(input_caption),
        video_path=str(context_video_path),
    )
    object_context, ablation_debug = infer0705._apply_object_context_ablation(
        object_context,
        mode=str(getattr(model, "_object_context_ablation_mode", "none")),
        random_seed=getattr(model, "_object_context_random_seed", None),
        random_scale=float(getattr(model, "_object_context_random_scale", 1.0)),
    )
    object_debug["object_context_ablation"] = ablation_debug

    pipe = model.pipe
    pipe.dit.eval()
    with torch.no_grad():
        pipe_kwargs = dict(
            prompt=str(input_caption),
            negative_prompt="",
            context_video=context_pil,
            seed=int(seed),
            tiled=True,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            num_inference_steps=int(sampling_steps),
            cfg_scale=float(cfg_scale),
        )
        if bool(getattr(model, "enable_object_branch", False)):
            pipe_kwargs["object_context"] = object_context
        video = pipe(**pipe_kwargs)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    context_sheet_path = output_video.with_name(
        f"{output_video.stem}_input_ctx{effective_context_frames:02d}.jpg"
    )
    _save_context_contact_sheet(context_pil=context_pil, output_path=context_sheet_path)
    save_video(video, str(output_video), fps=int(fps), quality=int(quality))

    result = {
        "input_json": str(input_json_path),
        "input_video": str(context_sheet_path),
        "source_video": str(source_video),
        "input_caption": str(input_caption),
        "output_video": str(output_video),
        "seed": int(seed),
        "step": int(sampling_steps),
        "guidance": float(cfg_scale),
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
        },
        "model_args": {
            "height": int(height),
            "width": int(width),
            "num_frames": int(num_frames),
            "context_frames": effective_context_frames,
            "input_resize_mode": "cover_crop",
            "input_cover_crop_height": int(input_cover_crop_height),
            "input_cover_crop_width": int(input_cover_crop_width),
            "enable_object_branch": bool(getattr(model, "enable_object_branch", False)),
            "lora_checkpoint": str(lora_checkpoint),
            "stage1a_init_from": str(stage1a_init_from),
        },
        "vjepa": vjepa_summary,
    }
    return result, logs


def main() -> None:
    _install_kubric_runtime_hooks()

    cli_args = parse_args()
    if cli_args.output_num_frames is not None:
        cli_args.num_frames = int(cli_args.output_num_frames)

    infer0705.apply_vjepa_preset_if_requested(cli_args)
    weights_root = cli_args.weights_root.expanduser().resolve()
    input_json_list_path = cli_args.input_json_list_path.expanduser().resolve()
    model_name = str(cli_args.model_name).strip()
    output_root = resolve_output_root(
        explicit_output_root=cli_args.output_root,
        base_output_root="/data/gaoya/AAA_test_video/0623/test/v2v",
        model_name=model_name,
    )

    if not weights_root.exists():
        raise FileNotFoundError(f"weights-root not found: {weights_root}")

    cli_args.device, cli_args.aux_device = _resolve_runtime_devices(cli_args)
    torch.manual_seed(int(cli_args.seed))
    np.random.seed(int(cli_args.seed))

    json_paths = core._read_list_file(input_json_list_path)
    if cli_args.limit is not None:
        json_paths = json_paths[: max(0, int(cli_args.limit))]

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_json_list_path": str(input_json_list_path),
        "weights_root": str(weights_root),
        "num_items": len(json_paths),
        "num_inference_steps": int(cli_args.num_inference_steps),
        "cfg_scale": float(cli_args.cfg_scale),
        "seed": int(cli_args.seed),
        "height": int(cli_args.height),
        "width": int(cli_args.width),
        "input_resize_mode": "cover_crop",
        "input_cover_crop_height": int(cli_args.input_cover_crop_height),
        "input_cover_crop_width": int(cli_args.input_cover_crop_width),
        "num_frames": int(cli_args.num_frames),
        "context_frames": int(cli_args.context_frames),
        "sampling_mode": str(cli_args.sampling_mode),
        "initialize_model_on_cpu": bool(cli_args.initialize_model_on_cpu),
        "disable_object_branch": bool(cli_args.disable_object_branch),
        "device": str(cli_args.device),
        "aux_device": cli_args.aux_device,
        "inference_devices": cli_args.inference_devices,
        "object_context_ablation": {
            "mode": str(cli_args.object_context_ablation),
            "random_seed": cli_args.object_context_random_seed,
            "random_scale": float(cli_args.object_context_random_scale),
        },
        "vjepa": infer0705.summarize_vjepa_args(cli_args),
    }
    with (output_root / "batch_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    step_output_dir = output_root / weights_root.name
    step_output_dir.mkdir(parents=True, exist_ok=True)
    method_name = _build_method_name_from_checkpoint_dir(
        weights_root,
        context_frames=int(cli_args.context_frames),
        num_frames=int(cli_args.num_frames),
    )

    runtime_args = _build_runtime_args(cli_args, weights_root, step_output_dir)
    model, _, load_info = infer0705._build_runtime_model(runtime_args)
    model.to(torch.device(cli_args.device))
    model.eval()
    model.pipe.dit.eval()
    model._object_context_ablation_mode = str(cli_args.object_context_ablation)
    model._object_context_random_seed = cli_args.object_context_random_seed
    model._object_context_random_scale = float(cli_args.object_context_random_scale)

    step_success = 0
    step_failed = 0
    step_skipped = 0
    step_entries: list[dict[str, object]] = []
    step_log_lines = [
        f"[checkpoint] {weights_root}",
        f"[device] main={cli_args.device} aux={cli_args.aux_device}",
    ]

    for input_json_path in json_paths:
        payload = core._load_input_json(input_json_path)
        try:
            source_video = _resolve_source_video(payload, input_json_path)
            input_caption = core._ensure_str_field(payload, "input_caption", input_json_path)
        except (KeyError, ValueError) as exc:
            print(f"[skip] {weights_root.name} {input_json_path.stem}: {exc}")
            step_skipped += 1
            continue

        sample_stem = input_json_path.stem
        output_video = step_output_dir / f"{sample_stem}.mp4"
        output_json = step_output_dir / f"{sample_stem}.json"
        output_log = step_output_dir / f"{sample_stem}.log"

        if _has_complete_existing_output(output_video, output_json) and not (
            cli_args.force or cli_args.overwrite
        ):
            print(f"[skip] {weights_root.name} {sample_stem}")
            step_skipped += 1
            continue

        try:
            result, case_logs = _run_single_case_in_process(
                model=model,
                checkpoint_dir=weights_root,
                input_json_path=input_json_path,
                source_video=source_video,
                input_caption=input_caption,
                output_dir=step_output_dir,
                output_video=output_video,
                num_frames=int(cli_args.num_frames),
                context_frames=int(cli_args.context_frames),
                sampling_mode=str(cli_args.sampling_mode),
                sampling_steps=int(cli_args.num_inference_steps),
                fps=int(cli_args.fps),
                seed=int(cli_args.seed),
                cfg_scale=float(cli_args.cfg_scale),
                height=int(cli_args.height),
                width=int(cli_args.width),
                input_cover_crop_height=int(cli_args.input_cover_crop_height),
                input_cover_crop_width=int(cli_args.input_cover_crop_width),
                quality=int(cli_args.quality),
                load_info=load_info,
                lora_checkpoint=str(cli_args.lora_checkpoint),
                stage1a_init_from=str(cli_args.stage1a_init_from),
                vjepa_summary=infer0705.summarize_vjepa_args(cli_args),
            )
        except Exception as exc:
            error_lines = step_log_lines + [f"[error] {sample_stem}: {exc}"]
            core._write_text_lines(output_log, error_lines)
            print(f"[error] {weights_root.name} {sample_stem}: {exc}")
            step_failed += 1
            if _is_cuda_oom(exc):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise SystemExit(86) from exc
            continue

        success_lines = step_log_lines + case_logs + [f"[done] {weights_root.name} {sample_stem}"]
        core._write_text_lines(output_log, success_lines)
        result["method"] = method_name
        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        step_entries.append(result)
        print(f"[done] {weights_root.name} {sample_stem} -> {output_video}")
        step_success += 1

    step_summary = {
        "checkpoint_dir": str(weights_root),
        "method": method_name,
        "num_total": len(json_paths),
        "num_success": step_success,
        "num_failed": step_failed,
        "num_skipped": step_skipped,
        "entries": step_entries,
    }
    with (step_output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(step_summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    summary = {
        "weights_root": str(weights_root),
        "output_root": str(output_root),
        "step": weights_root.name,
        "num_total": len(json_paths),
        "num_success": step_success,
        "num_failed": step_failed,
        "num_skipped": step_skipped,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
