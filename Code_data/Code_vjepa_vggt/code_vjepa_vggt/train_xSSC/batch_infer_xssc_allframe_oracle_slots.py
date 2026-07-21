
#!/usr/bin/env python3
"""Standalone batch v2v inference for Scheme A all-frame oracle xSSC slots.

This script copies the original json-list batch inference flow, but calls the
Scheme A runtime/model/object-context builders directly instead of installing
runtime hooks into the legacy infer0705 module.
"""
from __future__ import annotations

"""
Typical single-GPU run:
  PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
  CUDA_VISIBLE_DEVICES=2 \
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/batch_infer_xssc_allframe_oracle_slots.py \
    --weights-root /path/to/scheme_a/checkpoints/step-001500 \
    --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
    --model-name xssc_allframe_oracle_step1500 \
    --output-root /data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/test_5 \
    --num-inference-steps 40 \
    --context-frames 8 \
    --num-frames 49

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


def _cli_flag_present(argv: list[str], names: tuple[str, ...]) -> bool:
    return any(name in argv for name in names)


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
from code_vjepa_vggt.context_wan_v_newtrain import ObjectBranchInstabilityError
from code_vjepa_vggt.train0705 import infer_stage1b_context_only_no_gt_box_v_newtrain0705 as infer0705
from code_vjepa_vggt.train_xSSC import infer_xssc_allframe_oracle_slots as oracle_infer
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

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def _tensor_numeric_stats(tensor: torch.Tensor | None) -> dict[str, float | int | list[int] | None]:
    if tensor is None:
        return {
            "shape": None,
            "dtype": None,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "abs_max": None,
            "nan_count": None,
            "inf_count": None,
        }
    with torch.no_grad():
        tensor_f32 = tensor.detach().float()
        finite = torch.isfinite(tensor_f32)
        safe = torch.nan_to_num(tensor_f32, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "mean": float(safe.mean().item()),
            "std": float(safe.std(unbiased=False).item()),
            "min": float(safe.min().item()),
            "max": float(safe.max().item()),
            "abs_max": float(safe.abs().max().item()),
            "nan_count": int(torch.isnan(tensor_f32).sum().item()),
            "inf_count": int(torch.isinf(tensor_f32).sum().item()),
            "finite_ratio": float(finite.float().mean().item()),
        }
def _normalize_ckpt_method_name(name: str) -> str:
    normalized = re.sub(r"^[A-Za-z]+\d+_", "", name, count=1)
    return normalized or name


def _build_method_name_from_checkpoint_dir(
    checkpoint_dir: Path,
    *,
    context_frames: int,
    num_frames: int,
    sampling_steps: int,
    height: int,
    width: int,
    negative_prompt: str | None,
) -> str:
    step_name = checkpoint_dir.name
    checkpoint_parent = checkpoint_dir.parent
    if negative_prompt is None:
        neg_tag = "nullnegprompt"
    elif negative_prompt == "":
        neg_tag = "emptynegprompt"
    elif negative_prompt == DEFAULT_NEGATIVE_PROMPT:
        neg_tag = "defaultnegprompt"
    else:
        neg_tag = "customnegprompt"
    suffix = (
        f"_steps{int(sampling_steps):02d}"
        f"_{int(height)}x{int(width)}"
        f"_ctx{int(context_frames):02d}"
        f"_{int(num_frames):02d}f"
        f"_{neg_tag}"
    )
    if checkpoint_parent.name == "checkpoints" and checkpoint_parent.parent.name:
        method_root = _normalize_ckpt_method_name(checkpoint_parent.parent.name)
        return f"{method_root}_{step_name}{suffix}"
    if checkpoint_parent.name:
        method_root = _normalize_ckpt_method_name(checkpoint_parent.name)
        return f"{method_root}_{step_name}{suffix}"
    return f"{step_name}{suffix}"


def _append_method_suffix(method_name: str, suffix: str | None) -> str:
    if suffix is None:
        return method_name
    suffix_norm = str(suffix).strip()
    if not suffix_norm:
        return method_name
    return f"{method_name}_{suffix_norm}"


def _normalize_shard_tag(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None
    sanitized = re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("._-")
    return sanitized or None


def _resolve_step_output_dir_name(
    raw_value: str | None,
    *,
    checkpoint_dir: Path,
    method_name: str,
) -> str:
    if raw_value is None:
        return checkpoint_dir.name
    value = str(raw_value).strip()
    if not value:
        return checkpoint_dir.name
    if value == "__METHOD_NAME__":
        return method_name
    return value


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


def _resolve_negative_prompt_from_cli(cli_args: argparse.Namespace) -> str | None:
    if getattr(cli_args, "_negative_prompt_provided", False):
        return str(cli_args.negative_prompt)
    return None


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


def _dump_pipe_inputs(
    *,
    dump_root: Path,
    sample_stem: str,
    context_pil: list[Image.Image],
    prompt: str,
    negative_prompt: str | None,
    pipe_kwargs: dict[str, object],
    source_video: str,
    frame_indices,
) -> None:
    sample_dir = dump_root / sample_stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    _save_context_contact_sheet(
        context_pil=context_pil,
        output_path=sample_dir / "pipe_context_contact_sheet.jpg",
    )
    for frame_idx, image in enumerate(context_pil):
        image.save(sample_dir / f"context_frame_{frame_idx:02d}.png")

    (sample_dir / "prompt.txt").write_text(str(prompt) + "\n", encoding="utf-8")
    (sample_dir / "negative_prompt.txt").write_text(
        "" if negative_prompt is None else str(negative_prompt),
        encoding="utf-8",
    )
    payload = {
        "source_video": str(source_video),
        "prompt": str(prompt),
        "negative_prompt": negative_prompt,
        "frame_indices": frame_indices.tolist(),
        "num_context_frames": len(context_pil),
        "pipe_kwargs": {
            key: value
            for key, value in pipe_kwargs.items()
            if key not in {"context_video", "object_context"}
        },
        "has_object_context": "object_context" in pipe_kwargs,
    }
    (sample_dir / "pipe_inputs.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
            "Batch-run Scheme A all-frame oracle xSSC slot inference over a txt file "
            "containing one input json path per line."
        )
    )
    parser.add_argument("--weights-root", type=Path, required=True, help="step-* dir containing checkpoint.safetensors")
    parser.add_argument("--input-json-list-path", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--step-output-dir-name",
        type=str,
        default=None,
        help="Optional final result subdirectory name under output-root. Defaults to weights_root.name. Use __METHOD_NAME__ to derive the full method suffix name automatically.",
    )
    parser.add_argument("--method-suffix", type=str, default=None)
    parser.add_argument(
        "--shard-tag",
        type=str,
        default=None,
        help="Optional shard tag used to disambiguate batch-level summary filenames during multi-process launches.",
    )
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
        default=512,
        help="Resize input video proportionally to cover this height before center cropping.",
    )
    parser.add_argument(
        "--input-cover-crop-width",
        type=int,
        default=896,
        help="Resize input video proportionally to cover this width before center cropping.",
    )
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument(
        "--output-num-frames",
        type=int,
        default=None,
        help="Alias of --num-frames. If set, overrides --num-frames.",
    )
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help=(
            "Negative prompt to pass into the pipeline. "
            "Pass an empty string to use ''. If this argument is omitted, the default "
            "DEFAULT_NEGATIVE_PROMPT is used."
        ),
    )
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
    parser.add_argument("--xssc-root", type=Path, default=Path(oracle_infer.train.DEFAULT_XSSC_ROOT))
    parser.add_argument("--xssc-config", type=Path, default=Path(oracle_infer.train.DEFAULT_XSSC_CONFIG))
    parser.add_argument("--xssc-checkpoint", type=Path, default=Path(oracle_infer.train.DEFAULT_XSSC_CHECKPOINT))
    parser.add_argument("--xssc-input-size", type=int, default=256)
    parser.add_argument("--xssc-max-time-steps", type=int, default=64)
    parser.add_argument("--xssc-oracle-video-frames", type=int, default=oracle_infer.train.DEFAULT_XSSC_ORACLE_VIDEO_FRAMES)
    parser.add_argument("--xssc-vae-temporal-stride", type=int, default=oracle_infer.train.DEFAULT_WAN_VAE_TEMPORAL_STRIDE)
    parser.add_argument(
        "--xssc-oracle-sampling-mode",
        choices=["prefix", "uniform"],
        default="prefix",
        help="How to read the full source video for oracle xSSC slots.",
    )
    parser.add_argument(
        "--xssc-oracle-video-resize-mode",
        choices=["cover_crop", "stretch"],
        default="cover_crop",
        help="Preprocess mode for the 49-frame oracle video before xSSC.",
    )
    parser.add_argument("--xssc-preprocess-mode", default="center_crop")
    parser.add_argument("--object-lora-rank", type=int, default=32)
    parser.add_argument("--object-lora-alpha", type=float, default=32.0)
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
    parser.add_argument(
        "--grounding-caption-prompt-mode",
        choices=["known_terms", "physical_noun_phrases"],
        default="known_terms",
    )
    parser.add_argument("--grounding-caption-max-phrases", type=int, default=4)
    parser.add_argument("--grounding-caption-min-score", type=float, default=4.0)
    parser.add_argument("--grounding-disable-caption-terms", action="store_true", default=True)
    parser.add_argument(
        "--grounding-enable-caption-terms",
        dest="grounding_disable_caption_terms",
        action="store_false",
        help="Include caption-derived nouns in the GroundingDINO prompt.",
    )
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
        choices=["none", "zero", "random", "keep_slot"],
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
    parser.add_argument(
        "--object-context-keep-slot-ids",
        type=str,
        default=None,
        help="Comma-separated slot ids to keep when --object-context-ablation=keep_slot.",
    )
    parser.add_argument(
        "--object-context-scale-factor",
        type=float,
        default=1.0,
        help="Optional multiplicative factor applied to the final object_context after ablation.",
    )
    parser.add_argument(
        "--object-context-token-norm-max",
        type=float,
        default=None,
        help="Optional per-token L2 norm clamp applied to the final object_context after ablation.",
    )
    parser.add_argument(
        "--object-adapter-mlp-residual-max-ratio",
        type=float,
        default=None,
        help="Optional per-token RMS cap for the adapter MLP residual before out_norm.",
    )
    parser.add_argument(
        "--compact-object-context-slots",
        action="store_true",
        help="Physically remove invalid slot tokens before DiT object cross-attention.",
    )
    parser.add_argument(
        "--object-branch-ratio-guard-max-ratio",
        type=float,
        default=None,
        help="Optional L2 ratio cap for gated object residual vs x_before_object inside Wan blocks.",
    )
    parser.add_argument(
        "--object-branch-residual-scale",
        type=float,
        default=1.0,
        help="Inference-only multiplier applied to the gated object residual before the ratio guard.",
    )
    parser.add_argument(
        "--object-branch-ratio-guard-max-block-id",
        type=int,
        default=None,
        help="Apply the optional object-branch ratio guard only up to this Wan block id.",
    )
    parser.add_argument(
        "--object-branch-auto-fallback-max-active-slots",
        type=int,
        default=None,
        help="If set, retry with the first N ranked slots when the guard repeatedly triggers.",
    )
    parser.add_argument(
        "--object-branch-auto-fallback-trigger-count",
        type=int,
        default=5,
        help="Abort the initial inference after this many guard triggers before retrying.",
    )
    parser.add_argument(
        "--dump-pipe-inputs-root",
        type=Path,
        default=None,
        help="Optional directory used to export the actual pipe conditioning inputs per sample.",
    )
    parser.add_argument(
        "--dump-numeric-trace-root",
        type=Path,
        default=None,
        help="Optional directory used to dump per-step latent/noise numeric stats per sample.",
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
        compact_object_context_slots=bool(cli_args.compact_object_context_slots),
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
        grounding_caption_prompt_mode=str(cli_args.grounding_caption_prompt_mode),
        grounding_caption_max_phrases=int(cli_args.grounding_caption_max_phrases),
        grounding_caption_min_score=float(cli_args.grounding_caption_min_score),
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


def _configure_oracle_environment(cli_args: argparse.Namespace) -> None:
    os.environ["XSSC_ROOT"] = str(cli_args.xssc_root.expanduser().resolve())
    os.environ["XSSC_CONFIG"] = str(cli_args.xssc_config.expanduser().resolve())
    os.environ["XSSC_CHECKPOINT"] = str(cli_args.xssc_checkpoint.expanduser().resolve())
    os.environ["XSSC_INPUT_SIZE"] = str(int(cli_args.xssc_input_size))
    os.environ["XSSC_MAX_TIME_STEPS"] = str(int(cli_args.xssc_max_time_steps))
    os.environ["OBJECT_LORA_RANK"] = str(int(cli_args.object_lora_rank))
    os.environ["OBJECT_LORA_ALPHA"] = str(float(cli_args.object_lora_alpha))
    os.environ["XSSC_ORACLE_VIDEO_FRAMES"] = str(int(cli_args.xssc_oracle_video_frames))
    os.environ["XSSC_VAE_TEMPORAL_STRIDE"] = str(int(cli_args.xssc_vae_temporal_stride))
    os.environ["XSSC_ORACLE_SAMPLING_MODE"] = str(cli_args.xssc_oracle_sampling_mode)
    os.environ["XSSC_ORACLE_VIDEO_RESIZE_MODE"] = str(cli_args.xssc_oracle_video_resize_mode)
    os.environ["XSSC_PREPROCESS_MODE"] = str(cli_args.xssc_preprocess_mode)


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
    negative_prompt: str | None,
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
    object_context, object_debug = oracle_infer._build_object_context(
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
        slot_count=int(getattr(model, "aux_max_objects", 0)),
        keep_slot_ids=getattr(model, "_object_context_keep_slot_ids", None),
        scale_factor=float(getattr(model, "_object_context_scale_factor", 1.0)),
        token_norm_max=getattr(model, "_object_context_token_norm_max", None),
    )
    object_debug["object_context_ablation"] = ablation_debug
    object_debug["object_context_stats"] = _tensor_numeric_stats(object_context)

    pipe = model.pipe
    numeric_trace_root = getattr(model, "_dump_numeric_trace_root", None)
    if numeric_trace_root is not None:
        pipe._numeric_trace_enabled = True
        pipe._numeric_trace_path = str(Path(numeric_trace_root) / f"{output_video.stem}_numeric_trace.json")
    else:
        pipe._numeric_trace_enabled = False
        pipe._numeric_trace_path = None
    pipe.dit.eval()
    fallback_debug: dict[str, object] = {
        "enabled": False,
        "triggered": False,
        "reason": None,
        "keep_slot_ids": None,
    }
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
            _dump_pipe_inputs(
                dump_root=Path(dump_pipe_inputs_root),
                sample_stem=output_video.stem,
                context_pil=context_pil,
                prompt=str(input_caption),
                negative_prompt=negative_prompt,
                pipe_kwargs=pipe_kwargs,
                source_video=str(source_video),
                frame_indices=frame_indices,
            )
        fallback_max_slots = getattr(
            model, "_object_branch_auto_fallback_max_active_slots", None
        )
        fallback_trigger_count = int(
            getattr(model, "_object_branch_auto_fallback_trigger_count", 5)
        )
        valid_object_count = int(round(float(object_debug.get("object_valid_count", 0.0))))
        fallback_enabled = (
            fallback_max_slots is not None
            and int(fallback_max_slots) > 0
            and valid_object_count > int(fallback_max_slots)
            and str(getattr(model, "_object_context_ablation_mode", "none")).strip().lower()
            in {"", "none"}
        )
        fallback_debug["enabled"] = bool(fallback_enabled)
        fallback_debug["max_active_slots"] = (
            None if fallback_max_slots is None else int(fallback_max_slots)
        )
        fallback_debug["trigger_count"] = int(fallback_trigger_count)
        pipe.dit._object_branch_guard_abort_count = 0
        pipe.dit._object_branch_guard_abort_after_count = (
            int(fallback_trigger_count) if fallback_enabled else None
        )
        try:
            video = pipe(**pipe_kwargs)
        except ObjectBranchInstabilityError as exc:
            keep_slot_ids = list(range(int(fallback_max_slots)))
            fallback_context, fallback_ablation = infer0705._apply_object_context_ablation(
                object_context,
                mode="keep_slot",
                slot_count=int(getattr(model, "aux_max_objects", 0)),
                keep_slot_ids=keep_slot_ids,
            )
            fallback_debug.update(
                {
                    "triggered": True,
                    "reason": str(exc),
                    "keep_slot_ids": keep_slot_ids,
                    "ablation": fallback_ablation,
                }
            )
            logs.append(
                "[object-fallback] "
                f"reason={exc} keep_slot_ids={keep_slot_ids}"
            )
            pipe.dit._object_branch_guard_abort_after_count = None
            pipe.dit._object_branch_guard_abort_count = 0
            pipe_kwargs["object_context"] = fallback_context
            video = pipe(**pipe_kwargs)
        finally:
            pipe.dit._object_branch_guard_abort_after_count = None
            pipe.dit._object_branch_guard_abort_count = 0

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
        "object_branch_ratio_guard": {
            "max_ratio": getattr(model.pipe.dit, "_object_branch_ratio_guard_max_ratio", None),
            "max_block_id": getattr(model.pipe.dit, "_object_branch_ratio_guard_max_block_id", None),
        },
        "object_branch_auto_fallback": fallback_debug,
        "object_debug": object_debug,
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
        },
        "vjepa": vjepa_summary,
    }
    return result, logs


def main() -> None:
    cli_args = parse_args()
    cli_args._negative_prompt_provided = _cli_flag_present(sys.argv, ("--negative-prompt",))
    if cli_args.output_num_frames is not None:
        cli_args.num_frames = int(cli_args.output_num_frames)
    if int(cli_args.context_frames) != oracle_infer.train.XSSC_NUM_CONTEXT_FRAMES:
        raise ValueError(
            "Scheme A oracle xSSC inference expects "
            f"--context-frames={oracle_infer.train.XSSC_NUM_CONTEXT_FRAMES}, "
            f"got {cli_args.context_frames}"
        )
    if int(cli_args.num_frames) != int(cli_args.xssc_oracle_video_frames):
        raise ValueError(
            "--num-frames must match --xssc-oracle-video-frames for Scheme A oracle "
            f"inference, got {cli_args.num_frames} vs {cli_args.xssc_oracle_video_frames}"
        )
    if (int(cli_args.xssc_oracle_video_frames) - 1) % int(cli_args.xssc_vae_temporal_stride) != 0:
        raise ValueError(
            "--xssc-oracle-video-frames must satisfy 1 + n * --xssc-vae-temporal-stride"
        )
    _configure_oracle_environment(cli_args)

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
    shard_tag = _normalize_shard_tag(cli_args.shard_tag)

    output_root.mkdir(parents=True, exist_ok=True)
    resolved_negative_prompt = _resolve_negative_prompt_from_cli(cli_args)
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
        "negative_prompt": resolved_negative_prompt,
        "initialize_model_on_cpu": bool(cli_args.initialize_model_on_cpu),
        "disable_object_branch": bool(cli_args.disable_object_branch),
        "device": str(cli_args.device),
        "aux_device": cli_args.aux_device,
        "inference_devices": cli_args.inference_devices,
        "shard_tag": shard_tag,
        "object_context_ablation": {
            "mode": str(cli_args.object_context_ablation),
            "random_seed": cli_args.object_context_random_seed,
            "random_scale": float(cli_args.object_context_random_scale),
            "scale_factor": float(cli_args.object_context_scale_factor),
            "token_norm_max": cli_args.object_context_token_norm_max,
            "keep_slot_ids": None
            if cli_args.object_context_keep_slot_ids is None
            else [int(part.strip()) for part in str(cli_args.object_context_keep_slot_ids).split(",") if part.strip()],
        },
        "object_adapter_mlp_residual_max_ratio": cli_args.object_adapter_mlp_residual_max_ratio,
        "compact_object_context_slots": bool(cli_args.compact_object_context_slots),
        "object_branch_ratio_guard": {
            "max_ratio": cli_args.object_branch_ratio_guard_max_ratio,
            "max_block_id": cli_args.object_branch_ratio_guard_max_block_id,
        },
        "object_branch_auto_fallback": {
            "max_active_slots": cli_args.object_branch_auto_fallback_max_active_slots,
            "trigger_count": int(cli_args.object_branch_auto_fallback_trigger_count),
        },
        "vjepa": infer0705.summarize_vjepa_args(cli_args),
        "scheme_a_xssc_oracle": {
            "xssc_root": str(cli_args.xssc_root),
            "xssc_config": str(cli_args.xssc_config),
            "xssc_checkpoint": str(cli_args.xssc_checkpoint),
            "xssc_input_size": int(cli_args.xssc_input_size),
            "xssc_max_time_steps": int(cli_args.xssc_max_time_steps),
            "xssc_oracle_video_frames": int(cli_args.xssc_oracle_video_frames),
            "xssc_vae_temporal_stride": int(cli_args.xssc_vae_temporal_stride),
            "latent_time_steps": 1
            + (int(cli_args.xssc_oracle_video_frames) - 1)
            // int(cli_args.xssc_vae_temporal_stride),
            "slots_per_frame": 7,
            "object_token_count": 7
            * (
                1
                + (int(cli_args.xssc_oracle_video_frames) - 1)
                // int(cli_args.xssc_vae_temporal_stride)
            ),
            "oracle_sampling_mode": str(cli_args.xssc_oracle_sampling_mode),
            "oracle_video_resize_mode": str(cli_args.xssc_oracle_video_resize_mode),
            "xssc_preprocess_mode": str(cli_args.xssc_preprocess_mode),
            "object_lora_rank": int(cli_args.object_lora_rank),
            "object_lora_alpha": float(cli_args.object_lora_alpha),
            "uses_future_slots": True,
        },
    }
    manifest_name = "batch_manifest.json" if shard_tag is None else f"batch_manifest_{shard_tag}.json"
    with (output_root / manifest_name).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    method_name = _append_method_suffix(
        _build_method_name_from_checkpoint_dir(
            weights_root,
            context_frames=int(cli_args.context_frames),
            num_frames=int(cli_args.num_frames),
            sampling_steps=int(cli_args.num_inference_steps),
            height=int(cli_args.height),
            width=int(cli_args.width),
            negative_prompt=resolved_negative_prompt,
        ),
        cli_args.method_suffix,
    )
    step_output_dir_name = _resolve_step_output_dir_name(
        cli_args.step_output_dir_name,
        checkpoint_dir=weights_root,
        method_name=method_name,
    )
    step_output_dir = output_root / step_output_dir_name
    step_output_dir.mkdir(parents=True, exist_ok=True)

    runtime_args = _build_runtime_args(cli_args, weights_root, step_output_dir)
    model, _, load_info = oracle_infer._build_runtime_model(runtime_args)
    model.to(torch.device(cli_args.device))
    model.eval()
    model.pipe.dit.eval()
    model._object_context_ablation_mode = str(cli_args.object_context_ablation)
    model._object_context_random_seed = cli_args.object_context_random_seed
    model._object_context_random_scale = float(cli_args.object_context_random_scale)
    model._object_context_scale_factor = float(cli_args.object_context_scale_factor)
    model._object_context_token_norm_max = cli_args.object_context_token_norm_max
    model.compact_object_context_slots = bool(cli_args.compact_object_context_slots)
    model._object_context_keep_slot_ids = (
        None
        if cli_args.object_context_keep_slot_ids is None
        else [int(part.strip()) for part in str(cli_args.object_context_keep_slot_ids).split(",") if part.strip()]
    )
    model._object_branch_auto_fallback_max_active_slots = (
        None
        if cli_args.object_branch_auto_fallback_max_active_slots is None
        else int(cli_args.object_branch_auto_fallback_max_active_slots)
    )
    model._object_branch_auto_fallback_trigger_count = int(
        cli_args.object_branch_auto_fallback_trigger_count
    )
    model._dump_pipe_inputs_root = (
        None if cli_args.dump_pipe_inputs_root is None else str(cli_args.dump_pipe_inputs_root)
    )
    model._dump_numeric_trace_root = (
        None if cli_args.dump_numeric_trace_root is None else str(cli_args.dump_numeric_trace_root)
    )
    model.pipe.dit._object_branch_ratio_guard_max_ratio = (
        None
        if cli_args.object_branch_ratio_guard_max_ratio is None
        else float(cli_args.object_branch_ratio_guard_max_ratio)
    )
    model.pipe.dit._object_branch_residual_scale = float(cli_args.object_branch_residual_scale)
    model.pipe.dit._object_branch_ratio_guard_max_block_id = (
        None
        if cli_args.object_branch_ratio_guard_max_block_id is None
        else int(cli_args.object_branch_ratio_guard_max_block_id)
    )
    model.object_adapter.mlp_residual_max_ratio = (
        None
        if cli_args.object_adapter_mlp_residual_max_ratio is None
        or float(cli_args.object_adapter_mlp_residual_max_ratio) <= 0.0
        else float(cli_args.object_adapter_mlp_residual_max_ratio)
    )

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
                negative_prompt=resolved_negative_prompt,
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
        "shard_tag": shard_tag,
        "num_total": len(json_paths),
        "num_success": step_success,
        "num_failed": step_failed,
        "num_skipped": step_skipped,
        "entries": step_entries,
    }
    step_result_name = "result.json" if shard_tag is None else f"result_{shard_tag}.json"
    with (step_output_dir / step_result_name).open("w", encoding="utf-8") as handle:
        json.dump(step_summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    summary = {
        "weights_root": str(weights_root),
        "output_root": str(output_root),
        "step": weights_root.name,
        "shard_tag": shard_tag,
        "num_total": len(json_paths),
        "num_success": step_success,
        "num_failed": step_failed,
        "num_skipped": step_skipped,
    }
    summary_name = "summary.json" if shard_tag is None else f"summary_{shard_tag}.json"
    with (output_root / summary_name).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
