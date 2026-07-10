from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from code_vjepa_vggt import batch_infer_v_newtrain_from_jsonl as core
from code_vjepa_vggt.train0705 import infer_stage1b_context_only_no_gt_box_v_newtrain0705 as infer0705
from code_vjepa_vggt.train0705.tmp_cross_object_context_between_ckpts import (
    DEFAULT_BASE_LORA,
    DEFAULT_DIFFSYNTH_ROOT,
    DEFAULT_STAGE1A,
    DEFAULT_WAN_ROOT,
    _build_object_context_for_checkpoint,
    _make_runtime_args,
    _resolve_runtime_device,
    _tensor_stats,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box.wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v import (
    DEFAULT_NEGATIVE_PROMPT,
)
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, read_video_prefix, read_video_uniform
from diffsynth.utils.data import save_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single object-branch diagnosis case with optional none/zero/random/cross "
            "object_context source selection and Wan numeric trace dump."
        )
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint used for Wan denoising.")
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--diffsynth-root", type=Path, default=DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument("--lora-checkpoint", type=Path, default=DEFAULT_BASE_LORA)
    parser.add_argument("--stage1a-init-from", type=Path, default=DEFAULT_STAGE1A)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--input-cover-crop-height", type=int, default=480)
    parser.add_argument("--input-cover-crop-width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=5)
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
    parser.add_argument(
        "--variant",
        choices=["none", "zero", "random", "cross"],
        required=True,
        help="Object-context variant to feed into the denoiser.",
    )
    parser.add_argument(
        "--cross-object-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint used to build external object_context when variant=cross.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt passed into Wan inference.",
    )
    parser.add_argument(
        "--random-object-seed",
        type=int,
        default=123,
        help="Random seed used when variant=random.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save video, trace, and per-run summary.",
    )
    parser.add_argument(
        "--trace-json-name",
        type=str,
        default="numeric_trace.json",
        help="Filename for the raw numeric trace JSON.",
    )
    infer0705.add_vjepa_cli_args(parser)
    args = parser.parse_args()
    if str(args.variant) == "cross" and args.cross_object_checkpoint is None:
        parser.error("--cross-object-checkpoint is required when --variant=cross")
    return args


def _resolve_source_video(payload: dict[str, object], json_path: Path) -> str:
    source_video = payload.get("source_video")
    if isinstance(source_video, str) and source_video.strip():
        return source_video.strip()
    return core._resolve_input_video(payload, json_path)


def _gate_summary(model) -> dict[str, object]:
    pipe = model.pipe
    blocks = getattr(pipe.dit, "blocks", [])
    per_layer: list[dict[str, object]] = []
    gate_abs_means: list[float] = []
    gate_abs_maxes: list[float] = []
    for block_id, block in enumerate(blocks):
        object_gate = getattr(block, "object_gate", None)
        if object_gate is None:
            continue
        raw = object_gate.detach().float()
        tanh = torch.tanh(raw)
        abs_mean = float(tanh.abs().mean().item())
        abs_max = float(tanh.abs().max().item())
        gate_abs_means.append(abs_mean)
        gate_abs_maxes.append(abs_max)
        per_layer.append(
            {
                "block_id": int(block_id),
                "raw_mean": float(raw.mean().item()),
                "raw_abs_mean": float(raw.abs().mean().item()),
                "raw_abs_max": float(raw.abs().max().item()),
                "tanh_mean": float(tanh.mean().item()),
                "tanh_abs_mean": abs_mean,
                "tanh_abs_max": abs_max,
            }
        )
    return {
        "num_layers": int(len(per_layer)),
        "layers": per_layer,
        "tanh_abs_mean_global_mean": float(np.mean(gate_abs_means)) if gate_abs_means else None,
        "tanh_abs_mean_global_max": float(np.max(gate_abs_maxes)) if gate_abs_maxes else None,
    }


def _summarize_numeric_trace(trace_path: Path) -> dict[str, object]:
    if not trace_path.exists():
        return {"exists": False}
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    object_steps = [entry for entry in payload if entry.get("kind") == "object_branch_step"]
    layers: dict[int, list[dict[str, object]]] = {}
    for step_entry in object_steps:
        for layer in step_entry.get("layers", []):
            block_id = int(layer.get("block_id", -1))
            layers.setdefault(block_id, []).append(layer)

    def _mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    def _max(values: list[float]) -> float | None:
        return float(np.max(values)) if values else None

    per_layer = []
    all_ratio: list[float] = []
    all_object_delta_l2: list[float] = []
    all_gated_delta_l2: list[float] = []
    for block_id in sorted(layers):
        entries = layers[block_id]
        ratios = [float(entry.get("gated_to_x_ratio_l2", 0.0)) for entry in entries]
        raw_delta = [
            float(((entry.get("object_delta") or {}).get("l2")) or 0.0)
            for entry in entries
        ]
        gated_delta = [
            float(((entry.get("gated_object_delta") or {}).get("l2")) or 0.0)
            for entry in entries
        ]
        x_l2 = [
            float(((entry.get("x_before_object") or {}).get("l2")) or 0.0)
            for entry in entries
        ]
        gate_abs_mean = [
            float((((entry.get("object_gate_tanh") or {}).get("abs_max")) or 0.0))
            for entry in entries
        ]
        all_ratio.extend(ratios)
        all_object_delta_l2.extend(raw_delta)
        all_gated_delta_l2.extend(gated_delta)
        per_layer.append(
            {
                "block_id": int(block_id),
                "num_steps": int(len(entries)),
                "mean_gated_to_x_ratio_l2": _mean(ratios),
                "max_gated_to_x_ratio_l2": _max(ratios),
                "mean_object_delta_l2": _mean(raw_delta),
                "mean_gated_object_delta_l2": _mean(gated_delta),
                "mean_x_before_object_l2": _mean(x_l2),
                "mean_gate_tanh_abs_max": _mean(gate_abs_mean),
            }
        )
    return {
        "exists": True,
        "num_entries": int(len(payload)),
        "num_object_branch_steps": int(len(object_steps)),
        "overall": {
            "mean_gated_to_x_ratio_l2": _mean(all_ratio),
            "max_gated_to_x_ratio_l2": _max(all_ratio),
            "mean_object_delta_l2": _mean(all_object_delta_l2),
            "mean_gated_object_delta_l2": _mean(all_gated_delta_l2),
        },
        "layers": per_layer,
    }


def _read_video_rgb(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"failed to decode video: {path}")
    return np.stack(frames).astype(np.float32) / 255.0


def _read_video_resized_rgb(path: Path, width: int, height: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"failed to decode video: {path}")
    return np.stack(frames).astype(np.float32) / 255.0


def _mse_summary(pred: np.ndarray, ref: np.ndarray) -> dict[str, object]:
    n = min(int(pred.shape[0]), int(ref.shape[0]))
    diff = pred[:n] - ref[:n]
    mse_pf = np.mean(diff * diff, axis=(1, 2, 3))
    return {
        "num_frames": int(n),
        "mean": float(mse_pf.mean()),
        "first8": float(mse_pf[: min(8, n)].mean()),
        "last8": float(mse_pf[max(0, n - 8) :].mean()),
        "max_frame_idx": int(np.argmax(mse_pf)),
        "max_frame_mse": float(np.max(mse_pf)),
    }


def _prepare_object_context(
    *,
    args: argparse.Namespace,
    denoise_checkpoint_dir: Path,
    context_video_single: torch.Tensor,
    prompt: str,
    video_path: Path,
) -> tuple[torch.Tensor, dict[str, object]]:
    if str(args.variant) == "cross":
        source_checkpoint = Path(args.cross_object_checkpoint).expanduser().resolve()
        object_context_cpu, source_debug = _build_object_context_for_checkpoint(
            args=args,
            checkpoint_dir=source_checkpoint,
            context_video_single=context_video_single,
            prompt=prompt,
            video_path=video_path,
        )
        return object_context_cpu, {
            "variant": "cross",
            "object_context_source_checkpoint": str(source_checkpoint),
            "source_object_context": source_debug,
            "ablation_debug": {
                "mode": "cross",
                "applied": True,
            },
        }

    object_context_cpu, source_debug = _build_object_context_for_checkpoint(
        args=args,
        checkpoint_dir=denoise_checkpoint_dir,
        context_video_single=context_video_single,
        prompt=prompt,
        video_path=video_path,
    )
    if str(args.variant) == "none":
        return object_context_cpu, {
            "variant": "none",
            "object_context_source_checkpoint": str(denoise_checkpoint_dir),
            "source_object_context": source_debug,
            "ablation_debug": {
                "mode": "none",
                "applied": False,
            },
        }

    object_context = object_context_cpu.to(dtype=torch.float32)
    ablated, ablation_debug = infer0705._apply_object_context_ablation(
        object_context,
        mode=str(args.variant),
        random_seed=int(args.random_object_seed),
        random_scale=1.0,
    )
    if ablated is None:
        raise RuntimeError(f"ablation produced None object_context for variant={args.variant}")
    return ablated.detach().cpu(), {
        "variant": str(args.variant),
        "object_context_source_checkpoint": str(denoise_checkpoint_dir),
        "source_object_context": source_debug,
        "ablation_debug": ablation_debug,
    }


def _load_context_from_source_video(args: argparse.Namespace, source_video: Path) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    if str(args.sampling_mode) == "uniform":
        frames, frame_indices = read_video_uniform(source_video, int(args.context_frames))
    else:
        frames, frame_indices = read_video_prefix(source_video, int(args.context_frames))
    if int(frames.shape[0]) > int(args.context_frames):
        frames = frames[: int(args.context_frames)]
        frame_indices = frame_indices[: int(args.context_frames)]
    context_video_single = preprocess_video_rgb_uint8(
        frames,
        (int(args.height), int(args.width)),
        resize_mode="cover_crop",
        cover_crop_hw=(int(args.input_cover_crop_height), int(args.input_cover_crop_width)),
    )
    return frames, frame_indices, context_video_single


def main() -> None:
    args = parse_args()
    infer0705.apply_vjepa_preset_if_requested(args)
    args.device = _resolve_runtime_device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    checkpoint_dir = Path(args.checkpoint).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_root = output_dir

    payload = core._load_input_json(args.input_json)
    source_video = Path(_resolve_source_video(payload, args.input_json)).expanduser().resolve()
    input_caption = core._ensure_str_field(payload, "input_caption", args.input_json)
    _, frame_indices, context_video_single = _load_context_from_source_video(args, source_video)

    object_context_cpu, object_context_info = _prepare_object_context(
        args=args,
        denoise_checkpoint_dir=checkpoint_dir,
        context_video_single=context_video_single,
        prompt=input_caption,
        video_path=source_video,
    )

    runtime_args = _make_runtime_args(args, checkpoint_dir, output_dir)
    model, model_args, load_info = infer0705._build_runtime_model(runtime_args)
    try:
        pipe = model.pipe
        pipe.dit.eval()
        pipe._numeric_trace_enabled = True
        trace_path = output_dir / str(args.trace_json_name)
        pipe._numeric_trace_path = str(trace_path)

        context_pil = infer0705._tensor_video_to_pil_list(context_video_single)
        object_context = object_context_cpu.to(device=pipe.device, dtype=pipe.torch_dtype)
        with torch.no_grad():
            video = pipe(
                prompt=str(input_caption),
                negative_prompt=str(args.negative_prompt),
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

        sample_stem = args.input_json.stem
        output_video = output_dir / f"{sample_stem}_{checkpoint_dir.name}_{args.variant}.mp4"
        save_video(video, str(output_video), fps=int(args.fps), quality=int(args.quality))

        gate_summary = _gate_summary(model)
        trace_summary = _summarize_numeric_trace(trace_path)

        pred_video = _read_video_rgb(output_video)
        source_video_resized = _read_video_resized_rgb(source_video, pred_video.shape[2], pred_video.shape[1])
        source_mse = _mse_summary(pred_video, source_video_resized)

        result = {
            "input_json": str(args.input_json),
            "source_video": str(source_video),
            "input_caption": str(input_caption),
            "output_video": str(output_video),
            "trace_json": str(trace_path),
            "checkpoint": str(checkpoint_dir),
            "variant": str(args.variant),
            "cross_object_checkpoint": None
            if args.cross_object_checkpoint is None
            else str(Path(args.cross_object_checkpoint).expanduser().resolve()),
            "negative_prompt": str(args.negative_prompt),
            "seed": int(args.seed),
            "sampling_steps": int(args.sampling_steps),
            "cfg_scale": float(args.cfg_scale),
            "frame_indices": frame_indices.tolist(),
            "load_info": infer0705._summarize_load_info(load_info),
            "object_context_info": object_context_info,
            "external_object_context_stats": _tensor_stats(object_context_cpu),
            "gate_summary": gate_summary,
            "trace_summary": trace_summary,
            "source_mse": source_mse,
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
        result_path = output_dir / "result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "result": str(result_path),
                    "output_video": str(output_video),
                    "trace_json": str(trace_path),
                    "trace_summary": trace_summary.get("overall", {}),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
