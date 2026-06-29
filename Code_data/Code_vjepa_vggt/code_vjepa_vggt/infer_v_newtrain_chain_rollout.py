from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from decord import VideoReader, cpu

from code_vjepa_vggt.batch_infer_v_newtrain_from_jsonl import (
    _apply_config_defaults,
    _build_model_args,
)
from code_vjepa_vggt.infer_v_newtrain_context_video_wan import (
    _build_object_context,
    _load_context_video,
    _load_v_newtrain_state_into_model,
    _resolve_checkpoint_file,
    _resolve_launch_device,
    _tensor_video_to_pil_list,
    build_model,
)
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8
from diffsynth.utils.data import save_video


def _load_input_json(json_path: Path) -> dict[str, object]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"input json must be an object: {json_path}")
    return payload


def _resolve_prompt_and_video(
    *,
    input_json: Path | None,
    context_video: str | None,
    prompt: str | None,
) -> tuple[Path, str, dict[str, object] | None]:
    if input_json is not None:
        payload = _load_input_json(input_json)
        payload_prompt = payload.get("input_caption")
        if not isinstance(payload_prompt, str) or not payload_prompt.strip():
            raise ValueError(f"missing or empty 'input_caption' in {input_json}")
        payload_video = payload.get("input_video")
        if not isinstance(payload_video, str) or not payload_video.strip():
            raise ValueError(f"missing or empty 'input_video' in {input_json}")
        return Path(payload_video).expanduser().resolve(), payload_prompt.strip(), payload

    if context_video is None or prompt is None:
        raise ValueError("provide either --input-json or both --context-video and --prompt")
    return Path(context_video).expanduser().resolve(), str(prompt), None


def _read_video_frames(video_path: Path) -> np.ndarray:
    vr = VideoReader(str(video_path), ctx=cpu(0))
    if len(vr) <= 0:
        raise RuntimeError(f"video has no frames: {video_path}")
    frame_idx = np.arange(len(vr), dtype=np.int64)
    return vr.get_batch(frame_idx).asnumpy()


def _read_video_tail(video_path: Path, tail_frames: int) -> tuple[np.ndarray, np.ndarray]:
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)
    if total_frames <= 0:
        raise RuntimeError(f"video has no frames: {video_path}")
    if tail_frames <= 0:
        raise ValueError(f"tail_frames must be positive, got {tail_frames}")
    start = max(0, total_frames - int(tail_frames))
    frame_idx = np.arange(start, total_frames, dtype=np.int64)
    frames = vr.get_batch(frame_idx).asnumpy()
    if int(frames.shape[0]) != int(tail_frames):
        raise RuntimeError(
            f"video {video_path} only has {int(frames.shape[0])} tail frames, expected {int(tail_frames)}"
        )
    return frames, frame_idx


def _infer_segment(
    *,
    model,
    context_frames_rgb: np.ndarray,
    prompt: str,
    seed: int,
    sampling_steps: int,
    cfg_scale: float,
    num_frames: int,
    height: int,
    width: int,
) -> tuple[object, dict[str, object]]:
    context_video_single = preprocess_video_rgb_uint8(context_frames_rgb, (int(height), int(width)))
    context_pil = _tensor_video_to_pil_list(context_video_single)
    object_context, object_debug = _build_object_context(
        model=model,
        context_video_single=context_video_single,
    )

    pipe = model.pipe
    pipe.dit.eval()
    with torch.no_grad():
        video = pipe(
            prompt=prompt,
            negative_prompt="",
            context_video=context_pil,
            seed=int(seed),
            tiled=True,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            num_inference_steps=int(sampling_steps),
            cfg_scale=float(cfg_scale),
            object_context=object_context,
        )
    return video, object_debug


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _build_case_stem(input_json: Path | None, context_video: Path) -> str:
    if input_json is not None:
        return input_json.stem
    return context_video.stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run chained v_newtrain rollout: use 8-frame context to generate 24 frames, "
            "reuse the generated tail-8 as context, generate another 24 frames, and save both segments plus a merged video."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-json", default=None, help="input case json containing input_video and input_caption")
    parser.add_argument("--context-video", default=None, help="used when --input-json is not provided")
    parser.add_argument("--prompt", default=None, help="used when --input-json is not provided")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wan-root", default="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--disable-object-branch", action="store_true")
    parser.add_argument("--object-num-queries", type=int, default=8)
    parser.add_argument("--aux-max-objects", type=int, default=4)
    parser.add_argument("--jepa-ckpt-path", default="/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth")
    parser.add_argument("--jepa-input-size", type=int, default=384)
    parser.add_argument("--jepa-patch-size", type=int, default=16)
    parser.add_argument("--jepa-tubelet-size", type=int, default=2)
    parser.add_argument("--cotracker-checkpoint", default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")
    parser.add_argument("--cotracker-input-h", type=int, default=384)
    parser.add_argument("--cotracker-input-w", type=int, default=512)
    parser.add_argument("--cotracker-window-len", type=int, default=60)
    parser.add_argument("--object-pooler-latent-dim", type=int, default=16)
    parser.add_argument("--cond-proj-dim", type=int, default=4096)
    parser.add_argument("--jepa-window-radius", type=int, default=1)
    parser.add_argument("--latent-window-radius", type=int, default=1)
    args = parser.parse_args()

    input_json = Path(args.input_json).expanduser().resolve() if args.input_json is not None else None
    config = load_yaml_config(args.config)
    _apply_config_defaults(args, parser, config)

    initial_context_video, prompt_text, source_payload = _resolve_prompt_and_video(
        input_json=input_json,
        context_video=args.context_video,
        prompt=args.prompt,
    )
    if not initial_context_video.is_file():
        raise FileNotFoundError(f"context video not found: {initial_context_video}")

    args.device = _resolve_launch_device()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    model_args: SimpleNamespace = _build_model_args(args)
    model = build_model(model_args)
    model.to(torch.device(args.device))
    model.eval()
    load_info = _load_v_newtrain_state_into_model(model, Path(args.checkpoint))

    initial_frames_rgb, initial_frame_indices = _load_context_video(
        video_path=initial_context_video,
        target_context_frames=int(args.context_frames),
        sampling_mode=args.sampling_mode,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    case_stem = _build_case_stem(input_json, initial_context_video)
    segment1_path = output_dir / f"{case_stem}__segment1.mp4"
    segment2_path = output_dir / f"{case_stem}__segment2.mp4"
    merged_path = output_dir / f"{case_stem}__merged.mp4"
    result_path = output_dir / f"{case_stem}__chain_rollout.json"

    segment1_video, segment1_debug = _infer_segment(
        model=model,
        context_frames_rgb=initial_frames_rgb,
        prompt=prompt_text,
        seed=int(args.seed),
        sampling_steps=int(args.sampling_steps),
        cfg_scale=float(args.cfg_scale),
        num_frames=int(args.num_frames),
        height=int(args.height),
        width=int(args.width),
    )
    save_video(segment1_video, str(segment1_path), fps=int(args.fps), quality=int(args.quality))

    chained_context_frames_rgb, chained_frame_indices = _read_video_tail(segment1_path, int(args.context_frames))
    segment2_video, segment2_debug = _infer_segment(
        model=model,
        context_frames_rgb=chained_context_frames_rgb,
        prompt=prompt_text,
        seed=int(args.seed) + 1,
        sampling_steps=int(args.sampling_steps),
        cfg_scale=float(args.cfg_scale),
        num_frames=int(args.num_frames),
        height=int(args.height),
        width=int(args.width),
    )
    save_video(segment2_video, str(segment2_path), fps=int(args.fps), quality=int(args.quality))

    segment1_saved_frames = _read_video_frames(segment1_path)
    segment2_saved_frames = _read_video_frames(segment2_path)
    merged_frames = np.concatenate([segment1_saved_frames, segment2_saved_frames], axis=0)
    save_video(merged_frames, str(merged_path), fps=int(args.fps), quality=int(args.quality))

    result = {
        "checkpoint": str(_resolve_checkpoint_file(Path(args.checkpoint))),
        "config": str(Path(args.config).expanduser().resolve()),
        "input_json": str(input_json) if input_json is not None else None,
        "source_payload": source_payload,
        "prompt": prompt_text,
        "initial_context_video": str(initial_context_video),
        "segment1_video": str(segment1_path),
        "segment2_video": str(segment2_path),
        "merged_video": str(merged_path),
        "sampling_mode": str(args.sampling_mode),
        "context_frames": int(args.context_frames),
        "segment_frames": int(args.num_frames),
        "merged_frames": int(merged_frames.shape[0]),
        "fps": int(args.fps),
        "seed_segment1": int(args.seed),
        "seed_segment2": int(args.seed) + 1,
        "initial_frame_indices": initial_frame_indices.tolist(),
        "segment1_tail_indices_for_segment2": chained_frame_indices.tolist(),
        "load_info": load_info,
        "segment1_object_debug": segment1_debug,
        "segment2_object_debug": segment2_debug,
    }
    _save_json(result_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
