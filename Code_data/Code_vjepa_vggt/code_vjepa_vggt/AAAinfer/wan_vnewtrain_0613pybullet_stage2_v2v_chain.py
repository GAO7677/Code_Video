from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import torch

from code_vjepa_vggt import batch_infer_v_newtrain_from_jsonl as core
from code_vjepa_vggt.AAAinfer.utils.named_paths import resolve_output_root
from code_vjepa_vggt.infer_v_newtrain_chain_rollout import (
    _infer_segment,
    _read_video_frames,
    _read_video_tail,
    _resolve_prompt_and_video,
    _save_json,
    _save_rgb_frames_video,
)
from code_vjepa_vggt.infer_v_newtrain_context_video_wan import (
    _load_context_video,
    _load_v_newtrain_state_into_model,
    _resolve_checkpoint_file,
    _resolve_launch_device,
    build_model,
)
from code_vjepa_vggt.utils.config import load_yaml_config
from diffsynth.utils.data import save_video

"""
Run command example:

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=0 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_vnewtrain_0613pybullet_stage2_v2v_chain.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun/checkpoints/step-007000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun_chain

Default output root:
- /data/gaoya/AAA_test_video/0623/test/v2v/pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun_chain
"""

DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")


def _normalize_ckpt_method_name(name: str) -> str:
    normalized = re.sub(r"^[A-Za-z]+\d+_", "", name, count=1)
    return normalized or name


def _build_method_name_from_checkpoint_dir(checkpoint_dir: Path) -> str:
    step_name = checkpoint_dir.name
    checkpoint_parent = checkpoint_dir.parent
    if checkpoint_parent.name == "checkpoints" and checkpoint_parent.parent.name:
        method_root = _normalize_ckpt_method_name(checkpoint_parent.parent.name)
        return f"{method_root}_{step_name}"
    if checkpoint_parent.name:
        method_root = _normalize_ckpt_method_name(checkpoint_parent.name)
        return f"{method_root}_{step_name}"
    return step_name


def _run_chain_case(
    *,
    model,
    checkpoint_dir: Path,
    input_json_path: Path,
    output_dir: Path,
    num_frames: int,
    context_frames: int,
    sampling_mode: str,
    sampling_steps: int,
    fps: int,
    seed: int,
    cfg_scale: float,
    height: int,
    width: int,
    quality: int,
) -> dict[str, object]:
    initial_context_video, prompt_text, source_payload = _resolve_prompt_and_video(
        input_json=input_json_path,
        context_video=None,
        prompt=None,
    )
    if not initial_context_video.is_file():
        raise FileNotFoundError(f"context video not found: {initial_context_video}")

    initial_frames_rgb, initial_frame_indices = _load_context_video(
        video_path=initial_context_video,
        target_context_frames=int(context_frames),
        sampling_mode=sampling_mode,
    )

    case_stem = input_json_path.stem
    segment1_path = output_dir / f"{case_stem}__segment1.mp4"
    segment2_path = output_dir / f"{case_stem}__segment2.mp4"
    merged_path = output_dir / f"{case_stem}__merged.mp4"
    segment1_context_path = output_dir / f"{case_stem}__segment1_context.mp4"
    segment2_context_path = output_dir / f"{case_stem}__segment2_context.mp4"
    result_path = output_dir / f"{case_stem}.json"

    _save_rgb_frames_video(
        frames_rgb=initial_frames_rgb,
        output_path=segment1_context_path,
        fps=int(fps),
        quality=int(quality),
    )

    segment1_video, segment1_debug = _infer_segment(
        model=model,
        context_frames_rgb=initial_frames_rgb,
        prompt=prompt_text,
        seed=int(seed),
        sampling_steps=int(sampling_steps),
        cfg_scale=float(cfg_scale),
        num_frames=int(num_frames),
        height=int(height),
        width=int(width),
    )
    save_video(segment1_video, str(segment1_path), fps=int(fps), quality=int(quality))

    chained_context_frames_rgb, chained_frame_indices = _read_video_tail(segment1_path, int(context_frames))
    _save_rgb_frames_video(
        frames_rgb=chained_context_frames_rgb,
        output_path=segment2_context_path,
        fps=int(fps),
        quality=int(quality),
    )

    segment2_video, segment2_debug = _infer_segment(
        model=model,
        context_frames_rgb=chained_context_frames_rgb,
        prompt=prompt_text,
        seed=int(seed) + 1,
        sampling_steps=int(sampling_steps),
        cfg_scale=float(cfg_scale),
        num_frames=int(num_frames),
        height=int(height),
        width=int(width),
    )
    save_video(segment2_video, str(segment2_path), fps=int(fps), quality=int(quality))

    segment1_saved_frames = _read_video_frames(segment1_path)
    segment2_saved_frames = _read_video_frames(segment2_path)
    merged_frames = np.concatenate([segment1_saved_frames, segment2_saved_frames], axis=0)
    save_video(merged_frames, str(merged_path), fps=int(fps), quality=int(quality))

    result: dict[str, object] = {
        "input_json": str(input_json_path),
        "input_caption": prompt_text,
        "output_video": str(merged_path),
        "seed": int(seed),
        "step": int(sampling_steps),
        "guidance": float(cfg_scale),
        "ckpt": str(_resolve_checkpoint_file(checkpoint_dir)),
        "context_video": str(initial_context_video),
        "segment1_context_video": str(segment1_context_path),
        "segment1_video": str(segment1_path),
        "segment2_context_video": str(segment2_context_path),
        "segment2_video": str(segment2_path),
        "merged_video": str(merged_path),
        "segment1_seed": int(seed),
        "segment2_seed": int(seed) + 1,
        "fps": int(fps),
        "num_frames": int(num_frames),
        "context_frames": int(context_frames),
        "sampling_mode": str(sampling_mode),
        "merged_frames": int(merged_frames.shape[0]),
        "initial_frame_indices": initial_frame_indices.tolist(),
        "segment1_tail_indices_for_segment2": chained_frame_indices.tolist(),
        "segment1_object_debug": segment1_debug,
        "segment2_object_debug": segment2_debug,
        "status": "generated",
    }
    if isinstance(source_payload, dict):
        source_video = source_payload.get("source_video")
        if isinstance(source_video, str) and source_video:
            result["source_video"] = source_video
        input_image = source_payload.get("input_image")
        if isinstance(input_image, str) and input_image:
            result["input_image"] = input_image

    _save_json(result_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run chained current v_newtrain/object-branch checkpoints over an input json list. "
            "Each case generates segment1, then reuses the generated tail-8 as segment2 context, "
            "and saves merged output plus context videos."
        )
    )
    parser.add_argument("--weights-root", type=Path, required=True, help="step-* dir containing checkpoint.safetensors")
    parser.add_argument("--input-json-list-path", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/"
            "train_0624pybullet_freeze_lora_other_modules_gpu67.yaml"
        ),
    )
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=32)
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    weights_root = cli_args.weights_root.expanduser().resolve()
    input_json_list_path = cli_args.input_json_list_path.expanduser().resolve()
    model_name = str(cli_args.model_name).strip()
    output_root = resolve_output_root(
        explicit_output_root=cli_args.output_root,
        base_output_root="/data/gaoya/AAA_test_video/0623/test/v2v",
        model_name=model_name,
    )

    config_path = cli_args.config.expanduser().resolve()
    config = load_yaml_config(config_path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--wan-root", default=str(DEFAULT_WAN_ROOT))
    parser.add_argument("--lora-rank", type=int, default=32)
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
    core._apply_config_defaults(cli_args, parser, config)

    cli_args.device = _resolve_launch_device()

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
        "num_frames": int(cli_args.num_frames),
        "context_frames": int(cli_args.context_frames),
        "sampling_mode": str(cli_args.sampling_mode),
        "chain_rollout": True,
    }
    with (output_root / "batch_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    model_args = core._build_model_args(cli_args)
    summary: dict[str, object] = {
        "input_json_list_path": str(input_json_list_path),
        "weights_root": str(weights_root),
        "output_root": str(output_root),
        "step": weights_root.name,
    }

    if not weights_root.exists():
        raise FileNotFoundError(f"weights-root not found: {weights_root}")

    step_output_dir = output_root / weights_root.name
    step_output_dir.mkdir(parents=True, exist_ok=True)
    method_name = _build_method_name_from_checkpoint_dir(weights_root)

    model = build_model(model_args)
    model.to(torch.device(cli_args.device))
    model.eval()
    load_info = _load_v_newtrain_state_into_model(model, weights_root)
    model.pipe.dit.eval()

    step_generated = 0
    step_failed = 0
    step_skipped = 0

    for input_json_path in json_paths:
        sample_stem = input_json_path.stem
        merged_video = step_output_dir / f"{sample_stem}__merged.mp4"
        output_json = step_output_dir / f"{sample_stem}.json"
        if merged_video.exists() and output_json.exists() and not (cli_args.force or cli_args.overwrite):
            print(f"[skip] {weights_root.name} {sample_stem}")
            step_skipped += 1
            continue

        try:
            result = _run_chain_case(
                model=model,
                checkpoint_dir=weights_root,
                input_json_path=input_json_path,
                output_dir=step_output_dir,
                num_frames=int(cli_args.num_frames),
                context_frames=int(cli_args.context_frames),
                sampling_mode=str(cli_args.sampling_mode),
                sampling_steps=int(cli_args.num_inference_steps),
                fps=int(cli_args.fps),
                seed=int(cli_args.seed),
                cfg_scale=float(cli_args.cfg_scale),
                height=int(cli_args.height),
                width=int(cli_args.width),
                quality=int(cli_args.quality),
            )
            print(f"[done] {weights_root.name} {sample_stem}")
            step_generated += 1
        except Exception as exc:
            failed_payload = {
                "input_json": str(input_json_path),
                "output_video": str(merged_video),
                "step": int(cli_args.num_inference_steps),
                "guidance": float(cli_args.cfg_scale),
                "seed": int(cli_args.seed),
                "ckpt": str(_resolve_checkpoint_file(weights_root)),
                "status": "failed",
                "error": repr(exc),
            }
            _save_json(output_json, failed_payload)
            print(f"[error] {weights_root.name} {sample_stem}: {exc}")
            step_failed += 1
            continue

        if isinstance(result, dict):
            result["method"] = method_name
            _save_json(output_json, result)

    summary["run"] = {
        "step": weights_root.name,
        "method_name": method_name,
        "checkpoint_dir": str(weights_root),
        "load_info": load_info,
        "generated": step_generated,
        "failed": step_failed,
        "skipped_existing": step_skipped,
        "output_dir": str(step_output_dir),
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _save_json(output_root / "summary.json", summary)


if __name__ == "__main__":
    main()
