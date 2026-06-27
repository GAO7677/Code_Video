from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re

import numpy as np
import torch

from code_vjepa_vggt import batch_infer_v_newtrain_from_jsonl as core
from code_vjepa_vggt.utils.config import load_yaml_config

"""
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=0 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_vnewtrain_0613pybullet_stage2_v2v.py \
  --checkpoint-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun/checkpoints \
  --steps step-006500 \
  --num-frames 81
"""

DEFAULT_INPUT_JSON_LIST_PATH = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
DEFAULT_CHECKPOINT_ROOT = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
    "pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun/checkpoints"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/AAA_test_video/0623/test/v2v/"
    "pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun"
)
DEFAULT_CONFIG_PATH = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/"
    "train_0624pybullet_freeze_lora_other_modules_gpu67.yaml"
)
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


def _list_step_dirs(checkpoint_root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in checkpoint_root.glob("step-*")
            if path.is_dir() and (path / "checkpoint.safetensors").is_file()
        ],
        key=lambda path: path.name,
    )


def _resolve_steps(checkpoint_root: Path, requested_steps: list[str], latest_only: bool) -> list[str]:
    if requested_steps:
        return [str(step).strip() for step in requested_steps if str(step).strip()]
    discovered = _list_step_dirs(checkpoint_root)
    if not discovered:
        raise FileNotFoundError(f"no step-* checkpoints found under {checkpoint_root}")
    if latest_only:
        return [discovered[-1].name]
    return [path.name for path in discovered]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run current v_newtrain/object-branch checkpoints over test_100.txt. "
            "This is a thin wrapper around batch_infer_v_newtrain_from_jsonl.py."
        )
    )
    parser.add_argument("--input-json-list-path", type=Path, default=DEFAULT_INPUT_JSON_LIST_PATH)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--steps", nargs="*", default=None, help="checkpoint step dirs such as step-006000")
    parser.add_argument("--latest-only", action="store_true", help="only run the latest available step-* checkpoint")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
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
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    checkpoint_root = cli_args.checkpoint_root.expanduser().resolve()
    input_json_list_path = cli_args.input_json_list_path.expanduser().resolve()
    output_root = cli_args.output_root.expanduser().resolve()

    if cli_args.config is not None:
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

    cli_args.device = core._resolve_launch_device()
    step_names = _resolve_steps(checkpoint_root, cli_args.steps or [], bool(cli_args.latest_only))

    torch.manual_seed(int(cli_args.seed))
    np.random.seed(int(cli_args.seed))

    json_paths = core._read_list_file(input_json_list_path)
    if cli_args.limit is not None:
        json_paths = json_paths[: max(0, int(cli_args.limit))]

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_json_list_path": str(input_json_list_path),
        "checkpoint_root": str(checkpoint_root),
        "steps": step_names,
        "num_items": len(json_paths),
        "num_inference_steps": int(cli_args.num_inference_steps),
        "cfg_scale": float(cli_args.cfg_scale),
        "seed": int(cli_args.seed),
        "height": int(cli_args.height),
        "width": int(cli_args.width),
        "num_frames": int(cli_args.num_frames),
        "context_frames": int(cli_args.context_frames),
        "sampling_mode": str(cli_args.sampling_mode),
    }
    with (output_root / "batch_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    model_args = core._build_model_args(cli_args)
    summary: dict[str, object] = {
        "input_json_list_path": str(input_json_list_path),
        "checkpoint_root": str(checkpoint_root),
        "output_root": str(output_root),
        "steps": [],
    }

    for step_name in step_names:
        checkpoint_dir = checkpoint_root / step_name
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"checkpoint dir not found: {checkpoint_dir}")

        step_output_dir = output_root / step_name
        step_output_dir.mkdir(parents=True, exist_ok=True)
        method_name = _build_method_name_from_checkpoint_dir(checkpoint_dir)

        model = core.build_model(model_args)
        model.to(torch.device(cli_args.device))
        model.eval()
        load_info = core._load_v_newtrain_state_into_model(model, checkpoint_dir)
        model.pipe.dit.eval()

        step_success = 0
        step_failed = 0
        step_skipped = 0
        step_entries: list[dict[str, object]] = []
        step_log_lines = [
            f"[checkpoint] {checkpoint_dir}",
            f"[load_info] {json.dumps(load_info, ensure_ascii=False)}",
        ]

        for input_json_path in json_paths:
            payload = core._load_input_json(input_json_path)
            try:
                input_video = core._resolve_input_video(payload, input_json_path)
                input_caption = core._ensure_str_field(payload, "input_caption", input_json_path)
            except (KeyError, ValueError) as exc:
                print(f"[skip] {step_name} {input_json_path.stem}: {exc}")
                step_skipped += 1
                continue

            sample_stem = input_json_path.stem
            output_video = step_output_dir / f"{sample_stem}.mp4"
            output_json = step_output_dir / f"{sample_stem}.json"
            output_log = step_output_dir / f"{sample_stem}.log"

            if output_video.exists() and output_json.exists() and not cli_args.force:
                print(f"[skip] {step_name} {sample_stem}")
                step_skipped += 1
                continue

            try:
                result, case_logs = core._run_single_case_in_process(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    input_json_path=input_json_path,
                    input_video=input_video,
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
                    quality=int(cli_args.quality),
                )
            except Exception as exc:
                error_lines = step_log_lines + [f"[error] {sample_stem}: {exc}"]
                core._write_text_lines(output_log, error_lines)
                print(f"[error] {step_name} {sample_stem}: {exc}")
                step_failed += 1
                continue

            success_lines = step_log_lines + case_logs + [f"[done] {step_name} {sample_stem}"]
            core._write_text_lines(output_log, success_lines)
            result["method"] = method_name
            with output_json.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            step_entries.append(result)
            step_success += 1
            print(f"[done] {step_name} {sample_stem}")

        step_summary = {
            "step": step_name,
            "checkpoint_dir": str(checkpoint_dir),
            "output_dir": str(step_output_dir),
            "load_info": load_info,
            "num_success": step_success,
            "num_failed": step_failed,
            "num_skipped": step_skipped,
            "num_total_requested": len(json_paths),
            "entries": step_entries,
        }
        with (step_output_dir / "result.json").open("w", encoding="utf-8") as handle:
            json.dump(step_summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        summary["steps"].append(step_summary)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(output_root / "summary.json")


if __name__ == "__main__":
    main()
