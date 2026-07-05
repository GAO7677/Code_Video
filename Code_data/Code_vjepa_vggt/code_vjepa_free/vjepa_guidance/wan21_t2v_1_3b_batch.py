#!/usr/bin/env python3
from __future__ import annotations

"""
Batch Wan2.1 T2V 1.3B inference over a txt file listing one input json per line.

Baseline:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
CUDA_VISIBLE_DEVICES=5 \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan21_t2v_1_3b_batch.py \
  --input-list /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --output-root /data/gaoya/agent-data/outputs/wan21_test5/baseline \
  --model-name wan21_t2v_1p3b_baseline \
  --device-id 0 \
  --disable-vjepa-guidance

Guided:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
CUDA_VISIBLE_DEVICES=5,6 \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan21_t2v_1_3b_batch.py \
  --input-list /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --output-root /data/gaoya/agent-data/outputs/wan21_test5/guided \
  --model-name wan21_t2v_1p3b_guided \
  --device-id 0 \
  --vjepa-device-id 1 \
  --vjepa-guidance-steps 2 \
  --vjepa-min-step-percent 0.35 \
  --vjepa-max-step-percent 0.65 \
  --vjepa-latent-step-size 0.02 \
  --preview-downsample-factor 4 \
  --preview-frame-stride 2 \
  --window-size 8 \
  --context-frames 4 \
  --stride 2
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


THIS_FILE = Path(__file__).resolve()
DEFAULT_SCRIPT_PATH = THIS_FILE.parent / "archive" / "2026-07-cleanup" / "wan21_t2v_1_3b_vjepa.py"
DEFAULT_PYTHON_BIN = Path("/data/gaoya/miniconda3/envs/wan/bin/python")
DEFAULT_CKPT_DIR = Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B-Diffusers")
DEFAULT_VJEPA_CKPT = Path("/data/gaoya/ckpt/VJEPA2/vith.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run Wan2.1 T2V 1.3B baseline/guided generations from a txt file of input json paths."
    )
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--script-path", type=Path, default=DEFAULT_SCRIPT_PATH)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--vjepa-device-id", type=int, default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--transformer-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--vae-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")

    parser.add_argument("--disable-vjepa-guidance", action="store_true")
    parser.add_argument("--vjepa-model", type=str, default="vith")
    parser.add_argument("--vjepa-ckpt", type=Path, default=DEFAULT_VJEPA_CKPT)
    parser.add_argument("--vjepa-guidance-steps", type=int, default=2)
    parser.add_argument("--vjepa-min-step-percent", type=float, default=0.35)
    parser.add_argument("--vjepa-max-step-percent", type=float, default=0.65)
    parser.add_argument("--vjepa-latent-step-size", type=float, default=0.02)
    parser.add_argument("--preview-downsample-factor", type=int, default=4)
    parser.add_argument("--preview-frame-stride", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--reduction", choices=["mean", "max"], default="mean")
    parser.add_argument("--gradient-normalization", choices=["rms", "l2", "none"], default="rms")
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def read_unique_json_paths(path: Path) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value:
            continue
        candidate = Path(value).expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def build_command(args: argparse.Namespace, *, prompt: str, output_video: Path) -> list[str]:
    cmd = [
        str(args.python_bin),
        str(args.script_path),
        "--ckpt_dir",
        str(args.ckpt_dir),
        "--prompt",
        prompt,
        "--output",
        str(output_video),
        "--seed",
        str(args.seed),
        "--device_id",
        str(args.device_id),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num_frames",
        str(args.num_frames),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--guidance_scale",
        str(args.guidance_scale),
        "--flow_shift",
        str(args.flow_shift),
        "--fps",
        str(args.fps),
        "--max_sequence_length",
        str(args.max_sequence_length),
        "--transformer_dtype",
        str(args.transformer_dtype),
        "--vae_dtype",
        str(args.vae_dtype),
        "--log_level",
        str(args.log_level),
    ]
    if args.vjepa_device_id is not None:
        cmd.extend(["--vjepa_device_id", str(args.vjepa_device_id)])
    if args.cpu_offload:
        cmd.append("--cpu_offload")
    if args.disable_vjepa_guidance:
        cmd.append("--disable_vjepa_guidance")
        return cmd

    cmd.extend(
        [
            "--vjepa_model",
            str(args.vjepa_model),
            "--vjepa_ckpt",
            str(args.vjepa_ckpt),
            "--vjepa_guidance_steps",
            str(args.vjepa_guidance_steps),
            "--vjepa_min_step_percent",
            str(args.vjepa_min_step_percent),
            "--vjepa_max_step_percent",
            str(args.vjepa_max_step_percent),
            "--vjepa_latent_step_size",
            str(args.vjepa_latent_step_size),
            "--preview_downsample_factor",
            str(args.preview_downsample_factor),
            "--preview_frame_stride",
            str(args.preview_frame_stride),
            "--window_size",
            str(args.window_size),
            "--context_frames",
            str(args.context_frames),
            "--stride",
            str(args.stride),
            "--reduction",
            str(args.reduction),
            "--gradient_normalization",
            str(args.gradient_normalization),
            "--max_grad_norm",
            str(args.max_grad_norm),
        ]
    )
    return cmd


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "input_list": str(args.input_list.expanduser().resolve()),
        "model_name": str(args.model_name),
        "ckpt_dir": str(args.ckpt_dir),
        "seed": int(args.seed),
        "height": int(args.height),
        "width": int(args.width),
        "num_frames": int(args.num_frames),
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "flow_shift": float(args.flow_shift),
        "fps": int(args.fps),
        "vjepa": {
            "enabled": not bool(args.disable_vjepa_guidance),
            "model": str(args.vjepa_model),
            "checkpoint": str(args.vjepa_ckpt),
            "guidance_steps": int(args.vjepa_guidance_steps),
            "min_step_percent": float(args.vjepa_min_step_percent),
            "max_step_percent": float(args.vjepa_max_step_percent),
            "latent_step_size": float(args.vjepa_latent_step_size),
            "preview_downsample_factor": int(args.preview_downsample_factor),
            "preview_frame_stride": int(args.preview_frame_stride),
            "window_size": int(args.window_size),
            "context_frames": int(args.context_frames),
            "stride": int(args.stride),
            "reduction": str(args.reduction),
            "gradient_normalization": str(args.gradient_normalization),
            "max_grad_norm": float(args.max_grad_norm),
        },
    }
    (args.output_root / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    json_paths = read_unique_json_paths(args.input_list.expanduser().resolve())
    summary_entries: list[dict[str, Any]] = []
    for input_json_path in json_paths:
        payload = load_json(input_json_path)
        prompt = str(payload.get("input_caption") or "").strip()
        if not prompt:
            print(f"[skip] missing input_caption: {input_json_path}", flush=True)
            continue

        sample_stem = input_json_path.stem
        output_video = args.output_root / f"{sample_stem}.mp4"
        output_json = args.output_root / f"{sample_stem}.json"
        output_log = args.output_root / f"{sample_stem}.log"
        if output_video.exists() and output_json.exists() and not args.force:
            print(f"[skip] {sample_stem}", flush=True)
            continue

        cmd = build_command(args, prompt=prompt, output_video=output_video)
        started = time.time()
        with output_log.open("w", encoding="utf-8") as handle:
            handle.write("COMMAND:\n")
            handle.write(subprocess.list2cmdline(cmd) + "\n\n")
            handle.flush()
            process = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
        runtime_sec = time.time() - started

        result = {
            "input_json": str(input_json_path),
            "input_caption": prompt,
            "source_video": payload.get("source_video"),
            "output_video": str(output_video),
            "method": str(args.model_name),
            "seed": int(args.seed),
            "step": int(args.num_inference_steps),
            "guidance": float(args.guidance_scale),
            "ckpt": str(args.ckpt_dir),
            "status": "generated" if process.returncode == 0 and output_video.exists() else "failed",
            "runtime_sec": runtime_sec,
            "returncode": int(process.returncode),
            "command": cmd,
            "vjepa": manifest["vjepa"],
        }
        output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary_entries.append(result)
        print(f"[{'done' if result['status'] == 'generated' else 'fail'}] {sample_stem}", flush=True)
        if process.returncode != 0:
            raise SystemExit(process.returncode)

    summary = {
        "model_name": str(args.model_name),
        "output_root": str(args.output_root),
        "num_total": len(json_paths),
        "num_generated": sum(1 for item in summary_entries if item["status"] == "generated"),
        "num_failed": sum(1 for item in summary_entries if item["status"] != "generated"),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
