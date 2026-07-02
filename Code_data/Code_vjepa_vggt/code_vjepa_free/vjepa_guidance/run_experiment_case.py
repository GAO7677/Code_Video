#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one manifest case for Wan2.2 TI2V V-JEPA experiments.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exp_id", type=str, required=True)
    parser.add_argument("--device_id", type=int, default=None)
    parser.add_argument("--vjepa_device_id", type=int, default=None)
    parser.add_argument("--offload_model", action="store_true")
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--convert_model_dtype", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_manifest_row(manifest_path: Path, exp_id: str) -> dict[str, str]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["exp_id"] == exp_id:
                return row
    raise KeyError(f"exp_id not found in manifest: {exp_id}")


def build_command(
    row: dict[str, str],
    *,
    device_id_override: int | None = None,
    vjepa_device_id_override: int | None = None,
    offload_model: bool = False,
    t5_cpu: bool = False,
    convert_model_dtype: bool = False,
) -> list[str]:
    device_id = str(device_id_override) if device_id_override is not None else row["device_id"]
    vjepa_device_id = (
        str(vjepa_device_id_override) if vjepa_device_id_override is not None else row["vjepa_device_id"]
    )
    cmd = [
        row["python_bin"],
        row["script_path"],
        "--ckpt_dir",
        row["ckpt_dir"],
        "--prompt",
        row["prompt"],
        "--image",
        row["source_image"],
        "--output",
        row["output_video"],
        "--sample_steps",
        row["sample_steps"],
        "--sample_solver",
        row["sample_solver"],
        "--sample_shift",
        row["sample_shift"],
        "--sample_guide_scale",
        row["sample_guide_scale"],
        "--frame_num",
        row["frame_num"],
        "--size",
        row["size"],
        "--seed",
        row["seed"],
        "--device_id",
        device_id,
        "--vjepa_device_id",
        vjepa_device_id,
        "--vjepa_model",
        row["vjepa_model"],
        "--vjepa_ckpt",
        row["vjepa_ckpt"],
        "--vjepa_guidance_steps",
        row["vjepa_guidance_steps"],
        "--vjepa_min_step_percent",
        row["vjepa_min_step_percent"],
        "--vjepa_max_step_percent",
        row["vjepa_max_step_percent"],
        "--vjepa_latent_step_size",
        row["vjepa_latent_step_size"],
        "--vjepa_preview_downsample_factor",
        row["vjepa_preview_downsample_factor"],
        "--vjepa_preview_frame_stride",
        row["vjepa_preview_frame_stride"],
        "--vjepa_window_size",
        row["vjepa_window_size"],
        "--vjepa_context_frames",
        row["vjepa_context_frames"],
        "--vjepa_stride",
        row["vjepa_stride"],
        "--vjepa_reduction",
        row["vjepa_reduction"],
        "--vjepa_grad_norm_mode",
        row["vjepa_grad_norm_mode"],
        "--vjepa_max_grad_norm",
        row["vjepa_max_grad_norm"],
    ]
    if row["disable_vjepa_guidance"] == "1":
        cmd.append("--disable_vjepa_guidance")
    if offload_model:
        cmd.append("--offload_model")
    if t5_cpu:
        cmd.append("--t5_cpu")
    if convert_model_dtype:
        cmd.append("--convert_model_dtype")
    return cmd


def write_sidecar_json(
    row: dict[str, str],
    *,
    command: list[str],
    status: str,
    runtime_sec: float,
    returncode: int | None,
    stdout_path: Path,
) -> None:
    payload = dict(row)
    payload.update(
        {
            "status": status,
            "runtime_sec": runtime_sec,
            "returncode": returncode,
            "command": command,
            "stdout_log": str(stdout_path),
            "output_exists": Path(row["output_video"]).exists(),
        }
    )
    output_json = Path(row["output_json"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    row = load_manifest_row(args.manifest, args.exp_id)
    output_video = Path(row["output_video"])
    output_video.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = output_video.with_suffix(".log")
    cmd = build_command(
        row,
        device_id_override=args.device_id,
        vjepa_device_id_override=args.vjepa_device_id,
        offload_model=args.offload_model,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    if args.dry_run:
        print(" ".join(subprocess.list2cmdline([part]) for part in cmd))
        return

    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND:\n")
        handle.write(subprocess.list2cmdline(cmd) + "\n\n")
        handle.flush()
        process = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
    runtime_sec = time.time() - started
    status = "ok" if process.returncode == 0 and output_video.exists() else "failed"
    write_sidecar_json(
        row,
        command=cmd,
        status=status,
        runtime_sec=runtime_sec,
        returncode=process.returncode,
        stdout_path=stdout_path,
    )
    if process.returncode != 0:
        raise SystemExit(process.returncode)
    print(output_video)


if __name__ == "__main__":
    main()
