#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one manifest case for Wan2.1 T2V 1.3B V-JEPA experiments.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exp_id", type=str, required=True)
    parser.add_argument("--device_id", type=int, default=None)
    parser.add_argument("--vjepa_device_id", type=int, default=None)
    parser.add_argument("--cpu_offload", action="store_true")
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
    cpu_offload: bool = False,
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
        "--output",
        row["output_video"],
        "--seed",
        row["seed"],
        "--device_id",
        device_id,
        "--vjepa_device_id",
        vjepa_device_id,
        "--height",
        row["height"],
        "--width",
        row["width"],
        "--num_frames",
        row["num_frames"],
        "--num_inference_steps",
        row["num_inference_steps"],
        "--guidance_scale",
        row["guidance_scale"],
        "--flow_shift",
        row["flow_shift"],
        "--fps",
        row["fps"],
        "--transformer_dtype",
        row["transformer_dtype"],
        "--vae_dtype",
        row["vae_dtype"],
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
        "--preview_downsample_factor",
        row["vjepa_preview_downsample_factor"],
        "--preview_frame_stride",
        row["vjepa_preview_frame_stride"],
        "--window_size",
        row["vjepa_window_size"],
        "--context_frames",
        row["vjepa_context_frames"],
        "--stride",
        row["vjepa_stride"],
        "--reduction",
        row["vjepa_reduction"],
        "--gradient_normalization",
        row["vjepa_grad_norm_mode"],
        "--max_grad_norm",
        row["vjepa_max_grad_norm"],
    ]
    if row["disable_vjepa_guidance"] == "1":
        cmd.append("--disable_vjepa_guidance")
    if cpu_offload:
        cmd.append("--cpu_offload")
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
        cpu_offload=args.cpu_offload,
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
