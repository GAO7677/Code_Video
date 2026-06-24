from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _list_checkpoint_steps(checkpoint_dir: Path) -> list[str]:
    steps: list[str] = []
    for path in sorted(checkpoint_dir.glob("step-*")):
        if path.is_dir() and (path / "checkpoint.safetensors").is_file():
            steps.append(path.name)
    return steps


def _missing_outputs(checkpoint_dir: Path, output_root: Path) -> list[str]:
    run_output_dir = output_root / checkpoint_dir.name
    missing: list[str] = []
    for step_name in _list_checkpoint_steps(checkpoint_dir):
        mp4_path = run_output_dir / f"{step_name}.mp4"
        json_path = run_output_dir / f"{step_name}.json"
        result_dir = run_output_dir / step_name
        result_json = result_dir / "result.json"
        if not (mp4_path.is_file() and json_path.is_file() and result_json.is_file()):
            missing.append(step_name)
    return missing


def _build_batch_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(args.python_bin),
        str(args.batch_script),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--infer-script",
        str(args.infer_script),
        "--context-video",
        str(args.context_video),
        "--prompt",
        str(args.prompt),
        "--output-root",
        str(args.output_root),
        "--gpu",
        str(args.gpu),
        "--num-frames",
        str(args.num_frames),
        "--sampling-mode",
        str(args.sampling_mode),
        "--sampling-steps",
        str(args.sampling_steps),
        "--fps",
        str(args.fps),
        "--seed",
        str(args.seed),
    ]
    if args.config is not None:
        cmd.extend(["--config", str(args.config)])
    if args.force:
        cmd.append("--force")
    return cmd


def _run_batch(args: argparse.Namespace) -> int:
    cmd = _build_batch_command(args)
    print(f"[{_utc_now()}] batch infer start", flush=True)
    print("command:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    repo_root = str(Path(__file__).resolve().parent.parent)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_root if not existing_pythonpath else f"{repo_root}:{existing_pythonpath}"
    process = subprocess.run(cmd, env=env, check=False)
    print(f"[{_utc_now()}] batch infer done returncode={process.returncode}", flush=True)
    return int(process.returncode)


def _write_status(args: argparse.Namespace, payload: dict[str, object]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "watch_v_newtrain_status.json"
    status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_checkpoint_dir = Path(
        "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_diffsynth_object_v_newtrain_gpu67/checkpoints"
    )
    default_output_root = Path("/data/gaoya/AAA_test_video/0623/train/train0624/infer_v_newtrain_batch")
    parser = argparse.ArgumentParser(
        description="Watch v_newtrain checkpoint directories and repeatedly call the proven batch inference pipeline."
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=default_checkpoint_dir)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--batch-script",
        type=Path,
        default=repo_root / "code_vjepa_vggt" / "batch_infer_checkpoints.py",
    )
    parser.add_argument(
        "--infer-script",
        type=Path,
        default=repo_root / "code_vjepa_vggt" / "infer_v_newtrain_context_video_wan.py",
    )
    parser.add_argument(
        "--context-video",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4"),
    )
    parser.add_argument("--prompt", default="industrial rigid body simulation sphere")
    parser.add_argument("--output-root", type=Path, default=default_output_root)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.batch_script = args.batch_script.expanduser().resolve()
    args.infer_script = args.infer_script.expanduser().resolve()
    args.context_video = args.context_video.expanduser().resolve()
    if args.config is not None:
        args.config = args.config.expanduser().resolve()

    print(f"[{_utc_now()}] watch_v_newtrain_batch_infer start", flush=True)
    print(f"checkpoint_dir: {args.checkpoint_dir}", flush=True)
    print(f"output_root: {args.output_root}", flush=True)
    print(f"gpu: {args.gpu}", flush=True)

    while True:
        missing = _missing_outputs(args.checkpoint_dir, args.output_root)
        status_payload = {
            "timestamp": _utc_now(),
            "checkpoint_dir": str(args.checkpoint_dir),
            "output_root": str(args.output_root),
            "missing_steps": missing,
            "gpu": str(args.gpu),
        }
        _write_status(args, status_payload)
        if missing or args.force:
            returncode = _run_batch(args)
            refreshed_missing = _missing_outputs(args.checkpoint_dir, args.output_root)
            status_payload["last_batch_returncode"] = returncode
            status_payload["timestamp"] = _utc_now()
            status_payload["missing_steps"] = refreshed_missing
            _write_status(args, status_payload)
        else:
            print(f"[{_utc_now()}] no new checkpoints to infer", flush=True)
        if args.once:
            break
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    main()
