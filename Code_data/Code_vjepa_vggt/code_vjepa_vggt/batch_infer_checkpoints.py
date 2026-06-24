from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _parse_step(path: Path) -> int:
    candidates: list[str] = []
    if path.is_dir():
        candidates.append(path.name)
    else:
        candidates.append(path.stem)
        if path.name == "checkpoint.safetensors":
            candidates.append(path.parent.name)
    for candidate in candidates:
        if candidate.startswith("step_"):
            suffix = candidate.split("_", 1)[1]
        elif candidate.startswith("step-"):
            suffix = candidate.split("-", 1)[1]
        else:
            continue
        try:
            return int(suffix)
        except ValueError:
            continue
    return -1


def _list_checkpoints(checkpoint_dir: Path) -> list[Path]:
    legacy_files = [path for path in checkpoint_dir.glob("step_*.pt") if path.is_file()]
    safetensor_dirs = [
        path
        for path in checkpoint_dir.glob("step-*")
        if path.is_dir() and (path / "checkpoint.safetensors").is_file()
    ]
    return sorted(legacy_files + safetensor_dirs, key=lambda path: (_parse_step(path), path.name))


def _checkpoint_name(checkpoint_path: Path) -> str:
    if checkpoint_path.is_dir():
        return checkpoint_path.name
    if checkpoint_path.name == "checkpoint.safetensors":
        return checkpoint_path.parent.name
    return checkpoint_path.stem


def _build_command(
    python_bin: str,
    infer_script: Path,
    checkpoint_path: Path,
    config_path: Path,
    context_video: Path,
    prompt: str,
    output_dir: Path,
    output_video: Path,
    num_frames: int,
    sampling_mode: str,
    sampling_steps: int,
    fps: int,
    seed: int,
        wan_lora_only: bool,
) -> list[str]:
    cmd = [
        python_bin,
        str(infer_script),
        "--checkpoint",
        str(checkpoint_path),
        "--context-video",
        str(context_video),
        "--prompt",
        prompt,
        "--output-dir",
        str(output_dir),
        "--output-video",
        str(output_video),
        "--num-frames",
        str(num_frames),
        "--sampling-mode",
        str(sampling_mode),
        "--sampling-steps",
        str(sampling_steps),
        "--fps",
        str(fps),
        "--seed",
        str(seed),
    ]
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])
    if wan_lora_only:
        cmd.append("--wan-lora-only")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for every step_*.pt under a checkpoint directory.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--infer-script", required=True)
    parser.add_argument("--context-video", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wan-lora-only",
        action="store_true",
        help="Run Wan backbone + configured LoRA only, without initializing object branches or loading step_*.pt trainable modules.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    infer_script = Path(args.infer_script).expanduser().resolve()
    context_video = Path(args.context_video).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    run_name = checkpoint_dir.name
    if bool(args.wan_lora_only):
        run_name = f"{run_name}_wan_lora_only"
    run_output_dir = output_root / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = _list_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise FileNotFoundError(f"no supported checkpoints found under {checkpoint_dir}")

    summary: dict[str, object] = {
        "checkpoint_dir": str(checkpoint_dir),
        "run_name": run_name,
        "context_video": str(context_video),
        "prompt": str(args.prompt),
        "seed": int(args.seed),
        "gpu": str(args.gpu),
        "items": [],
    }

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    repo_root = str(infer_script.parent.parent)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_root if not existing_pythonpath else f"{repo_root}:{existing_pythonpath}"

    for checkpoint_path in checkpoints:
        ckpt_name = _checkpoint_name(checkpoint_path)
        step_output_dir = run_output_dir / ckpt_name
        step_output_dir.mkdir(parents=True, exist_ok=True)
        output_video = run_output_dir / f"{ckpt_name}.mp4"
        output_json = run_output_dir / f"{ckpt_name}.json"
        log_path = step_output_dir / "infer.log"

        if output_video.exists() and output_json.exists() and not args.force:
            summary["items"].append(
                {
                    "checkpoint": str(checkpoint_path),
                    "video": str(output_video),
                    "json": str(output_json),
                    "status": "skip_existing",
                }
            )
            continue

        cmd = _build_command(
            python_bin=str(args.python_bin),
            infer_script=infer_script,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            context_video=context_video,
            prompt=str(args.prompt),
            output_dir=step_output_dir,
            output_video=output_video,
            num_frames=int(args.num_frames),
            sampling_mode=str(args.sampling_mode),
            sampling_steps=int(args.sampling_steps),
            fps=int(args.fps),
            seed=int(args.seed),
            wan_lora_only=bool(args.wan_lora_only),
        )

        print(f"running: {checkpoint_path.name}", flush=True)
        print("command:", " ".join(cmd), flush=True)

        with open(log_path, "a", encoding="utf-8") as log_handle:
            process = subprocess.run(
                cmd,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        result_json = step_output_dir / "result.json"
        if result_json.exists():
            output_json.write_text(result_json.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            output_json.write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint_path),
                        "status": "missing_result_json",
                        "returncode": int(process.returncode),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        summary["items"].append(
            {
                "checkpoint": str(checkpoint_path),
                "video": str(output_video),
                "json": str(output_json),
                "result_dir": str(step_output_dir),
                "log": str(log_path),
                "returncode": int(process.returncode),
                "status": "ok" if process.returncode == 0 and output_video.exists() else "failed",
            }
        )

    summary_path = run_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
