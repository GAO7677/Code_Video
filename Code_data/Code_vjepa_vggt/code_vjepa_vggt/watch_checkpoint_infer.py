from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from code_vjepa_vggt.utils.config import load_yaml_config


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _load_state(state_file: Path) -> dict[str, object]:
    if not state_file.exists():
        return {"processed": {}}
    with open(state_file, "r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict):
        return {"processed": {}}
    processed = state.get("processed")
    if not isinstance(processed, dict):
        state["processed"] = {}
    return state


def _save_state(state_file: Path, state: dict[str, object]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_file.with_suffix(".tmp.json")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)
    tmp_path.replace(state_file)


def _is_complete(step_output_dir: Path) -> bool:
    result_path = step_output_dir / "result.json"
    prediction_path = step_output_dir / "prediction.mp4"
    return result_path.is_file() and prediction_path.is_file()


@dataclass
class WatchConfig:
    checkpoint_dir: Path
    infer_script: Path
    config_path: Path | None
    context_video: Path
    output_dir: Path
    prompt: str
    python_bin: str
    gpu: str
    poll_seconds: float
    num_frames: int
    fps: int
    sampling_mode: str
    sampling_steps: int
    state_file: Path
    once: bool
    process_existing: bool


def _build_command(cfg: WatchConfig, checkpoint_path: Path, step_output_dir: Path) -> list[str]:
    cmd = [
        cfg.python_bin,
        str(cfg.infer_script),
        "--checkpoint",
        str(checkpoint_path),
        "--context-video",
        str(cfg.context_video),
        "--prompt",
        cfg.prompt,
        "--output-dir",
        str(step_output_dir),
        "--output-video",
        str(step_output_dir / "prediction.mp4"),
        "--num-frames",
        str(cfg.num_frames),
        "--sampling-mode",
        cfg.sampling_mode,
        "--sampling-steps",
        str(cfg.sampling_steps),
        "--fps",
        str(cfg.fps),
    ]
    if cfg.config_path is not None:
        cmd.extend(["--config", str(cfg.config_path)])
    return cmd


def _run_infer(cfg: WatchConfig, checkpoint_path: Path, state: dict[str, object]) -> int:
    checkpoint_name = _checkpoint_name(checkpoint_path)
    step_output_dir = cfg.output_dir / checkpoint_name
    step_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = step_output_dir / "infer.log"
    cmd = _build_command(cfg, checkpoint_path, step_output_dir)

    print(f"[{_utc_now()}] infer start: {checkpoint_path}", flush=True)
    print("command:", " ".join(cmd), flush=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    repo_root = str(Path(__file__).resolve().parent.parent)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_root if not existing_pythonpath else f"{repo_root}:{existing_pythonpath}"

    started_at = _utc_now()
    with open(log_path, "a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{started_at}] START {checkpoint_path}\n")
        log_handle.write("COMMAND: " + " ".join(cmd) + "\n")
        log_handle.flush()
        process = subprocess.run(
            cmd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        finished_at = _utc_now()
        log_handle.write(f"[{finished_at}] END returncode={process.returncode}\n")
        log_handle.flush()

    processed = state.setdefault("processed", {})
    assert isinstance(processed, dict)
    processed[checkpoint_name] = {
        "checkpoint": str(checkpoint_path),
        "step": _parse_step(checkpoint_path),
        "output_dir": str(step_output_dir),
        "log_path": str(log_path),
        "status": "ok" if process.returncode == 0 and _is_complete(step_output_dir) else "failed",
        "returncode": int(process.returncode),
        "started_at": started_at,
        "finished_at": finished_at,
    }
    _save_state(cfg.state_file, state)

    print(
        f"[{finished_at}] infer done: checkpoint={checkpoint_path.name} "
        f"returncode={process.returncode} output_dir={step_output_dir}",
        flush=True,
    )
    return int(process.returncode)


def _build_watch_config(args: argparse.Namespace) -> WatchConfig:
    config_path = None
    config = {}
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        config = load_yaml_config(str(config_path))
    fps = int(args.fps if args.fps is not None else config.get("data", {}).get("fps", 30))
    return WatchConfig(
        checkpoint_dir=Path(args.checkpoint_dir).expanduser().resolve(),
        infer_script=Path(args.infer_script).expanduser().resolve(),
        config_path=config_path,
        context_video=Path(args.context_video).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        prompt=str(args.prompt),
        python_bin=str(args.python_bin),
        gpu=str(args.gpu),
        poll_seconds=float(args.poll_seconds),
        num_frames=int(args.num_frames),
        fps=fps,
        sampling_mode=str(args.sampling_mode),
        sampling_steps=int(args.sampling_steps),
        state_file=Path(args.state_file).expanduser().resolve(),
        once=bool(args.once),
        process_existing=bool(args.process_existing),
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_config = repo_root / "code_vjepa_vggt" / "configs" / "train_0624pybullet_freeze_lora_other_modules_gpu67.yaml"
    default_infer_script = repo_root / "code_vjepa_vggt" / "infer_context_video_wan.py"
    default_checkpoint_dir = Path(
        "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_freeze_lora_other_modules_gpu67"
    )
    default_output_dir = Path("/data/gaoya/AAA_test_video/0623/train/train0624/infer_test")

    parser = argparse.ArgumentParser(description="Watch a checkpoint directory and run inference on each new step_*.pt.")
    parser.add_argument("--checkpoint-dir", default=str(default_checkpoint_dir))
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--infer-script", default=str(default_infer_script))
    parser.add_argument(
        "--context-video",
        default="/data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4",
    )
    parser.add_argument("--prompt", default="industrial rigid body simulation sphere")
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--state-file", default=str(default_output_dir / "watch_state.json"))
    parser.add_argument("--once", action="store_true", help="Process currently available checkpoints and exit.")
    parser.add_argument(
        "--process-existing",
        action="store_true",
        help="Also process checkpoints that already exist when the watcher starts. "
        "Default behavior is to start from checkpoints created after watcher startup.",
    )
    args = parser.parse_args()

    cfg = _build_watch_config(args)
    state = _load_state(cfg.state_file)
    startup_checkpoint_names = {path.name for path in _list_checkpoints(cfg.checkpoint_dir)}
    if not cfg.process_existing and "startup_existing" not in state:
        state["startup_existing"] = sorted(startup_checkpoint_names)
        _save_state(cfg.state_file, state)
    print(f"[{_utc_now()}] watcher start", flush=True)
    print(f"checkpoint_dir: {cfg.checkpoint_dir}", flush=True)
    print(f"output_dir: {cfg.output_dir}", flush=True)
    print(f"gpu: {cfg.gpu}", flush=True)

    while True:
        seen_any = False
        checkpoints = _list_checkpoints(cfg.checkpoint_dir)
        processed = state.setdefault("processed", {})
        assert isinstance(processed, dict)
        startup_existing = set(state.get("startup_existing", [])) if not cfg.process_existing else set()
        for checkpoint_path in checkpoints:
            seen_any = True
            checkpoint_name = _checkpoint_name(checkpoint_path)
            step_output_dir = cfg.output_dir / checkpoint_name
            entry = processed.get(checkpoint_name)
            if checkpoint_name in startup_existing and entry is None:
                continue
            if _is_complete(step_output_dir) and isinstance(entry, dict) and entry.get("status") == "ok":
                continue
            if _is_complete(step_output_dir) and entry is None:
                processed[checkpoint_name] = {
                    "checkpoint": str(checkpoint_path),
                    "step": _parse_step(checkpoint_path),
                    "output_dir": str(step_output_dir),
                    "log_path": str(step_output_dir / "infer.log"),
                    "status": "ok",
                    "returncode": 0,
                    "started_at": None,
                    "finished_at": _utc_now(),
                }
                _save_state(cfg.state_file, state)
                continue
            _run_infer(cfg, checkpoint_path, state)

        if cfg.once:
            if not seen_any:
                print(f"[{_utc_now()}] no checkpoints found under {cfg.checkpoint_dir}", flush=True)
            break

        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
