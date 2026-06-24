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


def _list_checkpoint_steps(checkpoint_dir: Path) -> list[Path]:
    steps: list[Path] = []
    for path in sorted(checkpoint_dir.glob("step-*")):
        if path.is_dir() and (path / "checkpoint.safetensors").is_file():
            steps.append(path)
    return steps


def _checkpoint_signature(checkpoints: list[Path]) -> list[str]:
    return [path.name for path in checkpoints]


def _select_checkpoints(checkpoints: list[Path], latest_k: int | None) -> list[Path]:
    if latest_k is None or latest_k <= 0 or len(checkpoints) <= latest_k:
        return checkpoints
    return checkpoints[-latest_k:]


def _load_state(state_file: Path) -> dict[str, object]:
    if not state_file.is_file():
        return {"last_signature": [], "history": []}
    state = json.loads(state_file.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        return {"last_signature": [], "history": []}
    if not isinstance(state.get("last_signature"), list):
        state["last_signature"] = []
    if not isinstance(state.get("history"), list):
        state["history"] = []
    return state


def _save_state(state_file: Path, state: dict[str, object]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_file.with_suffix(".tmp.json")
    tmp_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(state_file)


def _build_command(args: argparse.Namespace, checkpoints: list[Path]) -> list[str]:
    cmd = [
        str(args.python_bin),
        str(args.compare_script),
        "--checkpoints",
    ]
    cmd.extend(str(path) for path in checkpoints)
    cmd.extend(
        [
            "--indices",
            *[str(index) for index in args.indices],
            "--output-dir",
            str(args.output_dir),
            "--port",
            str(args.port),
            "--fps",
            str(args.fps),
            "--device",
            args.device,
            "--no-serve",
        ]
    )
    if args.dataset_root is not None:
        cmd.extend(["--dataset-root", str(args.dataset_root)])
    if args.split is not None:
        cmd.extend(["--split", str(args.split)])
    if args.wan_root is not None:
        cmd.extend(["--wan-root", str(args.wan_root)])
    return cmd


def _run_compare(args: argparse.Namespace, checkpoints: list[Path], state: dict[str, object]) -> int:
    selected = _select_checkpoints(checkpoints, args.latest_k)
    cmd = _build_command(args, selected)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    repo_root = str(Path(__file__).resolve().parent.parent)
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [repo_root, str(args.diffsynth_root)]
    env["PYTHONPATH"] = ":".join(
        [part for part in pythonpath_parts if part] + ([existing_pythonpath] if existing_pythonpath else [])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "watch_v_newtrain_aux_compare.log"
    started_at = _utc_now()
    print(f"[{started_at}] aux compare start", flush=True)
    print("command:", " ".join(cmd), flush=True)
    with open(log_path, "a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{started_at}] START {' '.join(cmd)}\n")
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

    history = state.setdefault("history", [])
    assert isinstance(history, list)
    history.append(
        {
            "timestamp": finished_at,
            "returncode": int(process.returncode),
            "selected_checkpoints": [str(path) for path in selected],
            "output_dir": str(args.output_dir),
            "log_path": str(log_path),
        }
    )
    state["last_signature"] = _checkpoint_signature(checkpoints)
    state["last_rendered"] = [str(path) for path in selected]
    state["last_returncode"] = int(process.returncode)
    state["last_timestamp"] = finished_at
    _save_state(args.state_file, state)
    print(f"[{finished_at}] aux compare done returncode={process.returncode}", flush=True)
    return int(process.returncode)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Watch v_newtrain checkpoint directories and regenerate the dual-view aux-loss comparison page."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
            "pybullet0624_diffsynth_object_v_newtrain_gpu67_lossfix_v18_boxwhmax/checkpoints"
        ),
    )
    parser.add_argument(
        "--compare-script",
        type=Path,
        default=repo_root / "code_vjepa_vggt" / "inspect_train_aux_losses_v_newtrain_compare.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/0623/train/train0624/aux_loss_vis_v18_watch"),
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/0623/train/train0624/aux_loss_vis_v18_watch/watch_state.json"),
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
    )
    parser.add_argument(
        "--diffsynth-root",
        type=Path,
        default=Path("/home/gaoya/Code_Video/DiffSynth-Studio-main"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/"
            "industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500"
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--indices", type=int, nargs="+", default=[339, 27])
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--port", type=int, default=8816)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--latest-k", type=int, default=4)
    parser.add_argument(
        "--wan-root",
        type=Path,
        default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"),
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--process-existing", action="store_true")
    args = parser.parse_args()

    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.compare_script = args.compare_script.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.state_file = args.state_file.expanduser().resolve()
    args.diffsynth_root = args.diffsynth_root.expanduser().resolve()
    if args.dataset_root is not None:
        args.dataset_root = args.dataset_root.expanduser().resolve()
    if args.wan_root is not None:
        args.wan_root = args.wan_root.expanduser().resolve()

    state = _load_state(args.state_file)
    startup_signature = _checkpoint_signature(_list_checkpoint_steps(args.checkpoint_dir))
    if not args.process_existing and "startup_existing" not in state:
        state["startup_existing"] = startup_signature
        _save_state(args.state_file, state)

    print(f"[{_utc_now()}] watch_v_newtrain_aux_compare start", flush=True)
    print(f"checkpoint_dir: {args.checkpoint_dir}", flush=True)
    print(f"output_dir: {args.output_dir}", flush=True)
    print(f"gpu: {args.gpu}", flush=True)

    while True:
        checkpoints = _list_checkpoint_steps(args.checkpoint_dir)
        current_signature = _checkpoint_signature(checkpoints)
        last_signature = state.get("last_signature", [])
        startup_existing = set(state.get("startup_existing", [])) if not args.process_existing else set()
        eligible_checkpoints = [path for path in checkpoints if path.name not in startup_existing]
        eligible_signature = _checkpoint_signature(eligible_checkpoints)

        if eligible_checkpoints and eligible_signature != last_signature:
            _run_compare(args, eligible_checkpoints, state)
        else:
            print(f"[{_utc_now()}] no new checkpoints for aux compare", flush=True)

        if args.once:
            break
        time.sleep(float(args.poll_seconds))
        state = _load_state(args.state_file)


if __name__ == "__main__":
    main()
