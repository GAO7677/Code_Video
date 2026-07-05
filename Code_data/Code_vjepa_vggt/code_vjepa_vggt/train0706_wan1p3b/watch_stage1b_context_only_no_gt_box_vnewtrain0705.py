from __future__ import annotations

# Run command example:
# PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
# /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
# /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/watch_stage1b_context_only_no_gt_box_vnewtrain0705.py

"""
Foreground watcher for train0706 stage1b checkpoints.

Behavior:
1. Poll `step-*` under the training checkpoint directory.
2. When a new `checkpoint.safetensors` becomes stable, run batch inference on
   `test_5.txt`.
3. After at least one new step finishes inference successfully, run `bench.sh`
   over the whole V2V result root.

State is stored under:
  <result-root>/.watch_stage1b_context_only_no_gt_box_vnewtrain0706_wan21_13b/state.json
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_PROJECT_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
DEFAULT_DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
DEFAULT_PYTHON_BIN = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
DEFAULT_INFER_SCRIPT = (
    DEFAULT_PROJECT_ROOT
    / "code_vjepa_vggt/train0706_wan1p3b/wan_stage1b_context_only_no_gt_box_wan21_13b_v2v.py"
)
DEFAULT_BENCH_SCRIPT = DEFAULT_PROJECT_ROOT / "code_vjepa_vggt/AAAinfer/bench.sh"
DEFAULT_CHECKPOINT_ROOT = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
    "train_stage1b_diffsynth_native0706_wan21_13b/run_gpu3567_20260705_retry2/checkpoints"
)
DEFAULT_INPUT_JSON_LIST = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
DEFAULT_RESULT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/v2v_1p3b")
DEFAULT_MODEL_NAME = "train_stage1b_diffsynth_native0706_wan21_13b"
WATCH_STATE_DIRNAME = ".watch_stage1b_context_only_no_gt_box_vnewtrain0706_wan21_13b"
DEFAULT_BENCH_IDLE_GPUS = "0,1,2"


@dataclass(frozen=True)
class StepInfo:
    step_dir: Path
    checkpoint_file: Path
    step_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch train0706 step checkpoints, run batch inference for new steps, "
            "then run bench.sh on the shared V2V result root."
        )
    )
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--input-json-list-path", type=Path, default=DEFAULT_INPUT_JSON_LIST)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--infer-script", type=Path, default=DEFAULT_INFER_SCRIPT)
    parser.add_argument("--bench-script", type=Path, default=DEFAULT_BENCH_SCRIPT)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--diffsynth-root", type=Path, default=DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument("--infer-gpu", default="7")
    parser.add_argument("--bench-gpu", default="0")
    parser.add_argument(
        "--bench-idle-gpus",
        default=DEFAULT_BENCH_IDLE_GPUS,
        help="Comma-separated GPU indices that must be idle before bench.sh runs.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--poll-interval-seconds", type=int, default=120)
    parser.add_argument("--min-checkpoint-age-seconds", type=int, default=60)
    parser.add_argument("--retry-cooldown-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}", flush=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return default
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def ensure_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Watcher already running: {lock_path}") from exc
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def build_state_paths(result_root: Path) -> tuple[Path, Path]:
    state_root = result_root / WATCH_STATE_DIRNAME
    return state_root / "state.json", state_root / "watch.lock"


def default_state(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": 1,
        "checkpoint_root": str(args.checkpoint_root),
        "result_root": str(args.result_root),
        "model_name": str(args.model_name),
        "steps": {},
        "bench": {
            "last_attempt_at": None,
            "last_success_at": None,
            "last_returncode": None,
            "last_error": None,
        },
    }


def discover_ready_steps(checkpoint_root: Path, min_age_seconds: int) -> list[StepInfo]:
    now = time.time()
    step_infos: list[StepInfo] = []
    for step_dir in sorted(checkpoint_root.glob("step-*")):
        if not step_dir.is_dir():
            continue
        checkpoint_file = step_dir / "checkpoint.safetensors"
        if not checkpoint_file.is_file():
            continue
        age_seconds = now - checkpoint_file.stat().st_mtime
        if age_seconds < float(min_age_seconds):
            continue
        step_infos.append(
            StepInfo(
                step_dir=step_dir.resolve(),
                checkpoint_file=checkpoint_file.resolve(),
                step_name=step_dir.name,
            )
        )
    return step_infos


def result_json_path(result_root: Path, model_name: str, step_name: str) -> Path:
    return result_root / model_name / step_name / "result.json"


def result_json_indicates_success(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    if int(payload.get("num_success", 0)) <= 0:
        return False
    return int(payload.get("num_failed", 0)) == 0


def should_retry(step_state: dict[str, Any], now: float, retry_cooldown_seconds: int) -> bool:
    last_attempt_at = step_state.get("last_infer_attempt_at")
    if last_attempt_at is None:
        return True
    return (float(now) - float(last_attempt_at)) >= float(retry_cooldown_seconds)


def build_infer_command(args: argparse.Namespace, step_dir: Path) -> list[str]:
    return [
        str(args.python_bin),
        str(args.infer_script),
        "--weights-root",
        str(step_dir),
        "--input-json-list-path",
        str(args.input_json_list_path),
        "--model-name",
        str(args.model_name),
        "--num-inference-steps",
        str(int(args.num_inference_steps)),
    ]


def build_bench_command(args: argparse.Namespace) -> list[str]:
    return ["bash", str(args.bench_script), str(args.result_root)]


def build_infer_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.project_root}:{args.diffsynth_root}"
    env["DIFFSYNTH_ROOT"] = str(args.diffsynth_root)
    env["CUDA_VISIBLE_DEVICES"] = str(args.infer_gpu)
    return env


def build_bench_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.bench_gpu)
    env["BENCH_CUDA_VISIBLE_DEVICES"] = str(args.bench_gpu)
    return env


def _parse_gpu_index_list(value: str | None) -> list[int]:
    if value is None:
        return []
    parsed: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(int(item))
    return parsed


def bench_idle_gpus_ready(gpu_indices: list[int], *, max_memory_used_mib: int = 1024) -> tuple[bool, str]:
    if not gpu_indices:
        return True, "no idle-gpu gate configured"
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return False, f"nvidia-smi failed with return code {completed.returncode}"
    gpu_stats: dict[int, tuple[int, int]] = {}
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpu_index = int(parts[0])
            memory_used = int(parts[1].split()[0])
            utilization = int(parts[2].split()[0])
        except ValueError:
            continue
        gpu_stats[gpu_index] = (memory_used, utilization)
    missing = [idx for idx in gpu_indices if idx not in gpu_stats]
    if missing:
        return False, f"missing gpu stats for indices: {missing}"
    busy = []
    for idx in gpu_indices:
        memory_used, utilization = gpu_stats[idx]
        if memory_used > int(max_memory_used_mib) or utilization > 0:
            busy.append(f"gpu{idx}(mem={memory_used}MiB,util={utilization}%)")
    if busy:
        return False, "busy: " + ", ".join(busy)
    return True, "idle"


def run_command(command: list[str], *, env: dict[str, str], dry_run: bool) -> tuple[int, str | None]:
    command_str = " ".join(command)
    if dry_run:
        log(f"[dry-run] {command_str}")
        return 0, None
    log(f"[run] {command_str}")
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode == 0:
        return 0, None
    return completed.returncode, f"command failed with return code {completed.returncode}"


def mark_inference_success(step_state: dict[str, Any], step_dir: Path, result_path: Path, now: float) -> None:
    step_state["checkpoint_dir"] = str(step_dir)
    step_state["result_json_path"] = str(result_path)
    step_state["inference_done"] = True
    step_state["bench_done"] = False
    step_state["last_infer_success_at"] = now
    step_state["last_infer_returncode"] = 0
    step_state["last_infer_error"] = None


def run_inference_for_step(
    *,
    args: argparse.Namespace,
    step: StepInfo,
    step_state: dict[str, Any],
    now: float,
) -> bool:
    step_state["last_seen_at"] = now
    step_state["checkpoint_dir"] = str(step.step_dir)
    result_path = result_json_path(args.result_root, args.model_name, step.step_name)

    if result_json_indicates_success(result_path):
        log(f"[skip] inference already complete for {step.step_name}")
        mark_inference_success(step_state, step.step_dir, result_path, now)
        return True

    if not should_retry(step_state, now, int(args.retry_cooldown_seconds)):
        log(f"[wait] retry cooldown active for {step.step_name}")
        return False

    step_state["last_infer_attempt_at"] = now
    command = build_infer_command(args, step.step_dir)
    returncode, error = run_command(command, env=build_infer_env(args), dry_run=bool(args.dry_run))
    step_state["last_infer_returncode"] = returncode
    step_state["last_infer_error"] = error
    if returncode != 0:
        step_state["inference_done"] = False
        return False

    if args.dry_run:
        mark_inference_success(step_state, step.step_dir, result_path, now)
        return True

    if not result_json_indicates_success(result_path):
        step_state["inference_done"] = False
        step_state["last_infer_error"] = f"missing or incomplete result json: {result_path}"
        return False

    mark_inference_success(step_state, step.step_dir, result_path, now)
    return True


def run_bench_if_needed(args: argparse.Namespace, state: dict[str, Any], now: float) -> bool:
    steps = state.setdefault("steps", {})
    pending_steps = [
        step_name
        for step_name, step_state in sorted(steps.items())
        if bool(step_state.get("inference_done")) and not bool(step_state.get("bench_done"))
    ]
    if not pending_steps:
        return False

    idle_gpu_indices = _parse_gpu_index_list(getattr(args, "bench_idle_gpus", None))
    ready, reason = bench_idle_gpus_ready(idle_gpu_indices)
    if not ready:
        log(f"[wait] bench gated by gpu idle check: {reason}")
        return False

    bench_state = state.setdefault("bench", {})
    last_attempt_at = bench_state.get("last_attempt_at")
    if (
        last_attempt_at is not None
        and not args.dry_run
        and (float(now) - float(last_attempt_at)) < float(args.retry_cooldown_seconds)
        and bench_state.get("last_returncode") not in (None, 0)
    ):
        log("[wait] bench retry cooldown active")
        return False

        log(f"[bench] pending steps: {', '.join(pending_steps)}")
        log(f"[bench] gpu idle check: {reason}")
        bench_state["last_attempt_at"] = now
        command = build_bench_command(args)
    returncode, error = run_command(command, env=build_bench_env(args), dry_run=bool(args.dry_run))
    bench_state["last_returncode"] = returncode
    bench_state["last_error"] = error
    if returncode != 0:
        return False

    bench_state["last_success_at"] = now
    bench_state["last_error"] = None
    for step_name in pending_steps:
        steps[step_name]["bench_done"] = True
        steps[step_name]["last_bench_success_at"] = now
    return True


def scan_once(args: argparse.Namespace, state: dict[str, Any]) -> bool:
    now = time.time()
    discovered_steps = discover_ready_steps(args.checkpoint_root, int(args.min_checkpoint_age_seconds))
    if not discovered_steps:
        log("[scan] no ready checkpoints found")
        return False

    log(f"[scan] ready checkpoints: {', '.join(step.step_name for step in discovered_steps)}")
    bench_needed = False
    steps_state = state.setdefault("steps", {})
    for step in discovered_steps:
        step_state = steps_state.setdefault(step.step_name, {})
        success = run_inference_for_step(args=args, step=step, step_state=step_state, now=now)
        if success and not bool(step_state.get("bench_done")):
            bench_needed = True

    if bench_needed:
        run_bench_if_needed(args, state, now)
        return True

    run_bench_if_needed(args, state, now)
    return False


def main() -> None:
    args = parse_args()
    args.checkpoint_root = args.checkpoint_root.expanduser().resolve()
    args.input_json_list_path = args.input_json_list_path.expanduser().resolve()
    args.result_root = args.result_root.expanduser().resolve()
    args.python_bin = args.python_bin.expanduser().resolve()
    args.infer_script = args.infer_script.expanduser().resolve()
    args.bench_script = args.bench_script.expanduser().resolve()
    args.project_root = args.project_root.expanduser().resolve()
    args.diffsynth_root = args.diffsynth_root.expanduser().resolve()

    if not args.input_json_list_path.is_file():
        raise FileNotFoundError(f"input-json-list-path not found: {args.input_json_list_path}")
    if not args.python_bin.is_file():
        raise FileNotFoundError(f"python-bin not found: {args.python_bin}")
    if not args.infer_script.is_file():
        raise FileNotFoundError(f"infer-script not found: {args.infer_script}")
    if not args.bench_script.is_file():
        raise FileNotFoundError(f"bench-script not found: {args.bench_script}")

    state_path, lock_path = build_state_paths(args.result_root)
    state = load_json(state_path, default_state(args))
    lock_handle = ensure_lock(lock_path)
    try:
        log(f"[watch] checkpoint_root={args.checkpoint_root}")
        log(f"[watch] result_root={args.result_root}")
        log(f"[watch] model_name={args.model_name}")
        log(f"[watch] infer_gpu={args.infer_gpu} bench_gpu={args.bench_gpu}")
        while True:
            if not args.checkpoint_root.is_dir():
                log(f"[wait] checkpoint root not ready yet: {args.checkpoint_root}")
                atomic_write_json(state_path, state)
                if args.once:
                    break
                time.sleep(int(args.poll_interval_seconds))
                continue
            changed = scan_once(args, state)
            atomic_write_json(state_path, state)
            if args.once:
                break
            if not changed:
                log(f"[sleep] {int(args.poll_interval_seconds)}s")
            time.sleep(int(args.poll_interval_seconds))
    finally:
        lock_handle.close()


if __name__ == "__main__":
    main()
