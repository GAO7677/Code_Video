#!/usr/bin/env python3
"""Resume strict-CYCLES RigidBench evaluation serially on physical GPU5.

The strict adapter writes large tracker/depth intermediates.  This queue
keeps the durable per-case metrics and task report, then removes only those
rebuildable intermediates after each task so the queue can cover the full
test70 inventory without exhausting the data volume.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


STRICT_SCRIPT = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/run_test70_rigidbench_strict.py")
BUILDER = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/build_test70_rigidbench_metrics.py")
PYTHON = Path("/home/gaoya/miniconda3/envs/sam/bin/python")
DASHBOARD = Path(
    "/data/gaoya/agent-data/physv_v2v_0819/visualization/hub/"
    "physv-v2v-0819-test70-no-event-timing-40step/dashboard.json"
)
REPORT_ROOT = Path("/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_strict_test70")
RUNS = REPORT_ROOT / "runs"
LOG = REPORT_ROOT / "logs" / "gpu5_strict_queue.log"
PYTHONPATH = ":".join(
    (
        "/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src",
        "/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/vendor/Video-Depth-Anything",
    )
)
MIN_FREE_BYTES = 12 * 1024**3


def load_strict_module():
    spec = importlib.util.spec_from_file_location("strict_queue_adapter", STRICT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(STRICT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dashboard_tasks() -> list[str]:
    payload = json.loads(DASHBOARD.read_text())
    return [str(row["task_id"]) for row in payload.get("models", [])]


def report(task: str) -> dict:
    path = RUNS / task / "strict_cycles_test70.json"
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def cleanup_intermediates(task: str) -> None:
    """Drop only files that can be regenerated from the durable videos/GT."""
    run = RUNS / task
    for name in ("tracks", "masks", "depth"):
        path = run / name
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
    generated = run / "generated"
    if generated.is_dir() and not generated.is_symlink():
        for path in generated.iterdir():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)


def append_log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()
    print(message, flush=True)


def main() -> int:
    lock_path = REPORT_ROOT / "gpu5_strict_queue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            append_log("another GPU5 strict queue is already running")
            return 2

        strict = load_strict_module()
        ids = strict.strict_case_ids()
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "5",
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": PYTHONPATH,
            }
        )
        tasks = dashboard_tasks()
        append_log(f"queue start tasks={len(tasks)} gpu=5 strict_ids={len(ids)}")
        completed = 0
        partial = 0
        failed = 0
        for index, task in enumerate(tasks, 1):
            old = report(task)
            if int(old.get("evaluated_case_count", 0)) >= 70:
                completed += 1
                continue
            if shutil.disk_usage(REPORT_ROOT).free < MIN_FREE_BYTES:
                append_log(f"stop: free data volume below {MIN_FREE_BYTES // 2**30} GiB before {task}")
                break
            # Do not spend GPU time on a task whose dashboard has no usable
            # prediction video.  The failed step-400 task still has 69 videos
            # and is intentionally allowed to produce a partial report.
            available = sum(1 for sample_id in ids if strict.output_video(task, sample_id))
            if available == 0:
                append_log(f"[{index}/{len(tasks)}] skip task={task} no prediction videos")
                continue
            append_log(f"[{index}/{len(tasks)}] start task={task} videos={available}")
            command = [str(PYTHON), str(STRICT_SCRIPT), "--task-id", task]
            result = subprocess.run(command, env=env, check=False)
            current = report(task)
            if current:
                cleanup_intermediates(task)
                evaluated = int(current.get("evaluated_case_count", 0))
                if evaluated >= 70:
                    completed += 1
                else:
                    partial += 1
            else:
                # A failed run may have left several gigabytes of tracker
                # output behind; none of it is durable without a report.
                cleanup_intermediates(task)
                failed += 1
            subprocess.run(["/usr/bin/python3", str(BUILDER)], check=False)
            append_log(
                f"[{index}/{len(tasks)}] end task={task} rc={result.returncode} "
                f"evaluated={current.get('evaluated_case_count', 0)}"
            )
        subprocess.run(["/usr/bin/python3", str(BUILDER)], check=False)
        append_log(f"queue end complete={completed} partial={partial} failed={failed}")
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
