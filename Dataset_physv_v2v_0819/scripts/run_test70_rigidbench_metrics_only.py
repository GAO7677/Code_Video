#!/usr/bin/env python3
"""Score ready strict-test70 tracker artifacts without rerunning trackers."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT = Path("/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_strict_test70")
RUNS = ROOT / "runs"
LOCKS = ROOT / "metric_locks"
DATASET = ROOT / "staging" / "rigidbench_dataset"
DASHBOARD = Path("/data/gaoya/agent-data/physv_v2v_0819/visualization/hub/physv-v2v-0819-test70-no-event-timing-40step/dashboard.json")
STRICT_SCRIPT = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/run_test70_rigidbench_strict.py")
BUILDER = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/build_test70_rigidbench_metrics.py")
FPS = 30


def load_strict_module():
    spec = importlib.util.spec_from_file_location("strict_test70_adapter", STRICT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(STRICT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dashboard_tasks() -> list[str]:
    data = json.loads(DASHBOARD.read_text())
    return [str(m["task_id"]) for m in data.get("models", []) if m.get("status") == "complete"]


def active_full_tasks() -> set[str]:
    active: set[str] = set()
    for line in os.popen("ps -eo args").read().splitlines():
        if "run_test70_rigidbench_strict.py --task-id " not in line:
            continue
        active.add(line.split("--task-id ", 1)[1].split()[0])
    return active


def ready_ids(task: str, case_ids: list[str]) -> list[str]:
    run = RUNS / task
    ready = []
    for case in case_ids:
        if (
            (run / "generated" / case / "00000.jpg").is_file()
            and (run / "tracks" / case / "tracks.npz").is_file()
            and (run / "masks" / case / "mask.npz").is_file()
            and (run / "depth" / case / "depth.npz").is_file()
        ):
            ready.append(case)
    return ready


def metrics_count(task: str) -> int:
    return len(list((RUNS / task / "metrics").glob("*.json")))


def score_task(task: str, case_ids: list[str], force: bool = False) -> bool:
    strict = load_strict_module()
    ids = ready_ids(task, case_ids)
    if not ids:
        return False
    if metrics_count(task) >= len(ids) and not force:
        return False

    sys.path.insert(0, str(strict.RIGIDBENCH_ROOT / "src"))
    sys.path.insert(0, str(strict.RIGIDBENCH_ROOT / "vendor" / "Video-Depth-Anything"))
    strict.patch_local_trackers()
    import rigidbench.eval.score.context as score_context
    from rigidbench.eval.pipeline import EvalPipeline
    from rigidbench.eval.samples import load_samples
    from rigidbench.eval.score.aggregate import aggregate_metrics

    score_context.GT_FPS = FPS
    samples = [s for s in load_samples(DATASET, "eval") if s.id in set(ids)]
    pipeline = EvalPipeline(task, str(DATASET), str(RUNS), generated_fps=FPS)
    stage = pipeline._run_evaluation(samples, force=force)
    if stage.failed:
        print(f"[metric-only] {task} failed: {stage.failed[:5]}", flush=True)
        return False
    aggregate = aggregate_metrics(task, str(RUNS), expected_sample_ids=ids, official=False)
    strict.write_metadata(RUNS / task, task, ids, sorted(set(case_ids) - set(ids)), aggregate)
    subprocess_result = os.system(f"/usr/bin/python3 {BUILDER}")
    print(f"[metric-only] scored task={task} cases={len(ids)} aggregate_rc={subprocess_result}", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    strict = load_strict_module()
    case_ids = strict.strict_case_ids()
    LOCKS.mkdir(parents=True, exist_ok=True)
    print(f"[metric-only] worker={args.worker_id} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} started", flush=True)
    while True:
        active = active_full_tasks()
        for task in dashboard_tasks():
            if task in active or metrics_count(task) >= len(case_ids):
                continue
            if not ready_ids(task, case_ids):
                continue
            lock_path = LOCKS / f"{task}.lock"
            with lock_path.open("w") as lock_file:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                try:
                    if task not in active_full_tasks():
                        score_task(task, case_ids)
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            break
        else:
            if args.once:
                return 0
            print(f"[metric-only] worker={args.worker_id} no ready task; waiting", flush=True)
            import time
            time.sleep(max(1, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
