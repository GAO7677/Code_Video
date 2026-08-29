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


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write a small metadata repair atomically so a worker crash cannot truncate it."""
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def normalize_metric_metadata(task: str, expected_ids: list[str]) -> int:
    """Backfill task_type in legacy per-case metric JSONs.

    Early strict runs wrote the complete metric set but omitted ``task_type``.
    RigidBench's Result loader requires that field even though the metric
    values themselves are otherwise valid.  The authoritative task type is
    the strict GT metadata for the same sample ID, so repair only that missing
    field and leave all measured values untouched.
    """
    metric_dir = RUNS / task / "metrics"
    if not metric_dir.is_dir():
        return 0
    expected = set(expected_ids)
    changed = 0
    unresolved: list[str] = []
    for path in sorted(metric_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        sample_id = str(payload.get("sample_id") or path.stem)
        if sample_id not in expected:
            continue
        updates = False
        if payload.get("sample_id") != sample_id:
            payload["sample_id"] = sample_id
            updates = True
        if not payload.get("task_type"):
            metadata_paths = (
                DATASET / "samples" / sample_id / "metadata.json",
                Path("/data/gaoya/AAA_test_video/physv_v2v_0819_strict")
                / "truth" / "cases" / sample_id / "rigidbench" / "metadata.json",
            )
            task_type = None
            for metadata_path in metadata_paths:
                try:
                    metadata = json.loads(metadata_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(metadata, dict) and metadata.get("task_type"):
                    task_type = str(metadata["task_type"])
                    break
            if task_type is None:
                unresolved.append(sample_id)
                continue
            payload["task_type"] = task_type
            updates = True
        if updates:
            atomic_write_json(path, payload)
            changed += 1
    if unresolved:
        raise RuntimeError(
            f"Cannot infer task_type for {task} samples: {', '.join(sorted(set(unresolved))[:10])}"
        )
    if changed:
        print(f"[metric-only] normalized legacy metadata task={task} files={changed}", flush=True)
    return changed


def metric_sample_ids(task: str, case_ids: list[str]) -> list[str]:
    """Return valid expected sample IDs already represented by metric files."""
    metric_dir = RUNS / task / "metrics"
    expected = set(case_ids)
    found: set[str] = set()
    for path in metric_dir.glob("*.json") if metric_dir.is_dir() else ():
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        sample_id = str(payload.get("sample_id") or path.stem)
        if sample_id in expected:
            found.add(sample_id)
    return sorted(found)


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
    ready = ready_ids(task, case_ids)
    metric_ids = metric_sample_ids(task, case_ids)
    # A few historical runs contain a complete 70-case metric set but lack
    # the tracker intermediates needed to recompute it.  Aggregate those
    # measured values directly instead of treating the task as unprocessable.
    ids = metric_ids if len(metric_ids) >= 70 and not force else ready
    if not ids:
        return False
    normalize_metric_metadata(task, ids)
    if metrics_count(task) >= len(ids) and not force:
        # Older runs may already contain every per-case JSON but have never
        # written the task-level aggregate report.  In that situation there
        # is no scoring work left; build the report directly instead of
        # returning to the outer queue, which would select the same task
        # forever because the global case list can be larger than ``ids``.
        report_path = RUNS / task / "strict_cycles_test70.json"
        if report_path.is_file():
            return False
        sys.path.insert(0, str(strict.RIGIDBENCH_ROOT / "src"))
        sys.path.insert(0, str(strict.RIGIDBENCH_ROOT / "vendor" / "Video-Depth-Anything"))
        import rigidbench.eval.score.context as score_context
        from rigidbench.eval.score.aggregate import aggregate_metrics

        score_context.GT_FPS = FPS
        aggregate = aggregate_metrics(task, str(RUNS), expected_sample_ids=ids, official=False)
        strict.write_metadata(
            RUNS / task,
            task,
            ids,
            sorted(set(case_ids) - set(ids)),
            aggregate,
        )
        subprocess_result = os.system(f"/usr/bin/python3 {BUILDER}")
        print(
            f"[metric-only] aggregated task={task} cases={len(ids)} "
            f"aggregate_rc={subprocess_result}",
            flush=True,
        )
        return True

    sys.path.insert(0, str(strict.RIGIDBENCH_ROOT / "src"))
    sys.path.insert(0, str(strict.RIGIDBENCH_ROOT / "vendor" / "Video-Depth-Anything"))
    strict.load_single_runner().patch_local_trackers()
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
            if task in active:
                continue
            ready = ready_ids(task, case_ids)
            report_path = RUNS / task / "strict_cycles_test70.json"
            if metrics_count(task) >= len(ready) and report_path.is_file():
                continue
            if len(metric_sample_ids(task, case_ids)) >= 70 or ready:
                # score_task chooses direct aggregation for complete legacy
                # metric sets and the metric-only pipeline for ready cases.
                pass
            else:
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
