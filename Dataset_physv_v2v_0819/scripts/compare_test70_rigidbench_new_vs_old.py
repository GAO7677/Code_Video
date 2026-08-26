#!/usr/bin/env python3
"""Regression-check video-only prediction extraction against old metric JSONs.

This script never writes task metrics.  It recomputes the selected metrics from
strict GT inputs plus the generated video and compares the scalars with the
previous cache-backed results.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

RUNNER_DIR = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts")
sys.path.insert(0, str(RUNNER_DIR))
import run_test70_rigidbench_metric_backfill as backfill  # noqa: E402


METRIC_KEYS = backfill.METRICS


def close_value(old, new, atol: float, rtol: float) -> bool:
    if old is None or new is None:
        return old == new
    try:
        old_f, new_f = float(old), float(new)
    except (TypeError, ValueError):
        return old == new
    if math.isnan(old_f) and math.isnan(new_f):
        return True
    return bool(np.isclose(old_f, new_f, atol=atol, rtol=rtol, equal_nan=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare video-only RigidBench recomputation with old JSON metrics")
    parser.add_argument("--input-root", type=Path, default=backfill.DEFAULT_INPUT_ROOT)
    parser.add_argument("--strict-root", type=Path, default=backfill.DEFAULT_STRICT_ROOT)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--all-complete", action="store_true", help="Use cases with old metric fields already present")
    parser.add_argument("--metric", choices=METRIC_KEYS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metrics", default=",".join(METRIC_KEYS))
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    tasks = backfill.task_dirs(args.input_root, args.task_id)
    if not tasks:
        raise SystemExit(f"No requested task found: {args.task_id}")
    rows = []
    overall_ok = True
    metrics = [args.metric] if args.metric else [x.strip() for x in args.metrics.split(",") if x.strip()]
    for metric in metrics:
        model = backfill.load_shared_model(metric, args.device)
        for task in tasks:
            if args.all_complete:
                candidates = []
                for path in sorted((task / "metrics").glob("*.json")):
                    payload = backfill.read_json(path)
                    complete = payload.get(metric) is not None if args.metric else all(payload.get(k) is not None for k in METRIC_KEYS)
                    if complete and (task / "generated" / path.stem).is_dir():
                        candidates.append(path.stem)
            else:
                candidates = list(args.case_id)
            for case_id in candidates:
                if not backfill.metric_inputs_ready(task, case_id, metric, args.strict_root):
                    rows.append({"task": task.name, "metric": metric, "case": case_id, "status": "not_ready"})
                    overall_ok = False
                    continue
                old_payload = backfill.read_json(backfill.task_sample_json(task, case_id))
                try:
                    new_payload = backfill.compute(metric, task, case_id, args.strict_root, model, args.device)
                    key = metric
                    old_value = old_payload.get(key)
                    new_value = new_payload.get(key)
                    ok = close_value(old_value, new_value, args.atol, args.rtol)
                    overall_ok = overall_ok and ok
                    rows.append({
                        "task": task.name,
                        "metric": metric,
                        "case": case_id,
                        "old": old_value,
                        "new": new_value,
                        "abs_diff": abs(float(old_value) - float(new_value)) if old_value is not None and new_value is not None else None,
                        "status": "match" if ok else "mismatch",
                    })
                except Exception as exc:
                    overall_ok = False
                    rows.append({"task": task.name, "metric": metric, "case": case_id, "status": "error", "error": repr(exc)})
        del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"tasks": args.task_id, "cases": args.case_id, "all_complete": args.all_complete, "ok": overall_ok, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": overall_ok, "report": str(args.report), "rows": len(rows)}, ensure_ascii=False))
    return 0 if overall_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
