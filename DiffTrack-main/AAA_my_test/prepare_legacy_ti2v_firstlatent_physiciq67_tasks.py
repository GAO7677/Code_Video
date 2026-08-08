#!/usr/bin/env python3
"""Write PhysicIQ67 case manifest and missing PCK task list."""

from __future__ import annotations

import argparse
import json

from AAA_my_test.legacy_ti2v_firstlatent_physiciq67_common import (
    CASES,
    OUTPUT_ROOT,
    TASKS_JSONL,
    all_tasks,
    read_seeds,
    run_dir,
    write_case_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_case_manifest()
    tasks = []
    completed = 0
    for case, seed in all_tasks():
        output = run_dir(case.key, seed)
        if (output / "complete.json").is_file() and not args.overwrite_complete:
            completed += 1
            continue
        tasks.append({"case_key": case.key, "seed": int(seed)})

    tmp = TASKS_JSONL.with_suffix(TASKS_JSONL.suffix + ".tmp")
    tmp.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in tasks),
        encoding="utf-8",
    )
    tmp.replace(TASKS_JSONL)
    summary = {
        "case_count": len(CASES),
        "seed_count": len(read_seeds()),
        "expected_runs": len(CASES) * len(read_seeds()),
        "completed_runs": completed,
        "missing_runs": len(tasks),
        "tasks_jsonl": str(TASKS_JSONL),
        "output_root": str(OUTPUT_ROOT),
    }
    (OUTPUT_ROOT / "task_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
