#!/usr/bin/env python3
"""Run an explicit subset of legacy TI2V first-latent PCK tasks."""

from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path

import torch

from AAA_my_test.legacy_ti2v_firstlatent_common import CASES, OUTPUT_ROOT, run_dir
from AAA_my_test.run_legacy_ti2v_firstlatent_pck_worker import (
    build_args,
    build_wan_ti2v_pipeline,
    load_cotracker,
    process_task,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-jsonl", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_tasks(path: Path) -> list[tuple[str, int]]:
    tasks: list[tuple[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        tasks.append((str(item["case_key"]), int(item["seed"])))
    return tasks


def main() -> None:
    args = parse_args()
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0, num-workers)")

    case_by_key = {case.key: case for case in CASES}
    raw_tasks = read_tasks(args.task_jsonl)
    tasks = []
    for index, (case_key, seed) in enumerate(raw_tasks):
        if index % args.num_workers != args.worker_id:
            continue
        case = case_by_key[case_key]
        output = run_dir(case.key, seed)
        if (output / "complete.json").is_file() and not args.overwrite:
            continue
        tasks.append((case, seed))

    print(
        f"selected worker {args.worker_id}/{args.num_workers}: {len(tasks)} tasks",
        flush=True,
    )
    if not tasks:
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pipe = build_wan_ti2v_pipeline(build_args(tasks[0][1]))
    cotracker = load_cotracker(str(args.device))
    for index, (case, seed) in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] start {case.key} seed={seed}", flush=True)
        output = run_dir(case.key, seed)
        try:
            process_task(pipe, cotracker, case, seed, str(args.device), bool(args.overwrite))
        except Exception:
            output.mkdir(parents=True, exist_ok=True)
            (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(traceback.format_exc(), flush=True)
            raise
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{index}/{len(tasks)}] complete {case.key} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
