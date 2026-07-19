#!/usr/bin/env python3
"""Score the Stage1b/LoRA/GT comparison videos with PhysV metrics.

Each metric writes one atomic JSON sidecar per case. Run metric workers in
separate environments/GPUs, then use the dashboard builder to collect them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


PHYSV_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
DEFAULT_DASHBOARD = Path(
    "/data/gaoya/agent-data/outputs/sam2_region_generation_comparison"
)
DEFAULT_SCORE_ROOT = DEFAULT_DASHBOARD / "physv_scores"

if str(PHYSV_ROOT) not in sys.path:
    sys.path.insert(0, str(PHYSV_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric", required=True, choices=["wmreward", "videophy2", "cosmos_reason1"]
    )
    parser.add_argument("--dashboard-dir", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--score-root", type=Path, default=DEFAULT_SCORE_ROOT)
    parser.add_argument("--models", nargs="+", default=["stage1b", "lora", "gt"])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    payload_path = args.dashboard_dir / "dashboard_data.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    selected = set(args.models)
    jobs: list[dict[str, Any]] = []
    for model in payload["models"]:
        if model["name"] not in selected:
            continue
        for case in model["cases"]:
            video = (args.dashboard_dir / case["asset_root"] / model["video_file"]).resolve()
            if not video.is_file():
                raise FileNotFoundError(video)
            jobs.append(
                {
                    "model": model["name"],
                    "model_label": model["label"],
                    "case_key": case["case_key"],
                    "prompt": case["prompt"],
                    "video": video,
                }
            )
    if not jobs:
        raise RuntimeError(f"No jobs found for models: {sorted(selected)}")
    if args.case_limit is not None:
        jobs = jobs[: args.case_limit]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    return [job for index, job in enumerate(jobs) if index % args.num_shards == args.shard_index]


def create_runner(metric: str) -> Any:
    if metric == "wmreward":
        from physv_eval.wmreward_official import WMRewardRunner

        return WMRewardRunner()
    if metric == "videophy2":
        from physv_eval.videophy2_auto import VideoPhy2Runner

        return VideoPhy2Runner(device="cuda")
    from physv_eval.cosmos_reason1_official import OfficialCosmosReason1Runner

    return OfficialCosmosReason1Runner()


def score_job(metric: str, runner: Any, job: dict[str, Any]) -> dict[str, Any]:
    video = job["video"]
    if metric == "wmreward":
        return runner.score(video)
    if metric == "videophy2":
        sa = runner.score_video(video, task="sa", caption=job["prompt"])
        pc = runner.score_video(video, task="pc")
        return {"sa": sa, "pc": pc}
    return runner.score(video)


def is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "ok"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    args = parse_args()
    jobs = load_jobs(args)
    pending = []
    for job in jobs:
        output = args.score_root / job["model"] / job["case_key"] / f"{args.metric}.json"
        if args.force or not is_complete(output):
            pending.append((job, output))

    print(
        f"metric={args.metric} shard={args.shard_index}/{args.num_shards} "
        f"selected={len(jobs)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return

    runner = create_runner(args.metric)
    failures = 0
    for index, (job, output) in enumerate(pending, start=1):
        started = time.time()
        record: dict[str, Any] = {
            "status": "ok",
            "metric": args.metric,
            "model": job["model"],
            "model_label": job["model_label"],
            "case_key": job["case_key"],
            "video": str(job["video"]),
            "prompt": job["prompt"],
        }
        try:
            record["result"] = score_job(args.metric, runner, job)
        except Exception as exc:  # Keep the batch resumable after a corrupt case.
            failures += 1
            record.update(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
        record["elapsed_seconds"] = time.time() - started
        atomic_write_json(output, record)
        print(
            f"[{index}/{len(pending)}] {record['status']} "
            f"{job['model']}/{job['case_key']} {record['elapsed_seconds']:.2f}s",
            flush=True,
        )
    if failures:
        raise SystemExit(f"{failures} case(s) failed")


if __name__ == "__main__":
    main()
