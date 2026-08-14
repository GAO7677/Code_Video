#!/usr/bin/env python3
"""Backfill only missing Object Query ablation metrics under a result directory.

The command recognizes both strict legacy inventories (``video_similarity_top100.json``)
and dynamic M1/M2/M3 Head-Scope directories.  Existing records are accepted only when
their video signature and required audit assets still match the generated video.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AAA_my_test.object_query_ablation_metrics.compute_head_scope_baseline_metrics import (  # noqa: E402
    atomic_json,
    collect_candidates,
    discover_seed_dirs,
)
from AAA_my_test.object_query_ablation_metrics.compute_head_scope_trajectory_metrics import (  # noqa: E402
    locate_baseline,
    validate_frozen_baseline_inputs,
)
from AAA_my_test.object_query_ablation_metrics.common import sha256_file  # noqa: E402


DEFAULT_OUTPUT_BASE = Path(
    "/data/gaoya/agent-data/outputs/object_query_ablation_metrics"
)
STAGES = ("fast", "trajectory", "survival", "complete25", "legacy25")
FAST_KEYS = (
    "impact_score_0_100",
    "spillover_score_0_100",
    "global",
    "target_roi",
    "outside_objects",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("GPU", "0")))
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help=f"comma-separated subset of: {','.join(STAGES)}",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-vbench", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the missing-stage plan without writing inventories or running jobs",
    )
    parser.add_argument("--max-seeds", type=int, default=0, help="debug/smoke limit")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def signature_matches(record: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return record.get("video_signature") == candidate.get("video_signature")


def existing_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    return {
        str(row.get("variant_id") or row.get("id")): row
        for row in payload.get("records", [])
        if isinstance(row, dict) and (row.get("variant_id") or row.get("id"))
    }


def fast_complete(
    output_base: Path, case: str, seed: int, candidates: list[dict[str, Any]]
) -> bool:
    records = existing_records(
        output_base / "head_scope_baseline_fast" / case / f"seed_{seed:05d}" / "report.json"
    )
    for candidate in candidates:
        row = records.get(str(candidate["variant_id"]), {})
        metrics = row.get("metrics", {})
        if not signature_matches(row, candidate) or any(key not in metrics for key in FAST_KEYS):
            return False
    return bool(candidates)


def trajectory_complete(
    output_base: Path, case: str, seed: int, candidates: list[dict[str, Any]]
) -> bool:
    records = existing_records(
        output_base / "head_scope_trajectory" / case / f"seed_{seed:05d}" / "report.json"
    )
    for candidate in candidates:
        row = records.get(str(candidate["variant_id"]), {})
        metrics = row.get("metrics", {})
        if (
            not signature_matches(row, candidate)
            or "quality_pass" not in metrics
            or "target_worst_track_loss_score_0_100" not in metrics
            or not Path(str(row.get("track_path") or "")).is_file()
            or not Path(str(row.get("overlay_path") or "")).is_file()
        ):
            return False
    return bool(candidates)


def survival_complete(
    output_base: Path, case: str, seed: int, candidates: list[dict[str, Any]]
) -> bool:
    records = existing_records(
        output_base
        / "head_scope_trajectory"
        / case
        / f"seed_{seed:05d}"
        / "object_survival_report.json"
    )
    required = (
        "quality_pass",
        "target_worst_disappearance_score_0_100",
        "target_worst_mask_absence_score_0_100",
    )
    for candidate in candidates:
        row = records.get(str(candidate["variant_id"]), {})
        metrics = row.get("metrics", {})
        if (
            not signature_matches(row, candidate)
            or any(key not in metrics for key in required)
            or not Path(str(row.get("mask_path") or "")).is_file()
            or not Path(str(row.get("feature_path") or "")).is_file()
            or not Path(str(row.get("overlay_path") or "")).is_file()
        ):
            return False
    return bool(candidates)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def complete25_report_complete(
    report_path: Path, expected_ids: set[str], require_vbench: bool
) -> bool:
    payload = read_json(report_path)
    records = payload.get("records", [])
    if (
        not isinstance(records, list)
        or int(payload.get("video_count", -1)) != len(expected_ids) + 1
        or int(payload.get("ablation_count", -1)) != len(expected_ids)
        or len(payload.get("metric_definitions", [])) != 25
        or {str(row.get("id")) for row in records} != expected_ids
    ):
        return False
    root = report_path.parent
    if require_vbench:
        baseline_vbench = payload.get("baseline", {}).get("vbench", {})
        if len(baseline_vbench) != 7 or not all(
            finite(item.get("score")) for item in baseline_vbench.values()
        ):
            return False
    for row in records:
        if require_vbench:
            vbench = row.get("vbench", {})
            if len(vbench) != 7 or not all(finite(item.get("score")) for item in vbench.values()):
                return False
        assets = row.get("assets", {})
        for kind in ("trajectory", "mask", "pixel", "raft"):
            if not (root / str(assets.get(kind) or "")).is_file():
                return False
        for object_name in ("object_A", "object_B"):
            for reference in ("baseline", "source_gt_video"):
                try:
                    montage = root / str(assets["perceptual"][object_name][reference])
                except (KeyError, TypeError):
                    return False
                if not montage.is_file() or montage.stat().st_size == 0:
                    return False
    return True


def complete25_eligibility(seed_dir: Path, case: str, seed: int) -> tuple[bool, str]:
    frozen_valid, frozen_reason = validate_frozen_baseline_inputs(seed_dir)
    if not frozen_valid:
        return False, frozen_reason
    try:
        baseline = locate_baseline(case, seed, seed_dir)
        baseline_manifest = read_json(baseline.parent / "manifest.json")
        input_json = Path(str(baseline_manifest.get("input_json") or "")).expanduser().resolve()
        source_payload = read_json(input_json)
        source_video = Path(str(source_payload.get("source_video") or "")).expanduser().resolve()
        source_root = source_video.parent
    except (OSError, ValueError, TypeError) as exc:
        return False, f"cannot resolve Baseline/source: {exc}"
    required = (source_video, source_root / "states.npz", source_root / "meta.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return False, "missing simulator source assets: " + ", ".join(missing)
    try:
        import numpy as np

        with np.load(source_root / "states.npz", allow_pickle=False) as arrays:
            required_keys = {"positions", "quats", "camera_eye", "camera_target", "camera_up"}
            if not required_keys.issubset(arrays.files) or arrays["positions"].shape[1] != 2:
                return False, "states.npz is not the two-object ball-block protocol"
        metadata = read_json(source_root / "meta.json")
        objects = metadata.get("objects", [])
        if (
            len(objects) != 2
            or objects[0].get("shape") != "sphere"
            or objects[1].get("shape") != "box"
        ):
            return False, "simulator objects are not ordered [sphere, box]"
    except (OSError, ValueError, KeyError) as exc:
        return False, f"invalid simulator states: {exc}"
    return True, "eligible"


def baseline_reference_eligibility(
    seed_dir: Path, case: str, seed: int
) -> tuple[bool, str]:
    frozen_valid, frozen_reason = validate_frozen_baseline_inputs(seed_dir)
    if not frozen_valid:
        return False, frozen_reason
    try:
        locate_baseline(case, seed, seed_dir)
    except (FileNotFoundError, OSError) as exc:
        return False, str(exc)
    return True, "eligible"


def build_dynamic_inventory(
    seed_dir: Path, case: str, seed: int, candidates: list[dict[str, Any]], write: bool
) -> tuple[Path, dict[str, Any]]:
    baseline = locate_baseline(case, seed, seed_dir)
    videos: list[dict[str, Any]] = [
        {
            "id": "baseline",
            "protocol": "baseline",
            "target_scope": None,
            "region": None,
            "mask_mode": None,
            "head_scope": None,
            "path": str(baseline.resolve()),
            "file_sha256": sha256_file(baseline),
        }
    ]
    for candidate in sorted(candidates, key=lambda row: str(row["variant_id"])):
        path = Path(str(candidate["path"]))
        videos.append(
            {
                "id": str(candidate["variant_id"]),
                "protocol": "head_scope",
                "target_scope": candidate["target_scope"],
                "region": candidate.get("region"),
                "mask_mode": candidate["mask_mode"],
                "head_scope": candidate["head_scope"],
                "path": str(path.resolve()),
                "file_sha256": sha256_file(path),
            }
        )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_kind": "dynamic_m123_head_scope",
        "case": case,
        "seed": seed,
        "video_count": len(videos),
        "videos": videos,
    }
    path = seed_dir / "metrics_inventory_all_generated.json"
    if write:
        atomic_json(path, payload)
    return path, payload


def command_text(command: list[str]) -> str:
    import shlex

    return shlex.join(command)


def select_balanced_shard(
    entries: list[dict[str, Any]], num_shards: int, shard_index: int
) -> tuple[list[dict[str, Any]], list[int]]:
    """Greedily balance immutable case-seed units by generated video count."""
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(
            f"invalid shard {shard_index}/{num_shards}; require 0 <= index < count"
        )
    bins: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
    loads = [0 for _ in range(num_shards)]
    ordered = sorted(
        entries,
        key=lambda row: (
            -len(row["candidates"]),
            str(row["case"]),
            int(row["seed"]),
        ),
    )
    for entry in ordered:
        target = min(range(num_shards), key=lambda index: (loads[index], index))
        bins[target].append(entry)
        loads[target] += len(entry["candidates"])
    bins[shard_index].sort(key=lambda row: (str(row["case"]), int(row["seed"])))
    return bins[shard_index], loads


def run(command: list[str], dry_run: bool) -> None:
    print("+ " + command_text(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def bench_command(
    input_path: Path, args: argparse.Namespace, *extra: str, output_base: Path | None = None
) -> list[str]:
    command = [
        "bash",
        str(SCRIPT_DIR / "bench.sh"),
        str(input_path),
        "--gpu",
        str(args.gpu),
        "--output-base",
        str(output_base or args.output_base),
        *extra,
    ]
    if args.skip_vbench:
        command.append("--skip-vbench")
    if args.overwrite:
        command.append("--overwrite")
    return command


def main() -> None:
    args = parse_args()
    root = args.result_dir.expanduser().resolve()
    output_base = args.output_base.expanduser().resolve()
    stages = {value.strip() for value in args.stages.split(",") if value.strip()}
    unknown = stages - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stages: {sorted(unknown)}")
    if args.gpu == 4:
        raise SystemExit("GPU 4 is forbidden by /home/gaoya/AGENTS.md")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit(
            f"invalid shard {args.shard_index}/{args.num_shards}; "
            "require 0 <= shard-index < num-shards"
        )
    if not root.is_dir():
        raise SystemExit(f"result directory does not exist: {root}")
    if output_base == Path("/home/gaoya") or Path("/home/gaoya") in output_base.parents:
        raise SystemExit("large metric outputs may not be stored under /home/gaoya")

    seed_entries: list[dict[str, Any]] = []
    for seed_dir in discover_seed_dirs(root):
        candidates = collect_candidates(
            seed_dir,
            {
                "top100",
                "bottom100",
                "random100_layer_matched_draw0",
                "all720",
            },
        )
        if not candidates:
            continue
        case = str(candidates[0]["case"])
        seed = int(candidates[0]["seed"])
        reference_eligible, reference_reason = baseline_reference_eligibility(
            seed_dir, case, seed
        )
        seed_entries.append(
            {
                "seed_dir": seed_dir,
                "case": case,
                "seed": seed,
                "candidates": candidates,
                "reference_eligible": reference_eligible,
                "reference_reason": reference_reason,
            }
        )
    if args.max_seeds > 0:
        seed_entries = seed_entries[: args.max_seeds]
    all_seed_entries = seed_entries[:]
    seed_entries, shard_loads = select_balanced_shard(
        seed_entries, args.num_shards, args.shard_index
    )

    global_fast_missing = any(
        entry["reference_eligible"]
        and not fast_complete(
            output_base,
            entry["case"],
            entry["seed"],
            entry["candidates"],
        )
        for entry in all_seed_entries
    )

    plans = []
    for entry in seed_entries:
        case, seed, candidates = entry["case"], entry["seed"], entry["candidates"]
        eligible, reason = complete25_eligibility(entry["seed_dir"], case, seed)
        plans.append(
            {
                "case": case,
                "seed": seed,
                "seed_dir": str(entry["seed_dir"]),
                "generated_videos": len(candidates),
                "missing": {
                    "fast": entry["reference_eligible"]
                    and not fast_complete(output_base, case, seed, candidates),
                    "trajectory": entry["reference_eligible"]
                    and not trajectory_complete(output_base, case, seed, candidates),
                    "survival": entry["reference_eligible"]
                    and not survival_complete(output_base, case, seed, candidates),
                    "complete25": eligible
                    and not complete25_report_complete(
                        output_base
                        / "head_scope_complete25"
                        / case
                        / f"seed_{seed:05d}"
                        / "report.json",
                        {str(row["variant_id"]) for row in candidates},
                        not args.skip_vbench,
                    ),
                },
                "baseline_reference_eligible": entry["reference_eligible"],
                "baseline_reference_reason": entry["reference_reason"],
                "complete25_eligible": eligible,
                "complete25_reason": reason,
            }
        )

    legacy = []
    if "legacy25" in stages and args.shard_index == 0:
        for inventory in sorted(root.rglob("video_similarity_top100.json")):
            payload = read_json(inventory)
            ids = {str(row.get("id")) for row in payload.get("videos", [])[1:]}
            case = str(payload.get("case") or "")
            seed = int(payload.get("seed", -1))
            if case and seed >= 0 and ids:
                legacy.append(
                    {
                        "inventory": str(inventory),
                        "case": case,
                        "seed": seed,
                        "missing": not complete25_report_complete(
                            output_base / case / f"seed_{seed:05d}" / "report.json",
                            ids,
                            not args.skip_vbench,
                        ),
                    }
                )

    plan_payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(root),
        "output_base": str(output_base),
        "stages": sorted(stages),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "planned_generated_ablation_loads": shard_loads,
        "global_fast_missing": global_fast_missing,
        "case_seed_count": len(plans),
        "generated_ablation_count": sum(row["generated_videos"] for row in plans),
        "head_scope": plans,
        "legacy25": legacy,
    }
    print(json.dumps(plan_payload, ensure_ascii=False, indent=2), flush=True)
    if args.plan_only:
        return

    if (
        "fast" in stages
        and args.shard_index == 0
        and global_fast_missing
    ):
        command = bench_command(root, args, "--head-scope-baseline")
        command.extend(["--workers", str(args.workers)])
        run(command, args.dry_run)

    for entry, plan in zip(seed_entries, plans, strict=True):
        seed_dir = entry["seed_dir"]
        if "trajectory" in stages and plan["missing"]["trajectory"]:
            run(bench_command(seed_dir, args, "--head-scope-trajectory"), args.dry_run)
        if "survival" in stages and plan["missing"]["survival"]:
            run(bench_command(seed_dir, args, "--head-scope-object-survival"), args.dry_run)
        if "complete25" in stages and plan["missing"]["complete25"]:
            inventory_path, _payload = build_dynamic_inventory(
                seed_dir,
                entry["case"],
                entry["seed"],
                entry["candidates"],
                write=not args.dry_run,
            )
            run(
                bench_command(
                    inventory_path,
                    args,
                    "--no-aggregate",
                    output_base=output_base / "head_scope_complete25",
                ),
                args.dry_run,
            )

    for row in legacy:
        if row["missing"]:
            run(
                bench_command(Path(row["inventory"]), args, "--no-aggregate"),
                args.dry_run,
            )

    if not args.dry_run:
        audit_root = output_base / "fill_missing_runs"
        audit_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        atomic_json(audit_root / f"{stamp}.json", plan_payload)
    print("[fill-missing:done] requested missing stages completed", flush=True)


if __name__ == "__main__":
    main()
