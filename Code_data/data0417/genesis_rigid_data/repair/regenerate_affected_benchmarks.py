#!/usr/bin/env python3
# 用途：按审计结果重生成受影响的 benchmark 样本。
"""Regenerate affected benchmark samples using the existing Genesis pipelines.

该脚本用于按审计结果重建受影响的 benchmark 样本；输入为 benchmark 根目录、审计报告和现有生成脚本，输出为重生成的样本目录、过滤结果、stage1 子集及日志。

This script intentionally reuses the current generation entry points instead of
duplicating scene-construction logic:
- pool benchmark objects are regenerated from benchmark_manifest.json
- stage1 heldout single-object motion samples are regenerated with the existing
  case900/case901 command line
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from repair.audit_benchmark_inertial_origins import DEFAULT_BENCHMARK_ROOT, build_report, detect_benchmark_type

DEFAULT_WAN_PYTHON = Path("/data/gaoya/miniconda3/envs/wan/bin/python")
DEFAULT_TRY1_SCRIPT = PROJECT_ROOT / "generators" / "try1_physxnet_articulation_mpm0417.py"
DEFAULT_FILTER_SCRIPT = SCRIPT_DIR / "filter_single_object_motion_cases.py"
DEFAULT_BUILD_SCRIPT = SCRIPT_DIR.parent.parent.parent / "Code_train" / "train_0419" / "state_adapter" / "build_stage1_subsets.py"
DEFAULT_STAGE1_TRAIN_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases")


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_sample_dirs(sample_paths: Sequence[str]) -> Iterable[Path]:
    for sample_path in sample_paths:
        yield Path(str(sample_path))


def _remove_path(path: Path, *, dry_run: bool) -> None:
    if not path.exists():
        return
    print(f"[remove] {path}")
    if dry_run:
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink()


def _run(cmd: List[str], *, dry_run: bool, log_path: Optional[Path] = None) -> int:
    print("[run]", " ".join(cmd))
    if dry_run:
        return 0
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=False)
        return int(result.returncode)
    result = subprocess.run(cmd, check=False)
    return int(result.returncode)


def _select_flagged_objects(report: Dict, requested_root: Path, explicit_ids: Sequence[str]) -> List[str]:
    explicit = {str(item) for item in explicit_ids}
    for root_report in report.get("benchmark_roots", []):
        if Path(root_report["benchmark_root"]).resolve() != requested_root.resolve():
            continue
        flagged = [str(item) for item in root_report.get("flagged_object_ids", [])]
        if explicit:
            flagged = [item for item in flagged if item in explicit]
        return sorted(flagged)
    return sorted(explicit)


def regenerate_stage1_object(
    *,
    benchmark_root: Path,
    object_id: str,
    args: argparse.Namespace,
) -> Dict:
    base = benchmark_root / "train" / "rigid" / "single_object_preview"
    targets = [
        base / "count_01" / f"{object_id}__case900_random_parabola",
        base / "count_01" / f"{object_id}__case901_high_drop",
        base / "invalid_case900_901" / f"{object_id}__case900_random_parabola",
        base / "invalid_case900_901" / f"{object_id}__case901_high_drop",
        benchmark_root / "_asset_cache" / "physxnet_objects" / object_id,
    ]
    for target in targets:
        _remove_path(target, dry_run=bool(args.dry_run))

    cmd = [
        str(args.wan_python),
        str(args.try1_script),
        "--physx_root",
        str(args.physx_root),
        "--version",
        str(args.version),
        "--object_id",
        str(object_id),
        "--output_root",
        str(benchmark_root),
        "--run_genesis",
        "--num_random_cases",
        str(args.stage1_num_random_cases),
        "--case_scene_mode",
        "diverse",
        "--case_index_filter",
        "900",
        "901",
        "--prefer_existing_runtime_meshes",
        "--dt",
        str(args.stage1_dt),
        "--substeps",
        str(args.stage1_substeps),
        "--steps",
        str(args.stage1_steps),
        "--fps",
        str(args.stage1_fps),
        "--simulator_mode",
        "rigid",
        "--rigid_target_object_count",
        str(args.stage1_rigid_target_object_count),
        "--physxnet_volume_threshold_m3",
        str(args.stage1_physxnet_volume_threshold_m3),
        "--case_seed",
        str(args.stage1_case_seed),
        "--motion_case_max_retries",
        str(args.stage1_motion_case_max_retries),
        "--disable_striker",
    ]
    log_path = benchmark_root / "logs_regen_inertial_fix" / f"{object_id}.log"
    rc = _run(cmd, dry_run=bool(args.dry_run), log_path=log_path)

    if rc == 0 and not bool(args.dry_run):
        filter_cmd = [
            str(args.wan_python),
            str(args.filter_script),
            "--root",
            str(base),
            "--write_metrics",
        ]
        _run(filter_cmd, dry_run=False, log_path=benchmark_root / "logs_regen_inertial_fix" / f"{object_id}.qa.log")

    return {
        "benchmark_root": str(benchmark_root),
        "benchmark_type": "stage1_heldout",
        "object_id": str(object_id),
        "returncode": int(rc),
        "log_path": str(log_path),
    }


def regenerate_pool_object(
    *,
    benchmark_root: Path,
    object_entry: Dict,
    manifest: Dict,
    args: argparse.Namespace,
) -> Dict:
    object_id = str(object_entry["object_id"])
    for sample_dir in _iter_sample_dirs(object_entry.get("samples", [])):
        _remove_path(sample_dir, dry_run=bool(args.dry_run))

    work_root = Path(str(manifest.get("work_root", "")))
    for cache_candidate in [
        work_root / "_asset_cache" / "physxnet_objects" / object_id,
        work_root / object_id,
    ]:
        _remove_path(cache_candidate, dry_run=bool(args.dry_run))

    speed_cfg = dict(object_entry.get("speed_config", {}))
    case_indices = [str(int(idx)) for idx in object_entry.get("case_indices", [])]
    cmd = [
        str(args.wan_python),
        str(args.try1_script),
        "--physx_root",
        str(manifest["physx_root"]),
        "--version",
        str(manifest["version"]),
        "--object_id",
        object_id,
        "--output_root",
        str(benchmark_root),
        "--run_genesis",
        "--num_random_cases",
        str(manifest.get("cases_per_object", len(case_indices))),
        "--case_scene_mode",
        "diverse",
        "--case_index_filter",
        *case_indices,
        "--prefer_existing_runtime_meshes",
        "--dt",
        str(args.pool_dt),
        "--substeps",
        str(args.pool_substeps),
        "--ball_posx",
        str(args.pool_ball_posx),
        "--steps",
        str(args.pool_steps),
        "--fps",
        str(args.pool_fps),
        "--striker_speed",
        str(speed_cfg.get("striker_speed", args.pool_default_striker_speed)),
        "--physxnet_entry_speed_min",
        str(speed_cfg.get("entry_speed_min", args.pool_default_entry_speed_min)),
        "--physxnet_entry_speed_max",
        str(speed_cfg.get("entry_speed_max", args.pool_default_entry_speed_max)),
        "--simulator_mode",
        "rigid",
        "--rigid_target_object_count",
        str(args.pool_rigid_target_object_count),
        "--physxnet_volume_threshold_m3",
        str(args.pool_physxnet_volume_threshold_m3),
    ]
    if int(args.pool_rigid_target_object_count) <= 1:
        cmd.append("--disable_striker")

    log_path = benchmark_root / "logs_regen_inertial_fix" / f"{object_id}.log"
    rc = _run(cmd, dry_run=bool(args.dry_run), log_path=log_path)
    return {
        "benchmark_root": str(benchmark_root),
        "benchmark_type": "physxnet_pool",
        "object_id": object_id,
        "returncode": int(rc),
        "log_path": str(log_path),
        "case_indices": [int(idx) for idx in object_entry.get("case_indices", [])],
    }


def rebuild_stage1_subsets(benchmark_root: Path, args: argparse.Namespace) -> int:
    cmd = [
        str(args.wan_python),
        str(args.build_script),
        "--benchmark_root",
        str(benchmark_root),
        "--stage1_train_root",
        str(args.stage1_train_root),
        "--max_source_samples",
        str(args.max_source_samples),
        "--max_windows_per_subset",
        str(args.max_windows_per_subset),
        "--future_main_visibility_threshold",
        str(args.future_main_visibility_threshold),
        "--subset_count_buckets",
        str(args.subset_count_buckets),
    ]
    log_path = benchmark_root / "logs_regen_inertial_fix" / "build_stage1_subsets.log"
    return _run(cmd, dry_run=bool(args.dry_run), log_path=log_path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate affected benchmark samples after inertial-origin fixes.")
    parser.add_argument("--benchmark_root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--audit_report", type=Path, default=None, help="Optional precomputed audit report JSON.")
    parser.add_argument("--benchmark_subroot", type=Path, action="append", default=None, help="Restrict regeneration to specific benchmark roots.")
    parser.add_argument("--object_ids", nargs="*", default=None, help="Optional explicit object ids to regenerate.")
    parser.add_argument("--wan_python", type=Path, default=DEFAULT_WAN_PYTHON)
    parser.add_argument("--try1_script", type=Path, default=DEFAULT_TRY1_SCRIPT)
    parser.add_argument("--filter_script", type=Path, default=DEFAULT_FILTER_SCRIPT)
    parser.add_argument("--build_script", type=Path, default=DEFAULT_BUILD_SCRIPT)
    parser.add_argument("--physx_root", type=Path, default=Path("/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet"))
    parser.add_argument("--version", type=str, default="version_1")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--rebuild_stage1_subsets", action="store_true")

    parser.add_argument("--stage1_case_seed", type=int, default=20260421)
    parser.add_argument("--stage1_num_random_cases", type=int, default=12)
    parser.add_argument("--stage1_dt", type=float, default=0.003)
    parser.add_argument("--stage1_substeps", type=int, default=40)
    parser.add_argument("--stage1_steps", type=int, default=49)
    parser.add_argument("--stage1_fps", type=int, default=12)
    parser.add_argument("--stage1_rigid_target_object_count", type=int, default=1)
    parser.add_argument("--stage1_physxnet_volume_threshold_m3", type=float, default=999999.0)
    parser.add_argument("--stage1_motion_case_max_retries", type=int, default=8)
    parser.add_argument("--stage1_train_root", type=Path, default=DEFAULT_STAGE1_TRAIN_ROOT)
    parser.add_argument("--max_source_samples", type=int, default=0)
    parser.add_argument("--max_windows_per_subset", type=int, default=100)
    parser.add_argument("--future_main_visibility_threshold", type=float, default=0.5)
    parser.add_argument("--subset_count_buckets", type=str, default="count_01")

    parser.add_argument("--pool_dt", type=float, default=0.003)
    parser.add_argument("--pool_substeps", type=int, default=40)
    parser.add_argument("--pool_steps", type=int, default=12)
    parser.add_argument("--pool_fps", type=int, default=12)
    parser.add_argument("--pool_ball_posx", type=float, default=0.03)
    parser.add_argument("--pool_rigid_target_object_count", type=int, default=3)
    parser.add_argument("--pool_physxnet_volume_threshold_m3", type=float, default=999999.0)
    parser.add_argument("--pool_default_striker_speed", type=float, default=2.8)
    parser.add_argument("--pool_default_entry_speed_min", type=float, default=0.6)
    parser.add_argument("--pool_default_entry_speed_max", type=float, default=2.2)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    benchmark_root = Path(args.benchmark_root).resolve()
    if args.audit_report is not None and Path(args.audit_report).exists():
        audit_report = _load_json(Path(args.audit_report))
    else:
        audit_report = build_report(benchmark_root)

    selected_roots = [Path(path).resolve() for path in (args.benchmark_subroot or [])]
    if not selected_roots:
        selected_roots = [Path(item["benchmark_root"]).resolve() for item in audit_report.get("benchmark_roots", [])]

    explicit_ids = [str(item) for item in (args.object_ids or [])]
    regen_results: List[Dict] = []

    for root in selected_roots:
        benchmark_type = detect_benchmark_type(root)
        flagged_ids = _select_flagged_objects(audit_report, root, explicit_ids)
        if explicit_ids and not flagged_ids:
            flagged_ids = sorted(explicit_ids)
        if not flagged_ids:
            print(f"[skip] root={root} no flagged objects selected")
            continue

        print(f"[root] {root} type={benchmark_type} objects={flagged_ids}")
        if benchmark_type == "stage1_heldout":
            for object_id in flagged_ids:
                regen_results.append(regenerate_stage1_object(benchmark_root=root, object_id=object_id, args=args))
            if bool(args.rebuild_stage1_subsets):
                rc = rebuild_stage1_subsets(root, args)
                regen_results.append(
                    {
                        "benchmark_root": str(root),
                        "benchmark_type": benchmark_type,
                        "object_id": "__rebuild_stage1_subsets__",
                        "returncode": int(rc),
                    }
                )
        elif benchmark_type == "physxnet_pool":
            manifest = _load_json(root / "benchmark_manifest.json")
            entries = {str(item["object_id"]): item for item in manifest.get("objects", [])}
            for object_id in flagged_ids:
                entry = entries.get(object_id)
                if entry is None:
                    regen_results.append(
                        {
                            "benchmark_root": str(root),
                            "benchmark_type": benchmark_type,
                            "object_id": object_id,
                            "returncode": -1,
                            "error": "object_not_in_manifest",
                        }
                    )
                    continue
                regen_results.append(regenerate_pool_object(benchmark_root=root, object_entry=entry, manifest=manifest, args=args))
        else:
            for object_id in flagged_ids:
                regen_results.append(
                    {
                        "benchmark_root": str(root),
                        "benchmark_type": benchmark_type,
                        "object_id": object_id,
                        "returncode": -1,
                        "error": "unsupported_benchmark_type",
                    }
                )

    summary = {
        "benchmark_root": str(benchmark_root),
        "selected_roots": [str(path) for path in selected_roots],
        "results": regen_results,
        "success": sum(1 for item in regen_results if int(item.get("returncode", -1)) == 0),
        "failed": sum(1 for item in regen_results if int(item.get("returncode", -1)) != 0),
    }
    report_path = benchmark_root / "regen_inertial_fix_report.json"
    if not bool(args.dry_run):
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"success={summary['success']} failed={summary['failed']} report={report_path}")


if __name__ == "__main__":
    main()
