#!/usr/bin/env python3
"""Build and verify the full remaining-block ablation pipeline."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ALL_BLOCKS = tuple(range(30))
EXISTING_BLOCKS = {0, 5, 11, 17, 19, 29}
REMAINING_BLOCKS = tuple(block for block in ALL_BLOCKS if block not in EXISTING_BLOCKS)
MODEL_MODES = {
    "wan_lora": ("whole_block", "self_attn_zero"),
    "xssc": ("whole_block", "self_attn_zero", "object_cross_attn"),
    "physrvg": (
        "whole_block",
        "self_attn_zero",
        "text_cross_attn_zero",
        "ffn_zero",
        "lora_off",
    ),
}
METRIC_GROUPS = {
    "cpu": (
        "physics_iq_with_context",
        "physics_iq_without_context",
        "physics_iq_verified_proxy",
        "pmf_with_context",
        "pmf_without_context",
    ),
    "gpu_common": (
        "wmreward",
        "vbench_subject_consistency",
        "vbench_background_consistency",
        "vbench_temporal_flickering",
        "vbench_motion_smoothness",
        "vbench_dynamic_degree",
        "vbench_aesthetic_quality",
        "vbench_imaging_quality",
    ),
    "videophy2": ("videophy2",),
    "cosmos": ("cosmos_reason1",),
}
REQUIRED_FIELDS = {
    "physics_iq_with_context": ("score",),
    "physics_iq_without_context": ("score",),
    "physics_iq_verified_proxy": ("score",),
    "pmf_with_context": ("score",),
    "pmf_without_context": ("score",),
    "wmreward": ("surprise",),
    "videophy2": ("sa_score", "pc_score", "joint_pass", "pc_raw_score"),
    "cosmos_reason1": ("score",),
    "vbench_subject_consistency": ("score",),
    "vbench_background_consistency": ("score",),
    "vbench_temporal_flickering": ("score",),
    "vbench_motion_smoothness": ("score",),
    "vbench_dynamic_degree": ("score",),
    "vbench_aesthetic_quality": ("score",),
    "vbench_imaging_quality": ("score",),
}
IGNORED_JSON_NAMES = {
    "summary.json",
    "result.json",
    "batch_manifest.json",
    "eval_summary.json",
}


def read_paths(path: Path) -> list[Path]:
    return [
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def input_json_path(payload: dict[str, Any]) -> Path | None:
    value = payload.get("input_json") or payload.get("case_json")
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser().resolve()


def result_payloads(directory: Path, allowed: set[Path]) -> dict[Path, dict[str, Any]]:
    payloads: dict[Path, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name in IGNORED_JSON_NAMES or path.name.startswith("eval_summary_"):
            continue
        payload = load_json(path)
        if payload is None:
            continue
        source = input_json_path(payload)
        if source is not None and source in allowed:
            payloads[source] = payload
    return payloads


def find_leaf(config_root: Path, allowed: set[Path]) -> Path:
    candidates: list[Path] = []
    for directory in (config_root, *sorted(path for path in config_root.rglob("*") if path.is_dir())):
        if len(result_payloads(directory, allowed)) == len(allowed):
            candidates.append(directory)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one {len(allowed)}-case leaf below {config_root}, "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def model_root(output_base: Path, model: str) -> Path:
    return output_base / ("PhyRVG" if model == "physrvg" else model)


def config_root(output_base: Path, model: str, mode: str, block: int | None) -> Path:
    if mode == "baseline":
        tag = "baseline"
    else:
        if block is None:
            raise ValueError(f"Missing block for {model}/{mode}")
        tag = f"{mode}_block{block:02d}"
    return model_root(output_base, model) / tag


def expected_configs(include_baseline: bool) -> list[tuple[str, str, int | None]]:
    configs: list[tuple[str, str, int | None]] = []
    for model, modes in MODEL_MODES.items():
        if include_baseline:
            configs.append((model, "baseline", None))
        for block in ALL_BLOCKS:
            for mode in modes:
                configs.append((model, mode, block))
    return configs


def metric_is_complete(payload: dict[str, Any], metric: str) -> bool:
    value = payload.get(metric)
    if not isinstance(value, dict):
        return False
    for field in REQUIRED_FIELDS[metric]:
        field_value = value.get(field)
        if isinstance(field_value, bool):
            if metric == "videophy2" and field == "joint_pass":
                continue
            return False
        if not isinstance(field_value, (int, float)) or not math.isfinite(float(field_value)):
            return False
    return True


def build_generation(args: argparse.Namespace) -> None:
    rows: list[str] = []
    jobs: list[dict[str, Any]] = []
    index = 0
    for model, modes in MODEL_MODES.items():
        for block in REMAINING_BLOCKS:
            for mode in modes:
                root = config_root(args.output_base, model, mode, block)
                rows.append(
                    f"gen-{index:04d}\t{model}\t{mode}\t{block}\t{root}\n"
                )
                jobs.append(
                    {
                        "task_id": f"gen-{index:04d}",
                        "model": model,
                        "mode": mode,
                        "block": block,
                        "config_root": str(root),
                    }
                )
                index += 1
    if index != 240:
        raise RuntimeError(f"Expected 240 generation jobs, got {index}")
    args.queue.parent.mkdir(parents=True, exist_ok=True)
    args.queue.write_text("".join(rows), encoding="utf-8")
    report = {
        "num_jobs": index,
        "remaining_blocks": list(REMAINING_BLOCKS),
        "existing_blocks": sorted(EXISTING_BLOCKS),
        "jobs": jobs,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"num_jobs": index, "remaining_blocks": list(REMAINING_BLOCKS)}, indent=2))


def validate_config(args: argparse.Namespace) -> None:
    allowed = set(read_paths(args.input_list))
    leaf = find_leaf(args.config_root, allowed)
    payloads = result_payloads(leaf, allowed)
    failures: list[dict[str, Any]] = []
    for source, payload in payloads.items():
        output_video = payload.get("output_video")
        if not isinstance(output_video, str) or not Path(output_video).expanduser().is_file():
            failures.append({"input_json": str(source), "error": "missing_output_video"})
    report = {
        "complete": not failures and len(payloads) == len(allowed),
        "config_root": str(args.config_root.resolve()),
        "leaf_root": str(leaf),
        "expected_cases": len(allowed),
        "matched_cases": len(payloads),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "complete", "leaf_root", "expected_cases", "matched_cases"
    )}, indent=2))
    if not report["complete"]:
        raise SystemExit(1)


def prepare_metrics(args: argparse.Namespace) -> None:
    allowed = set(read_paths(args.input_list))
    roots: list[Path] = []
    new_roots: list[Path] = []
    config_records: list[dict[str, Any]] = []
    for model, mode, block in expected_configs(include_baseline=True):
        root = config_root(args.output_base, model, mode, block)
        leaf = find_leaf(root, allowed)
        roots.append(leaf)
        if block in REMAINING_BLOCKS:
            new_roots.append(leaf)
        config_records.append(
            {
                "model": model,
                "mode": mode,
                "block": block,
                "config_root": str(root),
                "leaf_root": str(leaf),
            }
        )
    if len(roots) != 303 or len(set(roots)) != 303 or len(new_roots) != 240:
        raise RuntimeError(
            f"Unexpected root counts: all={len(roots)} unique={len(set(roots))} "
            f"new={len(new_roots)}"
        )

    args.all_roots.write_text(
        "".join(f"{root}\n" for root in roots), encoding="utf-8"
    )
    args.new_roots.write_text(
        "".join(f"{root}\n" for root in new_roots), encoding="utf-8"
    )
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    task_index = 0
    group_counts: dict[str, int] = {}
    for group, metrics in METRIC_GROUPS.items():
        rows: list[str] = []
        for root in roots:
            for metric in metrics:
                rows.append(
                    f"metric-{task_index:05d}\t{metric}\t{root}\n"
                )
                task_index += 1
        (args.queue_dir / f"{group}.tsv").write_text(
            "".join(rows), encoding="utf-8"
        )
        group_counts[group] = len(rows)
    report = {
        "num_all_roots": len(roots),
        "num_new_roots": len(new_roots),
        "num_metric_tasks": task_index,
        "metric_group_counts": group_counts,
        "configs": config_records,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "num_all_roots", "num_new_roots", "num_metric_tasks", "metric_group_counts"
    )}, indent=2))


def build_retry(args: argparse.Namespace) -> None:
    allowed = set(read_paths(args.input_list))
    roots = read_paths(args.all_roots)
    rows: list[str] = []
    missing_cases = 0
    metric_counts: Counter[str] = Counter()
    for root in roots:
        payloads = result_payloads(root, allowed)
        if len(payloads) != len(allowed):
            raise ValueError(f"{root}: expected {len(allowed)} payloads, got {len(payloads)}")
        for metric in REQUIRED_FIELDS:
            missing = sum(
                1 for payload in payloads.values()
                if not metric_is_complete(payload, metric)
            )
            if missing:
                task_id = f"retry-{len(rows):05d}"
                rows.append(f"{task_id}\t{metric}\t{root}\n")
                missing_cases += missing
                metric_counts[metric] += 1
    args.queue.write_text("".join(rows), encoding="utf-8")
    report = {
        "num_retry_tasks": len(rows),
        "num_missing_case_metrics": missing_cases,
        "metric_task_counts": dict(sorted(metric_counts.items())),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def verify_all(args: argparse.Namespace) -> None:
    allowed = set(read_paths(args.input_list))
    roots = read_paths(args.all_roots)
    failures: list[dict[str, Any]] = []
    verified = Counter()
    for root in roots:
        payloads = result_payloads(root, allowed)
        if len(payloads) != len(allowed):
            failures.append(
                {
                    "root": str(root),
                    "error": "case_count",
                    "expected": len(allowed),
                    "found": len(payloads),
                }
            )
            continue
        for metric in REQUIRED_FIELDS:
            missing = [
                str(source) for source, payload in payloads.items()
                if not metric_is_complete(payload, metric)
            ]
            if missing:
                failures.append(
                    {
                        "root": str(root),
                        "metric": metric,
                        "error": "missing_or_invalid_metric",
                        "count": len(missing),
                        "examples": missing[:5],
                    }
                )
            else:
                verified[metric] += len(payloads)
    expected_per_metric = len(roots) * len(allowed)
    report = {
        "complete": not failures,
        "num_roots": len(roots),
        "expected_cases_per_root": len(allowed),
        "expected_records_per_metric": expected_per_metric,
        "verified_records": dict(sorted(verified.items())),
        "failures": failures,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "complete": report["complete"],
        "num_roots": report["num_roots"],
        "expected_records_per_metric": expected_per_metric,
        "num_failures": len(failures),
    }, indent=2))
    if failures:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-generation")
    build.add_argument("--output-base", type=Path, required=True)
    build.add_argument("--queue", type=Path, required=True)
    build.add_argument("--report", type=Path, required=True)
    build.set_defaults(func=build_generation)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config-root", type=Path, required=True)
    validate.add_argument("--input-list", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.set_defaults(func=validate_config)

    prepare = subparsers.add_parser("prepare-metrics")
    prepare.add_argument("--output-base", type=Path, required=True)
    prepare.add_argument("--input-list", type=Path, required=True)
    prepare.add_argument("--all-roots", type=Path, required=True)
    prepare.add_argument("--new-roots", type=Path, required=True)
    prepare.add_argument("--queue-dir", type=Path, required=True)
    prepare.add_argument("--report", type=Path, required=True)
    prepare.set_defaults(func=prepare_metrics)

    retry = subparsers.add_parser("build-retry")
    retry.add_argument("--all-roots", type=Path, required=True)
    retry.add_argument("--input-list", type=Path, required=True)
    retry.add_argument("--queue", type=Path, required=True)
    retry.add_argument("--report", type=Path, required=True)
    retry.set_defaults(func=build_retry)

    verify = subparsers.add_parser("verify-all")
    verify.add_argument("--all-roots", type=Path, required=True)
    verify.add_argument("--input-list", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.set_defaults(func=verify_all)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
