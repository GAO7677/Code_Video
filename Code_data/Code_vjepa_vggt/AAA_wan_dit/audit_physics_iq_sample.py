#!/usr/bin/env python3
"""Recompute a deterministic stratified sample of Physics-IQ records read-only."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
TRY0526_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
BENCH_PATH = REPO_ROOT / "code_vjepa_vggt/AAAinfer/bench.py"
DEFAULT_ROOTS = REPO_ROOT / "code_vjepa_vggt/train0705_kubric_no_gt_box/AAAevalphysiq.txt"
DEFAULT_ALLOWLIST = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/aaaevalphysiq_metric_code_audit/"
    "physics_iq_random100_recompute.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-roots", type=Path, default=DEFAULT_ROOTS)
    parser.add_argument("--input-json-allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_paths(path: Path) -> list[Path]:
    return [
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def load_bench_module() -> Any:
    spec = importlib.util.spec_from_file_location("aaainfer_bench_audit", BENCH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load bench module: {BENCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stratified_sample(
    records_by_root: list[tuple[Path, list[Any]]], sample_size: int, seed: int
) -> list[tuple[Path, Any]]:
    rng = random.Random(seed)
    nonempty = [(root, records) for root, records in records_by_root if records]
    if sample_size < len(nonempty):
        raise ValueError(
            f"sample_size={sample_size} cannot cover all {len(nonempty)} non-empty roots"
        )

    selected: list[tuple[Path, Any]] = []
    remaining: list[tuple[Path, Any]] = []
    base_count = sample_size // len(nonempty)
    for root, records in nonempty:
        shuffled = list(records)
        rng.shuffle(shuffled)
        take = min(base_count, len(shuffled))
        selected.extend((root, record) for record in shuffled[:take])
        remaining.extend((root, record) for record in shuffled[take:])

    rng.shuffle(remaining)
    selected.extend(remaining[: sample_size - len(selected)])
    rng.shuffle(selected)
    if len(selected) != sample_size:
        raise ValueError(f"Only {len(selected)} eligible records for sample_size={sample_size}")
    return selected


def scalar(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("score")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def main() -> None:
    args = parse_args()
    for path in (REPO_ROOT, TRY0526_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    bench = load_bench_module()
    from physv_eval.single_case import physics_iq

    # Writing aligned videos is presentation-only and does not enter the score.
    physics_iq._write_video = lambda *unused_args, **unused_kwargs: None

    roots = read_paths(args.result_roots.expanduser().resolve())
    allowed_inputs = set(read_paths(args.input_json_allowlist.expanduser().resolve()))
    records_by_root: list[tuple[Path, list[Any]]] = []
    prepare_errors: list[dict[str, Any]] = []
    for root in roots:
        records, errors = bench.prepare_cases(root)
        records = [record for record in records if record.input_json_path in allowed_inputs]
        records_by_root.append((root, records))
        prepare_errors.extend({"result_root": str(root), **error} for error in errors)

    sample = stratified_sample(records_by_root, args.sample_size, args.seed)
    comparisons: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, (root, record) in enumerate(sample, start=1):
        try:
            case, context_frames = bench.build_context_metric_case_payload(record)
            row: dict[str, Any] = {
                "sample_index": index,
                "result_root": str(root),
                "result_json": str(record.result_json_path),
                "input_json": str(record.input_json_path),
                "candidate_video": str(record.candidate_video_path),
                "source_video": str(record.gt_video_path),
                "context_frames_override": context_frames,
                "metrics": {},
            }
            for field, mode in (
                ("physics_iq_with_context", "with_context"),
                ("physics_iq_without_context", "without_context"),
            ):
                current = physics_iq.score_case(
                    case,
                    source_video_path=record.gt_video_path,
                    context_mode=mode,
                    context_frames=context_frames,
                    threshold_value=10,
                    downsample_factor=4,
                    aligned_video_dir=args.output.parent / "unused_aligned" / str(index) / field,
                )
                old_payload = record.result_payload.get(field)
                old_score = scalar(old_payload)
                new_score = scalar(current)
                difference = None if old_score is None or new_score is None else new_score - old_score
                row["metrics"][field] = {
                    "stored_score": old_score,
                    "recomputed_score": new_score,
                    "difference": difference,
                    "exact_match": difference == 0.0,
                    "stored_payload": old_payload,
                    "recomputed_payload": current,
                }
            comparisons.append(row)
            print(f"[{index:03d}/{len(sample):03d}] {record.result_json_path.name}", flush=True)
        except Exception as exc:
            failures.append(
                {
                    "sample_index": index,
                    "result_root": str(root),
                    "result_json": str(record.result_json_path),
                    "error": repr(exc),
                }
            )
            print(f"[{index:03d}/{len(sample):03d}] ERROR {record.result_json_path}: {exc}", flush=True)

    metric_summary: dict[str, Any] = {}
    for field in ("physics_iq_with_context", "physics_iq_without_context"):
        values = [row["metrics"][field] for row in comparisons]
        diffs = [abs(item["difference"]) for item in values if item["difference"] is not None]
        metric_summary[field] = {
            "num_compared": len(diffs),
            "num_exact_match": sum(item["exact_match"] for item in values),
            "num_mismatch": sum(not item["exact_match"] for item in values),
            "max_abs_difference": max(diffs, default=None),
            "mean_abs_difference": sum(diffs) / len(diffs) if diffs else None,
        }

    output = {
        "protocol": {
            "result_roots_file": str(args.result_roots.expanduser().resolve()),
            "input_json_allowlist": str(args.input_json_allowlist.expanduser().resolve()),
            "sample_size": args.sample_size,
            "seed": args.seed,
            "sampling": "stratified_random_covering_every_result_root",
            "threshold_value": 10,
            "downsample_factor": 4,
            "score_implementation": str(
                TRY0526_ROOT / "physv_eval/single_case/physics_iq.py"
            ),
            "aligned_video_write_disabled": True,
        },
        "population": {
            "num_roots": len(roots),
            "num_eligible_records": sum(len(records) for _, records in records_by_root),
            "eligible_by_root": {
                str(root): len(records) for root, records in records_by_root
            },
        },
        "sample_by_root": dict(Counter(str(root) for root, _ in sample)),
        "summary": metric_summary,
        "num_failures": len(failures),
        "prepare_errors": prepare_errors,
        "failures": failures,
        "comparisons": comparisons,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metric_summary, ensure_ascii=False, indent=2))
    print(f"report={output_path}")


if __name__ == "__main__":
    main()
