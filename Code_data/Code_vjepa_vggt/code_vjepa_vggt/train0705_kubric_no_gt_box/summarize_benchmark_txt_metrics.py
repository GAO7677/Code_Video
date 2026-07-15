from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


EXCLUDED_JSON_NAMES = {"summary.json", "result.json", "batch_manifest.json", "eval_summary.json"}


def nested_get(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


METRICS: tuple[tuple[str, str, Any], ...] = (
    ("wmreward_surprise_count", "wmreward_surprise_mean", lambda payload: to_float(nested_get(payload, "wmreward", "surprise"))),
    ("physics_iq_with_context_score_count", "physics_iq_with_context_score_mean", lambda payload: to_float(nested_get(payload, "physics_iq_with_context", "score"))),
    ("physics_iq_without_context_score_count", "physics_iq_without_context_score_mean", lambda payload: to_float(nested_get(payload, "physics_iq_without_context", "score"))),
    ("pmf_with_context_score_count", "pmf_with_context_score_mean", lambda payload: to_float(nested_get(payload, "pmf_with_context", "score"))),
    ("pmf_without_context_score_count", "pmf_without_context_score_mean", lambda payload: to_float(nested_get(payload, "pmf_without_context", "score"))),
    ("videophy2_score_count", "videophy2_score_mean", lambda payload: to_float(nested_get(payload, "videophy2", "score"))),
    ("phyground_general_avg_count", "phyground_general_avg_mean", lambda payload: to_float(nested_get(payload, "phyground", "general_avg"))),
    ("cosmos_reason1_score_count", "cosmos_reason1_score_mean", lambda payload: to_float(nested_get(payload, "cosmos_reason1", "score"))),
    ("vbench_subject_consistency_score_count", "vbench_subject_consistency_score_mean", lambda payload: to_float(nested_get(payload, "vbench_subject_consistency", "score"))),
    ("vbench_background_consistency_score_count", "vbench_background_consistency_score_mean", lambda payload: to_float(nested_get(payload, "vbench_background_consistency", "score"))),
    ("vbench_temporal_flickering_score_count", "vbench_temporal_flickering_score_mean", lambda payload: to_float(nested_get(payload, "vbench_temporal_flickering", "score"))),
    ("vbench_motion_smoothness_score_count", "vbench_motion_smoothness_score_mean", lambda payload: to_float(nested_get(payload, "vbench_motion_smoothness", "score"))),
    ("vbench_dynamic_degree_score_count", "vbench_dynamic_degree_score_mean", lambda payload: to_float(nested_get(payload, "vbench_dynamic_degree", "score"))),
    ("vbench_aesthetic_quality_score_count", "vbench_aesthetic_quality_score_mean", lambda payload: to_float(nested_get(payload, "vbench_aesthetic_quality", "score"))),
    ("vbench_imaging_quality_score_count", "vbench_imaging_quality_score_mean", lambda payload: to_float(nested_get(payload, "vbench_imaging_quality", "score"))),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize metric backfill results for result folders listed in a txt file."
    )
    parser.add_argument("--input-txt", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--input-json-allowlist", type=Path, default=None)
    return parser.parse_args()


def read_result_roots(list_path: Path) -> list[Path]:
    roots: list[Path] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        roots.append(Path(line).expanduser().resolve())
    return roots


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def is_result_payload(payload: dict[str, Any]) -> bool:
    return any(
        isinstance(payload.get(key), str) and bool(payload[key].strip())
        for key in ("input_json", "case_json")
    )


def discover_payloads(result_root: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    if not result_root.is_dir():
        return payloads
    for json_path in sorted(result_root.glob("*.json")):
        if json_path.name in EXCLUDED_JSON_NAMES or json_path.name.startswith("eval_summary_"):
            continue
        payload = load_json(json_path)
        if payload is None or not is_result_payload(payload):
            continue
        payloads.append(payload)
    return payloads


def resolve_payload_input_json(payload: dict[str, Any]) -> Path | None:
    for key in ("input_json", "case_json"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        return Path(value).expanduser().resolve()
    return None


def find_batch_manifest_path(result_root: Path) -> Path | None:
    # Parent manifests are shared by multiple method directories and can be
    # overwritten by an unrelated partial or single-case inference run.
    candidate = result_root / "batch_manifest.json"
    return candidate if candidate.is_file() else None


def read_reference_input_jsons(result_root: Path) -> list[Path] | None:
    manifest_path = find_batch_manifest_path(result_root)
    if manifest_path is None:
        return None
    manifest_payload = load_json(manifest_path)
    if manifest_payload is None:
        return None
    input_json_list_path = manifest_payload.get("input_json_list_path")
    if not isinstance(input_json_list_path, str) or not input_json_list_path.strip():
        return None
    list_path = Path(input_json_list_path).expanduser().resolve()
    if not list_path.is_file():
        return None

    reference_paths: list[Path] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        reference_paths.append(Path(line).expanduser().resolve())
    return reference_paths


def infer_method_name(result_root: Path, payloads: list[dict[str, Any]]) -> str:
    values = []
    for payload in payloads:
        value = payload.get("method")
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    unique = sorted(set(values))
    if unique:
        return unique[0]
    return result_root.name


def build_row(
    result_root: Path,
    payloads: list[dict[str, Any]],
    allowed_input_jsons: set[Path] | None = None,
) -> dict[str, Any]:
    if allowed_input_jsons is not None:
        payloads = [
            payload
            for payload in payloads
            if (resolved_input_json := resolve_payload_input_json(payload)) is not None
            and resolved_input_json in allowed_input_jsons
        ]
    reference_input_jsons = read_reference_input_jsons(result_root)
    if reference_input_jsons is not None:
        reference_set = set(reference_input_jsons)
        payloads = [
            payload
            for payload in payloads
            if (resolved_input_json := resolve_payload_input_json(payload)) is not None
            and resolved_input_json in reference_set
        ]
    num_json = len(payloads)

    row: dict[str, Any] = {
        "method": infer_method_name(result_root, payloads),
        "result_root": str(result_root),
        "num_json": num_json,
    }
    for count_column, mean_column, extractor in METRICS:
        values: list[float] = []
        for payload in payloads:
            value = extractor(payload)
            if value is not None:
                values.append(value)
        row[count_column] = len(values)
        row[mean_column] = round(sum(values) / len(values), 4) if values else 0.0
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "result_root", "num_json"]
    for count_column, mean_column, _ in METRICS:
        fieldnames.append(count_column)
        fieldnames.append(mean_column)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    input_txt = args.input_txt.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    allowed_input_jsons = None
    if args.input_json_allowlist is not None:
        allowlist_path = args.input_json_allowlist.expanduser().resolve()
        allowed_input_jsons = {
            Path(line.strip()).expanduser().resolve()
            for line in allowlist_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

    rows: list[dict[str, Any]] = []
    for result_root in read_result_roots(input_txt):
        payloads = discover_payloads(result_root)
        rows.append(build_row(result_root, payloads, allowed_input_jsons))

    write_csv(output_csv, rows)
    print(
        json.dumps(
            {
                "input_txt": str(input_txt),
                "output_csv": str(output_csv),
                "num_rows": len(rows),
                "allowlist_size": None if allowed_input_jsons is None else len(allowed_input_jsons),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
