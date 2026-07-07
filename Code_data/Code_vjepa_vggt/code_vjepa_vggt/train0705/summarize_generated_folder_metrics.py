from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_RESULT_ROOTS = (
    Path("/data/gaoya/AAA_test_video/0623/test/v2v"),
    Path("/data/gaoya/AAA_test_video/0623/test/ti2v"),
    Path("/data/gaoya/AAA_test_video/0623/test/t2v"),
)
DEFAULT_DATASET_LISTS = (
    Path("/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt"),
    Path("/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_morpheus_real_world.txt"),
)
DEFAULT_OUTPUT_MD = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/AAAresults/generated_folder_metric_summary.md"
)
DEFAULT_OUTPUT_CSV = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/AAAresults/generated_folder_metric_summary.csv"
)
EXCLUDED_JSON_NAMES = {"summary.json", "result.json", "batch_manifest.json", "eval_summary.json"}


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    list_path: Path
    case_paths: set[str]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    mean_column: str
    count_column: str
    extractor: Callable[[dict[str, Any]], float | None]


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


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("pdi_score", "pdi_score_mean", "pdi_score_count", lambda payload: to_float(nested_get(payload, "pdi", "pdi_score"))),
    MetricSpec(
        "wmreward_surprise",
        "wmreward_surprise_mean",
        "wmreward_surprise_count",
        lambda payload: to_float(nested_get(payload, "wmreward", "surprise")),
    ),
    MetricSpec(
        "proxy_relraw",
        "proxy_relraw_mean",
        "proxy_relraw_count",
        lambda payload: to_float(nested_get(payload, "proxy", "details", "temporal_relation_raw_error")),
    ),
    MetricSpec(
        "proxy_deltarel",
        "proxy_deltarel_mean",
        "proxy_deltarel_count",
        lambda payload: to_float(nested_get(payload, "proxy", "details", "delta_relation_raw_error")),
    ),
    MetricSpec(
        "proxy_deltaprof",
        "proxy_deltaprof_mean",
        "proxy_deltaprof_count",
        lambda payload: to_float(nested_get(payload, "proxy", "details", "delta_profile_error")),
    ),
    MetricSpec(
        "physics_iq_with_context_score",
        "physics_iq_with_context_score_mean",
        "physics_iq_with_context_score_count",
        lambda payload: to_float(nested_get(payload, "physics_iq_with_context", "score")),
    ),
    MetricSpec(
        "physics_iq_without_context_score",
        "physics_iq_without_context_score_mean",
        "physics_iq_without_context_score_count",
        lambda payload: to_float(nested_get(payload, "physics_iq_without_context", "score")),
    ),
    MetricSpec(
        "pmf_with_context_score",
        "pmf_with_context_score_mean",
        "pmf_with_context_score_count",
        lambda payload: to_float(nested_get(payload, "pmf_with_context", "score")),
    ),
    MetricSpec(
        "pmf_without_context_score",
        "pmf_without_context_score_mean",
        "pmf_without_context_score_count",
        lambda payload: to_float(nested_get(payload, "pmf_without_context", "score")),
    ),
    MetricSpec(
        "videophy2_score",
        "videophy2_score_mean",
        "videophy2_score_count",
        lambda payload: to_float(nested_get(payload, "videophy2", "score")),
    ),
    MetricSpec(
        "phyground_general_avg",
        "phyground_general_avg_mean",
        "phyground_general_avg_count",
        lambda payload: to_float(nested_get(payload, "phyground", "general_avg")),
    ),
    MetricSpec(
        "cosmos_reason1_score",
        "cosmos_reason1_score_mean",
        "cosmos_reason1_score_count",
        lambda payload: to_float(nested_get(payload, "cosmos_reason1", "score")),
    ),
)

CSV_EXCLUDED_METRIC_KEYS = {
    "pdi_score",
    "proxy_relraw",
    "proxy_deltarel",
    "proxy_deltaprof",
    "phyground_general_avg",
}
CSV_METRICS: tuple[MetricSpec, ...] = tuple(metric for metric in METRICS if metric.key not in CSV_EXCLUDED_METRIC_KEYS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize all generated result folders under v2v/ti2v/t2v whose per-dataset generation "
            "count matches the target dataset list size, and report metric mean/count per folder."
        )
    )
    parser.add_argument("--result-root", dest="result_roots", action="append", type=Path, default=[])
    parser.add_argument("--dataset-list", dest="dataset_lists", action="append", type=Path, default=[])
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def resolve_path_string(path_str: str) -> str:
    return str(Path(path_str).expanduser().resolve())


def read_list_paths(list_path: Path) -> set[str]:
    paths: set[str] = set()
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        paths.add(resolve_path_string(line))
    return paths


def build_dataset_specs(list_paths: list[Path]) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    for list_path in list_paths:
        resolved = list_path.expanduser().resolve()
        case_paths = read_list_paths(resolved)
        label = resolved.stem
        if label.startswith("v2v_jsons_"):
            label = label[len("v2v_jsons_") :]
        specs.append(DatasetSpec(label=label, list_path=resolved, case_paths=case_paths))
    return specs


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def discover_folder_payloads(result_roots: list[Path]) -> dict[Path, list[dict[str, Any]]]:
    folders: dict[Path, list[dict[str, Any]]] = {}
    for result_root in result_roots:
        resolved_root = result_root.expanduser().resolve()
        if not resolved_root.is_dir():
            continue
        for json_path in sorted(resolved_root.rglob("*.json")):
            if json_path.name in EXCLUDED_JSON_NAMES or json_path.name.startswith("eval_summary_"):
                continue
            payload = load_json(json_path)
            if payload is None:
                continue
            input_json = payload.get("input_json")
            if not isinstance(input_json, str) or not input_json.strip():
                input_json = payload.get("case_json")
            if not isinstance(input_json, str) or not input_json.strip():
                continue
            payload["_result_json_path"] = str(json_path.resolve())
            payload["_input_json_resolved"] = resolve_path_string(input_json)
            folder = json_path.parent.resolve()
            folders.setdefault(folder, []).append(payload)
    return folders


def infer_root_kind(folder: Path, result_roots: list[Path]) -> str:
    for result_root in result_roots:
        resolved_root = result_root.expanduser().resolve()
        try:
            folder.relative_to(resolved_root)
        except ValueError:
            continue
        return resolved_root.name
    return "<unknown>"


def infer_method_name(folder_payloads: list[dict[str, Any]], folder: Path) -> str:
    method_names = []
    for payload in folder_payloads:
        value = payload.get("method")
        if isinstance(value, str) and value.strip():
            method_names.append(value.strip())
    if method_names:
        unique = sorted(set(method_names))
        if len(unique) == 1:
            return unique[0]
        return unique[0]
    return folder.name


def mean_or_zero_if_incomplete(values: list[float], expected_count: int) -> float:
    if len(values) != expected_count or expected_count <= 0:
        return 0.0
    return float(sum(values) / len(values))


def build_rows(
    folder_payloads: dict[Path, list[dict[str, Any]]],
    result_roots: list[Path],
    dataset_specs: list[DatasetSpec],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for folder, payloads in sorted(folder_payloads.items()):
        method_name = infer_method_name(payloads, folder)
        root_kind = infer_root_kind(folder, result_roots)
        for dataset in dataset_specs:
            matched_payloads = [
                payload for payload in payloads if payload.get("_input_json_resolved") in dataset.case_paths
            ]
            expected_count = len(dataset.case_paths)
            generated_count = len(matched_payloads)
            if generated_count != expected_count:
                continue

            row: dict[str, Any] = {
                "dataset_label": dataset.label,
                "dataset_list_path": str(dataset.list_path),
                "dataset_size": expected_count,
                "root_kind": root_kind,
                "method": method_name,
                "folder_path": str(folder),
                "generated_count": generated_count,
            }
            for metric in METRICS:
                values = []
                for payload in matched_payloads:
                    value = metric.extractor(payload)
                    if value is not None:
                        values.append(value)
                row[metric.count_column] = len(values)
                row[metric.mean_column] = round(mean_or_zero_if_incomplete(values, expected_count), 4)
            rows.append(row)
    rows.sort(key=lambda row: (str(row["dataset_label"]), str(row["root_kind"]), str(row["method"]), str(row["folder_path"])))
    return rows


def format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "0.0000"


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "method",
        "dataset_label",
        "dataset_list_path",
        "dataset_size",
        "root_kind",
        "folder_path",
        "generated_count",
    ]
    for metric in CSV_METRICS:
        fieldnames.append(metric.count_column)
        fieldnames.append(metric.mean_column)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, rows: list[dict[str, Any]], dataset_specs: list[DatasetSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Generated Folder Metric Summary", ""]
    for dataset in dataset_specs:
        dataset_rows = [row for row in rows if row["dataset_label"] == dataset.label]
        lines.append(f"## {dataset.label}")
        lines.append("")
        lines.append(f"- dataset_list_path: `{dataset.list_path}`")
        lines.append(f"- dataset_size: `{len(dataset.case_paths)}`")
        lines.append(f"- qualified_folders: `{len(dataset_rows)}`")
        lines.append("")
        if not dataset_rows:
            lines.append("No qualified folders found.")
            lines.append("")
            continue

        headers = ["method", "root_kind", "generated_count", "folder_path"]
        for metric in METRICS:
            headers.append(metric.mean_column)
            headers.append(metric.count_column)

        table_rows: list[list[str]] = []
        for row in dataset_rows:
            table_row = [
                str(row["method"]),
                str(row["root_kind"]),
                str(row["generated_count"]),
                str(row["folder_path"]),
            ]
            for metric in METRICS:
                table_row.append(format_metric(row[metric.mean_column]))
                table_row.append(str(row[metric.count_column]))
            table_rows.append(table_row)
        lines.append(markdown_table(headers, table_rows))
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_roots = [path.expanduser().resolve() for path in (args.result_roots or list(DEFAULT_RESULT_ROOTS))]
    dataset_lists = [path.expanduser().resolve() for path in (args.dataset_lists or list(DEFAULT_DATASET_LISTS))]

    dataset_specs = build_dataset_specs(dataset_lists)
    folder_payloads = discover_folder_payloads(result_roots)
    rows = build_rows(folder_payloads, result_roots, dataset_specs)

    write_csv_rows(args.output_csv.expanduser().resolve(), rows)
    write_markdown(args.output_md.expanduser().resolve(), rows, dataset_specs)

    summary = {
        "result_roots": [str(path) for path in result_roots],
        "dataset_lists": [str(path) for path in dataset_lists],
        "output_md": str(args.output_md.expanduser().resolve()),
        "output_csv": str(args.output_csv.expanduser().resolve()),
        "num_rows": len(rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
