from __future__ import annotations

# Run command example:
# /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
# /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/collect_stage1b_metric_table.py \
#   --report-csv /data/gaoya/AAA_test_video/0623/test/report/v2v/method_summary.csv \
#   --result-substring /train_stage1b_diffsynth_native0705_0705/step- \
#   --output-csv /data/gaoya/AAA_test_video/0623/test/report/v2v/train_stage1b_diffsynth_native0705_0705_metric_table.csv \
#   --output-md /data/gaoya/AAA_test_video/0623/test/report/v2v/train_stage1b_diffsynth_native0705_0705_metric_table.md

import argparse
import csv
import re
from pathlib import Path


DEFAULT_REPORT_CSV = Path("/data/gaoya/AAA_test_video/0623/test/report/v2v/method_summary.csv")
DEFAULT_RESULT_SUBSTRING = "/train_stage1b_diffsynth_native0705_0705/step-"

METRIC_COLUMNS = (
    "wmreward_surprise_mean",
    "physics_iq_score_mean",
    "videophy2_score_mean",
    "phyground_general_avg_mean",
    "cosmos_reason1_score_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a compact per-step metric table from render_v2v_metric_report.py output."
        )
    )
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--result-substring", default=DEFAULT_RESULT_SUBSTRING)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def step_sort_key(step_name: str) -> tuple[int, str]:
    match = re.search(r"step-(\d+)", step_name)
    if match is None:
        return (10**12, step_name)
    return (int(match.group(1)), step_name)


def to_display(value: str) -> str:
    if value is None:
        return "-"
    stripped = str(value).strip()
    if not stripped:
        return "-"
    try:
        numeric = float(stripped)
    except ValueError:
        return stripped
    return f"{numeric:.4f}"


def to_raw(value: str) -> str:
    if value is None:
        return ""
    stripped = str(value).strip()
    return stripped


def load_rows(report_csv: Path, result_substring: str) -> list[dict[str, str]]:
    with report_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    filtered = [
        row
        for row in rows
        if result_substring in str(row.get("result_dir", ""))
    ]
    filtered.sort(key=lambda row: step_sort_key(str(row.get("method", ""))))
    return filtered


def build_table_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    table_rows: list[dict[str, str]] = []
    for row in rows:
        table_rows.append(
            {
                "method": str(row.get("method", "")).strip(),
                "dataset_label": str(row.get("dataset_label", "")).strip(),
                "num_cases": to_raw(row.get("num_cases", "")),
                "wmreward_surprise": to_display(row.get("wmreward_surprise_mean", "")),
                "physics_iq": to_display(row.get("physics_iq_score_mean", "")),
                "videophy2": to_display(row.get("videophy2_score_mean", "")),
                "phyground": to_display(row.get("phyground_general_avg_mean", "")),
                "cosmos_reason1": to_display(row.get("cosmos_reason1_score_mean", "")),
                "result_dir": str(row.get("result_dir", "")).strip(),
            }
        )
    return table_rows


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "method",
        "dataset_label",
        "num_cases",
        "wmreward_surprise",
        "physics_iq",
        "videophy2",
        "phyground",
        "cosmos_reason1",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.report_csv = args.report_csv.expanduser().resolve()
    if not args.report_csv.is_file():
        raise FileNotFoundError(f"report csv not found: {args.report_csv}")

    rows = load_rows(args.report_csv, str(args.result_substring))
    if not rows:
        raise RuntimeError(
            f"no rows matched result_substring={args.result_substring!r} in {args.report_csv}"
        )
    table_rows = build_table_rows(rows)

    if args.output_csv is not None:
        write_csv_rows(args.output_csv.expanduser().resolve(), table_rows)
    if args.output_md is not None:
        write_markdown(args.output_md.expanduser().resolve(), table_rows)

    headers = [
        "method",
        "dataset_label",
        "num_cases",
        "wmreward_surprise",
        "physics_iq",
        "videophy2",
        "phyground",
        "cosmos_reason1",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in table_rows:
        print("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")


if __name__ == "__main__":
    main()
