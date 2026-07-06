from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS = (
    "physics_iq",
    "videophy2",
    "wmreward",
    "cosmos_reason1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill selected metric payloads from per-case JSON files into "
            "batch result.json entries under one prediction root."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Prediction root containing per-directory result.json files and per-case JSON files.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric field names copied from per-case JSON into result.json entries.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def backfill_result_json(result_path: Path, metrics: tuple[str, ...]) -> dict[str, Any]:
    payload = load_json(result_path)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{result_path}: entries is not a list")

    updated_entries = 0
    missing_case_json: list[str] = []
    missing_metrics: list[dict[str, Any]] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        output_video_value = entry.get("output_video")
        if not isinstance(output_video_value, str) or not output_video_value:
            continue

        case_json_path = Path(output_video_value).with_suffix(".json")
        if not case_json_path.is_file():
            missing_case_json.append(str(case_json_path))
            continue

        case_payload = load_json(case_json_path)
        copied_any = False
        for metric_name in metrics:
            if metric_name in case_payload:
                entry[metric_name] = case_payload[metric_name]
                copied_any = True
            else:
                missing_metrics.append(
                    {
                        "result_json": str(result_path),
                        "case_json": str(case_json_path),
                        "metric": metric_name,
                    }
                )
        if copied_any:
            updated_entries += 1

    payload["metric_backfill"] = {
        "metrics": list(metrics),
        "entries_total": len(entries),
        "entries_updated": updated_entries,
        "missing_case_json_count": len(missing_case_json),
        "missing_metric_count": len(missing_metrics),
    }
    save_json(result_path, payload)
    return {
        "result_json": str(result_path),
        "entries_total": len(entries),
        "entries_updated": updated_entries,
        "missing_case_json": missing_case_json,
        "missing_metrics": missing_metrics,
    }


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    metrics = tuple(args.metrics)

    result_paths = sorted(root.glob("*/result.json"))
    if not result_paths:
        raise FileNotFoundError(f"No result.json files found under {root}")

    summaries: list[dict[str, Any]] = []
    total_entries = 0
    total_updated = 0
    total_missing_case_json = 0
    total_missing_metrics = 0

    for result_path in result_paths:
        summary = backfill_result_json(result_path, metrics)
        summaries.append(summary)
        total_entries += int(summary["entries_total"])
        total_updated += int(summary["entries_updated"])
        total_missing_case_json += len(summary["missing_case_json"])
        total_missing_metrics += len(summary["missing_metrics"])
        print(
            f"{result_path}: updated {summary['entries_updated']}/{summary['entries_total']} entries"
        )

    print(
        json.dumps(
            {
                "root": str(root),
                "metrics": list(metrics),
                "num_result_json": len(result_paths),
                "entries_total": total_entries,
                "entries_updated": total_updated,
                "missing_case_json_count": total_missing_case_json,
                "missing_metric_count": total_missing_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
