from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


DEFAULT_RESULT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/v2v")
EXCLUDED_JSON_NAMES = {"summary.json", "result.json", "batch_manifest.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a txt file of input json paths, collect all matching method outputs "
            "from the v2v result root, copy videos into per-case folders, and write "
            "per-case metric metadata."
        )
    )
    parser.add_argument("--txt-path", type=Path, required=True, help="Txt file listing input json paths.")
    parser.add_argument("--output-root", type=Path, required=True, help="Destination root for copied videos and metrics.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT, help="Root containing v2v result jsons.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


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
        return float(value)
    return None


def extract_metric_values(payload: dict[str, Any]) -> dict[str, float | None]:
    return {
        "pdi_score": to_float(nested_get(payload, "pdi", "pdi_score")),
        "wmreward_surprise": to_float(nested_get(payload, "wmreward", "surprise")),
        "wmreward_similarity": to_float(nested_get(payload, "wmreward", "similarity")),
        "proxy_score": to_float(nested_get(payload, "proxy", "score")),
        "videophy2_score": to_float(nested_get(payload, "videophy2", "score")),
        "phyground_general_avg": to_float(nested_get(payload, "phyground", "general_avg")),
        "cosmos_reason1_score": to_float(nested_get(payload, "cosmos_reason1", "score")),
    }


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    slug = slug.strip("._-")
    return slug or "unknown"


def resolve_path_string(path_str: str) -> str:
    return str(Path(path_str).expanduser().resolve())


def load_input_json_targets(txt_path: Path) -> list[str]:
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    targets: list[str] = []
    seen: set[str] = set()
    for line in lines:
        candidate = line.strip()
        if not candidate or candidate.startswith("#"):
            continue
        resolved = resolve_path_string(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        targets.append(resolved)
    return targets


def discover_result_jsons(result_root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(result_root.rglob("*.json")):
        if path.name in EXCLUDED_JSON_NAMES:
            continue
        if path.name.startswith("eval_summary_"):
            continue
        payload = load_json(path)
        if payload is None:
            continue
        if "input_json" not in payload:
            continue
        paths.append(path)
    return paths


def derive_method_name(payload: dict[str, Any], result_json_path: Path) -> str:
    method = payload.get("method")
    if isinstance(method, str) and method.strip():
        return method.strip()
    return result_json_path.parent.name or result_json_path.stem


def count_available_metrics(metrics: dict[str, float | None]) -> int:
    return sum(1 for value in metrics.values() if value is not None)


def source_variant_priority(result_json_path: str) -> int:
    lowered = result_json_path.lower()
    if "frame49" in lowered:
        return 1
    if "chain" in lowered:
        return 0
    return 2


def select_best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)

    selected: list[dict[str, Any]] = []
    for method, candidates in grouped.items():
        ranked = sorted(
            candidates,
            key=lambda item: (
                -count_available_metrics(item["metrics"]),
                -source_variant_priority(str(item["result_json_path"])),
                str(item["result_json_path"]),
            ),
        )
        best = dict(ranked[0])
        best["num_method_candidates"] = len(candidates)
        best["discarded_result_json_paths"] = [
            str(candidate["result_json_path"])
            for candidate in ranked[1:]
        ]
        selected.append(best)

    return sorted(selected, key=lambda item: str(item["method"]))


def main() -> None:
    args = parse_args()
    txt_path = args.txt_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    result_root = args.result_root.expanduser().resolve()

    targets = load_input_json_targets(txt_path)
    target_set = set(targets)
    output_root.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[dict[str, Any]]] = {target: [] for target in targets}
    unmatched_result_jsons: list[str] = []

    for result_json_path in discover_result_jsons(result_root):
        payload = load_json(result_json_path)
        if payload is None:
            continue
        input_json = payload.get("input_json")
        if not isinstance(input_json, str) or not input_json.strip():
            continue
        resolved_input_json = resolve_path_string(input_json)
        if resolved_input_json not in target_set:
            continue

        output_video = payload.get("output_video")
        output_video_path = None
        if isinstance(output_video, str) and output_video.strip():
            candidate = Path(output_video).expanduser().resolve()
            if candidate.exists():
                output_video_path = candidate
            else:
                unmatched_result_jsons.append(str(result_json_path))

        buckets[resolved_input_json].append(
            {
                "method": derive_method_name(payload, result_json_path),
                "result_json_path": str(result_json_path),
                "output_video": str(output_video_path) if output_video_path is not None else None,
                "metrics": extract_metric_values(payload),
                "raw_metric_keys": sorted(
                    key
                    for key in ("pdi", "wmreward", "proxy", "videophy2", "phyground", "cosmos_reason1")
                    if key in payload
                ),
            }
        )

    export_summary: list[dict[str, Any]] = []
    for input_json_path in targets:
        case_name = Path(input_json_path).stem
        case_dir = output_root / slugify(case_name)
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        raw_rows = buckets[input_json_path]
        rows = select_best_rows(raw_rows)
        copied_rows: list[dict[str, Any]] = []
        for row in rows:
            copied_video_path = None
            if row["output_video"] is not None:
                src = Path(str(row["output_video"]))
                dst = case_dir / f"{slugify(str(row['method']))}.mp4"
                shutil.copy2(src, dst)
                copied_video_path = str(dst)

            copied_rows.append(
                {
                    "method": row["method"],
                    "input_json": input_json_path,
                    "copied_video": copied_video_path,
                    "source_output_video": row["output_video"],
                    "result_json_path": row["result_json_path"],
                    "num_method_candidates": row["num_method_candidates"],
                    "discarded_result_json_paths": row["discarded_result_json_paths"],
                    **row["metrics"],
                }
            )

        metrics_json_path = case_dir / "metrics.json"
        metrics_json_path.write_text(
            json.dumps(
                {
                    "input_json": input_json_path,
                    "txt_path": str(txt_path),
                    "result_root": str(result_root),
                    "num_candidate_rows": len(raw_rows),
                    "num_methods": len(copied_rows),
                    "selection_rule": (
                        "Select one row per method by highest non-null metric count, "
                        "then prefer non-frame49/non-chain result paths, then "
                        "lexicographically smallest result_json_path."
                    ),
                    "rows": copied_rows,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        export_summary.append(
            {
                "input_json": input_json_path,
                "case_dir": str(case_dir),
                "metrics_json": str(metrics_json_path),
                "num_candidate_rows": len(raw_rows),
                "num_methods": len(copied_rows),
            }
        )

    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "txt_path": str(txt_path),
                "result_root": str(result_root),
                "output_root": str(output_root),
                "num_cases": len(targets),
                "cases": export_summary,
                "unmatched_result_jsons": unmatched_result_jsons,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "txt_path": str(txt_path),
                "result_root": str(result_root),
                "output_root": str(output_root),
                "num_cases": len(targets),
                "summary_json": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
