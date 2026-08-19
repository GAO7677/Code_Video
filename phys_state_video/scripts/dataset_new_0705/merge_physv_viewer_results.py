"""Merge audited difficulty and V2V result JSONL files for the PhysV viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
            raise ValueError(f"invalid result row at {path}:{line_number}")
        rows.append(row)
    return rows


def _initialization_qa(row: dict[str, object]) -> dict[str, object] | None:
    direct = row.get("initialization_qa")
    if isinstance(direct, dict):
        return direct
    v2v = row.get("v2v")
    if isinstance(v2v, dict) and isinstance(v2v.get("initialization"), dict):
        return v2v["initialization"]
    return None


def merge_results(
    difficulty_results: Path,
    v2v_results: Path,
    output: Path,
) -> dict[str, object]:
    rows = _load_rows(difficulty_results) + _load_rows(v2v_results)
    seen: set[str] = set()
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        qa = _initialization_qa(row)
        if not qa or not bool(qa.get("passed")):
            raise ValueError(f"missing or failed initialization QA for {case_id}")
        if str(row.get("status", "rendered")) != "rendered":
            raise ValueError(f"non-rendered row cannot be published: {case_id}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "output": str(output),
        "total_cases": len(rows),
        "difficulty_cases": sum(not str(row.get("case_id", "")).startswith("v2v_") for row in rows),
        "v2v_cases": sum(str(row.get("case_id", "")).startswith("v2v_") for row in rows),
        "all_initialization_qa_passed": True,
    }
    summary_path = output.parent / "reports" / "combined_results_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difficulty-results", type=Path, required=True)
    parser.add_argument("--v2v-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        merge_results(args.difficulty_results, args.v2v_results, args.output),
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
