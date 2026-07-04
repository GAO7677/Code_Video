#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a manifest of samples whose probe feature files are still missing under an extract root."
        )
    )
    parser.add_argument("--manifest_csv", type=Path, required=True)
    parser.add_argument("--extract_root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument(
        "--split_count",
        type=int,
        default=1,
        help="Optionally split pending rows into multiple pair-preserving shard CSVs.",
    )
    return parser.parse_args()


def sample_id_from_row(row: dict[str, str]) -> str:
    return f"{row['pair_id']}__{row['role']}"


def is_done(extract_root: Path, sample_id: str) -> bool:
    sample_root = extract_root / sample_id
    return (sample_root / "probe_features.pt").is_file() and (sample_root / "meta.json").is_file()


def group_rows_by_pair(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    groups: list[list[dict[str, str]]] = []
    current_pair_id: str | None = None
    current_rows: list[dict[str, str]] = []
    for row in rows:
        pair_id = row["pair_id"]
        if current_pair_id is None or pair_id == current_pair_id:
            current_rows.append(row)
        else:
            groups.append(current_rows)
            current_rows = [row]
        current_pair_id = pair_id
    if current_rows:
        groups.append(current_rows)
    return groups


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with args.manifest_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    pending_rows = [row for row in rows if not is_done(args.extract_root, sample_id_from_row(row))]
    write_csv(args.output_csv, fieldnames, pending_rows)

    summary = {
        "manifest_csv": str(args.manifest_csv),
        "extract_root": str(args.extract_root),
        "output_csv": str(args.output_csv),
        "total_rows": len(rows),
        "pending_rows": len(pending_rows),
        "done_rows": len(rows) - len(pending_rows),
        "pending_pair_count": len({row["pair_id"] for row in pending_rows}),
        "split_count": args.split_count,
    }

    if args.split_count > 1:
        pending_groups = group_rows_by_pair(pending_rows)
        shards = [[] for _ in range(args.split_count)]
        for idx, pair_rows in enumerate(pending_groups):
            shards[idx % args.split_count].extend(pair_rows)

        shard_paths: list[str] = []
        for shard_idx, shard_rows in enumerate(shards):
            shard_path = args.output_csv.parent / f"{args.output_csv.stem}_shard{shard_idx}{args.output_csv.suffix}"
            write_csv(shard_path, fieldnames, shard_rows)
            shard_paths.append(str(shard_path))
        summary["shard_csvs"] = shard_paths

    summary_path = args.output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output_csv)
    print(summary_path)


if __name__ == "__main__":
    main()
