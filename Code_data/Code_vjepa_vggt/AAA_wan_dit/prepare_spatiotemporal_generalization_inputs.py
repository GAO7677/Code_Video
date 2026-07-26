#!/usr/bin/env python3
"""Prepare unique test cases and a 50-seed single-case sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-list", type=Path, required=True)
    parser.add_argument("--seed-case", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_lines(path: Path, values: list[Path]) -> None:
    path.write_text(
        "".join(f"{value.expanduser().resolve()}\n" for value in values),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    original_paths: list[Path] = []
    seen: set[Path] = set()
    for line in args.source_list.expanduser().resolve().read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        path = Path(line.strip()).expanduser().resolve()
        if path not in seen:
            original_paths.append(path)
            seen.add(path)
    if len(original_paths) != 20:
        raise RuntimeError(
            f"expected 20 unique test_5 cases, found {len(original_paths)}"
        )

    seed_case = args.seed_case.expanduser().resolve()
    if seed_case not in seen:
        raise RuntimeError("seed sweep case must also be in the test case list")
    seed_payload = json.loads(seed_case.read_text(encoding="utf-8"))
    seed_dir = output_dir / "seed_jsons"
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_paths: list[Path] = []
    seed_map: dict[str, dict] = {}

    for path in original_paths:
        groups = ["test5"]
        if path == seed_case:
            groups.append("seed_sweep")
        seed_map[path.stem] = {
            "case_key": path.stem,
            "source_case": path.stem,
            "seed": 42,
            "groups": groups,
        }

    for seed in range(50):
        if seed == 42:
            continue
        stem = f"{seed_case.stem}__seed_{seed:03d}"
        path = seed_dir / f"{stem}.json"
        path.write_text(
            json.dumps(seed_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        seed_paths.append(path)
        seed_map[stem] = {
            "case_key": stem,
            "source_case": seed_case.stem,
            "seed": seed,
            "groups": ["seed_sweep"],
        }

    combined = [*original_paths, *seed_paths]
    _write_lines(output_dir / "test5_unique.txt", original_paths)
    _write_lines(output_dir / "seed_sweep_49_plus_reused_seed42.txt", seed_paths)
    _write_lines(output_dir / "combined_69_runs.txt", combined)
    (output_dir / "seed_map.json").write_text(
        json.dumps(seed_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "source_list": str(args.source_list.expanduser().resolve()),
        "unique_test_cases": len(original_paths),
        "seed_sweep_case": str(seed_case),
        "seed_sweep_seeds": list(range(50)),
        "seed42_reused_from_test5": True,
        "actual_runs_per_model": len(combined),
        "combined_list": str(output_dir / "combined_69_runs.txt"),
        "seed_map": str(output_dir / "seed_map.json"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
