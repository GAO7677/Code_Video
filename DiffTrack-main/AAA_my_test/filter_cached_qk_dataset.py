#!/usr/bin/env python3
"""Create a lightweight dataset containing only cases with valid region caches."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dataset.resolve()
    cache = args.cache_root.resolve()
    output = args.output_dataset.resolve()
    included = []
    excluded = []
    for manifest_path in sorted((source / "cases").glob("case_*/case_manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        case_key = str(payload["case_key"])
        if (cache / case_key / "complete.json").is_file():
            destination = output / "cases" / case_key / "case_manifest.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest_path, destination)
            included.append(payload)
        else:
            error_path = cache / case_key / "error.txt"
            excluded.append(
                {
                    "case_key": case_key,
                    "reason": error_path.read_text(encoding="utf-8")[-4000:]
                    if error_path.is_file()
                    else "missing SAM2 cache",
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(
            {"included_count": len(included), "excluded_count": len(excluded), "excluded": excluded, "cases": included},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Filtered dataset: {len(included)} included, {len(excluded)} excluded", flush=True)
    if not included:
        raise SystemExit("no cases have valid SAM2 caches")


if __name__ == "__main__":
    main()
