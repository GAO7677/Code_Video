#!/usr/bin/env python3
"""Assemble viewer data from an existing media build and fresh result rows.

The media bytes depend only on the input videos and preprocessing settings, not
on the GPU used for inference.  This helper reuses an existing media build
while replacing its per-case inference metadata and captions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    fresh = read_jsonl(args.results)
    template = json.loads((args.template / "viewer_data.json").read_text(encoding="utf-8"))["cases"]
    by_id = {row["case_id"]: row for row in template}
    if not fresh:
        raise ValueError("No fresh result rows")
    missing = [row["case_id"] for row in fresh if row["case_id"] not in by_id]
    if missing:
        raise ValueError(f"Template is missing cases: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    public: list[dict[str, Any]] = []
    for row in fresh:
        old = by_id[row["case_id"]]
        merged = dict(row)
        # Keep the media/audit fields produced from the exact same input clip.
        for key in ("dataset", "family", "sample", "case_slug", "assets", "audit"):
            if key in old:
                merged[key] = old[key]
        public.append(merged)
        slug = merged["case_slug"]
        src = args.media_dir / slug
        dst = args.output_dir / slug
        if not dst.exists():
            dst.symlink_to(src, target_is_directory=True)

    (args.output_dir / "viewer_data.json").write_text(
        json.dumps({"cases": public}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"viewer_data={args.output_dir / 'viewer_data.json'} cases={len(public)}")


if __name__ == "__main__":
    main()
