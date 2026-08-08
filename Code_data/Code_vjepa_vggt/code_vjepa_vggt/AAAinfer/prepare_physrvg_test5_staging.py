#!/usr/bin/env python3
"""Prepare a flat, deduplicated staging directory for PhysRVG test_5 inference."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = list(
        dict.fromkeys(
            line.strip()
            for line in args.input_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prepared: list[dict[str, str]] = []
    for entry in entries:
        input_json = Path(entry).resolve()
        payload = json.loads(input_json.read_text(encoding="utf-8"))
        input_video = Path(payload["input_video"]).resolve()
        if not input_video.is_file():
            raise FileNotFoundError(input_video)
        stem = input_json.stem
        shutil.copy2(input_json, args.output_dir / f"{stem}.json")
        shutil.copy2(input_video, args.output_dir / f"{stem}.mp4")
        prepared.append(
            {
                "case": stem,
                "input_json": str(input_json),
                "input_video": str(input_video),
            }
        )

    manifest = {
        "input_list": str(args.input_list.resolve()),
        "raw_entries": sum(
            bool(line.strip())
            for line in args.input_list.read_text(encoding="utf-8").splitlines()
        ),
        "prepared_count": len(prepared),
        "prepared": prepared,
    }
    (args.output_dir / "staging_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
