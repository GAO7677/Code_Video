#!/usr/bin/env python3
"""Render one model's representative all-block overlays for parallel execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_allblock_moving_query_gallery import (
    _load_maps,
    _write_overlay_video,
)
from motion_query_map import _center_crop_resize, _read_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-root", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--query-map", type=Path, required=True)
    parser.add_argument(
        "--model", choices=("wan_lora", "xssc", "physrvg"), required=True
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--step", type=int, default=35)
    parser.add_argument("--blocks", required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(
        args.classification.expanduser().resolve().read_text(encoding="utf-8")
    )
    query_map = json.loads(
        args.query_map.expanduser().resolve().read_text(encoding="utf-8")
    )["cases"][args.case]
    source_frames = [
        _center_crop_resize(frame)
        for frame in _read_video(Path(query_map["source_video"]))
    ]
    rows = {
        (int(row["block"]), int(row["head"])): str(row["role"])
        for row in payload["heads"]
        if row["model"] == args.model
    }
    assets = args.assets_dir.expanduser().resolve()
    assets.mkdir(parents=True, exist_ok=True)
    for block in (int(value) for value in args.blocks.split(",")):
        selections = {
            role: int(head)
            for role, head in payload["representatives"][args.model][
                str(block)
            ].items()
        }
        actual_roles = {
            role: rows[(block, head)] for role, head in selections.items()
        }
        attention, coords = _load_maps(
            args.maps_root.expanduser().resolve(),
            args.model,
            args.case,
            block,
            args.step,
        )
        _write_overlay_video(
            source_frames=source_frames,
            attention=attention,
            query_coords=coords,
            selections=selections,
            actual_roles=actual_roles,
            model=args.model,
            block=block,
            step=args.step,
            output_path=assets
            / f"{args.model}_block{block:02d}_representatives.mp4",
        )
        print(f"[render] {args.model} block {block:02d}", flush=True)


if __name__ == "__main__":
    main()
