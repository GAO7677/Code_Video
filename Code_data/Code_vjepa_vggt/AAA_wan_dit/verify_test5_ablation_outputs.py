#!/usr/bin/env python3
"""Verify one five-case DiT ablation output and identify its metric root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_inputs(path: Path) -> list[Path]:
    entries = [
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(entries) != 5 or len({entry.stem for entry in entries}) != 5:
        raise ValueError(f"expected exactly five unique input JSONs, got {len(entries)}")
    return entries


def unique_path(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {name} under {root}, found {len(matches)}: {matches}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument(
        "--model", choices=("wan_lora", "xssc", "physrvg"), required=True
    )
    parser.add_argument("--mode", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--head", default="none")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_root = args.config_root.expanduser().resolve()
    inputs = read_inputs(args.input_list.expanduser().resolve())
    expected_block = None if args.block == "none" else int(args.block)
    expected_head = None if args.head == "none" else int(args.head)
    metadata_key = "physrvg_ablation" if args.model == "physrvg" else "dit_ablation"
    records: list[dict] = []
    result_roots: set[Path] = set()

    for input_json in inputs:
        video_path = unique_path(config_root, f"{input_json.stem}.mp4")
        json_path = unique_path(config_root, f"{input_json.stem}.json")
        if video_path.stat().st_size <= 0:
            raise RuntimeError(f"empty video: {video_path}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        metadata = payload.get(metadata_key)
        if not isinstance(metadata, dict):
            raise RuntimeError(f"{json_path} is missing {metadata_key}")
        if metadata.get("mode") != args.mode:
            raise RuntimeError(
                f"{json_path} mode={metadata.get('mode')!r}, expected {args.mode!r}"
            )
        if metadata.get("block_id") != expected_block:
            raise RuntimeError(
                f"{json_path} block_id={metadata.get('block_id')!r}, "
                f"expected {expected_block!r}"
            )
        if metadata.get("head_id") != expected_head:
            raise RuntimeError(
                f"{json_path} head_id={metadata.get('head_id')!r}, "
                f"expected {expected_head!r}"
            )
        if args.mode == "self_attn_head_zero":
            if metadata.get("num_attention_heads") != 24:
                raise RuntimeError(
                    f"{json_path} num_attention_heads="
                    f"{metadata.get('num_attention_heads')!r}, expected 24"
                )
            expected_calls = 40 if args.model == "physrvg" else 400
            if metadata.get("observed_target_forward_calls") != expected_calls:
                raise RuntimeError(
                    f"{json_path} observed_target_forward_calls="
                    f"{metadata.get('observed_target_forward_calls')!r}, "
                    f"expected {expected_calls}"
                )
        recorded_input = Path(str(payload.get("input_json", ""))).expanduser().resolve()
        if recorded_input != input_json:
            raise RuntimeError(
                f"{json_path} input_json={recorded_input}, expected {input_json}"
            )
        result_roots.add(video_path.parent)
        records.append(
            {
                "input_json": str(input_json),
                "output_video": str(video_path),
                "output_json": str(json_path),
                "video_bytes": int(video_path.stat().st_size),
            }
        )

    if len(result_roots) != 1:
        raise RuntimeError(f"outputs span multiple metric roots: {sorted(result_roots)}")
    result_root = next(iter(result_roots))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "model": args.model,
                "mode": args.mode,
                "block_id": expected_block,
                "head_id": expected_head,
                "config_root": str(config_root),
                "result_root": str(result_root),
                "num_cases": len(records),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
