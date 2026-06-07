#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.proxy_state import read_video_frames
from phys_state_video.wan_state_v2_helpers import compute_future_latent_steps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a fixed random-clip manifest for the tailquery benchmark dashboard."
    )
    parser.add_argument("--bench-json-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--sample-per-json", type=int, default=5)
    parser.add_argument("--json-names", nargs="+", default=["A.json", "B.json", "D.json"])
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--context-min", type=int, default=4)
    parser.add_argument("--context-max", type=int, default=12)
    parser.add_argument("--future-min", type=int, default=8)
    parser.add_argument("--future-max", type=int, default=20)
    parser.add_argument("--temporal-stride", type=int, required=True)
    parser.add_argument("--max-context-latent-steps", type=int, required=True)
    parser.add_argument("--max-future-latent-steps", type=int, required=True)
    return parser.parse_args()


def resolve_clip_bounds(
    total_frames: int,
    *,
    temporal_stride: int,
    max_context_latent_steps: int,
    max_future_latent_steps: int,
    context_min: int,
    context_max: int,
    future_min: int,
    future_max: int,
    rng: random.Random,
) -> tuple[int, int, int]:
    candidates: list[tuple[int, int]] = []
    for context_steps in range(context_min, context_max + 1):
        if context_steps >= total_frames:
            continue
        context_latent_steps = 1 + max(context_steps - 1, 0) // temporal_stride
        if context_latent_steps > max_context_latent_steps:
            continue
        for future_steps in range(future_min, future_max + 1):
            if context_steps + future_steps > total_frames:
                continue
            try:
                future_latent_steps = compute_future_latent_steps(context_steps, future_steps, temporal_stride)
            except ValueError:
                continue
            if future_latent_steps > max_future_latent_steps:
                continue
            candidates.append((context_steps, future_steps))
    if not candidates:
        raise RuntimeError(
            "no valid random clip can satisfy latent-step limits: "
            f"total_frames={total_frames}, context=[{context_min},{context_max}], future=[{future_min},{future_max}]"
        )
    context_steps, future_steps = rng.choice(candidates)
    start_max = total_frames - (context_steps + future_steps)
    start_idx = rng.randint(0, max(start_max, 0))
    return start_idx, context_steps, future_steps


def choose_bench_cases(bench_json_root: Path, json_names: list[str], sample_per_json: int, seed: int):
    rng = random.Random(seed)
    chosen = []
    for json_name in json_names:
        payload = json.loads((bench_json_root / json_name).read_text(encoding="utf-8"))
        indices = list(range(len(payload)))
        rng.shuffle(indices)
        for source_index in indices[: min(sample_per_json, len(indices))]:
            item = payload[source_index]
            chosen.append(
                {
                    "source_name": Path(json_name).stem,
                    "source_index": int(source_index),
                    "category": str(item.get("category") or "unknown"),
                    "source_video": str(item["source_video"]),
                    "caption": str(item.get("caption") or ""),
                }
            )
    return chosen


def main():
    args = parse_args()
    bench_json_root = Path(args.bench_json_root)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    chosen = choose_bench_cases(bench_json_root, args.json_names, args.sample_per_json, args.seed)
    cases = []
    for case_idx, spec in enumerate(chosen):
        source_path = Path(spec["source_video"])
        if not source_path.exists():
            print(f"skip missing source video: {source_path}")
            continue
        full_video = read_video_frames(
            source_path,
            resize_height=args.height,
            resize_width=args.width,
        )
        start_idx, context_steps, future_steps = resolve_clip_bounds(
            int(full_video.shape[0]),
            temporal_stride=args.temporal_stride,
            max_context_latent_steps=args.max_context_latent_steps,
            max_future_latent_steps=args.max_future_latent_steps,
            context_min=args.context_min,
            context_max=args.context_max,
            future_min=args.future_min,
            future_max=args.future_max,
            rng=rng,
        )
        stem = source_path.stem.replace(" ", "_")
        case_id = f"{spec['source_name'].lower()}_{spec['source_index']:03d}_{case_idx:02d}_{stem}"
        cases.append(
            {
                "case_id": case_id,
                "source_name": spec["source_name"],
                "source_index": int(spec["source_index"]),
                "category": spec["category"],
                "source_video": str(source_path),
                "caption": spec["caption"],
                "source_total_frames": int(full_video.shape[0]),
                "clip_start": int(start_idx),
                "context_steps": int(context_steps),
                "future_steps": int(future_steps),
                "height": int(args.height),
                "width": int(args.width),
            }
        )
    payload = {
        "bench_json_root": str(bench_json_root),
        "seed": int(args.seed),
        "sample_per_json": int(args.sample_per_json),
        "json_names": list(args.json_names),
        "height": int(args.height),
        "width": int(args.width),
        "temporal_stride": int(args.temporal_stride),
        "max_context_latent_steps": int(args.max_context_latent_steps),
        "max_future_latent_steps": int(args.max_future_latent_steps),
        "cases": cases,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {output_json}")
    print(f"cases: {len(cases)}")


if __name__ == "__main__":
    main()
