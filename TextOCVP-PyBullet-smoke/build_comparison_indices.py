#!/usr/bin/env python3
"""Build fixed absolute-path indices for the three SAVi Stage 1 experiments."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pybullet-root", type=Path, required=True)
    parser.add_argument("--kubric-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")


def pybullet_records(root, split):
    records = []
    for video_path in sorted((root / split).glob("*/sample_*/video.mp4")):
        records.append(
            {
                "source": "pybullet",
                "video_path": str(video_path.resolve()),
                "metadata_path": str((video_path.parent / "meta.json").resolve()),
                "group": video_path.parents[1].name,
                "sample_id": video_path.parent.name,
                "sampling_frame_range": [0, 49],
            }
        )
    return records


def kubric_records(root, list_name):
    list_path = root / list_name
    records = []
    for raw_path in list_path.read_text(encoding="utf-8").splitlines():
        if not raw_path.strip():
            continue
        sample_path = Path(raw_path.strip())
        if not sample_path.is_absolute():
            sample_path = root / sample_path
        video_path = sample_path / "rgba.mp4"
        if not video_path.is_file():
            continue
        records.append(
            {
                "source": "kubric",
                "video_path": str(video_path.resolve()),
                "metadata_path": str((sample_path / "metadata.json").resolve()),
                "group": sample_path.relative_to(root).parts[0],
                "sample_id": sample_path.name,
                "sampling_frame_range": [0, 49],
            }
        )
    return records


def stratified_sample(records, count, seed):
    if count > len(records):
        raise ValueError(f"Cannot select {count} records from {len(records)}")
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    groups = {}
    for record in shuffled:
        groups.setdefault(record["group"], []).append(record)
    selected = []
    offset = 0
    ordered_groups = sorted(groups)
    while len(selected) < count:
        added = False
        for group in ordered_groups:
            values = groups[group]
            if offset < len(values):
                selected.append(values[offset])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        offset += 1
    rng.shuffle(selected)
    return selected


def summarize(records):
    return {
        "count": len(records),
        "sources": dict(sorted(Counter(record["source"] for record in records).items())),
        "groups": dict(sorted(Counter(record["group"] for record in records).items())),
    }


def main():
    args = parse_args()
    output_root = args.output_root.resolve()
    pybullet_root = args.pybullet_root.resolve()
    kubric_root = args.kubric_root.resolve()

    pybullet_train = pybullet_records(pybullet_root, "train")
    pybullet_handoff = pybullet_records(pybullet_root, "val")
    kubric_train_all = kubric_records(kubric_root, "train.txt")
    kubric_handoff_all = kubric_records(kubric_root, "val.txt")
    if len(pybullet_train) != 1200 or len(pybullet_handoff) != 150:
        raise RuntimeError(
            f"Unexpected PyBullet counts: train={len(pybullet_train)} val={len(pybullet_handoff)}"
        )

    kubric_train = stratified_sample(kubric_train_all, 1200, args.seed)
    kubric_handoff = stratified_sample(kubric_handoff_all, 150, args.seed + 1)
    mixed_pybullet = stratified_sample(pybullet_train, 600, args.seed + 2)
    mixed_kubric = stratified_sample(kubric_train, 600, args.seed + 3)
    mixed_train = mixed_pybullet + mixed_kubric
    random.Random(args.seed + 4).shuffle(mixed_train)
    mixed_handoff = list(pybullet_handoff) + list(kubric_handoff)

    indices = {
        "pybullet/train.jsonl": pybullet_train,
        "pybullet/handoff_monitor.jsonl": pybullet_handoff,
        "kubric/train.jsonl": kubric_train,
        "kubric/handoff_monitor.jsonl": kubric_handoff,
        "kubric/handoff_full.jsonl": kubric_handoff_all,
        "mixed/train.jsonl": mixed_train,
        "mixed/handoff_monitor.jsonl": mixed_handoff,
    }
    for relative_path, records in indices.items():
        write_jsonl(output_root / relative_path, records)

    manifest = {
        "seed": args.seed,
        "pybullet_root": str(pybullet_root),
        "kubric_root": str(kubric_root),
        "sampling_frame_range": [0, 49],
        "num_frames": 10,
        "frame_stride": 1,
        "resolution_hw": [216, 384],
        "indices": {
            relative_path: summarize(records) for relative_path, records in indices.items()
        },
        "source_lists": {
            "kubric_train_all": len(kubric_train_all),
            "kubric_handoff_all": len(kubric_handoff_all),
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
