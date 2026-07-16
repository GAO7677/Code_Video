#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_comparison_indices import kubric_records, stratified_sample


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/0623_savi/indices"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--kubric-root",
        type=Path,
        default=Path("/data/gaoya/dataset/nnsriram97-phyco_kubric"),
    )
    parser.add_argument("--kubric-count", type=int, default=9600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pybullet_train = read_jsonl(args.source_root / "pybullet" / "train.jsonl")
    kubric_train_all = kubric_records(args.kubric_root.resolve(), "train.txt")
    kubric_train = stratified_sample(kubric_train_all, args.kubric_count, args.seed)
    pybullet_valid = read_jsonl(args.source_root / "pybullet" / "handoff_monitor.jsonl")
    kubric_valid = read_jsonl(args.source_root / "kubric" / "handoff_monitor.jsonl")
    train = pybullet_train + kubric_train
    valid = pybullet_valid + kubric_valid
    write_jsonl(args.output_root / "mixed" / "train.jsonl", train)
    write_jsonl(args.output_root / "mixed" / "handoff_monitor.jsonl", valid)
    manifest = {
        "purpose": "Full source pool for per-epoch source-aware sampling",
        "seed": args.seed,
        "kubric_root": str(args.kubric_root.resolve()),
        "kubric_official_train_available": len(kubric_train_all),
        "train": {
            "total": len(train),
            "pybullet": len(pybullet_train),
            "kubric": len(kubric_train),
        },
        "handoff_monitor": {
            "total": len(valid),
            "pybullet": len(pybullet_valid),
            "kubric": len(kubric_valid),
        },
        "paths_are_absolute": True,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
