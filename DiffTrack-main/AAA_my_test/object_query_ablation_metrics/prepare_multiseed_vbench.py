#!/usr/bin/env python3
"""Create a focused VBench index for the strict six-seed ablation cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CASE = "0613pybullet_sample_001460_w002"
DEFAULT_SEEDS = (13248, 32466, 35075, 47326, 68613, 90094)
INVENTORY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1"
) / CASE
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/object_query_ablation_metrics_vbench_6seed"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    index = args.output_root / "index"
    index.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in dict.fromkeys(args.seeds):
        inventory_path = INVENTORY_ROOT / f"seed_{seed:05d}" / "video_similarity_top100.json"
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
        videos = payload.get("videos", [])
        if len(videos) != 49:
            raise RuntimeError(f"seed {seed}: expected 49 videos, found {len(videos)}")
        for video in videos:
            run_dir = Path(video["path"]).parent.resolve()
            manifest = run_dir / "manifest.json"
            if not manifest.is_file():
                raise FileNotFoundError(manifest)
            identity = f"{CASE}:{seed}:{video['id']}"
            name = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:20]
            link = index / name
            if link.is_symlink() and link.resolve() != run_dir:
                raise RuntimeError(f"stale conflicting link: {link}")
            if not link.exists():
                link.symlink_to(run_dir, target_is_directory=True)
            records.append(
                {
                    "case": CASE,
                    "seed": seed,
                    "video_id": video["id"],
                    "run_dir": str(run_dir),
                    "index_link": str(link),
                }
            )
    snapshot = {
        "case": CASE,
        "seeds": list(dict.fromkeys(args.seeds)),
        "sample_count": len(dict.fromkeys(args.seeds)),
        "video_count": len(records),
        "index_root": str(index),
        "records": records,
    }
    (args.output_root / "snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: snapshot[key] for key in ("seeds", "video_count", "index_root")}, indent=2))


if __name__ == "__main__":
    main()
