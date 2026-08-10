#!/usr/bin/env python3
"""Create a focused VBench index from one result directory or the legacy cohort."""

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
    parser.add_argument(
        "--result-dir",
        type=Path,
        action="append",
        help=(
            "seed result directory containing video_similarity_top100.json; "
            "repeat to index more than one directory"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    index = output_root / "index"
    index.mkdir(parents=True, exist_ok=True)
    records = []
    if args.result_dir:
        inventory_paths = [
            path.expanduser().resolve() / "video_similarity_top100.json"
            for path in args.result_dir
        ]
    else:
        inventory_paths = [
            INVENTORY_ROOT / f"seed_{seed:05d}" / "video_similarity_top100.json"
            for seed in dict.fromkeys(args.seeds)
        ]
    identities = []
    expected_links = set()
    for inventory_path in inventory_paths:
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
        case = str(payload.get("case") or "")
        seed = int(payload.get("seed", -1))
        if not case or seed < 0:
            raise RuntimeError(f"invalid case/seed identity: {inventory_path}")
        identities.append((case, seed))
        videos = payload.get("videos", [])
        if len(videos) != 49:
            raise RuntimeError(
                f"{case} seed {seed}: expected 49 videos, found {len(videos)}"
            )
        for video in videos:
            run_dir = Path(video["path"]).parent.resolve()
            manifest = run_dir / "manifest.json"
            if not manifest.is_file():
                raise FileNotFoundError(manifest)
            identity = f"{case}:{seed}:{video['id']}"
            name = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:20]
            expected_links.add(name)
            link = index / name
            if link.is_symlink() and link.resolve() != run_dir:
                raise RuntimeError(f"stale conflicting link: {link}")
            if not link.exists():
                link.symlink_to(run_dir, target_is_directory=True)
            records.append(
                {
                    "case": case,
                    "seed": seed,
                    "video_id": video["id"],
                    "video": str(Path(video["path"]).resolve()),
                    "run_dir": str(run_dir),
                    "index_link": str(link),
                }
            )
    for link in index.iterdir():
        if link.is_symlink() and link.name not in expected_links:
            link.unlink()
    case_names = sorted({case for case, _seed in identities})
    snapshot = {
        "case": case_names[0] if len(case_names) == 1 else None,
        "cases": case_names,
        "seeds": sorted({seed for _case, seed in identities}),
        "sample_count": len(set(identities)),
        "video_count": len(records),
        "index_root": str(index),
        "records": records,
    }
    (output_root / "snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: snapshot[key] for key in ("seeds", "video_count", "index_root")}, indent=2))


if __name__ == "__main__":
    main()
