#!/usr/bin/env python3
"""Print resumable progress for the 20-case multi-object M1 search."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/training_free_m1_multi_object_search_v1"
)
MANIFEST = ROOT / "search_manifest.json"


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    samples = payload["samples"]
    baseline = sum(Path(row["baseline_video"]).is_file() for row in samples)
    tracks = sum(
        (ROOT / "tracks" / row["case"] / f"seed_{int(row['seed']):05d}" /
         "frozen_baseline_tracks/complete.json").is_file()
        for row in samples
    )
    completes = list((ROOT / "guided").glob("*/seed_*/multi_object_*/complete.json"))
    errors = list((ROOT / "guided").glob("*/seed_*/multi_object_*/error.txt"))
    by_case = Counter(path.parents[2].name for path in completes)
    print(
        f"baselines={baseline}/{len(samples)} tracks={tracks}/{len(samples)} "
        f"guided={len(completes)}/{payload['search_grid']['guided_video_count']} "
        f"errors={len(errors)}"
    )
    for case in sorted({row["case"] for row in samples}):
        print(f"{case}: {by_case[case]}/80")


if __name__ == "__main__":
    main()
