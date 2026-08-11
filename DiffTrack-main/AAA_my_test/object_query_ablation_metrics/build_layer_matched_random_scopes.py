#!/usr/bin/env python3
"""Build frozen Random100 controls matched to latest3350 Top100 layer counts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SOURCE = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326/pck_head_scopes_s039_latest3350.json"
)
OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/head_scopes_latest3350_with_random100.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--draws", type=int, default=3)
    args = parser.parse_args()

    payload = json.loads(args.source.read_text(encoding="utf-8"))
    entries = list(payload["entries"])
    if len(entries) != 720:
        raise RuntimeError(f"expected 720 ranked heads, got {len(entries)}")
    top = entries[:100]
    bottom = entries[-100:]
    top_pairs = {(int(row["block"]), int(row["head"])) for row in top}
    bottom_pairs = {(int(row["block"]), int(row["head"])) for row in bottom}
    if top_pairs & bottom_pairs:
        raise RuntimeError("Top100 and Bottom100 overlap")
    layer_counts = Counter(int(row["block"]) for row in top)
    rng = np.random.default_rng(args.seed)

    scope_definitions = dict(payload["head_scopes"])
    draws = {}
    for draw in range(args.draws):
        pairs = []
        for block in range(30):
            count = layer_counts.get(block, 0)
            candidates = [
                head
                for head in range(24)
                if (block, head) not in top_pairs and (block, head) not in bottom_pairs
            ]
            if len(candidates) < count:
                raise RuntimeError(
                    f"L{block}: need {count} random heads but only {len(candidates)} remain"
                )
            chosen = sorted(int(value) for value in rng.choice(candidates, count, replace=False))
            pairs.extend([[block, head] for head in chosen])
        if len(pairs) != 100 or len({tuple(pair) for pair in pairs}) != 100:
            raise RuntimeError(f"draw{draw}: invalid random head count")
        name = f"random100_layer_matched_draw{draw}"
        scope_definitions[name] = {
            "pairs": pairs,
            "count": 100,
            "matched_to": "top100_per_layer_histogram",
            "sampling_pool": "all720_excluding_top100_and_bottom100",
            "random_seed": int(args.seed),
            "draw_index": draw,
        }
        draws[name] = pairs

    result = {
        **payload,
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_head_scopes": str(args.source),
        "source_head_scopes_sha256": sha256(args.source),
        "random_control_seed": int(args.seed),
        "random_control_policy": (
            "independent draws; exact Top100 per-layer counts; exclude fixed Top100/Bottom100"
        ),
        "head_scopes": scope_definitions,
        "random_draws": draws,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(args.output)
    print(json.dumps(dict(sorted(layer_counts.items())), indent=2))


if __name__ == "__main__":
    main()
