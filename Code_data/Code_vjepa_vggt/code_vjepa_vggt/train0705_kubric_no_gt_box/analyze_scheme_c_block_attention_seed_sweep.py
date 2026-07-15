#!/usr/bin/env python3
"""Compare stale-anchor and expected-object noun attention across random seeds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LAYERS = (8, 11, 25)
PROGRESS_INDICES = (0, 10, 20, 30, 39)
FUTURE_LATENT_START = 2
# Normalized-image regions mapped to the captured 16x28 attention grid.
STALE_ANCHOR = (slice(0, 5), slice(22, 28))
EXPECTED_BLOCK = (slice(6, 13), slice(17, 24))


def _single(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern!r} under {root}, found {matches}")
    return matches[0]


def _attention_dir(root: Path) -> Path:
    return _single(
        root,
        "**/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-"
        "ball-and-block-fall_motion_to_end_text_noun_attention",
    )


def _measure(seed: int, attention_dir: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with np.load(attention_dir / "all_attention_maps_fp16.npz") as maps:
        for progress in PROGRESS_INDICES:
            for layer in LAYERS:
                key = f"block__layer_{layer:02d}__progress_{progress:02d}"
                values = maps[key].astype(np.float32)[FUTURE_LATENT_START:]
                stale = float(values[:, STALE_ANCHOR[0], STALE_ANCHOR[1]].sum())
                expected = float(values[:, EXPECTED_BLOCK[0], EXPECTED_BLOCK[1]].sum())
                total = float(values.sum())
                rows.append(
                    {
                        "seed": seed,
                        "progress_index": progress,
                        "remaining_steps": {0: 40, 10: 30, 20: 20, 30: 10, 39: 1}[progress],
                        "layer": layer,
                        "stale_anchor_mass": stale,
                        "expected_block_mass": expected,
                        "stale_to_expected_ratio": stale / max(expected, 1.0e-12),
                        "stale_anchor_percent": 100.0 * stale / max(total, 1.0e-12),
                        "expected_block_percent": 100.0 * expected / max(total, 1.0e-12),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--seed-sweep-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runs = {42: _attention_dir(args.baseline_root)}
    for seed_dir in sorted(args.seed_sweep_root.glob("seed_*")):
        if seed_dir.is_dir() and seed_dir.name[5:].isdigit():
            runs[int(seed_dir.name[5:])] = _attention_dir(seed_dir)

    rows: list[dict[str, float | int]] = []
    top_layers: dict[str, list[int]] = {}
    for seed, attention_dir in sorted(runs.items()):
        rows.extend(_measure(seed, attention_dir))
        manifest = json.loads((attention_dir / "manifest.json").read_text(encoding="utf-8"))
        top_layers[str(seed)] = [int(value) for value in manifest["nouns"]["block"]["top_layers"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "block_attention_roi_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    focus: dict[str, float] = {}
    for seed in sorted(runs):
        selected = [
            float(row["stale_to_expected_ratio"])
            for row in rows
            if row["seed"] == seed
            and row["progress_index"] == 20
            and row["layer"] in (8, 11)
        ]
        focus[str(seed)] = float(np.mean(selected))

    summary = {
        "seeds": sorted(runs),
        "fixed_layers": list(LAYERS),
        "progress_indices": list(PROGRESS_INDICES),
        "future_latent_start": FUTURE_LATENT_START,
        "stale_anchor_grid_region": {"y": [0, 5], "x": [22, 28]},
        "expected_block_grid_region": {"y": [6, 13], "x": [17, 24]},
        "block_top_layers_by_seed": top_layers,
        "mid_denoise_l8_l11_stale_to_expected_mean": focus,
        "metrics_csv": str(csv_path),
    }
    summary_path = args.output_dir / "block_attention_seed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
