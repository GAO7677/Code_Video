#!/usr/bin/env python3
"""Rebuild the complete S039 head ranking behind a frozen Top100 manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50"
)
DEFAULT_MANIFEST = (
    DEFAULT_ROOT / "visual_samples/attention_zero_seed47326/cases.json"
)
DEFAULT_OUTPUT = (
    DEFAULT_MANIFEST.parent / "pck_head_scopes_s039_frozen134.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    run_count = int(manifest["completed_runs_at_selection"])
    completed = sorted(
        (path.stat().st_mtime_ns, path)
        for path in (args.root / "runs").glob("*/seed_*/complete.json")
        if (path.parent / "metrics.npz").is_file()
    )[:run_count]
    if len(completed) != run_count:
        raise RuntimeError(f"expected {run_count} source runs, found {len(completed)}")

    correct = np.zeros((40, 30, 24), dtype=np.int64)
    comparisons = np.zeros_like(correct)
    error_sum = np.zeros_like(correct, dtype=np.float64)
    sources = []
    for mtime_ns, complete_path in completed:
        metrics_path = complete_path.parent / "metrics.npz"
        with np.load(metrics_path) as arrays:
            correct += arrays["correct32"].astype(np.int64)
            comparisons += arrays["comparisons"].astype(np.int64)
            error_sum += arrays["error_sum"].astype(np.float64)
        sources.append(
            {
                "metrics": str(metrics_path),
                "complete_mtime_ns": int(mtime_ns),
            }
        )

    step = 39
    with np.errstate(divide="ignore", invalid="ignore"):
        pck = np.where(
            comparisons[step] > 0,
            100.0 * correct[step] / comparisons[step],
            np.nan,
        )
        mean_error = np.where(
            comparisons[step] > 0,
            error_sum[step] / comparisons[step],
            np.nan,
        )
    entries = [
        {
            "step": step,
            "block": block,
            "head": head,
            "pck32": None if not np.isfinite(pck[block, head]) else float(pck[block, head]),
            "mean_error_px": (
                None
                if not np.isfinite(mean_error[block, head])
                else float(mean_error[block, head])
            ),
            "comparisons": int(comparisons[step, block, head]),
        }
        for block in range(30)
        for head in range(24)
    ]
    entries.sort(
        key=lambda row: (
            row["pck32"] is None,
            -(row["pck32"] if row["pck32"] is not None else -1.0),
            row["mean_error_px"]
            if row["mean_error_px"] is not None
            else float("inf"),
            row["block"],
            row["head"],
        )
    )
    frozen_top = list(manifest.get("entries") or [])
    if entries[: len(frozen_top)] != frozen_top:
        raise RuntimeError("reconstructed ranking does not reproduce the frozen Top100")

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ranking_step": step,
        "ranking_metric": "micro PCK@32; tie-break mean error, block, head",
        "source_manifest": str(args.manifest_path),
        "completed_runs_at_selection": run_count,
        "reproduces_frozen_top100": True,
        "head_scopes": {
            "top100": {"rank_start": 1, "rank_end": 100},
            "bottom100": {"rank_start": 621, "rank_end": 720},
            "all720": {"rank_start": 1, "rank_end": 720},
        },
        "entries": entries,
        "source_runs": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        f"wrote {args.output} · top={entries[0]['pck32']:.6f} · "
        f"bottom100_start={entries[-100]['pck32']:.6f} · "
        f"last={entries[-1]['pck32']:.6f}"
    )


if __name__ == "__main__":
    main()
