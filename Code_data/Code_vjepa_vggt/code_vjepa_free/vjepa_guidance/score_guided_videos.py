#!/usr/bin/env python3
"""Phase 0a: read-only scorer for guided vs baseline videos.

Loads a single cached WMRewardRunner (primary metric) and scores every video in
a directory, then joins the scores against the per-condition `mean_delta_post`
recorded in a probe-sweep `phaseN_summary.json`. Optionally also runs the
source-referenced physics_iq metric when a real continuation clip is given.

This does NOT modify any metric code under physv_eval/. It only imports
`score_case` / `WMRewardRunner` and calls them.

Run under an env that has diffusers + decord (e.g. conda `wan`):

    PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_try0526 \
    CUDA_VISIBLE_DEVICES=2 \
    /data/gaoya/miniconda3/envs/wan/bin/python score_guided_videos.py \
      --videos-dir /data/gaoya/agent-data/outputs/probe_sweep/phase4/videos \
      --summary-json /data/gaoya/agent-data/outputs/probe_sweep/phase4/phase4_summary.json \
      --out-json /data/gaoya/agent-data/outputs/probe_sweep/phase4/wmreward_scores.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos-dir", type=Path, required=True,
                   help="Directory of .mp4 videos to score (baseline + guided).")
    p.add_argument("--summary-json", type=Path, default=None,
                   help="phaseN_summary.json with per-condition mean_delta_post to join.")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--baseline-label", default="baseline",
                   help="Filename stem treated as the baseline for delta-wmreward.")
    p.add_argument("--source-video", type=Path, default=None,
                   help="Optional real continuation clip; enables physics_iq scoring.")
    p.add_argument("--physics-iq", action="store_true",
                   help="Also score physics_iq vs --source-video (needs the source).")
    return p.parse_args()


def load_delta_map(summary_json: Path | None) -> dict[str, float]:
    if summary_json is None or not summary_json.is_file():
        return {}
    data = json.loads(summary_json.read_text())
    out: dict[str, float] = {}
    for row in data.get("ranked", []):
        label = row.get("label")
        if label is not None and row.get("mean_delta_post") is not None:
            out[label] = float(row["mean_delta_post"])
    return out


def main() -> None:
    args = parse_args()

    # Import metrics read-only. WMRewardRunner caches the V-JEPA model in-process.
    from physv_eval.wmreward_official import WMRewardRunner

    videos = sorted(p for p in args.videos_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 files under {args.videos_dir}")

    delta_map = load_delta_map(args.summary_json)

    runner = WMRewardRunner()
    physics_iq_score_case = None
    if args.physics_iq:
        if args.source_video is None or not args.source_video.is_file():
            raise SystemExit("--physics-iq requires a valid --source-video")
        from physv_eval.single_case.physics_iq import score_case as physics_iq_score_case

    rows: list[dict[str, Any]] = []
    for video in videos:
        label = video.stem
        rec: dict[str, Any] = {"label": label, "video": str(video)}
        wm = runner.score(video)
        rec["surprise"] = wm["surprise"]
        rec["similarity"] = wm["similarity"]
        rec["mean_delta_post"] = delta_map.get(label)
        if physics_iq_score_case is not None:
            try:
                piq = physics_iq_score_case(str(video), source_video_path=str(args.source_video))
                rec["physics_iq_score"] = piq.get("score") if piq else None
            except Exception as exc:  # keep going; physics_iq is secondary
                rec["physics_iq_score"] = None
                rec["physics_iq_error"] = repr(exc)
        rows.append(rec)
        print(f"{label:24s} surprise={rec['surprise']:.4f} "
              f"sim={rec['similarity']:.4f} "
              f"mean_delta_post={rec['mean_delta_post']}")

    # Compute delta-wmreward vs baseline.
    base = next((r for r in rows if r["label"] == args.baseline_label), None)
    base_surprise = base["surprise"] if base else None
    for r in rows:
        r["delta_surprise_vs_base"] = (
            r["surprise"] - base_surprise if base_surprise is not None else None
        )

    out = {
        "baseline_label": args.baseline_label,
        "baseline_surprise": base_surprise,
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {args.out_json}")

    # Print the coupling table sorted by energy delta (most-negative first).
    coupled = [r for r in rows if r.get("mean_delta_post") is not None]
    coupled.sort(key=lambda r: r["mean_delta_post"])
    if coupled:
        print("\nlabel                    mean_delta_post   surprise   dSurprise_vs_base")
        for r in coupled:
            ds = r["delta_surprise_vs_base"]
            print(f"{r['label']:24s} {r['mean_delta_post']:+.5f}        "
                  f"{r['surprise']:.4f}    {ds:+.4f}" if ds is not None
                  else f"{r['label']:24s} {r['mean_delta_post']:+.5f}        {r['surprise']:.4f}    n/a")


if __name__ == "__main__":
    main()
