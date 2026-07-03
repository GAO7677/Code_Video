#!/usr/bin/env python3
"""Read-only scorer for guided vs baseline videos.

Loads a single cached WMRewardRunner (primary metric) and scores every video in
a directory, then joins the scores against the per-condition `mean_delta_post`
recorded in a probe-sweep `phaseN_summary.json`. Optionally also runs
source-referenced physics_iq plus reference-free judge metrics (VideoPhy-2 PC /
Cosmos-Reason1) using the official single-case `score_case` entry points.

This does NOT modify any metric code under physv_eval/. It only imports
`score_case` / runner classes and calls them read-only.

Run under an env that has the needed deps (e.g. conda `wan` or `vphy`):

    PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_try0526 \
    CUDA_VISIBLE_DEVICES=2 \
    /data/gaoya/miniconda3/envs/wan/bin/python score_guided_videos.py \
      --videos-dir /data/gaoya/agent-data/outputs/probe_sweep/phase5/videos \
      --summary-json /data/gaoya/agent-data/outputs/probe_sweep/phase5/phase5_summary.json \
      --out-json /data/gaoya/agent-data/outputs/probe_sweep/phase5/wmreward_scores.json \
      --source-video /data/gaoya/AAA_test_video/0623/testdataset/025_Solid_Mechanics_0002_perspective-center_trimmed/physicIQ_0002_clip_2p5s_3p5s.mp4 \
      --physics-iq \
      --videophy2-task pc \
      --cosmos-reason1
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
    p.add_argument("--merge-from-json", type=Path, default=None,
                   help="Optional existing score JSON to merge per-label fields from before adding new metrics.")
    p.add_argument("--baseline-label", default="baseline",
                   help="Filename stem treated as the baseline for delta-wmreward.")
    p.add_argument("--skip-wmreward", action="store_true",
                   help="Do not run WMReward again. Useful when reusing an existing wmreward JSON and only adding judge metrics.")
    p.add_argument("--source-video", type=Path, default=None,
                   help="Optional real continuation clip; enables physics_iq scoring.")
    p.add_argument("--physics-iq", action="store_true",
                   help="Also score physics_iq vs --source-video (needs the source).")
    p.add_argument("--videophy2-task", default=None, choices=["sa", "pc", "rule"],
                   help="Optional VideoPhy-2 single-case task to run. Use 'pc' for physics consistency.")
    p.add_argument("--videophy2-device", default="cuda",
                   help="Device passed to VideoPhy2Runner when --videophy2-task is set.")
    p.add_argument("--videophy2-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="dtype passed to VideoPhy2Runner when --videophy2-task is set.")
    p.add_argument("--videophy2-num-frames", type=int, default=32,
                   help="num_frames passed to VideoPhy2Runner when --videophy2-task is set.")
    p.add_argument("--cosmos-reason1", action="store_true",
                   help="Also score Cosmos-Reason1 single-case metric.")
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


def load_existing_rows(existing_json: Path | None) -> dict[str, dict[str, Any]]:
    if existing_json is None or not existing_json.is_file():
        return {}
    data = json.loads(existing_json.read_text())
    rows = data.get("rows", [])
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = row.get("label")
        if isinstance(label, str):
            out[label] = dict(row)
    return out


def main() -> None:
    args = parse_args()

    videos = sorted(p for p in args.videos_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 files under {args.videos_dir}")

    delta_map = load_delta_map(args.summary_json)
    existing_rows = load_existing_rows(args.merge_from_json)

    runner = None
    physics_iq_score_case = None
    videophy2_score_case = None
    videophy2_runner = None
    cosmos_score_case = None
    cosmos_runner = None
    if not args.skip_wmreward:
        # Import metrics read-only. WMRewardRunner caches the V-JEPA model in-process.
        from physv_eval.wmreward_official import WMRewardRunner

        runner = WMRewardRunner()
    if args.physics_iq:
        if args.source_video is None or not args.source_video.is_file():
            raise SystemExit("--physics-iq requires a valid --source-video")
        from physv_eval.single_case.physics_iq import score_case as physics_iq_score_case
    if args.videophy2_task is not None:
        from physv_eval.single_case.videophy2 import score_case as videophy2_score_case
        from physv_eval.videophy2_auto import VideoPhy2Runner

        videophy2_runner = VideoPhy2Runner(
            device=args.videophy2_device,
            dtype=args.videophy2_dtype,
            num_frames=args.videophy2_num_frames,
        )
    if args.cosmos_reason1:
        from physv_eval.single_case.cosmos_reason1 import score_case as cosmos_score_case
        from physv_eval.cosmos_reason1_official import OfficialCosmosReason1Runner

        cosmos_runner = OfficialCosmosReason1Runner()

    rows: list[dict[str, Any]] = []
    for video in videos:
        label = video.stem
        rec: dict[str, Any] = dict(existing_rows.get(label, {}))
        rec["label"] = label
        rec["video"] = str(video)
        if runner is not None:
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
        if videophy2_score_case is not None and videophy2_runner is not None:
            try:
                vp2 = videophy2_score_case(
                    str(video),
                    task=args.videophy2_task,
                    runner=videophy2_runner,
                )
                rec["videophy2_task"] = args.videophy2_task
                rec["videophy2_score"] = vp2.get("score") if vp2 else None
            except Exception as exc:
                rec["videophy2_score"] = None
                rec["videophy2_error"] = repr(exc)
        if cosmos_score_case is not None and cosmos_runner is not None:
            try:
                cosmos = cosmos_score_case(str(video), runner=cosmos_runner)
                rec["cosmos_reason1_score"] = cosmos.get("score") if cosmos else None
            except Exception as exc:
                rec["cosmos_reason1_score"] = None
                rec["cosmos_reason1_error"] = repr(exc)
        rows.append(rec)
        extras: list[str] = []
        if rec.get("physics_iq_score") is not None:
            extras.append(f"physics_iq={rec['physics_iq_score']}")
        if rec.get("videophy2_score") is not None:
            extras.append(f"videophy2_{args.videophy2_task}={rec['videophy2_score']}")
        if rec.get("cosmos_reason1_score") is not None:
            extras.append(f"cosmos={rec['cosmos_reason1_score']}")
        extras_str = (" " + " ".join(extras)) if extras else ""
        surprise_str = f"{rec['surprise']:.4f}" if rec.get("surprise") is not None else "n/a"
        sim_str = f"{rec['similarity']:.4f}" if rec.get("similarity") is not None else "n/a"
        print(f"{label:24s} surprise={surprise_str} "
              f"sim={sim_str} "
              f"mean_delta_post={rec['mean_delta_post']}{extras_str}")

    # Compute delta-wmreward vs baseline.
    base = next((r for r in rows if r["label"] == args.baseline_label), None)
    base_surprise = base.get("surprise") if base else None
    base_similarity = base.get("similarity") if base else None
    base_physics_iq = base.get("physics_iq_score") if base else None
    base_videophy2 = base.get("videophy2_score") if base else None
    base_cosmos = base.get("cosmos_reason1_score") if base else None
    for r in rows:
        r["delta_surprise_vs_base"] = (
            r["surprise"] - base_surprise if base_surprise is not None else None
        )
        r["delta_similarity_vs_base"] = (
            r["similarity"] - base_similarity if base_similarity is not None else None
        )
        r["delta_physics_iq_vs_base"] = (
            r.get("physics_iq_score") - base_physics_iq
            if base_physics_iq is not None and r.get("physics_iq_score") is not None
            else None
        )
        r["delta_videophy2_vs_base"] = (
            r.get("videophy2_score") - base_videophy2
            if base_videophy2 is not None and r.get("videophy2_score") is not None
            else None
        )
        r["delta_cosmos_reason1_vs_base"] = (
            r.get("cosmos_reason1_score") - base_cosmos
            if base_cosmos is not None and r.get("cosmos_reason1_score") is not None
            else None
        )

    out = {
        "baseline_label": args.baseline_label,
        "baseline_surprise": base_surprise,
        "baseline_similarity": base_similarity,
        "baseline_physics_iq_score": base_physics_iq,
        "baseline_videophy2_score": base_videophy2,
        "baseline_cosmos_reason1_score": base_cosmos,
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
