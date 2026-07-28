#!/usr/bin/env python3
"""Build same-baseline low/high Impact video comparisons."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_stc_motion import load_features, rms, track_state


DEFAULT_ANALYSIS_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_stc_motion_analysis"
)
DEFAULT_GALLERY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery"
)
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
COMPONENTS = (
    ("flow_vector", "RAFT 全像素光流"),
    ("flow_top05", "Top-5% 强运动"),
    ("object_trajectory", "物体轨迹"),
    ("object_speed", "物体速度"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--gallery-root", type=Path, default=DEFAULT_GALLERY_ROOT)
    parser.add_argument("--output-name", default="impact-examples")
    parser.add_argument("--context-frames", type=int, default=8)
    return parser.parse_args()


def replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.resolve())


def component_scores(
    analysis_root: Path,
    entry: dict[str, Any],
    baseline: dict[str, Any],
    context_frames: int,
) -> dict[str, float]:
    loaded = load_features(analysis_root, entry)
    baseline_loaded = load_features(analysis_root, baseline)
    if loaded is None or baseline_loaded is None:
        raise RuntimeError(f"Missing motion features for {entry['entry_id']}")
    arrays, metadata = loaded
    baseline_arrays, baseline_metadata = baseline_loaded
    start = context_frames - 1
    state = track_state(arrays, metadata, start)
    baseline_state = track_state(baseline_arrays, baseline_metadata, start)
    shared_objects = (
        state["object_mask"]
        & baseline_state["object_mask"]
        & state["valid_points"]
        & baseline_state["valid_points"]
    )

    def rmse(first: np.ndarray, second: np.ndarray) -> float:
        mask = np.isfinite(first) & np.isfinite(second)
        if not mask.any():
            return float("nan")
        return float(np.sqrt(np.mean(np.square(first[mask] - second[mask]))))

    errors = {
        "flow_vector": rmse(
            arrays["flow_norm"][start:].astype(np.float32),
            baseline_arrays["flow_norm"][start:].astype(np.float32),
        ),
        "flow_top05": rmse(
            arrays["flow_top05"][start:],
            baseline_arrays["flow_top05"][start:],
        ),
        "object_trajectory": rmse(
            state["corrected_displacement"][:, shared_objects],
            baseline_state["corrected_displacement"][:, shared_objects],
        ),
        "object_speed": rmse(
            state["object_speed_curve"],
            baseline_state["object_speed_curve"],
        ),
    }
    denominators = {
        "flow_vector": max(
            rms(baseline_arrays["flow_norm"][start:].astype(np.float32)), 0.002
        ),
        "flow_top05": max(rms(baseline_arrays["flow_top05"][start:]), 0.002),
        "object_trajectory": max(
            rms(
                baseline_state["corrected_displacement"][
                    :, baseline_state["object_mask"]
                ]
            ),
            0.01,
        ),
        "object_speed": max(rms(baseline_state["object_speed_curve"]), 0.002),
    }
    return {
        key: float(np.log1p(errors[key] / denominators[key]))
        for key, _ in COMPONENTS
    }


def choose_examples(frame: pd.DataFrame) -> list[dict[str, Any]]:
    candidates = frame[
        (frame["variant"] != "baseline")
        & frame["role"].isin(("S", "T", "ST"))
        & ~frame["tracking_failure"].astype(bool)
    ]
    examples = []
    for model in MODEL_LABELS:
        best: dict[str, Any] | None = None
        for seed, group in candidates[candidates["model"] == model].groupby("seed"):
            low = group.loc[group["impact_score"].idxmin()]
            high = group.loc[group["impact_score"].idxmax()]
            spread = float(high["impact_score"] - low["impact_score"])
            if best is None or spread > best["spread"]:
                best = {
                    "model": model,
                    "seed": int(seed),
                    "low": low,
                    "high": high,
                    "spread": spread,
                }
        if best is None:
            raise RuntimeError(f"No complete S/T/ST examples found for {model}")
        baseline = frame[
            (frame["model"] == model)
            & (frame["seed"] == best["seed"])
            & (frame["variant"] == "baseline")
        ]
        if len(baseline) != 1:
            raise RuntimeError(f"Expected one baseline for {model} seed {best['seed']}")
        best["baseline"] = baseline.iloc[0]
        examples.append(best)
    return examples


def metric_block(title: str, impact: float, scores: dict[str, float]) -> str:
    rows = "".join(
        f"<div><span>{html.escape(label)}</span><strong>{scores[key]:.3f}</strong></div>"
        for key, label in COMPONENTS
    )
    dominant_key = max(scores, key=scores.get)
    dominant_label = dict(COMPONENTS)[dominant_key]
    return (
        f'<div class="metrics"><div class="impact"><span>{html.escape(title)}</span>'
        f"<strong>{impact:.3f}</strong></div>{rows}"
        f'<p>最大贡献：{html.escape(dominant_label)}</p></div>'
    )


def main() -> None:
    args = parse_args()
    analysis_root = args.analysis_root.resolve()
    output_dir = args.gallery_root.resolve() / "multiseed" / args.output_name
    media_dir = output_dir / "media"
    inventory = json.loads((analysis_root / "inventory.json").read_text(encoding="utf-8"))
    entries = {entry["entry_id"]: entry for entry in inventory["entries"]}
    frame = pd.read_csv(analysis_root / "results" / "per_video_metrics.csv")
    examples = choose_examples(frame)
    manifest = []
    sections = []

    for example in examples:
        model = example["model"]
        seed = example["seed"]
        baseline_row = example["baseline"]
        baseline_entry = entries[baseline_row["entry_id"]]
        cards = []
        record: dict[str, Any] = {
            "model": model,
            "seed": seed,
            "spread": example["spread"],
            "videos": [],
        }
        for kind, label in (
            ("baseline", "Baseline"),
            ("low", "Low Impact"),
            ("high", "High Impact"),
        ):
            row = example[kind]
            entry = entries[row["entry_id"]]
            source = Path(entry["source"]["path"])
            link = media_dir / f"{model}__seed-{seed:06d}__{kind}.mp4"
            replace_symlink(link, source)
            impact = float(row["impact_score"])
            scores = (
                {key: 0.0 for key, _ in COMPONENTS}
                if kind == "baseline"
                else component_scores(
                    analysis_root,
                    entry,
                    baseline_entry,
                    args.context_frames,
                )
            )
            variant = str(row["variant"])
            cards.append(
                '<article class="video-cell">'
                f"<h3>{html.escape(label)}</h3>"
                f'<p class="variant">{html.escape(variant)}</p>'
                f'<video controls preload="metadata" src="media/{link.name}"></video>'
                f"{metric_block('Impact', impact, scores)}"
                "</article>"
            )
            record["videos"].append(
                {
                    "kind": kind,
                    "variant": variant,
                    "impact": impact,
                    "components": scores,
                    "source": str(source),
                    "media": str(link.relative_to(output_dir)),
                }
            )
        sections.append(
            '<section class="comparison">'
            f"<div class=\"section-title\"><h2>{MODEL_LABELS[model]}</h2>"
            f"<span>seed {seed} · Impact 跨度 {example['spread']:.3f}</span></div>"
            f'<div class="video-grid">{"".join(cards)}</div>'
            "</section>"
        )
        manifest.append(record)

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Impact 大小视频对照</title>
<style>
:root{{--bg:#f5f6f8;--panel:#fff;--line:#d9dde3;--text:#20242a;--muted:#66707c;--blue:#1769aa}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Arial,sans-serif}}
header{{position:sticky;top:0;z-index:3;background:#fff;border-bottom:1px solid var(--line);padding:12px 18px;display:flex;align-items:center;gap:14px}}
h1{{font-size:20px;margin:0}}header p{{margin:0;color:var(--muted);flex:1}}
button{{border:1px solid #aeb5bf;background:#fff;padding:7px 11px;border-radius:5px;cursor:pointer}}
main{{max-width:1500px;margin:auto;padding:18px}}.comparison{{margin-bottom:28px}}
.section-title{{display:flex;align-items:baseline;gap:12px;margin-bottom:8px}}h2{{font-size:18px;margin:0}}.section-title span,.variant{{color:var(--muted)}}
.video-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
.video-cell{{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px}}
h3{{font-size:15px;margin:0}}.variant{{margin:2px 0 8px}}video{{display:block;width:100%;aspect-ratio:16/9;background:#111}}
.metrics{{margin-top:8px}}.metrics>div{{display:flex;justify-content:space-between;border-top:1px solid #edf0f3;padding:4px 0}}
.metrics .impact{{font-size:16px;color:var(--blue);border-top:0}}.metrics p{{margin:5px 0 0;color:var(--muted);font-size:12px}}
.note{{background:#eef3f7;border-left:3px solid var(--blue);padding:9px 11px;margin:0 0 18px}}
@media(max-width:800px){{.video-grid{{grid-template-columns:1fr}}header p{{display:none}}}}
</style></head><body>
<header><h1>Impact 大小视频对照</h1><p>同一模型、同一 seed、同一 Baseline</p>
<button id="play">播放全部</button><button id="replay">从头播放</button></header>
<main><p class="note">低/高 Impact 仅表示相对 Baseline 的运动差异大小，不表示生成质量高低。四项数值是进入总分平均前的 log(1 + 归一化误差)。</p>
{''.join(sections)}</main>
<script>
const videos=[...document.querySelectorAll('video')];
document.getElementById('play').onclick=()=>videos.forEach(v=>v.play());
document.getElementById('replay').onclick=()=>videos.forEach(v=>{{v.currentTime=0;v.play()}});
</script></body></html>"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
