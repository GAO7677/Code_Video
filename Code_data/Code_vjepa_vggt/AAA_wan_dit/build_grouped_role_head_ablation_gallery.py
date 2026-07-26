#!/usr/bin/env python3
"""Build a baseline-versus-category gallery for one grouped ablation case."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

from grouped_head_targets import CATEGORY_TARGETS


CASE = "0613pybullet_sample_001460_w002"
MODELS = ("wan_lora", "xssc", "physrvg")
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _one_video(root: Path) -> Path:
    matches = sorted(root.glob(f"**/{CASE}.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {CASE}.mp4 under {root}, found {matches}")
    return matches[0]


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    baseline_root = args.baseline_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    videos: dict[str, dict[str, str]] = {}
    for model in MODELS:
        model_videos: dict[str, str] = {}
        baseline = _one_video(baseline_root / "generated" / model)
        baseline_target = assets / f"{model}_baseline.mp4"
        shutil.copy2(baseline, baseline_target)
        model_videos["baseline"] = baseline_target.name
        for category in CATEGORY_TARGETS:
            tag = f"self_attn_grouped_head_zero_category_{category.lower()}"
            source = _one_video(root / model / tag)
            target = assets / f"{model}_{category.lower()}.mp4"
            shutil.copy2(source, target)
            model_videos[category] = target.name
        videos[model] = model_videos

    source_json = Path(
        "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/"
        f"{CASE}.json"
    )
    source_payload = json.loads(source_json.read_text(encoding="utf-8"))
    source_video = Path(source_payload["source_video"])
    source_target = assets / "source_gt.mp4"
    shutil.copy2(source_video, source_target)

    target_rows = "".join(
        "<tr>"
        f"<th>{category}</th><td>"
        + ", ".join(f"B{block}H{head}" for block, head in targets)
        + "</td></tr>"
        for category, targets in CATEGORY_TARGETS.items()
    )
    sections = []
    for model in MODELS:
        cards = [
            "<figure><video controls loop muted preload='metadata' "
            f"src='assets/{videos[model]['baseline']}'></video>"
            "<figcaption>Baseline</figcaption></figure>"
        ]
        for category in CATEGORY_TARGETS:
            cards.append(
                "<figure><video controls loop muted preload='metadata' "
                f"src='assets/{videos[model][category]}'></video>"
                f"<figcaption>{category} category heads = 0</figcaption></figure>"
            )
        sections.append(
            f"<section><h2>{html.escape(MODEL_LABELS[model])}</h2>"
            f"<div class='grid'>{''.join(cards)}</div></section>"
        )

    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grouped Head-category ablations</title>
<style>
body{{margin:0;background:#0d0f11;color:#f4f5f6;font:15px/1.45 system-ui,sans-serif}}
header,section{{padding:18px 22px;border-bottom:1px solid #30343a}}
h1,h2{{margin:0 0 10px}}p,figcaption,td{{color:#bcc3ca}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}
figure{{margin:0}}video{{width:100%;display:block;background:#000}}
figcaption{{padding-top:5px}}table{{border-collapse:collapse}}th,td{{padding:4px 12px 4px 0;text-align:left}}
@media(max-width:1000px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>{CASE} · grouped Head-category ablations</h1>
<p>每个类别在六个 Block 中各关闭一个代表 self-attention Head。关闭发生在
Head 输出拼接后、output projection 前；其他 Head 和模块保持不变。</p>
<figure><video controls loop muted preload="metadata" src="assets/source_gt.mp4"></video>
<figcaption>Source/GT</figcaption></figure>
<table>{target_rows}</table></header>
{''.join(sections)}
</body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "case": CASE,
                "root": str(root),
                "targets": CATEGORY_TARGETS,
                "videos": videos,
                "source_gt": str(source_target),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output / "index.html")


if __name__ == "__main__":
    main()
