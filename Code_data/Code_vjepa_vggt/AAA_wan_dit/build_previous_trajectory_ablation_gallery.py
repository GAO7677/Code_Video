#!/usr/bin/env python3
"""Build source/baseline/previous-trajectory grouped ablation comparison."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path


CASE = "0613pybullet_sample_001460_w002"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--baseline-video", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _one_video(root: Path) -> Path:
    matches = [
        path
        for path in root.glob(f"**/{CASE}.mp4")
        if "_runtime" not in path.parts
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one video under {root}, found {matches}")
    return matches[0]


def main() -> None:
    args = parse_args()
    ablation_root = args.ablation_root.expanduser().resolve()
    baseline_video = args.baseline_video.expanduser().resolve()
    source_json = args.source_json.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    sources = {
        "source": Path(payload["source_video"]),
        "baseline": baseline_video,
        "prev_a": _one_video(ablation_root / "prev_a"),
        "prev_b": _one_video(ablation_root / "prev_b"),
    }
    names = {
        "source": "source_gt.mp4",
        "baseline": "baseline_no_ablation.mp4",
        "prev_a": "ablate_prev_A_47_heads.mp4",
        "prev_b": "ablate_prev_B_38_heads.mp4",
    }
    for key, source in sources.items():
        shutil.copy2(source, assets / names[key])
    labels = {
        "source": "Input context · 8 frames",
        "baseline": "Baseline · no ablation",
        "prev_a": "A-group ablation · 47 heads = 0",
        "prev_b": "B-group ablation · 38 heads = 0",
    }
    cards = "".join(
        "<figure><video controls loop muted preload='metadata' "
        f"src='assets/{names[key]}'></video>"
        f"<figcaption>{html.escape(labels[key])}</figcaption></figure>"
        for key in ("source", "baseline", "prev_a", "prev_b")
    )
    keyframes = output / "comparison_keyframes.jpg"
    metrics_path = output / "difference_metrics.json"
    analysis = ""
    if keyframes.is_file():
        analysis += (
            "<section class='analysis'><h2>相同帧关键帧对比</h2>"
            "<img src='comparison_keyframes.jpg' alt='Baseline, A47 and B38 keyframes'></section>"
        )
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        analysis += (
            "<section class='analysis'><h2>相对基线的逐像素差异</h2>"
            "<table><thead><tr><th>实验</th><th>生成段 MAE</th><th>生成段 PSNR</th>"
            "<th>峰值 MAE / 帧</th></tr></thead><tbody>"
            f"<tr><td>A47</td><td>{metrics['A47']['mae_generated_f8_48']:.3f}</td>"
            f"<td>{metrics['A47']['psnr_generated_f8_48_db']:.2f} dB</td>"
            f"<td>{metrics['A47']['peak_mae']:.3f} / {metrics['A47']['peak_frame']}</td></tr>"
            f"<tr><td>B38</td><td>{metrics['B38']['mae_generated_f8_48']:.3f}</td>"
            f"<td>{metrics['B38']['psnr_generated_f8_48_db']:.2f} dB</td>"
            f"<td>{metrics['B38']['peak_mae']:.3f} / {metrics['B38']['peak_frame']}</td></tr>"
            "</tbody></table><p>MAE 越高、PSNR 越低，表示与完全不消融结果差异越大。</p></section>"
        )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Previous-trajectory Head group ablation</title>
<style>
body{{margin:0;background:#111417;color:#f4f5f6;font:14px Arial,sans-serif;letter-spacing:0}}
header{{padding:15px 20px;border-bottom:1px solid #353b40}}h1{{font-size:21px;margin:0 0 6px}}p{{margin:4px 0;color:#bbc2c8}}
main{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;padding:14px 18px}}
figure{{margin:0}}video{{display:block;width:100%;background:#000}}figcaption{{padding:6px 0;color:#d8dde1;font-weight:700}}
section.analysis{{padding:14px 18px;border-top:1px solid #353b40}}section.analysis h2{{font-size:18px}}
section.analysis img{{display:block;width:100%;height:auto}}table{{border-collapse:collapse}}th,td{{padding:7px 12px;border:1px solid #454d53;text-align:left}}
@media(max-width:900px){{main{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>{CASE} · 前一轨迹 Head 组消融</h1>
<p>Wan+LoRA，8帧context，40步，CFG 5.0，seed 42。消融在self-attention heads拼接后、output projection前置零。</p>
<p>Prompt: {html.escape(str(payload["input_caption"]))}</p></header>
<main>{cards}</main>{analysis}</body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "case": CASE,
                "source_json": str(source_json),
                "caption": payload["input_caption"],
                "videos": {
                    key: {
                        "source": str(sources[key]),
                        "asset": f"assets/{names[key]}",
                    }
                    for key in sources
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[ablation-gallery] wrote {output / 'index.html'}")


if __name__ == "__main__":
    main()
