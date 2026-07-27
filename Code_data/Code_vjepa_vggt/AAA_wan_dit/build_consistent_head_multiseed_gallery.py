#!/usr/bin/env python3
"""Build a seed-paired gallery for baseline and six Head-zero variants."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

from consistent_head_targets import CATEGORIES


CASE = "0613pybullet_sample_001460_w002"
VARIANTS = ("baseline", *CATEGORIES)
NAMES = {
    "baseline": "Baseline",
    "S": "S-zero · 帧内空间",
    "ST": "ST-zero · 帧内 + 相邻轨迹",
    "T": "T-zero · 轨迹传播",
    "P": "P-zero · 固定位置",
    "C": "C-zero · 历史 / context",
    "G": "G-zero · 全局聚合",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--legacy-ablation-root", type=Path, required=True)
    parser.add_argument("--legacy-baseline-video", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--verification-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    return parser.parse_args()


def _tag(variant: str) -> str:
    if variant == "baseline":
        return "baseline"
    return f"self_attn_consistent_head_zero_category_{variant.lower()}"


def main() -> None:
    args = parse_args()
    new_root = args.new_root.expanduser().resolve()
    legacy_root = args.legacy_ablation_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    verification = json.loads(
        args.verification_report.expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )
    if verification.get("all_checks") is not True:
        raise RuntimeError("Refusing to build gallery from unverified outputs")

    source_payload = json.loads(
        args.source_json.expanduser().resolve().read_text(encoding="utf-8")
    )
    reference_assets = {}
    for key, source in (
        ("source_gt", Path(source_payload["source_video"])),
        ("context_8f", Path(source_payload["input_video"])),
    ):
        target = assets / f"{key}.mp4"
        shutil.copy2(source, target)
        reference_assets[key] = target.name

    videos: dict[str, dict[str, str]] = {}
    for seed in args.seeds:
        videos[str(seed)] = {}
        for variant in VARIANTS:
            if seed == 42:
                if variant == "baseline":
                    source = args.legacy_baseline_video.expanduser().resolve()
                else:
                    source = legacy_root / _tag(variant) / f"{CASE}.mp4"
            else:
                source = (
                    new_root
                    / f"seed_{seed:04d}"
                    / _tag(variant)
                    / f"{CASE}.mp4"
                )
            if not source.is_file():
                raise FileNotFoundError(source)
            target = assets / f"seed{seed:04d}_{variant.lower()}.mp4"
            shutil.copy2(source, target)
            videos[str(seed)][variant] = target.name

    seed_buttons = "".join(
        f'<button class="seed-button" data-seed="{seed}">Seed {seed}</button>'
        for seed in args.seeds
    )
    sections = []
    for seed in args.seeds:
        cards = "".join(
            f"""<figure>
<video controls loop muted preload="metadata"
 src="assets/{html.escape(videos[str(seed)][variant])}"></video>
<figcaption><strong>{html.escape(NAMES[variant])}</strong></figcaption>
</figure>"""
            for variant in VARIANTS
        )
        sections.append(
            f'<section class="seed-section" data-seed="{seed}">'
            f"<h2>Seed {seed}</h2><div class=\"grid\">{cards}</div></section>"
        )
    prompt = html.escape(str(source_payload.get("input_caption", "")))
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multi-seed consistent Head-zero ablations</title>
<style>
:root{{--bg:#101214;--panel:#1a1d20;--line:#343a40;--text:#f2f3f4;--muted:#b7bec5;--accent:#e5bd55}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}}
header,main{{max-width:1600px;margin:auto;padding:18px 22px}}header{{border-bottom:1px solid var(--line)}}
h1,h2,p{{margin:0 0 10px}}h1{{font-size:24px}}h2{{font-size:18px}}
.toolbar{{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:8px;padding:10px 22px;background:#101214ee;border-bottom:1px solid var(--line)}}
button{{border:1px solid #59616a;background:#25292d;color:var(--text);padding:7px 11px;cursor:pointer}}
button:hover,.seed-button.active{{border-color:var(--accent);color:#ffe29a}}
.reference{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0 20px}}
.seed-section{{display:none}}.seed-section.active{{display:block}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}
figure{{margin:0;background:var(--panel);border:1px solid var(--line)}}video{{display:block;width:100%;background:#000;aspect-ratio:7/4}}
figcaption{{padding:8px 10px;color:var(--muted)}}figcaption strong{{color:var(--text)}}
.note{{color:var(--muted)}}@media(max-width:1000px){{.grid,.reference{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>{CASE} · 多 Seed 六类 Head-zero 消融</h1>
<p>Prompt: {prompt}</p>
<p class="note">Seed 42–46；每个 seed 内 baseline 与六类消融共享完全相同的初始噪声。
其余配置固定为 8 帧 context、49 帧输出、40 steps、CFG 5.0。</p></header>
<div class="toolbar">{seed_buttons}
<button id="play">同步播放当前 Seed</button>
<button id="pause">暂停</button><button id="reset">回到开头</button></div>
<main><h2>参考视频</h2><div class="reference">
<figure><video controls loop muted preload="metadata" src="assets/{reference_assets["source_gt"]}"></video><figcaption><strong>Ground truth</strong></figcaption></figure>
<figure><video controls loop muted preload="metadata" src="assets/{reference_assets["context_8f"]}"></video><figcaption><strong>Conditioning context · 8 frames</strong></figcaption></figure>
</div>{''.join(sections)}</main>
<script>
const sections=[...document.querySelectorAll(".seed-section")],buttons=[...document.querySelectorAll(".seed-button")];
function select(seed){{sections.forEach(s=>s.classList.toggle("active",s.dataset.seed===seed));buttons.forEach(b=>b.classList.toggle("active",b.dataset.seed===seed));history.replaceState(null,"",`#seed-${{seed}}`);}}
buttons.forEach(button=>button.onclick=()=>select(button.dataset.seed));
function currentVideos(){{return [...document.querySelector(".seed-section.active").querySelectorAll("video")];}}
document.getElementById("play").onclick=()=>currentVideos().forEach(video=>{{video.currentTime=0;video.play();}});
document.getElementById("pause").onclick=()=>currentVideos().forEach(video=>video.pause());
document.getElementById("reset").onclick=()=>currentVideos().forEach(video=>{{video.pause();video.currentTime=0;}});
const requested=location.hash.match(/^#seed-(\\d+)$/)?.[1],fallback="{args.seeds[0]}";
select(buttons.some(button=>button.dataset.seed===requested)?requested:fallback);
</script></body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "case": CASE,
                "seeds": args.seeds,
                "variants": list(VARIANTS),
                "videos": videos,
                "reference_assets": reference_assets,
                "verification_report": str(
                    args.verification_report.expanduser().resolve()
                ),
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
