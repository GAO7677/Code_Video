#!/usr/bin/env python3
"""Build a one-case gallery for six protocol-consistent Head-zero runs."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

from consistent_head_targets import CATEGORIES, load_consistent_category_targets


CASE = "0613pybullet_sample_001460_w002"
CATEGORY_NAMES = {
    "S": "帧内空间",
    "ST": "帧内 + 相邻轨迹",
    "T": "轨迹传播",
    "P": "固定位置",
    "C": "历史 / context",
    "G": "全局聚合",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--classification-metadata", type=Path, required=True)
    parser.add_argument("--baseline-video", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _copy(source: Path, target: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, target)
    return target.name


def _one_result(root: Path) -> tuple[Path, dict[str, object]]:
    matches = list(root.glob(f"{CASE}.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one result JSON under {root}, found {matches}")
    payload = json.loads(matches[0].read_text(encoding="utf-8"))
    video = Path(str(payload["output_video"]))
    if not video.is_file():
        raise FileNotFoundError(video)
    return video, payload


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    targets, source = load_consistent_category_targets(
        args.classification_metadata,
    )

    case_payload = json.loads(
        args.source_json.expanduser().resolve().read_text(encoding="utf-8")
    )
    source_video = Path(case_payload["source_video"])
    context_video = Path(case_payload["input_video"])
    copied = {
        "source": _copy(source_video, assets / "source_gt.mp4"),
        "context": _copy(context_video, assets / "conditioning_context_8f.mp4"),
        "baseline": _copy(
            args.baseline_video.expanduser().resolve(),
            assets / "baseline.mp4",
        ),
    }

    result_payloads: dict[str, dict[str, object]] = {}
    for category in CATEGORIES:
        tag = f"self_attn_consistent_head_zero_category_{category.lower()}"
        video, payload = _one_result(root / tag)
        copied[category] = _copy(video, assets / f"zero_{category.lower()}.mp4")
        result_payloads[category] = payload

    cards = [
        (
            "baseline",
            "Baseline",
            "不做消融，seed=42",
            copied["baseline"],
        )
    ]
    for category in CATEGORIES:
        cards.append(
            (
                category,
                f"{category}-zero · {CATEGORY_NAMES[category]}",
                f"关闭 {len(targets[category])} 个一致分类 head",
                copied[category],
            )
        )
    card_html = "".join(
        f"""<figure data-kind="{html.escape(key)}">
<video controls loop muted preload="metadata" src="assets/{html.escape(video)}"></video>
<figcaption><strong>{html.escape(title)}</strong>
<span>{html.escape(detail)}</span></figcaption></figure>"""
        for key, title, detail, video in cards
    )
    count_rows = "".join(
        f"<tr><th>{category}</th><td>{CATEGORY_NAMES[category]}</td>"
        f"<td>{len(targets[category])}</td>"
        f"<td>{len({block for block, _ in targets[category]})}</td></tr>"
        for category in CATEGORIES
    )
    prompt = html.escape(str(case_payload.get("input_caption", "")))
    source_sha = html.escape(str(source["sha256"]))
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Six consistent Head-zero ablations</title>
<style>
:root{{--bg:#111315;--panel:#1b1e21;--line:#34393f;--text:#f2f3f4;--muted:#b7bec5;--accent:#e5bd55}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}}
header,main{{max-width:1600px;margin:auto;padding:18px 22px}}header{{border-bottom:1px solid var(--line)}}
h1,h2,p{{margin:0 0 10px}}h1{{font-size:24px}}h2{{font-size:17px}}
.toolbar{{position:sticky;top:0;z-index:5;display:flex;gap:8px;padding:10px 22px;background:#111315ee;border-bottom:1px solid var(--line)}}
button{{border:1px solid #59616a;background:#25292d;color:var(--text);padding:7px 11px;cursor:pointer}}
button:hover{{border-color:var(--accent)}}.reference{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0 18px}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}
figure{{margin:0;background:var(--panel);border:1px solid var(--line)}}video{{display:block;width:100%;background:#000;aspect-ratio:7/4}}
figcaption{{display:flex;justify-content:space-between;gap:12px;padding:8px 10px;color:var(--muted)}}
figcaption strong{{color:var(--text)}}table{{border-collapse:collapse;margin-top:10px}}th,td{{padding:4px 14px 4px 0;text-align:left;border-bottom:1px solid #292d31}}
code{{color:#f0cf78;overflow-wrap:anywhere}}.note{{color:var(--muted)}}
@media(max-width:1000px){{.grid,.reference{{grid-template-columns:1fr}}}}
</style></head><body>
<header>
<h1>{CASE} · 六类一致 Head-zero 消融</h1>
<p>Prompt: {prompt}</p>
<p class="note">所有结果使用同一 Wan+LoRA 权重、8 帧 context、49 帧输出、
40 个去噪步、CFG 5.0、seed 42。每次仅将一个类别的 self-attention head 输出在
output projection 前置零，positive/negative CFG 两条分支都执行。</p>
<table><thead><tr><th>类别</th><th>解释</th><th>Head 数</th><th>涉及 Block</th></tr></thead>
<tbody>{count_rows}</tbody></table>
<p class="note">分类源：step {source["denoise_step_one_based"]} · {html.escape(str(source["cfg_branch"]))} CFG branch ·
三协议一致 338/720 · SHA256 <code>{source_sha}</code></p>
</header>
<div class="toolbar">
<button id="play">同步播放生成结果</button>
<button id="pause">全部暂停</button>
<button id="reset">回到开头</button>
</div>
<main>
<h2>参考视频</h2>
<div class="reference">
<figure><video controls loop muted preload="metadata" src="assets/{copied["source"]}"></video>
<figcaption><strong>Ground truth</strong><span>完整原视频</span></figcaption></figure>
<figure><video controls loop muted preload="metadata" src="assets/{copied["context"]}"></video>
<figcaption><strong>Conditioning context</strong><span>实际输入的前 8 帧</span></figcaption></figure>
</div>
<h2>Baseline 与六类消融</h2>
<div class="grid" id="generated">{card_html}</div>
</main>
<script>
const generated=[...document.querySelectorAll("#generated video")];
document.getElementById("play").onclick=()=>{{for(const video of generated){{video.currentTime=0;video.play();}}}};
document.getElementById("pause").onclick=()=>generated.forEach(video=>video.pause());
document.getElementById("reset").onclick=()=>generated.forEach(video=>{{video.pause();video.currentTime=0;}});
</script></body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "case": CASE,
                "classification_source": source,
                "targets": {
                    category: [
                        {"block": block, "head": head}
                        for block, head in targets[category]
                    ]
                    for category in CATEGORIES
                },
                "results": result_payloads,
                "assets": copied,
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
