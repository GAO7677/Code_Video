#!/usr/bin/env python3
"""Combine DINOv3 and official DINOv2 reconstruction outputs in one viewer."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov3-dir", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def prefixed(path, source_dir, output_dir):
    return (source_dir / path).resolve().relative_to(output_dir.resolve()).as_posix()


def build_html(payload):
    data = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>xSSC MOVi-C val reconstruction comparison</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#101317;color:#edf0f3;font:14px system-ui,sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:2;background:#101317f2;border-bottom:1px solid #353b43}}.bar,main{{max-width:1460px;margin:auto}}
.bar{{min-height:58px;padding:10px 16px;display:flex;align-items:center;gap:10px}}h1{{font-size:17px;margin:0 auto 0 0}}
select,button{{height:34px;border:1px solid #4a525c;border-radius:5px;background:#20252b;color:#f4f5f6;padding:0 10px}}button{{font-size:15px;cursor:pointer}}
main{{padding:18px 16px 40px}}.intro{{color:#abb4be;line-height:1.6;margin:0 0 18px}}.model{{border-top:3px solid var(--accent);padding-top:12px;margin:0 0 30px}}
.model h2{{font-size:16px;margin:0 0 5px}}.model-note{{color:#aab3bd;margin:0 0 10px}}video{{display:block;width:100%;background:#050607;border:1px solid #353b43}}
.metrics{{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:1px;background:#353b43;border:1px solid #353b43;margin:10px 0}}
.metric{{background:#191d22;padding:9px}}.metric span{{display:block;color:#96a0ab;font-size:12px;margin-bottom:3px}}.metric strong{{font-size:15px}}.chart{{display:block;width:min(760px,100%);background:white}}
@media(max-width:760px){{h1{{font-size:14px}}.metrics{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><header><div class="bar"><h1>xSSC MOVi-C val reconstruction comparison</h1><select id="case"></select><button id="restart">从头播放</button><button id="play">播放两行</button></div></header><main id="app"></main>
<script>
const DATA={data};const app=document.getElementById('app'),sel=document.getElementById('case'),play=document.getElementById('play'),restart=document.getElementById('restart');let videos=[];
function metric(k,v){{return `<div class="metric"><span>${{k}}</span><strong>${{v}}</strong></div>`}}function modelRow(m){{const x=m.metrics;return `<section class="model" style="--accent:${{m.color}}"><h2>${{m.model_label}}</h2><p class="model-note">${{m.backbone_label}} feature space · feature ${{m.feature_dim}} · slot ${{m.slot_dim}} · ${{m.slots}} slots</p><video muted playsinline preload="metadata" src="${{m.video}}"></video><div class="metrics">${{metric('MSE',x.mse.toFixed(4))}}${{metric('RMSE',x.rmse.toFixed(4))}}${{metric('Normalized RMSE',x.normalized_rmse.toFixed(4))}}${{metric('Mean patch cosine',x.mean_patch_cosine.toFixed(4))}}${{metric('Error q99',x.error_rmse_q99.toFixed(4))}}</div><img class="chart" src="${{m.frame_metrics_plot}}" alt="逐帧指标"></section>`}}
function render(i){{const c=DATA.cases[i];app.innerHTML=`<p class="intro">MOVi-C test index ${{c.index}} · ${{c.frames}} frames · 相同中心 crop 与相同 frame-0 GT bbox condition。dataset batch 保留完整 bbox 序列，但两个 RandSFQ2 都只读取 condit[:, 0] 初始化首帧 slots，后续帧由 transition + SlotAttention 递推。每行依次为 RGB、该 backbone 的目标特征 PCA、重构 PCA、误差热力图和 slots。DINOv2/DINOv3 的特征空间与尺度不同，MSE 只能在各自模型内部解释，不能用绝对值直接判定哪一个模型更好。</p>${{c.models.map(modelRow).join('')}}`;videos=[...app.querySelectorAll('video')];}}
DATA.cases.forEach((c,i)=>{{const o=document.createElement('option');o.value=i;o.textContent=`Case ${{c.index}}`;sel.appendChild(o)}});sel.onchange=()=>render(Number(sel.value));play.onclick=async()=>{{const paused=videos.every(v=>v.paused);if(paused){{const t=videos[0].currentTime;videos.forEach(v=>v.currentTime=t);await Promise.all(videos.map(v=>v.play().catch(()=>null)));play.textContent='暂停两行'}}else{{videos.forEach(v=>v.pause());play.textContent='播放两行'}}}};restart.onclick=async()=>{{videos.forEach(v=>v.currentTime=0);await Promise.all(videos.map(v=>v.play().catch(()=>null)));play.textContent='暂停两行'}};render(0);
</script></body></html>"""


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    sources = [
        (args.dinov3_dir.resolve(), "#22c55e"),
        (args.official_dir.resolve(), "#a855f7"),
    ]
    metadata = [(path, json.loads((path / "metadata.json").read_text()), color) for path, color in sources]
    expected = metadata[0][1]["case_indices"]
    for path, item, _ in metadata[1:]:
        if item["case_indices"] != expected:
            raise RuntimeError(f"case mismatch in {path}: {item['case_indices']} != {expected}")

    cases = []
    for position, index in enumerate(expected):
        models = []
        for source_dir, item, color in metadata:
            case = item["cases"][position]
            shapes = case["shapes"]
            models.append(
                {
                    "model_label": item["model_label"],
                    "backbone_label": item["backbone_label"],
                    "color": color,
                    "video": prefixed(case["video"], source_dir, output_dir),
                    "frame_metrics_plot": prefixed(case["frame_metrics_plot"], source_dir, output_dir),
                    "metrics": case["metrics"],
                    "feature_dim": shapes["feature"][2],
                    "slot_dim": shapes["slotz"][-1],
                    "slots": shapes["slotz"][2],
                }
            )
        cases.append({"index": index, "frames": metadata[0][1]["cases"][position]["frames"], "models": models})

    payload = {
        "title": "xSSC MOVi-C val reconstruction comparison",
        "case_indices": expected,
        "models": [
            {
                "model_label": item["model_label"],
                "backbone_label": item["backbone_label"],
                "checkpoint": item["checkpoint"],
                "config": item["config"],
            }
            for _, item, _ in metadata
        ],
        "cases": cases,
    }
    (output_dir / "comparison_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output_dir / "index.html").write_text(build_html(payload))
    print(json.dumps({"index": str(output_dir / "index.html"), "cases": expected}, indent=2))


if __name__ == "__main__":
    main()
