#!/usr/bin/env python3
"""Aggregate Stage1b ToyDataset correspondence metrics into a static dashboard."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/stage1b_kubric_generation_analysis_step004000"
)
METRICS = (
    "mean_error_px",
    "pck8",
    "pck16",
    "pck32",
    "pck_one_token",
    "top1_gt_rate",
    "mean_gt_rank",
    "mean_gt_probability",
    "normalized_entropy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def aggregate_rows(case_payloads: list[dict]) -> list[dict]:
    accumulators = defaultdict(lambda: defaultdict(float))
    for case in case_payloads:
        for row in case["rows"]:
            key = (
                row["method"],
                int(row["layer"]),
                int(row["step_index"]),
                float(row["timestep"]),
                None if row["sigma"] is None else float(row["sigma"]),
            )
            comparisons = int(row.get("comparisons", 0))
            accumulator = accumulators[key]
            accumulator["comparisons"] += comparisons
            accumulator["cases"] += 1
            for metric in METRICS:
                accumulator[metric] += float(row.get(metric, 0.0)) * comparisons
    output = []
    for key, accumulator in accumulators.items():
        comparisons = int(accumulator["comparisons"])
        row = {
            "method": key[0],
            "layer": key[1],
            "step_index": key[2],
            "timestep": key[3],
            "sigma": key[4],
            "cases": int(accumulator["cases"]),
            "comparisons": comparisons,
        }
        for metric in METRICS:
            row[metric] = accumulator[metric] / comparisons if comparisons else None
        output.append(row)
    return sorted(output, key=lambda row: (row["method"], row["layer"], row["step_index"]))


def find_row(rows: list[dict], method: str, layer: int, step: int) -> dict:
    matches = [
        row
        for row in rows
        if row["method"] == method
        and int(row["layer"]) == layer
        and int(row["step_index"]) == step
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {method}/L{layer}/S{step}, got {len(matches)}")
    return matches[0]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    case_payloads = []
    for case_dir in sorted((root / "cases").glob("case_*")):
        manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
        rows = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
        case_payloads.append(
            {
                "case_key": case_dir.name,
                "prompt": manifest["prompt"],
                "context_video": manifest["context_video"],
                "rows": rows,
                "qk_l29_s19": find_row(rows, "qk", 29, 19),
                "qk_l17_s19": find_row(rows, "qk", 17, 19),
                "hidden_l29_s14": find_row(rows, "hidden", 29, 14),
                "generated": f"cases/{case_dir.name}/generated.mp4",
                "qk_track": f"cases/{case_dir.name}/tracks_qk_L17_S019.mp4",
                "hidden_track": f"cases/{case_dir.name}/tracks_hidden_L17_S019.mp4",
                "qk_heatmap": f"cases/{case_dir.name}/heatmap_qk_L17_S019.png",
                "hidden_heatmap": f"cases/{case_dir.name}/heatmap_hidden_L17_S019.png",
            }
        )
    if len(case_payloads) != 50:
        raise RuntimeError(f"expected 50 completed cases, got {len(case_payloads)}")

    aggregate = aggregate_rows(case_payloads)
    best_qk = max(
        (row for row in aggregate if row["method"] == "qk"),
        key=lambda row: (row["pck32"], -row["mean_error_px"]),
    )
    best_hidden = max(
        (row for row in aggregate if row["method"] == "hidden"),
        key=lambda row: (row["pck32"], -row["mean_error_px"]),
    )
    summary = {
        "case_count": len(case_payloads),
        "record_count": sum(len(case["rows"]) for case in case_payloads),
        "checkpoint": "step-004000",
        "context": "first 8 pixel frames encoded as 2 clean latent frames",
        "token_grid": [7, 16, 28],
        "best_qk": best_qk,
        "best_hidden": best_hidden,
        "aggregate": aggregate,
        "cases": [{key: value for key, value in case.items() if key != "rows"} for case in case_payloads],
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(root / "aggregate_metrics.csv", aggregate)

    report = f"""# Stage1b Kubric step-004000 generation analysis

- Cases: {len(case_payloads)}
- Records: {summary['record_count']}
- Context: first 8 pixel frames, encoded by the actual generation path into 2 clean latent frames
- Token grid: 7 x 16 x 28
- Query: latent index 1
- Future targets: latent indices 2-6

## Best Q/K

- Layer: {best_qk['layer']}
- Denoising step: {best_qk['step_index']}
- Timestep / sigma: {best_qk['timestep']:.0f} / {best_qk['sigma']:.4f}
- Mean error: {best_qk['mean_error_px']:.2f}px
- PCK@32: {best_qk['pck32']:.2f}%
- GT top-1 rate: {best_qk['top1_gt_rate']:.2f}%
- Mean GT rank: {best_qk['mean_gt_rank']:.2f}

## Best hidden baseline

- Layer: {best_hidden['layer']}
- Denoising step: {best_hidden['step_index']}
- Timestep / sigma: {best_hidden['timestep']:.0f} / {best_hidden['sigma']:.4f}
- Mean error: {best_hidden['mean_error_px']:.2f}px
- PCK@32: {best_hidden['pck32']:.2f}%
- GT top-1 rate: {best_hidden['top1_gt_rate']:.2f}%

`PCK@32` is the meaningful token-scale metric for this 32-pixel token grid. `PCK@8` is dominated by token-center quantization and should not be used to compare layers.
"""
    (root / "aggregate_report.md").write_text(report, encoding="utf-8")

    payload = json.dumps(summary, ensure_ascii=False).replace("</", "<\\/")
    html = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage1b step-004000 · Context-to-Future Atlas</title><style>
:root{--ink:#17211e;--muted:#67716d;--paper:#efe9dc;--card:#fffdf7;--line:#d3c8b3;--red:#c44a32;--green:#137863;--blue:#216fa4;--shadow:0 18px 50px #2b352f1f}*{box-sizing:border-box}
body{margin:0;color:var(--ink);background:radial-gradient(circle at 8% 0,#e9a77b55,transparent 33rem),radial-gradient(circle at 92% 18%,#62a99544,transparent 32rem),var(--paper);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(1500px,calc(100% - 28px));margin:auto;padding:34px 0 70px}h1,h2,h3{font-family:Georgia,"Noto Serif CJK SC",serif;margin:0}h1{font-size:clamp(46px,7vw,92px);line-height:.9;letter-spacing:-.055em}.lead{max-width:900px;color:var(--muted);line-height:1.7}.cards{display:grid;grid-template-columns:1.4fr repeat(3,1fr);gap:13px;margin:25px 0}.card,.viewer{background:#fffdf7e8;border:1px solid var(--line);box-shadow:var(--shadow);border-radius:4px 22px 4px 4px;padding:18px}.verdict{background:#182521;color:#fff}.verdict p{color:#c9d4cf}.metric span{font-size:11px;text-transform:uppercase;color:var(--muted)}.metric b{display:block;font:600 35px/1 Georgia;margin:10px 0 6px}.controls{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:26px 0 13px}select{width:100%;padding:12px;border:1px solid var(--ink);background:var(--card);font-weight:700}.viewer-head{display:flex;justify-content:space-between;gap:20px;margin-bottom:14px}.viewer-head p{color:var(--muted);margin:5px 0}.videos{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.panel{background:#121a18;color:#fff;padding:8px;border-radius:3px 15px 3px 3px}.panel h3{font-size:17px;margin:4px 4px 9px}.panel video,.panel img{width:100%;display:block;background:#090d0c}.case-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:12px}.case-metrics .card{box-shadow:none}.case-metrics b{font-size:23px}.matrix{overflow:auto;margin-top:15px}.matrix table{width:100%;border-collapse:collapse;background:var(--card)}th,td{padding:8px 10px;border:1px solid var(--line);text-align:right;font-size:12px}th:first-child,td:first-child{text-align:left}.foot{color:var(--muted);margin-top:25px;font-size:12px}@media(max-width:900px){.cards,.videos,.case-metrics,.controls{grid-template-columns:1fr}.viewer-head{display:block}}
</style></head><body><main><header><div style="color:var(--red);font-weight:800;letter-spacing:.18em;text-transform:uppercase;font-size:12px">Wan2.2-TI2V-5B · Stage1b step-004000</div><h1>Context → Future<br>Correspondence Atlas</h1><p class="lead">50 个 ToyDataset base case；完整视频前 8 帧作为 context。Q/K 捕获位置是 self-attention 的 RMSNorm + 3D RoPE 之后、FlashAttention 之前，只统计 conditional 分支。</p></header>
<section class="cards"><article class="card verdict"><h2>最佳 Q/K 集中在最后一层、低噪声阶段</h2><p id="verdict"></p></article><article class="card metric"><span>Best Q/K PCK@32</span><b id="best-pck"></b><small>L29 / step19</small></article><article class="card metric"><span>Mean error</span><b id="best-error"></b><small>3641 visible matches</small></article><article class="card metric"><span>GT top-1</span><b id="best-top1"></b><small>正确 token 排名第一</small></article></section>
<section class="controls"><label>Case<select id="case"></select></label><label>View<select id="view"><option value="tracks">轨迹对照</option><option value="heatmaps">相关性热图</option></select></label></section>
<section class="viewer"><div class="viewer-head"><div><h2 id="case-title"></h2><p id="prompt"></p></div><a id="manifest" href="#">manifest</a></div><div class="videos" id="videos"></div><div class="case-metrics" id="case-metrics"></div></section>
<section class="viewer matrix"><h2>Q/K PCK@32 layer × step</h2><div id="matrix"></div></section><p class="foot">32px 是该 DiT token grid 的空间步长，因此 PCK@32 是主定位指标；PCK@8 受 token 中心量化下限支配。</p>
</main><script id="payload" type="application/json">__PAYLOAD__</script><script>
const d=JSON.parse(document.getElementById('payload').textContent),fmt=(x,n=2)=>Number(x).toFixed(n);document.getElementById('best-pck').textContent=fmt(d.best_qk.pck32)+'%';document.getElementById('best-error').textContent=fmt(d.best_qk.mean_error_px)+'px';document.getElementById('best-top1').textContent=fmt(d.best_qk.top1_gt_rate)+'%';document.getElementById('verdict').textContent=`L${d.best_qk.layer} / step ${d.best_qk.step_index} / sigma ${fmt(d.best_qk.sigma,4)}：PCK@32 ${fmt(d.best_qk.pck32)}%，平均误差 ${fmt(d.best_qk.mean_error_px)}px。`;
const cs=document.getElementById('case'),view=document.getElementById('view');cs.innerHTML=d.cases.map((c,i)=>`<option value="${i}">${c.case_key}</option>`).join('');function metric(title,r){return `<article class="card"><h3>${title}</h3><p><b>${fmt(r.pck32)}%</b> PCK@32 · <b>${fmt(r.mean_error_px)}px</b> error</p><p>GT top-1 ${fmt(r.top1_gt_rate)}% · rank ${fmt(r.mean_gt_rank)}</p></article>`}function render(){const c=d.cases[Number(cs.value)||0],heat=view.value==='heatmaps';document.getElementById('case-title').textContent=c.case_key;document.getElementById('prompt').textContent=c.prompt;document.getElementById('manifest').href=`cases/${c.case_key}/manifest.json`;document.getElementById('videos').innerHTML=heat?`<article class="panel"><h3>Q/K L17 S19 heatmap</h3><img src="${c.qk_heatmap}"></article><article class="panel"><h3>Hidden L17 S19 heatmap</h3><img src="${c.hidden_heatmap}"></article>`:`<article class="panel"><h3>Generated</h3><video controls muted loop src="${c.generated}"></video></article><article class="panel"><h3>Q/K L17 S19</h3><video controls muted loop src="${c.qk_track}"></video></article><article class="panel"><h3>Hidden L17 S19</h3><video controls muted loop src="${c.hidden_track}"></video></article>`;document.getElementById('case-metrics').innerHTML=metric('Q/K L29 S19',c.qk_l29_s19)+metric('Q/K L17 S19',c.qk_l17_s19)+metric('Hidden L29 S14',c.hidden_l29_s14)}cs.addEventListener('change',render);view.addEventListener('change',render);render();
const layers=[0,5,11,17,23,29],steps=[0,5,10,14,19],rows=d.aggregate.filter(x=>x.method==='qk');let h='<table><tr><th>Layer</th>'+steps.map(s=>`<th>S${s}</th>`).join('')+'</tr>';for(const l of layers){h+=`<tr><th>L${l}</th>`+steps.map(s=>{const r=rows.find(x=>x.layer===l&&x.step_index===s),a=Math.max(0,Math.min(1,r.pck32/100));return `<td style="background:rgba(19,120,99,${.08+.65*a})">${fmt(r.pck32)}%</td>`}).join('')+'</tr>'}h+='</table>';document.getElementById('matrix').innerHTML=h;
</script></body></html>'''.replace("__PAYLOAD__", payload)
    (root / "index.html").write_text(html, encoding="utf-8")
    print(f"dashboard: {root / 'index.html'}")


if __name__ == "__main__":
    main()
