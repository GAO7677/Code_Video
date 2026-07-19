#!/usr/bin/env python3
"""Build a static dashboard for motion-region score inputs and results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/data/gaoya/agent-data/outputs/sam2_region_motion_roi_scores")
MODEL_LABELS = {
    "stage1b": "Stage1b step-004000",
    "lora": "LoRA step-000500",
    "gt": "GT matched first 25 frames",
}
METRIC_NAMES = (
    "motion_region_surprise",
    "static_region_surprise",
    "official_window_surprise",
    "videophy2_sa",
    "videophy2_pc",
    "cosmos_reason1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def load_ok(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok":
        raise RuntimeError(f"Incomplete score: {path}")
    return payload["result"]


def score_bundle(model_dir: Path) -> dict[str, Any]:
    wm = load_ok(model_dir / "scores" / "wmreward_region.json")
    vp = load_ok(model_dir / "scores" / "videophy2_roi.json")
    cosmos = load_ok(model_dir / "scores" / "cosmos_roi.json")
    return {
        "motion_region_surprise": wm["motion_region_surprise"],
        "static_region_surprise": wm["static_region_surprise"],
        "motion_minus_static": wm["motion_minus_static"],
        "official_window_surprise": wm["official_window_surprise"],
        "motion_token_ratio": wm["motion_token_ratio"],
        "motion_token_count": wm["motion_token_count"],
        "videophy2_sa": vp["sa"]["score"],
        "videophy2_pc": vp["pc"]["score"],
        "cosmos_reason1": cosmos["score"],
    }


def mean(values: list[float]) -> float:
    return float(statistics.mean(values))


def build_payload(root: Path) -> dict[str, Any]:
    models = [
        {"name": name, "label": label, "cases": []}
        for name, label in MODEL_LABELS.items()
    ]
    by_name = {model["name"]: model for model in models}
    crop_ratios = []
    for case_dir in sorted((root / "cases").glob("case_*")):
        metadata = json.loads((case_dir / "metadata.json").read_text())
        crop_ratios.append(float(metadata["crop_area_ratio"]))
        for model_name in MODEL_LABELS:
            model_dir = case_dir / model_name
            scores = score_bundle(model_dir)
            by_name[model_name]["cases"].append(
                {
                    "case_key": metadata["case_key"],
                    "prompt": metadata["prompt"],
                    "crop_box_xyxy": metadata["crop_box_xyxy"],
                    "crop_size_wh": metadata["crop_size_wh"],
                    "crop_area_ratio": metadata["crop_area_ratio"],
                    "shared_motion_area_ratio": metadata["shared_motion_area_ratio"],
                    "flow_threshold": metadata["flow_threshold"],
                    "scores": scores,
                    "wm_input_video": f"cases/{metadata['case_key']}/{model_name}/wm_input_full25.mp4",
                    "roi_input_video": f"cases/{metadata['case_key']}/{model_name}/motion_roi_input.mp4",
                    "overlay_video": f"cases/{metadata['case_key']}/{model_name}/motion_roi_overlay.mp4",
                    "metadata_json": f"cases/{metadata['case_key']}/metadata.json",
                    "score_root": f"cases/{metadata['case_key']}/{model_name}/scores",
                }
            )
    for model in models:
        if len(model["cases"]) != 50:
            raise RuntimeError(f"{model['name']}: expected 50 cases, got {len(model['cases'])}")
        model["summary"] = {
            name: mean([float(case["scores"][name]) for case in model["cases"]])
            for name in METRIC_NAMES
        }
        model["summary"]["motion_minus_static"] = mean(
            [float(case["scores"]["motion_minus_static"]) for case in model["cases"]]
        )
    return {
        "models": models,
        "protocol": {
            "case_count": 50,
            "video_count": 150,
            "frame_count": 25,
            "roi_definition": "Per-model dominant residual-flow tube, then shared union across Stage1b/LoRA/GT; fixed 16:9 crop with context margin.",
            "wmreward": "Full matched 25-frame input; token surprise is aggregated only where the shared motion mask overlaps at least 10% of a V-JEPA patch.",
            "vlm": "VideoPhy2 and Cosmos-Reason1 read the displayed fixed motion_roi_input.mp4 directly.",
            "crop_area_mean": mean(crop_ratios),
            "crop_area_median": float(statistics.median(crop_ratios)),
            "full_frame_crop_count": sum(value >= 0.999 for value in crop_ratios),
        },
    }


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Motion ROI Physical Scores</title><style>
:root{--ink:#17211f;--paper:#e9e3d4;--card:#fffdf7;--line:#bfb49d;--red:#bd3a28;--blue:#176c78;--green:#176b55;--muted:#67706c;--shadow:0 18px 50px #18231e1c}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 6% 0,#d9653e44,transparent 34rem),radial-gradient(circle at 92% 14%,#2a918044,transparent 32rem),repeating-linear-gradient(90deg,#0000 0 39px,#877b6710 40px),var(--paper);font-family:"Avenir Next","Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(1580px,calc(100% - 28px));margin:auto;padding:34px 0 72px}.eyebrow{color:var(--red);font-weight:900;font-size:12px;letter-spacing:.18em;text-transform:uppercase}h1,h2,h3{font-family:Georgia,"Noto Serif CJK SC",serif;margin:0}h1{font-size:clamp(48px,7vw,92px);line-height:.88;letter-spacing:-.055em}.lead{max-width:1100px;color:var(--muted);line-height:1.65}.controls{display:grid;grid-template-columns:2fr 1fr;gap:12px;margin:26px 0 14px}.control label{font-size:11px;font-weight:900;letter-spacing:.1em;text-transform:uppercase}.control select{display:block;width:100%;margin-top:5px;padding:12px;border:1px solid var(--ink);background:var(--card);font-weight:800}.viewer,.score-table,.protocol{background:#fffdf8e8;border:1px solid var(--line);box-shadow:var(--shadow);border-radius:4px 24px 4px 4px}.viewer{padding:16px}.viewer-head{display:flex;justify-content:space-between;gap:22px;align-items:flex-start;margin-bottom:12px}.viewer-head p{margin:6px 0;color:var(--muted)}.viewer-head a,.links a{color:var(--blue);font-weight:800}.media{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.panel{background:#101816;color:#fff;border-radius:3px 18px 3px 3px;padding:9px;min-width:0}.panel h3{font-size:17px;margin:4px 4px 8px}.panel .mode-note{font-size:11px;color:#b8c5c0;min-height:31px;margin:4px}.panel video{display:block;width:100%;background:#070b0a;aspect-ratio:16/9}.links{display:flex;flex-wrap:wrap;gap:10px;margin:10px 4px 2px}.score-table{padding:16px;margin-top:12px;overflow:auto}.score-table table{border-collapse:collapse;width:100%;min-width:850px}.score-table th,.score-table td{border:1px solid var(--line);padding:10px 12px;text-align:center}.score-table th:first-child,.score-table td:first-child{text-align:left}.score-table thead th{background:#17211f;color:#fff}.score-table td strong{display:block;font:700 24px/1 Georgia}.score-table td small{display:block;color:var(--muted);margin-top:6px}.score-table td.best{background:#d8eadf}.direction{font-size:10px;color:var(--muted);text-transform:uppercase}.protocol{padding:17px;margin-top:12px;line-height:1.6}.protocol strong{color:var(--red)}@media(max-width:1000px){.media{grid-template-columns:1fr}.panel .mode-note{min-height:0}}@media(max-width:650px){.controls{grid-template-columns:1fr}.viewer-head{display:block}}
</style></head><body><main><header><div class="eyebrow">One source case · three methods side by side</div><h1>Motion-Region<br>Physical Scores</h1><p class="lead">同一个 source case 的 Stage1b、LoRA、GT 始终在同一页横向展示。切换视频视图时三列同步变化，分数表逐指标比较当前 case，并同时给出各方法的 50-case 均值。</p></header><section class="controls"><div class="control"><label>Source case<select id="case"></select></label></div><div class="control"><label>Video view<select id="view"><option value="roi_input_video">Actual VideoPhy2 / Cosmos ROI input</option><option value="wm_input_video">Actual regional WMReward full input</option><option value="overlay_video">Motion mask and crop boundary</option></select></label></div></section><section class="viewer"><div class="viewer-head"><div><h2 id="title"></h2><p id="prompt"></p><p id="geometry"></p></div><a id="metadata">shared metadata.json</a></div><div class="media" id="media"></div></section><section class="score-table"><table><thead id="score-head"></thead><tbody id="score-body"></tbody></table></section><section class="protocol" id="protocol"></section></main><script id="payload" type="application/json">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('payload').textContent),caseEl=document.getElementById('case'),viewEl=document.getElementById('view');const fmt=(v,n=3)=>v==null?'NA':Number(v).toFixed(n);const cases=data.models[0].cases;caseEl.innerHTML=cases.map((c,i)=>`<option value="${i}">${c.case_key}</option>`).join('');const metricRows=[['Motion WM surprise','motion_region_surprise','lower',3],['Static WM surprise','static_region_surprise','reference',3],['Motion minus static','motion_minus_static','reference',3],['Full WM official','official_window_surprise','lower',3],['VideoPhy2-SA','videophy2_sa','higher',1],['VideoPhy2-PC','videophy2_pc','higher',1],['Cosmos-R1','cosmos_reason1','higher',1]];function entries(){const i=Number(caseEl.value)||0;return data.models.map(m=>({model:m,case:m.cases[i]}))}function bestIndex(key,direction){if(direction==='reference')return -1;const values=entries().map(x=>Number(x.case.scores[key]));const target=direction==='lower'?Math.min(...values):Math.max(...values);return values.indexOf(target)}function render(){const list=entries(),shared=list[0].case,view=viewEl.value,viewNotes={roi_input_video:'该视频直接送入 VideoPhy2 和 Cosmos-R1。',wm_input_video:'完整 25 帧输入；WM 只聚合共享运动 token。',overlay_video:'红色为共享主运动区域，黄色为固定 ROI。'};document.getElementById('title').textContent=shared.case_key;document.getElementById('prompt').textContent=shared.prompt;document.getElementById('geometry').textContent=`shared crop ${shared.crop_size_wh[0]}×${shared.crop_size_wh[1]} · ${fmt(100*shared.crop_area_ratio,1)}% of frame · shared motion ${fmt(100*shared.shared_motion_area_ratio,1)}%`;document.getElementById('metadata').href=shared.metadata_json;document.getElementById('media').innerHTML=list.map(({model,case:c})=>`<article class="panel"><h3>${model.label}</h3><p class="mode-note">${viewNotes[view]}</p><video controls muted loop playsinline preload="metadata" src="${c[view]}"></video><div class="links"><a href="${c.score_root}/wmreward_region.json">WM JSON</a><a href="${c.score_root}/videophy2_roi.json">VideoPhy2 JSON</a><a href="${c.score_root}/cosmos_roi.json">Cosmos JSON</a></div></article>`).join('');document.getElementById('score-head').innerHTML='<tr><th>Metric</th>'+list.map(x=>`<th>${x.model.label}</th>`).join('')+'</tr>';document.getElementById('score-body').innerHTML=metricRows.map(([label,key,direction,digits])=>{const best=bestIndex(key,direction);return `<tr><td><b>${label}</b><div class="direction">${direction==='lower'?'lower is better':direction==='higher'?'higher is better':'diagnostic'}</div></td>`+list.map(({model,case:c},i)=>`<td class="${i===best?'best':''}"><strong>${fmt(c.scores[key],digits)}</strong><small>50-case mean ${fmt(model.summary[key],digits)}</small></td>`).join('')+'</tr>'}).join('');const p=data.protocol;document.getElementById('protocol').innerHTML=`<strong>Protocol.</strong> ${p.roi_definition} Across 50 cases, mean crop area is ${fmt(100*p.crop_area_mean,1)}%, median ${fmt(100*p.crop_area_median,1)}%; ${p.full_frame_crop_count} cases retain the full frame. <strong>WMReward.</strong> ${p.wmreward} <strong>VLM.</strong> ${p.vlm}`}
caseEl.addEventListener('change',render);viewEl.addEventListener('change',render);render();
</script></body></html>'''


def main() -> None:
    root = parse_args().root.resolve()
    payload = build_payload(root)
    (root / "dashboard_data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    (root / "index.html").write_text(HTML.replace("__PAYLOAD__", serialized))
    print(f"dashboard: {root / 'index.html'}")


if __name__ == "__main__":
    main()
