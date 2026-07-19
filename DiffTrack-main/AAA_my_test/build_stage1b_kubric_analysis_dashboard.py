#!/usr/bin/env python3
"""Build a combined Stage1b/LoRA SAM2-region correspondence dashboard."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


OUTPUTS_ROOT = Path("/data/gaoya/agent-data/outputs")
DEFAULT_STAGE1B = OUTPUTS_ROOT / "stage1b_kubric_step004000_sam2_regions_steps40"
DEFAULT_LORA = OUTPUTS_ROOT / "wan_openvid_0613pybullet_lora_step000500_sam2_regions_steps40"
DEFAULT_GT = (
    OUTPUTS_ROOT
    / "wan22_ti2v_5b_gt_real_sam2_regions_steps40"
)
DEFAULT_OUTPUT = OUTPUTS_ROOT / "sam2_region_generation_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1b-root", type=Path, default=DEFAULT_STAGE1B)
    parser.add_argument("--lora-root", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_model_payload(
    name: str,
    label: str,
    root: Path,
    asset_name: str,
    *,
    video_file: str = "generated.mp4",
    secondary_method: str = "hidden",
    secondary_label: str = "Hidden",
    supports_heatmaps: bool = True,
    token_stride: int = 32,
) -> dict:
    cases = []
    case_parent = root / "cases" if (root / "cases").is_dir() else root
    for case_dir in sorted(case_parent.glob("case_*")):
        manifest_path = case_dir / "manifest.json"
        metrics_path = case_dir / "metrics.json"
        if not manifest_path.is_file() or not metrics_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        display_layers = [
            layer for layer in manifest["layers"] if layer in (0, 5, 11, 17, 23, 29)
        ]
        metrics = [row for row in metrics if int(row["layer"]) in display_layers]
        cases.append(
            {
                "case_key": case_dir.name,
                "prompt": manifest["prompt"],
                "regions": manifest["query_regions"],
                "layers": display_layers,
                "steps": manifest["step_indices"],
                "metrics": metrics,
                "asset_root": (
                    f"{asset_name}/cases/{case_dir.name}"
                    if case_parent.name == "cases"
                    else f"{asset_name}/{case_dir.name}"
                ),
            }
        )
    if len(cases) != 50:
        raise RuntimeError(f"{name}: expected 50 complete cases, got {len(cases)}")
    return {
        "name": name,
        "label": label,
        "video_file": video_file,
        "secondary_method": secondary_method,
        "secondary_label": secondary_label,
        "supports_heatmaps": supports_heatmaps,
        "token_stride": token_stride,
        "cases": cases,
    }


def link_assets(output_dir: Path, name: str, target: Path) -> None:
    link = output_dir / name
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"dashboard asset path exists and is not a symlink: {link}")
    os.symlink(target.resolve(), link, target_is_directory=True)


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAM2 Region Correspondence</title><style>
:root{--ink:#14201d;--paper:#eee8da;--card:#fffdf8;--line:#cfc4af;--accent:#c5482e;--green:#126b58;--muted:#68716d;--shadow:0 18px 48px #1f2b261c}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 8% 0,#ecaa7d55,transparent 35rem),radial-gradient(circle at 92% 10%,#70ad9844,transparent 32rem),var(--paper);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(1500px,calc(100% - 28px));margin:auto;padding:34px 0 70px}h1,h2,h3{font-family:Georgia,"Noto Serif CJK SC",serif;margin:0}h1{font-size:clamp(44px,6.5vw,88px);line-height:.9;letter-spacing:-.05em}.eyebrow{color:var(--accent);font-size:12px;font-weight:900;letter-spacing:.17em;text-transform:uppercase}.lead{max-width:980px;color:var(--muted);line-height:1.65}.controls{display:grid;grid-template-columns:1fr 1.6fr 1fr 1fr;gap:10px;margin:25px 0 12px}label{font-size:11px;font-weight:900;letter-spacing:.08em;text-transform:uppercase}select{display:block;width:100%;margin-top:5px;padding:11px;border:1px solid var(--ink);background:var(--card);font-weight:800}.viewer,.card{background:#fffdf8e8;border:1px solid var(--line);box-shadow:var(--shadow);border-radius:4px 22px 4px 4px;padding:16px}.viewer-head{display:flex;justify-content:space-between;gap:20px;margin-bottom:13px}.viewer-head p{color:var(--muted);margin:6px 0}.media{display:grid;grid-template-columns:1.1fr 1fr 1fr;gap:10px}.panel{background:#101816;color:#fff;border-radius:3px 16px 3px 3px;padding:8px;min-width:0}.panel h3{font-size:16px;margin:4px 4px 9px}.panel video,.panel img{display:block;width:100%;background:#080c0b}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0}.metric span{font-size:11px;color:var(--muted);text-transform:uppercase}.metric b{display:block;font:700 29px/1 Georgia;margin:9px 0 5px}.tables{display:grid;grid-template-columns:1fr 1fr;gap:12px}.matrix{overflow:auto}.matrix h3{margin-bottom:10px}.matrix table{border-collapse:collapse;width:100%;background:var(--card)}th,td{border:1px solid var(--line);padding:7px 9px;text-align:right;font-size:12px}th:first-child,td:first-child{text-align:left}.note{font-size:12px;color:var(--muted);margin-top:18px}@media(max-width:900px){.controls,.media,.metrics,.tables{grid-template-columns:1fr}.viewer-head{display:block}}
</style></head><body><main><header><div class="eyebrow">Wan2.2-TI2V-5B · 40 denoising steps</div><h1>Object-wise<br>Correspondence Atlas</h1><p class="lead">query 来自最后一个 clean-context latent 对齐的像素帧 4。GroundingDINO 按物体短语分配独立框，SAM2 传播 mask；每个 object/background 区域独立抽取 8 个点。圆为 CoTracker，方框为 Q/K 或 hidden 匹配。</p></header>
<section class="controls"><label>Model<select id="model"></select></label><label>Case<select id="case"></select></label><label>Region<select id="region"></select></label><label>View<select id="view"><option value="tracks">Tracks</option><option value="heatmaps">Heatmaps</option></select></label></section>
<section class="viewer"><div class="viewer-head"><div><h2 id="title"></h2><p id="prompt"></p></div><a id="manifest">manifest.json</a></div><div class="media" id="media"></div></section>
<section class="metrics" id="metrics"></section><section class="tables"><article class="viewer matrix"><h3>Q/K PCK@32</h3><div id="qk-table"></div></article><article class="viewer matrix"><h3>Hidden PCK@32</h3><div id="hidden-table"></div></article></section>
<p class="note" id="protocol-note"></p>
</main><script id="payload" type="application/json">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('payload').textContent);const modelEl=document.getElementById('model'),caseEl=document.getElementById('case'),regionEl=document.getElementById('region'),viewEl=document.getElementById('view');const fmt=(x,n=2)=>x==null?'NA':Number(x).toFixed(n);modelEl.innerHTML=data.models.map((m,i)=>`<option value="${i}">${m.label}</option>`).join('');function currentModel(){return data.models[Number(modelEl.value)||0]}function currentCase(){return currentModel().cases[Number(caseEl.value)||0]}function currentRegion(){return currentCase().regions[Number(regionEl.value)||0]}function resetCases(){caseEl.innerHTML=currentModel().cases.map((c,i)=>`<option value="${i}">${c.case_key}</option>`).join('');caseEl.value=0;resetRegions()}function resetRegions(){regionEl.innerHTML=currentCase().regions.map((r,i)=>`<option value="${i}">${r.region_name}${r.region_phrase?' · '+r.region_phrase:''}</option>`).join('');regionEl.value=0;render()}function row(method,layer,step){const r=currentRegion();return currentCase().metrics.find(x=>x.region_name===r.region_name&&x.method===method&&x.layer===layer&&x.step_index===step)}function card(label,r,key,suffix=''){return `<article class="card metric"><span>${label}</span><b>${fmt(r?.[key])}${suffix}</b><small>L17 / S39 · ${r?.comparisons??0} matches</small></article>`}function matrix(method){const c=currentCase(),layers=c.layers,steps=c.steps;let h='<table><tr><th>Layer</th>'+steps.map(s=>`<th>S${s}</th>`).join('')+'</tr>';for(const l of layers){h+=`<tr><th>L${l}</th>`+steps.map(s=>{const r=row(method,l,s),v=r?.pck32??0,a=Math.max(0,Math.min(1,v/100));return `<td style="background:rgba(18,107,88,${.06+.68*a})">${fmt(v)}%</td>`}).join('')+'</tr>'}return h+'</table>'}function render(){const m=currentModel(),c=currentCase(),r=currentRegion(),base=c.asset_root,dir=`${base}/regions/${r.region_name}`,canHeat=viewEl.value==='heatmaps'&&m.supports_heatmaps,tag=`L17_S039`,secondary=m.secondary_method;document.getElementById('title').textContent=`${m.label} · ${c.case_key} · ${r.region_name}`;document.getElementById('prompt').textContent=r.region_phrase?`${r.region_phrase} | ${c.prompt}`:c.prompt;document.getElementById('manifest').href=`${base}/manifest.json`;document.getElementById('media').innerHTML=canHeat?`<article class="panel"><h3>SAM2 mask + query</h3><img src="${dir}/mask_points.png"></article><article class="panel"><h3>Q/K heatmap</h3><img src="${dir}/heatmap_qk_${tag}.png"></article><article class="panel"><h3>${m.secondary_label} heatmap</h3><img src="${dir}/heatmap_${secondary}_${tag}.png"></article>`:`<article class="panel"><h3>${m.name==='gt'?'GT video':'Generated video'}</h3><video controls muted loop src="${base}/${m.video_file}"></video></article><article class="panel"><h3>Q/K + CoTracker</h3><video controls muted loop src="${dir}/tracks_qk_${tag}.mp4"></video></article><article class="panel"><h3>${m.secondary_label} + CoTracker</h3><video controls muted loop src="${dir}/tracks_${secondary}_${tag}.mp4"></video></article>`;const q=row('qk',17,39),h=row(secondary,17,39);document.getElementById('metrics').innerHTML=card('Q/K PCK@32',q,'pck32','%')+card('Q/K mean error',q,'mean_error_px','px')+card(`${m.secondary_label} PCK@32`,h,'pck32','%')+card(`${m.secondary_label} error`,h,'mean_error_px','px');document.getElementById('qk-table').innerHTML=matrix('qk');document.getElementById('hidden-table').previousElementSibling.textContent=`${m.secondary_label} PCK@32`;document.getElementById('hidden-table').innerHTML=matrix(secondary);document.getElementById('protocol-note').textContent=m.name==='gt'?'GT real 使用同一 Wan2.2-TI2V-5B：完整25帧由 Wan VAE 编码，前8帧另编码为2个 clean latents；未来 GT latents 在各 scheduler step 加同一固定噪声后单次前向。token stride=32px，PCK@32 为一-token主指标。':'Wan2.2 生成分析：前8帧编码为2个 clean latents，query 对齐源 frame 4，空间 token stride=32px，因此 PCK@32 是一-token主指标。'}modelEl.addEventListener('change',resetCases);caseEl.addEventListener('change',resetRegions);regionEl.addEventListener('change',render);viewEl.addEventListener('change',render);resetCases();
</script></body></html>'''


def main() -> None:
    args = parse_args()
    stage1b = args.stage1b_root.expanduser().resolve()
    lora = args.lora_root.expanduser().resolve()
    gt = args.gt_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    link_assets(output, "stage1b", stage1b)
    link_assets(output, "lora", lora)
    link_assets(output, "gt", gt)
    payload = {
        "models": [
            load_model_payload("stage1b", "Stage1b step-004000", stage1b, "stage1b"),
            load_model_payload("lora", "LoRA step-000500", lora, "lora"),
            load_model_payload(
                "gt",
                "GT real · Wan2.2-TI2V-5B",
                gt,
                "gt",
                video_file="gt.mp4",
                secondary_method="hidden",
                secondary_label="Hidden",
                supports_heatmaps=True,
                token_stride=32,
            ),
        ]
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    (output / "dashboard_data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "index.html").write_text(
        HTML.replace("__PAYLOAD__", serialized), encoding="utf-8"
    )
    print(f"dashboard: {output / 'index.html'}")


if __name__ == "__main__":
    main()
