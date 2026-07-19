#!/usr/bin/env python3
"""Build a compact GT/Stage1b/LoRA/base-Wan trajectory comparison page."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_ROOT = Path("/data/gaoya/agent-data/outputs/physiciq_selected_three_model_qk")
DEFAULT_OUTPUT = Path("/data/gaoya/agent-data/outputs/physiciq_selected_three_model_qk_dashboard")
MODELS = (
    ("gt", "GT source · CoTracker", "gt_cotracker"),
    ("stage1b", "Stage1b step-004000 · Q/K L23 S39", "stage1b"),
    ("lora", "LoRA step-000500 · Q/K L23 S39", "lora"),
    ("baseline", "Wan2.2-TI2V-5B baseline · Q/K L23 S39", "baseline"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PhysicIQ Q/K tracks</title><style>
:root{--paper:#ece8dc;--ink:#17211e;--card:#fffdf8;--line:#b7b0a0;--accent:#b8492f}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 0 0,#d2704930,transparent 34rem),radial-gradient(circle at 100% 0,#2b806530,transparent 34rem),var(--paper);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(1820px,calc(100% - 24px));margin:auto;padding:28px 0 56px}h1,h2{font-family:Georgia,"Noto Serif CJK SC",serif}h1{font-size:clamp(38px,5vw,72px);line-height:.95;margin:0}.lead{max-width:1100px;color:#59635f;line-height:1.55}.controls{display:grid;grid-template-columns:2fr 1fr;gap:12px;margin:20px 0}label{font-size:11px;font-weight:900;text-transform:uppercase;letter-spacing:.08em}select{display:block;width:100%;margin-top:5px;padding:10px;background:var(--card);border:1px solid var(--ink);font-weight:800}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.card{background:#111815;color:#fff;padding:10px;min-width:0;border-radius:3px 18px 3px 3px}.card h2{font-size:17px;min-height:42px;margin:2px}.card video{display:block;width:100%;aspect-ratio:7/4;object-fit:contain;background:#020403}.note{margin-top:14px;padding:12px;background:var(--card);border:1px solid var(--line);line-height:1.55}@media(max-width:1100px){.grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:650px){.grid,.controls{grid-template-columns:1fr}}
</style></head><body><main><div style="color:var(--accent);font-weight:900;letter-spacing:.12em;font-size:12px">PHYSICIQ · TOKEN CORRESPONDENCE</div><h1>Q/K trajectories<br>against real motion</h1><p class="lead">同一组 SAM2 区域起始点。GT 列为原视频 CoTracker 轨迹；其余三列为各模型 Q/K argmax 轨迹，分别叠加在各自生成视频上。</p><div class="controls"><label>Case<select id="case"></select></label><label>Region<select id="region"></select></label></div><div class="grid" id="grid"></div><div class="note">轨迹起点对应 context/source pixel frame 4。三个生成模型均使用 seed 42、40 步去噪；当前视频固定显示 Q/K layer 23、step 39。不同生成视频的后续内容并不与 GT 像素对齐，因此这里只比较轨迹形态，不计算跨视频点误差。</div></main><script type="application/json" id="payload">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('payload').textContent),caseSel=document.getElementById('case'),regionSel=document.getElementById('region'),grid=document.getElementById('grid');
for(const item of data.cases){const o=document.createElement('option');o.value=item.case_key;o.textContent=item.label;caseSel.appendChild(o)}
function selected(){return data.cases.find(x=>x.case_key===caseSel.value)}
function regions(){const item=selected();regionSel.innerHTML='';for(const r of item.regions){const o=document.createElement('option');o.value=r;o.textContent=r;regionSel.appendChild(o)}}
function render(){const item=selected(),region=regionSel.value;grid.innerHTML='';for(const model of data.models){const card=document.createElement('section');card.className='card';const h=document.createElement('h2');h.textContent=model.label;const v=document.createElement('video');v.controls=true;v.muted=true;v.loop=true;v.preload='metadata';v.src=model.name==='gt'?`${model.asset}/cases/${item.case_key}/regions/${region}/tracks_cotracker.mp4`:`${model.asset}/cases/${item.case_key}/regions/${region}/tracks_qk_L23_S039.mp4`;card.append(h,v);grid.append(card)}}
caseSel.addEventListener('change',()=>{regions();render()});regionSel.addEventListener('change',render);regions();render();
</script></body></html>'''


def main() -> None:
    args = parse_args()
    result_root = args.result_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_payload = []
    for name, label, directory in MODELS:
        source = result_root / directory
        if not source.is_dir():
            raise FileNotFoundError(source)
        link = output / name
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            raise FileExistsError(link)
        os.symlink(source, link, target_is_directory=True)
        model_payload.append({"name": name, "label": label, "asset": name})
    gt_root = result_root / "gt_cotracker" / "cases"
    cases = []
    for case_dir in sorted(gt_root.iterdir()):
        if not (case_dir / "complete.json").is_file():
            continue
        manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
        regions = [item["region_name"] for item in manifest["query_regions"]]
        cases.append(
            {
                "case_key": case_dir.name,
                "label": Path(manifest["input_json"]).stem,
                "regions": regions,
            }
        )
        for model_name in ("stage1b", "lora", "baseline"):
            generated_case = result_root / model_name / "cases" / case_dir.name
            if not (generated_case / "complete.json").is_file():
                raise RuntimeError(f"missing completed {model_name}/{case_dir.name}")
    payload = json.dumps({"models": model_payload, "cases": cases}, ensure_ascii=False).replace("</", "<\\/")
    (output / "index.html").write_text(HTML.replace("__PAYLOAD__", payload), encoding="utf-8")
    print(f"Built {output / 'index.html'} with {len(cases)} cases")


if __name__ == "__main__":
    main()
