#!/usr/bin/env python3
"""Build an incremental per-case gallery for the test_5 S/T/ST sweep."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "test5_st_phased_seed851.json"
DEFAULT_GALLERY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/test5-st-phased-seed851"
)
MODEL_NAMES = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
ROLE_NAMES = {
    "S": "仅消融 S",
    "T": "仅消融 T",
    "ST": "联合消融 S+T",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gallery-root", type=Path, default=DEFAULT_GALLERY_ROOT)
    return parser.parse_args()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def safe_link(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(f"Refusing to replace non-link gallery asset: {destination}")
    destination.symlink_to(source)


def video_map(root: Path, cases: set[str]) -> dict[str, Path]:
    if not root.exists():
        return {}
    root = root.resolve()
    result: dict[str, Path] = {}
    for path in root.rglob("*.mp4"):
        if path.stem not in cases or path.stat().st_size <= 1024:
            continue
        if path.stem in result:
            raise RuntimeError(f"Duplicate case {path.stem} under {root}")
        result[path.stem] = path
    return result


def case_page(case_id: str) -> str:
    title = html.escape(case_id)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · S/T 分阶段消融</title>
<style>
:root{{--bg:#111416;--panel:#1a1e21;--line:#343a40;--text:#f2f4f5;--muted:#aab1b7;--accent:#56b9a6}}
*{{box-sizing:border-box}}body{{margin:0;padding-bottom:62px;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:5;padding:12px 18px;background:#111416f2;border-bottom:1px solid var(--line)}}
.top{{display:flex;gap:14px;align-items:center;justify-content:space-between}}.case-picker{{display:flex;gap:8px;align-items:center;min-width:0;flex:1}}.case-picker label{{font-weight:750}}select{{min-width:260px;max-width:760px;width:70%;padding:7px 9px;border:1px solid var(--line);background:#24292d;color:var(--text)}}
h1,h2,h3,p{{margin:0}}h1{{margin-top:7px;font-size:18px;overflow-wrap:anywhere}}h2{{font-size:17px;margin:22px 0 8px}}h3{{font-size:14px}}
.prompt{{margin-top:6px;color:var(--muted)}}.status{{white-space:nowrap;color:var(--accent);font-weight:700}}
main{{padding:14px 18px}}.references{{display:grid;grid-template-columns:repeat(2,minmax(280px,448px));gap:10px}}
.model-section{{margin-top:30px;border-top:1px solid var(--line)}}.model-banner{{display:flex;align-items:center;gap:12px;padding:11px 12px;background:#24292d;border-bottom:3px solid var(--accent)}}.model-banner h2{{margin:0;font-size:25px}}.seed-badge{{padding:3px 8px;border:1px solid #7d8790;background:#111416;color:#fff;font-size:14px;font-weight:800}}
table{{width:100%;border-collapse:collapse;table-layout:fixed}}th,td{{border:1px solid var(--line);padding:6px;vertical-align:top}}
thead th{{background:#22272b}}th:first-child{{width:125px;text-align:left}}tbody th{{background:#181c1f}}.baseline-cell{{background:#171b1e}}
figure{{margin:0}}video{{display:block;width:100%;aspect-ratio:7/4;object-fit:contain;background:#050607}}
figcaption{{padding-top:4px;color:var(--muted)}}.missing{{display:grid;place-items:center;aspect-ratio:7/4;background:#20252a;color:#858d94}}
.playbar{{position:fixed;z-index:8;left:0;right:0;bottom:0;display:grid;grid-template-columns:auto auto auto minmax(180px,1fr) auto;gap:8px;align-items:center;padding:9px 18px;background:#171a1df2;border-top:1px solid var(--line)}}
button{{border:1px solid var(--line);background:#24292d;color:#fff;padding:6px 10px;cursor:pointer}}input{{width:100%;accent-color:var(--accent)}}.time{{min-width:92px;text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}}
@media(max-width:900px){{.references{{grid-template-columns:1fr}}.model-section{{overflow-x:auto}}table{{min-width:980px}}}}
</style></head><body>
<header><div class="top"><div class="case-picker"><label for="case-select">Case</label><select id="case-select"><option>{title}</option></select></div><span class="status" id="status">读取中</span></div><h1 id="title">{title}</h1><p class="prompt" id="prompt"></p></header>
<main><h2>Reference</h2><div class="references" id="references"></div><div id="models"></div></main>
<div class="playbar"><button id="play" type="button">全部播放</button><button id="replay" type="button">重新播放</button><button id="pause" type="button">暂停</button><input id="timeline" type="range" min="0" max="1000" value="0"><span class="time" id="time">00:00 / 00:00</span></div>
<script>
let DATA=null,seeking=false;
const q=id=>document.getElementById(id);
function media(src,label){{return src?`<figure><video muted playsinline preload="metadata" src="${{src}}"></video><figcaption>${{label}}</figcaption></figure>`:`<div class="missing">Pending</div>`}}
function render(){{
 q("title").textContent=DATA.id;q("prompt").textContent=DATA.prompt||"";
 q("case-select").value=DATA.id;
 q("references").innerHTML=media(DATA.references.source,"Source / GT")+media(DATA.references.context,"8-frame context");
 q("models").innerHTML=DATA.models.map(m=>{{
  const rows=DATA.roles.map((r,index)=>`<tr><th>${{DATA.role_names[r]}}</th>${{index===0?`<td class="baseline-cell" rowspan="${{DATA.roles.length}}">${{media(DATA.videos.baseline[m],"Baseline · 无消融")}}</td>`:""}}${{DATA.stages.map(stage=>`<td>${{media(DATA.videos.stages[stage.key][m][r],stage.label)}}</td>`).join("")}}</tr>`).join("");
  return `<section class="model-section"><div class="model-banner"><h2>${{DATA.model_names[m]}}</h2><span class="seed-badge">SEED 851</span></div><table><thead><tr><th>消融类型</th><th>Baseline</th>${{DATA.stages.map(stage=>`<th>${{stage.label}}</th>`).join("")}}</tr></thead><tbody>${{rows}}</tbody></table></section>`;
 }}).join("");
 q("status").textContent=`结果 ${{DATA.completed_outputs}} / ${{DATA.expected_outputs}}`;
}}
function videos(){{return [...document.querySelectorAll("video")].filter(v=>Number.isFinite(v.duration)&&v.duration>0)}}
function fmt(s){{const x=Math.max(0,Math.floor(Number.isFinite(s)?s:0));return `${{String(Math.floor(x/60)).padStart(2,"0")}}:${{String(x%60).padStart(2,"0")}}`}}
function seek(f){{videos().forEach(v=>{{v.currentTime=Math.min(v.duration,Math.max(0,v.duration*f))}})}}
function sync(){{const v=videos()[0];if(!v)return;if(!seeking)q("timeline").value=String(Math.round(1000*v.currentTime/v.duration));q("time").textContent=`${{fmt(v.currentTime)}} / ${{fmt(v.duration)}}`}}
function play(){{seek(Number(q("timeline").value)/1000);videos().forEach(v=>{{v.loop=false;v.play().catch(()=>{{}})}})}}
function pause(){{videos().forEach(v=>v.pause())}}
async function load(){{try{{const r=await fetch(`case.json?t=${{Date.now()}}`,{{cache:"no-store"}});if(!r.ok)throw new Error(`HTTP ${{r.status}}`);const next=await r.json();const changed=!DATA||next.completed_outputs!==DATA.completed_outputs;DATA=next;if(changed&&!videos().some(v=>!v.paused))render();else q("status").textContent=`结果 ${{DATA.completed_outputs}} / ${{DATA.expected_outputs}}`}}catch(e){{q("status").textContent=`刷新失败: ${{e.message}}`}}}}
async function loadCaseOptions(){{const r=await fetch(`../../manifest.json?t=${{Date.now()}}`,{{cache:"no-store"}});if(!r.ok)return;const manifest=await r.json();q("case-select").innerHTML=manifest.cases.map(c=>`<option value="${{c.id}}">${{c.id}} · ${{c.completed_outputs}}/${{c.expected_outputs}}</option>`).join("");if(DATA)q("case-select").value=DATA.id}}
q("case-select").onchange=e=>{{window.location.href=`../${{encodeURIComponent(e.target.value)}}/`}};
q("play").onclick=play;q("replay").onclick=()=>{{q("timeline").value=0;seek(0);play()}};q("pause").onclick=pause;
q("timeline").onpointerdown=()=>{{seeking=true}};q("timeline").oninput=e=>{{seek(Number(e.target.value)/1000);sync()}};q("timeline").onchange=()=>{{seeking=false}};
setInterval(sync,250);load();loadCaseOptions();setInterval(()=>{{load();loadCaseOptions()}},15000);
</script></body></html>"""


def index_page() -> str:
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>test_5 · S/T 分阶段消融</title>
<style>
:root{--bg:#111416;--text:#f2f4f5;--muted:#aab1b7}body{display:grid;place-items:center;min-height:100vh;margin:0;background:var(--bg);color:var(--text);font:15px system-ui,sans-serif}.status{color:var(--muted)}
</style></head><body><div class="status">正在进入 case 页面…</div>
<script>
async function load(){const r=await fetch(`manifest.json?t=${Date.now()}`,{cache:"no-store"});const d=await r.json();if(d.cases.length)window.location.replace(`cases/${encodeURIComponent(d.cases[0].id)}/`)}
load();
</script></body></html>"""


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    output_root = Path(config["storage"]["output_root"]).expanduser().resolve()
    gallery_root = args.gallery_root.expanduser().resolve()
    input_paths = [
        Path(line.strip()).expanduser().resolve()
        for line in Path(config["input_list"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = {path.stem for path in input_paths}
    models = [str(value) for value in config["models"]]
    roles = [str(value) for value in config["roles"]]
    seed = int(config["seed"])
    stages = [
        {
            "key": f"{int(start):02d}_{int(end):02d}",
            "label": f"去噪步骤 [{int(start)},{int(end)})",
            "start": int(start),
            "end": int(end),
        }
        for start, end in config["step_ranges"]
    ]
    maps: dict[tuple[str, str], dict[str, Path]] = {}
    for model in models:
        baseline_root = (
            output_root
            / "generated"
            / model
            / f"seed-{seed:06d}"
            / "role-baseline"
        )
        maps[(model, "baseline")] = video_map(baseline_root, cases)
        for stage in stages:
            for role in roles:
                variant = f"{role}_steps{stage['key']}"
                variant_root = (
                    output_root
                    / "generated"
                    / model
                    / f"seed-{seed:06d}"
                    / f"role-{variant}"
                )
                maps[(model, variant)] = video_map(variant_root, cases)

    summaries = []
    expected_outputs = len(models) * (1 + len(roles) * len(stages))
    for source_json in input_paths:
        payload = json.loads(source_json.read_text(encoding="utf-8"))
        case_id = source_json.stem
        case_root = gallery_root / "cases" / case_id
        references: dict[str, str | None] = {"source": None, "context": None}
        for key, payload_key in (("source", "source_video"), ("context", "input_video")):
            value = payload.get(payload_key)
            if isinstance(value, str) and Path(value).is_file():
                source = Path(value)
                destination = case_root / "media" / "references" / f"{key}{source.suffix}"
                safe_link(source, destination)
                references[key] = str(destination.relative_to(case_root))

        videos: dict[str, Any] = {"baseline": {}, "stages": {}}
        completed = 0
        for model in models:
            source = maps[(model, "baseline")].get(case_id)
            url = None
            if source is not None:
                destination = case_root / "media" / "generated" / model / "baseline.mp4"
                safe_link(source, destination)
                url = str(destination.relative_to(case_root))
                completed += 1
            videos["baseline"][model] = url
        for stage in stages:
            videos["stages"][stage["key"]] = {}
            for model in models:
                videos["stages"][stage["key"]][model] = {}
                for role in roles:
                    variant = f"{role}_steps{stage['key']}"
                    source = maps[(model, variant)].get(case_id)
                    url = None
                    if source is not None:
                        destination = (
                            case_root
                            / "media"
                            / "generated"
                            / model
                            / f"{variant}.mp4"
                        )
                        safe_link(source, destination)
                        url = str(destination.relative_to(case_root))
                        completed += 1
                    videos["stages"][stage["key"]][model][role] = url
        case_data = {
            "schema_version": 1,
            "id": case_id,
            "source_json": str(source_json),
            "prompt": str(payload.get("input_caption", "")),
            "models": models,
            "model_names": MODEL_NAMES,
            "roles": roles,
            "role_names": ROLE_NAMES,
            "stages": stages,
            "references": references,
            "videos": videos,
            "completed_outputs": completed,
            "expected_outputs": expected_outputs,
        }
        atomic_write(
            case_root / "case.json",
            json.dumps(case_data, ensure_ascii=False, indent=2) + "\n",
        )
        atomic_write(case_root / "index.html", case_page(case_id))
        summaries.append(
            {
                "id": case_id,
                "prompt": case_data["prompt"],
                "completed_outputs": completed,
                "expected_outputs": expected_outputs,
            }
        )

    manifest = {
        "schema_version": 1,
        "experiment": config["experiment_name"],
        "seed": seed,
        "models": models,
        "roles": roles,
        "stages": stages,
        "cases": summaries,
    }
    atomic_write(
        gallery_root / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write(gallery_root / "index.html", index_page())
    total = sum(case["completed_outputs"] for case in summaries)
    print(
        f"[test5-gallery] cases={len(summaries)} outputs={total}/"
        f"{len(summaries) * expected_outputs} {gallery_root / 'index.html'}"
    )


if __name__ == "__main__":
    main()
