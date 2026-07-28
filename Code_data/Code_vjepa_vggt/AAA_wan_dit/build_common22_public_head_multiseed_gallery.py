#!/usr/bin/env python3
"""Build an incremental case/seed gallery for common22 Head-role ablations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from common22_public_head_targets import load_public_head_targets


MODEL_NAMES = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
ROLE_NAMES = {
    "baseline": "Baseline",
    "S": "S-zero 空间局部",
    "T": "T-zero 运动轨迹",
    "P": "P-zero 固定位置时序",
    "C": "C-zero 上下文/历史",
    "G": "G-zero 全局聚合",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _ensure_symlink(path: Path, target: Path) -> None:
    if path.is_symlink():
        if path.resolve() == target.resolve():
            return
        path.unlink()
    elif path.exists():
        raise RuntimeError(f"Refusing to replace non-symlink gallery path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=target.is_dir())


def _cases(path: Path) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        json_path = Path(line.strip()).expanduser().resolve()
        if json_path in seen:
            continue
        seen.add(json_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        output.append(
            {
                "id": json_path.stem,
                "json": str(json_path),
                "prompt": payload.get("input_caption", ""),
                "source_video": payload.get("source_video"),
                "context_video": payload.get("input_video"),
            }
        )
    return output


def _find_video(root: Path, case: str) -> Path | None:
    matches = [path for path in root.rglob(f"{case}.mp4") if path.stem == case]
    if len(matches) > 1:
        raise RuntimeError(f"Duplicate videos for {case} under {root}: {matches}")
    return matches[0] if matches else None


def _state_counts(root: Path, expected_jobs: int) -> dict[str, int]:
    counts = {"complete": 0, "running": 0, "failed": 0, "pending": 0}
    for path in root.glob("state/*/seed-*/role-*.json"):
        try:
            status = json.loads(path.read_text(encoding="utf-8")).get("status")
        except json.JSONDecodeError:
            status = "failed"
        counts[status if status in counts else "failed"] += 1
    counts["pending"] = expected_jobs - sum(
        counts[key] for key in ("complete", "running", "failed")
    )
    return counts


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["storage"]["output_root"]).expanduser().resolve()
    baseline_root = Path(config["storage"]["baseline_root"]).expanduser().resolve()
    gallery = Path(config["storage"]["gallery_root"]).expanduser().resolve()
    report = Path(config["public_head_report"]).expanduser().resolve()
    targets, source = load_public_head_targets(report)
    cases = _cases(Path(config["input_list"]).expanduser().resolve())
    seeds = [int(value) for value in config["seeds"]]
    models = [str(value) for value in config["models"]]
    roles = [str(value) for value in config["roles"]]
    expected_jobs = len(seeds) * len(models) * len(roles)

    gallery.mkdir(parents=True, exist_ok=True)
    media = gallery / "media"
    _ensure_symlink(media / "ablations", output_root / "generated")
    _ensure_symlink(media / "baselines", baseline_root)
    references = media / "references"
    references.mkdir(parents=True, exist_ok=True)
    for case in cases:
        for kind in ("source_video", "context_video"):
            value = case.get(kind)
            if isinstance(value, str) and Path(value).expanduser().is_file():
                suffix = Path(value).suffix or ".mp4"
                _ensure_symlink(
                    references / f"{case['id']}__{kind}{suffix}",
                    Path(value).expanduser().resolve(),
                )

    videos: dict[str, dict[str, dict[str, dict[str, str | None]]]] = {}
    completed_cells = 0
    total_cells = len(cases) * len(seeds) * len(models) * (1 + len(roles))
    for case in cases:
        case_map: dict[str, dict[str, dict[str, str | None]]] = {}
        for seed in seeds:
            seed_map: dict[str, dict[str, str | None]] = {}
            for model in models:
                variants: dict[str, str | None] = {}
                baseline_dir = (
                    baseline_root / model / f"seed-{seed:06d}" / "generated" / model
                )
                baseline = _find_video(baseline_dir, case["id"])
                variants["baseline"] = (
                    f"media/baselines/{model}/seed-{seed:06d}/generated/"
                    f"{model}/{baseline.relative_to(baseline_dir).as_posix()}"
                    if baseline
                    else None
                )
                completed_cells += int(baseline is not None)
                for role in roles:
                    role_dir = (
                        output_root
                        / "generated"
                        / model
                        / f"seed-{seed:06d}"
                        / f"role-{role}"
                    )
                    video = _find_video(role_dir, case["id"]) if role_dir.is_dir() else None
                    variants[role] = (
                        f"media/ablations/{model}/seed-{seed:06d}/role-{role}/"
                        f"{video.relative_to(role_dir).as_posix()}"
                        if video
                        else None
                    )
                    completed_cells += int(video is not None)
                seed_map[model] = variants
            case_map[str(seed)] = seed_map
        videos[case["id"]] = case_map

    manifest = {
        "experiment": config["experiment_name"],
        "cases": cases,
        "seeds": seeds,
        "models": models,
        "model_names": MODEL_NAMES,
        "roles": roles,
        "role_names": ROLE_NAMES,
        "target_counts": {role: len(targets[role]) for role in roles},
        "target_source": source,
        "generation": config["generation"],
        "expected_jobs": expected_jobs,
        "state_counts": _state_counts(output_root, expected_jobs),
        "completed_video_cells": completed_cells,
        "total_video_cells": total_cells,
        "videos": videos,
    }
    _atomic_text(
        gallery / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n",
    )

    document = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Common22 public Head ablations</title>
<style>
:root{--bg:#101214;--panel:#1a1d20;--line:#383d43;--text:#f3f4f5;--muted:#b8bec5;--accent:#e8bd50}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,sans-serif;letter-spacing:0}
header{padding:14px 18px;border-bottom:1px solid var(--line)}h1,h2,p{margin:0 0 8px}h1{font-size:22px}h2{font-size:16px}
.toolbar{position:sticky;top:0;z-index:4;display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px 18px;background:#101214ee;border-bottom:1px solid var(--line)}
select,button{height:34px;border:1px solid #59616a;background:#25292d;color:var(--text);padding:0 10px}button{cursor:pointer}.status{color:var(--accent);font-weight:700}
main{padding:14px 18px;overflow-x:auto}.refs{display:grid;grid-template-columns:repeat(2,minmax(260px,448px));gap:10px;margin-bottom:14px}
table{border-collapse:collapse;min-width:1540px;width:100%}th,td{border:1px solid var(--line);padding:6px;vertical-align:top}th{background:#202429;position:sticky;top:55px;z-index:2}
th:first-child,td:first-child{position:sticky;left:0;background:#171a1d;z-index:1;width:120px}.card{min-width:220px}
video{display:block;width:100%;aspect-ratio:7/4;background:#000}.missing{display:grid;place-items:center;aspect-ratio:7/4;background:#24282c;color:#8f969e}
figcaption{padding-top:4px;color:var(--muted)}figure{margin:0}.note{color:var(--muted)}
@media(max-width:850px){.refs{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Common22 跨模型公共稳定 Head 分类消融</h1>
<p>20 cases × 22 paired seeds × 3 models；每个类别同时将该类全部公共 Head 的 self-attention 输出在 output projection 前置零。</p>
<p class="note" id="prompt"></p></header>
<div class="toolbar"><label>Case <select id="case"></select></label><label>Seed <select id="seed"></select></label>
<button id="play">同步播放</button><button id="pause">暂停</button><button id="reset">回到开头</button><span class="status" id="status">读取中</span></div>
<main><div class="refs" id="refs"></div><table><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table></main>
<script>
let DATA=null;const variants=["baseline","S","T","P","C","G"];
const q=id=>document.getElementById(id);
function options(node,values,label){node.innerHTML=values.map(v=>`<option value="${v}">${label(v)}</option>`).join("")}
function refPath(item,key){if(!item[key])return null;const ext=item[key].split(".").pop();return `media/references/${item.id}__${key}.${ext}`}
function card(src,label){return src?`<figure><video controls loop muted preload="metadata" src="${src}"></video><figcaption>${label}</figcaption></figure>`:`<div class="missing">${label} · 等待生成</div>`}
function render(){
 const c=q("case").value,s=q("seed").value,item=DATA.cases.find(x=>x.id===c);q("prompt").textContent=`Prompt: ${item.prompt}`;
 q("refs").innerHTML=card(refPath(item,"source_video"),"Source / GT")+card(refPath(item,"context_video"),"8-frame context");
 q("head").innerHTML="<th>模型</th>"+variants.map(v=>`<th>${DATA.role_names[v]}${v==="baseline"?"":` · ${DATA.target_counts[v]} Heads`}</th>`).join("");
 q("body").innerHTML=DATA.models.map(m=>`<tr><td><strong>${DATA.model_names[m]}</strong></td>${variants.map(v=>`<td class="card">${card(DATA.videos[c][s][m][v],DATA.role_names[v])}</td>`).join("")}</tr>`).join("");
}
async function load(){
 DATA=await fetch("manifest.json?"+Date.now(),{cache:"no-store"}).then(r=>r.json());
 const oldCase=q("case").value,oldSeed=q("seed").value;options(q("case"),DATA.cases,x=>x.id);options(q("seed"),DATA.seeds,x=>`Seed ${x}`);
 if(DATA.cases.some(x=>x.id===oldCase))q("case").value=oldCase;if(DATA.seeds.map(String).includes(oldSeed))q("seed").value=oldSeed;
 const sc=DATA.state_counts;q("status").textContent=`视频 ${DATA.completed_video_cells}/${DATA.total_video_cells} · 任务 complete ${sc.complete}/${DATA.expected_jobs} · running ${sc.running} · failed ${sc.failed}`;
 render();
}
q("case").onchange=render;q("seed").onchange=render;
const videos=()=>[...document.querySelectorAll("video")];
q("play").onclick=()=>videos().forEach(v=>{v.currentTime=0;v.play()});q("pause").onclick=()=>videos().forEach(v=>v.pause());q("reset").onclick=()=>videos().forEach(v=>{v.pause();v.currentTime=0});
load();setInterval(load,10000);
</script></body></html>"""
    _atomic_text(gallery / "index.html", document)
    print(
        f"[gallery] {gallery / 'index.html'}; "
        f"videos={completed_cells}/{total_cells}; states={manifest['state_counts']}"
    )


if __name__ == "__main__":
    main()
