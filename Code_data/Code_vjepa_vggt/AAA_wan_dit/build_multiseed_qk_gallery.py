#!/usr/bin/env python3
"""Build a case-selectable seed-by-model video and Q@K gallery."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

from run_paired_query_50seeds_worker import _deduplicated_paths, _video_map


MODEL_NAMES = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _status(path: Path) -> str:
    if not path.is_file():
        return "missing"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status", "missing"))
    except json.JSONDecodeError:
        return "invalid"


def _link(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return str(target)
    if target.exists() or target.is_symlink():
        target.unlink()
    os.symlink(source.resolve(), target)
    return str(target)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    snapshot = json.loads(
        args.snapshot.expanduser().resolve().read_text(encoding="utf-8")
    )
    output_root = Path(config["storage"]["output_root"]).expanduser().resolve()
    prerequisite = Path(config["storage"]["prerequisite_root"]).expanduser().resolve()
    pending_root = output_root / "pending_selected_qk"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = _deduplicated_paths(
        Path(config["input"]["json_list"]).expanduser().resolve()
    )
    cases = [path.stem for path in source_paths]
    case_set = set(cases)
    models = [str(model) for model in config["models"]]
    seeds = [int(seed) for seed in config["seeds"]]
    pending = {
        (model, int(item["seed"]))
        for model, rows in snapshot["pending"].items()
        for item in rows
    }
    records = {}
    for seed in seeds:
        seed_name = f"seed-{seed:06d}"
        records[str(seed)] = {}
        for model in models:
            videos = _video_map(
                prerequisite / "pass1" / model / seed_name / "generated" / model,
                case_set,
            )
            qk_status = _status(
                pending_root / "state" / model / f"{seed_name}.json"
            )
            model_cases = {}
            for case in cases:
                video_rel = None
                if case in videos:
                    target = (
                        output_dir
                        / "media"
                        / seed_name
                        / model
                        / f"{case}.mp4"
                    )
                    _link(videos[case], target)
                    video_rel = str(target.relative_to(output_dir))
                qk_source = (
                    pending_root
                    / "heatmaps"
                    / seed_name
                    / model
                    / case
                    / "all_roles_softmax_qk.png"
                )
                qk_rel = None
                if qk_source.is_file():
                    target = (
                        output_dir
                        / "heatmaps"
                        / seed_name
                        / model
                        / case
                        / qk_source.name
                    )
                    _link(qk_source, target)
                    qk_rel = str(target.relative_to(output_dir))
                temporal_source = (
                    pending_root
                    / "heatmaps"
                    / seed_name
                    / model
                    / case
                    / "all_roles_temporal_13x13.png"
                )
                temporal_rel = None
                if temporal_source.is_file():
                    target = (
                        output_dir
                        / "heatmaps"
                        / seed_name
                        / model
                        / case
                        / temporal_source.name
                    )
                    _link(temporal_source, target)
                    temporal_rel = str(target.relative_to(output_dir))
                model_cases[case] = {
                    "video": video_rel,
                    "qk": qk_rel,
                    "temporal": temporal_rel,
                    "compact_status": _status(
                        output_root / "state" / model / f"{seed_name}.json"
                    ),
                    "qk_status": qk_status,
                    "qk_requested": (model, seed) in pending,
                }
            records[str(seed)][model] = model_cases

    payload = {
        "cases": cases,
        "models": models,
        "model_names": MODEL_NAMES,
        "seeds": seeds,
        "records": records,
        "pending_model_seed_jobs": len(pending),
        "policy": (
            "Rows are seeds and columns are models. Q@K backfill is limited to "
            "model-seed jobs unfinished at confirmation time."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    model_headers = "".join(
        f"<th>{html.escape(MODEL_NAMES[model])}</th>" for model in models
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>多 seed Head Softmax Q@K 对比</title>
<style>
:root{{--bg:#f5f6f3;--panel:#fff;--line:#c8cbc4;--ink:#20231f;--muted:#656b63;--accent:#08665b}}
*{{box-sizing:border-box}} body{{margin:0;padding-bottom:58px;background:var(--bg);color:var(--ink);font:14px Arial,sans-serif}}
main{{max-width:1800px;margin:auto;padding:16px}} h1{{font-size:23px;margin:0 0 10px;letter-spacing:0}}
.toolbar{{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:10px;padding:10px 0;background:var(--bg);border-bottom:1px solid var(--line)}}
label{{font-weight:700}} select{{min-width:min(720px,75vw);padding:7px 9px;border:1px solid #969c93;background:#fff}}
.note{{color:var(--muted);line-height:1.55;margin:10px 0}}
.table-wrap{{overflow:auto}} table{{border-collapse:separate;border-spacing:0;width:100%;table-layout:fixed}}
th,td{{border-right:1px solid var(--line);border-bottom:1px solid var(--line);padding:8px;vertical-align:top;background:var(--panel)}}
th{{position:sticky;top:53px;z-index:4;background:#eceee9;text-align:left}}
th:first-child,td:first-child{{position:sticky;left:0;z-index:3;width:92px;background:#eceee9;font-weight:700}}
th:first-child{{z-index:6}} .cell{{min-width:350px}} video{{display:block;width:100%;aspect-ratio:16/9;background:#111}}
.qk{{display:block;width:100%;max-height:680px;object-fit:contain;background:#111;margin-top:7px}}
.asset-label{{margin-top:8px;font-size:12px;font-weight:700;color:var(--muted)}}
.meta{{display:flex;justify-content:space-between;gap:8px;margin-top:5px;color:var(--muted);font-size:12px}}
.missing{{display:grid;place-items:center;min-height:92px;margin-top:7px;border:1px dashed #a9ada6;color:var(--muted);text-align:center;padding:8px}}
.playbar{{position:fixed;left:0;right:0;bottom:0;z-index:20;display:flex;align-items:center;gap:8px;padding:9px 16px;background:#20231f;color:#fff;border-top:1px solid #000}}
.playbar button{{height:34px;padding:0 14px;border:1px solid #737a70;background:#fff;color:#20231f;font-weight:700;cursor:pointer}}
.playbar button:hover{{background:#e4e8e1}} .play-status{{margin-left:auto;color:#d7ddd4;font-variant-numeric:tabular-nums}}
a{{color:var(--accent)}} @media(max-width:900px){{.cell{{min-width:290px}} select{{min-width:0;width:100%}}}}
</style></head><body><main>
<h1>不同 seed、模型与代表 head 的 softmax Q@K 对比</h1>
<div class="toolbar"><label for="case-select">Case</label><select id="case-select"></select></div>
<p class="note">每行一个 seed，每列一个模型。每张拼图按 S/T/P/C/G 五个聚合代表 head 排列；
仅对确认时尚未完成的 model-seed 追加 512×512 softmax Q@K，较早完成项不回填。</p>
<div class="table-wrap"><table><thead><tr><th>Seed</th>{model_headers}</tr></thead>
<tbody id="body"></tbody></table></div>
</main>
<div class="playbar"><button id="play-all" type="button">Play All</button>
<button id="pause-all" type="button">Pause All</button>
<span class="play-status" id="play-status">0 / 0 playing</span></div>
<script>
const DATA={data};
const select=document.getElementById("case-select");
const body=document.getElementById("body");
const esc=value=>String(value).replace(/[&<>"']/g,ch=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[ch]));
select.innerHTML=DATA.cases.map(item=>`<option value="${{esc(item)}}">${{esc(item)}}</option>`).join("");
function missingText(item){{
  if(!item.qk_requested) return "确认时已完成，不进行 Q@K 回填";
  if(item.qk_status==="failed") return "Q@K 失败，等待自动重试";
  if(item.qk_status==="running") return "Q@K 正在计算";
  return "等待 compact 完成后计算 Q@K";
}}
function cell(item,model){{
  const video=item.video
    ? `<video controls muted playsinline preload="none" src="${{esc(item.video)}}"></video>`
    : `<div class="missing">视频尚未生成</div>`;
  const qk=item.qk
    ? `<div class="asset-label">512×512 softmax Q@K</div><a href="${{esc(item.qk)}}"><img class="qk" loading="lazy" src="${{esc(item.qk)}}"></a>`
    : `<div class="missing">${{missingText(item)}}</div>`;
  const temporal=item.temporal
    ? `<div class="asset-label">对应 head 的 13×13 temporal matrix</div><a href="${{esc(item.temporal)}}"><img class="qk" loading="lazy" src="${{esc(item.temporal)}}"></a>`
    : (item.qk_requested ? `<div class="missing">等待 13×13 temporal matrix</div>` : "");
  return `<td><div class="cell">${{video}}${{qk}}${{temporal}}<div class="meta"><span>${{esc(DATA.model_names[model])}}</span><span>compact: ${{esc(item.compact_status)}} · QK: ${{esc(item.qk_status)}}</span></div></div></td>`;
}}
const videos=()=>Array.from(body.querySelectorAll("video"));
function updatePlayStatus(){{
  const items=videos(), playing=items.filter(video=>!video.paused&&!video.ended).length;
  document.getElementById("play-status").textContent=`${{playing}} / ${{items.length}} playing`;
}}
async function playAll(){{
  const items=videos();
  items.forEach(video=>{{video.muted=true; video.currentTime=0;}});
  await Promise.allSettled(items.map(video=>video.play()));
  updatePlayStatus();
}}
function pauseAll(){{
  videos().forEach(video=>video.pause());
  updatePlayStatus();
}}
function render(){{
  pauseAll();
  const current=select.value;
  body.innerHTML=DATA.seeds.map(seed=>{{
    const cells=DATA.models.map(model=>cell(DATA.records[String(seed)][model][current],model)).join("");
    return `<tr><td>${{String(seed).padStart(6,"0")}}</td>${{cells}}</tr>`;
  }}).join("");
  videos().forEach(video=>{{
    video.addEventListener("play",updatePlayStatus);
    video.addEventListener("pause",updatePlayStatus);
    video.addEventListener("ended",updatePlayStatus);
  }});
  updatePlayStatus();
}}
document.getElementById("play-all").addEventListener("click",playAll);
document.getElementById("pause-all").addEventListener("click",pauseAll);
select.addEventListener("change",render); render();
</script></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    print(
        f"[multiseed-gallery] wrote {output_dir / 'index.html'} "
        f"cases={len(cases)} seeds={len(seeds)} models={len(models)}"
    )


if __name__ == "__main__":
    main()
