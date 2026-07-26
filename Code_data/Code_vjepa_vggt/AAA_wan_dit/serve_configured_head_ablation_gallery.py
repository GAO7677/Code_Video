#!/usr/bin/env python3
"""Serve completed per-head ablations as one live page per input case."""

from __future__ import annotations

import argparse
import html
import json
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse


DATA_ROOT = Path("/data/gaoya")
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
MODEL_ORDER = {name: index for index, name in enumerate(MODEL_LABELS)}
BASELINE_ROOTS = {
    "wan_lora": (
        Path("/data/gaoya/AAA_test_video/0623/test/v2v_wan_test5/wan_lora/baseline"),
        Path("/data/gaoya/agent-data/outputs/wan_dit_block17_self_attention/test5_first5/generated/wan_lora"),
        Path("/data/gaoya/AAA_test_video/0623/test/v2v_wan/wan_lora/baseline"),
    ),
    "xssc": (
        Path("/data/gaoya/AAA_test_video/0623/test/v2v_wan_test5/xssc/baseline/results"),
        Path("/data/gaoya/agent-data/outputs/wan_dit_block17_self_attention/test5_first5/generated/xssc/results"),
        Path("/data/gaoya/AAA_test_video/0623/test/v2v_wan/xssc/baseline/results"),
    ),
    "physrvg": (
        Path("/data/gaoya/AAA_test_video/0623/test/v2v_wan_test5/PhyRVG/baseline"),
        Path("/data/gaoya/agent-data/outputs/wan_dit_block17_self_attention/test5_first5/generated/physrvg/input_first5_unique/physRVG_steps40_512x896_08_49f"),
        Path("/data/gaoya/AAA_test_video/0623/test/v2v_wan/PhyRVG/baseline/physicIQ/physRVG_steps40_512x896_08_49f"),
    ),
}
METRICS = (
    ("physics_iq_with_context", "Physics-IQ · ctx", "higher", ("physics_iq_with_context", "score"), 2),
    ("physics_iq_without_context", "Physics-IQ · no ctx", "higher", ("physics_iq_without_context", "score"), 2),
    ("pmf_with_context", "PMF · ctx", "higher", ("pmf_with_context", "score"), 4),
    ("pmf_without_context", "PMF · no ctx", "higher", ("pmf_without_context", "score"), 4),
    ("wmreward", "WMReward surprise", "lower", ("wmreward", "surprise"), 4),
    ("vbench_subject_consistency", "Subject consistency", "higher", ("vbench_subject_consistency", "score"), 4),
    ("vbench_background_consistency", "Background consistency", "higher", ("vbench_background_consistency", "score"), 4),
    ("vbench_temporal_flickering", "Temporal flickering", "higher", ("vbench_temporal_flickering", "score"), 4),
    ("vbench_motion_smoothness", "Motion smoothness", "higher", ("vbench_motion_smoothness", "score"), 4),
    ("vbench_dynamic_degree", "Dynamic degree", "higher", ("vbench_dynamic_degree", "score"), 4),
    ("vbench_aesthetic_quality", "Aesthetic quality", "higher", ("vbench_aesthetic_quality", "score"), 4),
    ("vbench_imaging_quality", "Imaging quality", "higher", ("vbench_imaging_quality", "score"), 4),
    ("videophy2_sa", "VideoPhy2 SA", "higher", ("videophy2", "sa_score"), 2),
    ("videophy2_pc", "VideoPhy2 PC", "higher", ("videophy2", "pc_score"), 2),
    ("videophy2_joint", "VideoPhy2 joint", "higher", ("videophy2", "joint_pass"), 0),
    ("videophy2_pc_raw", "VideoPhy2 PC raw", "higher", ("videophy2", "pc_raw_score"), 2),
    ("cosmos_reason1", "Cosmos-Reason1", "higher", ("cosmos_reason1", "score"), 4),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/data/gaoya/AAA_test_video/0623/test/v2v_wan_test5"
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8916)
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def data_url(path: Path) -> str:
    relative = path.resolve().relative_to(DATA_ROOT.resolve()).as_posix()
    return "/_data/" + quote(relative, safe="/")


def slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not result:
        raise ValueError(f"invalid case name: {value!r}")
    return result


def load_metrics(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    result: dict[str, float] = {}
    for key, _label, _direction, value_path, _decimals in METRICS:
        value: object = payload
        for part in value_path:
            value = value.get(part) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = float(value)
    return result


def baseline_path(model: str, case_name: str) -> Path | None:
    for root in BASELINE_ROOTS[model]:
        candidate = root / f"{case_name}.mp4"
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
        matches = list(root.glob(f"*/{case_name}.mp4"))
        if len(matches) == 1 and matches[0].stat().st_size > 0:
            return matches[0]
        matches = list(root.glob(f"*/*/{case_name}.mp4"))
        if len(matches) == 1 and matches[0].stat().st_size > 0:
            return matches[0]
    return None


def build_manifest(root: Path, *, static: bool = False) -> dict[str, object]:
    run_root = root / "_pipeline"
    input_list = run_root / "input_unique.txt"
    completed = run_root / "generation" / "completed.tsv"
    validations = run_root / "generation" / "validations"

    case_sources: list[Path] = []
    seen: set[Path] = set()
    for line in input_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source = Path(line.strip()).expanduser().resolve()
        if source not in seen:
            seen.add(source)
            case_sources.append(source)

    cases: list[dict[str, object]] = []
    case_by_input: dict[str, dict[str, object]] = {}
    for source in case_sources:
        payload = load_json(source)
        source_video = payload.get("source_video")
        item: dict[str, object] = {
            "name": source.stem,
            "slug": slug(source.stem),
            "prompt": payload.get("input_caption", ""),
            "source_video": (
                (
                    f"/_gallery/sources/{quote(slug(source.stem))}.mp4"
                    if static
                    else data_url(Path(source_video))
                )
                if isinstance(source_video, str)
                and Path(source_video).is_file()
                else None
            ),
            "outputs": [],
        }
        cases.append(item)
        case_by_input[str(source)] = item

    completed_rows = []
    if completed.is_file():
        for line in completed.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) >= 4:
                completed_rows.append(fields)

    configurations: list[dict[str, object]] = []
    for model in MODEL_LABELS:
        if not any(
            baseline_path(model, str(case["name"])) is not None
            for case in cases
        ):
            continue
        config = {
            "task_id": f"baseline-{model}",
            "model": model,
            "model_label": MODEL_LABELS[model],
            "block": None,
            "head": None,
            "baseline": True,
        }
        configurations.append(config)
        for case in cases:
            video = baseline_path(model, str(case["name"]))
            if video is None:
                continue
            case["outputs"].append(
                {
                    **config,
                    "video": (
                        f"/_gallery/baselines/{model}/"
                        f"{quote(str(case['slug']))}.mp4"
                        if static
                        else data_url(video)
                    ),
                    "metrics": load_metrics(video.with_suffix(".json")),
                    "best_metrics": [],
                }
            )

    for task_id, model, block, head, *_ in completed_rows:
        validation_path = validations / f"{task_id}.json"
        if not validation_path.is_file():
            continue
        validation = load_json(validation_path)
        config = {
            "task_id": task_id,
            "model": model,
            "model_label": MODEL_LABELS.get(model, model),
            "block": int(block),
            "head": int(head),
        }
        configurations.append(config)
        records = validation.get("records", [])
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            input_json = record.get("input_json")
            output_video = record.get("output_video")
            output_json = record.get("output_json")
            case = case_by_input.get(str(Path(str(input_json)).resolve()))
            video = Path(str(output_video))
            if case is None or not video.is_file() or video.stat().st_size == 0:
                continue
            case["outputs"].append(
                {
                    **config,
                    "video": (
                        "/" + quote(
                            video.resolve().relative_to(root).as_posix(),
                            safe="/",
                        )
                        if static
                        else data_url(video)
                    ),
                    "metrics": load_metrics(Path(str(output_json))),
                    "best_metrics": [],
                }
            )

    priority_root = root / "_priority_case"
    priority_completed = priority_root / "completed.tsv"
    existing_keys = {
        (item["model"], item["block"], item["head"])
        for item in configurations
        if not item.get("baseline")
    }
    if priority_completed.is_file():
        for line in priority_completed.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) < 4:
                continue
            task_id, model, block, head = fields[:4]
            key = (model, int(block), int(head))
            if key in existing_keys:
                continue
            validation_path = priority_root / "validations" / f"{task_id}.json"
            if not validation_path.is_file():
                continue
            validation = load_json(validation_path)
            config = {
                "task_id": task_id,
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "block": int(block),
                "head": int(head),
                "priority_case_only": True,
            }
            configurations.append(config)
            existing_keys.add(key)
            records = validation.get("records", [])
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                input_json = record.get("input_json")
                output_video = Path(str(record.get("output_video")))
                output_json = Path(str(record.get("output_json")))
                case = case_by_input.get(str(Path(str(input_json)).resolve()))
                if (
                    case is None
                    or not output_video.is_file()
                    or output_video.stat().st_size == 0
                ):
                    continue
                case["outputs"].append(
                    {
                        **config,
                        "video": (
                            "/" + quote(
                                output_video.resolve()
                                .relative_to(root)
                                .as_posix(),
                                safe="/",
                            )
                            if static
                            else data_url(output_video)
                        ),
                        "metrics": load_metrics(output_json),
                        "best_metrics": [],
                    }
                )

    configurations.sort(
        key=lambda item: (
            MODEL_ORDER.get(str(item["model"]), 99),
            -1 if item["block"] is None else int(item["block"]),
            -1 if item["head"] is None else int(item["head"]),
        )
    )
    for case in cases:
        case["outputs"].sort(
            key=lambda item: (
                MODEL_ORDER.get(str(item["model"]), 99),
                -1 if item["block"] is None else int(item["block"]),
                -1 if item["head"] is None else int(item["head"]),
            )
        )
        outputs = case["outputs"]
        for key, _label, direction, _value_path, _decimals in METRICS:
            available = [
                (index, output["metrics"].get(key))
                for index, output in enumerate(outputs)
                if isinstance(output.get("metrics"), dict)
                and isinstance(output["metrics"].get(key), (int, float))
            ]
            if not available:
                continue
            best = (
                min(value for _index, value in available)
                if direction == "lower"
                else max(value for _index, value in available)
            )
            for index, value in available:
                if value == best:
                    outputs[index]["best_metrics"].append(key)
    return {
        "cases": cases,
        "configurations": configurations,
        "metrics": [
            {
                "key": key,
                "label": label,
                "direction": direction,
                "decimals": decimals,
            }
            for key, label, direction, _value_path, decimals in METRICS
        ],
        "num_cases": len(cases),
        "num_completed_configurations": len(configurations),
        "num_available_videos": sum(len(case["outputs"]) for case in cases),
    }


def materialize_static_gallery(root: Path) -> dict[str, object]:
    """Write a pyport-compatible snapshot below ROOT/_gallery."""
    manifest = build_manifest(root, static=True)
    gallery_dir = root / "_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    (gallery_dir / "index.html").write_text(
        case_html(""), encoding="utf-8"
    )
    (gallery_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for case in manifest["cases"]:
        case_dir = gallery_dir / "case" / str(case["slug"])
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "index.html").write_text(
            case_html(str(case["slug"])), encoding="utf-8"
        )

    sources_dir = gallery_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    input_list = root / "_pipeline" / "input_unique.txt"
    for line in input_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source_json = Path(line.strip()).expanduser().resolve()
        payload = load_json(source_json)
        source_video = payload.get("source_video")
        if not isinstance(source_video, str):
            continue
        target = Path(source_video).expanduser().resolve()
        if not target.is_file():
            continue
        link = sources_dir / f"{slug(source_json.stem)}.mp4"
        if link.is_symlink() and link.resolve() == target:
            continue
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)

    for model in MODEL_LABELS:
        baseline_dir = gallery_dir / "baselines" / model
        baseline_dir.mkdir(parents=True, exist_ok=True)
        for case in manifest["cases"]:
            target = baseline_path(model, str(case["name"]))
            if target is None:
                continue
            link = baseline_dir / f"{case['slug']}.mp4"
            if link.is_symlink() and link.resolve() == target:
                continue
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(target)
    return manifest


def index_html(manifest: dict[str, object]) -> str:
    links = "\n".join(
        f'<a href="./case/{quote(str(case["slug"]))}/">'
        f'{index + 1:02d}. {html.escape(str(case["name"]))}</a>'
        for index, case in enumerate(manifest["cases"])
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Head Ablation Cases</title>
<style>
body{{margin:0;background:#f1f3f2;color:#18211c;font:14px Arial,sans-serif;letter-spacing:0}}
header{{padding:18px 24px;background:#18382a;color:white}} h1{{margin:0;font-size:21px}}
p{{margin:6px 0 0;color:#c2d5ca}} main{{max-width:1000px;margin:auto;padding:20px}}
.cases{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}}
a{{padding:12px;background:white;border:1px solid #ccd4cf;border-radius:4px;color:#204c36;text-decoration:none;overflow-wrap:anywhere}}
a:hover{{border-color:#347052}} @media(max-width:700px){{.cases{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Per-head Ablation Cases</h1>
<p>{manifest["num_cases"]} cases · {manifest["num_completed_configurations"]} completed configurations</p>
</header><main><div class="cases">{links}</div></main></body></html>"""


def case_html(case_slug: str) -> str:
    safe_slug = html.escape(case_slug, quote=True)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Head Ablation · {safe_slug}</title>
<style>
:root{{--bg:#eef1ef;--surface:#fff;--text:#17201b;--muted:#65716b;--line:#cbd4cf;--nav:#18382a}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px Arial,sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:5;display:grid;grid-template-columns:minmax(260px,1fr) minmax(360px,720px) auto;gap:12px;align-items:center;padding:11px 18px;background:var(--nav);color:#fff}}
h1{{margin:0;font-size:18px;overflow-wrap:anywhere}} #status{{margin-top:3px;color:#bdd1c5;font-size:12px}}
.case-nav{{display:grid;grid-template-columns:40px minmax(0,1fr) 40px;gap:7px}} .nav{{display:flex;gap:7px}} a,button,select{{min-height:36px;padding:8px 11px;border:1px solid #557765;border-radius:4px}} a,button{{color:#fff;background:#24503a;text-decoration:none;cursor:pointer}} select{{min-width:0;background:#fff;color:var(--text)}}
main{{max-width:2200px;margin:auto;padding:18px 18px 42px}} .case-meta{{display:grid;grid-template-columns:minmax(0,1fr) 420px;gap:16px;align-items:start}}
h2{{margin:0;font-size:20px;overflow-wrap:anywhere}} .prompt{{color:var(--muted);line-height:1.5}} video{{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#080b09}}
.model{{margin-top:26px;border-top:3px solid #285f99;padding-top:12px}} .model[data-model=xssc]{{border-color:#24734f}} .model[data-model=physrvg]{{border-color:#a65325}}
.model h3{{margin:0 0 12px;font-size:19px}} .block{{margin:0 0 22px}} .block h4{{margin:0;padding:8px 10px;background:#dfe7e2;border:1px solid var(--line);font-size:15px}}
.heads{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-top:0}}
.cell{{min-width:0;padding:6px;background:var(--surface)}} .label{{margin-bottom:5px;color:#405149;font-size:11px;font-weight:700}}
.empty{{padding:30px;background:white;border:1px solid var(--line);color:var(--muted)}}
.metrics{{margin-top:28px;background:#fff;border:1px solid var(--line)}} .metrics-head{{display:flex;justify-content:space-between;gap:12px;padding:10px;border-bottom:1px solid var(--line)}} .metrics-head h3{{margin:0;font-size:16px}} .metrics-head span{{color:var(--muted);font-size:11px}}
.table-wrap{{overflow:auto;max-height:720px}} table{{width:max-content;min-width:100%;border-collapse:separate;border-spacing:0;font-size:10px}} th,td{{min-width:86px;padding:5px;border-right:1px solid #dce4df;border-bottom:1px solid #dce4df;text-align:center;white-space:nowrap}} thead th{{position:sticky;top:0;z-index:2;background:#e5ece8}} th.method{{position:sticky;left:0;z-index:1;min-width:190px;text-align:left;background:#f4f7f5}} thead th.method{{z-index:3;background:#dce7e1}} td.best{{color:#075d37;background:#dff3e7;font-weight:700}} td.missing{{color:#a1aaa5}}
@media(max-width:1100px){{header{{grid-template-columns:1fr}}.heads{{grid-template-columns:repeat(3,minmax(0,1fr))}}.case-meta{{grid-template-columns:1fr}}}}
@media(max-width:650px){{header{{grid-template-columns:1fr}}.heads{{grid-template-columns:1fr}}}}
</style></head><body>
<header><div><h1 id="title">Loading...</h1><div id="status"></div></div>
<div class="case-nav"><button id="prev" title="Previous case">←</button><select id="caseSelect"></select><button id="next" title="Next case">→</button></div>
<nav class="nav"><button id="playAll">Play all</button><button id="restartAll">Restart all</button><button id="refresh">Refresh</button></nav></header>
<main><section class="case-meta"><div><h2 id="name"></h2><p class="prompt" id="prompt"></p></div><div id="source"></div></section><div id="models"></div><section class="metrics" id="metrics"></section></main>
<script>
const INITIAL_CASE_SLUG={json.dumps(case_slug)};
const MANIFEST_URL=INITIAL_CASE_SLUG?"../../manifest.json":"./manifest.json";
const MODEL_ORDER=["wan_lora","xssc","physrvg"];
const MODEL_LABELS={{wan_lora:"Wan+LoRA",xssc:"Wan+xSSC",physrvg:"PhysRVG"}};
let activeSlug=INITIAL_CASE_SLUG;
function video(url){{const v=document.createElement("video");v.controls=true;v.preload="none";v.playsInline=true;v.src=url;return v}}
function add(parent,tag,className,text){{const node=document.createElement(tag);node.className=className;node.textContent=text;parent.appendChild(node);return node}}
async function render(){{
 const data=await fetch(MANIFEST_URL+"?t="+Date.now(),{{cache:"no-store"}}).then(r=>r.json());
 if(!activeSlug){{const match=location.hash.match(/case=([^&]+)/);activeSlug=match?decodeURIComponent(match[1]):data.cases[0].slug}}
 let index=data.cases.findIndex(c=>c.slug===activeSlug);if(index<0){{index=0;activeSlug=data.cases[0].slug}}
 const item=data.cases[index]; document.title="Head Ablation · "+item.name; document.getElementById("title").textContent=item.name;
 document.getElementById("name").textContent=item.name; document.getElementById("prompt").textContent=item.prompt||"";
 document.getElementById("status").textContent=data.num_completed_configurations+" completed configurations · "+item.outputs.length+" videos";
 const select=document.getElementById("caseSelect");select.replaceChildren();data.cases.forEach((c,i)=>{{const option=document.createElement("option");option.value=c.slug;option.textContent=(i+1)+". "+c.name;select.appendChild(option)}});select.value=item.slug;
 const source=document.getElementById("source");source.replaceChildren();if(item.source_video)source.appendChild(video(item.source_video));
 const models=document.getElementById("models");models.replaceChildren();
 MODEL_ORDER.forEach(model=>{{const outputs=item.outputs.filter(o=>o.model===model);const baseline=outputs.find(o=>o.baseline);const ablations=outputs.filter(o=>!o.baseline);
  const section=document.createElement("section");section.className="model";section.dataset.model=model;
  const h3=document.createElement("h3");h3.textContent=MODEL_LABELS[model];section.appendChild(h3);
  const baselineGroup=document.createElement("section");baselineGroup.className="block";const baselineTitle=document.createElement("h4");baselineTitle.textContent="Baseline · no ablation";baselineGroup.appendChild(baselineTitle);
  const baselineGrid=document.createElement("div");baselineGrid.className="heads";if(baseline){{const cell=document.createElement("div");cell.className="cell";const label=document.createElement("div");label.className="label";label.textContent="Baseline";cell.append(label,video(baseline.video));baselineGrid.appendChild(cell)}}else{{const pending=document.createElement("div");pending.className="empty";pending.textContent="Pending baseline";baselineGrid.appendChild(pending)}}baselineGroup.appendChild(baselineGrid);section.appendChild(baselineGroup);
  [...new Set(ablations.map(o=>o.block))].sort((a,b)=>a-b).forEach(block=>{{
   const group=document.createElement("section");group.className="block";const h4=document.createElement("h4");h4.textContent="Block "+String(block).padStart(2,"0");group.appendChild(h4);
   const grid=document.createElement("div");grid.className="heads";
   ablations.filter(o=>o.block===block).sort((a,b)=>a.head-b.head).forEach(out=>{{const cell=document.createElement("div");cell.className="cell";
    const label=document.createElement("div");label.className="label";label.textContent="Head "+String(out.head).padStart(2,"0")+" = 0";cell.append(label,video(out.video));grid.appendChild(cell)}});
   group.appendChild(grid);section.appendChild(group)}});
  models.appendChild(section)}});
 if(!models.children.length){{const empty=document.createElement("div");empty.className="empty";empty.textContent="No completed configurations yet";models.appendChild(empty)}}
 const metrics=document.getElementById("metrics");metrics.replaceChildren();const mh=document.createElement("div");mh.className="metrics-head";add(mh,"h3","","Case metric comparison");add(mh,"span","","↑ higher is better · ↓ lower is better · best value highlighted");metrics.appendChild(mh);
 const wrap=document.createElement("div");wrap.className="table-wrap";const table=document.createElement("table");const thead=document.createElement("thead");const hr=document.createElement("tr");add(hr,"th","method","Configuration");data.metrics.forEach(metric=>add(hr,"th","",metric.label+" "+(metric.direction==="lower"?"↓":"↑")));thead.appendChild(hr);table.appendChild(thead);
 const tbody=document.createElement("tbody");item.outputs.forEach(out=>{{const row=document.createElement("tr");add(row,"th","method",out.baseline?MODEL_LABELS[out.model]+" · Baseline":MODEL_LABELS[out.model]+" · B"+String(out.block).padStart(2,"0")+" · H"+String(out.head).padStart(2,"0"));const best=new Set(out.best_metrics||[]);
  data.metrics.forEach(metric=>{{const value=out.metrics?out.metrics[metric.key]:null;const valid=Number.isFinite(value);add(row,"td",(valid&&best.has(metric.key)?"best ":"")+(valid?"":"missing"),valid?Number(value).toFixed(metric.decimals):"—")}});tbody.appendChild(row)}});
 table.appendChild(tbody);wrap.appendChild(table);metrics.appendChild(wrap);
 history.replaceState(null,"","#case="+encodeURIComponent(item.slug));
}}
function move(delta){{fetch(MANIFEST_URL+"?t="+Date.now(),{{cache:"no-store"}}).then(r=>r.json()).then(data=>{{const i=data.cases.findIndex(c=>c.slug===activeSlug);activeSlug=data.cases[(i+delta+data.cases.length)%data.cases.length].slug;render()}})}}
function allVideos(){{return Array.from(document.querySelectorAll("video"))}}
document.getElementById("playAll").onclick=()=>allVideos().forEach(item=>item.play().catch(()=>{{}}));
document.getElementById("restartAll").onclick=()=>allVideos().forEach(item=>{{item.currentTime=0;item.play().catch(()=>{{}})}});
document.getElementById("prev").onclick=()=>move(-1);document.getElementById("next").onclick=()=>move(1);document.getElementById("caseSelect").onchange=e=>{{activeSlug=e.target.value;render()}};document.getElementById("refresh").onclick=()=>render();setInterval(()=>render().catch(()=>{{}}),30000);render();
</script></body></html>"""


class Handler(SimpleHTTPRequestHandler):
    server_version = "HeadAblationGallery/1.0"

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/_gallery/manifest.json":
            body = json.dumps(
                self.server.build_manifest(), ensure_ascii=False  # type: ignore[attr-defined]
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path in {"/_gallery", "/_gallery/"}:
            self.send_html(case_html(""))
            return
        match = re.fullmatch(r"/_gallery/case/([^/]+)/?", path)
        if match:
            self.send_html(case_html(unquote(match.group(1))))
            return
        if path.startswith("/_data/"):
            self.path = "/" + path.removeprefix("/_data/")
        super().do_GET()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "_pipeline").is_dir():
        raise FileNotFoundError(root / "_pipeline")

    def live_manifest() -> dict[str, object]:
        return build_manifest(root)

    first = live_manifest()
    materialize_static_gallery(root)
    if args.build_only:
        print(
            json.dumps(
                {
                    "gallery": str(root / "_gallery" / "index.html"),
                    "cases": first["num_cases"],
                    "completed_configurations": first[
                        "num_completed_configurations"
                    ],
                    "videos": first["num_available_videos"],
                },
                ensure_ascii=False,
            )
        )
        return
    print(
        json.dumps(
            {
                "url": f"http://localhost:{args.port}/_gallery/",
                "cases": first["num_cases"],
                "completed_configurations": first[
                    "num_completed_configurations"
                ],
                "videos": first["num_available_videos"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    handler = partial(Handler, directory=str(DATA_ROOT))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.build_manifest = live_manifest  # type: ignore[attr-defined]
    server.serve_forever()


if __name__ == "__main__":
    main()
