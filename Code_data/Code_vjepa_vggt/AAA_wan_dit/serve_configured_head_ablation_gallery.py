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


def build_manifest(root: Path) -> dict[str, object]:
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
                data_url(Path(source_video))
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
            case = case_by_input.get(str(Path(str(input_json)).resolve()))
            video = Path(str(output_video))
            if case is None or not video.is_file() or video.stat().st_size == 0:
                continue
            case["outputs"].append(
                {
                    **config,
                    "video": data_url(video),
                }
            )

    configurations.sort(
        key=lambda item: (
            MODEL_ORDER.get(str(item["model"]), 99),
            int(item["block"]),
            int(item["head"]),
        )
    )
    for case in cases:
        case["outputs"].sort(
            key=lambda item: (
                MODEL_ORDER.get(str(item["model"]), 99),
                int(item["block"]),
                int(item["head"]),
            )
        )
    return {
        "cases": cases,
        "configurations": configurations,
        "num_cases": len(cases),
        "num_completed_configurations": len(configurations),
        "num_available_videos": sum(len(case["outputs"]) for case in cases),
    }


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
header{{position:sticky;top:0;z-index:5;display:grid;grid-template-columns:minmax(260px,1fr) auto;gap:12px;align-items:center;padding:11px 18px;background:var(--nav);color:#fff}}
h1{{margin:0;font-size:18px;overflow-wrap:anywhere}} #status{{margin-top:3px;color:#bdd1c5;font-size:12px}}
.nav{{display:flex;gap:7px}} a,button{{min-height:36px;padding:8px 11px;border:1px solid #557765;border-radius:4px;color:#fff;background:#24503a;text-decoration:none;cursor:pointer}}
main{{max-width:2200px;margin:auto;padding:18px 18px 42px}} .case-meta{{display:grid;grid-template-columns:minmax(0,1fr) 420px;gap:16px;align-items:start}}
h2{{margin:0;font-size:20px;overflow-wrap:anywhere}} .prompt{{color:var(--muted);line-height:1.5}} video{{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#080b09}}
.model{{margin-top:26px;border-top:3px solid #285f99;padding-top:12px}} .model[data-model=xssc]{{border-color:#24734f}} .model[data-model=physrvg]{{border-color:#a65325}}
.model h3{{margin:0 0 12px;font-size:19px}} .block{{margin:0 0 22px}} .block h4{{margin:0;padding:8px 10px;background:#dfe7e2;border:1px solid var(--line);font-size:15px}}
.heads{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-top:0}}
.cell{{min-width:0;padding:6px;background:var(--surface)}} .label{{margin-bottom:5px;color:#405149;font-size:11px;font-weight:700}}
.empty{{padding:30px;background:white;border:1px solid var(--line);color:var(--muted)}} @media(max-width:1100px){{.heads{{grid-template-columns:repeat(3,minmax(0,1fr))}}.case-meta{{grid-template-columns:1fr}}}}
@media(max-width:650px){{header{{grid-template-columns:1fr}}.heads{{grid-template-columns:1fr}}}}
</style></head><body>
<header><div><h1 id="title">Loading...</h1><div id="status"></div></div>
<nav class="nav"><a id="prev" title="Previous case">←</a><a href="../../">Cases</a><a id="next" title="Next case">→</a><button id="refresh">Refresh</button></nav></header>
<main><section class="case-meta"><div><h2 id="name"></h2><p class="prompt" id="prompt"></p></div><div id="source"></div></section><div id="models"></div></main>
<script>
const CASE_SLUG={json.dumps(case_slug)};
const MODEL_ORDER=["wan_lora","xssc","physrvg"];
const MODEL_LABELS={{wan_lora:"Wan+LoRA",xssc:"Wan+xSSC",physrvg:"PhysRVG"}};
function video(url){{const v=document.createElement("video");v.controls=true;v.preload="none";v.playsInline=true;v.src=url;return v}}
async function render(){{
 const data=await fetch("../../manifest.json?t="+Date.now(),{{cache:"no-store"}}).then(r=>r.json());
 const index=data.cases.findIndex(c=>c.slug===CASE_SLUG); if(index<0)throw new Error("Case not found");
 const item=data.cases[index]; document.title="Head Ablation · "+item.name; document.getElementById("title").textContent=item.name;
 document.getElementById("name").textContent=item.name; document.getElementById("prompt").textContent=item.prompt||"";
 document.getElementById("status").textContent=data.num_completed_configurations+" completed configurations · "+item.outputs.length+" videos";
 document.getElementById("prev").href="../"+data.cases[(index-1+data.cases.length)%data.cases.length].slug+"/";
 document.getElementById("next").href="../"+data.cases[(index+1)%data.cases.length].slug+"/";
 const source=document.getElementById("source");source.replaceChildren();if(item.source_video)source.appendChild(video(item.source_video));
 const models=document.getElementById("models");models.replaceChildren();
 MODEL_ORDER.forEach(model=>{{const outputs=item.outputs.filter(o=>o.model===model);if(!outputs.length)return;
  const section=document.createElement("section");section.className="model";section.dataset.model=model;
  const h3=document.createElement("h3");h3.textContent=MODEL_LABELS[model];section.appendChild(h3);
  [...new Set(outputs.map(o=>o.block))].sort((a,b)=>a-b).forEach(block=>{{
   const group=document.createElement("section");group.className="block";const h4=document.createElement("h4");h4.textContent="Block "+String(block).padStart(2,"0");group.appendChild(h4);
   const grid=document.createElement("div");grid.className="heads";
   outputs.filter(o=>o.block===block).sort((a,b)=>a.head-b.head).forEach(out=>{{const cell=document.createElement("div");cell.className="cell";
    const label=document.createElement("div");label.className="label";label.textContent="Head "+String(out.head).padStart(2,"0")+" = 0";cell.append(label,video(out.video));grid.appendChild(cell)}});
   group.appendChild(grid);section.appendChild(group)}});
  models.appendChild(section)}});
 if(!models.children.length){{const empty=document.createElement("div");empty.className="empty";empty.textContent="No completed configurations yet";models.appendChild(empty)}}
}}
document.getElementById("refresh").onclick=()=>render();setInterval(()=>render().catch(()=>{{}}),30000);render();
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
            self.send_html(index_html(self.server.build_manifest()))  # type: ignore[attr-defined]
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
