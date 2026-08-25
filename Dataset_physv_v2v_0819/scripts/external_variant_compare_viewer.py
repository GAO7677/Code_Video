#!/usr/bin/env python3
"""Local side-by-side viewer for original and external-condition demo cases."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import unquote, urlparse


DEFAULT_DERIVED_ROOT = Path(
    "/data/gaoya/agent-data/datasets/pybullet0717_prompt_physics_consistency_v1/external_variants_demo"
)


def compact_source(meta: dict, video: str) -> dict:
    motion = meta.get("motion_metrics") or {}
    return {
        "case_id": str(meta.get("case_id", meta.get("key", ""))),
        "family": str(meta.get("family_key", meta.get("blueprint", {}).get("family_key", ""))),
        "title": str(meta.get("title", "Original simulation")),
        "caption": str(meta.get("caption", "")),
        "video": video,
        "objects": [
            {
                "name": str(obj.get("name", "")),
                "shape": str(obj.get("shape", "")),
                "dynamic": bool(obj.get("dynamic", False)),
                "mass": obj.get("mass"),
                "material": obj.get("material_key", ""),
                "position": obj.get("position", []),
                "velocity": obj.get("linear_velocity", []),
            }
            for obj in meta.get("objects", [])
        ],
        "motion": {
            "speed": motion.get("motion_object_diag_pct_per_second"),
            "moving_area": motion.get("moving_area_ratio"),
            "presence": motion.get("motion_presence_ratio"),
        },
    }


def load_groups(root: Path) -> tuple[dict[str, dict], list[str]]:
    raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    groups: dict[str, dict] = {}
    for item in raw:
        source_id = str(item["source_case_id"])
        source_meta_path = Path(str(item["source_meta"]))
        source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
        group = groups.setdefault(
            source_id,
            {
                "source": compact_source(source_meta, str(item["source_video"])),
                "variants": [],
            },
        )
        variant_meta = json.loads(Path(str(item["meta"])).read_text(encoding="utf-8"))
        group["variants"].append(
            {
                "case_id": str(item["sample_key"]),
                "family": str(item.get("family_key", "")),
                "video": str(item["video"]),
                "seed": item.get("variant_seed"),
                "index": item.get("variant_index"),
                "caption": str(variant_meta.get("caption", "")),
                "perturbations": item.get("perturbations", {}).get("objects", []),
                "collision": item.get("collision_summary", {}),
                "objects": variant_meta.get("objects", []),
                "background_profile": str(
                    item.get("background_profile", variant_meta.get("background_profile", ""))
                ),
                "display_colors": (
                    variant_meta.get("render_metadata", {}).get(
                        "display_material_assignments", {}
                    )
                    or variant_meta.get("display_material_assignments", {})
                ),
            }
        )
    ordered = sorted(groups, key=lambda key: (groups[key]["source"]["family"], key))
    return groups, ordered


def html_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>External Conditions · Original / Variant Atlas</title>
  <style>
    :root { --ink:#101419; --panel:#1b232a; --paper:#e9eef0; --muted:#91a0a7; --lime:#d9ff63; --copper:#e37b5f; --line:rgba(233,238,240,.15); }
    * { box-sizing:border-box; } body { margin:0; color:var(--paper); background:var(--ink); font-family:Arial,"Noto Sans SC",sans-serif; }
    body:before { content:""; position:fixed; inset:0; pointer-events:none; opacity:.16; background-image:linear-gradient(rgba(255,255,255,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.04) 1px,transparent 1px); background-size:36px 36px; mask-image:linear-gradient(to bottom,#000,transparent 80%); }
    .shell { max-width:1540px; margin:auto; padding:30px 34px 72px; position:relative; }
    .eyebrow { color:var(--lime); font:700 11px ui-monospace,SFMono-Regular,monospace; letter-spacing:.17em; text-transform:uppercase; }
    header { display:flex; justify-content:space-between; gap:34px; align-items:end; padding:24px 0 28px; border-bottom:1px solid var(--line); }
    h1 { margin:11px 0 13px; font:700 clamp(38px,6vw,78px)/.94 Georgia,serif; letter-spacing:-.055em; } h1 em { color:var(--lime); font-style:normal; }
    .lede { max-width:720px; margin:0; color:#bdc9ce; font-size:15px; line-height:1.7; }
    .stats { min-width:390px; display:grid; grid-template-columns:repeat(3,1fr); border:1px solid var(--line); background:var(--line); }
    .stat { padding:18px; min-height:98px; background:rgba(27,35,42,.94); } .stat strong { display:block; color:var(--lime); font:700 30px ui-monospace,SFMono-Regular,monospace; } .stat span { display:block; margin-top:12px; color:var(--muted); font-size:10px; letter-spacing:.12em; text-transform:uppercase; }
    .toolbar { display:flex; justify-content:space-between; align-items:end; flex-wrap:wrap; gap:16px; padding:23px 0 18px; }
    .filters { display:flex; gap:7px; flex-wrap:wrap; margin-top:9px; } button,select { border:1px solid var(--line); border-radius:0; color:var(--paper); background:var(--panel); font:inherit; } button { padding:9px 13px; cursor:pointer; font:700 11px ui-monospace,SFMono-Regular,monospace; letter-spacing:.08em; } button:hover,button.active { color:var(--ink); border-color:var(--lime); background:var(--lime); } select { min-width:350px; padding:11px 14px; }
    button:focus-visible,select:focus { outline:2px solid #8ed1fc; outline-offset:3px; }
    .groups { display:grid; gap:25px; } .group { border:1px solid var(--line); background:rgba(27,35,42,.84); box-shadow:0 18px 50px rgba(0,0,0,.22); }
    .group-head { display:flex; align-items:baseline; justify-content:space-between; gap:16px; padding:17px 20px; border-bottom:1px solid var(--line); } .group-head h2 { margin:0; font:700 24px Georgia,serif; letter-spacing:-.03em; } .group-head small { color:var(--muted); font:11px ui-monospace,SFMono-Regular,monospace; }
    .compare { display:grid; grid-template-columns:minmax(300px,.9fr) minmax(0,1.8fr); }
    .original { padding:18px; border-right:1px solid var(--line); background:rgba(16,20,25,.5); } .original .label { color:var(--copper); }
    .label { margin-bottom:9px; font:700 11px ui-monospace,SFMono-Regular,monospace; letter-spacing:.12em; text-transform:uppercase; } .original video,.variant video { display:block; width:100%; aspect-ratio:16/9; object-fit:cover; background:#080b0e; }
    .case-name { margin:12px 0 7px; font:700 15px ui-monospace,SFMono-Regular,monospace; } .caption { margin:0; color:#c5d0d4; font-size:13px; line-height:1.55; }
    .object-line { margin-top:13px; color:var(--muted); font:10px/1.6 ui-monospace,SFMono-Regular,monospace; }
    .variants { padding:18px; } .variant-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:13px; } .variant { min-width:0; border:1px solid var(--line); background:#151b20; } .variant:hover { border-color:rgba(217,255,99,.7); } .variant-body { padding:11px 12px 13px; } .variant-top { display:flex; justify-content:space-between; gap:8px; color:var(--lime); font:700 10px ui-monospace,SFMono-Regular,monospace; } .variant-id { color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; } .delta { margin-top:9px; color:#b9c7cb; font:10px/1.55 ui-monospace,SFMono-Regular,monospace; } .contact-badge { display:inline-block; margin-top:9px; padding:4px 6px; color:var(--ink); background:var(--lime); font-weight:700; letter-spacing:.04em; }
    footer { margin-top:30px; padding-top:16px; border-top:1px solid var(--line); color:#718087; font:11px/1.6 ui-monospace,SFMono-Regular,monospace; }
    @media(max-width:900px) { .shell{padding:20px 15px 50px} header{display:block} .stats{min-width:0;margin-top:24px} .compare{grid-template-columns:1fr}.original{border-right:0;border-bottom:1px solid var(--line)} select{min-width:min(350px,100%)} }
    @media(prefers-reduced-motion:reduce) { *{scroll-behavior:auto!important} }
  </style>
</head>
<body><main class="shell">
  <header><div><div class="eyebrow">0717 / __MODE__ / DEMO</div><h1>Same physics.<br><em>New motion.</em></h1><p class="lede">Each row keeps the original material and physical parameters, then changes only the external starting conditions. Compare the source simulation on the left with its randomized variants on the right.</p></div><div class="stats"><div class="stat"><strong id="source-count">—</strong><span>source cases</span></div><div class="stat"><strong id="variant-count">—</strong><span>derived cases</span></div><div class="stat"><strong id="profile-count">—</strong><span>background profiles</span></div></div></header>
  <section class="toolbar"><div><div class="eyebrow">CASE GROUPS</div><div class="filters" id="filters"></div></div><label><span class="label">Jump to source</span><br><select id="select"></select></label></section>
  <section class="groups" id="groups"></section>
  <footer>Demo output: __ROOT__ · No VAE, prompt, or Utonia cache was generated.</footer>
</main>
<script id="payload" type="application/json">__DATA__</script>
<script>
  const DATA=JSON.parse(document.getElementById('payload').textContent), groups=DATA.groups, order=DATA.order;
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt=(v,d=3)=>v==null?'—':Number(v).toFixed(d), media=(kind,id)=>`/media/${kind}/${encodeURIComponent(id)}`;
  document.getElementById('source-count').textContent=order.length; document.getElementById('variant-count').textContent=order.reduce((n,id)=>n+groups[id].variants.length,0); document.getElementById('profile-count').textContent=new Set(order.flatMap(id=>groups[id].variants.map(v=>v.background_profile).filter(Boolean))).size||'—';
  const families=[...new Set(order.map(id=>groups[id].source.family))].sort();
  const select=document.getElementById('select'); order.forEach(id=>{const o=document.createElement('option');o.value=id;o.textContent=`${groups[id].source.family} · ${id}`;select.appendChild(o)});
  function objectText(objects){return (objects||[]).filter(o=>o.dynamic).map(o=>`${o.name}: ${o.material_key||o.material||o.shape}`).join(' · ')||'no dynamic object';}
  function variantCard(v){const first=(v.perturbations||[])[0]||{},rows=(v.perturbations||[]).map(p=>`${esc(p.object)}: Δxy ${esc((p.position_delta_m||[]).map(x=>Number(x).toFixed(3)).join(', '))} m · speed ×${fmt(p.velocity_scale)} · heading ${fmt(p.heading_delta_deg,2)}°`).join('<br>');const c=v.collision||{},pairs=Object.entries(c.pair_contact_frames||{}).map(([k,n])=>`${esc(k.replaceAll('__',' ↔ '))}: ${esc(n)} frames`).join(' · ')||'contact not recorded';const colors=Object.entries(v.display_colors||{}).map(([name,info])=>`${name}: ${info.label||'—'}${info.texture_asset_id?' · '+info.texture_asset_id:''}`).join(' · ');return `<article class="variant"><video controls playsinline preload="metadata" src="${media('variant',v.case_id)}"></video><div class="variant-body"><div class="variant-top"><span>EXT V${String(v.index).padStart(2,'0')}</span><span class="variant-id">${esc(v.case_id)}</span></div><div class="delta"><b>${esc(v.background_profile||'background')}</b><br>${esc(colors||'textured PBR objects')}<br>${rows}<br>angular scale ${fmt(first.angular_velocity_scale)} · seed ${esc(v.seed)}<br><span class="contact-badge">CONTACT · ${pairs}</span></div></div></article>`}
  function groupCard(id){const g=groups[id],s=g.source;return `<article class="group" id="group-${esc(id)}"><div class="group-head"><h2>${esc(s.title)} <span style="color:var(--copper)">/ ${esc(s.family)}</span></h2><small>${esc(id)} · ${g.variants.length} derived samples</small></div><div class="compare"><div class="original"><div class="label">__SOURCE_LABEL__</div><video controls playsinline preload="metadata" src="${media('source',id)}"></video><div class="case-name">${esc(id)}</div><p class="caption">${esc(s.caption)}</p><div class="object-line">${esc(objectText(s.objects))}</div></div><div class="variants"><div class="label">__VARIANT_LABEL__</div><div class="variant-grid">${g.variants.map(variantCard).join('')}</div></div></div></article>`}
  function render(family='ALL'){const ids=family==='ALL'?order:order.filter(id=>groups[id].source.family===family);document.getElementById('groups').innerHTML=ids.map(groupCard).join('');document.querySelectorAll('#filters button').forEach(b=>b.classList.toggle('active',b.dataset.family===family));}
  document.getElementById('filters').innerHTML=['ALL',...families].map(f=>`<button data-family="${f}">${f}</button>`).join('');document.getElementById('filters').addEventListener('click',e=>{if(e.target.matches('button'))render(e.target.dataset.family)});select.addEventListener('change',e=>document.getElementById('group-'+e.target.value)?.scrollIntoView({behavior:'smooth',block:'start'}));render();
</script></body></html>"""


class Viewer:
    def __init__(self, root: Path):
        self.root = root
        self.groups, self.order = load_groups(root)
        if "texture" in root.name.lower():
            mode = "TEXTURE REALISM"
            source_label = "Existing fast pyrender baseline"
            variant_label = "Eevee PBR · downloaded varied backgrounds + textured objects"
        else:
            mode = "COLLISION VARIANTS" if "collision" in root.name.lower() else "EXTERNAL CONDITIONS"
            source_label = "Original simulation"
            variant_label = "External-condition variants"
        self.page = (
            PAGE.replace("__DATA__", html_json({"groups": self.groups, "order": self.order}))
            .replace("__MODE__", mode)
            .replace("__ROOT__", str(root))
            .replace("__SOURCE_LABEL__", source_label)
            .replace("__VARIANT_LABEL__", variant_label)
            .encode("utf-8")
        )
        self.source_paths = {key: Path(value["source"]["video"]) for key, value in self.groups.items()}
        self.variant_paths = {v["case_id"]: Path(v["video"]) for value in self.groups.values() for v in value["variants"]}


class Handler(BaseHTTPRequestHandler):
    server_version = "ExternalVariantViewer/1.0"

    @property
    def viewer(self) -> Viewer:
        return self.server.viewer  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.address_string()}] {fmt % args}")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.send_bytes(self.viewer.page, "text/html; charset=utf-8")
            return
        match = re.fullmatch(r"/media/(source|variant)/(.+)", path)
        if match:
            kind, key = match.group(1), unquote(match.group(2))
            table = self.viewer.source_paths if kind == "source" else self.viewer.variant_paths
            target = table.get(key)
            if target is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Unknown case")
                return
            self.send_file(target)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def send_file(self, path: Path) -> None:
        if not path.is_file(): self.send_error(HTTPStatus.NOT_FOUND, "Video missing"); return
        size=path.stat().st_size; start,end=0,size-1; status=HTTPStatus.OK; header=self.headers.get('Range')
        if header:
            m=re.fullmatch(r'bytes=(\d*)-(\d*)',header.strip())
            if not m: self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE); return
            start=int(m.group(1) or 0); end=int(m.group(2) or min(size-1,start+4*1024*1024-1)); end=min(end,size-1)
            if start>=size or end<start: self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE); return
            status=HTTPStatus.PARTIAL_CONTENT
        length=end-start+1; self.send_response(status); self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'video/mp4'); self.send_header('Accept-Ranges','bytes'); self.send_header('Content-Length',str(length))
        if status==HTTPStatus.PARTIAL_CONTENT: self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        with path.open('rb') as handle:
            handle.seek(start); remaining=length
            while remaining:
                chunk=handle.read(min(1024*1024,remaining))
                if not chunk: break
                self.wfile.write(chunk); remaining-=len(chunk)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--derived-root',type=Path,default=DEFAULT_DERIVED_ROOT); parser.add_argument('--host',default='0.0.0.0'); parser.add_argument('--port',type=int,default=8782); args=parser.parse_args()
    viewer=Viewer(args.derived_root.resolve()); server=ThreadingHTTPServer((args.host,args.port),Handler); server.viewer=viewer  # type: ignore[attr-defined]
    print(f'External-variant viewer: http://127.0.0.1:{args.port}/'); print(f'Loaded {len(viewer.order)} source groups / {len(viewer.variant_paths)} variants')
    try: server.serve_forever()
    except KeyboardInterrupt: print('\nViewer stopped')
    finally: server.server_close()


if __name__=='__main__': main()
