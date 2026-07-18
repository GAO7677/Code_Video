#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0718ToyDataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the static viewer for the 0718 controlled toy dataset.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--page-title", default="0718 Controlled Toy Physics Dataset")
    return parser.parse_args()


def _safe(value: object) -> str:
    return html.escape(str(value), quote=True)


def _relpath(path: str | Path, output_root: Path) -> str:
    path = Path(path).resolve()
    dataset_root = output_root.parent.resolve()
    try:
        return path.relative_to(dataset_root).as_posix()
    except ValueError:
        return os.path.relpath(str(path), str(output_root)).replace(os.sep, "/")


def _ensure_viewer_links(dataset_root: Path, output_root: Path) -> None:
    for name in ("cases", "config", "logs", "dataset_manifest.json"):
        target = dataset_root / name
        link = output_root / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"viewer link points to the wrong target: {link}")
            continue
        if link.exists():
            raise RuntimeError(f"cannot create viewer link because this path already exists: {link}")
        link.symlink_to(Path(os.path.relpath(target, output_root)), target_is_directory=target.is_dir())


def _media_card(label: str, manifest: dict, output_root: Path) -> str:
    rgb = _relpath(manifest["video"], output_root)
    mask = _relpath(manifest["mask_video"], output_root)
    mask_ids = _relpath(manifest["mask_ids"], output_root)
    meta = _relpath(manifest["meta"], output_root)
    id_map = ", ".join(f"{name}={instance_id}" for name, instance_id in manifest.get("instance_id_map", {}).items())
    return f"""
      <article class="media-card">
        <div class="media-head">
          <div><span class="media-label">{_safe(label)}</span><h3>{_safe(manifest['sample_key'])}</h3></div>
          <span class="id-map">{_safe(id_map)}</span>
        </div>
        <div class="video-pair">
          <figure><video controls muted loop preload="metadata" src="{_safe(rgb)}"></video><figcaption>RGB</figcaption></figure>
          <figure><video controls muted loop preload="metadata" src="{_safe(mask)}"></video><figcaption>Instance mask preview</figcaption></figure>
        </div>
        <p>{_safe(manifest.get('short_caption', ''))}</p>
        <div class="links"><a href="{_safe(meta)}">metadata</a><a href="{_safe(mask_ids)}">lossless mask IDs (.npz)</a></div>
      </article>
    """


def _case_section(case: dict, output_root: Path) -> str:
    base = _media_card("Shared anchor / base", case["base"], output_root)
    variants = "".join(
        _media_card(f"Variant: {pair['attribute'].replace('_', ' ')}", pair["variant"], output_root)
        for pair in case.get("pairs", [])
    )
    search_text = f"{case['case_id']} {case['case_key']} {case.get('title', '')}".lower()
    return f"""
    <section class="case" data-count="{int(case['object_count'])}" data-search="{_safe(search_text)}">
      <header class="case-head">
        <div><span class="case-index">{_safe(case['case_id'])}</span><h2>{_safe(case.get('title', case['case_key']))}</h2></div>
        <div class="case-meta"><span>{int(case['object_count'])} object{'s' if int(case['object_count']) != 1 else ''}</span><span>seed {int(case['seed'])}</span></div>
      </header>
      <p class="pair-note">The base is the common anchor for all three controlled pairs. Each variant changes only the named attribute.</p>
      <div class="media-grid">{base}{variants}</div>
    </section>
    """


def build_html(manifest: dict, output_root: Path, page_title: str) -> str:
    cases = manifest.get("cases", [])
    sections = "".join(_case_section(case, output_root) for case in cases)
    dataset_manifest = _relpath(Path(manifest["dataset_root"]) / "dataset_manifest.json", output_root)
    case_catalog = _relpath(Path(manifest["dataset_root"]) / "config" / "case_catalog.json", output_root)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{_safe(page_title)}</title>
  <style>
    :root{{--paper:#f1eadc;--card:#fffaf1;--ink:#20231f;--muted:#6f6b61;--line:#d8cdb9;--red:#bd4b2f;--blue:#246b78;--shadow:0 18px 55px rgba(66,48,28,.12)}}
    *{{box-sizing:border-box}} body{{margin:0;color:var(--ink);font-family:"Avenir Next","Gill Sans",sans-serif;background:radial-gradient(circle at 14% 0,#fff9ed 0,transparent 32%),repeating-linear-gradient(90deg,transparent 0 39px,rgba(77,63,42,.025) 40px),var(--paper)}}
    main{{width:min(1500px,calc(100vw - 32px));margin:auto;padding:28px 0 64px}} a{{color:var(--blue);text-decoration:none}} a:hover{{text-decoration:underline}}
    .hero{{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:30px;padding:32px;background:linear-gradient(135deg,#fffdf8,#eadcc4);box-shadow:var(--shadow)}}
    .hero:after{{content:"";position:absolute;width:260px;height:260px;right:-80px;top:-110px;border:34px solid rgba(189,75,47,.12);border-radius:50%}}
    .eyebrow,.case-index{{font:700 11px/1.2 "Courier New",monospace;letter-spacing:.15em;text-transform:uppercase;color:var(--red)}} h1{{font-family:"Baskerville","Iowan Old Style",serif;font-size:clamp(36px,6vw,72px);line-height:.94;max-width:900px;margin:12px 0 16px}}
    .hero p{{font-size:17px;line-height:1.6;max-width:830px;color:var(--muted)}} .stats,.case-meta,.links{{display:flex;gap:10px;flex-wrap:wrap}} .stats span,.case-meta span{{padding:8px 11px;border:1px solid var(--line);border-radius:99px;background:rgba(255,255,255,.6);font-size:13px}}
    .toolbar{{position:sticky;top:10px;z-index:5;margin:22px 0;padding:12px;display:flex;gap:10px;flex-wrap:wrap;background:rgba(255,250,241,.91);backdrop-filter:blur(14px);border:1px solid var(--line);border-radius:18px;box-shadow:0 8px 24px rgba(66,48,28,.10)}}
    .toolbar input{{flex:1;min-width:230px;padding:11px 14px;border:1px solid var(--line);border-radius:10px;background:white;font:inherit}} button{{border:1px solid var(--line);border-radius:10px;padding:10px 14px;background:white;color:var(--ink);cursor:pointer;font-weight:700}} button.active{{background:var(--ink);color:white;border-color:var(--ink)}}
    .case{{margin-top:24px;padding:22px;border:1px solid var(--line);border-radius:26px;background:rgba(255,250,241,.82);box-shadow:var(--shadow)}} .case.hidden{{display:none}} .case-head{{display:flex;justify-content:space-between;align-items:end;gap:16px}} h2{{font-family:"Baskerville","Iowan Old Style",serif;font-size:32px;margin:5px 0 0}} .pair-note{{margin:10px 0 18px;color:var(--muted)}}
    .media-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}} .media-card{{padding:15px;border:1px solid #e2d8c7;border-radius:18px;background:var(--card)}} .media-head{{display:flex;justify-content:space-between;gap:12px;align-items:start;margin-bottom:12px}} .media-label{{font-size:12px;color:var(--red);font-weight:800;text-transform:uppercase;letter-spacing:.08em}} h3{{margin:5px 0 0;font:600 13px/1.3 "Courier New",monospace;overflow-wrap:anywhere}} .id-map{{color:var(--muted);font:11px/1.4 "Courier New",monospace;text-align:right}}
    .video-pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px}} figure{{margin:0}} video{{display:block;width:100%;aspect-ratio:16/9;background:#090a09;border-radius:10px}} figcaption{{padding:6px 2px 0;color:var(--muted);font-size:12px}} .media-card p{{color:var(--muted);line-height:1.45;margin:8px 0}} .links a{{font-size:12px;border-bottom:1px solid currentColor}}
    .empty{{display:none;text-align:center;color:var(--muted);padding:50px}} @media(max-width:920px){{.media-grid{{grid-template-columns:1fr}}}} @media(max-width:620px){{main{{width:min(100% - 18px,1500px)}}.hero,.case{{padding:18px;border-radius:20px}}.case-head{{align-items:start;flex-direction:column}}.video-pair{{grid-template-columns:1fr}}}}
  </style>
</head>
<body><main>
  <section class="hero"><div class="eyebrow">Controlled-variable rigid-body simulation</div><h1>{_safe(page_title)}</h1><p>Each case contains at most three scene objects. One shared base acts as the anchor for background-color, object-color, and object-shape pairs. RGB and exact instance masks are rendered from the same geometry at the same simulation frame.</p>
    <div class="stats"><span id="visible-count">{len(cases)} / 50 cases</span><span>4 unique simulations per case</span><span>RGB + mask</span><span>1280 x 720, 30 fps</span><a href="{_safe(dataset_manifest)}">dataset manifest</a><a href="{_safe(case_catalog)}">case catalog</a></div>
  </section>
  <nav class="toolbar"><input id="search" type="search" placeholder="Search case ID or motion..."><button class="filter active" data-count="all">All</button><button class="filter" data-count="1">1 object</button><button class="filter" data-count="2">2 objects</button><button class="filter" data-count="3">3 objects</button></nav>
  <div id="cases">{sections}</div><p id="empty" class="empty">No cases match this filter.</p>
</main>
<script>
  const cards=[...document.querySelectorAll('.case')], search=document.querySelector('#search'), count=document.querySelector('#visible-count'), empty=document.querySelector('#empty'); let active='all';
  function apply(){{const q=search.value.trim().toLowerCase();let shown=0;for(const card of cards){{const okCount=active==='all'||card.dataset.count===active;const okSearch=!q||card.dataset.search.includes(q);card.classList.toggle('hidden',!(okCount&&okSearch));if(okCount&&okSearch)shown++}}count.textContent=`${{shown}} / 50 cases`;empty.style.display=shown?'none':'block'}}
  document.querySelectorAll('.filter').forEach(button=>button.addEventListener('click',()=>{{document.querySelector('.filter.active').classList.remove('active');button.classList.add('active');active=button.dataset.count;apply()}}));search.addEventListener('input',apply);
</script></body></html>"""


def main() -> None:
    args = parse_args()
    output_root = args.output_root or args.dataset_root / "html"
    output_root.mkdir(parents=True, exist_ok=True)
    _ensure_viewer_links(args.dataset_root, output_root)
    manifest = json.loads((args.dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    output_path = output_root / "index.html"
    output_path.write_text(build_html(manifest, output_root, args.page_title), encoding="utf-8")
    print(json.dumps({"case_count": len(manifest.get("cases", [])), "index_html": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
