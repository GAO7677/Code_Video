#!/usr/bin/env python3
"""Group Scheme-C outputs by source case and build a browser video gallery."""
from __future__ import annotations

import argparse
import html
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote


CASE_JSON_IGNORES = {
    "batch_manifest.json",
    "result.json",
    "summary.json",
    "gallery_manifest.json",
}
DISPLAY_PATH_PREFIX = Path("/data/gaoya")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--input-list", type=Path, default=None)
    return parser.parse_args()


def load_case_payload(path: Path) -> dict[str, Any] | None:
    if path.name in CASE_JSON_IGNORES:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    required = ("input_json", "source_video", "output_video")
    if not all(isinstance(payload.get(key), str) and payload[key] for key in required):
        return None
    output_video = Path(payload["output_video"]).expanduser().resolve()
    source_video = Path(payload["source_video"]).expanduser().resolve()
    if not output_video.is_file() or not source_video.is_file():
        return None
    payload["_metadata_path"] = str(path.resolve())
    payload["_output_video"] = str(output_video)
    payload["_source_video"] = str(source_video)
    return payload


def variant_from_path(root: Path, metadata_path: Path) -> tuple[str, str, int, float]:
    relative = metadata_path.resolve().relative_to(root.resolve())
    parts = relative.parts[:-1]
    step = next((part for part in parts if re.fullmatch(r"step-\d+", part)), "unknown-step")
    residual_part = next((part for part in parts if part.startswith("object_residual_")), None)
    if residual_part:
        match = re.fullmatch(r"object_residual_(\d+)p(\d+)x", residual_part)
        scale = float(f"{match.group(1)}.{match.group(2)}") if match else -1.0
        mode = residual_part.replace("object_residual_", "residual ").replace("p", ".")
    else:
        scale = -1.0
        mode = "default"
    step_number = int(step.split("-")[-1]) if step != "unknown-step" else -1
    variant_id = f"{step}__{residual_part or 'default'}"
    return variant_id, f"{step} / {mode}", step_number, scale


def safe_link(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"refusing to replace non-symlink gallery asset: {link}")
    link.symlink_to(target.resolve())


def ordered_case_stems(input_list: Path | None) -> dict[str, int]:
    if input_list is None or not input_list.is_file():
        return {}
    stems = [
        Path(line.strip()).stem
        for line in input_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {stem: index for index, stem in enumerate(stems)}


def binding_summary(payload: dict[str, Any]) -> str:
    binding = payload.get("object_debug", {}).get("entity_id_binding", {})
    if not binding.get("enabled"):
        return "binding unavailable"
    matched = binding.get("matched", [])
    if not matched:
        return "binding enabled, no matched slot"
    labels = [
        f"slot {item.get('slot_id')} -> ID {item.get('entity_id')} ({item.get('grounding_phrase', '?')})"
        for item in matched
    ]
    return "; ".join(labels)


def display_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(DISPLAY_PATH_PREFIX))
    except ValueError:
        return str(resolved)


def write_gallery(root: Path, output_dir: Path, input_list: Path | None) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metadata_path in root.rglob("*.json"):
        if output_dir in metadata_path.parents:
            continue
        payload = load_case_payload(metadata_path)
        if payload is None:
            continue
        case_stem = Path(payload["input_json"]).stem
        variant_id, label, step_number, scale = variant_from_path(root, metadata_path)
        grouped[case_stem].append(
            {
                "payload": payload,
                "variant_id": variant_id,
                "label": label,
                "step_number": step_number,
                "scale": scale,
            }
        )

    order = ordered_case_stems(input_list)
    case_stems = sorted(grouped, key=lambda stem: (order.get(stem, 10**9), stem))
    videos_dir = output_dir / "videos"
    records: list[dict[str, Any]] = []
    for case_stem in case_stems:
        variants_by_id: dict[str, dict[str, Any]] = {}
        for item in grouped[case_stem]:
            variants_by_id[item["variant_id"]] = item
        variants = sorted(
            variants_by_id.values(),
            key=lambda item: (item["step_number"], item["scale"], item["label"]),
        )
        first_payload = variants[0]["payload"]
        case_dir = videos_dir / case_stem
        source_link = case_dir / "source.mp4"
        safe_link(Path(first_payload["_source_video"]), source_link)
        rendered_variants = []
        for item in variants:
            variant_link = case_dir / f"{item['variant_id']}.mp4"
            safe_link(Path(item["payload"]["_output_video"]), variant_link)
            rendered_variants.append(
                {
                    "id": item["variant_id"],
                    "label": item["label"],
                    "video": str(variant_link.relative_to(output_dir)),
                    "output_path": item["payload"]["_output_video"],
                    "display_output_path": display_path(item["payload"]["_output_video"]),
                    "metadata": os.path.relpath(item["payload"]["_metadata_path"], output_dir),
                    "binding": binding_summary(item["payload"]),
                }
            )
        records.append(
            {
                "case": case_stem,
                "input_json": first_payload["input_json"],
                "display_input_json": display_path(first_payload["input_json"]),
                "caption": first_payload.get("input_caption", ""),
                "source_video": str(source_link.relative_to(output_dir)),
                "source_path": first_payload["_source_video"],
                "display_source_path": display_path(first_payload["_source_video"]),
                "variants": rendered_variants,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "root": str(root),
        "num_cases": len(records),
        "num_generated_videos": sum(len(record["variants"]) for record in records),
        "cases": records,
    }
    (output_dir / "gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    sections = []
    for index, record in enumerate(records, start=1):
        figures = [
            "<figure class='source'>"
            f"<video controls preload='metadata' src='{quote(record['source_video'])}'></video>"
            "<figcaption><strong>Source video</strong><span>original input timeline</span>"
            f"<code class='path'>{html.escape(record['display_source_path'])}</code></figcaption>"
            "</figure>"
        ]
        for variant in record["variants"]:
            figures.append(
                "<figure>"
                f"<video controls preload='metadata' src='{quote(variant['video'])}'></video>"
                f"<figcaption><strong>{html.escape(variant['label'])}</strong>"
                f"<span>{html.escape(variant['binding'])}</span>"
                f"<code class='path'>{html.escape(variant['display_output_path'])}</code>"
                f"<a href='{quote(variant['metadata'])}'>metadata</a></figcaption>"
                "</figure>"
            )
        sections.append(
            f"<section data-case='{html.escape(record['case'].lower())}'>"
            "<header>"
            f"<div><span class='case-index'>{index:02d}</span><h2>{html.escape(record['case'])}</h2></div>"
            "<div class='actions'><button data-action='play'>Play all</button>"
            "<button data-action='pause'>Pause all</button><button data-action='reset'>Reset</button></div>"
            "</header>"
            f"<p class='input-path'><strong>Input JSON</strong><code>{html.escape(record['display_input_json'])}</code></p>"
            f"<p class='caption'>{html.escape(record['caption'])}</p>"
            f"<div class='media'>{''.join(figures)}</div>"
            "</section>"
        )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scheme-C case comparison</title>
<style>
:root{{--ink:#1f2522;--muted:#65706b;--line:#cbd2ce;--paper:#f4f6f3;--panel:#fff;--source:#26734d;--accent:#a53f2b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px "IBM Plex Sans","Noto Sans",sans-serif;letter-spacing:0}}
main{{max-width:1800px;margin:auto;padding:24px 28px 80px}}.top{{position:sticky;top:0;z-index:4;background:rgba(244,246,243,.96);border-bottom:1px solid var(--line);padding:18px 0 14px}}
h1{{font:700 28px "IBM Plex Serif","Noto Serif",serif;margin:0 0 6px}}.summary{{color:var(--muted);margin:0 0 14px}}
.tools{{display:flex;gap:14px;align-items:center;flex-wrap:wrap}}input[type=search]{{width:min(520px,100%);padding:10px 12px;border:1px solid #9da9a3;background:#fff;font:inherit}}
label{{display:flex;align-items:center;gap:7px;color:var(--muted)}}section{{padding:28px 0 34px;border-bottom:1px solid var(--line)}}section>header{{display:flex;justify-content:space-between;gap:20px;align-items:center;margin-bottom:8px}}
section>header>div:first-child{{display:flex;align-items:baseline;gap:12px;min-width:0}}h2{{font:650 19px "IBM Plex Sans","Noto Sans",sans-serif;margin:0;overflow-wrap:anywhere}}.case-index{{color:var(--accent);font:700 13px monospace}}
.input-path{{display:flex;gap:9px;align-items:baseline;margin:0 0 7px;color:var(--muted);min-width:0}}.input-path strong{{flex:none;font-size:12px;text-transform:uppercase}}.input-path code,.path{{font:12px ui-monospace,"SFMono-Regular",Consolas,monospace;overflow-wrap:anywhere;word-break:break-word;color:#37423d}}.caption{{max-width:1200px;color:#45504b;margin:0 0 16px;line-height:1.5}}.actions{{display:flex;gap:7px;flex:none}}button{{border:1px solid #8e9b95;background:#fff;padding:7px 10px;color:var(--ink);font:inherit;cursor:pointer}}button:hover{{border-color:var(--accent);color:var(--accent)}}
.media{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}}figure{{margin:0;background:var(--panel);border:1px solid var(--line);min-width:0}}figure.source{{border:2px solid var(--source)}}video{{display:block;width:100%;aspect-ratio:7/4;object-fit:contain;background:#111}}
figcaption{{display:grid;gap:5px;padding:10px 11px;line-height:1.35}}figcaption span{{color:var(--muted);font-size:13px;overflow-wrap:anywhere}}figcaption .path{{padding-top:5px;border-top:1px solid #e4e8e5}}figcaption a{{color:var(--accent);font-size:13px;width:max-content}}
.empty{{padding:60px 0;color:var(--muted)}}@media(max-width:1100px){{.media{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:700px){{main{{padding:16px 14px 60px}}.media{{grid-template-columns:1fr}}section>header{{align-items:flex-start;flex-direction:column}}}}
</style></head><body><main>
<div class="top"><h1>Scheme-C Outputs By Source Case</h1>
<p class="summary">{len(records)} cases / {manifest['num_generated_videos']} generated videos. Source is always first; generated videos are grouped by checkpoint and object residual scale. Displayed paths omit <code>/data/gaoya/</code>.</p>
<div class="tools"><input id="search" type="search" placeholder="Filter case name"><label><input id="sync" type="checkbox" checked> Synchronize playback within each case</label></div></div>
{''.join(sections) if sections else '<p class="empty">No completed output videos found yet.</p>'}
</main><script>
const syncBox=document.querySelector('#sync');
document.querySelector('#search').addEventListener('input',e=>{{const q=e.target.value.trim().toLowerCase();document.querySelectorAll('section').forEach(s=>s.hidden=!s.dataset.case.includes(q));}});
document.querySelectorAll('section').forEach(section=>{{
  const videos=[...section.querySelectorAll('video')]; let lock=false;
  const apply=(source,fn)=>{{if(!syncBox.checked||lock)return;lock=true;videos.filter(v=>v!==source).forEach(fn);setTimeout(()=>lock=false,0);}};
  videos.forEach(v=>{{v.addEventListener('play',()=>apply(v,x=>{{x.currentTime=v.currentTime;x.play().catch(()=>{{}})}}));v.addEventListener('pause',()=>apply(v,x=>x.pause()));v.addEventListener('seeking',()=>apply(v,x=>x.currentTime=v.currentTime));}});
  section.querySelector('[data-action=play]').onclick=()=>videos.forEach(v=>v.play().catch(()=>{{}}));
  section.querySelector('[data-action=pause]').onclick=()=>videos.forEach(v=>v.pause());
  section.querySelector('[data-action=reset]').onclick=()=>videos.forEach(v=>{{v.pause();v.currentTime=0}});
}});
</script></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = (args.output_dir or root / "_case_gallery").expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest = write_gallery(root, output_dir, args.input_list)
    print(
        f"gallery={output_dir / 'index.html'} cases={manifest['num_cases']} "
        f"generated_videos={manifest['num_generated_videos']}"
    )


if __name__ == "__main__":
    main()
