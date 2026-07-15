#!/usr/bin/env python3
"""Build a synchronized gallery for Scheme-C's largest per-case regressions."""
from __future__ import annotations

import argparse
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote


DEFAULT_FORMAL_ROOT = Path(
    "/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ"
)
DEFAULT_ALLOWLIST = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/scheme_c_largest_regression_gallery_20260715"
)
DISPLAY_PREFIX = Path("/data/gaoya")
SCALES = ("1p0", "1p5", "2p0")


@dataclass(frozen=True)
class Method:
    method_id: str
    label: str
    root: Path
    family: str
    scale: str | None = None
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--input-list", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-per-scale", type=int, default=3)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def case_stems(input_list: Path) -> list[str]:
    return [
        Path(line.strip()).stem
        for line in input_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def find_case_json(root: Path, stem: str) -> Path | None:
    direct = root / f"{stem}.json"
    if direct.is_file():
        return direct
    matches = [
        path
        for path in root.rglob(f"{stem}.json")
        if path.name not in {"summary.json", "batch_manifest.json", "result.json"}
    ]
    return sorted(matches, key=lambda path: (len(path.parts), str(path)))[0] if matches else None


def score(payload: dict[str, Any] | None, key: str) -> float | None:
    if not payload:
        return None
    block = payload.get(key)
    if not isinstance(block, dict):
        return None
    value = block.get("score")
    return float(value) if isinstance(value, (int, float)) else None


def display_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(DISPLAY_PREFIX))
    except ValueError:
        return str(resolved)


def safe_link(target: Path, link: Path) -> None:
    target = target.expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"refusing to replace non-symlink asset: {link}")
    link.symlink_to(target)


def method_specs(formal_root: Path) -> list[Method]:
    scheme_root = formal_root / "train_stage1b_scheme_c_entity_caption_physical_fresh_20260714T174707Z/step-003500"
    mix_root = formal_root / "train_stage1b_mixdataset/step-003500"
    methods: list[Method] = []
    for scale in SCALES:
        label_scale = scale.replace("p", ".") + "x"
        methods.append(
            Method(
                f"scheme_c_{scale}",
                f"Scheme-C · residual {label_scale}",
                scheme_root / f"object_residual_{scale}x/results",
                "scheme",
                scale,
                "step-003500 · null negative prompt",
            )
        )
    for scale in SCALES:
        label_scale = scale.replace("p", ".") + "x"
        methods.append(
            Method(
                f"mixdataset_{scale}",
                f"Mixdataset · residual {label_scale}",
                mix_root / f"object_residual_{scale}x",
                "mix",
                scale,
                "step-003500 · default negative prompt",
            )
        )
    methods.extend(
        [
            Method(
                "wan_base",
                "Wan2.2 TI2V-5B base",
                formal_root / "basemodel/wan2p2_ti2v5B_aligned49_steps40_512x896_49f_defaultnegprompt",
                "baseline",
                note="base model · default negative prompt",
            ),
            Method(
                "raw_physics_lora",
                "Raw physics LoRA",
                formal_root / "loramodel/wan_openvid_0613pybullet_lorav2v_step000500_aligned49_steps40_512x896_ctx08_49f_defaultnegprompt",
                "baseline",
                note="step-000500 · default negative prompt",
            ),
            Method(
                "stability_v3",
                "Stability-v3",
                formal_root / "train_stage1b_kubric0708/train_stage1b_kubric0708_stability_v3_from_scratch_20260711T144000Z_step-003500_steps40_512x896_ctx08_49f_defaultnegprompt",
                "baseline",
                note="step-003500 · default negative prompt",
            ),
            Method(
                "physrvg",
                "PhysRVG",
                formal_root / "physRVG_steps40_512x896_08_49f",
                "baseline",
                note="40 steps · context 8 · 49 frames",
            ),
        ]
    )
    return methods


def metric_values(payload: dict[str, Any]) -> dict[str, float | None]:
    return {
        "physics_iq_ctx": score(payload, "physics_iq_with_context"),
        "physics_iq_noctx": score(payload, "physics_iq_without_context"),
        "pmf_ctx": score(payload, "pmf_with_context"),
        "videophy2": score(payload, "videophy2"),
        "cosmos": score(payload, "cosmos_reason1"),
    }


def build_index(methods: list[Method], stems: list[str]) -> dict[str, dict[str, tuple[Path, dict[str, Any]]]]:
    index: dict[str, dict[str, tuple[Path, dict[str, Any]]]] = {}
    for method in methods:
        method_cases: dict[str, tuple[Path, dict[str, Any]]] = {}
        for stem in stems:
            metadata_path = find_case_json(method.root, stem)
            if metadata_path is None:
                continue
            payload = load_json(metadata_path)
            if payload is None:
                continue
            output_video = Path(str(payload.get("output_video", ""))).expanduser()
            if not output_video.is_file():
                sibling = metadata_path.with_suffix(".mp4")
                if not sibling.is_file():
                    continue
                payload["output_video"] = str(sibling)
            method_cases[stem] = (metadata_path, payload)
        index[method.method_id] = method_cases
    return index


def choose_cases(
    index: dict[str, dict[str, tuple[Path, dict[str, Any]]]],
    stems: list[str],
    top_per_scale: int,
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, list[dict[str, Any]]]]:
    deltas: dict[str, dict[str, float]] = {}
    rankings: dict[str, list[dict[str, Any]]] = {}
    selected: set[str] = set()
    for scale in SCALES:
        scheme_cases = index[f"scheme_c_{scale}"]
        mix_cases = index[f"mixdataset_{scale}"]
        rows: list[dict[str, Any]] = []
        for stem in stems:
            if stem not in scheme_cases or stem not in mix_cases:
                continue
            scheme_score = score(scheme_cases[stem][1], "physics_iq_with_context")
            mix_score = score(mix_cases[stem][1], "physics_iq_with_context")
            if scheme_score is None or mix_score is None:
                continue
            delta = scheme_score - mix_score
            deltas.setdefault(stem, {})[scale] = delta
            rows.append(
                {
                    "case": stem,
                    "scale": scale,
                    "scheme_c": scheme_score,
                    "mixdataset": mix_score,
                    "delta": delta,
                }
            )
        rows.sort(key=lambda row: (row["delta"], row["case"]))
        rankings[scale] = rows[:top_per_scale]
        selected.update(row["case"] for row in rows[:top_per_scale])
    ordered = sorted(selected, key=lambda stem: (min(deltas[stem].values()), stem))
    return ordered, deltas, rankings


def fmt_metric(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def make_gallery(
    output_dir: Path,
    methods: list[Method],
    index: dict[str, dict[str, tuple[Path, dict[str, Any]]]],
    selected: list[str],
    deltas: dict[str, dict[str, float]],
    rankings: dict[str, list[dict[str, Any]]],
    allowlist: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for case_index, stem in enumerate(selected, start=1):
        reference = index["scheme_c_1p0"].get(stem)
        if reference is None:
            continue
        _, reference_payload = reference
        source_path = Path(str(reference_payload["source_video"]))
        source_link = output_dir / "videos" / stem / "source.mp4"
        safe_link(source_path, source_link)
        videos: list[dict[str, Any]] = []
        for method in methods:
            item = index[method.method_id].get(stem)
            if item is None:
                missing.append({"case": stem, "method": method.method_id, "reason": "metadata/output missing"})
                continue
            metadata_path, payload = item
            output_path = Path(str(payload["output_video"]))
            video_link = output_dir / "videos" / stem / f"{method.method_id}.mp4"
            metadata_link = output_dir / "videos" / stem / f"{method.method_id}.json"
            safe_link(output_path, video_link)
            safe_link(metadata_path, metadata_link)
            values = metric_values(payload)
            videos.append(
                {
                    "id": method.method_id,
                    "label": method.label,
                    "family": method.family,
                    "scale": method.scale,
                    "note": method.note,
                    "video": str(video_link.relative_to(output_dir)),
                    "metadata": str(metadata_link.relative_to(output_dir)),
                    "output_path": str(output_path.resolve()),
                    "display_output_path": display_path(output_path),
                    "metrics": values,
                    "delta_vs_mix": deltas.get(stem, {}).get(method.scale) if method.family == "scheme" else None,
                }
            )
        records.append(
            {
                "rank": case_index,
                "case": stem,
                "caption": reference_payload.get("input_caption", ""),
                "input_json": reference_payload.get("input_json", ""),
                "source_video": str(source_link.relative_to(output_dir)),
                "source_path": str(source_path.resolve()),
                "display_source_path": display_path(source_path),
                "deltas": deltas[stem],
                "worst_delta": min(deltas[stem].values()),
                "videos": videos,
            }
        )

    manifest = {
        "ranking_metric": "physics_iq_with_context.score",
        "comparison": "Scheme-C step-003500 minus mixdataset step-003500 at matching residual scale",
        "selection": "union of the worst 3 allowlisted cases at each residual scale",
        "allowlist": str(allowlist.resolve()),
        "num_allowlisted_cases": len(case_stems(allowlist)),
        "num_selected_cases": len(records),
        "num_generated_videos": sum(len(record["videos"]) for record in records),
        "rankings": rankings,
        "missing": missing,
        "cases": records,
    }
    (output_dir / "gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "index.html").write_text(render_html(manifest), encoding="utf-8")
    return manifest


def metric_html(metrics: dict[str, float | None]) -> str:
    items = [
        ("PhysicsIQ ctx", fmt_metric(metrics["physics_iq_ctx"])),
        ("PhysicsIQ no ctx", fmt_metric(metrics["physics_iq_noctx"])),
        ("PMF ctx", fmt_metric(metrics["pmf_ctx"], 3)),
        ("VideoPhy2", fmt_metric(metrics["videophy2"], 1)),
        ("Cosmos", fmt_metric(metrics["cosmos"], 1)),
    ]
    return "".join(
        f"<span><b>{html.escape(label)}</b>{html.escape(value)}</span>" for label, value in items
    )


def render_html(manifest: dict[str, Any]) -> str:
    sections: list[str] = []
    for record in manifest["cases"]:
        delta_chips = "".join(
            f"<span class='delta'>residual {scale.replace('p', '.')}x <b>{value:+.2f}</b></span>"
            for scale, value in sorted(record["deltas"].items())
        )
        figures = [
            "<figure class='video-item source'>"
            f"<video controls preload='metadata' src='{quote(record['source_video'])}'></video>"
            "<figcaption><div class='item-title'><strong>Source / reference</strong>"
            "<span class='tag source-tag'>GT timeline</span></div>"
            "<p>Original case video used as the visual reference.</p>"
            f"<code>{html.escape(record['display_source_path'])}</code></figcaption></figure>"
        ]
        for video in record["videos"]:
            delta = video["delta_vs_mix"]
            delta_badge = (
                f"<span class='tag regression'>Δ {delta:+.2f}</span>" if delta is not None else ""
            )
            leader = " data-sync-leader='true'" if video["id"] == "scheme_c_1p0" else ""
            figures.append(
                f"<figure class='video-item {html.escape(video['family'])}'>"
                f"<video controls preload='metadata' src='{quote(video['video'])}'{leader}></video>"
                "<figcaption>"
                f"<div class='item-title'><strong>{html.escape(video['label'])}</strong>{delta_badge}</div>"
                f"<p>{html.escape(video['note'])}</p>"
                f"<div class='metrics'>{metric_html(video['metrics'])}</div>"
                f"<code>{html.escape(video['display_output_path'])}</code>"
                f"<a href='{quote(video['metadata'])}' target='_blank'>metadata JSON</a>"
                "</figcaption></figure>"
            )
        sections.append(
            f"<section data-case='{html.escape(record['case'].lower())}'>"
            "<header class='case-head'><div class='case-name'>"
            f"<span class='rank'>#{record['rank']:02d}</span><h2>{html.escape(record['case'])}</h2></div>"
            f"<div class='deltas'>{delta_chips}</div></header>"
            f"<p class='caption'>{html.escape(record['caption'])}</p>"
            "<div class='timeline'><button data-action='play' title='Play synchronized'>▶</button>"
            "<button data-action='pause' title='Pause all'>Ⅱ</button>"
            "<button data-action='reset' title='Reset timeline'>↺</button>"
            "<input data-timeline type='range' min='0' max='1000' value='0' aria-label='Normalized frame timeline'>"
            "<output>0.0%</output><span>normalized frame progress</span></div>"
            f"<div class='media'>{''.join(figures)}</div></section>"
        )

    scale_rows = []
    for scale, rows in manifest["rankings"].items():
        for rank, row in enumerate(rows, start=1):
            scale_rows.append(
                f"<tr><td>{scale.replace('p', '.')}x</td><td>{rank}</td>"
                f"<td>{html.escape(row['case'])}</td><td>{row['scheme_c']:.2f}</td>"
                f"<td>{row['mixdataset']:.2f}</td><td class='negative'>{row['delta']:+.2f}</td></tr>"
            )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scheme-C Largest Regressions</title><style>
:root{{--ink:#17211d;--muted:#68736e;--paper:#f1f3ef;--panel:#fff;--line:#c7cec9;--red:#a92f2f;--green:#216844;--blue:#285f85;--amber:#9a5b13}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px "IBM Plex Sans","Noto Sans",sans-serif;letter-spacing:0}}
main{{max-width:1920px;margin:auto;padding:0 26px 72px}}.top{{position:sticky;top:0;z-index:8;padding:18px 0 14px;background:rgba(241,243,239,.97);border-bottom:1px solid var(--line)}}
h1{{margin:0 0 5px;font:700 27px "IBM Plex Serif","Noto Serif",serif}}.lede{{margin:0 0 12px;color:var(--muted);line-height:1.45}}.warning{{color:#713b06;font-weight:650}}
.tools{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}#search{{width:min(560px,100%);padding:9px 11px;border:1px solid #89958e;background:white;font:inherit}}button{{width:34px;height:32px;border:1px solid #89958e;background:#fff;color:var(--ink);cursor:pointer;font:inherit}}button:hover{{border-color:var(--red);color:var(--red)}}
details{{margin-top:12px}}summary{{cursor:pointer;color:var(--blue);width:max-content}}table{{margin-top:10px;border-collapse:collapse;background:#fff;font-variant-numeric:tabular-nums}}th,td{{padding:6px 9px;border:1px solid var(--line);text-align:left}}.negative{{color:var(--red);font-weight:700}}
section{{padding:27px 0 34px;border-bottom:1px solid var(--line)}}.case-head{{display:flex;justify-content:space-between;gap:18px;align-items:flex-start}}.case-name{{display:flex;gap:11px;align-items:baseline;min-width:0}}.rank{{color:var(--red);font:700 13px ui-monospace,monospace}}h2{{margin:0;font-size:19px;overflow-wrap:anywhere}}.deltas{{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}}.delta{{padding:5px 7px;background:#fff;border:1px solid var(--line);font-variant-numeric:tabular-nums}}.delta b{{color:var(--red)}}.caption{{max-width:1250px;margin:8px 0 12px;color:#46514c;line-height:1.45}}
.timeline{{display:grid;grid-template-columns:34px 34px 34px minmax(180px,720px) 58px auto;gap:6px;align-items:center;margin-bottom:13px}}.timeline input{{width:100%;accent-color:var(--red)}}.timeline output{{font:12px ui-monospace,monospace;text-align:right}}.timeline>span{{color:var(--muted);font-size:12px}}
.media{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px}}figure{{margin:0;background:var(--panel);border:1px solid var(--line);min-width:0}}figure.source{{border:2px solid var(--green)}}figure.scheme{{border-top:4px solid var(--red)}}figure.mix{{border-top:4px solid var(--blue)}}figure.baseline{{border-top:4px solid #707a75}}video{{display:block;width:100%;aspect-ratio:7/4;object-fit:contain;background:#101310}}
figcaption{{display:grid;gap:7px;padding:10px}}.item-title{{display:flex;justify-content:space-between;gap:8px;align-items:flex-start}}.item-title strong{{font-size:14px}}.tag{{padding:2px 5px;border:1px solid currentColor;font:700 11px ui-monospace,monospace;white-space:nowrap}}.regression{{color:var(--red)}}.source-tag{{color:var(--green)}}figcaption p{{min-height:18px;margin:0;color:var(--muted);font-size:12px}}.metrics{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:3px 8px;padding:7px 0;border-top:1px solid #e2e6e3;border-bottom:1px solid #e2e6e3}}.metrics span{{display:flex;justify-content:space-between;gap:7px;font:11px ui-monospace,monospace}}.metrics b{{font-weight:500;color:var(--muted)}}code{{font:10px ui-monospace,monospace;color:#44504a;overflow-wrap:anywhere;word-break:break-word}}a{{color:var(--blue);font-size:12px;width:max-content}}
@media(max-width:1450px){{.media{{grid-template-columns:repeat(3,minmax(0,1fr))}}}}@media(max-width:1050px){{.media{{grid-template-columns:repeat(2,minmax(0,1fr))}}.case-head{{flex-direction:column}}.deltas{{justify-content:flex-start}}}}@media(max-width:680px){{main{{padding:0 13px 55px}}.media{{grid-template-columns:1fr}}.timeline{{grid-template-columns:34px 34px 34px minmax(90px,1fr) 54px}}.timeline>span{{display:none}}h1{{font-size:23px}}}}
</style></head><body><main><div class="top"><h1>Scheme-C Largest Per-Case Regressions</h1>
<p class="lede">{manifest['num_selected_cases']} cases selected from the strict 67-case allowlist: union of the worst three at residual 1.0x, 1.5x, and 2.0x. Ranking metric is PhysicsIQ with context, Δ = Scheme-C − mixdataset at matching scale. <span class="warning">Scheme-C used a null negative prompt; mixdataset and most baselines used the default negative prompt.</span></p>
<div class="tools"><input id="search" type="search" placeholder="Filter case name"><span>{manifest['num_generated_videos']} generated outputs + source references</span></div>
<details><summary>Show exact top-3 ranking</summary><table><thead><tr><th>Residual</th><th>Rank</th><th>Case</th><th>Scheme-C</th><th>Mixdataset</th><th>Δ</th></tr></thead><tbody>{''.join(scale_rows)}</tbody></table></details></div>
{''.join(sections)}</main><script>
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
document.querySelector('#search').addEventListener('input',e=>{{const q=e.target.value.trim().toLowerCase();document.querySelectorAll('section').forEach(s=>s.hidden=!s.dataset.case.includes(q));}});
document.querySelectorAll('section').forEach(section=>{{
  const videos=[...section.querySelectorAll('video')], leader=section.querySelector('[data-sync-leader]')||videos[0];
  const slider=section.querySelector('[data-timeline]'), output=section.querySelector('output'); let syncing=false, timer=null;
  const ready=v=>Number.isFinite(v.duration)&&v.duration>0;
  const fraction=()=>ready(leader)?clamp(leader.currentTime/leader.duration,0,1):Number(slider.value)/1000;
  const seekAll=f=>{{syncing=true;videos.forEach(v=>{{if(ready(v))v.currentTime=clamp(f,0,1)*v.duration}});slider.value=Math.round(f*1000);output.value=`${{(f*100).toFixed(1)}}%`;setTimeout(()=>syncing=false,0)}};
  const syncRates=()=>{{if(!ready(leader))return;videos.forEach(v=>{{if(ready(v))v.playbackRate=clamp(v.duration/leader.duration,.25,4)}})}};
  const stopTimer=()=>{{if(timer!==null){{clearInterval(timer);timer=null}}}};
  slider.addEventListener('input',()=>seekAll(Number(slider.value)/1000));
  videos.forEach(v=>v.addEventListener('loadedmetadata',()=>seekAll(Number(slider.value)/1000)));
  leader.addEventListener('timeupdate',()=>{{if(!syncing){{const f=fraction();slider.value=Math.round(f*1000);output.value=`${{(f*100).toFixed(1)}}%`}}}});
  section.querySelector('[data-action=play]').onclick=()=>{{const f=fraction();seekAll(f>=.999?0:f);syncRates();videos.forEach(v=>v.play().catch(()=>{{}}));stopTimer();timer=setInterval(()=>{{const p=fraction();videos.forEach(v=>{{if(v!==leader&&ready(v)&&Math.abs(v.currentTime/v.duration-p)>.035)v.currentTime=p*v.duration}})}},250)}};
  section.querySelector('[data-action=pause]').onclick=()=>{{videos.forEach(v=>v.pause());stopTimer()}};
  section.querySelector('[data-action=reset]').onclick=()=>{{videos.forEach(v=>v.pause());stopTimer();seekAll(0)}};
  leader.addEventListener('ended',()=>{{videos.forEach(v=>v.pause());stopTimer();seekAll(1)}});
}});
</script></body></html>"""


def main() -> None:
    args = parse_args()
    formal_root = args.formal_root.expanduser().resolve()
    input_list = args.input_list.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    methods = method_specs(formal_root)
    absent_roots = [str(method.root) for method in methods if not method.root.is_dir()]
    if absent_roots:
        raise FileNotFoundError("missing method roots:\n" + "\n".join(absent_roots))
    stems = case_stems(input_list)
    index = build_index(methods, stems)
    selected, deltas, rankings = choose_cases(index, stems, args.top_per_scale)
    manifest = make_gallery(output_dir, methods, index, selected, deltas, rankings, input_list)
    print(
        f"gallery={output_dir / 'index.html'} cases={manifest['num_selected_cases']} "
        f"outputs={manifest['num_generated_videos']} missing={len(manifest['missing'])}"
    )
    for scale, rows in rankings.items():
        print(f"residual_{scale}x")
        for rank, row in enumerate(rows, start=1):
            print(
                f"  {rank}. {row['case']} scheme={row['scheme_c']:.2f} "
                f"mix={row['mixdataset']:.2f} delta={row['delta']:+.2f}"
            )


if __name__ == "__main__":
    main()
