#!/usr/bin/env python3
"""Build a dedicated five-case CoTracker trajectory-overlay comparison page."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path(__file__).with_name("validation_30cases_config.json")
ENTRY_IDS = (
    "cotracker_trajectory_step0500",
    "cotracker_trajectory_step1000",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def replace_symlink(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source.resolve())


def metric(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    config = load_json(args.config.expanduser().resolve())
    validation_root = Path(config["output_root"]).expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else validation_root / "trajectory_overlay_hub"
    )
    media_root = output_root / "media"
    media_root.mkdir(parents=True, exist_ok=True)

    reports = {
        entry_id: load_json(
            validation_root / "trajectory_overlays" / entry_id / "report.json"
        )
        for entry_id in ENTRY_IDS
    }
    entries = [reports[entry_id]["entry"] for entry_id in ENTRY_IDS]
    cases_by_entry = {
        entry_id: {row["case_id"]: row for row in reports[entry_id]["cases"]}
        for entry_id in ENTRY_IDS
    }
    case_ids = [row["case_id"] for row in reports[ENTRY_IDS[0]]["cases"]]
    if len(case_ids) != 5:
        raise RuntimeError(f"expected five cases, found {len(case_ids)}")
    for entry_id in ENTRY_IDS[1:]:
        if set(cases_by_entry[entry_id]) != set(case_ids):
            raise RuntimeError(f"case mismatch for {entry_id}")

    aggregate_rows = []
    for entry in entries:
        entry_id = entry["entry_id"]
        rows = list(cases_by_entry[entry_id].values())
        mean_ade = sum(float(row["trajectory_normalized_ade"]) for row in rows) / len(rows)
        aggregate_rows.append(
            f'<tr><td><span class="swatch" style="background:{html.escape(entry["color"], quote=True)}"></span>'
            f'<strong>step-{int(entry["step"]):04d}</strong></td>'
            f'<td>{metric(reports[entry_id]["mean_trajectory_loss"], 9)}</td>'
            f'<td>{metric(reports[entry_id]["mean_trajectory_coordinate_loss"], 9)}</td>'
            f'<td>{metric(reports[entry_id]["mean_trajectory_visibility_penalty"], 9)}</td>'
            f'<td>{metric(mean_ade, 6)}</td></tr>'
        )

    case_options = []
    case_sections = []
    for case_index, case_id in enumerate(case_ids, start=1):
        first = cases_by_entry[ENTRY_IDS[0]][case_id]
        case_options.append(
            f'<option value="{html.escape(case_id, quote=True)}">'
            f'{case_index:02d} · {html.escape(first["prompt"])}</option>'
        )
        losses = {
            entry_id: float(cases_by_entry[entry_id][case_id]["trajectory_loss"])
            for entry_id in ENTRY_IDS
        }
        best_entry = min(losses, key=losses.get)
        checkpoint_blocks = []
        for entry in entries:
            entry_id = entry["entry_id"]
            row = cases_by_entry[entry_id][case_id]
            case_root = (
                validation_root
                / "trajectory_overlays"
                / entry_id
                / case_id
            )
            stem = f"{case_id}__{entry_id}"
            links = {
                "video": media_root / f"{stem}.mp4",
                "poster": media_root / f"{stem}.jpg",
                "metrics": media_root / f"{stem}.json",
                "tracks": media_root / f"{stem}.npz",
            }
            replace_symlink(case_root / "trajectory_overlay.mp4", links["video"])
            replace_symlink(case_root / "trajectory_preview.jpg", links["poster"])
            replace_symlink(case_root / "metrics.json", links["metrics"])
            replace_symlink(case_root / "trajectories.npz", links["tracks"])
            best = " checkpoint-best" if entry_id == best_entry else ""
            checkpoint_blocks.append(
                f'<article class="checkpoint{best}"><header><div><span class="checkpoint-tag">'
                f'step-{int(entry["step"]):04d}</span><h3>{html.escape(entry["method_label"])}</h3></div>'
                f'<div class="primary-metric"><strong>{metric(row["trajectory_loss"], 9)}</strong>'
                f'<span>trajectory loss</span></div></header>'
                f'<dl class="metrics"><div><dt>Coordinate</dt><dd>{metric(row["trajectory_coordinate_loss"], 9)}</dd></div>'
                f'<div><dt>Visibility penalty</dt><dd>{metric(row["trajectory_visibility_penalty"], 9)}</dd></div>'
                f'<div><dt>Normalized ADE</dt><dd>{metric(row["trajectory_normalized_ade"], 6)}</dd></div>'
                f'<div><dt>Valid objects</dt><dd>{float(row["trajectory_valid_object_fraction"]):.0%}</dd></div></dl>'
                f'<div class="video-scroll"><video controls muted playsinline preload="metadata" data-sync-video '
                f'poster="media/{html.escape(links["poster"].name)}" src="media/{html.escape(links["video"].name)}"></video></div>'
                f'<footer><span>cached GT · F04 anchor · F08-F48 · 24 points/object</span>'
                f'<nav><a href="media/{html.escape(links["metrics"].name)}">metrics.json</a>'
                f'<a href="media/{html.escape(links["tracks"].name)}">trajectories.npz</a></nav></footer></article>'
            )
        hidden = "" if case_index == 1 else " hidden"
        case_sections.append(
            f'<section class="case" data-case-id="{html.escape(case_id, quote=True)}"{hidden}>'
            f'<div class="case-heading"><div><span>CASE {case_index:02d} / 05</span>'
            f'<h2>{html.escape(case_id)}</h2><p>{html.escape(first["prompt"])}</p></div>'
            f'<div class="case-delta"><strong>{metric(abs(losses[ENTRY_IDS[0]] - losses[ENTRY_IDS[1]]), 9)}</strong>'
            f'<span>checkpoint gap</span></div></div>{"".join(checkpoint_blocks)}</section>'
        )

    mean_0500 = float(reports[ENTRY_IDS[0]]["mean_trajectory_loss"])
    mean_1000 = float(reports[ENTRY_IDS[1]]["mean_trajectory_loss"])
    relative_improvement = (mean_0500 - mean_1000) / mean_0500 * 100.0
    style = """<style>
:root{--canvas:#101312;--surface:#171c1a;--panel:#202724;--line:#3d4a44;--ink:#f2f5ef;--muted:#9baaa2;--teal:#45c7a5;--amber:#f0b84b;--red:#e56355;--magenta:#a23b72}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--canvas);color:var(--ink);font:14px/1.5 Inter,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;letter-spacing:0}
a{color:var(--teal);text-underline-offset:3px}button,select{font:inherit;letter-spacing:0}
.topbar{position:sticky;top:0;z-index:10;border-bottom:1px solid var(--line);background:#101312f2;backdrop-filter:blur(10px)}
.topbar-inner{display:flex;align-items:center;justify-content:space-between;gap:16px;max-width:1700px;margin:auto;padding:12px 24px}
.brand{display:flex;align-items:baseline;gap:12px;min-width:0}.brand strong{font-size:15px}.brand span{color:var(--muted);font:11px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap}
.nav{display:flex;align-items:center;gap:14px;flex:none}.nav a{font-size:12px}.nav button{min-height:36px;padding:0 13px;border:1px solid var(--teal);border-radius:6px;background:#1c6556;color:#fff;cursor:pointer}.nav button:hover{background:#237765}.nav button:focus-visible,.case-picker select:focus-visible,a:focus-visible{outline:3px solid var(--amber);outline-offset:2px}
main{max-width:1700px;margin:auto;padding:26px 24px 70px}
.intro{display:grid;grid-template-columns:minmax(0,1fr) minmax(520px,.8fr);gap:36px;align-items:end;padding:10px 0 24px;border-bottom:1px solid var(--line)}
.eyebrow{display:block;color:var(--amber);font:700 11px ui-monospace,SFMono-Regular,Consolas,monospace}.intro h1{margin:6px 0 8px;font-size:30px;line-height:1.15}.intro p{max-width:760px;margin:0;color:var(--muted)}
.summary-table-wrap{overflow:auto;border:1px solid var(--line);border-radius:6px}.summary-table{width:100%;border-collapse:collapse;font:11px ui-monospace,SFMono-Regular,Consolas,monospace}.summary-table th,.summary-table td{padding:8px 9px;border-bottom:1px solid #2d3934;text-align:right;white-space:nowrap}.summary-table th:first-child,.summary-table td:first-child{text-align:left}.summary-table th{background:#222b27;color:var(--muted)}.summary-table tr:last-child td{border-bottom:0}.swatch{display:inline-block;width:8px;height:8px;margin-right:7px;border-radius:2px}
.control-band{display:flex;align-items:center;justify-content:space-between;gap:20px;padding:16px 0;border-bottom:1px solid var(--line)}.case-picker{display:flex;align-items:center;gap:10px;min-width:0}.case-picker label{color:var(--muted);font-size:12px;font-weight:700;white-space:nowrap}.case-picker select{width:min(760px,70vw);height:38px;padding:0 34px 0 10px;border:1px solid #53645d;border-radius:6px;background:var(--surface);color:var(--ink)}.comparison{color:var(--muted);font:11px ui-monospace,SFMono-Regular,Consolas,monospace;text-align:right}.comparison strong{display:block;color:var(--teal);font-size:16px}
.case{padding:25px 0}.case[hidden]{display:none}.case-heading{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:14px}.case-heading>div>span{color:var(--amber);font:700 11px ui-monospace,SFMono-Regular,Consolas,monospace}.case h2{margin:4px 0;font-size:20px;overflow-wrap:anywhere}.case-heading p{margin:0;color:var(--muted)}.case-delta{text-align:right}.case-delta strong{display:block;color:var(--red);font:16px ui-monospace,SFMono-Regular,Consolas,monospace}.case-delta span{color:var(--muted);font-size:11px}
.checkpoint{margin:12px 0 18px;padding:14px;border:1px solid var(--line);border-left:4px solid var(--magenta);border-radius:6px;background:var(--panel)}.checkpoint-best{border-left-color:var(--teal)}.checkpoint>header{display:flex;align-items:start;justify-content:space-between;gap:16px}.checkpoint-tag{color:var(--amber);font:700 11px ui-monospace,SFMono-Regular,Consolas,monospace}.checkpoint h3{margin:3px 0 0;font-size:14px}.primary-metric{text-align:right}.primary-metric strong{display:block;color:var(--ink);font:18px ui-monospace,SFMono-Regular,Consolas,monospace}.checkpoint-best .primary-metric strong{color:var(--teal)}.primary-metric span{color:var(--muted);font-size:10px}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));margin:12px 0;border:1px solid #35413c}.metrics div{padding:8px 10px;border-right:1px solid #35413c}.metrics div:last-child{border-right:0}.metrics dt{color:var(--muted);font-size:10px}.metrics dd{margin:2px 0 0;font:12px ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}
.video-scroll{width:100%;overflow:auto;background:#070908}.video-scroll video{display:block;width:100%;min-width:900px;aspect-ratio:2688/512;background:#070908;object-fit:contain}.checkpoint footer{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-top:10px;color:var(--muted);font:10px ui-monospace,SFMono-Regular,Consolas,monospace}.checkpoint nav{display:flex;gap:12px;flex:none}
@media(max-width:900px){.intro{grid-template-columns:1fr}.control-band,.case-heading,.checkpoint>header,.checkpoint footer{align-items:stretch;flex-direction:column}.comparison,.case-delta,.primary-metric{text-align:left}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.metrics div:nth-child(2){border-right:0}.metrics div:nth-child(-n+2){border-bottom:1px solid #35413c}.topbar-inner{align-items:flex-start}.brand{align-items:flex-start;flex-direction:column;gap:1px}.case-picker{align-items:stretch;flex-direction:column}.case-picker select{width:100%}}
@media(max-width:560px){main{padding:18px 12px 50px}.topbar-inner{padding:10px 12px}.brand span{white-space:normal}.nav a{display:none}.checkpoint{padding:10px}.video-scroll video{width:900px}.checkpoint nav{flex-wrap:wrap}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
</style>"""
    script = """<script>
(() => {
  const select = document.getElementById('case-select');
  const replay = document.getElementById('sync-replay');
  const cases = [...document.querySelectorAll('.case')];
  function showCase(caseId, updateHash = true) {
    const target = cases.find((item) => item.dataset.caseId === caseId) || cases[0];
    if (!target) return;
    cases.forEach((item) => {
      item.hidden = item !== target;
      if (item.hidden) item.querySelectorAll('video').forEach((video) => video.pause());
    });
    select.value = target.dataset.caseId;
    if (updateHash) history.replaceState(null, '', `#${encodeURIComponent(target.dataset.caseId)}`);
    window.scrollTo({top: 0, behavior: 'smooth'});
  }
  select.addEventListener('change', () => showCase(select.value));
  window.addEventListener('hashchange', () => showCase(decodeURIComponent(location.hash.slice(1)), false));
  replay.addEventListener('click', async () => {
    const videos = [...document.querySelectorAll('.case:not([hidden]) video')];
    replay.disabled = true;
    videos.forEach((video) => { video.pause(); video.currentTime = 0; });
    await Promise.allSettled(videos.map((video) => video.play()));
    replay.disabled = false;
  });
  showCase(decodeURIComponent(location.hash.slice(1)) || select.value, false);
})();
</script>"""
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CoTracker trajectory overlay · 5-case</title>{style}</head><body>
<header class="topbar"><div class="topbar-inner"><div class="brand"><strong>CoTracker trajectory overlay</strong><span>TRAIN-CACHE TARGET · FULL 40-STEP INFERENCE</span></div><nav class="nav"><a href="../train-validation-30cases/">30-case validation</a><a href="../">项目 Hub</a><button id="sync-replay" type="button">同步播放</button></nav></div></header>
<main><section class="intro"><div><span class="eyebrow">5 FIXED PYBULLET CASES · 2 CHECKPOINTS</span><h1>轨迹 Overlay 对比</h1><p>左：缓存 GT 轨迹；中：生成视频预测轨迹；右：白色 GT、彩色预测、红色误差。Loss 使用训练阶段相同的缓存 GT tracks、confidence 与 geometric visibility。</p></div><div class="summary-table-wrap"><table class="summary-table"><thead><tr><th>Checkpoint</th><th>Trajectory</th><th>Coordinate</th><th>Vis penalty</th><th>ADE</th></tr></thead><tbody>{''.join(aggregate_rows)}</tbody></table></div></section>
<section class="control-band"><div class="case-picker"><label for="case-select">选择 case</label><select id="case-select">{''.join(case_options)}</select></div><div class="comparison"><strong>step-1000 低 {relative_improvement:.2f}%</strong>5-case mean trajectory loss</div></section>
{''.join(case_sections)}</main>{script}</body></html>"""
    (output_root / "index.html").write_text(page, encoding="utf-8")
    print(output_root)


if __name__ == "__main__":
    main()
