#!/usr/bin/env python3
"""Build the all-checkpoint nine-case gallery with GT-relative losses."""

from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path

from run_test5_all_checkpoints_train_cases import (
    DEFAULT_CONFIG,
    discover_inventory,
)


HUB_ROOT = Path("/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub")
PAGE_NAME = "test5-all-checkpoints-train-cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        raise RuntimeError(f"Refusing to replace real directory: {link}")
    link.symlink_to(target)


def read_json(path: Path, fallback: dict) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else fallback


def loss_text(record: dict | None) -> str:
    if not record:
        return '<span class="metric pending-metric">GT loss 待计算</span>'
    mse = record.get("mse_loss")
    trajectory = record.get("trajectory_loss")
    mse_text = "—" if mse is None else f"{float(mse):.6f}"
    trajectory_text = "—" if trajectory is None else f"{float(trajectory):.6f}"
    return (
        f'<span class="metric"><b>MSE↓</b> {mse_text}</span>'
        f'<span class="metric"><b>Trajectory↓</b> {trajectory_text}</span>'
    )


def ranking_table(
    rows: list[dict],
    metric_key: str,
    title: str,
    case_labels: list[tuple[str, str]],
) -> str:
    complete = [row for row in rows if len(row[metric_key]) == len(case_labels)]
    complete.sort(key=lambda row: sum(row[metric_key].values()) / len(case_labels))
    pending = [row for row in rows if len(row[metric_key]) != len(case_labels)]
    body = []
    for rank, row in enumerate(complete, start=1):
        values = row[metric_key]
        mean = sum(values.values()) / len(values)
        cells = "".join(
            f'<td>{values[case_id]:.6f}</td>' for case_id, _ in case_labels
        )
        body.append(
            f'<tr><td class="rank">{rank}</td><td class="method-name">'
            f'{escape(row["method_label"])}</td><td>step-{int(row["step"])}</td>'
            f'<td class="mean">{mean:.6f}</td>{cells}</tr>'
        )
    for row in pending:
        values = row[metric_key]
        cells = "".join(
            f'<td>{values[case_id]:.6f}</td>' if case_id in values else '<td>—</td>'
            for case_id, _ in case_labels
        )
        body.append(
            f'<tr class="pending-row"><td class="rank">—</td>'
            f'<td class="method-name">{escape(row["method_label"])}</td>'
            f'<td>step-{int(row["step"])}</td><td>{len(values)}/9</td>{cells}</tr>'
        )
    headers = "".join(
        f'<th title="{escape(case_id)}">{escape(short)}</th>'
        for case_id, short in case_labels
    )
    return f'''<details class="ranking-block" open><summary><span>{escape(title)}</span><b>{len(complete)}/{len(rows)} weights ready</b></summary>
  <div class="table-wrap"><table><thead><tr><th>Rank</th><th>方案</th><th>权重</th><th>9-case mean↓</th>{headers}</tr></thead>
  <tbody>{''.join(body)}</tbody></table></div></details>'''


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"]).expanduser().resolve()
    inventory = read_json(
        output_root / "all_checkpoint_inventory.json",
        discover_inventory(config),
    )
    cases = json.loads(
        Path(config["cases_manifest"]).read_text(encoding="utf-8")
    )["cases"]
    metrics = read_json(output_root / "gt_losses.json", {"results": {}})
    inference_status = read_json(
        output_root / "all_checkpoint_runtime_status.json",
        {"state": "queued", "entries": {}},
    )
    metrics_status = read_json(
        output_root / "gt_losses_runtime_status.json",
        {"state": "queued"},
    )
    site = output_root / "all_checkpoints_site"
    media = site / "media"
    media.mkdir(parents=True, exist_ok=True)

    entries_by_method: dict[str, list[dict]] = defaultdict(list)
    for entry in inventory["entries"]:
        entries_by_method[entry["method_key"]].append(entry)
    method_order = [method["key"] for method in config["methods"]]
    source_counts: dict[str, int] = defaultdict(int)
    case_labels = []
    for case in cases:
        source = str(case["source"])
        source_counts[source] += 1
        case_labels.append(
            (str(case["case_id"]), f"{source[0].upper()}{source_counts[source]}")
        )
    ranking_rows = []
    for entry in inventory["entries"]:
        mse_values = {}
        trajectory_values = {}
        for case_id, _ in case_labels:
            record = metrics.get("results", {}).get(entry["entry_id"], {}).get(case_id)
            if record and record.get("mse_loss") is not None:
                mse_values[case_id] = float(record["mse_loss"])
            if record and record.get("trajectory_loss") is not None:
                trajectory_values[case_id] = float(record["trajectory_loss"])
        ranking_rows.append(
            {
                "method_label": entry["method_label"],
                "step": entry["step"],
                "mse": mse_values,
                "trajectory": trajectory_values,
            }
        )
    ranking_html = (
        ranking_table(ranking_rows, "mse", "MSE 排名 · 9-case 独立均值", case_labels)
        + ranking_table(
            ranking_rows,
            "trajectory",
            "轨迹 loss 排名 · 9-case 独立均值",
            case_labels,
        )
    )
    total_videos = 0
    total_metrics = 0
    case_sections = []
    case_options = ['<option value="all">全部 9 个 case</option>']

    for case_number, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        case_options.append(
            f'<option value="{escape(case_id)}">CASE {case_number:02d} · '
            f'{escape(case["source"])}</option>'
        )
        case_media = media / case_id
        replace_symlink(case_media / "context.mp4", Path(case["context_video"]).resolve())
        replace_symlink(case_media / "gt.mp4", Path(case["gt_video"]).resolve())
        method_sections = []
        for method_key in method_order:
            method_entries = entries_by_method[method_key]
            panels = []
            ready_for_method = 0
            for entry in method_entries:
                entry_id = str(entry["entry_id"])
                video = Path(entry["result_root"]) / f"{case_id}.mp4"
                metadata = Path(entry["result_root"]) / f"{case_id}.json"
                metric_record = metrics.get("results", {}).get(entry_id, {}).get(case_id)
                if video.is_file() and metadata.is_file() and video.stat().st_size > 0:
                    video_link = case_media / f"{method_key}_step{int(entry['step']):06d}.mp4"
                    replace_symlink(video_link, video.resolve())
                    body = (
                        f'<video controls muted playsinline preload="none" '
                        f'src="media/{escape(case_id)}/{escape(video_link.name)}"></video>'
                    )
                    ready_for_method += 1
                    total_videos += 1
                else:
                    state = inference_status.get("entries", {}).get(entry_id, {}).get(
                        "state", "pending"
                    )
                    message = "正在推理" if state == "running" else "等待推理"
                    body = f'<div class="pending-video">{message}</div>'
                if metric_record and "trajectory_loss" in metric_record:
                    total_metrics += 1
                panels.append(
                    f'''<figure style="--method-color:{escape(entry['color'])}">
  <div class="video-shell">{body}</div>
  <figcaption><strong>step-{int(entry['step'])}</strong><div class="metrics">{loss_text(metric_record)}</div></figcaption>
</figure>'''
                )
            method_sections.append(
                f'''<details class="method" style="--method-color:{escape(method_entries[0]['color'])}">
  <summary><span>{escape(method_entries[0]['method_label'])}</span><b>{ready_for_method}/{len(method_entries)} weights</b></summary>
  <div class="step-grid">{''.join(panels)}</div>
</details>'''
            )
        case_sections.append(
            f'''<section class="case" data-case="{escape(case_id)}" id="{escape(case_id)}">
  <header><span>CASE {case_number:02d} · {escape(case['source'])}</span><h2>{escape(case_id)}</h2><p>{escape(case['prompt'])}</p></header>
  <div class="reference-grid">
    <figure class="reference"><div class="video-shell"><video controls muted playsinline preload="metadata" src="media/{escape(case_id)}/context.mp4"></video></div><figcaption>条件输入 · 8 帧</figcaption></figure>
    <figure class="reference gt"><div class="video-shell"><video controls muted playsinline preload="metadata" src="media/{escape(case_id)}/gt.mp4"></video></div><figcaption>训练 GT · 49 帧</figcaption></figure>
  </div>
  <div class="methods">{''.join(method_sections)}</div>
</section>'''
        )

    expected = inventory["num_checkpoints"] * len(cases)
    refresh = (
        '<meta http-equiv="refresh" content="60">'
        if total_videos < expected or total_metrics < expected
        else ""
    )
    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh}
<title>全部 checkpoint · 9-case GT loss</title>
<style>
:root{{--ink:#172b3a;--paper:#edf2f5;--panel:#fbfcfd;--line:#c8d4dc;--muted:#607486;--blue:#315b7d;--amber:#c7852c}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth;background:var(--paper);color:var(--ink);font-family:"Avenir Next","Segoe UI",sans-serif}}body{{margin:0}}.mast{{padding:27px max(24px,4vw) 22px;background:#dce7ed;border-bottom:1px solid #aebfc9}}.eyebrow{{font:700 11px ui-monospace,monospace;letter-spacing:.15em;color:var(--blue)}}h1{{margin:8px 0 10px;font:600 clamp(28px,4vw,48px)/1.06 Georgia,serif}}.mast p{{max-width:1200px;margin:6px 0;color:var(--muted);line-height:1.55}}.counts{{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}}.counts b{{padding:8px 11px;background:var(--ink);color:white;font:700 12px ui-monospace,monospace}}.toolbar{{position:sticky;top:0;z-index:9;display:flex;gap:9px;align-items:center;padding:10px max(24px,4vw);background:#edf2f5f2;backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}}button,select{{padding:9px 12px;border:1px solid #9eafba;background:white;color:var(--ink);font:700 12px inherit}}button{{background:var(--amber);border-color:#9b6722;color:white;cursor:pointer}}main{{padding:28px max(24px,4vw) 90px}}.case{{scroll-margin-top:68px}}.case+.case{{margin-top:58px;padding-top:44px;border-top:2px solid #aebfc9}}.case>header span{{font:800 10px ui-monospace,monospace;letter-spacing:.12em;color:var(--blue)}}h2{{margin:7px 0 4px;font:600 23px Georgia,serif}}.case>header p{{margin:0 0 15px;color:var(--muted)}}.reference-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:11px;max-width:1050px;margin-bottom:16px}}figure{{margin:0;background:var(--panel);border:1px solid var(--line);border-top:5px solid var(--method-color,#315b7d)}}.reference{{--method-color:#315b7d}}.reference.gt{{--method-color:#c7852c}}.video-shell{{aspect-ratio:896/512;display:grid;place-items:center;overflow:hidden;background:#1a2c39}}video{{width:100%;height:100%;object-fit:contain;background:#111d25}}figcaption{{min-height:65px;padding:9px 10px;font:700 11px/1.4 ui-monospace,monospace;color:#40576a}}.method{{margin:7px 0;border:1px solid var(--line);border-left:6px solid var(--method-color);background:#f7fafb}}summary{{display:flex;justify-content:space-between;gap:20px;padding:11px 13px;cursor:pointer;font-size:12px}}summary b{{font:700 11px ui-monospace,monospace;color:var(--muted)}}.step-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;padding:0 11px 12px}}.pending-video{{width:100%;height:100%;display:grid;place-items:center;color:#dce8ef;font:700 11px ui-monospace,monospace;background:repeating-linear-gradient(135deg,#253d50,#253d50 9px,#2d485d 9px,#2d485d 18px)}}.metrics{{display:flex;gap:6px;flex-wrap:wrap;margin-top:7px}}.metric{{padding:3px 5px;background:#e2eaf0;color:#344c5e;font-size:10px}}.pending-metric{{color:#728594}}.rankings{{padding:24px max(24px,4vw) 0}}.ranking-block{{margin:8px 0;border:1px solid var(--line);background:#f7fafb}}.ranking-block summary{{background:#dfe8ed}}.table-wrap{{overflow:auto;max-height:620px}}table{{width:100%;min-width:1500px;border-collapse:collapse;font:11px ui-monospace,monospace;font-variant-numeric:tabular-nums}}th,td{{padding:7px 8px;border:1px solid #d6e0e6;text-align:right;white-space:nowrap}}thead th{{position:sticky;top:0;z-index:2;background:#213c4d;color:white}}th:nth-child(2),td.method-name{{text-align:left}}td.rank,td.mean{{font-weight:800;color:#174e69}}tbody tr:nth-child(even){{background:#edf3f6}}.pending-row{{color:#8596a1}}[hidden]{{display:none!important}}@media(max-width:1100px){{.step-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:650px){{.reference-grid,.step-grid{{grid-template-columns:1fr}}.toolbar{{align-items:stretch;flex-wrap:wrap}}summary{{align-items:start;flex-direction:column;gap:4px}}}}
</style></head><body>
<header class="mast"><div class="eyebrow">test5 methods / all available checkpoints / GT-relative losses</div><h1>全部权重 × 训练集 9-case</h1><p>共 {inventory['num_methods']} 个方法、{inventory['num_checkpoints']} 个 checkpoint。MSE 与轨迹 loss 均只统计生成段第 8–48 帧；轨迹使用 CoTracker3 20×20 网格，在第 7 帧初始化并对共同可见点计算归一化轨迹 RMSE，数值越低越好。</p><div class="counts"><b>视频 {total_videos}/{expected}</b><b>GT loss {total_metrics}/{expected}</b><b>推理 {escape(str(inference_status.get('state','queued')))}</b><b>指标 {escape(str(metrics_status.get('state','queued')))}</b></div></header>
<nav class="toolbar"><select id="caseFilter">{''.join(case_options)}</select><button id="openAll" type="button">展开当前 case</button><button id="replay" type="button">当前页全部重播</button><button id="pause" type="button">全部暂停</button></nav>
<section class="rankings">{ranking_html}</section>
<main>{''.join(case_sections)}</main>
<script>
const cases=[...document.querySelectorAll('.case')];
const visibleVideos=()=>cases.filter(c=>!c.hidden).flatMap(c=>[...c.querySelectorAll('video')]);
document.getElementById('caseFilter').addEventListener('change',e=>cases.forEach(c=>c.hidden=e.target.value!=='all'&&c.dataset.case!==e.target.value));
document.getElementById('openAll').addEventListener('click',()=>cases.filter(c=>!c.hidden).forEach(c=>c.querySelectorAll('details').forEach(d=>d.open=true)));
document.getElementById('replay').addEventListener('click',()=>visibleVideos().forEach(v=>{{v.currentTime=0;v.play().catch(()=>{{}})}}));
document.getElementById('pause').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>v.pause()));
</script></body></html>'''
    (site / "index.html").write_text(html, encoding="utf-8")
    replace_symlink(HUB_ROOT / PAGE_NAME, site)
    print(site / "index.html")


if __name__ == "__main__":
    main()
