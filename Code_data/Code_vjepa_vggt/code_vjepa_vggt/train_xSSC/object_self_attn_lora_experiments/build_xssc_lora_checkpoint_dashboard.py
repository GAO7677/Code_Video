#!/usr/bin/env python3
"""Build checkpoint video galleries and per-metric training-step curves."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def link_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(source.resolve())


def link_directory(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.is_file():
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(f"Refusing to replace directory: {destination}")
    destination.symlink_to(source.resolve(), target_is_directory=True)


def render_gt_clip(ffmpeg: str, source: Path, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            "fps=30",
            "-frames:v",
            "49",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ],
        check=True,
    )


def read_inputs(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        input_path = Path(line.strip()).resolve()
        if input_path in seen:
            continue
        seen.add(input_path)
        payload = load_json(input_path)
        cases.append(
            {
                "stem": input_path.stem,
                "input_json": str(input_path),
                "prompt": payload["input_caption"],
                "source_video": payload["source_video"],
                "input_video": payload["input_video"],
            }
        )
    return cases


def load_manifests(watch_root: Path) -> list[dict[str, Any]]:
    manifests = [
        load_json(path)
        for path in sorted((watch_root / "state" / "checkpoints").glob("*/step-*.json"))
        if path.is_file()
    ]
    return sorted(manifests, key=lambda row: (row["method_index"], row["step"]))


def metric_done_count(watch_root: Path, method_key: str, step: int) -> int:
    marker_root = (
        watch_root / "state" / "metrics" / method_key / f"step-{step:06d}"
    )
    return len(list(marker_root.glob("*.json"))) if marker_root.is_dir() else 0


def build_video_media(
    config: dict[str, Any],
    manifests: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    watch_root = Path(config["paths"]["watch_root"]).resolve()
    videos_root = watch_root / "site" / "videos"
    media_root = videos_root / "media"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        ffmpeg = str(Path(config["paths"]["python"]).with_name("ffmpeg"))
    for case in cases:
        source_root = media_root / "_source" / case["stem"]
        gt_path = source_root / "gt_49f_30fps.mp4"
        render_gt_clip(ffmpeg, Path(case["source_video"]), gt_path)
        context_path = source_root / "context_8f.mp4"
        link_file(Path(case["input_video"]), context_path)
        case["gt"] = gt_path.relative_to(videos_root).as_posix()
        case["context"] = context_path.relative_to(videos_root).as_posix()
    records: list[dict[str, Any]] = []
    for manifest in manifests:
        method_key = manifest["method_key"]
        step = int(manifest["step"])
        result_root = Path(manifest["result_root"])
        record = {
            "method_key": method_key,
            "method_label": manifest["method_label"],
            "step": step,
            "checkpoint_dir": manifest["checkpoint_dir"],
            "origin": manifest["origin"],
            "videos": {},
        }
        for case in cases:
            source = result_root / f"{case['stem']}.mp4"
            destination = (
                media_root
                / method_key
                / f"step-{step:06d}"
                / f"{case['stem']}.mp4"
            )
            link_file(source, destination)
            record["videos"][case["stem"]] = destination.relative_to(
                videos_root
            ).as_posix()
        records.append(record)
    return records


def build_videos_page(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> str:
    methods = [
        {
            "key": method["key"],
            "label": method["label"],
            "color": method["color"],
        }
        for method in config["methods"]
        if any(record["method_key"] == method["key"] for record in records)
    ]
    data = json.dumps(
        {"methods": methods, "records": records, "cases": cases},
        ensure_ascii=False,
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>xSSC LoRA checkpoint 视频</title>
  <style>
    :root {{
      --bg:#f3f5f6; --surface:#fff; --ink:#172126; --muted:#657278;
      --line:#d6dde0; --accent:#006d77; --warm:#9b3a31;
    }}
    *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);
      font-family:Inter,"Noto Sans SC",Arial,sans-serif}}
    .toolbar{{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:9px;
      min-height:58px;padding:9px 16px;background:rgba(255,255,255,.97);
      border-bottom:1px solid var(--line)}}
    .title{{margin-right:auto;font-size:16px;font-weight:800}}
    select,button{{height:38px;border:1px solid var(--line);border-radius:6px;
      background:var(--surface);color:var(--ink);font:inherit}}
    select{{max-width:360px;padding:0 9px}} button{{width:38px;padding:0;cursor:pointer}}
    main{{max-width:1500px;margin:auto;padding:18px}}
    .case-head{{margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid var(--line)}}
    h1{{margin:0 0 5px;font-size:19px;overflow-wrap:anywhere}}
    .prompt{{margin:0;max-width:1150px;color:var(--muted);line-height:1.5}}
    .videos{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
    .cell{{min-width:0;padding:9px;background:var(--surface);border:1px solid var(--line);
      border-radius:6px}}
    .label{{min-height:24px;padding:1px 2px 7px;color:var(--warm);
      font-size:13px;font-weight:800}}
    video{{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#101416}}
    .checkpoint{{margin-top:12px;color:var(--muted);font-size:12px;overflow-wrap:anywhere}}
    a{{color:var(--accent);font-weight:750;text-decoration:none}}
    @media(max-width:900px){{.toolbar{{flex-wrap:wrap}}.title{{width:100%}}
      .videos{{grid-template-columns:1fr}}select{{max-width:calc(100vw - 36px)}}}}
  </style>
</head>
<body>
  <div class="toolbar">
    <a href="../">返回监控页</a>
    <div class="title">Checkpoint · test_5 视频</div>
    <select id="method" aria-label="方法"></select>
    <select id="step" aria-label="训练 step"></select>
    <select id="case" aria-label="案例"></select>
    <button id="play" title="同步播放" aria-label="播放">▶</button>
    <button id="pause" title="同步暂停" aria-label="暂停">Ⅱ</button>
    <button id="replay" title="从头播放" aria-label="重新播放">↺</button>
  </div>
  <main>
    <div class="case-head"><h1 id="case-title"></h1><p class="prompt" id="prompt"></p></div>
    <div class="videos">
      <div class="cell"><div class="label">GT · 49 frames @ 30 FPS</div>
        <video id="gt" preload="metadata" playsinline muted></video></div>
      <div class="cell"><div class="label">Input context · 8 frames</div>
        <video id="context" preload="metadata" playsinline muted></video></div>
      <div class="cell"><div class="label" id="generated-label"></div>
        <video id="generated" preload="metadata" playsinline muted></video></div>
    </div>
    <div class="checkpoint" id="checkpoint"></div>
  </main>
  <script>
    const D={data};
    const method=document.getElementById("method"),step=document.getElementById("step");
    const caseSelect=document.getElementById("case");
    const videoIds=["gt","context","generated"];
    D.methods.forEach(m=>method.add(new Option(m.label,m.key)));
    D.cases.forEach((c,i)=>caseSelect.add(new Option(`${{String(i+1).padStart(2,"0")}} · ${{c.stem}}`,c.stem)));
    function records(){{return D.records.filter(r=>r.method_key===method.value).sort((a,b)=>a.step-b.step)}}
    function fillSteps(){{
      const prior=step.value;step.innerHTML="";
      records().forEach(r=>step.add(new Option(`step ${{r.step}}`,String(r.step))));
      if([...step.options].some(o=>o.value===prior))step.value=prior;
      else if(step.options.length)step.selectedIndex=step.options.length-1;
    }}
    function render(){{
      const c=D.cases.find(x=>x.stem===caseSelect.value);
      const r=records().find(x=>String(x.step)===step.value);
      if(!c||!r)return;
      videoIds.forEach(id=>document.getElementById(id).pause());
      document.getElementById("case-title").textContent=c.stem;
      document.getElementById("prompt").textContent=c.prompt;
      document.getElementById("gt").src=c.gt;
      document.getElementById("context").src=c.context;
      document.getElementById("generated").src=r.videos[c.stem];
      document.getElementById("generated-label").textContent=`${{r.method_label}} · step ${{r.step}}`;
      document.getElementById("checkpoint").textContent=r.checkpoint_dir;
    }}
    method.addEventListener("change",()=>{{fillSteps();render()}});
    step.addEventListener("change",render);caseSelect.addEventListener("change",render);
    document.getElementById("play").onclick=()=>videoIds.forEach(id=>document.getElementById(id).play().catch(()=>{{}}));
    document.getElementById("pause").onclick=()=>videoIds.forEach(id=>document.getElementById(id).pause());
    document.getElementById("replay").onclick=()=>videoIds.forEach(id=>{{const v=document.getElementById(id);v.currentTime=0;v.play().catch(()=>{{}})}});
    fillSteps();render();
  </script>
</body>
</html>
"""


def finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def build_metric_plots(
    config: dict[str, Any],
    manifests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    watch_root = Path(config["paths"]["watch_root"]).resolve()
    plot_root = watch_root / "site" / "metrics" / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    summary_path = watch_root / "metric_summary.csv"
    frame = pd.read_csv(summary_path) if summary_path.is_file() else pd.DataFrame()
    manifest_by_root = {
        str(Path(manifest["result_root"]).resolve()): manifest for manifest in manifests
    }
    method_by_key = {method["key"]: method for method in config["methods"]}
    expected = int(config["runtime"]["expected_cases"])
    points: list[dict[str, Any]] = []
    plots: list[dict[str, Any]] = []
    for metric in config["plot_metrics"]:
        field = metric["field"]
        count_field = metric["count"]
        figure, axis = plt.subplots(figsize=(6.4, 3.6), dpi=140)
        has_data = False
        for method in config["methods"]:
            values: list[tuple[int, float]] = []
            if not frame.empty and field in frame.columns and count_field in frame.columns:
                for _, row in frame.iterrows():
                    result_root = str(Path(str(row["result_root"])).resolve())
                    manifest = manifest_by_root.get(result_root)
                    if not manifest or manifest["method_key"] != method["key"]:
                        continue
                    count = finite_number(row[count_field])
                    value = finite_number(row[field])
                    if count is None or value is None or count < expected:
                        continue
                    values.append((int(manifest["step"]), value))
            if not values:
                continue
            values.sort()
            has_data = True
            axis.plot(
                [value[0] for value in values],
                [value[1] for value in values],
                marker="o",
                linewidth=2,
                markersize=4,
                color=method["color"],
                label=method["label"],
            )
            for step, value in values:
                points.append(
                    {
                        "method_key": method["key"],
                        "method_label": method["label"],
                        "step": step,
                        "metric": field,
                        "metric_label": metric["label"],
                        "value": value,
                        "count": expected,
                    }
                )
        axis.set_title(metric["label"], fontsize=11, fontweight="bold")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Score")
        axis.grid(True, color="#dfe4e6", linewidth=0.8)
        if has_data:
            axis.legend(frameon=False, fontsize=8)
        else:
            axis.text(
                0.5,
                0.5,
                "Pending complete metric results",
                ha="center",
                va="center",
                color="#718087",
                transform=axis.transAxes,
            )
        figure.tight_layout()
        filename = f"{field}.png"
        figure.savefig(plot_root / filename, bbox_inches="tight")
        plt.close(figure)
        plots.append(
            {
                "field": field,
                "label": metric["label"],
                "image": f"plots/{filename}",
                "has_data": has_data,
            }
        )
    point_path = watch_root / "site" / "metrics" / "metric_points.csv"
    point_path.parent.mkdir(parents=True, exist_ok=True)
    with point_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "method_key",
            "method_label",
            "step",
            "metric",
            "metric_label",
            "value",
            "count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)
    return plots


def build_metrics_page(plots: list[dict[str, Any]]) -> str:
    cards = "".join(
        f"""<article><h2>{escape(plot['label'])}</h2>
        <img src="{escape(plot['image'])}" alt="{escape(plot['label'])}"></article>"""
        for plot in plots
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>xSSC LoRA checkpoint 指标曲线</title>
  <style>
    :root{{--bg:#f3f5f6;--surface:#fff;--ink:#172126;--muted:#657278;--line:#d6dde0;--accent:#006d77}}
    *{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
      font-family:Inter,"Noto Sans SC",Arial,sans-serif}}
    header{{position:sticky;top:0;z-index:4;display:flex;align-items:center;gap:16px;
      padding:14px 20px;background:rgba(255,255,255,.97);border-bottom:1px solid var(--line)}}
    h1{{margin:0;font-size:19px}}header span{{color:var(--muted)}}a{{margin-left:auto;color:var(--accent);
      font-weight:800;text-decoration:none}}main{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
      gap:12px;max-width:1460px;margin:auto;padding:16px}}
    article{{min-width:0;padding:10px;background:var(--surface);border:1px solid var(--line);
      border-radius:6px}}h2{{margin:0 0 6px;font-size:13px}}img{{display:block;width:100%;height:auto}}
    @media(max-width:800px){{main{{grid-template-columns:1fr;padding:10px}}header span{{display:none}}}}
  </style>
</head>
<body>
  <header><h1>Checkpoint 指标曲线</h1>
    <span>横轴：训练 step · 颜色：训练方法 · 仅绘制完整 20-case 指标</span>
    <a href="../">返回监控页</a></header>
  <main>{cards}</main>
</body>
</html>
"""


def build_status(
    config: dict[str, Any],
    manifests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    watch_root = Path(config["paths"]["watch_root"]).resolve()
    discovery_path = watch_root / "state" / "discovery.json"
    discovery = load_json(discovery_path) if discovery_path.is_file() else {}
    discovered = discovery.get("checkpoints", [])
    total_metrics = len(config["metrics"]["cpu"]) + len(config["metrics"]["gpu"])
    rows: list[dict[str, Any]] = []
    for method in config["methods"]:
        method_discovered = [
            row for row in discovered if row["method_key"] == method["key"]
        ]
        method_manifests = [
            row for row in manifests if row["method_key"] == method["key"]
        ]
        metric_done = sum(
            metric_done_count(watch_root, method["key"], int(row["step"]))
            for row in method_manifests
        )
        rows.append(
            {
                "method_key": method["key"],
                "method_label": method["label"],
                "color": method["color"],
                "discovered": len(method_discovered),
                "inferred": len(method_manifests),
                "latest_step": max(
                    [int(row["step"]) for row in method_manifests],
                    default=None,
                ),
                "metric_done": metric_done,
                "metric_total": len(method_discovered) * total_metrics,
            }
        )
    return rows


def build_watch_index(status: list[dict[str, Any]], pending: bool) -> str:
    rows = "".join(
        f"""<tr><td><span class="swatch" style="background:{escape(row['color'])}"></span>
        {escape(row['method_label'])}</td><td>{row['discovered']}</td><td>{row['inferred']}</td>
        <td>{escape(row['latest_step'] if row['latest_step'] is not None else '—')}</td>
        <td>{row['metric_done']}/{row['metric_total']}</td></tr>"""
        for row in status
    )
    state = "推理排队中" if pending else "持续监听"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>xSSC LoRA checkpoint watcher</title>
  <style>
    :root{{--bg:#f3f5f6;--surface:#fff;--ink:#172126;--muted:#657278;--line:#d6dde0;
      --accent:#006d77;--warm:#9b3a31}}
    *{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
      font-family:Inter,"Noto Sans SC",Arial,sans-serif}}
    header{{padding:20px 24px;background:var(--surface);border-bottom:1px solid var(--line)}}
    h1{{margin:0 0 6px;font-size:23px}}header p{{margin:0;color:var(--muted)}}
    main{{max-width:1120px;margin:auto;padding:20px}}.links{{display:grid;
      grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-bottom:18px}}
    .link{{display:block;padding:15px;background:var(--surface);border:1px solid var(--line);
      border-radius:6px;color:var(--ink);text-decoration:none}}.link strong{{display:block;
      margin-bottom:4px;color:var(--accent)}}.link span{{color:var(--muted);font-size:13px}}
    .panel{{padding:15px;background:var(--surface);border:1px solid var(--line);border-radius:6px}}
    .panel-head{{display:flex;align-items:center;margin-bottom:10px}}h2{{margin:0;font-size:16px}}
    .state{{margin-left:auto;color:var(--accent);font-weight:800}}table{{width:100%;
      border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{padding:9px 8px;
      text-align:left;border-top:1px solid var(--line);font-size:13px}}th{{color:var(--muted)}}
    .swatch{{display:inline-block;width:10px;height:10px;margin-right:7px;border-radius:2px}}
    .home{{display:inline-block;margin-top:15px;color:var(--accent);font-weight:800;text-decoration:none}}
    @media(max-width:700px){{main{{padding:12px}}.links{{grid-template-columns:1fr}}}}
  </style>
</head>
<body>
  <header><h1>xSSC LoRA checkpoint watcher</h1>
    <p>test_5 · 8 context · 49 frames · 512×896 · 8 denoising steps</p></header>
  <main>
    <div class="links">
      <a class="link" href="videos/"><strong>Checkpoint 视频</strong>
        <span>按方法、训练 step 和 case 查看生成结果</span></a>
      <a class="link" href="metrics/"><strong>指标曲线</strong>
        <span>每个指标独立子图，比较不同训练方法</span></a>
    </div>
    <section class="panel"><div class="panel-head"><h2>自动任务状态</h2>
      <span class="state">{state}</span></div>
      <table><thead><tr><th>方法</th><th>发现权重</th><th>完成推理</th>
      <th>最新 step</th><th>指标任务</th></tr></thead><tbody>{rows}</tbody></table>
    </section>
    <a class="home" href="../">返回训练可视化总览</a>
  </main>
</body>
</html>
"""


def build_master_hub(config: dict[str, Any], status: list[dict[str, Any]]) -> None:
    hub_root = Path(config["paths"]["master_hub_root"]).resolve()
    watch_root = Path(config["paths"]["watch_root"]).resolve()
    baseline_gallery = Path(config["paths"]["baseline_gallery_root"]).resolve()
    hub_root.mkdir(parents=True, exist_ok=True)
    link_directory(baseline_gallery, hub_root / "gallery")
    link_directory(watch_root / "site", hub_root / "checkpoint-watch")
    total_inferred = sum(row["inferred"] for row in status)
    total_discovered = sum(row["discovered"] for row in status)
    total_metrics = sum(row["metric_done"] for row in status)
    expected_metrics = sum(row["metric_total"] for row in status)
    physiciq_entry = ""
    if config.get("physiciq", {}).get("enabled"):
        leaf_path = Path(config["physiciq"]["leaf_folders"]).resolve()
        plot_root = leaf_path.parent / "_metric_plots" / leaf_path.stem
        if (plot_root / "index.html").is_file():
            link_directory(plot_root, hub_root / "physiciq-step1500-metrics")
            action = '<a href="physiciq-step1500-metrics/">进入 PhysicIQ 指标图</a>'
            phys_status = "指标图已更新"
        else:
            action = ""
            phys_status = "等待 step 1500"
        physiciq_entry = f"""
    <section class="entry"><div><h2>Step 1500 · PhysicIQ 67-case</h2>
      <div class="meta">Full-SA、S-head59、T-head70 · 40 denoising steps · 完整 14 项指标</div>
      {action}</div><div class="status">{phys_status}<strong>67 cases / method</strong></div>
    </section>"""
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>xSSC LoRA 训练可视化总览</title>
  <style>
    :root{{--bg:#f3f5f6;--surface:#fff;--ink:#172126;--muted:#657278;--line:#d6dde0;
      --accent:#006d77;--warm:#9b3a31}}
    *{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
      font-family:Inter,"Noto Sans SC",Arial,sans-serif}}
    header{{padding:21px 25px;background:var(--surface);border-bottom:1px solid var(--line)}}
    h1{{margin:0 0 6px;font-size:24px}}header p{{margin:0;color:var(--muted)}}
    main{{max-width:1120px;margin:auto;padding:22px}}.section-title{{margin:0 0 9px;
      color:var(--muted);font-size:13px}}.entries{{display:grid;grid-template-columns:1fr;gap:12px}}
    .entry{{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:16px;padding:17px;
      background:var(--surface);border:1px solid var(--line);border-radius:6px}}
    h2{{margin:0 0 5px;font-size:18px}}.meta{{color:var(--muted);font-size:13px}}
    a{{display:inline-block;margin-top:13px;padding:8px 12px;color:#fff;background:var(--accent);
      border-radius:5px;text-decoration:none;font-weight:800}}.status{{align-self:start;min-width:150px;
      padding:9px 11px;border-left:3px solid var(--accent);background:#edf6f5;
      color:var(--accent);font-size:13px;font-weight:750}}.status strong{{display:block;
      margin-top:3px;color:var(--ink);font-size:16px}}@media(max-width:650px){{
      main{{padding:12px}}.entry{{grid-template-columns:1fr}}.status{{justify-self:stretch}}}}
  </style>
</head>
<body>
  <header><h1>xSSC LoRA 训练可视化总览</h1>
    <p>Object-only、Full-SA、S-head59 与 T-head70</p></header>
  <main><div class="section-title">训练与推理</div><div class="entries">
    <section class="entry"><div><h2>Checkpoint 自动评测</h2>
      <div class="meta">自动发现权重、生成 test_5、计算完整指标并绘制训练 step 曲线</div>
      <a href="checkpoint-watch/">进入自动评测</a></div>
      <div class="status">持续监听<strong>推理 {total_inferred}/{total_discovered} · 指标 {total_metrics}/{expected_metrics}</strong></div>
    </section>
    <section class="entry"><div><h2>初始四方案对比</h2>
      <div class="meta">Object-only step 529 · Full-SA/S-head59/T-head70 step 1000 · 20 cases</div>
      <a href="gallery/">进入固定对比</a></div>
      <div class="status">已完成<strong>20/20 cases</strong></div>
    </section>
    {physiciq_entry}
  </div></main>
</body>
</html>
"""
    (hub_root / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_json(args.config.resolve())
    watch_root = Path(config["paths"]["watch_root"]).resolve()
    site_root = watch_root / "site"
    site_root.mkdir(parents=True, exist_ok=True)
    manifests = load_manifests(watch_root)
    cases = read_inputs(Path(config["paths"]["input_list"]))
    records = build_video_media(config, manifests, cases)
    videos_root = site_root / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)
    (videos_root / "index.html").write_text(
        build_videos_page(config, records, cases),
        encoding="utf-8",
    )
    plots = build_metric_plots(config, manifests)
    metrics_root = site_root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    (metrics_root / "index.html").write_text(
        build_metrics_page(plots),
        encoding="utf-8",
    )
    status = build_status(config, manifests)
    pending = (watch_root / "state" / "inference.pending").is_file()
    (site_root / "index.html").write_text(
        build_watch_index(status, pending),
        encoding="utf-8",
    )
    manifest = {
        "num_cases": len(cases),
        "num_inference_results": len(manifests),
        "num_metric_plots": len(plots),
        "status": status,
        "records": records,
        "plots": plots,
    }
    (site_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    build_master_hub(config, status)
    print(site_root / "index.html")


if __name__ == "__main__":
    main()
