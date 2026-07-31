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


def physiciq_metric_names(config: dict[str, Any]) -> list[str]:
    return list(config["metrics"]["cpu"]) + list(config["metrics"]["gpu"])


def count_paired_result_cases(result_root: Path) -> int:
    if not result_root.is_dir():
        return 0
    count = 0
    for result_json in result_root.glob("*.json"):
        if result_json.name.startswith("eval_summary_"):
            continue
        if result_json.name in {"summary.json", "batch_manifest.json", "eval_summary.json"}:
            continue
        if result_json.with_suffix(".mp4").is_file():
            count += 1
    return count


def metric_marker_ok(path: Path, *, partial: bool) -> bool:
    if not path.is_file():
        return False
    if not partial:
        return True
    try:
        payload = load_json(path)
    except Exception:
        return False
    return bool(payload.get("ok"))


def metric_marker_count(root: Path, metrics: list[str], *, partial: bool) -> int:
    return sum(
        metric_marker_ok(root / f"{metric}.json", partial=partial)
        for metric in metrics
    )


def active_summary_text(summary_root: Path, marker_root: Path, metrics: list[str]) -> str:
    active: list[tuple[float, str]] = []
    for metric in metrics:
        summary_path = summary_root / f"{metric}.json"
        marker_path = marker_root / f"{metric}.json"
        if not summary_path.is_file() or metric_marker_ok(marker_path, partial=True):
            continue
        try:
            payload = load_json(summary_path)
        except Exception:
            continue
        status = payload.get("metric_status")
        if not isinstance(status, dict):
            continue
        completed = status.get("completed", 0)
        total = status.get("num_cases", 0)
        active.append((summary_path.stat().st_mtime, f"{metric} {completed}/{total}"))
    if not active:
        return ""
    active.sort(reverse=True)
    return active[0][1]


def build_physiciq_status(config: dict[str, Any]) -> dict[str, Any] | None:
    phys = config.get("physiciq", {})
    if not phys.get("enabled"):
        return None
    watch_root = Path(config["paths"]["watch_root"]).resolve()
    output_root = Path(phys["output_root"]).resolve()
    expected_cases = int(phys["expected_cases"])
    metrics = physiciq_metric_names(config)
    methods = {method["key"]: method for method in config["methods"]}
    pending_path = watch_root / "state" / "physiciq" / "inference.pending"
    pending = load_json(pending_path) if pending_path.is_file() else None
    rows: list[dict[str, Any]] = []
    for step in [int(step) for step in phys["trigger_steps"]]:
        for method_key in phys["method_keys"]:
            method = methods.get(method_key, {"label": method_key, "color": "#657278"})
            name = phys["method_name_template"].format(method_key=method_key, step=step)
            result_root = output_root / name
            generated = count_paired_result_cases(result_root)
            manifest_path = (
                watch_root
                / "state"
                / "physiciq"
                / "inference"
                / method_key
                / f"step-{step:06d}.json"
            )
            formal_marker_root = (
                watch_root
                / "state"
                / "physiciq"
                / "metrics"
                / method_key
                / f"step-{step:06d}"
            )
            partial_marker_root = (
                watch_root
                / "state"
                / "physiciq_partial_metrics"
                / method_key
                / f"step-{step:06d}"
            )
            partial_summary_root = (
                watch_root
                / "physiciq_partial_metric_task_summaries"
                / method_key
                / f"step-{step:06d}"
            )
            partial_allowlist = (
                watch_root
                / "state"
                / "physiciq_partial_metrics"
                / "allowlists"
                / method_key
                / f"step-{step:06d}.txt"
            )
            partial_cases = 0
            if partial_allowlist.is_file():
                partial_cases = len(
                    [
                        line
                        for line in partial_allowlist.read_text(encoding="utf-8").splitlines()
                        if line.strip() and not line.lstrip().startswith("#")
                    ]
                )
            rows.append(
                {
                    "method_key": method_key,
                    "method_label": method["label"],
                    "color": method["color"],
                    "step": step,
                    "result_root": str(result_root),
                    "generated": generated,
                    "expected_cases": expected_cases,
                    "manifest_done": manifest_path.is_file(),
                    "formal_metrics_done": metric_marker_count(
                        formal_marker_root,
                        metrics,
                        partial=False,
                    ),
                    "partial_metrics_done": metric_marker_count(
                        partial_marker_root,
                        metrics,
                        partial=True,
                    ),
                    "partial_cases": partial_cases,
                    "partial_active": active_summary_text(
                        partial_summary_root,
                        partial_marker_root,
                        metrics,
                    ),
                    "metric_total": len(metrics),
                }
            )
    return {
        "expected_cases": expected_cases,
        "metric_total": len(metrics),
        "rows": rows,
        "pending": pending,
        "generated_total": sum(row["generated"] for row in rows),
        "generated_expected": len(rows) * expected_cases,
        "formal_metric_total": sum(row["formal_metrics_done"] for row in rows),
        "formal_metric_expected": len(rows) * len(metrics),
        "partial_metric_total": sum(row["partial_metrics_done"] for row in rows),
        "partial_metric_expected": sum(
            len(metrics) for row in rows if row["partial_cases"] > 0
        ),
    }


def progress_cell(done: int, total: int, *, label: str | None = None) -> str:
    ratio = 0 if total <= 0 else max(0, min(1, done / total))
    width = ratio * 100
    text = label if label is not None else f"{done}/{total}"
    return (
        f"""<div class="progress"><span style="width:{width:.1f}%"></span></div>"""
        f"""<div class="progtext">{escape(text)}</div>"""
    )


def build_physiciq_section(phys_status: dict[str, Any] | None) -> str:
    if phys_status is None:
        return ""
    rows = []
    for row in phys_status["rows"]:
        partial_label = (
            f"{row['partial_metrics_done']}/{row['metric_total']} · {row['partial_cases']} cases"
            if row["partial_cases"] > 0
            else f"{row['partial_metrics_done']}/{row['metric_total']}"
        )
        active = (
            f"""<div class="active">进行中：{escape(row['partial_active'])}</div>"""
            if row["partial_active"]
            else ""
        )
        manifest = "yes" if row["manifest_done"] else "no"
        rows.append(
            f"""<tr><td><span class="swatch" style="background:{escape(row['color'])}"></span>
            {escape(row['method_label'])}</td><td>step {row['step']}</td>
            <td>{progress_cell(row['generated'], row['expected_cases'])}</td>
            <td>{escape(manifest)}</td>
            <td>{progress_cell(row['formal_metrics_done'], row['metric_total'])}</td>
            <td>{progress_cell(row['partial_metrics_done'], row['metric_total'], label=partial_label)}{active}</td></tr>"""
        )
    pending = phys_status.get("pending")
    pending_text = "无生成 pending"
    if isinstance(pending, dict):
        pending_text = f"生成 pending：{pending.get('num_pending', '?')}"
        next_task = pending.get("next")
        tasks = pending.get("tasks")
        if isinstance(next_task, dict):
            pending_text += f" · next {next_task.get('method_key')} step {next_task.get('step')}"
        elif isinstance(tasks, list) and tasks:
            pending_text += f" · queue {len(tasks)}"
    return f"""
    <section class="panel" id="physiciq"><div class="panel-head"><h2>PhysicIQ 67-case 监控</h2>
      <span class="state">{escape(pending_text)}</span></div>
      <div class="summary-grid">
        <div><b>{phys_status['generated_total']}/{phys_status['generated_expected']}</b><span>生成 case</span></div>
        <div><b>{phys_status['formal_metric_total']}/{phys_status['formal_metric_expected']}</b><span>正式指标</span></div>
        <div><b>{phys_status['partial_metric_total']}/{phys_status['partial_metric_expected']}</b><span>partial 指标</span></div>
      </div>
      <table><thead><tr><th>方法</th><th>Step</th><th>生成</th><th>Manifest</th>
      <th>正式 67-case 指标</th><th>已生成 case 指标</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    </section>"""


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
    main{{max-width:1800px;margin:auto;padding:18px}}
    .case-head{{margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid var(--line)}}
    h1{{margin:0 0 5px;font-size:19px;overflow-wrap:anywhere}}
    .prompt{{margin:0;max-width:1150px;color:var(--muted);line-height:1.5}}
    .source-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
    .generated-head{{display:flex;align-items:baseline;gap:10px;margin:20px 0 10px}}
    .generated-head h2{{margin:0;font-size:16px}}.count{{color:var(--muted);font-size:12px}}
    .matrix-wrap{{overflow-x:auto;padding-bottom:8px}}
    .generated-matrix{{display:grid;gap:10px;min-width:1180px;align-items:stretch}}
    .matrix-head,.step-label{{display:flex;align-items:center;justify-content:center;
      min-height:44px;padding:8px;background:var(--surface);border:1px solid var(--line);
      font-size:13px;font-weight:850;text-align:center}}
    .matrix-head.method{{border-top-width:4px}}
    .step-label{{position:sticky;left:0;z-index:2;color:var(--accent);
      font-variant-numeric:tabular-nums}}
    .cell{{min-width:0;padding:9px;background:var(--surface);border:1px solid var(--line);
      border-radius:6px}}
    .label{{min-height:24px;padding:1px 2px 7px;color:var(--warm);
      font-size:13px;font-weight:800}}
    video{{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#101416}}
    .checkpoint{{margin-top:7px;color:var(--muted);font-size:11px;overflow-wrap:anywhere}}
    .missing{{display:flex;align-items:center;justify-content:center;min-height:180px;
      background:#eef1f2;border:1px dashed #bdc6ca;color:var(--muted);font-size:12px}}
    .empty{{padding:26px;background:var(--surface);border:1px solid var(--line);
      color:var(--muted);text-align:center}}
    a{{color:var(--accent);font-weight:750;text-decoration:none}}
    @media(max-width:900px){{.toolbar{{flex-wrap:wrap}}.title{{width:100%}}
      .source-grid{{grid-template-columns:1fr}}
      select{{max-width:calc(100vw - 36px)}}}}
  </style>
</head>
<body>
  <div class="toolbar">
    <a href="../">返回监控页</a>
    <div class="title">Checkpoint · test_5 全部结果</div>
    <select id="case" aria-label="案例"></select>
    <select id="step-filter" aria-label="训练 step"></select>
    <button id="play" title="同步播放" aria-label="播放">▶</button>
    <button id="pause" title="同步暂停" aria-label="暂停">Ⅱ</button>
    <button id="replay" title="从头播放" aria-label="重新播放">↺</button>
  </div>
  <main>
    <div class="case-head"><h1 id="case-title"></h1><p class="prompt" id="prompt"></p></div>
    <div class="source-grid">
      <div class="cell"><div class="label">GT · 49 frames @ 30 FPS</div>
        <video id="gt" preload="metadata" playsinline muted></video></div>
      <div class="cell"><div class="label">Input context · 8 frames</div>
        <video id="context" preload="metadata" playsinline muted></video></div>
    </div>
    <div class="generated-head"><h2>已完成 checkpoint</h2><span class="count" id="count"></span></div>
    <div class="matrix-wrap"><div class="generated-matrix" id="generated-matrix"></div></div>
  </main>
  <script>
    const D={data};
    const caseSelect=document.getElementById("case");
    const stepFilter=document.getElementById("step-filter");
    D.cases.forEach((c,i)=>caseSelect.add(new Option(`${{String(i+1).padStart(2,"0")}} · ${{c.stem}}`,c.stem)));
    stepFilter.add(new Option("全部已完成 step","all"));
    [...new Set(D.records.map(record=>record.step))].sort((a,b)=>a-b)
      .forEach(step=>stepFilter.add(new Option(`step ${{step}}`,String(step))));
    function visibleSteps(){{
      const steps=[...new Set(D.records.map(record=>record.step))].sort((a,b)=>a-b);
      return stepFilter.value==="all"
        ? steps
        : steps.filter(step=>String(step)===stepFilter.value);
    }}
    function videos(){{return [...document.querySelectorAll("main video")]}}
    function render(){{
      const c=D.cases.find(x=>x.stem===caseSelect.value);
      if(!c)return;
      videos().forEach(video=>video.pause());
      document.getElementById("case-title").textContent=c.stem;
      document.getElementById("prompt").textContent=c.prompt;
      document.getElementById("gt").src=c.gt;
      document.getElementById("context").src=c.context;
      const steps=visibleSteps();
      const matrix=document.getElementById("generated-matrix");
      matrix.replaceChildren();
      matrix.style.gridTemplateColumns=`88px repeat(${{D.methods.length}},minmax(260px,1fr))`;
      const corner=document.createElement("div");corner.className="matrix-head";
      corner.textContent="训练 step";matrix.append(corner);
      D.methods.forEach(method=>{{
        const header=document.createElement("div");header.className="matrix-head method";
        header.textContent=method.label;header.style.color=method.color;
        header.style.borderTopColor=method.color;matrix.append(header);
      }});
      let count=0;
      steps.forEach(step=>{{
        const stepLabel=document.createElement("div");stepLabel.className="step-label";
        stepLabel.textContent=`step ${{step}}`;matrix.append(stepLabel);
        D.methods.forEach(method=>{{
          const record=D.records.find(item=>
            item.step===step&&item.method_key===method.key);
          if(!record){{
            const missing=document.createElement("div");missing.className="missing";
            missing.textContent="该 step 无此方法权重";matrix.append(missing);return;
          }}
          count+=1;
          const cell=document.createElement("div");cell.className="cell";
          cell.style.borderTop=`3px solid ${{method.color}}`;
          const label=document.createElement("div");label.className="label";
          label.textContent=`${{record.method_label}} · step ${{record.step}}`;
          label.style.color=method.color;
          const video=document.createElement("video");
          video.preload="metadata";video.playsInline=true;video.muted=true;
          video.src=record.videos[c.stem];
          const checkpoint=document.createElement("div");checkpoint.className="checkpoint";
          checkpoint.textContent=record.checkpoint_dir;
          cell.append(label,video,checkpoint);matrix.append(cell);
        }});
      }});
      document.getElementById("count").textContent=
        `${{steps.length}} 行 · ${{count}} 个结果`;
    }}
    stepFilter.addEventListener("change",render);caseSelect.addEventListener("change",render);
    document.getElementById("play").onclick=()=>videos().forEach(video=>video.play().catch(()=>{{}}));
    document.getElementById("pause").onclick=()=>videos().forEach(video=>video.pause());
    document.getElementById("replay").onclick=()=>videos().forEach(video=>{{video.currentTime=0;video.play().catch(()=>{{}})}});
    render();
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


def build_watch_index(
    status: list[dict[str, Any]],
    pending: bool,
    phys_status: dict[str, Any] | None,
) -> str:
    rows = "".join(
        f"""<tr><td><span class="swatch" style="background:{escape(row['color'])}"></span>
        {escape(row['method_label'])}</td><td>{row['discovered']}</td><td>{row['inferred']}</td>
        <td>{escape(row['latest_step'] if row['latest_step'] is not None else '—')}</td>
        <td>{row['metric_done']}/{row['metric_total']}</td></tr>"""
        for row in status
    )
    state = "推理排队中" if pending else "持续监听"
    phys_section = build_physiciq_section(phys_status)
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
    .progress{{position:relative;width:100%;height:7px;margin-bottom:4px;overflow:hidden;
      background:#e7ecee;border-radius:999px}}.progress span{{display:block;height:100%;
      background:var(--accent);border-radius:999px}}.progtext{{color:var(--muted);
      font-size:12px}}.active{{margin-top:3px;color:var(--warm);font-size:12px;font-weight:800}}
    .summary-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;
      margin:0 0 12px}}.summary-grid div{{padding:10px;background:#f7fafb;
      border:1px solid var(--line);border-radius:5px}}.summary-grid b{{display:block;
      font-size:18px}}.summary-grid span{{color:var(--muted);font-size:12px}}
    .home{{display:inline-block;margin-top:15px;color:var(--accent);font-weight:800;text-decoration:none}}
    @media(max-width:700px){{main{{padding:12px}}.links{{grid-template-columns:1fr}}
      .summary-grid{{grid-template-columns:1fr}}}}
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
    {phys_section}
    <a class="home" href="../">返回训练可视化总览</a>
  </main>
</body>
</html>
"""


def build_master_hub(
    config: dict[str, Any],
    status: list[dict[str, Any]],
    phys_status: dict[str, Any] | None,
) -> None:
    hub_root = Path(config["paths"]["master_hub_root"]).resolve()
    watch_root = Path(config["paths"]["watch_root"]).resolve()
    baseline_gallery = Path(config["paths"]["baseline_gallery_root"]).resolve()
    hub_root.mkdir(parents=True, exist_ok=True)
    link_directory(watch_root / "site" / "videos", hub_root / "gallery")
    link_directory(baseline_gallery, hub_root / "initial-gallery")
    link_directory(watch_root / "site", hub_root / "checkpoint-watch")
    total_inferred = sum(row["inferred"] for row in status)
    total_discovered = sum(row["discovered"] for row in status)
    total_metrics = sum(row["metric_done"] for row in status)
    expected_metrics = sum(row["metric_total"] for row in status)
    physiciq_entry = ""
    if config.get("physiciq", {}).get("enabled"):
        phys_config = config["physiciq"]
        leaf_path = Path(phys_config["leaf_folders"]).resolve()
        plot_root = leaf_path.parent / "_metric_plots" / leaf_path.stem
        trigger_steps = [int(step) for step in phys_config["trigger_steps"]]
        generated_text = "生成 pending"
        metric_text = "正式指标 pending"
        partial_text = "partial 指标 pending"
        if phys_status is not None:
            generated_text = (
                f"生成 {phys_status['generated_total']}/"
                f"{phys_status['generated_expected']}"
            )
            metric_text = (
                f"正式指标 {phys_status['formal_metric_total']}/"
                f"{phys_status['formal_metric_expected']}"
            )
            partial_text = (
                f"partial {phys_status['partial_metric_total']}/"
                f"{phys_status['partial_metric_expected']}"
            )
        if (plot_root / "index.html").is_file():
            link_directory(plot_root, hub_root / "physiciq-step1500-metrics")
            action = (
                '<a href="checkpoint-watch/#physiciq">进入 PhysicIQ 监控</a>'
                '<a href="physiciq-step1500-metrics/">进入 PhysicIQ 指标图</a>'
            )
        else:
            action = '<a href="checkpoint-watch/#physiciq">进入 PhysicIQ 监控</a>'
        step_text = " / ".join(str(step) for step in trigger_steps)
        physiciq_entry = f"""
    <section class="entry"><div><h2>PhysicIQ 67-case · Checkpoint 对比</h2>
      <div class="meta">Full-SA、S-head59、T-head70 · 40 denoising steps · 完整 14 项指标</div>
      {action}</div><div class="status">{generated_text}<strong>{metric_text}</strong><em>{partial_text}</em><small>step {step_text}</small></div>
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
    a{{display:inline-block;margin:13px 8px 0 0;padding:8px 12px;color:#fff;background:var(--accent);
      border-radius:5px;text-decoration:none;font-weight:800}}.status{{align-self:start;min-width:150px;
      padding:9px 11px;border-left:3px solid var(--accent);background:#edf6f5;
      color:var(--accent);font-size:13px;font-weight:750}}.status strong{{display:block;
      margin-top:3px;color:var(--ink);font-size:16px}}.status em,.status small{{display:block;
      margin-top:3px;color:var(--muted);font-style:normal;font-size:12px}}@media(max-width:650px){{
      main{{padding:12px}}.entry{{grid-template-columns:1fr}}.status{{justify-self:stretch}}}}
  </style>
</head>
<body>
  <header><h1>xSSC LoRA 训练可视化总览</h1>
    <p>Object-only、Full-SA、S-head59 与 T-head70</p></header>
  <main><div class="section-title">训练与推理</div><div class="entries">
    <section class="entry"><div><h2>Checkpoint 自动评测</h2>
      <div class="meta">自动发现权重、生成 test_5、计算完整指标并绘制训练 step 曲线</div>
      <a href="checkpoint-watch/">进入自动评测</a>
      <a href="checkpoint-watch/metrics/">test_5 指标曲线</a></div>
      <div class="status">持续监听<strong>推理 {total_inferred}/{total_discovered} · 指标 {total_metrics}/{expected_metrics}</strong></div>
    </section>
    <section class="entry"><div><h2>全部 Checkpoint case 对比</h2>
      <div class="meta">按 case 展示 GT、context 和所有已完成方法/训练 step；支持 step 筛选</div>
      <a href="gallery/">进入案例对比</a></div>
      <div class="status">自动更新<strong>{total_inferred} 个结果</strong></div>
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
    phys_status = build_physiciq_status(config)
    pending = (watch_root / "state" / "inference.pending").is_file()
    (site_root / "index.html").write_text(
        build_watch_index(status, pending, phys_status),
        encoding="utf-8",
    )
    manifest = {
        "num_cases": len(cases),
        "num_inference_results": len(manifests),
        "num_metric_plots": len(plots),
        "status": status,
        "physiciq_status": phys_status,
        "records": records,
        "plots": plots,
    }
    (site_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    build_master_hub(config, status, phys_status)
    print(site_root / "index.html")


if __name__ == "__main__":
    main()
