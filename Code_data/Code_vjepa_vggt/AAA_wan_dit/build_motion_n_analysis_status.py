#!/usr/bin/env python3
"""Build a live status page for the S-feature and S-depth ablations."""

from __future__ import annotations

import argparse
import html
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_CONFIG = SCRIPT_DIR / "head_role_s_feature_split_pilot.json"
DEPTH_CONFIG = SCRIPT_DIR / "s_depth_strata_experiment.json"
UNION_CONFIG = SCRIPT_DIR / "head_role_s_feature_union_pilot.json"
PHASED_CONFIG = SCRIPT_DIR / "head_role_s_feature_phased_pilot.json"
REPORT_DIR = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/multiseed/motion-n-analysis"
)
CASE_GALLERY_URL = "/s-head-ablation/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-config", type=Path, default=FEATURE_CONFIG)
    parser.add_argument("--depth-config", type=Path, default=DEPTH_CONFIG)
    parser.add_argument("--union-config", type=Path, default=UNION_CONFIG)
    parser.add_argument("--phased-config", type=Path, default=PHASED_CONFIG)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def metric_status(root: Path) -> dict[str, Any]:
    metrics = root / "metrics"
    incremental = root / "incremental_metrics_live"
    candidate = metrics if metrics.is_dir() else incremental
    if not candidate.is_dir():
        return {
            "status": "not_started",
            "label": "未启动",
            "complete": 0,
            "failed": 0,
            "expected": 0,
            "detail": "尚未建立 benchmark 指标队列。",
        }

    expected = sum(
        line_count(path) for path in (candidate / "queues").glob("*.tsv")
    )
    complete = line_count(candidate / "completed_tasks.tsv")
    failed = line_count(candidate / "failed_tasks.tsv")
    workers_running = len(list((candidate / "state").glob("*.running")))
    workers_complete = len(list((candidate / "state").glob("*.complete")))
    if expected == 0:
        status, label = "not_started", "未启动"
    elif complete + failed >= expected:
        status, label = "complete", "已结束"
    elif complete or workers_running or workers_complete:
        status, label = "running", "计算中"
    else:
        status, label = "queued", "已排队"
    return {
        "status": status,
        "label": label,
        "complete": complete,
        "failed": failed,
        "expected": expected,
        "detail": (
            f"worker: running {workers_running}, complete {workers_complete}"
            if expected
            else "指标目录存在，但尚未写入任务队列。"
        ),
    }


def motion_status(root: Path) -> dict[str, Any]:
    candidates = [
        root / "motion_metrics",
        root / "motion-analysis",
        root / "motion_analysis",
    ]
    analysis = next((path for path in candidates if path.is_dir()), None)
    if analysis is None:
        return {
            "status": "not_started",
            "label": "未启动",
            "complete": 0,
            "expected": 0,
            "detail": "尚未提取 RAFT/轨迹特征，Motion Impact 与 GT gain 未计算。",
        }
    expected = 0
    inventory = analysis / "inventory.json"
    if inventory.is_file():
        payload = read_json(inventory)
        expected = len(payload.get("entries", payload.get("items", [])))
    complete = sum(1 for path in (analysis / "features").glob("*/features.npz"))
    return {
        "status": "complete" if expected and complete >= expected else "running",
        "label": "已结束" if expected and complete >= expected else "计算中",
        "complete": complete,
        "expected": expected,
        "detail": str(analysis),
    }


def expected_tasks(config: dict[str, Any]) -> int:
    manifest = read_json(Path(config["matched_subset_manifest"]).expanduser().resolve())
    return (
        len(config["models"])
        * len(config["seeds"])
        * len(config["step_ranges"])
        * len(manifest["subsets"])
    )


def generation_status(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["storage"]["output_root"]).expanduser().resolve()
    states: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((root / "state").glob("*.json")):
        try:
            states.append((path, read_json(path)))
        except (OSError, json.JSONDecodeError):
            states.append((path, {"status": "invalid"}))
    counts = Counter(str(payload.get("status", "invalid")) for _, payload in states)
    expected = expected_tasks(config)
    expected_videos = expected * int(config["expected_cases"])
    ready_videos = sum(
        1
        for video in (root / "generation").rglob("*.mp4")
        if video.stat().st_size > 1024
        and video.with_suffix(".json").is_file()
        and not video.with_suffix(".json.lock").exists()
    )
    pending = []
    for path, payload in states:
        status = str(payload.get("status", "invalid"))
        if status != "complete":
            pending.append(
                {
                    "task": payload.get("task_id", path.stem),
                    "status": status,
                    "attempt": payload.get("attempt"),
                    "gpu": payload.get("gpu"),
                    "error": payload.get("error", ""),
                }
            )
    return {
        "root": str(root),
        "expected_tasks": expected,
        "state_counts": dict(counts),
        "ready_videos": ready_videos,
        "expected_videos": expected_videos,
        "pending": pending,
        "benchmark": metric_status(root),
        "motion": motion_status(root),
    }


def progress_percent(complete: int, expected: int) -> float:
    return 0.0 if expected <= 0 else min(100.0, 100.0 * complete / expected)


def status_badge(status: str, label: str) -> str:
    return (
        f'<span class="badge {html.escape(status)}">{html.escape(label)}</span>'
    )


def metric_block(title: str, metric: dict[str, Any]) -> str:
    expected = int(metric["expected"])
    complete = int(metric["complete"])
    if expected:
        value = f"{complete}/{expected}"
        percent = progress_percent(complete, expected)
    else:
        value = metric["label"]
        percent = 0.0
    failed = int(metric.get("failed", 0))
    failed_text = f" · failed {failed}" if failed else ""
    return f"""
      <div class="metric">
        <div class="metric-head"><b>{html.escape(title)}</b>{status_badge(metric["status"], metric["label"])}</div>
        <div class="metric-value">{html.escape(str(value))}{failed_text}</div>
        <div class="bar"><span style="width:{percent:.2f}%"></span></div>
        <p>{html.escape(str(metric["detail"]))}</p>
      </div>
    """


def experiment_section(name: str, description: str, data: dict[str, Any]) -> str:
    counts = data["state_counts"]
    complete_tasks = int(counts.get("complete", 0))
    expected_tasks_value = int(data["expected_tasks"])
    generation_label = (
        "已完成"
        if complete_tasks == expected_tasks_value
        else ("存在失败" if counts.get("failed") else "生成中")
    )
    generation_status_name = (
        "complete"
        if complete_tasks == expected_tasks_value
        else ("failed" if counts.get("failed") else "running")
    )
    pending_rows = []
    for item in data["pending"]:
        pending_rows.append(
            "<tr>"
            f"<td>{html.escape(str(item['task']))}</td>"
            f"<td>{status_badge(str(item['status']), str(item['status']))}</td>"
            f"<td>{html.escape(str(item.get('attempt') or '-'))}</td>"
            f"<td>{html.escape(str(item.get('gpu') if item.get('gpu') is not None else '-'))}</td>"
            f"<td>{html.escape(str(item.get('error') or '-'))}</td>"
            "</tr>"
        )
    pending_html = (
        """
        <details open>
          <summary>未完成任务</summary>
          <div class="table-wrap"><table>
            <thead><tr><th>配置</th><th>状态</th><th>Attempt</th><th>GPU</th><th>说明</th></tr></thead>
            <tbody>%s</tbody>
          </table></div>
        </details>
        """
        % "".join(pending_rows)
        if pending_rows
        else '<p class="done-line">所有生成配置均已完成。</p>'
    )
    generation = {
        "status": generation_status_name,
        "label": generation_label,
        "complete": complete_tasks,
        "expected": expected_tasks_value,
        "failed": int(counts.get("failed", 0)),
        "detail": (
            f"视频 {data['ready_videos']}/{data['expected_videos']} · "
            f"running {counts.get('running', 0)} · failed {counts.get('failed', 0)}"
        ),
    }
    return f"""
    <section>
      <div class="section-title">
        <div><h2>{html.escape(name)}</h2><p>{html.escape(description)}</p></div>
        <a class="open-link" href="{CASE_GALLERY_URL}">查看视频对照</a>
      </div>
      <div class="metrics">
        {metric_block("消融视频生成", generation)}
        {metric_block("17 项 Benchmark", data["benchmark"])}
        {metric_block("Motion Impact / GT gain", data["motion"])}
      </div>
      {pending_html}
      <p class="path">输出：{html.escape(data["root"])}</p>
    </section>
    """


def build_page(
    feature: dict[str, Any],
    union: dict[str, Any],
    phased: dict[str, Any],
    depth: dict[str, Any],
    updated: str,
) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>S Head 消融与指标状态</title>
<style>
:root{{--ink:#18202b;--muted:#667085;--line:#d7dce3;--paper:#f5f7fa;--panel:#fff;--green:#16866f;--amber:#b97808;--red:#b13b46;--blue:#326caa}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:Arial,"Noto Sans SC",sans-serif;letter-spacing:0}}
main{{width:min(1380px,calc(100% - 32px));margin:24px auto 64px}} header{{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:18px}}
h1{{font-size:28px;margin:0 0 6px}} h2{{font-size:20px;margin:0 0 5px}} p{{margin:0;color:var(--muted);line-height:1.55}}
.stamp{{font-size:13px;text-align:right}} section{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:18px;margin:14px 0}}
.section-title{{display:flex;align-items:start;justify-content:space-between;gap:16px;margin-bottom:15px}}
.open-link{{color:var(--blue);text-decoration:none;white-space:nowrap}} .metrics{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
.metric{{border:1px solid var(--line);border-radius:6px;padding:13px;min-height:140px}} .metric-head{{display:flex;justify-content:space-between;align-items:center;gap:8px}}
.metric-value{{font-size:23px;font-weight:700;margin:13px 0 8px}} .metric p{{font-size:13px;margin-top:9px}}
.bar{{height:7px;background:#e8ebef;overflow:hidden}} .bar span{{display:block;height:100%;background:var(--blue)}}
.badge{{display:inline-flex;align-items:center;padding:3px 7px;border-radius:4px;font-size:12px;font-weight:700;background:#eceff3;color:#475467}}
.badge.complete{{background:#dff4ed;color:#106b59}} .badge.running,.badge.queued{{background:#fff1cf;color:#8a5b05}} .badge.failed{{background:#fbe2e5;color:#972f3a}}
details{{margin-top:14px}} summary{{cursor:pointer;font-weight:700;margin-bottom:9px}} .table-wrap{{overflow:auto}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid var(--line);padding:8px 9px;text-align:left;vertical-align:top}} th{{background:#f2f4f7}} td:first-child{{font-family:monospace;min-width:320px}}
.path{{font-family:monospace;font-size:12px;margin-top:12px;overflow-wrap:anywhere}} .done-line{{margin-top:12px;color:var(--green);font-weight:700}}
.note{{border-left:4px solid var(--amber);background:#fff8e7;padding:12px 14px;margin:14px 0;color:#6b4a08}}
@media(max-width:850px){{.metrics{{grid-template-columns:1fr}} header,.section-title{{align-items:start;flex-direction:column}} .stamp{{text-align:left}}}}
</style>
</head>
<body><main>
<header><div><h1>S Head 消融与指标状态</h1><p>S 分类依据拆分与网络深度拆分实验的统一进度页。</p></div><p class="stamp">更新时间：{html.escape(updated)}<br>页面每 30 秒自动刷新</p></header>
<div class="note"><b>状态口径：</b>“未启动”表示没有指标任务队列或运动特征产物，并非分数为 0。Benchmark 与 Motion 两类指标会分别统计，避免混为同一进度。</div>
{experiment_section("S 分类消融", "local_enrichment 主导组 vs same_frame_mass 主导组；两组 head 不重叠，并保持 block 数量匹配。", feature)}
{experiment_section("S 分类联合消融", "将上述两个互斥的 32-head subset 取并集，同时消融全部 64 个 head。", union)}
{experiment_section("S 分类分阶段消融", "对 Local-32、Same-frame-32、Union-64 分别应用去噪区间 0–10 与 10–20。", phased)}
{experiment_section("S 深度消融", "按 early / middle / late block 分层，比较不同去噪阶段下 S head 的深度效应。", depth)}
</main></body></html>
"""


def update_once(
    feature_config_path: Path,
    depth_config_path: Path,
    union_config_path: Path,
    phased_config_path: Path,
    report_dir: Path,
) -> dict[str, Any]:
    feature = generation_status(read_json(feature_config_path))
    depth = generation_status(read_json(depth_config_path))
    union = generation_status(read_json(union_config_path))
    phased = generation_status(read_json(phased_config_path))
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    payload = {
        "updated_utc": updated,
        "s_feature": feature,
        "s_feature_union": union,
        "s_feature_phased": phased,
        "s_depth": depth,
    }
    atomic_write(
        report_dir / "status.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write(
        report_dir / "index.html",
        build_page(feature, union, phased, depth, updated),
    )
    return payload


def main() -> None:
    args = parse_args()
    feature_config = args.feature_config.expanduser().resolve()
    depth_config = args.depth_config.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    while True:
        union_config = args.union_config.expanduser().resolve()
        phased_config = args.phased_config.expanduser().resolve()
        payload = update_once(
            feature_config,
            depth_config,
            union_config,
            phased_config,
            report_dir,
        )
        print(
            "[motion-n-status] "
            f"feature={payload['s_feature']['ready_videos']}/"
            f"{payload['s_feature']['expected_videos']} "
            f"union={payload['s_feature_union']['ready_videos']}/"
            f"{payload['s_feature_union']['expected_videos']} "
            f"phased={payload['s_feature_phased']['ready_videos']}/"
            f"{payload['s_feature_phased']['expected_videos']} "
            f"depth={payload['s_depth']['ready_videos']}/"
            f"{payload['s_depth']['expected_videos']}",
            flush=True,
        )
        if not args.watch:
            break
        time.sleep(max(10, args.poll_seconds))


if __name__ == "__main__":
    main()
