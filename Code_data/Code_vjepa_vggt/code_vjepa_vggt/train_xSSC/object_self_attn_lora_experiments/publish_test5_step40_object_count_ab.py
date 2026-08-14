#!/usr/bin/env python3
"""Publish the dedicated A/B watcher pages into the live 8844 hub."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re


OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/test5_step40_object_count_ab")
DEDICATED_HUB = OUTPUT_ROOT / "hub"
LIVE_HUB = Path("/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub")
WATCH_ROOT = OUTPUT_ROOT / "watch"
START_MARKER = "<!-- TEST5_STEP40_OBJECT_COUNT_AB_START -->"
END_MARKER = "<!-- TEST5_STEP40_OBJECT_COUNT_AB_END -->"
CPU_METRICS = (
    "physics_iq_with_context",
    "physics_iq_without_context",
    "pmf_with_context",
    "pmf_without_context",
)
GPU_METRICS = (
    "wmreward",
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_temporal_flickering",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
    "videophy2",
    "cosmos_reason1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="preparing")
    return parser.parse_args()


def ensure_link(target: Path, link: Path) -> None:
    if not target.is_dir():
        raise FileNotFoundError(f"Publish target is missing: {target}")
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink path: {link}")
    link.symlink_to(target.resolve(), target_is_directory=True)


def count_metric_markers(names: tuple[str, ...]) -> int:
    root = WATCH_ROOT / "state" / "metrics"
    return sum(len(list(root.glob(f"*/step-000500/{name}.json"))) for name in names)


def collect_status(stage: str) -> dict:
    manifests = len(
        list((WATCH_ROOT / "state" / "checkpoints").glob("*/step-000500.json"))
    )
    videos = len(list((WATCH_ROOT / "results").glob("*/step-000500_steps40_*/*.mp4")))
    status = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "gpu_id": 7,
        "completed_generation_groups": manifests,
        "expected_generation_groups": 36,
        "completed_videos": videos,
        "expected_videos": 720,
        "completed_cpu_metric_tasks": count_metric_markers(CPU_METRICS),
        "expected_cpu_metric_tasks": 144,
        "completed_gpu_metric_tasks": count_metric_markers(GPU_METRICS),
        "expected_gpu_metric_tasks": 360,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "pipeline_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return status


def update_live_index(status: dict) -> None:
    index = LIVE_HUB / "index.html"
    page = index.read_text(encoding="utf-8")
    entry = f"""{START_MARKER}
    <section class="entry"><div><h2>test_5 · 40-step Object-count A/B</h2>
      <div class="meta">18 个 step-500 权重；原始 prompt 与 object-count 约束 prompt 的 40-step 配对生成和完整指标</div>
      <a href="test5-step40-object-count-ab/">Case 视频对比</a>
      <a href="test5-step40-object-count-ab-average-metrics/">全 case 平均指标</a>
      <a href="test5-step40-object-count-ab-status/">流水线状态</a></div>
      <div class="status">{status['stage']}<strong>生成 {status['completed_videos']}/{status['expected_videos']}</strong>
      <em>CPU 指标 {status['completed_cpu_metric_tasks']}/{status['expected_cpu_metric_tasks']}</em>
      <small>GPU 指标 {status['completed_gpu_metric_tasks']}/{status['expected_gpu_metric_tasks']} · GPU 7</small></div>
    </section>
{END_MARKER}"""
    pattern = re.compile(
        re.escape(START_MARKER) + ".*?" + re.escape(END_MARKER),
        re.DOTALL,
    )
    if pattern.search(page):
        page = pattern.sub(entry, page)
    else:
        page = page.replace("</main>", entry + "\n</main>")
    temporary = index.with_name(f".{index.name}.tmp.{os.getpid()}")
    temporary.write_text(page, encoding="utf-8")
    os.replace(temporary, index)


def main() -> None:
    args = parse_args()
    links = {
        "test5-step40-object-count-ab": DEDICATED_HUB / "test5",
        "test5-step40-object-count-ab-average-metrics": (
            DEDICATED_HUB / "test5-average-metrics"
        ),
        "test5-step40-object-count-ab-status": DEDICATED_HUB / "checkpoint-watch",
    }
    for name, target in links.items():
        ensure_link(target, LIVE_HUB / name)
    status = collect_status(args.stage)
    update_live_index(status)
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
