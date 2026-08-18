#!/usr/bin/env python3
"""Build checkpoint video galleries and per-metric training-step curves."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from build_project_provenance_page import build_project_info_page
from build_physiciq_physrvg_worst_case_dashboard import (
    build_dashboard as build_physiciq_physrvg_worst_case_dashboard,
)
from build_physiciq_top3_physrvg_all_cases_dashboard import (
    build_dashboard as build_physiciq_top3_physrvg_all_cases_dashboard,
)
from build_physiciq_top3_physrvg_top10_videos import (
    build_dashboard as build_physiciq_top3_physrvg_top10_videos,
)


METHOD_PLOT_STYLES = {
    "object_only": {"marker": "P", "linestyle": "--"},
    "full_sa": {"marker": "o", "linestyle": "-"},
    "full_sa_resume": {"marker": "o", "linestyle": "-"},
    "full_sa_no_object": {"marker": "X", "linestyle": (0, (5, 1))},
    "full_sa_no_object_vjepa_loss": {
        "marker": "d",
        "linestyle": (0, (3, 1, 1, 1)),
    },
    "full_sa_physrvg_vjepa_loss": {
        "marker": "P",
        "linestyle": (0, (5, 1, 1, 1)),
    },
    "full_sa_physrvg_dit_gpu56": {
        "marker": ">",
        "linestyle": (0, (1, 1)),
    },
    "full_sa_no_object_xssc_loss_dinov3_movic_step50000": {
        "marker": "D",
        "linestyle": (0, (5, 1, 1, 1)),
    },
    "full_sa_object_slot_dedup_xssc50k_xssc_loss_dinov3_movic_step50000": {
        "marker": "H",
        "linestyle": (0, (4, 1, 1, 1)),
    },
    "full_sa_no_object_pybullet100": {"marker": "p", "linestyle": (0, (4, 1))},
    "full_sa_no_object_kubric100": {"marker": "*", "linestyle": (0, (2, 1))},
    "s_head59": {"marker": "s", "linestyle": "--"},
    "s_head59_resume": {"marker": "s", "linestyle": "--"},
    "t_head70": {"marker": "^", "linestyle": "-."},
    "t_head70_resume": {"marker": "^", "linestyle": "-."},
    "t_head70_no_object": {"marker": "h", "linestyle": (0, (3, 2))},
    "t_head100_lora_pck32_no_object": {
        "marker": "8",
        "linestyle": (0, (1, 1)),
    },
    "t_head70_slot_dedup_merge": {
        "marker": "v",
        "linestyle": (0, (3, 1, 1, 1)),
    },
    "t_head70_slot_dedup_merge_xssc_step050000": {
        "marker": "<",
        "linestyle": (0, (5, 1, 1, 1)),
    },
    "slot_dedup_merge": {"marker": "D", "linestyle": ":"},
}
DEFAULT_PLOT_STYLE = {"marker": "o", "linestyle": "-"}

CASE_METRIC_SPECS = [
    {"key": "videophy2_pc_raw", "label": "VideoPhy2 PC raw", "direction": "higher", "path": ("videophy2", "pc_raw_score")},
    {"key": "cosmos_reason1", "label": "Cosmos Reason", "direction": "higher", "path": ("cosmos_reason1", "score")},
    {"key": "physics_iq_with_context", "label": "Physics-IQ ctx", "direction": "higher", "path": ("physics_iq_with_context", "score")},
    {"key": "physics_iq_without_context", "label": "Physics-IQ no ctx", "direction": "higher", "path": ("physics_iq_without_context", "score")},
    {"key": "videophy2", "label": "VideoPhy2 joint", "direction": "higher", "path": ("videophy2", "score")},
    {"key": "videophy2_sa", "label": "VideoPhy2 SA", "direction": "higher", "path": ("videophy2", "sa_score")},
    {"key": "videophy2_pc", "label": "VideoPhy2 PC", "direction": "higher", "path": ("videophy2", "pc_score")},
    {"key": "videophy2_joint_rate", "label": "VideoPhy2 pass", "direction": "higher", "path": ("videophy2", "joint_rate")},
    {"key": "pmf_with_context", "label": "PMF ctx", "direction": "higher", "path": ("pmf_with_context", "score")},
    {"key": "pmf_without_context", "label": "PMF no ctx", "direction": "higher", "path": ("pmf_without_context", "score")},
    {"key": "wmreward", "label": "WMReward surprise", "direction": "lower", "path": ("wmreward", "surprise")},
    {"key": "vbench_subject_consistency", "label": "VBench subject", "direction": "higher", "path": ("vbench_subject_consistency", "score")},
    {"key": "vbench_background_consistency", "label": "VBench background", "direction": "higher", "path": ("vbench_background_consistency", "score")},
    {"key": "vbench_temporal_flickering", "label": "VBench temporal", "direction": "higher", "path": ("vbench_temporal_flickering", "score")},
    {"key": "vbench_motion_smoothness", "label": "VBench smoothness", "direction": "higher", "path": ("vbench_motion_smoothness", "score")},
    {"key": "vbench_dynamic_degree", "label": "VBench dynamic", "direction": "higher", "path": ("vbench_dynamic_degree", "score")},
    {"key": "vbench_aesthetic_quality", "label": "VBench aesthetic", "direction": "higher", "path": ("vbench_aesthetic_quality", "score")},
    {"key": "vbench_imaging_quality", "label": "VBench imaging", "direction": "higher", "path": ("vbench_imaging_quality", "score")},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def extract_case_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for spec in CASE_METRIC_SPECS:
        value: Any = payload
        for key in spec["path"]:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if pd.notna(number):
            metrics[str(spec["key"])] = number
    return metrics


def load_case_metrics(
    result_root: Path,
    cases: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for case in cases:
        result_path = result_root / f"{case['stem']}.json"
        if not result_path.is_file():
            continue
        try:
            payload = load_json(result_path)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        values = extract_case_metrics(payload)
        if values:
            metrics[str(case["stem"])] = values
    return metrics


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def link_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(source.resolve())
    os.replace(temporary, destination)


def link_directory(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not (destination.is_symlink() or destination.is_file()):
        raise RuntimeError(f"Refusing to replace directory: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(source.resolve(), target_is_directory=True)
    os.replace(temporary, destination)


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


def load_discovered_checkpoints(watch_root: Path) -> list[dict[str, Any]]:
    """Load discovered checkpoints from watcher state (including still pending checkpoints)."""
    discovery_path = watch_root / "state" / "discovery.json"
    if not discovery_path.is_file():
        return []
    payload = load_json(discovery_path)
    discovered = payload.get("checkpoints", [])
    records: list[dict[str, Any]] = []
    for item in discovered:
        if not isinstance(item, dict):
            continue
        if "method_key" not in item or "step" not in item:
            continue
        records.append(item)
    return sorted(records, key=lambda row: (row["method_key"], int(row["step"])))


def load_configured_checkpoints(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Scan current config roots so stale long-running watchers cannot hide new methods."""
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for method_index, method in enumerate(config["methods"]):
        min_step = int(method.get("min_step", 0))
        for item in method.get("static_checkpoints", []):
            step = int(item["step"])
            if step < min_step:
                continue
            records[(str(method["key"]), step)] = {
                "method_key": method["key"],
                "method_label": method["label"],
                "method_index": method_index,
                "step": step,
                "checkpoint_dir": str(Path(item["path"]).resolve()),
                "source": "configured-static",
            }
        for root_value in method.get("watch_roots", []):
            root = Path(root_value).resolve()
            if not root.is_dir():
                continue
            for checkpoint in sorted(root.glob("step-*")):
                try:
                    step = int(checkpoint.name.removeprefix("step-"))
                except ValueError:
                    continue
                if step < min_step:
                    continue
                records[(str(method["key"]), step)] = {
                    "method_key": method["key"],
                    "method_label": method["label"],
                    "method_index": method_index,
                    "step": step,
                    "checkpoint_dir": str(checkpoint),
                    "source": "configured-root",
                }
    return sorted(
        records.values(),
        key=lambda row: (int(row["method_index"]), int(row["step"])),
    )


def load_physiciq_manifests(watch_root: Path) -> list[dict[str, Any]]:
    manifests = [
        load_json(path)
        for path in sorted(
            (watch_root / "state" / "physiciq" / "inference").glob(
                "*/step-*.json"
            )
        )
        if path.is_file()
    ]
    method_order = {
        manifest["method_key"]: index
        for index, manifest in enumerate(manifests)
    }
    return sorted(
        manifests,
        key=lambda row: (method_order.get(row["method_key"], 999), int(row["step"])),
    )


def load_live_test_manifests(
    config: dict[str, Any], completed: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Add partially generated checkpoints to galleries without marking them complete."""
    watch_root = Path(config["paths"]["watch_root"]).resolve()
    runtime = config["runtime"]
    completed_pairs = {
        (str(row["method_key"]), int(row["step"])) for row in completed
    }
    live = list(completed)
    for task in load_configured_checkpoints(config):
        method_key = str(task["method_key"])
        step = int(task["step"])
        pair = (method_key, step)
        if pair in completed_pairs:
            continue
        output_name = (
            f"step-{step:06d}_steps{int(runtime['num_inference_steps'])}"
            f"_{int(runtime['height'])}x{int(runtime['width'])}"
            f"_ctx{int(runtime['context_frames']):02d}_{int(runtime['num_frames'])}f"
        )
        result_root = watch_root / "results" / method_key / output_name
        live.append(
            {
                "method_key": method_key,
                "method_label": task["method_label"],
                "method_index": task["method_index"],
                "step": step,
                "checkpoint_dir": task["checkpoint_dir"],
                "result_root": str(result_root),
                "origin": "watcher-live",
            }
        )
        completed_pairs.add(pair)
    return sorted(live, key=lambda row: (row.get("method_index", 999), row["step"]))


def load_live_physiciq_manifests(
    config: dict[str, Any], completed: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    phys = config.get("physiciq", {})
    if not phys.get("enabled"):
        return completed
    completed_pairs = {
        (str(row["method_key"]), int(row["step"])) for row in completed
    }
    methods = {method["key"]: method for method in config["methods"]}
    live = list(completed)
    for task in load_configured_checkpoints(config):
        method_key = str(task["method_key"])
        step = int(task["step"])
        pair = (method_key, step)
        if method_key not in phys["method_keys"] or pair in completed_pairs:
            continue
        name = phys["method_name_template"].format(
            method_key=method_key,
            step=step,
        )
        result_root = Path(phys["output_root"]).resolve() / name
        method = methods.get(method_key, {})
        live.append(
            {
                "method_key": method_key,
                "method_label": method.get("label", method_key),
                "step": step,
                "checkpoint_dir": task["checkpoint_dir"],
                "result_root": str(result_root),
                "origin": "watcher-live",
            }
        )
        completed_pairs.add(pair)
    return sorted(live, key=lambda row: (row["method_key"], row["step"]))


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
    configured_steps = phys.get("trigger_steps", "all")
    if configured_steps == "all":
        discovered = load_configured_checkpoints(config)
        task_pairs = sorted(
            {
                (manifest["method_key"], int(manifest["step"]))
                for manifest in discovered
                if manifest["method_key"] in phys["method_keys"]
            },
            key=lambda item: (item[1], phys["method_keys"].index(item[0])),
        )
    else:
        task_pairs = [
            (method_key, int(step))
            for step in configured_steps
            for method_key in phys["method_keys"]
        ]
    rows: list[dict[str, Any]] = []
    for method_key, step in task_pairs:
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
    *,
    site_name: str = "videos",
) -> list[dict[str, Any]]:
    watch_root = Path(config["paths"]["watch_root"]).resolve()
    videos_root = watch_root / "site" / site_name
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
            "origin": manifest.get("origin", "watcher"),
            "videos": {},
            "metrics": load_case_metrics(result_root, cases),
        }
        for case in cases:
            source = result_root / f"{case['stem']}.mp4"
            if not source.is_file():
                continue
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
    *,
    page_title: str = "Checkpoint · test_5 全部结果",
    methods_override: list[dict[str, str]] | None = None,
) -> str:
    if methods_override is None:
        method_order = config["methods"]
    else:
        method_order = methods_override
    methods = [
        {
            "key": method["key"],
            "label": method["label"],
            "color": method["color"],
            "condition": method.get("condition"),
            "schemeKey": method.get("scheme_key", method["key"]),
            "schemeLabel": method.get("scheme_label", method["label"]),
        }
        for method in method_order
    ]
    data = json.dumps(
        {
            "methods": methods,
            "records": records,
            "cases": cases,
            "abExperiment": bool(config.get("ab_experiment")),
            "autoRefresh": bool(config.get("ab_experiment")),
            "metricSpecs": [
                {
                    "key": spec["key"],
                    "label": spec["label"],
                    "direction": spec["direction"],
                }
                for spec in CASE_METRIC_SPECS
            ],
        },
        ensure_ascii=False,
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(page_title)}</title>
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
    .generated-matrix.ab{{min-width:960px;grid-template-columns:minmax(220px,.55fr)
      repeat(2,minmax(320px,1fr))}}
    .matrix-head,.step-label{{display:flex;align-items:center;justify-content:center;
      min-height:44px;padding:8px;background:var(--surface);border:1px solid var(--line);
      font-size:13px;font-weight:850;text-align:center}}
    .matrix-head.method{{border-top-width:4px}}
    .step-label{{position:sticky;left:0;z-index:2;color:var(--accent);
      font-variant-numeric:tabular-nums}}
    .scheme-label{{position:sticky;left:0;z-index:2;display:flex;align-items:center;
      min-height:220px;padding:14px;background:var(--surface);border:1px solid var(--line);
      border-left:4px solid var(--accent);font-size:13px;font-weight:850;line-height:1.4}}
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
    .metrics-section{{margin-top:24px;padding-top:16px;border-top:1px solid var(--line)}}
    .metrics-section h2{{margin:0 0 4px;font-size:16px}}.metrics-note{{margin:0 0 10px;
      color:var(--muted);font-size:12px}}.metrics-wrap{{overflow:auto;max-height:70vh;
      border:1px solid var(--line);background:var(--surface)}}
    .metrics-table{{width:max-content;min-width:100%;border-collapse:separate;border-spacing:0;
      font-size:11px;font-variant-numeric:tabular-nums}}
    .metrics-table th,.metrics-table td{{height:30px;padding:5px 7px;border-right:1px solid #e2e7e9;
      border-bottom:1px solid #e2e7e9;text-align:center;white-space:nowrap}}
    .metrics-table thead th{{position:sticky;top:0;z-index:3;background:#edf1f2;color:#45545b}}
    .metrics-table .method-col{{position:sticky;left:0;z-index:2;min-width:184px;
      text-align:left;background:#fff;font-weight:800}}
    .metrics-table thead .method-col{{z-index:4;background:#e5ebed}}
    .metrics-table .step-col{{position:sticky;left:184px;z-index:2;min-width:70px;
      background:#fff;color:var(--muted)}}
    .metrics-table thead .step-col{{z-index:4;background:#e5ebed}}
    .metrics-table td.best{{background:#dff3e7;color:#075d37;font-weight:900}}
    .metrics-table td.missing-value{{color:#9aa5aa}}.direction{{color:#66757c;font-size:10px}}
    a{{color:var(--accent);font-weight:750;text-decoration:none}}
    @media(max-width:900px){{.toolbar{{flex-wrap:wrap}}.title{{width:100%}}
      .source-grid{{grid-template-columns:1fr}}
      select{{max-width:calc(100vw - 36px)}}}}
  </style>
</head>
<body>
  <div class="toolbar">
    <a href="../">返回监控页</a>
    <div class="title">{escape(page_title)}</div>
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
    <div class="generated-head"><h2 id="generated-title">已完成 checkpoint</h2><span class="count" id="count"></span></div>
    <div class="matrix-wrap"><div class="generated-matrix" id="generated-matrix"></div></div>
    <section class="metrics-section"><h2>当前 case 指标对比</h2>
      <p class="metrics-note">每行一个方法 checkpoint；★ 表示当前筛选范围内该指标最佳。WMReward surprise 越低越好，其余越高越好。</p>
      <div class="metrics-wrap"><table class="metrics-table" id="metrics-table"></table></div>
    </section>
    <section class="metrics-section"><h2>所有 case 平均指标</h2>
      <p class="metrics-note">对本页全部 case 求均值；悬停数值可查看有效样本数。★ 表示当前 step 筛选范围内平均指标最佳。</p>
      <div class="metrics-wrap"><table class="metrics-table" id="average-metrics-table"></table></div>
    </section>
  </main>
  <script>
    const D={data};
    const caseSelect=document.getElementById("case");
    const stepFilter=document.getElementById("step-filter");
    function formatStepLabel(step){{
      return D.records.some(record=>record.step===step&&record.step_kind==="inference")
        ? `inference ${{step}}`
        : `step ${{step}}`;
    }}
    D.cases.forEach((c,i)=>caseSelect.add(new Option(`${{String(i+1).padStart(2,"0")}} · ${{c.stem}}`,c.stem)));
    stepFilter.add(new Option("全部已完成 step","all"));
    [...new Set(D.records.map(record=>record.step))].sort((a,b)=>a-b)
      .forEach(step=>stepFilter.add(new Option(formatStepLabel(step),String(step))));
    function visibleSteps(){{
      const steps=[...new Set(D.records.map(record=>record.step))].sort((a,b)=>a-b);
      return stepFilter.value==="all"
        ? steps
        : steps.filter(step=>String(step)===stepFilter.value);
    }}
    function videos(){{return [...document.querySelectorAll("main video")]}}
    function formatMetric(value){{
      const magnitude=Math.abs(value);
      if(magnitude>=10)return value.toFixed(2);
      if(magnitude>=1)return value.toFixed(3);
      return value.toFixed(4);
    }}
    function renderMetrics(c,steps){{
      const methodIndex=new Map(D.methods.map((method,index)=>[method.key,index]));
      const rows=D.records
        .filter(record=>steps.includes(record.step))
        .map(record=>({{
          record,
          method:D.methods.find(method=>method.key===record.method_key),
          values:record.metrics?.[c.stem]??{{}},
        }}))
        .sort((a,b)=>(methodIndex.get(a.record.method_key)??999)-
          (methodIndex.get(b.record.method_key)??999)||a.record.step-b.record.step);
      const best=new Map();
      D.metricSpecs.forEach(spec=>{{
        const values=rows.map(row=>Number(row.values[spec.key])).filter(Number.isFinite);
        if(!values.length)return;
        best.set(spec.key,spec.direction==="lower"?Math.min(...values):Math.max(...values));
      }});
      const table=document.getElementById("metrics-table");table.replaceChildren();
      const thead=document.createElement("thead");const header=document.createElement("tr");
      const methodHead=document.createElement("th");methodHead.className="method-col";
      methodHead.textContent="方法";header.append(methodHead);
      const stepHead=document.createElement("th");stepHead.className="step-col";
      stepHead.textContent="Step";header.append(stepHead);
      D.metricSpecs.forEach(spec=>{{
        const th=document.createElement("th");
        th.innerHTML=`${{spec.label}} <span class="direction">${{spec.direction==="lower"?"↓":"↑"}}</span>`;
        header.append(th);
      }});thead.append(header);table.append(thead);
      const tbody=document.createElement("tbody");
      rows.forEach(row=>{{
        const tr=document.createElement("tr");
        const methodCell=document.createElement("td");methodCell.className="method-col";
        methodCell.textContent=row.record.method_label;
        methodCell.style.color=row.method?.color??"#172126";tr.append(methodCell);
        const stepCell=document.createElement("td");stepCell.className="step-col";
        stepCell.textContent=row.record.step_kind==="inference"
          ? `infer ${{row.record.step}}`
          : row.record.step;tr.append(stepCell);
        D.metricSpecs.forEach(spec=>{{
          const td=document.createElement("td");const value=Number(row.values[spec.key]);
          if(!Number.isFinite(value)){{td.textContent="—";td.className="missing-value";}}
          else{{
            const isBest=best.has(spec.key)&&Math.abs(value-best.get(spec.key))<=1e-9;
            td.textContent=`${{isBest?"★ ":""}}${{formatMetric(value)}}`;
            if(isBest)td.className="best";
          }}
          tr.append(td);
        }});tbody.append(tr);
      }});table.append(tbody);
    }}
    function appendAbVideoCell(matrix,method,record,c,step){{
      const videoPath=record?.videos?.[c.stem];
      if(!videoPath){{
        const missing=document.createElement("div");missing.className="missing";
        missing.textContent="等待该方案生成";matrix.append(missing);return 0;
      }}
      const cell=document.createElement("div");cell.className="cell";
      cell.style.borderTop=`3px solid ${{method.color}}`;
      const label=document.createElement("div");label.className="label";
      const condition=method.condition==="control_original_prompt"
        ? "A · original prompt"
        : "B · identity/count prompt";
      label.textContent=`${{condition}} · ${{formatStepLabel(step)}}`;
      label.style.color=method.color;
      const video=document.createElement("video");
      video.preload="metadata";video.playsInline=true;video.muted=true;video.src=videoPath;
      const checkpoint=document.createElement("div");checkpoint.className="checkpoint";
      checkpoint.textContent=record.checkpoint_dir;
      cell.append(label,video,checkpoint);matrix.append(cell);return 1;
    }}
    function renderAbMatrix(c,steps,matrix){{
      matrix.className="generated-matrix ab";
      const headers=["训练方案","A · original prompt","B · identity/count prompt"];
      headers.forEach((text,index)=>{{
        const header=document.createElement("div");header.className="matrix-head";
        header.textContent=text;
        if(index===1)header.style.borderTop="4px solid #4D4D4D";
        if(index===2)header.style.borderTop="4px solid #167A52";
        matrix.append(header);
      }});
      const pairs=[];const pairByKey=new Map();
      D.methods.forEach(method=>{{
        if(!pairByKey.has(method.schemeKey)){{
          const pair={{key:method.schemeKey,label:method.schemeLabel,methods:[]}};
          pairByKey.set(method.schemeKey,pair);pairs.push(pair);
        }}
        pairByKey.get(method.schemeKey).methods.push(method);
      }});
      let count=0;
      steps.forEach(step=>pairs.forEach(pair=>{{
        const control=pair.methods.find(method=>method.condition==="control_original_prompt");
        const treatment=pair.methods.find(method=>method.condition!=="control_original_prompt");
        const scheme=document.createElement("div");scheme.className="scheme-label";
        scheme.textContent=pair.label;scheme.style.borderLeftColor=control?.color??"#006d77";
        matrix.append(scheme);
        [control,treatment].forEach(method=>{{
          if(!method){{
            const missing=document.createElement("div");missing.className="missing";
            missing.textContent="缺少 A/B 配置";matrix.append(missing);return;
          }}
          const record=D.records.find(item=>item.step===step&&item.method_key===method.key);
          count+=appendAbVideoCell(matrix,method,record,c,step);
        }});
      }}));
      document.getElementById("generated-title").textContent="方案 A/B 视频对比";
      document.getElementById("count").textContent=`${{pairs.length}} 个方案 · ${{count}} 个已生成结果`;
    }}
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
      if(D.abExperiment){{
        renderAbMatrix(c,steps,matrix);
        renderMetrics(c,steps);
        return;
      }}
      matrix.className="generated-matrix";
      matrix.style.gridTemplateColumns=`88px repeat(${{D.methods.length}},minmax(260px,1fr))`;
      const corner=document.createElement("div");corner.className="matrix-head";
      corner.textContent="Step / Reference";matrix.append(corner);
      D.methods.forEach(method=>{{
        const header=document.createElement("div");header.className="matrix-head method";
        header.textContent=method.label;header.style.color=method.color;
        header.style.borderTopColor=method.color;matrix.append(header);
      }});
      let count=0;
      steps.forEach(step=>{{
        const stepLabel=document.createElement("div");stepLabel.className="step-label";
        stepLabel.textContent=formatStepLabel(step);matrix.append(stepLabel);
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
          label.textContent=`${{record.method_label}} · ${{record.step_kind==="inference"?"inference":"step"}} ${{record.step}}`;
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
      renderMetrics(c,steps);
    }}
    stepFilter.addEventListener("change",render);caseSelect.addEventListener("change",render);
    document.getElementById("play").onclick=()=>videos().forEach(video=>video.play().catch(()=>{{}}));
    document.getElementById("pause").onclick=()=>videos().forEach(video=>video.pause());
    document.getElementById("replay").onclick=()=>videos().forEach(video=>{{video.currentTime=0;video.play().catch(()=>{{}})}});
    render();
    if(D.autoRefresh){{
      const loadedAt=Date.parse(document.lastModified);
      window.setInterval(async()=>{{
        try{{
          const response=await fetch(window.location.href,{{method:"HEAD",cache:"no-store"}});
          const modifiedAt=Date.parse(response.headers.get("Last-Modified")??"");
          const videoIsPlaying=videos().some(video=>!video.paused&&!video.ended);
          if(Number.isFinite(modifiedAt)&&modifiedAt>loadedAt&&!videoIsPlaying){{
            window.location.reload();
          }}
        }}catch{{}}
      }},30000);
    }}
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
                **METHOD_PLOT_STYLES.get(method["key"], DEFAULT_PLOT_STYLE),
                linewidth=2.4,
                markersize=5,
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


def build_metrics_page(
    plots: list[dict[str, Any]], pending_messages: list[str] | None = None
) -> str:
    cards = "".join(
        f"""<article><h2>{escape(plot['label'])}</h2>
        <img src="{escape(plot['image'])}" alt="{escape(plot['label'])}"></article>"""
        for plot in plots
    )
    pending_messages = pending_messages or []
    pending = ""
    if pending_messages:
        items = "".join(f"<li>{escape(message)}</li>" for message in pending_messages)
        pending = f'<section class="pending"><strong>正在接入</strong><ul>{items}</ul></section>'
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
      font-weight:800;text-decoration:none}}.pending{{max-width:1460px;margin:14px auto 0;padding:11px 14px;
      background:#e8f5f4;border:1px solid #9ccfca;border-radius:6px;color:#075d63}}
    .pending ul{{display:flex;flex-wrap:wrap;gap:6px 22px;margin:5px 0 0;padding-left:18px;font-size:12px}}
    main{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
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
  {pending}<main>{cards}</main>
</body>
</html>
"""


def annotate_metrics_index(path: Path, messages: list[str]) -> None:
    """Add an idempotent live-evaluation banner to an externally built plot page."""
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"<!-- xssc-live-start -->.*?<!-- xssc-live-end -->",
        "",
        text,
        flags=re.DOTALL,
    )
    if messages:
        items = "".join(f"<li>{escape(message)}</li>" for message in messages)
        banner = (
            '<!-- xssc-live-start --><section style="margin:12px;padding:12px 14px;'
            'border:1px solid #9ccfca;background:#e8f5f4;color:#075d63;'
            'font-family:Arial,sans-serif"><strong>正在接入</strong><ul style="margin:6px 0 0">'
            f"{items}</ul></section><!-- xssc-live-end -->"
        )
        if "<body>" in text:
            text = text.replace("<body>", f"<body>{banner}", 1)
        else:
            text = banner + text
    path.write_text(text, encoding="utf-8")


def build_merged_metric_plots_from_points(
    config: dict[str, Any],
    point_paths: list[Path],
    output_root: Path,
) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for path in point_paths:
        if path.is_file():
            frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not frame.empty:
        frame["method_key"] = frame["method_key"].map(
            lambda value: METHOD_KEY_ALIASES.get(str(value), str(value))
        )
        frame["method_label"] = frame["method_key"].map(
            lambda key: METHOD_BY_KEY.get(
                str(key),
                {"label": str(key)},
            )["label"]
        )
    plots: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    for metric in config["plot_metrics"]:
        field = metric["field"]
        figure, axis = plt.subplots(figsize=(6.4, 3.6), dpi=140)
        has_data = False
        for method in MERGED_METHODS:
            values: list[tuple[int, float, int]] = []
            if not frame.empty:
                subset = frame[
                    (frame["metric"] == field)
                    & (frame["method_key"] == method["key"])
                ]
                for _, row in subset.iterrows():
                    value = finite_number(row.get("value"))
                    step = finite_number(row.get("step"))
                    count = finite_number(row.get("count"))
                    if value is None or step is None:
                        continue
                    values.append((int(step), value, int(count or 0)))
            if not values:
                continue
            values.sort()
            has_data = True
            axis.plot(
                [value[0] for value in values],
                [value[1] for value in values],
                **METHOD_PLOT_STYLES.get(method["key"], DEFAULT_PLOT_STYLE),
                linewidth=2.4,
                markersize=5,
                color=method["color"],
                label=method["label"],
            )
            for step, value, count in values:
                points.append(
                    {
                        "method_key": method["key"],
                        "method_label": method["label"],
                        "step": step,
                        "metric": field,
                        "metric_label": metric["label"],
                        "value": value,
                        "count": count,
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
        figure.savefig(output_root / filename, bbox_inches="tight")
        plt.close(figure)
        plots.append(
            {
                "field": field,
                "label": metric["label"],
                "image": f"plots/{filename}",
                "has_data": has_data,
            }
        )
    with (output_root.parent / "metric_points.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method_key",
                "method_label",
                "step",
                "metric",
                "metric_label",
                "value",
                "count",
            ],
        )
        writer.writeheader()
        writer.writerows(points)
    return plots


def build_pending_page(title: str, message: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>
body{{margin:0;background:#f3f5f6;color:#172126;font-family:Inter,"Noto Sans SC",Arial,sans-serif}}
main{{max-width:760px;margin:15vh auto;padding:24px;background:#fff;border:1px solid #d6dde0;border-radius:6px}}
h1{{margin:0 0 8px;font-size:20px}}p{{margin:0 0 18px;color:#657278}}a{{color:#006d77;font-weight:800;text-decoration:none}}
</style></head><body><main><h1>{escape(title)}</h1><p>{escape(message)}</p>
<a href="../">返回监控页</a></main></body></html>"""


def build_combined_test_page(
    *,
    title: str,
    subtitle: str,
    tabs: list[dict[str, str]],
) -> str:
    first_src = tabs[0]["href"] if tabs else "about:blank"
    tab_buttons = "".join(
        f"""<button data-src="{escape(tab['href'])}" title="{escape(tab['label'])}">
        <strong>{escape(tab['label'])}</strong><span>{escape(tab['note'])}</span></button>"""
        for tab in tabs
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root{{--bg:#f3f5f6;--surface:#fff;--ink:#172126;--muted:#657278;--line:#d6dde0;
      --accent:#006d77;--warm:#9b3a31}}
    *{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
      font-family:Inter,"Noto Sans SC",Arial,sans-serif}}
    header{{position:sticky;top:0;z-index:10;padding:14px 18px;background:rgba(255,255,255,.98);
      border-bottom:1px solid var(--line)}}
    .top{{display:flex;align-items:baseline;gap:12px;margin-bottom:11px}}
    h1{{margin:0;font-size:20px}}.sub{{color:var(--muted);font-size:13px}}
    a{{margin-left:auto;color:var(--accent);font-weight:800;text-decoration:none}}
    .tabs{{display:flex;gap:8px;overflow-x:auto;padding-bottom:2px}}
    button{{min-width:168px;padding:8px 10px;border:1px solid var(--line);border-radius:6px;
      background:var(--surface);color:var(--ink);text-align:left;cursor:pointer}}
    button strong{{display:block;font-size:13px}}button span{{display:block;margin-top:2px;
      color:var(--muted);font-size:11px;white-space:nowrap}}
    button.active{{border-color:var(--accent);box-shadow:inset 0 3px 0 var(--accent)}}
    iframe{{display:block;width:100%;height:calc(100vh - 105px);border:0;background:var(--surface)}}
    @media(max-width:720px){{.top{{flex-wrap:wrap}}a{{margin-left:0}}iframe{{height:calc(100vh - 145px)}}}}
  </style>
</head>
<body>
  <header><div class="top"><h1>{escape(title)}</h1><span class="sub">{escape(subtitle)}</span>
    <a href="../">返回总览</a></div><div class="tabs">{tab_buttons}</div></header>
  <iframe id="frame" src="{escape(first_src)}" title="{escape(title)}"></iframe>
  <script>
    const frame=document.getElementById("frame");
    const buttons=[...document.querySelectorAll("button[data-src]")];
    function activate(button){{
      buttons.forEach(item=>item.classList.toggle("active",item===button));
      frame.src=button.dataset.src;
    }}
    buttons.forEach(button=>button.addEventListener("click",()=>activate(button)));
    if(buttons.length) activate(buttons[0]);
  </script>
</body>
</html>"""


MERGED_METHODS = [
    {
        "key": "physrvg_test5_lora_on",
        "label": "PhysRVG finetuned DiT + LoRA · reference",
        "color": "#0B6E4F",
    },
    {
        "key": "physrvg_test5_lora_off",
        "label": "PhysRVG finetuned DiT · LoRA OFF · reference",
        "color": "#315C87",
    },
    {"key": "object_only", "label": "Object-only", "color": "#4D4D4D"},
    {
        "key": "wan22_openvid_lora_baseline",
        "label": "Wan2.2 + OpenVid LoRA (No Additional Adapter)",
        "color": "#5B6770",
    },
    {"key": "full_sa", "label": "Full-SA + Object", "color": "#D62728"},
    {
        "key": "full_sa_physrvg_dit",
        "label": "Full-SA + Object (PhysRVG DiT)",
        "color": "#17BECF",
    },
    {
        "key": "full_sa_physrvg_vjepa_loss",
        "label": "PHYRVG-Full-SA + Object (PhysRVG DiT) + V-JEPA Loss",
        "color": "#C44E52",
    },
    {
        "key": "full_sa_physrvg_dit_gpu56",
        "label": "PHYRVG-Full-SA + Object (PhysRVG DiT) · GPU5/6 batch",
        "color": "#2E86AB",
    },
    {"key": "full_sa_no_object", "label": "Full-SA + No-Object", "color": "#FF7F0E"},
    {
        "key": "full_sa_no_object_vjepa_loss",
        "label": "Full-SA + No-Object + V-JEPA Loss",
        "color": "#009E73",
    },
    {
        "key": "full_sa_no_object_xssc_loss_dinov3_movic_step50000",
        "label": "Full-SA + No-Object + xSSC Loss (DINOv3 MOVi-C 50k)",
        "color": "#6F4EAD",
    },
    {
        "key": "full_sa_no_object_cotracker_trajectory_loss",
        "label": "Full-SA + No-Object + CoTracker Trajectory Loss",
        "color": "#A23B72",
    },
    {
        "key": "full_sa_no_object_gt_latent_mask_loss",
        "label": "Full-SA + No-Object + GT Latent-Mask CE",
        "color": "#E69F00",
    },
    {
        "key": "full_sa_object_slot_dedup_xssc50k_xssc_loss_dinov3_movic_step50000",
        "label": "Full-SA + Object + Slot-Dedup (xSSC-50k) + xSSC Loss (DINOv3 MOVi-C 50k)",
        "color": "#D55E00",
    },
    {
        "key": "t_head_pck32_s039_latest3350_top100_no_object_xssc_loss_dinov3_movic_step50000",
        "label": "PCKhead(S39/3350) + No-Object + xSSC Loss (DINOv3 MOVi-C 50k)",
        "color": "#CC79A7",
    },
    {"key": "s_head59", "label": "S-head59 + Object", "color": "#2CA02C"},
    {"key": "t_head70", "label": "T-head70 + Object", "color": "#9467BD"},
    {
        "key": "t_head70_no_object",
        "label": "T-head70 + No-Object",
        "color": "#E377C2",
    },
    {
        "key": "t_head100_lora_pck32_no_object",
        "label": "Motion-head100 (LoRA-PCK32 Top100) + No-Object",
        "color": "#0072B2",
    },
    {
        "key": "full_sa_no_object_pybullet100",
        "label": "Full-SA + No-Object (PyBullet 100%)",
        "color": "#00A6A6",
    },
    {
        "key": "full_sa_no_object_kubric100",
        "label": "Full-SA + No-Object (Kubric 100%)",
        "color": "#F28E2B",
    },
    {
        "key": "t_head70_slot_dedup_merge",
        "label": "T-head70 + Object + Slot-Dedup",
        "color": "#17BECF",
    },
    {
        "key": "slot_dedup_merge",
        "label": "Full-SA + Object + Slot-Dedup",
        "color": "#1F77B4",
    },
    {
        "key": "slot_dedup_merge_xssc_step050000",
        "label": "Full-SA + Object + Slot-Dedup (xSSC-50k)",
        "color": "#8C564B",
    },
    {
        "key": "t_head70_slot_dedup_merge_xssc_step050000",
        "label": "T-head70 + Object + Slot-Dedup (xSSC-50k)",
        "color": "#00B894",
    },
]

METHOD_KEY_ALIASES = {
    "full_sa_resume": "full_sa",
    "s_head59_resume": "s_head59",
}

METHOD_BY_KEY = {method["key"]: method for method in MERGED_METHODS}


def normalized_method(record: dict[str, Any]) -> dict[str, str]:
    key = METHOD_KEY_ALIASES.get(record["method_key"], record["method_key"])
    method = METHOD_BY_KEY.get(key)
    if method is not None:
        return method
    return {
        "key": key,
        "label": str(record.get("method_label", key)),
        "color": "#006d77",
    }


def prefix_case_media(
    cases: list[dict[str, Any]], media_prefix: str
) -> list[dict[str, Any]]:
    merged_cases: list[dict[str, Any]] = []
    for case in cases:
        row = dict(case)
        row["gt"] = f"{media_prefix}/_source/{case['stem']}/gt_49f_30fps.mp4"
        row["context"] = f"{media_prefix}/_source/{case['stem']}/context_8f.mp4"
        merged_cases.append(row)
    return merged_cases


def prefix_video_records(
    records: list[dict[str, Any]], videos_prefix: str
) -> list[dict[str, Any]]:
    prefixed: list[dict[str, Any]] = []
    for record in records:
        method = normalized_method(record)
        row = dict(record)
        row["method_key"] = method["key"]
        row["method_label"] = method["label"]
        row["videos"] = {
            stem: f"{videos_prefix}/{path}"
            for stem, path in record.get("videos", {}).items()
        }
        prefixed.append(row)
    return prefixed


def load_legacy_video_records(
    legacy_watch_root: str | None,
    videos_prefix: str,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not legacy_watch_root:
        return []
    manifest_path = Path(legacy_watch_root).resolve() / "manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = load_json(manifest_path)
    records = manifest.get("records", [])
    if not isinstance(records, list):
        return []
    state_root = manifest_path.parent.parent / "state" / "checkpoints"
    result_roots: dict[tuple[str, int], Path] = {}
    for state_path in state_root.glob("*/step-*.json"):
        if not state_path.is_file():
            continue
        state = load_json(state_path)
        result_root = state.get("result_root")
        if result_root:
            result_roots[(str(state["method_key"]), int(state["step"]))] = Path(
                str(result_root)
            )
    enriched_records: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        result_root = result_roots.get(
            (str(record["method_key"]), int(record["step"]))
        )
        row["metrics"] = (
            load_case_metrics(result_root, cases)
            if result_root is not None
            else {}
        )
        enriched_records.append(row)
    return prefix_video_records(enriched_records, videos_prefix)


def load_reference_records(
    manifest_path: Path,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        return []
    payload = load_json(manifest_path)
    records: list[dict[str, Any]] = []
    for model in payload.get("models", []):
        result_root = Path(model["result_root"]).resolve()
        video_prefix = str(model["video_prefix"]).rstrip("/")
        records.append(
            {
                "method_key": model["method_key"],
                "method_label": model["method_label"],
                "step": int(model["inference_steps"]),
                "step_kind": "inference",
                "checkpoint_dir": model["checkpoint_label"],
                "origin": "fixed-reference",
                "metrics": load_case_metrics(result_root, cases),
                "videos": {
                    case["stem"]: f"{video_prefix}/{case['stem']}.mp4"
                    for case in cases
                    if (result_root / f"{case['stem']}.mp4").is_file()
                },
            }
        )
    return records


def load_physiciq_video_records_from_state(
    state_root: Path,
    cases: list[dict[str, Any]],
    videos_prefix: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((state_root / "physiciq" / "inference").glob("*/step-*.json")):
        if not path.is_file():
            continue
        manifest = load_json(path)
        method = normalized_method(manifest)
        step = int(manifest["step"])
        original_method_key = manifest["method_key"]
        records.append(
            {
                "method_key": method["key"],
                "method_label": method["label"],
                "step": step,
                "checkpoint_dir": manifest.get("checkpoint_dir", ""),
                "origin": manifest.get("origin", "watcher"),
                "metrics": load_case_metrics(Path(manifest["result_root"]), cases),
                "videos": {
                    case["stem"]: (
                        f"{videos_prefix}/{original_method_key}/"
                        f"step-{step:06d}/{case['stem']}.mp4"
                    )
                    for case in cases
                },
            }
        )
    return records


def merge_video_records(
    current_records: list[dict[str, Any]],
    legacy_records: list[dict[str, Any]],
    current_site_prefix: str,
    methods_override: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    deduplicated: dict[tuple[str, int], dict[str, Any]] = {}
    for record in legacy_records + prefix_video_records(
        current_records,
        current_site_prefix,
    ):
        deduplicated[(str(record["method_key"]), int(record["step"]))] = record
    methods = MERGED_METHODS if methods_override is None else methods_override
    method_keys = [method["key"] for method in methods]
    return sorted(
        deduplicated.values(),
        key=lambda row: (
            method_keys.index(row["method_key"])
            if row["method_key"] in method_keys
            else 999,
            int(row.get("step", 0)),
            str(row["method_key"]),
        ),
    )


def format_average_metric(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 10:
        return f"{value:.2f}"
    if magnitude >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def build_average_metrics_page(
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    *,
    page_title: str,
    methods_override: list[dict[str, str]] | None = None,
) -> str:
    expected = len(cases)
    summaries: list[dict[str, Any]] = []
    for record in records:
        summary: dict[str, Any] = {
            "method_key": record["method_key"],
            "method_label": record["method_label"],
            "step": int(record["step"]),
            "step_kind": record.get("step_kind", "training"),
            "metrics": {},
        }
        record_metrics = record.get("metrics", {})
        for spec in CASE_METRIC_SPECS:
            values = [
                float(case_metrics[spec["key"]])
                for case in cases
                if isinstance(
                    (case_metrics := record_metrics.get(case["stem"])),
                    dict,
                )
                and spec["key"] in case_metrics
            ]
            summary["metrics"][spec["key"]] = {
                "count": len(values),
                "mean": sum(values) / len(values) if values else None,
            }
        summaries.append(summary)
    methods = MERGED_METHODS if methods_override is None else methods_override
    method_colors = {method["key"]: method["color"] for method in methods}
    method_order = {
        method["key"]: index for index, method in enumerate(methods)
    }
    metric_directions = {
        spec["key"]: spec["direction"] for spec in CASE_METRIC_SPECS
    }
    header_cells = "".join(
        f"<th>{escape(spec['label'])} <span>{'↓' if spec['direction'] == 'lower' else '↑'}</span></th>"
        for spec in CASE_METRIC_SPECS
    )
    body_rows: list[str] = []
    for row in summaries:
        cells: list[str] = []
        for spec in CASE_METRIC_SPECS:
            stat = row["metrics"][spec["key"]]
            if stat["count"] != expected or stat["mean"] is None:
                cells.append(
                    f'<td class="pending" data-metric="{escape(spec["key"])}" '
                    f'data-complete="0">pending {stat["count"]}/{expected}</td>'
                )
                continue
            value = float(stat["mean"])
            cells.append(
                f'<td data-metric="{escape(spec["key"])}" data-complete="1" '
                f'data-value="{value:.12g}">{format_average_metric(value)}</td>'
            )
        color = method_colors.get(str(row["method_key"]), "#172126")
        body_rows.append(
            f'<tr data-method="{escape(row["method_key"])}" '
            f'data-step="{row["step"]}"><td class="method" '
            f'style="color:{escape(color)}">'
            f'{escape(row["method_label"])}</td><td class="step">'
            f'{"infer " if row["step_kind"] == "inference" else ""}{row["step"]}'
            f'</td>{"".join(cells)}</tr>'
        )
    method_order_json = json.dumps(method_order, ensure_ascii=True)
    metric_directions_json = json.dumps(metric_directions, ensure_ascii=True)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(page_title)}</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f5f6;color:#172126;
font-family:Inter,"Noto Sans SC",Arial,sans-serif}}header{{padding:18px 22px;background:#fff;
border-bottom:1px solid #d6dde0}}h1{{margin:0 0 5px;font-size:20px}}p{{margin:0;color:#657278;
font-size:12px}}a{{color:#006d77;font-weight:800;text-decoration:none}}main{{padding:14px 20px}}
.toolbar{{display:flex;align-items:center;gap:12px;margin-bottom:10px}}.segmented{{display:inline-flex;
padding:2px;border:1px solid #c9d2d6;border-radius:6px;background:#e5ebed}}.segmented button{{height:30px;
padding:0 13px;border:0;border-radius:4px;background:transparent;color:#45545b;font-weight:800;cursor:pointer}}
.segmented button.active{{background:#fff;color:#006d77;box-shadow:0 1px 3px rgba(23,33,38,.16)}}
.view-note{{color:#657278;font-size:12px}}.wrap{{overflow:auto;max-height:calc(100vh - 174px);
border:1px solid #d6dde0;background:#fff}}
table{{width:max-content;min-width:100%;border-collapse:separate;border-spacing:0;
font-size:11px;font-variant-numeric:tabular-nums}}th,td{{height:31px;padding:5px 8px;
border-right:1px solid #e2e7e9;border-bottom:1px solid #e2e7e9;text-align:center;white-space:nowrap}}
thead th{{position:sticky;top:0;z-index:3;background:#e5ebed;color:#45545b}}thead span{{font-size:10px}}
.method{{position:sticky;left:0;z-index:2;min-width:190px;background:#fff;text-align:left;font-weight:850}}
thead .method{{z-index:4;background:#dce4e7}}.step{{position:sticky;left:190px;z-index:2;
min-width:72px;background:#fff;color:#657278}}thead .step{{z-index:4;background:#dce4e7}}
td.best{{background:#dff3e7;color:#075d37;font-weight:900}}td.pending{{color:#98a3a8;font-size:10px}}
.best::before{{content:"★ ";}}tr.group-start td{{border-top:3px solid #aebbc0}}
.back{{display:inline-block;margin-top:12px}}
</style></head><body><header><h1>{escape(page_title)}</h1>
<p>覆盖全部 {expected} 个 case 的均值才参与组内 ★ 最佳值比较；WMReward surprise 越低越好，其余越高越好。</p></header>
<main><div class="toolbar"><div class="segmented" role="group" aria-label="指标表分组方式">
<button type="button" data-view="model" class="active">按模型</button>
<button type="button" data-view="step">按 Step</button></div>
<span class="view-note" id="view-note">同一模型内比较不同训练 step</span></div>
<div class="wrap"><table><thead><tr><th class="method">方法</th><th class="step">Step</th>
{header_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>
<a class="back" href="../">返回总览</a></main><script>
const methodOrder={method_order_json};
const metricDirections={metric_directions_json};
const tbody=document.querySelector("tbody");
const rows=[...tbody.querySelectorAll("tr")];
const buttons=[...document.querySelectorAll("button[data-view]")];
const note=document.getElementById("view-note");
function orderOf(row){{return methodOrder[row.dataset.method]??999;}}
function applyView(view){{
  rows.sort((left,right)=>view==="model"
    ? orderOf(left)-orderOf(right)||Number(left.dataset.step)-Number(right.dataset.step)
    : Number(left.dataset.step)-Number(right.dataset.step)||orderOf(left)-orderOf(right));
  rows.forEach(row=>{{row.classList.remove("group-start");tbody.appendChild(row);}});
  const groups=new Map();
  let previous=null;
  rows.forEach(row=>{{
    const key=view==="model"?row.dataset.method:row.dataset.step;
    if(key!==previous)row.classList.add("group-start");
    previous=key;
    if(!groups.has(key))groups.set(key,[]);
    groups.get(key).push(row);
  }});
  rows.forEach(row=>row.querySelectorAll("td.best").forEach(cell=>cell.classList.remove("best")));
  groups.forEach(groupRows=>{{
    Object.entries(metricDirections).forEach(([metric,direction])=>{{
      const cells=groupRows.map(row=>row.querySelector(`td[data-metric="${{metric}}"]`))
        .filter(cell=>cell&&cell.dataset.complete==="1");
      if(!cells.length)return;
      const values=cells.map(cell=>Number(cell.dataset.value));
      const best=direction==="lower"?Math.min(...values):Math.max(...values);
      cells.filter(cell=>Math.abs(Number(cell.dataset.value)-best)<=1e-9)
        .forEach(cell=>cell.classList.add("best"));
    }});
  }});
  buttons.forEach(button=>button.classList.toggle("active",button.dataset.view===view));
  note.textContent=view==="model"?"同一模型内比较不同训练 step":"同一 step 内比较不同模型";
  localStorage.setItem("averageMetricsView",view);
}}
buttons.forEach(button=>button.addEventListener("click",()=>applyView(button.dataset.view)));
const saved=localStorage.getItem("averageMetricsView");
applyView(saved==="step"?"step":"model");
</script></body></html>"""


def write_unified_videos_page(
    *,
    config: dict[str, Any],
    page_root: Path,
    cases: list[dict[str, Any]],
    current_records: list[dict[str, Any]],
    legacy_records: list[dict[str, Any]],
    current_site_prefix: str,
    page_title: str,
) -> None:
    page_root.mkdir(parents=True, exist_ok=True)
    display_methods = config["methods"] if config.get("ab_experiment") else MERGED_METHODS
    merged_cases = prefix_case_media(cases, f"{current_site_prefix}/media")
    merged_records = merge_video_records(
        current_records,
        legacy_records,
        current_site_prefix,
        methods_override=display_methods,
    )
    (page_root / "index.html").write_text(
        build_videos_page(
            config,
            merged_records,
            merged_cases,
            page_title=page_title,
            methods_override=display_methods,
        ),
        encoding="utf-8",
    )


def build_status(
    config: dict[str, Any],
    manifests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    watch_root = Path(config["paths"]["watch_root"]).resolve()
    discovered = load_configured_checkpoints(config)
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


def build_weight_provenance_section() -> str:
    return """
    <section class="panel provenance"><div class="panel-head"><h2>训练方案、权重与数据</h2>
      <span class="state">15 种方案 · 3 个数据集</span></div>
      <p class="weights-note">所有方案共享同一 Wan + OpenVid LoRA 起点；“from scratch”仅表示不续接本轮实验 checkpoint，并非随机初始化。下列样本数与参数均按当前正式配置和数据索引核对。</p>
      <h3>基础模型与统一训练参数</h3>
      <div class="table-wrap"><table><thead><tr><th>项目</th><th>配置 / 规模</th><th>说明</th></tr></thead><tbody>
        <tr><td>生成底座</td><td><code>Wan-AI-Wan2.2-TI2V-5B</code>（约 5B 参数）</td><td>Wan 2.2 TI2V 主干；初始化后冻结</td></tr>
        <tr><td>初始化 LoRA</td><td><code>openvid_mixed_ctx24_384x672_lora · step-010000</code></td><td>rank/alpha = 32/32，覆盖 300 个模块；合并进 Wan 后卸载并冻结</td></tr>
        <tr><td>本轮适配结构</td><td>30 个 Self-Attention block × 24 heads；LoRA rank/alpha = 32/32</td><td>按方案训练 Object、Full-SA 或指定 Head；精确可训练参数见总表</td></tr>
        <tr><td>输入规格</td><td>49 frames · 8 context · 512×896 · bf16</td><td>xSSC 输入 256，最多 64 个时间步</td></tr>
        <tr><td>优化参数</td><td>LR 1e-4 · WD 0.01 · paged AdamW 8-bit · max grad norm 1.0</td><td>有效 batch 8；通常 20k steps，每 500 steps 保存；数据集消融为 1k steps</td></tr>
      </tbody></table></div>
      <h3>训练数据集规模与混合采样</h3>
      <div class="table-wrap"><table class="dataset-summary"><thead><tr><th>数据集</th><th>当前训练样本</th><th>标准混合占比</th><th>每个混合轮次的期望抽样数</th><th>使用方式</th></tr></thead><tbody>
        <tr><td>PyBullet 0713</td><td class="num">1,617</td><td class="num">30%</td><td class="num">约 50,818</td><td>49 帧 prefix replay；样本较少，带权重复抽样</td></tr>
        <tr><td>Kubric / PhyCo</td><td class="num">114,276</td><td class="num">30%</td><td class="num">约 50,818</td><td>69 帧索引筛选，训练输出 49 帧 replay</td></tr>
        <tr><td>OpenVidHD parquet</td><td class="num">53,500</td><td class="num">40%</td><td class="num">约 67,757</td><td>49 帧视频与文本提示</td></tr>
        <tr class="total-row"><td>标准混合合计</td><td class="num">169,393</td><td class="num">100%</td><td class="num">169,393</td><td><code>WeightedRandomSampler</code> 按来源占比采样</td></tr>
      </tbody></table></div>
      <p class="weights-note weights-footnote">样本数口径：PyBullet/Kubric 为当前确定性 <code>train</code> split，OpenVid 为当前 parquet 行数。PyBullet 100% 与 Kubric 100% 消融仅启用对应数据源，不使用另外两类样本。</p>
      <h3>训练方案总表</h3>
      <div class="table-wrap"><table class="scheme-weights"><thead><tr><th>类别</th><th>训练方案</th><th>训练数据</th><th>Object / xSSC 权重</th><th>本轮适配或额外权重</th><th>可训练参数</th></tr></thead><tbody>
        <tr><td class="category-cell">基线</td><td>Object-only</td><td>标准混合 30/30/40</td><td>xSSC step-026000 + DINOv3 ViT-L/16 + SAM2.1 Hiera-L</td><td>仅 Object 分支 LoRA</td><td class="num">25,458,688</td></tr>
        <tr><td class="category-cell" rowspan="2">Full Self-Attention</td><td>Full-SA + Object</td><td>标准混合 30/30/40</td><td>xSSC step-026000 + DINOv3 ViT-L/16 + SAM2.1 Hiera-L</td><td>全 Self-Attention LoRA + Object 分支</td><td class="num">49,051,648</td></tr>
        <tr><td>Full-SA + No-Object</td><td>标准混合 30/30/40</td><td>不加载</td><td>全 Self-Attention LoRA</td><td class="num">23,592,960</td></tr>
        <tr><td class="category-cell" rowspan="2">辅助损失</td><td>Full-SA + No-Object + V-JEPA Loss</td><td>标准混合 30/30/40</td><td>不加载</td><td>全 Self-Attention LoRA；冻结 V-JEPA2.1 ViT-L/ViT-G 与 Tiny-VAE <code>taew2_2</code>，辅助损失权重 0.01</td><td class="num">23,592,960</td></tr>
        <tr><td>Full-SA + No-Object + xSSC Loss (DINOv3 MOVi-C 50k)</td><td>标准混合 30/30/40</td><td>Tiny-VAE <code>taew2_2</code> + xSSC step-050000 + DINOv3 ViT-L/16 + SAM2.1 Hiera-L（均冻结）</td><td>全 Self-Attention LoRA；Tiny-VAE 解码后的未来帧 slot cosine loss，权重 0.1，并按 scheduler timestep 权重归一化</td><td class="num">23,592,960</td></tr>
        <tr><td class="category-cell" rowspan="4">Head 选择</td><td>S-head59 + Object</td><td>标准混合 30/30/40</td><td>xSSC step-026000 + DINOv3 ViT-L/16 + SAM2.1 Hiera-L</td><td>59 个 S-head LoRA + Object 分支</td><td class="num">34,682,880</td></tr>
        <tr><td>T-head70 + Object</td><td>标准混合 30/30/40</td><td>xSSC step-026000 + DINOv3 ViT-L/16 + SAM2.1 Hiera-L</td><td>70 个 T-head LoRA + Object 分支</td><td class="num">34,863,104</td></tr>
        <tr><td>T-head70 + No-Object</td><td>标准混合 30/30/40</td><td>不加载</td><td>70 个 T-head LoRA</td><td class="num">9,404,416</td></tr>
        <tr><td>Motion-head100 (LoRA-PCK32 Top100) + No-Object</td><td>标准混合 30/30/40</td><td>不加载</td><td>按 Wan + OpenVid LoRA 的 PCK@32 排名选择 Top100 Head</td><td class="num">11,075,584</td></tr>
        <tr><td class="category-cell" rowspan="2">数据集消融</td><td>Full-SA + No-Object (PyBullet 100%)</td><td>PyBullet 100%</td><td>不加载</td><td>全 Self-Attention LoRA；1k steps</td><td class="num">23,592,960</td></tr>
        <tr><td>Full-SA + No-Object (Kubric 100%)</td><td>Kubric 100%</td><td>不加载</td><td>全 Self-Attention LoRA；1k steps</td><td class="num">23,592,960</td></tr>
        <tr><td class="category-cell" rowspan="2">Slot 去重</td><td>T-head70 + Object + Slot-Dedup</td><td>标准混合 30/30/40</td><td>xSSC step-026000 + DINOv3 ViT-L/16 + SAM2.1 Hiera-L</td><td>70 个 T-head LoRA + Object 分支 + Slot-Dedup</td><td class="num">34,863,104</td></tr>
        <tr><td>Full-SA + Object + Slot-Dedup</td><td>标准混合 30/30/40</td><td>xSSC step-026000 + DINOv3 ViT-L/16 + SAM2.1 Hiera-L</td><td>全 Self-Attention LoRA + Object 分支 + Slot-Dedup</td><td class="num">49,051,648</td></tr>
        <tr><td class="category-cell" rowspan="2">xSSC-50k 初始化 / 续训</td><td>Full-SA + Object + Slot-Dedup (xSSC-50k)</td><td>标准混合 30/30/40</td><td>xSSC step-050000 + DINOv3 ViT-L/16 + SAM2.1 Hiera-L</td><td>全 Self-Attention LoRA + Object 分支 + Slot-Dedup</td><td class="num">49,051,648</td></tr>
        <tr><td>T-head70 + Object + Slot-Dedup (xSSC-50k)</td><td>标准混合 30/30/40</td><td>xSSC step-050000 + DINOv3 ViT-L/16 + SAM2.1 Hiera-L</td><td>70 个 T-head LoRA + Object 分支 + Slot-Dedup</td><td class="num">34,863,104</td></tr>
      </tbody></table></div>
      <p class="weights-note weights-footnote">较晚 step 可能从同一方案较早的 <code>training_state.pt</code> 续训，但底座、初始化 LoRA 与可训练参数规模不变。</p>
    </section>"""


def build_watch_index(
    config: dict[str, Any],
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
    weight_provenance_section = (
        "" if config.get("ab_experiment") else build_weight_provenance_section()
    )
    runtime = config["runtime"]
    run_summary = (
        f"test_5 · {int(runtime['context_frames'])} context · "
        f"{int(runtime['num_frames'])} frames · "
        f"{int(runtime['height'])}×{int(runtime['width'])} · "
        f"{int(runtime['num_inference_steps'])} denoising steps"
    )
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
    .provenance{{margin-bottom:18px}}.provenance h3{{margin:15px 0 7px;font-size:14px}}
    .table-wrap{{overflow-x:auto}}.scheme-weights{{min-width:1420px}}.dataset-summary{{min-width:850px}}
    .category-cell{{color:var(--accent);font-weight:800;vertical-align:top;white-space:nowrap}}
    .num{{font-variant-numeric:tabular-nums;white-space:nowrap}}.total-row td{{font-weight:800}}
    code{{font-size:12px}}
    .weights-note{{margin:0;color:var(--muted);font-size:13px;line-height:1.5}}
    .weights-footnote{{margin-top:10px}}
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
    <p>{escape(run_summary)}</p></header>
  <main>
    {weight_provenance_section}
    <div class="links">
      <a class="link" href="videos/"><strong>Checkpoint 视频</strong>
        <span>按方法、训练 step 和 case 查看生成结果</span></a>
      <a class="link" href="metrics/"><strong>指标曲线</strong>
        <span>每个指标独立子图，比较不同训练方法</span></a>
      <a class="link" href="physiciq-videos/"><strong>PhysicIQ case</strong>
        <span>67 个 case 的 checkpoint 视频对比</span></a>
      <a class="link" href="../physiciq-metrics/"><strong>PhysicIQ 指标曲线</strong>
        <span>每个新 checkpoint 的完整 Physics-IQ 评测</span></a>
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
    *,
    test_cases: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    phys_cases: list[dict[str, Any]],
    phys_records: list[dict[str, Any]],
) -> None:
    hub_root = Path(config["paths"]["master_hub_root"]).resolve()
    watch_root = Path(config["paths"]["watch_root"]).resolve()
    baseline_gallery = Path(config["paths"]["baseline_gallery_root"]).resolve()
    hub_root.mkdir(parents=True, exist_ok=True)
    build_project_info_page(config, hub_root / "project-info")
    link_directory(watch_root / "site" / "videos", hub_root / "gallery")
    phys_gallery = watch_root / "site" / "physiciq-videos"
    if phys_gallery.is_dir():
        link_directory(phys_gallery, hub_root / "physiciq-gallery")
    link_directory(baseline_gallery, hub_root / "initial-gallery")
    link_directory(watch_root / "site", hub_root / "checkpoint-watch")
    legacy_watch_root = config["paths"].get("legacy_watch_root")
    legacy_checkpoint_count = 0
    legacy_physiciq_count = 0
    if legacy_watch_root and Path(legacy_watch_root).is_dir():
        legacy_site = Path(legacy_watch_root).resolve()
        legacy_state = legacy_site.parent / "state"
        link_directory(legacy_site, hub_root / "legacy-checkpoint-watch")
        if (legacy_site / "videos").is_dir():
            link_directory(legacy_site / "videos", hub_root / "history-gallery")
        if (legacy_site / "metrics").is_dir():
            link_directory(legacy_site / "metrics", hub_root / "history-metrics")
        if (legacy_site / "physiciq-videos").is_dir():
            link_directory(
                legacy_site / "physiciq-videos",
                hub_root / "history-physiciq-gallery",
            )
        legacy_checkpoint_count = len(
            list((legacy_state / "checkpoints").glob("*/step-*.json"))
        )
        legacy_physiciq_count = len(
            list((legacy_state / "physiciq" / "inference").glob("*/step-*.json"))
        )
    legacy_physiciq_metrics_root = config["paths"].get(
        "legacy_physiciq_metrics_root"
    )
    if (
        legacy_physiciq_metrics_root
        and Path(legacy_physiciq_metrics_root).is_dir()
    ):
        link_directory(
            Path(legacy_physiciq_metrics_root),
            hub_root / "history-physiciq-metrics",
        )
    physrvg_physiciq_portal_root = config["paths"].get(
        "physrvg_physiciq_lora_portal_root"
    )
    if (
        physrvg_physiciq_portal_root
        and Path(physrvg_physiciq_portal_root).is_dir()
    ):
        link_directory(
            Path(physrvg_physiciq_portal_root),
            hub_root / "physrvg-physiciq-lora-ablation",
        )
    test5_metric_total = len(config["metrics"]["cpu"]) + len(config["metrics"]["gpu"])
    test5_live_messages = []
    for record in test_records:
        metric_done = metric_done_count(
            watch_root, str(record["method_key"]), int(record["step"])
        )
        if (
            record.get("origin") == "watcher-live"
            or str(record["method_key"]).endswith(("pybullet100", "kubric100"))
        ) and metric_done < test5_metric_total:
            test5_live_messages.append(
                f"{record['method_label']} · step {record['step']} · "
                f"生成 {len(record.get('videos', {}))}/{len(test_cases)}，"
                f"指标 {metric_done}/{test5_metric_total}"
            )
    test5_metric_root = hub_root / "test5-metrics"
    test5_plots = build_merged_metric_plots_from_points(
        config,
        [
            watch_root / "site" / "metrics" / "metric_points.csv",
            hub_root / "history-metrics" / "metric_points.csv",
        ],
        test5_metric_root / "plots",
    )
    test5_metric_root.mkdir(parents=True, exist_ok=True)
    (test5_metric_root / "index.html").write_text(
        build_metrics_page(test5_plots, test5_live_messages),
        encoding="utf-8",
    )
    legacy_test_records = load_legacy_video_records(
        legacy_watch_root,
        "../history-gallery",
        test_cases,
    )
    test5_reference_records = load_reference_records(
        hub_root / "physrvg-test5-lora-ablation" / "reference_models.json",
        test_cases,
    )
    legacy_test_records.extend(test5_reference_records)
    site_titles = config.get("site_titles", {})
    test5_page_root = hub_root / "test5"
    test5_media_prefix = "../gallery"
    if config.get("ab_experiment"):
        test5_page_root.mkdir(parents=True, exist_ok=True)
        link_directory(
            watch_root / "site" / "videos" / "media",
            test5_page_root / "media",
        )
        test5_media_prefix = "."
    write_unified_videos_page(
        config=config,
        page_root=test5_page_root,
        cases=test_cases,
        current_records=test_records,
        legacy_records=legacy_test_records,
        current_site_prefix=test5_media_prefix,
        page_title=str(
            site_titles.get("test5", "test_5 · 全 checkpoint 合并对比")
        ),
    )
    test5_average_root = hub_root / "test5-average-metrics"
    test5_average_root.mkdir(parents=True, exist_ok=True)
    (test5_average_root / "index.html").write_text(
        build_average_metrics_page(
            merge_video_records(
                test_records,
                legacy_test_records,
                test5_media_prefix,
                methods_override=(
                    config["methods"] if config.get("ab_experiment") else None
                ),
            ),
            test_cases,
            page_title=str(
                site_titles.get(
                    "test5_average_metrics",
                    "test_5 · 全 case 平均指标",
                )
            ),
            methods_override=(
                config["methods"] if config.get("ab_experiment") else None
            ),
        ),
        encoding="utf-8",
    )
    legacy_state_root = (
        Path(legacy_watch_root).resolve().parent / "state"
        if legacy_watch_root
        else None
    )
    legacy_phys_records = (
        load_physiciq_video_records_from_state(
            legacy_state_root,
            phys_cases,
            "../history-physiciq-gallery/media",
        )
        if legacy_state_root and legacy_state_root.is_dir()
        else []
    )
    physiciq_reference_records = load_reference_records(
        hub_root / "physrvg-physiciq-lora-ablation" / "reference_models.json",
        phys_cases,
    )
    legacy_phys_records.extend(physiciq_reference_records)
    solid_mechanics_cases: list[dict[str, Any]] = []
    if phys_cases:
        write_unified_videos_page(
            config=config,
            page_root=hub_root / "physiciq",
            cases=phys_cases,
            current_records=phys_records,
            legacy_records=legacy_phys_records,
            current_site_prefix="../physiciq-gallery",
            page_title="PhysicIQ 67-case · 全 checkpoint 合并对比",
        )
        phys_average_root = hub_root / "physiciq-average-metrics"
        phys_average_root.mkdir(parents=True, exist_ok=True)
        merged_phys_records = merge_video_records(
            phys_records,
            legacy_phys_records,
            "../physiciq-gallery",
        )
        phys_average_html = build_average_metrics_page(
            merged_phys_records,
            phys_cases,
            page_title="PhysicIQ · 67-case 平均指标",
        ).replace(
            "</header>",
            '<p><a href="../physiciq-solid-mechanics/">Solid Mechanics · 39-case 视频与逐 case 指标</a> · '
            '<a href="solid-mechanics/">39-case 平均指标表</a></p></header>',
        )
        (phys_average_root / "index.html").write_text(
            phys_average_html,
            encoding="utf-8",
        )
        solid_mechanics_cases = [
            case
            for case in phys_cases
            if "_Solid_Mechanics_" in json.dumps(case, ensure_ascii=False)
        ]
        solid_mechanics_video_root = hub_root / "physiciq-solid-mechanics"
        write_unified_videos_page(
            config=config,
            page_root=solid_mechanics_video_root,
            cases=solid_mechanics_cases,
            current_records=phys_records,
            legacy_records=legacy_phys_records,
            current_site_prefix="../physiciq-gallery",
            page_title=(
                f"PhysicIQ · Solid Mechanics · {len(solid_mechanics_cases)}-case 子集"
            ),
        )
        solid_mechanics_video_path = solid_mechanics_video_root / "index.html"
        solid_mechanics_video_path.write_text(
            solid_mechanics_video_path.read_text(encoding="utf-8").replace(
                '<a href="../">返回监控页</a>',
                '<a href="../">返回 8844 总览</a>'
                '<a href="../physiciq-average-metrics/solid-mechanics/">39-case 平均指标表</a>'
                '<a href="../physiciq/">返回 PhysicIQ 67-case</a>',
            ),
            encoding="utf-8",
        )
        solid_mechanics_root = phys_average_root / "solid-mechanics"
        solid_mechanics_root.mkdir(parents=True, exist_ok=True)
        solid_mechanics_html = build_average_metrics_page(
            merged_phys_records,
            solid_mechanics_cases,
            page_title=(
                f"PhysicIQ · Solid Mechanics · {len(solid_mechanics_cases)}-case 平均指标"
            ),
        ).replace(
            "</header>",
            '<p><a href="../../physiciq-solid-mechanics/">查看 39-case 视频与逐 case 指标</a> · '
            '<a href="../">返回 PhysicIQ 67-case 总平均</a></p></header>',
        )
        (solid_mechanics_root / "index.html").write_text(
            solid_mechanics_html,
            encoding="utf-8",
        )
    else:
        (hub_root / "physiciq").mkdir(parents=True, exist_ok=True)
        (hub_root / "physiciq" / "index.html").write_text(
            build_pending_page(
                "PhysicIQ 67-case · 全 checkpoint 合并对比",
                "正在等待 PhysicIQ case 信息生成。",
            ),
            encoding="utf-8",
        )
    total_inferred = sum(row["inferred"] for row in status)
    total_discovered = sum(row["discovered"] for row in status)
    total_metrics = sum(row["metric_done"] for row in status)
    expected_metrics = sum(row["metric_total"] for row in status)
    physrvg_test5_entry = ""
    physrvg_test5_root = hub_root / "physrvg-test5-lora-ablation"
    if (physrvg_test5_root / "index.html").is_file():
        physrvg_test5_entry = f"""
    <section class="entry"><div><h2>PhysRVG test_5 · LoRA ON/OFF</h2>
      <div class="meta">20-case 固定参考模型；40 denoising steps、30 FPS、49 帧、8 条件帧；逐 case 指标差异与双视频同步对比</div>
      <a href="physrvg-test5-lora-ablation/">进入 LoRA 消融对比</a>
      <a href="test5/">进入全模型合并视图</a></div>
      <div class="status">固定参考<strong>{len(test5_reference_records)}/2 组结果</strong><small>LoRA 是唯一模型变量</small></div>
    </section>"""
    physrvg_physiciq_entry = ""
    physrvg_physiciq_root = hub_root / "physrvg-physiciq-lora-ablation"
    if (physrvg_physiciq_root / "index.html").is_file():
        physrvg_physiciq_entry = f"""
    <section class="entry"><div><h2>PhysRVG PhysicIQ 67-case · LoRA ON/OFF</h2>
      <div class="meta">原始 PhysicIQ 固定参考实验；逐 case 指标差异、双视频同步对比，并已加入 PhysicIQ 全模型合并页</div>
      <a href="physrvg-physiciq-lora-ablation/">进入 LoRA 消融对比</a>
      <a href="physiciq/">进入全模型合并视图</a>
      <a href="physiciq-average-metrics/">查看 67-case 平均指标</a></div>
      <div class="status">固定参考<strong>{len(physiciq_reference_records)}/2 组结果</strong><small>67 case · 40 inference steps</small></div>
    </section>"""
    physiciq_entry = ""
    if config.get("physiciq", {}).get("enabled"):
        phys_config = config["physiciq"]
        leaf_path = Path(phys_config["leaf_folders"]).resolve()
        plot_root = leaf_path.parent / "_metric_plots" / leaf_path.stem
        global_plot_roots: list[Path] = []
        for leaf_value in phys_config.get("additional_leaf_folders", []):
            additional_leaf = Path(leaf_value).resolve()
            global_plot_roots.append(
                additional_leaf.parent / "_metric_plots" / additional_leaf.stem
            )
        preferred_plot_root = next(
            (
                root
                for root in global_plot_roots
                if (root / "index.html").is_file()
            ),
            plot_root if (plot_root / "index.html").is_file() else None,
        )
        phys_live_messages = []
        if phys_status is not None:
            phys_live_messages = [
                f"{row['method_label']} · step {row['step']} · "
                f"生成 {row['generated']}/{row['expected_cases']}，完整指标 pending"
                for row in phys_status["rows"]
                if row["generated"] > 0 and not row["manifest_done"]
            ]
        configured_steps = phys_config.get("trigger_steps", "all")
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
        if preferred_plot_root is not None:
            annotate_metrics_index(
                preferred_plot_root / "index.html", phys_live_messages
            )
            link_directory(preferred_plot_root, hub_root / "physiciq-metrics")
            action = (
                '<a href="checkpoint-watch/#physiciq">监控入口</a>'
                '<a href="physiciq/">Case 合并对比</a>'
                '<a href="physiciq-metrics/">指标曲线</a>'
                '<a href="physiciq-average-metrics/">67-case 平均指标表</a>'
                '<a href="physrvg-physiciq-lora-ablation/">PhysRVG LoRA ON/OFF</a>'
            )
        elif (
            legacy_physiciq_metrics_root
            and Path(legacy_physiciq_metrics_root).is_dir()
            and (Path(legacy_physiciq_metrics_root) / "index.html").is_file()
        ):
            link_directory(
                Path(legacy_physiciq_metrics_root),
                hub_root / "physiciq-metrics",
            )
            action = (
                '<a href="checkpoint-watch/#physiciq">监控入口</a>'
                '<a href="physiciq/">Case 合并对比</a>'
                '<a href="physiciq-metrics/">指标曲线</a>'
                '<a href="physiciq-average-metrics/">67-case 平均指标表</a>'
                '<a href="physrvg-physiciq-lora-ablation/">PhysRVG LoRA ON/OFF</a>'
            )
        else:
            pending_metrics = watch_root / "site" / "physiciq-metrics"
            pending_metrics.mkdir(parents=True, exist_ok=True)
            (pending_metrics / "index.html").write_text(
                build_pending_page(
                    "PhysicIQ 指标曲线",
                    "首个 checkpoint 的完整指标尚未计算完成。",
                ),
                encoding="utf-8",
            )
            link_directory(pending_metrics, hub_root / "physiciq-metrics")
            action = (
                '<a href="checkpoint-watch/#physiciq">监控入口</a>'
                '<a href="physiciq/">Case 合并对比</a>'
                '<a href="physiciq-metrics/">指标曲线</a>'
                '<a href="physiciq-average-metrics/">67-case 平均指标表</a>'
                '<a href="physrvg-physiciq-lora-ablation/">PhysRVG LoRA ON/OFF</a>'
            )
        step_text = (
            "每个新 checkpoint"
            if configured_steps == "all"
            else " / ".join(str(int(step)) for step in configured_steps)
        )
        method_text = "、".join(
            method["label"]
            for method in config["methods"]
            if method["key"] in phys_config["method_keys"]
        )
        physiciq_entry = f"""
    <section class="entry"><div><h2>PhysicIQ 67-case</h2>
      <div class="meta">{escape(method_text)} · 同一训练方案的不同 checkpoint step 合并展示 · 40 denoising steps</div>
      {action}</div><div class="status">{generated_text}<strong>{metric_text}</strong><em>{partial_text}</em><small>step {step_text}</small></div>
    </section>"""
    solid_mechanics_entry = ""
    if solid_mechanics_cases:
        solid_mechanics_entry = f"""
    <section class="entry"><div><h2>PhysicIQ · Solid Mechanics 子集</h2>
      <div class="meta">从 PhysicIQ 67-case 中筛选 Solid Mechanics；按 case 查看 GT、8 条件帧、全部模型结果与逐项指标</div>
      <a href="physiciq-solid-mechanics/">39-case 视频与逐 case 指标</a>
      <a href="physiciq-average-metrics/solid-mechanics/">39-case 平均指标表</a></div>
      <div class="status">固定子集<strong>{len(solid_mechanics_cases)} cases</strong><small>复用现有视频与指标</small></div>
    </section>"""
    physrvg_worst_case_entry = ""
    if phys_cases and (hub_root / "physiciq" / "index.html").is_file():
        worst_case_root = hub_root / "physiciq-vs-physrvg-worst-cases"
        build_physiciq_physrvg_worst_case_dashboard(
            hub_root / "physiciq" / "index.html",
            worst_case_root,
        )
        top3_compare_root = hub_root / "physiciq-top3-vs-physrvg-all-cases"
        build_physiciq_top3_physrvg_all_cases_dashboard(
            hub_root / "physiciq" / "index.html",
            top3_compare_root,
        )
        top10_video_root = hub_root / "physiciq-top3-vs-physrvg-top10-videos"
        build_physiciq_top3_physrvg_top10_videos(
            hub_root / "physiciq" / "index.html",
            top10_video_root,
        )
        physrvg_worst_case_entry = f"""
    <section class="entry"><div><h2>PhysicIQ · 相对 PhysRVG 最大劣势 case</h2>
      <div class="meta">覆盖全部已有方案；按 VideoPhy2 PC raw、Cosmos Reason、Physics-IQ ctx/no ctx
      的原始分差筛选最大劣势，四路视频同步核查</div>
      <a href="physiciq-vs-physrvg-worst-cases/">进入回归审计页</a>
      <a href="physiciq-top3-vs-physrvg-top10-videos/">四指标 Top 10 · 纯视频</a>
      <a href="physiciq-top3-vs-physrvg-all-cases/">综合 Top 3 × PhysRVG · 67 case</a></div>
      <div class="status">动态汇总<strong>全部方案</strong><small>PhysRVG OFF / +LoRA 可切换</small></div>
    </section>"""
    step40_ab_entry = ""
    if (hub_root / "test5-step40-object-identity-count-ab").exists():
        step40_ab_entry = """
    <!-- TEST5_STEP40_OBJECT_COUNT_AB_START -->
    <section class="entry"><div><h2>test_5 · 40-step Object Identity/Count A/B</h2>
      <div class="meta">18 个 step-500 权重；原始 prompt 与 identity/count 约束 prompt 的 40-step 配对生成和完整指标</div>
      <a href="test5-step40-object-identity-count-ab/">Case 视频对比</a>
      <a href="test5-step40-object-identity-count-ab-average-metrics/">全 case 平均指标</a>
      <a href="test5-step40-object-identity-count-ab-status/">流水线状态</a></div>
      <div class="status">GPU 7<strong>36 组 · 720 视频</strong><small>推理完成后自动评测</small></div>
    </section>
    <!-- TEST5_STEP40_OBJECT_COUNT_AB_END -->"""
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
    <p>{escape('、'.join(method['label'] for method in config['methods']))}</p></header>
  <main><div class="section-title">项目</div><div class="entries">
    <section class="entry"><div><h2>项目与权重溯源</h2>
      <div class="meta">完整方法流程、上游训练数据、训练模块与参数量、每个checkpoint配置链和Head分类依据</div>
      <a href="project-info/">查看项目信息</a></div>
      <div class="status">自动更新<strong>配置可追溯</strong><small>含Head证据入口</small></div>
    </section>
  </div><div class="section-title" style="margin-top:18px">训练与推理</div><div class="entries">
    <section class="entry"><div><h2>Checkpoint 自动评测</h2>
      <div class="meta">自动发现权重、生成 test_5、计算完整指标并绘制训练 step 曲线</div>
      <a href="checkpoint-watch/">进入自动评测</a></div>
      <div class="status">持续监听<strong>推理 {total_inferred}/{total_discovered} · 指标 {total_metrics}/{expected_metrics}</strong></div>
    </section>
    <section class="entry"><div><h2>test_5 · 全 checkpoint 合并</h2>
      <div class="meta">同一个 test_5 入口内按方法和 step 展示所有已完成 checkpoint</div>
      <a href="test5/">进入合并视图</a>
      <a href="test5-metrics/">指标曲线</a>
      <a href="test5-average-metrics/">全 case 平均指标表</a></div>
      <div class="status">全部 step<strong>{total_inferred + legacy_checkpoint_count} 组结果</strong><small>持续增量更新</small></div>
    </section>
    {step40_ab_entry}
    {physrvg_test5_entry}
    {physrvg_physiciq_entry}
    {physiciq_entry}
    {solid_mechanics_entry}
    {physrvg_worst_case_entry}
    <section class="entry"><div><h2>30-case train validation · 方法对比</h2>
      <div class="meta">固定 PyBullet train 30-case；当前三种方法 resume/latest 与标准 18-method inventory 的其他最新权重，包含视频同步对比和 validation loss</div>
      <a href="train-validation-30cases/">进入 30-case validation</a>
      <a href="cotracker-trajectory-overlay-5cases/">CoTracker 轨迹 Overlay · 5 case</a></div>
      <div class="status">固定验证集<strong>21 组方法权重</strong><small>视频与 val loss 持续补全</small></div>
    </section>
    <section class="entry"><div><h2>初始四方案 case 对比</h2>
      <div class="meta">Object-only、Full-SA、S-head59、T-head70 的早期固定 case 对照页面</div>
      <a href="initial-gallery/">查看初始 case</a></div>
      <div class="status">只读归档<strong>固定对照</strong></div>
    </section>
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
    completed_manifests = load_manifests(watch_root)
    manifests = load_live_test_manifests(config, completed_manifests)
    cases = read_inputs(Path(config["paths"]["input_list"]))
    records = build_video_media(config, manifests, cases)
    videos_root = site_root / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)
    (videos_root / "index.html").write_text(
        build_videos_page(config, records, cases),
        encoding="utf-8",
    )
    phys_config = config.get("physiciq", {})
    phys_cases: list[dict[str, Any]] = []
    phys_records: list[dict[str, Any]] = []
    if phys_config.get("enabled"):
        phys_cases = read_inputs(Path(phys_config["input_list"]))
        completed_phys_manifests = load_physiciq_manifests(watch_root)
        phys_manifests = load_live_physiciq_manifests(
            config, completed_phys_manifests
        )
        if phys_manifests:
            phys_records = build_video_media(
                config,
                phys_manifests,
                phys_cases,
                site_name="physiciq-videos",
            )
            phys_videos_root = site_root / "physiciq-videos"
            phys_videos_root.mkdir(parents=True, exist_ok=True)
            (phys_videos_root / "index.html").write_text(
                build_videos_page(
                    config,
                    phys_records,
                    phys_cases,
                    page_title="Checkpoint · PhysicIQ 67-case",
                ),
                encoding="utf-8",
            )
        else:
            phys_videos_root = site_root / "physiciq-videos"
            phys_videos_root.mkdir(parents=True, exist_ok=True)
            (phys_videos_root / "index.html").write_text(
                build_pending_page(
                    "Checkpoint · PhysicIQ 67-case",
                    "正在等待三个训练任务落下首个 checkpoint。",
                ),
                encoding="utf-8",
            )
    plots = build_metric_plots(config, completed_manifests)
    metrics_root = site_root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    metric_total = len(config["metrics"]["cpu"]) + len(config["metrics"]["gpu"])
    live_messages = []
    for record in records:
        metric_done = metric_done_count(
            watch_root, str(record["method_key"]), int(record["step"])
        )
        if (
            record.get("origin") == "watcher-live"
            or str(record["method_key"]).endswith(("pybullet100", "kubric100"))
        ) and metric_done < metric_total:
            live_messages.append(
                f"{record['method_label']} · step {record['step']} · "
                f"生成 {len(record.get('videos', {}))}/{len(cases)}，"
                f"指标 {metric_done}/{metric_total}"
            )
    (metrics_root / "index.html").write_text(
        build_metrics_page(plots, live_messages),
        encoding="utf-8",
    )
    status = build_status(config, completed_manifests)
    phys_status = build_physiciq_status(config)
    pending = (watch_root / "state" / "inference.pending").is_file()
    (site_root / "index.html").write_text(
        build_watch_index(config, status, pending, phys_status),
        encoding="utf-8",
    )
    manifest = {
        "num_cases": len(cases),
        "num_inference_results": len(completed_manifests),
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
    build_master_hub(
        config,
        status,
        phys_status,
        test_cases=cases,
        test_records=records,
        phys_cases=phys_cases,
        phys_records=phys_records,
    )
    print(site_root / "index.html")


if __name__ == "__main__":
    main()
