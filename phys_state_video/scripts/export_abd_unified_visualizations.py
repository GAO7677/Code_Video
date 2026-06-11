#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


ABD_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/ABD_test")
DEFAULT_OUTPUT_ROOT = ABD_ROOT / "_viz_v2"
DEFAULT_PORT = 18885

GROUP_TITLES = {
    "A": "A 组: PDI-Bench",
    "B": "B 组: Dataset_physV",
    "D": "D 组: Physics-IQ",
}

METRIC_SPECS = [
    {
        "key": "official_pdi",
        "label": "Official PDI ↓",
        "arrow": "↓",
        "desc": "几何与物理一致性审计分数，越低越好。",
    },
    {
        "key": "wmreward_surprise",
        "label": "WMReward Surprise ↓",
        "arrow": "↓",
        "desc": "短窗未来预测惊讶度，越低越好。",
    },
    {
        "key": "vjepa_relraw",
        "label": "V-JEPA RelRaw ↓",
        "arrow": "↓",
        "desc": "未来时序关系原始误差，越低越好。",
    },
    {
        "key": "vjepa_deltarel",
        "label": "V-JEPA DeltaRel ↓",
        "arrow": "↓",
        "desc": "未来变化关系误差，越低越好。",
    },
    {
        "key": "vjepa_deltaprof",
        "label": "V-JEPA DeltaProf ↓",
        "arrow": "↓",
        "desc": "未来变化轮廓误差，越低越好。",
    },
    {
        "key": "cosmos_reason1",
        "label": "Cosmos Reason1 ↑",
        "arrow": "↑",
        "desc": "LLM 物理常识打分，越高越好。",
    },
    {
        "key": "videophy2_pc",
        "label": "VideoPhy-2 PC ↑",
        "arrow": "↑",
        "desc": "VideoPhy-2 物理一致性分数，越高越好。",
    },
    {
        "key": "videophy2_sa",
        "label": "VideoPhy-2 SA ↑",
        "arrow": "↑",
        "desc": "VideoPhy-2 语义/动作分数，越高越好。",
    },
    {
        "key": "videophy2_joint",
        "label": "VideoPhy-2 Joint ↑",
        "arrow": "↑",
        "desc": "当 SA>=4 且 PC>=4 时记为 1，越高越好。",
    },
]

TRAINING_RUN_SPECS = [
    {
        "id": "baseline_v1",
        "label": "baseline_v1 显式状态基线",
        "desc": "显式 future state 作为主条件。",
        "run_root": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_baseline_v1"),
        "best_page": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_baseline_v1/viz/trained_cases_v1"),
        "timeline_page": None,
        "predictor_log": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_baseline_v1/logs/predictor_train.log"),
        "adapter_log": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_baseline_v1/logs/adapter_train.log"),
    },
    {
        "id": "latent_v1",
        "label": "latent_v1 隐式 latent 条件版",
        "desc": "future latent tokens 作为主条件，同时保留显式 head 监督。",
        "run_root": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v1"),
        "best_page": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v1/viz/training_ckpts/cases/adapter_best"),
        "timeline_page": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v1/viz/training_ckpts"),
        "predictor_log": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v1/logs/predictor_train.log"),
        "adapter_log": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v1/logs/adapter_train.log"),
    },
    {
        "id": "latent_v2",
        "label": "latent_v2 latent-only 生成版",
        "desc": "显式 state 主要做监督，视频生成主条件切换为 future latent tokens。",
        "run_root": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v2"),
        "best_page": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v2/viz/training_ckpts/cases/adapter_best"),
        "timeline_page": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v2/viz/training_ckpts"),
        "predictor_log": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v2/logs/predictor_train.log"),
        "adapter_log": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v2/logs/adapter_train.log"),
    },
    {
        "id": "visualctx_v3",
        "label": "visualctx_v3 视觉上下文 predictor-only",
        "desc": "predictor 主输入改为 context frames，页面展示 predictor case 与显式条件图。",
        "run_root": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_visualctx_predictor_v3_gpu0123"),
        "best_page": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_visualctx_predictor_v3_gpu0123/viz/predictor_cases_v1"),
        "timeline_page": None,
        "predictor_log": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_visualctx_predictor_v3_gpu0123/logs/predictor_train.log"),
        "adapter_log": None,
    },
    {
        "id": "tailquery_multictx_converge",
        "label": "tailquery_multictx_converge",
        "desc": "Wan state predictor v2 多 context 长度训练 + boundary 强化版。",
        "run_root": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v2/industrial_s1_scale2_wan_state_v2_tailquery_multictx_converge_gpu0123_20260606"),
        "best_page": None,
        "timeline_page": None,
        "predictor_log": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v2/industrial_s1_scale2_wan_state_v2_tailquery_multictx_converge_gpu0123_20260606/logs/predictor_train.log"),
        "adapter_log": None,
        "wandb_summary": Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v2/industrial_s1_scale2_wan_state_v2_tailquery_multictx_converge_gpu0123_20260606/wandb/wandb/run-20260606_121809-fnf5frld/files/wandb-summary.json"),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export unified ABD visualizations with baseline and training pages.")
    parser.add_argument("--abd-root", type=Path, default=ABD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-serve", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


ORIGINAL_JSON_CACHE: dict[str, dict[str, Any]] = {}


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_server(output_dir: Path, port: int) -> int:
    log_path = output_dir / f"http_{port}.log"
    pid_path = output_dir / f"http_{port}.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
            if is_port_open(port):
                return pid
        except Exception:
            pid_path.unlink(missing_ok=True)

    with open(log_path, "wb") as handle:
        proc = subprocess.Popen(
            ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1"],
            cwd=str(output_dir),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    time.sleep(1.0)
    return proc.pid


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src, target_is_directory=src.is_dir())


def copy_or_symlink_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def choose_case_keys(group_root: Path, group_name: str) -> list[str]:
    meta_dir = group_root / "_meta"
    gt_dir = group_root / "GT"
    subset_path = meta_dir / "report_subset_selected_cases.json"
    if group_name == "B" and subset_path.is_file():
        payload = load_json(subset_path)
        return [str(row["case_key"]) for row in payload.get("cases", [])]
    if subset_path.is_file():
        subset = load_json(subset_path)
        gt_payloads = [load_json(path) for path in sorted(gt_dir.glob("*.json"))]
        keys: list[str] = []
        for row in subset.get("cases", []):
            clip_name = str(row.get("clip_name", ""))
            task = str(row.get("task", row.get("theme", "")))
            for payload in gt_payloads:
                if clip_name and clip_name not in {payload.get("clip_name"), payload.get("sample_name")}:
                    continue
                if task and task not in {payload.get("category"), payload.get("theme")}:
                    if group_name == "D":
                        # D subset themes are semantic labels, not category names.
                        pass
                    elif group_name == "A":
                        pass
                keys.append(str(payload["case_key"]))
                break
        if keys:
            return keys

    rows = []
    for json_path in sorted(gt_dir.glob("*.json")):
        payload = load_json(json_path)
        rows.append((str(payload["category"]), str(payload["case_key"])))
    by_category: dict[str, list[str]] = {}
    for category, case_key in rows:
        by_category.setdefault(category, []).append(case_key)
    selected: list[str] = []
    for category in sorted(by_category):
        selected.append(by_category[category][0])
    return selected


def load_original_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    original_json = payload.get("original_json")
    if not original_json:
        return None
    path = str(original_json)
    if path in ORIGINAL_JSON_CACHE:
        return ORIGINAL_JSON_CACHE[path]
    try:
        loaded = load_json(Path(path))
    except Exception:
        loaded = None
    if loaded is not None:
        ORIGINAL_JSON_CACHE[path] = loaded
    return loaded


def resolve_metric_value(payload: dict[str, Any], metric_key: str) -> float | None:
    original_payload = load_original_payload(payload) or {}
    metric_results = payload.get("metric_results") or {}
    if not metric_results:
        metric_results = original_payload.get("metric_results") or {}
    metric_summary = payload.get("metric_summary") or {}
    if not metric_summary:
        metric_summary = original_payload.get("metric_summary") or {}
    if metric_key == "official_pdi":
        value = ((metric_results.get("official_pdi") or {}).get("pdi_score"))
        if value is None:
            value = (
                payload.get("pdi_score")
                or original_payload.get("pdi_score")
                or metric_summary.get("pdi_score")
                or ((original_payload.get("metrics") or {}).get("pdi_score"))
            )
        return float(value) if value is not None else None
    if metric_key == "wmreward_surprise":
        value = ((metric_results.get("wmreward_jepa") or {}).get("surprise"))
        if value is None:
            bucket = original_payload.get("metric_results", {}).get("wmreward_jepa") or {}
            value = bucket.get("surprise")
        if value is None:
            similarity = payload.get("wmreward_jepa")
            if similarity is None:
                similarity = original_payload.get("wmreward_jepa") or metric_summary.get("wmreward_jepa")
            if similarity is not None:
                value = 1.0 - float(similarity)
        return float(value) if value is not None else None
    if metric_key == "vjepa_relraw":
        value = (((metric_results.get("vjepa_proxy") or {}).get("details") or {}).get("temporal_relation_raw_error"))
        value = value if value is not None else (((payload.get("jepa") or {}).get("temporal_relation_raw_error")))
        value = value if value is not None else (((original_payload.get("jepa") or {}).get("temporal_relation_raw_error")))
        return float(value) if value is not None else None
    if metric_key == "vjepa_deltarel":
        value = (((metric_results.get("vjepa_proxy") or {}).get("details") or {}).get("delta_relation_raw_error"))
        value = value if value is not None else (((payload.get("jepa") or {}).get("delta_relation_raw_error")))
        value = value if value is not None else (((original_payload.get("jepa") or {}).get("delta_relation_raw_error")))
        return float(value) if value is not None else None
    if metric_key == "vjepa_deltaprof":
        value = (((metric_results.get("vjepa_proxy") or {}).get("details") or {}).get("delta_profile_error"))
        value = value if value is not None else (((payload.get("jepa") or {}).get("delta_profile_error")))
        value = value if value is not None else (((original_payload.get("jepa") or {}).get("delta_profile_error")))
        return float(value) if value is not None else None
    if metric_key == "cosmos_reason1":
        value = ((metric_results.get("cosmos_reason1") or {}).get("score"))
        value = value if value is not None else payload.get("cosmos_reason1_score")
        value = value if value is not None else original_payload.get("cosmos_reason1_score")
        value = value if value is not None else metric_summary.get("cosmos_reason1_score")
        return float(value) if value is not None else None
    if metric_key == "videophy2_pc":
        value = ((metric_results.get("videophy2_auto") or {}).get("pc_score"))
        value = value if value is not None else payload.get("videophy2_auto_pc")
        value = value if value is not None else original_payload.get("videophy2_auto_pc")
        value = value if value is not None else metric_summary.get("videophy2_auto_pc")
        return float(value) if value is not None else None
    if metric_key == "videophy2_sa":
        value = ((metric_results.get("videophy2_auto") or {}).get("sa_score"))
        value = value if value is not None else payload.get("videophy2_auto_sa")
        value = value if value is not None else original_payload.get("videophy2_auto_sa")
        value = value if value is not None else metric_summary.get("videophy2_auto_sa")
        return float(value) if value is not None else None
    if metric_key == "videophy2_joint":
        bucket = metric_results.get("videophy2_auto") or {}
        value = bucket.get("joint")
        if value is not None:
            return float(value)
        pc = bucket.get("pc_score", payload.get("videophy2_auto_pc"))
        sa = bucket.get("sa_score", payload.get("videophy2_auto_sa"))
        if pc is None:
            pc = original_payload.get("videophy2_auto_pc") or metric_summary.get("videophy2_auto_pc")
        if sa is None:
            sa = original_payload.get("videophy2_auto_sa") or metric_summary.get("videophy2_auto_sa")
        if pc is None or sa is None:
            return None
        return float(1.0 if float(pc) >= 4.0 and float(sa) >= 4.0 else 0.0)
    return None


def safe_mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def aggregate_method_metrics(group_root: Path) -> list[dict[str, Any]]:
    methods = [path.name for path in sorted(group_root.iterdir()) if path.is_dir() and not path.name.startswith("_")]
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_dir = group_root / method
        payloads = [load_json(path) for path in sorted(method_dir.glob("*.json"))]
        row: dict[str, Any] = {
            "method": method,
            "num_cases": len(payloads),
        }
        for spec in METRIC_SPECS:
            values = [resolve_metric_value(payload, spec["key"]) for payload in payloads]
            clean = [float(value) for value in values if value is not None]
            row[spec["key"]] = safe_mean(clean)
        rows.append(row)
    return rows


def find_ffmpeg() -> str:
    candidates = [
        shutil.which("ffmpeg"),
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/home/gaoya/miniconda3/envs/wan/bin/ffmpeg",
        "/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg",
        "/data/gaoya/miniconda3/envs/wan/bin/ffmpeg",
        "/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg",
        "/data/gaoya/home_miniconda3/pkgs/ffmpeg-8.0.0-gpl_hc3e963e_905/bin/ffmpeg",
        "/home/gaoya/.marscode/ai-chat/binary/1.6.38/modules/ai-agent/ffmpeg",
        "/home/gaoya/.marscode/ai-chat/binary/1.6.36/modules/ai-agent/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError("ffmpeg not found in PATH or known conda envs")


def ensure_browser_video(dst_dir: Path, source_path: Path, stem: str) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_path = dst_dir / f"{stem}.browser.mp4"
    if out_path.exists() and out_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns and out_path.stat().st_size > 0:
        return out_path.name
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path.name


def ensure_context_image(dst_dir: Path, source_path: Path, stem: str) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_path = dst_dir / f"{stem}{source_path.suffix}"
    if out_path.exists() or out_path.is_symlink():
        out_path.unlink()
    out_path.symlink_to(source_path)
    return out_path.name


def build_baseline_case(group_root: Path, output_dir: Path, case_key: str, methods: list[str]) -> dict[str, Any]:
    gt_payload = load_json(group_root / "GT" / f"{case_key}.json")
    case_dir = output_dir / "assets" / case_key
    context_image_rel = None
    if gt_payload.get("input_image"):
        context_image_rel = f"assets/{case_key}/" + ensure_context_image(case_dir, Path(str(gt_payload["input_image"])), "context_image")
    context_video_rel = None
    if gt_payload.get("input_context_video"):
        context_video_rel = f"assets/{case_key}/" + ensure_browser_video(case_dir, Path(str(gt_payload["input_context_video"])), "context_video")
    gt_full_video_rel = f"assets/{case_key}/" + ensure_browser_video(case_dir, Path(str(gt_payload["output_video"])), "gt_full")

    method_rows = []
    for method in methods:
        payload = load_json(group_root / method / f"{case_key}.json")
        method_case_dir = case_dir / method
        video_rel = f"assets/{case_key}/{method}/" + ensure_browser_video(method_case_dir, Path(str(payload["output_video"])), "output")
        method_rows.append(
            {
                "method": method,
                "video_rel": video_rel,
                "conditioning_mode": payload.get("conditioning_mode"),
                "context_frames": payload.get("context_frames"),
            }
        )

    return {
        "case_key": case_key,
        "category": str(gt_payload.get("category", "")),
        "prompt": str(gt_payload.get("input_prompt", "")),
        "clip_name": str(gt_payload.get("clip_name", gt_payload.get("sample_name", case_key))),
        "context_image_rel": context_image_rel,
        "context_video_rel": context_video_rel,
        "gt_full_video_rel": gt_full_video_rel,
        "methods": method_rows,
    }


def render_metric_value(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def build_baseline_html(group_name: str, report: dict[str, Any]) -> str:
    metric_headers = "".join(f"<th>{html.escape(spec['label'])}</th>" for spec in METRIC_SPECS)
    metric_rows = []
    for row in report["metric_rows"]:
        tds = "".join(f"<td>{render_metric_value(row.get(spec['key']))}</td>" for spec in METRIC_SPECS)
        metric_rows.append(f"<tr><td>{html.escape(row['method'])}</td><td>{row['num_cases']}</td>{tds}</tr>")

    metric_descs = "".join(
        f'<li><strong>{html.escape(spec["label"])}</strong>：{html.escape(spec["desc"])}</li>' for spec in METRIC_SPECS
    )

    case_blocks = []
    for case in report["cases"]:
        media_cards = []
        if case.get("context_image_rel"):
            media_cards.append(
                f"""
                <section class="media-card">
                  <div class="media-title">Context Image</div>
                  <img src="{html.escape(case['context_image_rel'])}" alt="context image" />
                </section>
                """
            )
        if case.get("context_video_rel"):
            media_cards.append(
                f"""
                <section class="media-card">
                  <div class="media-title">Context Video</div>
                  <video controls preload="metadata" src="{html.escape(case['context_video_rel'])}"></video>
                </section>
                """
            )
        media_cards.append(
            f"""
            <section class="media-card">
              <div class="media-title">GT / Full Video</div>
              <video controls preload="metadata" src="{html.escape(case['gt_full_video_rel'])}"></video>
            </section>
            """
        )
        outputs = []
        for method_row in case["methods"]:
            mode_text = method_row["conditioning_mode"] or "unknown"
            if method_row.get("context_frames") is not None:
                mode_text += f" | ctx={method_row['context_frames']}"
            outputs.append(
                f"""
                <section class="media-card">
                  <div class="media-title">{html.escape(method_row['method'])}</div>
                  <div class="media-note">{html.escape(mode_text)}</div>
                  <video controls preload="metadata" src="{html.escape(method_row['video_rel'])}"></video>
                </section>
                """
            )
        case_blocks.append(
            f"""
            <article class="case-card">
              <div class="case-head">
                <div>
                  <div class="eyebrow">{html.escape(case['category'])}</div>
                  <h2>{html.escape(case['case_key'])}</h2>
                </div>
                <div class="meta-chip">{html.escape(case['clip_name'])}</div>
              </div>
              <div class="prompt-box">{html.escape(case['prompt'])}</div>
              <div class="media-grid inputs">
                {''.join(media_cards)}
              </div>
              <div class="media-grid outputs">
                {''.join(outputs)}
              </div>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(report['title'])}</title>
  <style>
    :root {{
      --bg0: #f7f2ea;
      --bg1: #eadfce;
      --panel: rgba(255, 251, 245, 0.96);
      --line: #ddcfbc;
      --ink: #201b17;
      --muted: #6d665d;
      --accent: #0d5b54;
      --accent2: #b96b34;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at left top, rgba(185, 107, 52, 0.12), transparent 24%),
        radial-gradient(circle at right top, rgba(13, 91, 84, 0.12), transparent 28%),
        linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%);
    }}
    .page {{
      max-width: 1720px;
      margin: 0 auto;
      padding: 26px;
    }}
    .hero, .panel, .case-card, .media-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
    }}
    .hero, .panel, .case-card {{
      padding: 20px;
      margin-bottom: 18px;
    }}
    .eyebrow {{
      color: var(--accent2);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-size: 12px;
      margin-bottom: 6px;
    }}
    h1, h2, h3 {{
      margin-top: 0;
    }}
    .intro, .metric-desc, .meta-note, .media-note {{
      color: var(--muted);
      line-height: 1.7;
    }}
    .metric-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      margin-top: 12px;
    }}
    .metric-table th, .metric-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }}
    .metric-table th {{
      color: var(--accent);
      background: rgba(245, 239, 229, 0.88);
    }}
    .metric-desc {{
      margin: 0;
      padding-left: 18px;
    }}
    .case-head {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 12px;
      align-items: center;
    }}
    .meta-chip {{
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--muted);
      background: rgba(245, 239, 229, 0.88);
    }}
    .prompt-box {{
      background: rgba(246, 241, 232, 0.92);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      margin-bottom: 14px;
      line-height: 1.7;
    }}
    .media-grid {{
      display: grid;
      gap: 14px;
    }}
    .inputs {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin-bottom: 14px;
    }}
    .outputs {{
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }}
    .media-card {{
      padding: 12px;
    }}
    .media-title {{
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 8px;
    }}
    video, img {{
      width: 100%;
      display: block;
      border-radius: 12px;
      background: #000;
    }}
    img {{
      aspect-ratio: 16 / 9;
      object-fit: cover;
      background: #eee4d6;
    }}
    .back-link {{
      color: var(--accent);
      font-weight: 700;
      text-decoration: none;
    }}
    @media (max-width: 1260px) {{
      .inputs, .outputs {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 760px) {{
      .inputs, .outputs {{ grid-template-columns: 1fr; }}
      .case-head {{ flex-direction: column; align-items: flex-start; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">{html.escape(group_name)}</div>
      <h1>{html.escape(report['title'])}</h1>
      <p class="intro">{html.escape(report['intro'])}</p>
      <p><a class="back-link" href="../index.html">返回本组入口</a></p>
    </section>
    <section class="panel">
      <h2>指标总表</h2>
      <p class="meta-note">表中数值是该组 benchmark 在全部 case 上、按 baseline 方法分别求出的平均值。</p>
      <div style="overflow-x:auto;">
        <table class="metric-table">
          <thead>
            <tr>
              <th>Method</th>
              <th>样本数</th>
              {metric_headers}
            </tr>
          </thead>
          <tbody>
            {''.join(metric_rows)}
          </tbody>
        </table>
      </div>
      <h3>指标说明</h3>
      <ul class="metric-desc">
        {metric_descs}
      </ul>
    </section>
    {''.join(case_blocks)}
  </div>
</body>
</html>"""


def parse_simple_epoch_log(path: Path) -> dict[str, Any]:
    pattern = re.compile(r"epoch=(\d+)\s+train_loss=([0-9.eE+-]+)\s+val_loss=([0-9.eE+-]+)")
    train_points: list[tuple[float, float]] = []
    val_points: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        epoch = float(match.group(1))
        train_points.append((epoch, float(match.group(2))))
        val_points.append((epoch, float(match.group(3))))
    return {
        "train": train_points,
        "val": val_points,
    }


def parse_json_epoch_log(path: Path) -> dict[str, Any]:
    train_points: list[tuple[float, float]] = []
    val_points: list[tuple[float, float]] = []
    stage = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if "train_metrics" not in payload or "val_metrics" not in payload:
            continue
        step = float(payload.get("global_epoch", payload.get("epoch", len(train_points) + 1)))
        train_loss = ((payload.get("train_metrics") or {}).get("loss"))
        val_loss = ((payload.get("val_metrics") or {}).get("loss"))
        if train_loss is None or val_loss is None:
            continue
        train_points.append((step, float(train_loss)))
        val_points.append((step, float(val_loss)))
        stage = payload.get("stage", stage)
    return {
        "stage": stage,
        "train": train_points,
        "val": val_points,
    }


def parse_loss_series(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    if '"train_metrics"' in text and '"val_metrics"' in text:
        series = parse_json_epoch_log(path)
    else:
        series = parse_simple_epoch_log(path)
    if not series.get("train") and not series.get("val"):
        return None
    return series


def build_svg_chart(series: dict[str, Any] | None, width: int = 420, height: int = 170) -> str:
    if not series or (not series.get("train") and not series.get("val")):
        return '<div class="chart-missing">暂无 loss 曲线</div>'
    points = list(series.get("train", [])) + list(series.get("val", []))
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if math.isclose(y_min, y_max):
        y_max = y_min + 1e-6

    def map_xy(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        px = 16 + (x - x_min) / (x_max - x_min) * (width - 32)
        py = 12 + (1.0 - (y - y_min) / (y_max - y_min)) * (height - 24)
        return px, py

    def polyline(points_: list[tuple[float, float]], color: str) -> str:
        if not points_:
            return ""
        pts = " ".join(f"{px:.2f},{py:.2f}" for px, py in (map_xy(point) for point in points_))
        return f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{pts}" />'

    train_poly = polyline(series.get("train", []), "#c85f2a")
    val_poly = polyline(series.get("val", []), "#0d5b54")
    return f"""
    <svg viewBox="0 0 {width} {height}" class="loss-chart" aria-label="loss chart">
      <rect x="0" y="0" width="{width}" height="{height}" rx="14" fill="#fcf8f2" stroke="#eadfce" />
      <line x1="16" y1="{height-12}" x2="{width-16}" y2="{height-12}" stroke="#d8ccb9" />
      <line x1="16" y1="12" x2="16" y2="{height-12}" stroke="#d8ccb9" />
      {train_poly}
      {val_poly}
      <text x="18" y="24" fill="#7b4f2d" font-size="12">train</text>
      <text x="64" y="24" fill="#0d5b54" font-size="12">val</text>
      <text x="{width-16}" y="{height-16}" fill="#8a8074" font-size="11" text-anchor="end">epoch</text>
      <text x="20" y="{height-20}" fill="#8a8074" font-size="11">loss</text>
    </svg>
    """


def extract_best_val(series: dict[str, Any] | None) -> float | None:
    if not series or not series.get("val"):
        return None
    return min(value for _, value in series["val"])


def extract_last_val(series: dict[str, Any] | None) -> float | None:
    if not series or not series.get("val"):
        return None
    return float(series["val"][-1][1])


def load_wandb_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_training_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    links_root = output_dir / "runs"
    links_root.mkdir(parents=True, exist_ok=True)
    for spec in TRAINING_RUN_SPECS:
        predictor_series = parse_loss_series(spec.get("predictor_log"))
        adapter_series = parse_loss_series(spec.get("adapter_log"))
        wandb_summary = load_wandb_summary(spec.get("wandb_summary"))
        best_rel = None
        if spec.get("best_page") and Path(spec["best_page"]).exists():
            link = links_root / f"{spec['id']}_best"
            ensure_symlink(Path(spec["best_page"]), link)
            best_rel = f"runs/{link.name}"
        timeline_rel = None
        if spec.get("timeline_page") and Path(spec["timeline_page"]).exists():
            link = links_root / f"{spec['id']}_timeline"
            ensure_symlink(Path(spec["timeline_page"]), link)
            timeline_rel = f"runs/{link.name}"
        rows.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "desc": spec["desc"],
                "best_rel": best_rel,
                "timeline_rel": timeline_rel,
                "predictor_series": predictor_series,
                "adapter_series": adapter_series,
                "predictor_best_val": extract_best_val(predictor_series),
                "predictor_last_val": extract_last_val(predictor_series),
                "adapter_best_val": extract_best_val(adapter_series),
                "adapter_last_val": extract_last_val(adapter_series),
                "wandb_summary": wandb_summary,
            }
        )
    return rows


def build_training_html(group_name: str, report: dict[str, Any]) -> str:
    run_cards = []
    for row in report["runs"]:
        links = []
        if row.get("best_rel"):
            links.append(f'<a class="link" href="{html.escape(row["best_rel"])}/index.html">best case 页面</a>')
        if row.get("timeline_rel"):
            links.append(f'<a class="link secondary" href="{html.escape(row["timeline_rel"])}/index.html">不同 step ckpt 页面</a>')
        if not links:
            links.append('<span class="pending">当前没有现成的 case 页面</span>')
        wandb_line = ""
        summary = row.get("wandb_summary") or {}
        if summary:
            keys = []
            for key in ["joint_finetune/best_metric", "selection_metric", "stage_best_metric"]:
                if key in summary:
                    keys.append(f"{key}={summary[key]}")
            if keys:
                wandb_line = f'<p class="meta">wandb summary: {" | ".join(keys)}</p>'
        run_cards.append(
            f"""
            <article class="run-card">
              <div class="eyebrow">{html.escape(row['id'])}</div>
              <h2>{html.escape(row['label'])}</h2>
              <p class="meta">{html.escape(row['desc'])}</p>
              <div class="stats">
                <span>predictor best val: {"—" if row['predictor_best_val'] is None else f"{row['predictor_best_val']:.4f}"}</span>
                <span>predictor last val: {"—" if row['predictor_last_val'] is None else f"{row['predictor_last_val']:.4f}"}</span>
                <span>adapter best val: {"—" if row['adapter_best_val'] is None else f"{row['adapter_best_val']:.4f}"}</span>
                <span>adapter last val: {"—" if row['adapter_last_val'] is None else f"{row['adapter_last_val']:.4f}"}</span>
              </div>
              <div class="chart-grid">
                <section class="chart-card">
                  <div class="chart-title">Predictor Loss</div>
                  {build_svg_chart(row.get('predictor_series'))}
                </section>
                <section class="chart-card">
                  <div class="chart-title">Adapter Loss</div>
                  {build_svg_chart(row.get('adapter_series'))}
                </section>
              </div>
              {wandb_line}
              <div class="link-row">
                {''.join(links)}
              </div>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(report['title'])}</title>
  <style>
    :root {{
      --bg0: #f7f2ea;
      --bg1: #eadfce;
      --panel: rgba(255, 251, 245, 0.96);
      --line: #ddcfbc;
      --ink: #201b17;
      --muted: #6d665d;
      --accent: #0d5b54;
      --accent2: #b96b34;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at left top, rgba(185, 107, 52, 0.12), transparent 24%),
        radial-gradient(circle at right top, rgba(13, 91, 84, 0.12), transparent 28%),
        linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%);
    }}
    .page {{
      max-width: 1560px;
      margin: 0 auto;
      padding: 26px;
    }}
    .hero, .run-card, .chart-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
    }}
    .hero, .run-card {{
      padding: 20px;
      margin-bottom: 18px;
    }}
    .eyebrow {{
      color: var(--accent2);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .intro, .meta, .chart-missing {{
      color: var(--muted);
      line-height: 1.7;
    }}
    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 12px 0 14px;
    }}
    .stats span {{
      background: #f3eadf;
      border-radius: 999px;
      padding: 4px 10px;
      color: #7b4f2d;
      font-size: 13px;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }}
    .chart-card {{
      padding: 12px;
    }}
    .chart-title {{
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .loss-chart {{
      width: 100%;
      display: block;
    }}
    .chart-missing {{
      background: #fcf8f2;
      border: 1px dashed var(--line);
      border-radius: 14px;
      min-height: 160px;
      display: grid;
      place-items: center;
    }}
    .link-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
    }}
    .link {{
      color: var(--accent);
      font-weight: 700;
      text-decoration: none;
    }}
    .secondary {{
      color: #7f4f28;
    }}
    .pending {{
      color: #9b8f82;
      font-weight: 700;
    }}
    .back-link {{
      color: var(--accent);
      font-weight: 700;
      text-decoration: none;
    }}
    @media (max-width: 980px) {{
      .chart-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">{html.escape(group_name)}</div>
      <h1>{html.escape(report['title'])}</h1>
      <p class="intro">{html.escape(report['intro'])}</p>
      <p><a class="back-link" href="../index.html">返回本组入口</a></p>
    </section>
    {''.join(run_cards)}
  </div>
</body>
</html>"""


def build_group_index_html(group_name: str, title: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      color: #1f1f1b;
      background: linear-gradient(180deg, #f8f3ea 0%, #ede2d3 100%);
    }}
    .page {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero, .card {{
      background: rgba(255, 252, 246, 0.95);
      border: 1px solid #dccfbe;
      border-radius: 18px;
    }}
    .hero {{
      padding: 24px;
      margin-bottom: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .card {{
      padding: 20px;
    }}
    .eyebrow {{
      color: #b8642a;
      text-transform: uppercase;
      font-size: 13px;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }}
    .desc {{
      color: #6f675d;
      line-height: 1.7;
    }}
    .link {{
      color: #0f5a52;
      font-weight: 700;
      text-decoration: none;
    }}
    @media (max-width: 800px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">{html.escape(group_name)}</div>
      <h1>{html.escape(title)}</h1>
      <p class="desc">本组只保留两个页面：`baseline` 负责 benchmark case + 全 baseline 结果 + 指标总表；`training` 负责 `phys_state_video` 项目中不同训练阶段的 ckpt case 与 loss。</p>
      <p><a class="link" href="../index.html">返回 ABD 总入口</a></p>
    </section>
    <section class="grid">
      <article class="card">
        <div class="eyebrow">baseline</div>
        <h2>Benchmark Baselines</h2>
        <p class="desc">抽取代表性 case，统一展示 GT / baseline 视频，并在页首给出该组所有 baseline 的平均指标表。</p>
        <a class="link" href="baseline/index.html">打开 baseline 页面</a>
      </article>
      <article class="card">
        <div class="eyebrow">training</div>
        <h2>Project Checkpoints & Loss</h2>
        <p class="desc">汇总 `phys_state_video` 项目当前主要训练分支的 best case 页面、不同 step 的 ckpt 页面，以及 loss 曲线。</p>
        <a class="link" href="training/index.html">打开 training 页面</a>
      </article>
    </section>
  </div>
</body>
</html>"""


def build_root_index_html(port: int) -> str:
    cards = []
    for group_name in ["A", "B", "D"]:
        cards.append(
            f"""
            <article class="card">
              <div class="eyebrow">{group_name}</div>
              <h2>{html.escape(GROUP_TITLES[group_name])}</h2>
              <p class="desc">每组只保留两个页面：`baseline` 和 `training`。</p>
              <a class="link" href="{group_name}/index.html">打开 {group_name} 组入口</a>
            </article>
            """
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ABD Unified Visualization</title>
  <style>
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      color: #1f1f1b;
      background: linear-gradient(180deg, #f8f3ea 0%, #ede2d3 100%);
    }}
    .page {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero, .card {{
      background: rgba(255, 252, 246, 0.95);
      border: 1px solid #dccfbe;
      border-radius: 18px;
    }}
    .hero {{
      padding: 24px;
      margin-bottom: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
    }}
    .card {{
      padding: 20px;
    }}
    .eyebrow {{
      color: #b8642a;
      text-transform: uppercase;
      font-size: 13px;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }}
    .desc {{
      color: #6f675d;
      line-height: 1.7;
    }}
    .link {{
      color: #0f5a52;
      font-weight: 700;
      text-decoration: none;
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>ABD 可视化统一入口</h1>
      <p class="desc">这一版把目录重构为 `A/B/D` 三组，每组只保留两个页面：`baseline` 和 `training`。访问地址：<a class="link" href="http://127.0.0.1:{port}">http://127.0.0.1:{port}</a></p>
    </section>
    <section class="grid">
      {''.join(cards)}
    </section>
  </div>
</body>
</html>"""


def export_group_pages(abd_root: Path, output_root: Path, group_name: str) -> None:
    group_root = abd_root / group_name
    group_out = output_root / group_name
    group_out.mkdir(parents=True, exist_ok=True)

    methods = [path.name for path in sorted(group_root.iterdir()) if path.is_dir() and not path.name.startswith("_")]
    methods = [method for method in methods if method != "GT"]
    baseline_methods = ["GT"] + methods
    case_keys = choose_case_keys(group_root, group_name)
    baseline_out = group_out / "baseline"
    baseline_out.mkdir(parents=True, exist_ok=True)
    baseline_cases = [build_baseline_case(group_root, baseline_out, case_key, baseline_methods) for case_key in case_keys]
    metric_rows = aggregate_method_metrics(group_root)
    baseline_report = {
        "title": f"{GROUP_TITLES[group_name]} Baseline 对比",
        "intro": "本页展示该组 benchmark 的代表性 case，并统一对比 GT 与全部 baseline 的视频结果。页面顶部指标表使用该组全部 case 的平均值。",
        "group": group_name,
        "metric_rows": metric_rows,
        "cases": baseline_cases,
    }
    write_json(baseline_out / "report.json", baseline_report)
    (baseline_out / "index.html").write_text(build_baseline_html(group_name, baseline_report), encoding="utf-8")

    training_out = group_out / "training"
    training_out.mkdir(parents=True, exist_ok=True)
    training_rows = build_training_rows(training_out)
    training_report = {
        "title": f"{GROUP_TITLES[group_name]} Training / CKPT / Loss",
        "intro": "本页汇总 `phys_state_video` 项目当前主要训练分支。每个卡片给出可打开的 case 页面、不同 step ckpt 页面，以及 predictor / adapter loss 曲线。",
        "group": group_name,
        "runs": training_rows,
    }
    write_json(training_out / "report.json", training_report)
    (training_out / "index.html").write_text(build_training_html(group_name, training_report), encoding="utf-8")

    (group_out / "index.html").write_text(build_group_index_html(group_name, GROUP_TITLES[group_name]), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    for group_name in ["A", "B", "D"]:
        export_group_pages(args.abd_root, args.output_root, group_name)

    (args.output_root / "index.html").write_text(build_root_index_html(args.port), encoding="utf-8")

    if not args.no_serve:
        pid = start_server(args.output_root, args.port)
        print(f"served http://127.0.0.1:{args.port} pid={pid}")
    print(args.output_root / "index.html")


if __name__ == "__main__":
    main()
