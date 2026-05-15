#!/usr/bin/env python3
"""Rebuild a dataset portal with per-sample detail pages."""

from __future__ import annotations

import html
import hashlib
import json
import os
import shutil
import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

DEFAULT_ROOT = Path("/home/gaoya/portal_hub_sim/sum0504_portal")
STATE_VALIDATION_ROOT = Path("/home/gaoya/Code_Video/Code_data/data0417/data_check/state_validation_window")
DEFAULT_SUMMARY_ROOT = Path("/home/gaoya/Code_Video/Code_data/data0417/data_summary/sum0504")
MAX_ITEMS_PER_GROUP = 10
ROOT = DEFAULT_ROOT
SUMMARY_ROOT = DEFAULT_SUMMARY_ROOT
PORTAL_TITLE = "sum0504 Portal"
PREFER_GIF = False
MANIFEST_PATH = ROOT / "manifest.json"
INDEX_ASSET_ROOT = ROOT / "index_assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild a dataset portal with per-sample detail pages.")
    parser.add_argument(
        "--sample_substring",
        type=str,
        default="",
        help="Only include samples whose absolute sample_dir contains this substring, while keeping the original split/count/collision grouping.",
    )
    parser.add_argument(
        "--index_only",
        action="store_true",
        help="Only rebuild portal index and manifest from sum0504, without regenerating per-sample detail pages.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Portal output directory.",
    )
    parser.add_argument(
        "--summary_root",
        type=Path,
        default=DEFAULT_SUMMARY_ROOT,
        help="Summary root that contains split/simulator/count/collision/samples.txt leaves.",
    )
    parser.add_argument(
        "--portal_title",
        type=str,
        default="sum0504 Portal",
        help="Title shown on the portal homepage and detail pages.",
    )
    parser.add_argument(
        "--prefer_gif",
        action="store_true",
        help="Render video media as GIF previews whenever possible.",
    )
    parser.add_argument(
        "--collision_bucket",
        type=str,
        default="",
        help="Only include a specific collision bucket such as no_collision.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def relpath(path: Path, start: Path) -> str:
    return os.path.relpath(path, start)


def asset_src(path: Path, page_dir: Path) -> str:
    try:
        path.relative_to(ROOT)
        return relpath(path, page_dir)
    except ValueError:
        asset_dir = page_dir / "_assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:12]
        target = asset_dir / f"{digest}_{path.name}"
        if target.is_symlink():
            target.unlink()
        if (not target.exists()) or target.stat().st_size != path.stat().st_size:
            shutil.copy2(path, target)
        return relpath(target, page_dir)


def index_asset_src(path: Path) -> str:
    INDEX_ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:12]
    target = INDEX_ASSET_ROOT / f"{digest}_{path.name}"
    if target.is_symlink():
        target.unlink()
    if not target.exists():
        try:
            target.symlink_to(path)
        except Exception:
            shutil.copy2(path, target)
    elif target.is_file() and target.stat().st_size != path.stat().st_size:
        target.unlink()
        try:
            target.symlink_to(path)
        except Exception:
            shutil.copy2(path, target)
    return relpath(target, ROOT)


def index_asset_preview_src(path: Path) -> tuple[str, str]:
    if PREFER_GIF and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        gif_path = INDEX_ASSET_ROOT / f"{hashlib.md5(str(path).encode('utf-8')).hexdigest()[:12]}_{path.stem}.gif"
        if (not gif_path.exists()) and path.exists():
            make_gif_from_video(path, gif_path)
        if gif_path.exists():
            return "image", relpath(gif_path, ROOT)
    return ("video" if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"} else "image", index_asset_src(path))


def make_gif_from_video(video_path: Path, gif_path: Path, max_frames: int = 24) -> Path | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, frame_count // max_frames) if frame_count > max_frames and max_frames > 0 else 1
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        frame_idx += 1
    cap.release()
    if not frames:
        return None
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=False,
    )
    return gif_path


def save_gif_from_frame_paths(frame_paths: list[Path], gif_path: Path, duration_ms: int = 120) -> Path | None:
    frames: list[Image.Image] = []
    for frame_path in frame_paths:
        if not frame_path.exists():
            continue
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB").copy())
    if not frames:
        return None
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration_ms),
        loop=0,
        optimize=False,
    )
    return gif_path


def video_preview_src(video_path: Path, page_dir: Path) -> tuple[str, str]:
    video_src = asset_src(video_path, page_dir)
    gif_path = page_dir / "_assets" / f"{hashlib.md5(str(video_path).encode('utf-8')).hexdigest()[:12]}_{video_path.stem}.gif"
    if (not gif_path.exists()) and video_path.exists():
        make_gif_from_video(video_path, gif_path)
    if gif_path.exists():
        return "image", relpath(gif_path, page_dir)
    return "video", video_src


def fmt_bool(v: bool) -> str:
    return "yes" if v else "no"


def pick_existing(sample_dir: Path, names: list[str]) -> list[Path]:
    result = []
    for name in names:
        p = sample_dir / name
        if p.exists():
            result.append(p)
    return result


def load_json_maybe(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return load_json(path)


def detect_striker_info(sample_dir: Path, meta: dict[str, Any]) -> tuple[bool, float | None]:
    objects = list(meta.get("objects", []) or [])
    has_striker = any(
        str(obj.get("source_object_id", "")) == "yellow_striker_ball"
        or str(obj.get("motion_group", "")) == "striker"
        or str(obj.get("object_motion_group", "")) == "striker"
        for obj in objects
        if isinstance(obj, dict)
    )
    scene_input = load_json_maybe(sample_dir / "scene_input.json") or {}
    raw_speed = scene_input.get("striker_speed_mps", None)
    striker_speed = None
    if raw_speed is not None:
        try:
            striker_speed = float(raw_speed)
        except Exception:
            striker_speed = None
    return bool(has_striker), striker_speed


def resolve_source_sample_dir(record: dict[str, Any]) -> Path | None:
    meta = load_json_maybe(record.get("meta_path")) or {}
    pair_meta = load_json_maybe(record.get("pair_meta_path")) or {}
    candidates = [
        meta.get("source_sample_dir"),
        (meta.get("source_paths") or {}).get("source_sample_dir"),
        pair_meta.get("source_sample_dir"),
        (pair_meta.get("source_paths") or {}).get("source_sample_dir"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            path = Path(candidate)
            if path.exists():
                return path
    return None


def find_state_validation_case_dir(record: dict[str, Any]) -> Path | None:
    sample_name = str(record.get("sample_name", "")).strip()
    if not sample_name:
        return None
    for case_dir in STATE_VALIDATION_ROOT.glob("*/*/*"):
        if case_dir.is_dir() and case_dir.name.endswith(sample_name):
            return case_dir
    return None


def find_visualization_dir(sample_dir: Path | None) -> Path | None:
    if sample_dir is None:
        return None
    vis_dir = sample_dir / "visualizations"
    if vis_dir.exists():
        return vis_dir
    return None


def load_state_bundle(record: dict[str, Any]) -> dict[str, Any] | None:
    meta = load_json(record["meta_path"]) if record.get("meta_path") is not None else {}

    if record.get("state_pair_path") is not None:
        payload = np.load(record["state_pair_path"], allow_pickle=True)
        state_raw = np.asarray(payload["state_raw"], dtype=np.float32)
        object_ids = np.asarray(payload["object_ids"], dtype=np.int32)
        seg_ids = np.asarray(payload["seg_ids"], dtype=np.int32)
        return {
            "state_raw": state_raw,
            "object_ids": object_ids,
            "seg_ids": seg_ids,
            "meta": meta,
        }

    if record.get("state_9d_path") is not None:
        state_raw = np.asarray(np.load(record["state_9d_path"]), dtype=np.float32)
        object_ids = np.asarray([], dtype=np.int32)
        seg_ids = np.asarray([], dtype=np.int32)
        if record.get("anchor_targets_path") is not None:
            payload = np.load(record["anchor_targets_path"], allow_pickle=True)
            object_ids = np.asarray(payload["object_ids"], dtype=np.int32)
            seg_ids = np.asarray(payload["seg_ids"], dtype=np.int32)
        return {
            "state_raw": state_raw,
            "object_ids": object_ids,
            "seg_ids": seg_ids,
            "meta": meta,
        }

    if record.get("anchor_targets_path") is None:
        return None

    payload = np.load(record["anchor_targets_path"], allow_pickle=True)
    object_ids = np.asarray(payload["object_ids"], dtype=np.int32)
    seg_ids = np.asarray(payload["seg_ids"], dtype=np.int32)
    com_uv = np.asarray(payload["com_uv"], dtype=np.float32)
    center_depth = np.asarray(payload["center_depth"], dtype=np.float32)
    bbox_xyxy = np.asarray(payload["bbox_xyxy"], dtype=np.float32)
    visibility_mask = np.asarray(payload["visibility_mask"], dtype=np.uint8)
    x1 = bbox_xyxy[..., 0]
    y1 = bbox_xyxy[..., 1]
    x2 = bbox_xyxy[..., 2]
    y2 = bbox_xyxy[..., 3]
    width = np.maximum(0.0, x2 - x1).astype(np.float32)
    height = np.maximum(0.0, y2 - y1).astype(np.float32)
    u = com_uv[..., 0]
    v = com_uv[..., 1]
    d = center_depth.astype(np.float32)
    fps = float(meta.get("fps", meta.get("video_fps", 12.0)) or 12.0)
    dt = max(1.0 / max(fps, 1e-6), 1e-6)
    du = np.zeros_like(u, dtype=np.float32)
    dv = np.zeros_like(v, dtype=np.float32)
    dd = np.zeros_like(d, dtype=np.float32)
    if u.shape[0] > 1:
        du[1:] = (u[1:] - u[:-1]) / dt
        dv[1:] = (v[1:] - v[:-1]) / dt
        dd[1:] = (d[1:] - d[:-1]) / dt
    vis = visibility_mask.astype(np.float32)
    state_raw = np.stack([u, v, d, width, height, du, dv, dd, vis], axis=-1).astype(np.float32)
    return {
        "state_raw": state_raw,
        "object_ids": object_ids,
        "seg_ids": seg_ids,
        "meta": meta,
    }


def load_event_list(record: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("event_windows_path",):
        path = record.get(key)
        if path is not None and Path(path).exists():
            payload = load_json(Path(path))
            if isinstance(payload, list):
                return payload
    return []


def phase_name(value: int) -> str:
    mapping = {
        0: "unknown",
        1: "pre_motion",
        2: "simple_motion",
        3: "pre_contact",
        4: "contact",
        5: "post_contact",
    }
    return mapping.get(int(value), f"phase_{int(value)}")


def object_labels(bundle: dict[str, Any]) -> list[str]:
    meta = bundle.get("meta") or {}
    objects = meta.get("objects") if isinstance(meta.get("objects"), list) else []
    by_object_id = {
        int(obj.get("object_id")): obj
        for obj in objects
        if isinstance(obj, dict) and obj.get("object_id") is not None
    }
    labels: list[str] = []
    object_ids = np.asarray(bundle.get("object_ids", []), dtype=np.int32)
    seg_ids = np.asarray(bundle.get("seg_ids", []), dtype=np.int32)
    count = int(bundle["state_raw"].shape[1])
    for idx in range(count):
        object_id = int(object_ids[idx]) if idx < object_ids.shape[0] else idx
        seg_id = int(seg_ids[idx]) if idx < seg_ids.shape[0] else (idx + 1)
        obj = by_object_id.get(object_id, {})
        name = str(obj.get("name") or obj.get("source_object_id") or f"obj{object_id}")
        labels.append(f"{name} (obj={object_id}, seg={seg_id})")
    return labels


def object_name_map(record: dict[str, Any]) -> dict[int, str]:
    meta = load_json_maybe(record.get("meta_path")) or {}
    objects = meta.get("objects") if isinstance(meta.get("objects"), list) else []
    mapping: dict[int, str] = {}
    for obj in objects:
        if not isinstance(obj, dict) or obj.get("object_id") is None:
            continue
        obj_id = int(obj["object_id"])
        name = str(obj.get("name") or obj.get("source_object_id") or f"obj{obj_id}")
        role = str(obj.get("role", "")).strip()
        mapping[obj_id] = f"{role}:{name}" if role else name
    return mapping


def event_display_label(event: dict[str, Any], record: dict[str, Any]) -> str:
    obj_names = object_name_map(record)
    participants = event.get("object_indices", event.get("participant_indices", event.get("participants", [])))
    labels: list[str] = []
    for raw in participants:
        idx = int(raw)
        if idx < 0:
            labels.append(str(event.get("environment_name") or "environment"))
        else:
            labels.append(obj_names.get(idx, f"obj{idx}"))
    if labels:
        return " <-> ".join(labels)
    detail = str(event.get("environment_name") or event.get("pair_name") or "").strip()
    return detail or "event"


def event_kind_zh(kind: str) -> str:
    mapping = {
        "event": "碰撞事件",
        "contact_onset": "接触开始",
        "sustained_contact": "持续接触",
        "object_object_contact": "物体-物体接触",
        "environment_contact": "物体-环境接触",
        "object_environment": "物体-环境碰撞",
        "object_object": "物体-物体碰撞",
    }
    return mapping.get(kind, kind)


def explain_card(title: str) -> str:
    text = str(title).strip()
    mapping = {
        "Objects": "含义：这张卡片列出当前样本里有哪些物体，以及每个物体的实例编号、分割编号、角色和设定运动类型。",
        "Recorded Events": "含义：这张卡片用表格列出当前片段里记录到的碰撞/接触事件，方便快速查看事件开始帧、结束帧和事件类型。",
        "Collision Event GIFs": "含义：这张卡片把每个碰撞事件截成短 GIF，帮助你直观看到碰撞前、碰撞中和碰撞后的画面变化。",
        "Frame Phases": "含义：这张卡片统计当前视频各个阶段标签各占多少帧，用来判断片段主要处于运动前、接触前、接触中还是接触后。",
        "State Validation Metrics": "含义：这张卡片展示状态真值可靠性验证指标，用来判断记录的框、深度和速度等监督信号是否可信。",
        "Validation Overlay": "含义：这张卡片把记录的状态标注叠加到视频上，用来检查标注位置和物体真实位置是否对齐。",
        "Validation Comparisons": "含义：这张卡片汇总状态验证中的对比图，帮助快速检查误差、轨迹和多种诊断结果。",
        "Depth Video": "含义：这张卡片展示深度视频，可用来观察物体与相机之间的远近变化是否合理。",
        "Trajectory Overview": "含义：这张卡片把各个物体的中心轨迹画在图上，用来查看整体运动路径、相对位置和是否发生接近或交叉。",
        "State Curves And Collision Timeline": "含义：这张卡片展示速度、深度、可见性随时间的变化，并在时间轴上标出碰撞/接触事件。",
        "Segmentation And Depth Overview": "含义：这张卡片抽取若干关键帧，同时显示分割和深度，用来检查分割掩码与深度记录是否正常。",
        "Contact Heatmaps": "含义：这张卡片把接触图和接触冲量整理成热力图，用来观察哪些物体对在什么时间发生接触。",
        "First Frame": "含义：这张卡片展示片段的第一帧，用来快速确认初始画面、相机视角和物体是否完整进入画面。",
        "Recorded Files": "含义：这张卡片列出当前样本实际存在的记录文件路径，方便核对这个样本到底保存了哪些数据。",
        "Context Video": "含义：这张卡片展示上下文视频，也就是模型输入看到的前半段片段。",
        "Future GT Video": "含义：这张卡片展示未来真值视频，也就是上下文之后真实发生的后半段片段。",
        "Full Video": "含义：这张卡片展示完整片段，便于把上下文、未来和事件放在一起整体查看。",
        "Current Sample Video": "含义：这张卡片展示当前样本对应的视频内容，用来直接查看这条样本的主体运动。",
        "Raw Video": "含义：这张卡片展示原始完整视频，用来查看样本在未截取前的整体运动过程。",
    }
    if text in mapping:
        return mapping[text]
    lowered = text.lower()
    if "context" in lowered:
        return "含义：这张卡片展示上下文片段，也就是模型条件输入部分。"
    if "future" in lowered:
        return "含义：这张卡片展示未来真值片段，也就是模型需要预测的后续部分。"
    if "full" in lowered:
        return "含义：这张卡片展示完整视频片段，便于整体理解这条样本发生了什么。"
    if "video" in lowered:
        return "含义：这张卡片展示当前样本的一段视频，用来直接观察画面中的运动过程。"
    if "depth" in lowered:
        return "含义：这张卡片展示与深度相关的可视化，用来判断远近变化和几何记录是否合理。"
    return "含义：这张卡片展示当前样本的一类记录结果，用来从不同角度检查视频、状态和物理事件。"


def render_trajectory_overview(record: dict[str, Any], page_dir: Path) -> Path | None:
    bundle = load_state_bundle(record)
    if bundle is None:
        return None
    state = np.asarray(bundle["state_raw"], dtype=np.float32)
    labels = object_labels(bundle)
    meta = bundle.get("meta") or {}
    width = float((meta.get("resolution") or [0, 0])[0] or 0.0)
    height = float((meta.get("resolution") or [0, 0])[1] or 0.0)
    out_path = page_dir / "trajectory_overview.png"

    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=150)
    ax.set_title("Trajectory Overview")
    frame0 = record["sample_dir"] / "rgb" / "frame_000.png"
    if frame0.exists():
        with Image.open(frame0) as img:
            bg = np.asarray(img.convert("RGB"))
            if width <= 0.0 or height <= 0.0:
                width, height = float(img.width), float(img.height)
        ax.imshow(bg, extent=(0, width, height, 0), alpha=0.42)
    if width <= 0.0:
        width = max(1.0, float(np.nanmax(state[..., 0]) + 20.0))
    if height <= 0.0:
        height = max(1.0, float(np.nanmax(state[..., 1]) + 20.0))
    cmap = plt.get_cmap("tab10")
    for obj_idx in range(state.shape[1]):
        vis = state[:, obj_idx, 8] > 0.5
        if not np.any(vis):
            continue
        u = state[:, obj_idx, 0]
        v = state[:, obj_idx, 1]
        color = cmap(obj_idx % 10)
        ax.plot(u[vis], v[vis], color=color, linewidth=2.0, label=labels[obj_idx])
        ax.scatter(u[vis][0], v[vis][0], color=color, s=32, marker="o")
        ax.scatter(u[vis][-1], v[vis][-1], color=color, s=42, marker="X")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xlabel("u / px")
    ax.set_ylabel("v / px")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_state_curves(record: dict[str, Any], page_dir: Path) -> Path | None:
    bundle = load_state_bundle(record)
    if bundle is None:
        return None
    state = np.asarray(bundle["state_raw"], dtype=np.float32)
    labels = object_labels(bundle)
    events = load_event_list(record)
    out_path = page_dir / "state_curves.png"
    t = np.arange(state.shape[0], dtype=np.int32)
    speed = np.linalg.norm(state[..., 5:7], axis=-1)
    depth = state[..., 2]
    vis = state[..., 8]

    fig, axes = plt.subplots(4, 1, figsize=(8.4, 8.4), dpi=150, sharex=True)
    cmap = plt.get_cmap("tab10")
    panels = [
        (axes[0], speed, "Speed |du,dv|", "px/s"),
        (axes[1], depth, "Center Depth", "depth"),
        (axes[2], vis, "Visibility", "vis"),
    ]
    for ax, values, title, ylabel in panels:
        for obj_idx in range(values.shape[1]):
            ax.plot(t, values[:, obj_idx], color=cmap(obj_idx % 10), linewidth=1.8, label=labels[obj_idx])
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.92)

    event_ax = axes[3]
    event_ax.set_title("Collision / Contact Timeline", loc="left", fontsize=10)
    if events:
        for row, event in enumerate(events):
            start = int(event.get("start_frame", event.get("frame_idx", 0)))
            end = int(event.get("end_frame", start))
            label = str(event.get("kind") or event.get("window_type") or "event")
            if event.get("environment_name"):
                label += f" ({event.get('environment_name')})"
            event_ax.hlines(row, start, end + 0.001, color="#8a542d", linewidth=6)
            event_ax.text(end + 0.12, row, label, va="center", fontsize=7)
        event_ax.set_ylim(-1, max(1, len(events)))
        event_ax.set_yticks([])
    else:
        event_ax.text(0.02, 0.5, "No collision/contact events recorded in this clip.", transform=event_ax.transAxes, fontsize=9)
        event_ax.set_yticks([])
    event_ax.grid(alpha=0.18)
    event_ax.set_xlabel("frame index")

    for ax in axes[:3]:
        for event in events:
            start = int(event.get("start_frame", event.get("frame_idx", 0)))
            ax.axvline(start, color="#8a542d", alpha=0.24, linewidth=1.0, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_segmentation_depth_overview(record: dict[str, Any], page_dir: Path) -> Path | None:
    seg_path = record.get("seg_path")
    depth_path = record.get("depth_metric_path")
    rgb_dir = record["sample_dir"] / "rgb"
    if seg_path is None or depth_path is None or not rgb_dir.exists():
        return None
    frame_paths = sorted(rgb_dir.glob("frame_*.png"))
    if not frame_paths:
        return None
    seg = np.asarray(np.load(seg_path), dtype=np.int32)
    depth = np.asarray(np.load(depth_path), dtype=np.float32)
    total = min(len(frame_paths), seg.shape[0], depth.shape[0])
    if total <= 0:
        return None
    steps = min(4, total)
    indices = np.linspace(0, total - 1, num=steps, dtype=np.int32)
    out_path = page_dir / "seg_depth_overview.png"

    fig, axes = plt.subplots(2, steps, figsize=(4.2 * steps, 7.2), dpi=150)
    if steps == 1:
        axes = np.asarray(axes).reshape(2, 1)
    cmap = plt.get_cmap("tab20")
    for col, frame_idx in enumerate(indices.tolist()):
        with Image.open(frame_paths[frame_idx]) as img:
            rgb = np.asarray(img.convert("RGB"))
        seg_frame = seg[frame_idx]
        depth_frame = depth[frame_idx]
        overlay = rgb.astype(np.float32) * 0.72
        unique_ids = [int(x) for x in np.unique(seg_frame) if int(x) > 0]
        for seg_id in unique_ids:
            color = np.asarray(cmap(seg_id % 20)[:3], dtype=np.float32) * 255.0
            mask = seg_frame == seg_id
            overlay[mask] = overlay[mask] * 0.35 + color * 0.65
        axes[0, col].imshow(np.clip(overlay / 255.0, 0.0, 1.0))
        axes[0, col].set_title(f"RGB + Seg f={frame_idx}", fontsize=10)
        axes[0, col].axis("off")

        im = axes[1, col].imshow(depth_frame, cmap="viridis")
        axes[1, col].set_title(f"Depth f={frame_idx}", fontsize=10)
        axes[1, col].axis("off")
        fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_contact_heatmaps(record: dict[str, Any], page_dir: Path) -> Path | None:
    graph_path = record.get("contact_graph_path")
    impulse_path = record.get("contact_impulse_path")
    graph = np.asarray(np.load(graph_path), dtype=np.float32) if graph_path is not None else None
    impulse = np.asarray(np.load(impulse_path), dtype=np.float32) if impulse_path is not None else None
    if graph is None and impulse is None:
        return None
    out_path = page_dir / "contact_heatmaps.png"

    panels: list[tuple[str, np.ndarray]] = []
    if graph is not None:
        graph_2d = graph.reshape(graph.shape[0], -1)
        panels.append(("Contact Graph", graph_2d))
    if impulse is not None:
        impulse_2d = impulse.reshape(impulse.shape[0], -1)
        panels.append(("Contact Impulse", impulse_2d))
    if not panels:
        return None
    if all(float(np.nanmax(arr)) <= 0.0 for _, arr in panels):
        return None

    fig, axes = plt.subplots(len(panels), 1, figsize=(8.6, 2.8 * len(panels)), dpi=150, squeeze=False)
    for ax, (title, arr) in zip(axes[:, 0], panels):
        im = ax.imshow(arr.T, aspect="auto", interpolation="nearest", cmap="magma")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("frame index")
        ax.set_ylabel("pair idx")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_energy_plot(record: dict[str, Any], page_dir: Path) -> Path | None:
    energy_path = record.get("energy_path")
    series: list[tuple[str, np.ndarray]] = []
    if energy_path is not None and Path(energy_path).exists():
        payload = np.load(energy_path, allow_pickle=True)
        key_order = ["kinetic_trans", "kinetic_rot", "potential_gravity", "mechanical_total"]
        for key in key_order:
            if key in payload:
                values = np.asarray(payload[key], dtype=np.float32)
                if values.ndim == 1 and values.size > 0:
                    series.append((key, values))
    if not series:
        bundle = load_state_bundle(record)
        if bundle is None:
            return None
        state = np.asarray(bundle["state_raw"], dtype=np.float32)
        speed = np.linalg.norm(state[..., 5:7], axis=-1).sum(axis=1)
        depth = np.nanmean(state[..., 2], axis=1)
        series = [("speed_sum", speed), ("mean_depth", depth)]

    out_path = page_dir / "energy_curves.png"
    fig, ax = plt.subplots(figsize=(8.6, 3.8), dpi=150)
    t = np.arange(series[0][1].shape[0], dtype=np.int32)
    for idx, (label, values) in enumerate(series):
        ax.plot(t, values, linewidth=2.0, label=label)
    ax.set_title("Energy / Motion Curves", loc="left", fontsize=10)
    ax.set_xlabel("frame index")
    ax.set_ylabel("value")
    ax.grid(alpha=0.2)
    if series:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_object_table_html(record: dict[str, Any]) -> str:
    meta = load_json_maybe(record.get("meta_path")) or {}
    objects = meta.get("objects")
    if not isinstance(objects, list) or not objects:
        return ""
    rows: list[str] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(obj.get('object_id', 'n/a')))}</td>"
            f"<td>{html.escape(str(obj.get('seg_id', 'n/a')))}</td>"
            f"<td>{html.escape(str(obj.get('name') or obj.get('source_object_id') or 'n/a'))}</td>"
            f"<td>{html.escape(str(obj.get('role', 'n/a')))}</td>"
            f"<td>{html.escape(str(obj.get('motion_type', obj.get('motion_group', 'n/a'))))}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        "<section class=\"card wide\">"
        "<h2>Objects</h2>"
        f"<p class=\"card-note\">{html.escape(explain_card('Objects'))}</p>"
        "<table><thead><tr><th>object_id</th><th>seg_id</th><th>name</th><th>role</th><th>motion</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
    )


def render_event_table_html(record: dict[str, Any]) -> str:
    events = load_event_list(record)
    if not events:
        return ""
    rows: list[str] = []
    for idx, event in enumerate(events):
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{html.escape(str(event.get('kind') or event.get('window_type') or 'event'))}</td>"
            f"<td>{html.escape(str(event.get('start_frame', event.get('frame_idx', 'n/a'))))}</td>"
            f"<td>{html.escape(str(event.get('end_frame', event.get('frame_idx', 'n/a'))))}</td>"
            f"<td>{html.escape(str(event.get('environment_name') or event.get('pair_name') or ''))}</td>"
            "</tr>"
        )
    return (
        "<section class=\"card wide\">"
        "<h2>Recorded Events</h2>"
        f"<p class=\"card-note\">{html.escape(explain_card('Recorded Events'))}</p>"
        "<table><thead><tr><th>#</th><th>type</th><th>start</th><th>end</th><th>detail</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
    )


def render_event_gifs_html(record: dict[str, Any], page_dir: Path) -> str:
    events = load_event_list(record)
    rgb_dir = record["sample_dir"] / "rgb"
    frame_paths = sorted(rgb_dir.glob("frame_*.png"))
    if not events or not frame_paths:
        return ""

    last_idx = len(frame_paths) - 1
    cards: list[str] = []
    for idx, event in enumerate(events):
        start = int(event.get("start_frame", event.get("frame_idx", 0)))
        end = int(event.get("end_frame", event.get("frame_idx", start)))
        start = max(0, min(start, last_idx))
        end = max(start, min(end, last_idx))
        clip_start = max(0, start - 2)
        clip_end = min(last_idx, end + 2)
        clip_indices = list(range(clip_start, clip_end + 1))
        gif_path = page_dir / "_assets" / "event_gifs" / f"event_{idx:02d}_{clip_start:03d}_{clip_end:03d}.gif"
        gif = save_gif_from_frame_paths([frame_paths[i] for i in clip_indices], gif_path, duration_ms=140)
        if gif is None:
            continue
        kind = str(event.get("kind") or event.get("window_type") or "event")
        kind_zh = event_kind_zh(kind)
        detail = event_display_label(event, record)
        cards.append(
            f"""
<section class="media-card">
  <h3>Event {idx}: {html.escape(kind_zh)}</h3>
  <p class="event-note">{html.escape(detail)} | frames {start}-{end} | gif {clip_start}-{clip_end}</p>
  <p class="event-note">含义：这张卡片展示一次记录到的碰撞事件。GIF 从事件发生前 2 帧开始，到事件结束后 2 帧结束，用来直观看到碰撞前、碰撞中和碰撞后的变化；其中 <strong>{html.escape(detail)}</strong> 表示参与碰撞的物体或物体与环境。</p>
  <img src="{html.escape(relpath(gif, page_dir))}" alt="event gif {idx}">
</section>
"""
        )
    if not cards:
        return ""
    return (
        "<section class=\"card wide\">"
        "<h2>Collision Event GIFs</h2>"
        f"<p class=\"card-note\">{html.escape(explain_card('Collision Event GIFs'))}</p>"
        "<div class=\"event-gif-grid\">"
        + "".join(cards)
        + "</div></section>"
    )


def render_frame_phase_html(record: dict[str, Any]) -> str:
    frame_phase_path = record.get("frame_phase_path")
    if frame_phase_path is None or not Path(frame_phase_path).exists():
        return ""
    phase = np.asarray(np.load(frame_phase_path), dtype=np.int32).reshape(-1)
    if phase.size == 0:
        return ""
    unique, counts = np.unique(phase, return_counts=True)
    if unique.size <= 1:
        return ""
    rows = []
    for value, count in zip(unique.tolist(), counts.tolist()):
        rows.append(
            "<tr>"
            f"<td>{int(value)}</td>"
            f"<td>{html.escape(phase_name(int(value)))}</td>"
            f"<td>{int(count)}</td>"
            "</tr>"
        )
    return (
        "<section class=\"card\">"
        "<h2>Frame Phases</h2>"
        f"<p class=\"card-note\">{html.escape(explain_card('Frame Phases'))}</p>"
        "<table><thead><tr><th>phase id</th><th>phase</th><th>frames</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
    )


def render_validation_metrics_html(validation_summary_path: Path | None) -> str:
    payload = load_json_maybe(validation_summary_path)
    if not isinstance(payload, dict):
        return ""
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return ""
    rows = []
    curated_keys = [
        "center_projection_error_px",
        "bbox_iou",
        "depth_consistency_abs",
        "velocity_smoothness",
        "anomaly",
        "anomaly_reasons",
    ]
    for key in curated_keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, dict):
            short = ", ".join(
                f"{sub_k}={value[sub_k]:.4f}" for sub_k in ("mean", "median", "p95") if isinstance(value.get(sub_k), (int, float))
            )
        elif isinstance(value, list):
            short = ", ".join(str(x) for x in value[:6])
        else:
            short = str(value)
        rows.append(f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(short)}</td></tr>")
    if not rows:
        return ""
    return (
        "<section class=\"card wide\">"
        "<h2>State Validation Metrics</h2>"
        f"<p class=\"card-note\">{html.escape(explain_card('State Validation Metrics'))}</p>"
        "<table><thead><tr><th>metric</th><th>value</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
    )


def render_reused_diagnostics_html(record: dict[str, Any], page_dir: Path) -> str:
    blocks: list[str] = []
    validation_dir = find_state_validation_case_dir(record)
    if validation_dir is not None:
        mapping = [
            ("Validation Overlay", "overlay.gif", "image"),
            ("Validation Comparisons", "comparisons.png", "image"),
        ]
        for title, name, kind in mapping:
            path = validation_dir / name
            if not path.exists():
                continue
            src = asset_src(path, page_dir)
            if kind == "video":
                blocks.append(
                    f"""
<section class="media-card">
  <h3>{html.escape(title)}</h3>
  <p class="card-note">{html.escape(explain_card(title))}</p>
  <video src="{html.escape(src)}" controls preload="metadata"></video>
</section>
"""
                )
            else:
                blocks.append(
                    f"""
<section class="media-card">
  <h3>{html.escape(title)}</h3>
  <p class="card-note">{html.escape(explain_card(title))}</p>
  <img src="{html.escape(src)}" alt="{html.escape(title)}">
</section>
"""
                )

        summary_path = validation_dir / "summary.json"
        metrics_html = render_validation_metrics_html(summary_path)
        if metrics_html:
            blocks.append(metrics_html)
    return "".join(blocks)


def render_physics_summary_html(record: dict[str, Any], page_dir: Path) -> str:
    blocks: list[str] = []
    depth_video = record["sample_dir"] / "videos" / "depth.mp4"
    if not depth_video.exists():
        depth_video = record["sample_dir"] / "visualizations" / "depth_vis.mp4"
    if depth_video.exists():
        kind, src = video_preview_src(depth_video, page_dir)
        if kind == "image":
            blocks.append(
                f"""
<section class="media-card">
  <h3>Depth Video</h3>
  <p class="card-note">{html.escape(explain_card('Depth Video'))}</p>
  <img src="{html.escape(src)}" alt="depth video gif preview">
</section>
"""
            )
        else:
            blocks.append(
                f"""
<section class="media-card">
  <h3>Depth Video</h3>
  <p class="card-note">{html.escape(explain_card('Depth Video'))}</p>
  <video src="{html.escape(src)}" controls preload="metadata"></video>
</section>
"""
            )

    trajectory_path = render_trajectory_overview(record, page_dir)
    if trajectory_path is not None and trajectory_path.exists():
        src = asset_src(trajectory_path, page_dir)
        blocks.append(
            f"""
<section class="media-card">
  <h3>Trajectory Overview</h3>
  <p class="card-note">{html.escape(explain_card('Trajectory Overview'))}</p>
  <img src="{html.escape(src)}" alt="trajectory overview">
</section>
"""
        )

    curves_path = render_state_curves(record, page_dir)
    if curves_path is not None and curves_path.exists():
        src = asset_src(curves_path, page_dir)
        blocks.append(
            f"""
<section class="media-card wide">
  <h3>State Curves And Collision Timeline</h3>
  <p class="card-note">{html.escape(explain_card('State Curves And Collision Timeline'))}</p>
  <img src="{html.escape(src)}" alt="state curves">
</section>
"""
        )

    seg_depth_path = render_segmentation_depth_overview(record, page_dir)
    if seg_depth_path is not None and seg_depth_path.exists():
        src = asset_src(seg_depth_path, page_dir)
        blocks.append(
            f"""
<section class="media-card wide">
  <h3>Segmentation And Depth Overview</h3>
  <p class="card-note">{html.escape(explain_card('Segmentation And Depth Overview'))}</p>
  <img src="{html.escape(src)}" alt="segmentation and depth overview">
</section>
"""
        )

    contact_heatmap_path = render_contact_heatmaps(record, page_dir)
    if contact_heatmap_path is not None and contact_heatmap_path.exists():
        src = asset_src(contact_heatmap_path, page_dir)
        blocks.append(
            f"""
<section class="media-card wide">
  <h3>Contact Heatmaps</h3>
  <p class="card-note">{html.escape(explain_card('Contact Heatmaps'))}</p>
  <img src="{html.escape(src)}" alt="contact heatmaps">
</section>
"""
        )
    return "".join(blocks)


def build_sample_record(group: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    sample_dir = Path(item["sample_dir"])
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        meta_path = sample_dir / "metadata.json"
    pair_meta_path = sample_dir / "pair_meta.json"
    segment_state_path = sample_dir / "segment_state.npz"
    state_pair_path = sample_dir / "state_pair.npz"
    physics_dir = sample_dir / "physics"
    view_type = str(item.get("view_type", ""))

    data_files = pick_existing(
        sample_dir,
        [
            "meta.json",
            "metadata.json",
            "pair_meta.json",
            "segment_state.npz",
            "state_pair.npz",
            "first_frame.png",
            "full_video.mp4",
            "context_video.mp4",
            "future_gt_video.mp4",
        ],
    )
    data_files.extend(
        [
            p
            for p in pick_existing(
                physics_dir,
                [
                    "anchor_targets.npz",
                    "state_9d.npy",
                    "rigid_kinematics.npz",
                    "seg.npy",
                    "depth_metric.npy",
                    "event_windows.json",
                    "contact_graph.npy",
                    "contact_impulse.npy",
                    "frame_phase.npy",
                    "energy.npz",
                    "properties.json",
                ],
            )
            if p not in data_files
        ]
    )

    return {
        "group_slug": group["slug"],
        "group_title": group["title"],
        "sample_dir": sample_dir,
        "sample_name": str(item.get("sample_name", sample_dir.name)),
        "case_name": str(item.get("case_name", "")),
        "caption": str(item.get("caption", "")),
        "detail_caption": str(item.get("detail_caption", "")),
        "dataset": str(item.get("dataset", "")),
        "view_type": view_type,
        "has_striker": bool(item.get("has_striker", False)),
        "striker_init_speed": item.get("striker_init_speed", None),
        "media": list(item.get("media", [])),
        "meta_path": meta_path if meta_path.exists() else None,
        "pair_meta_path": pair_meta_path if pair_meta_path.exists() else None,
        "segment_state_path": segment_state_path if segment_state_path.exists() else None,
        "state_pair_path": state_pair_path if state_pair_path.exists() else None,
        "anchor_targets_path": (physics_dir / "anchor_targets.npz") if (physics_dir / "anchor_targets.npz").exists() else None,
        "state_9d_path": (physics_dir / "state_9d.npy") if (physics_dir / "state_9d.npy").exists() else None,
        "rigid_kinematics_path": (physics_dir / "rigid_kinematics.npz") if (physics_dir / "rigid_kinematics.npz").exists() else None,
        "seg_path": (physics_dir / "seg.npy") if (physics_dir / "seg.npy").exists() else None,
        "depth_metric_path": (physics_dir / "depth_metric.npy") if (physics_dir / "depth_metric.npy").exists() else None,
        "event_windows_path": (physics_dir / "event_windows.json") if (physics_dir / "event_windows.json").exists() else None,
        "contact_graph_path": (physics_dir / "contact_graph.npy") if (physics_dir / "contact_graph.npy").exists() else None,
        "contact_impulse_path": (physics_dir / "contact_impulse.npy") if (physics_dir / "contact_impulse.npy").exists() else None,
        "frame_phase_path": (physics_dir / "frame_phase.npy") if (physics_dir / "frame_phase.npy").exists() else None,
        "energy_path": (physics_dir / "energy.npz") if (physics_dir / "energy.npz").exists() else None,
        "properties_path": (physics_dir / "properties.json") if (physics_dir / "properties.json").exists() else None,
        "first_frame_path": (sample_dir / "first_frame.png") if (sample_dir / "first_frame.png").exists() else None,
        "data_files": data_files,
    }


def media_html(media: list[dict[str, Any]], page_dir: Path) -> str:
    blocks = []
    for m in media:
        path = Path(str(m["path"]))
        label = str(m.get("label", "media"))
        kind = str(m.get("kind", "video"))
        is_video_path = path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
        src_kind = "video" if is_video_path else kind
        src = asset_src(path, page_dir)
        if is_video_path and PREFER_GIF:
            preview_kind, preview_src = index_asset_preview_src(path)
            src_kind = preview_kind
            src = relpath(ROOT / preview_src, page_dir)
        if src_kind == "video":
            blocks.append(
                f"""
<section class="media-card">
  <h3>{html.escape(label)}</h3>
  <p class="card-note">{html.escape(explain_card(label))}</p>
  <video src="{html.escape(src)}" controls preload="metadata"></video>
</section>
"""
            )
        else:
            blocks.append(
                f"""
<section class="media-card">
  <h3>{html.escape(label)}</h3>
  <p class="card-note">{html.escape(explain_card(label))}</p>
  <img src="{html.escape(src)}" alt="{html.escape(label)}">
</section>
"""
            )
    return "".join(blocks)


def file_list_html(record: dict[str, Any], page_dir: Path) -> str:
    rows = []
    view_type = str(record.get("view_type", ""))
    path_rows = [
        ("meta", record["meta_path"]),
        ("anchor_targets", record["anchor_targets_path"]),
        ("state_9d", record["state_9d_path"]),
        ("rigid_kinematics", record["rigid_kinematics_path"]),
        ("seg", record["seg_path"]),
        ("depth_metric", record["depth_metric_path"]),
        ("event_windows", record["event_windows_path"]),
        ("contact_graph", record["contact_graph_path"]),
        ("contact_impulse", record["contact_impulse_path"]),
        ("frame_phase", record["frame_phase_path"]),
        ("energy", record["energy_path"]),
        ("properties", record["properties_path"]),
    ]
    if view_type == "window":
        path_rows.extend(
            [
                ("pair_meta", record["pair_meta_path"]),
                ("segment_state", record["segment_state_path"]),
                ("state_pair", record["state_pair_path"]),
                ("first_frame", record["first_frame_path"]),
            ]
        )
    else:
        path_rows.extend(
            [
                ("pair_meta", "n/a for raw"),
                ("segment_state", "n/a for raw"),
                ("state_pair", "n/a for raw"),
                ("first_frame", record["first_frame_path"] if record["first_frame_path"] is not None else "n/a for raw"),
            ]
        )

    for label, path in path_rows:
        if isinstance(path, str):
            rows.append(f"<tr><td>{html.escape(label)}</td><td>{html.escape(path)}</td></tr>")
            continue
        if path is None:
            status = "not generated yet" if label == "state_9d" else "missing"
            rows.append(f"<tr><td>{html.escape(label)}</td><td>{status}</td></tr>")
        else:
            rows.append(
                f"<tr><td>{html.escape(label)}</td><td><code>{html.escape(str(path))}</code></td></tr>"
            )
    return (
        "<table><thead><tr><th>field</th><th>path</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def build_sample_page(record: dict[str, Any]) -> str:
    page_dir = ROOT / "samples" / record["group_slug"] / record["sample_name"]
    page_dir.mkdir(parents=True, exist_ok=True)
    media_blocks = media_html(record["media"], page_dir)
    physics_blocks = render_physics_summary_html(record, page_dir)
    diagnostics_blocks = render_reused_diagnostics_html(record, page_dir)
    table_html = file_list_html(record, page_dir)
    object_table_html = render_object_table_html(record)
    event_table_html = render_event_table_html(record)
    event_gif_html = render_event_gifs_html(record, page_dir)
    frame_phase_html = render_frame_phase_html(record)
    first_frame_block = ""
    if record["first_frame_path"] is not None:
        src = asset_src(record["first_frame_path"], page_dir)
        first_frame_block = f"""
<section class="media-card">
  <h3>First Frame</h3>
  <p class="card-note">{html.escape(explain_card('First Frame'))}</p>
  <img src="{html.escape(src)}" alt="first frame">
</section>
"""
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(record['sample_name'])}</title>
  <style>
    :root {{
      --bg: #f3efe8;
      --panel: #fffdf8;
      --ink: #1f1c18;
      --muted: #6d6459;
      --line: #d8cbbb;
      --accent: #8a542d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #faf6ef 0%, var(--bg) 100%);
    }}
    .wrap {{ max-width: 1480px; margin: 0 auto; padding: 20px; }}
    .hero, .card {{
      background: rgba(255,253,248,0.96);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 8px 22px rgba(45, 30, 12, 0.05);
    }}
    .hero {{ padding: 18px 20px; margin-bottom: 16px; }}
    .hero h1 {{ margin: 0 0 6px; font-size: 26px; }}
    .hero p {{ margin: 6px 0; color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .card {{ padding: 14px; }}
    .media-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
    }}
    .media-card h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .card-note {{
      margin: 0 0 8px;
      font-size: 12px;
      line-height: 1.45;
      color: var(--muted);
    }}
    .event-gif-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .event-note {{
      margin: 0 0 8px;
      font-size: 12px;
      line-height: 1.4;
      color: var(--muted);
    }}
    video, img {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #121212;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }}
    code {{
      font-size: 12px;
      background: #f8f1e8;
      padding: 2px 6px;
      border-radius: 6px;
      word-break: break-all;
    }}
    .wide {{ grid-column: 1 / -1; }}
    iframe {{
      width: 100%;
      min-height: 540px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fff;
    }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .event-gif-grid {{ grid-template-columns: 1fr; }}
      iframe {{ min-height: 420px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <p><a href="../../../index.html">Back to {html.escape(PORTAL_TITLE)}</a></p>
      <h1>{html.escape(record['sample_name'])}</h1>
      <p>{html.escape(record['group_title'])}</p>
      <p><strong>Dataset:</strong> {html.escape(record['dataset'])} | <strong>View:</strong> {html.escape(record['view_type'])}</p>
      {f"<p><strong>Striker Initial Speed:</strong> {float(record['striker_init_speed']):.4f} m/s</p>" if record.get('has_striker') and record.get('striker_init_speed') is not None else ""}
      <p><strong>Caption:</strong> {html.escape(record['caption'] or 'n/a')}</p>
      <p><strong>Detail Caption:</strong> {html.escape(record['detail_caption'] or 'n/a')}</p>
      <p><strong>Sample Dir:</strong> <code>{html.escape(str(record['sample_dir']))}</code></p>
    </section>
    <section class="grid">
      {media_blocks}
      {event_gif_html}
      {physics_blocks}
      {diagnostics_blocks}
      {first_frame_block}
      {frame_phase_html}
      {object_table_html}
      {event_table_html}
      <section class="card wide">
        <h2>Recorded Files</h2>
        <p class="card-note">{html.escape(explain_card('Recorded Files'))}</p>
        {table_html}
      </section>
    </section>
  </div>
</body>
</html>
"""
    (page_dir / "index.html").write_text(html_text, encoding="utf-8")
    return relpath(page_dir / "index.html", ROOT)


def build_group_cards(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards = []
    for group in groups:
        for item in group["items"]:
            record = build_sample_record(group, item)
            detail_page = build_sample_page(record)
            item["detail_page"] = detail_page
        cards.append(group)
    return cards


def existing_detail_page_relpath(group: dict[str, Any], item: dict[str, Any]) -> str:
    sample_name = str(item.get("sample_name") or "")
    if not sample_name:
        sample_name = Path(str(item.get("sample_dir", ""))).name
    page_path = ROOT / "samples" / str(group["slug"]) / sample_name / "index.html"
    if page_path.exists():
        return relpath(page_path, ROOT)
    return "#"


def attach_placeholder_detail_pages(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards = []
    for group in groups:
        for item in group["items"]:
            item["detail_page"] = existing_detail_page_relpath(group, item)
        cards.append(group)
    return cards


def infer_media_for_sample(sample_dir: Path) -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    candidates = [
        ("Current Sample Video", sample_dir / "full_video.mp4", "video"),
        ("Context Video", sample_dir / "context_video.mp4", "video"),
        ("Future GT Video", sample_dir / "future_gt_video.mp4", "video"),
        ("RGB Video", sample_dir / "videos" / "rgb.mp4", "video"),
        ("Depth Video", sample_dir / "videos" / "depth.mp4", "video"),
        ("Depth Visualization Video", sample_dir / "visualizations" / "depth_vis.mp4", "video"),
        ("First Frame", sample_dir / "first_frame.png", "image"),
    ]
    for label, path, kind in candidates:
        if path.exists():
            media_kind, media_src = index_asset_preview_src(path)
            media.append(
                {
                    "label": label,
                    "path": str(path),
                    "portal_src": media_src,
                    "kind": media_kind if kind == "video" else kind,
                }
            )
    return media


def load_groups_from_summary(sample_substring: str = "", collision_bucket_filter: str = "") -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    substring = str(sample_substring or "")
    collision_bucket_filter = str(collision_bucket_filter or "")
    for samples_path in sorted(SUMMARY_ROOT.rglob("samples.txt")):
        rel = samples_path.relative_to(SUMMARY_ROOT)
        if len(rel.parts) != 5:
            continue
        split, simulator_type, count_bucket, collision_bucket, _ = rel.parts
        if collision_bucket_filter and collision_bucket != collision_bucket_filter:
            continue
        lines = [line.strip() for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        items: list[dict[str, Any]] = []
        for line in lines:
            if substring and substring not in line:
                continue
            sample_dir = Path(line)
            if substring == "__rs01" and not str(sample_dir.name).endswith("__rs01"):
                continue
            if not sample_dir.exists():
                continue
            meta_path = sample_dir / "meta.json"
            if not meta_path.exists():
                meta_path = sample_dir / "metadata.json"
            meta = load_json_maybe(meta_path) or {}
            has_striker, striker_init_speed = detect_striker_info(sample_dir, meta)
            view_type = "window" if (sample_dir / "pair_meta.json").exists() or (sample_dir / "segment_state.npz").exists() else "raw"
            items.append(
                {
                    "sample_dir": str(sample_dir),
                    "sample_name": str(meta.get("scene_id") or sample_dir.name),
                    "case_name": str(meta.get("case_name") or ""),
                    "caption": str(meta.get("caption") or ""),
                    "detail_caption": str(meta.get("detail_caption") or ""),
                    "dataset": str(meta.get("dataset") or "GenesisRigid"),
                    "view_type": view_type,
                    "has_striker": bool(has_striker),
                    "striker_init_speed": striker_init_speed,
                    "media": infer_media_for_sample(sample_dir),
                }
            )
            if len(items) >= MAX_ITEMS_PER_GROUP:
                break
        if not items:
            continue
        slug = f"{split}__{simulator_type}__{count_bucket}__{collision_bucket}"
        title = f"{split} / {simulator_type} / {count_bucket} / {collision_bucket}"
        groups.append(
            {
                "slug": slug,
                "title": title,
                "split": split,
                "simulator_type": simulator_type,
                "count_bucket": count_bucket,
                "collision_bucket": collision_bucket,
                "items": items,
                "total": len(items),
            }
        )
    return groups


def build_nav_tree(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tree: dict[str, Any] = {}
    for group in groups:
        split = str(group.get("split", "unknown"))
        simulator = str(group.get("simulator_type", "unknown"))
        count_bucket = str(group.get("count_bucket", "unknown"))
        collision_bucket = str(group.get("collision_bucket", "unknown"))
        split_node = tree.setdefault(split, {})
        sim_node = split_node.setdefault(simulator, {})
        count_node = sim_node.setdefault(count_bucket, {})
        count_node[collision_bucket] = {
            "slug": group["slug"],
            "title": group["title"],
            "shown": int(len(group.get("items", []))),
            "total": int(group.get("total", len(group.get("items", [])))),
        }

    result = []
    for split, split_node in sorted(tree.items()):
        split_entry = {"name": split, "children": []}
        for simulator, sim_node in sorted(split_node.items()):
            sim_entry = {"name": simulator, "children": []}
            for count_bucket, count_node in sorted(sim_node.items()):
                count_entry = {"name": count_bucket, "children": []}
                for collision_bucket, leaf in sorted(count_node.items()):
                    count_entry["children"].append(
                        {
                            "name": collision_bucket,
                            "slug": leaf["slug"],
                            "title": leaf["title"],
                            "shown": leaf["shown"],
                            "total": leaf["total"],
                        }
                    )
                sim_entry["children"].append(count_entry)
            split_entry["children"].append(sim_entry)
        result.append(split_entry)
    return result


def card_media_preview(item: dict[str, Any]) -> str:
    media = item.get("media", [])
    if not media:
        return "<p class='empty'>No exported media.</p>"
    blocks = []
    for m in media:
        label = str(m.get("label", "media"))
        kind = str(m.get("kind", "video"))
        path = Path(str(m["path"]))
        src = relpath(path, ROOT)
        if kind == "video":
            blocks.append(
                f"""
<div class="media-block">
  <div class="media-label">{html.escape(label)}</div>
  <video src="{html.escape(src)}" controls preload="metadata"></video>
</div>
"""
            )
        else:
            blocks.append(
                f"""
<div class="media-block">
  <div class="media-label">{html.escape(label)}</div>
  <img src="{html.escape(src)}" alt="{html.escape(label)}">
</div>
"""
            )
    return "".join(blocks)


def build_index(groups: list[dict[str, Any]]) -> str:
    payload_json = json.dumps(groups, ensure_ascii=False)
    nav_tree_json = json.dumps(build_nav_tree(groups), ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(PORTAL_TITLE)}</title>
  <style>
    :root {{
      --bg: #f3efe8;
      --panel: #fffdf8;
      --ink: #1f1c18;
      --muted: #6d6459;
      --line: #d8cbbb;
      --accent: #8a542d;
      --accent-soft: #f7ebdd;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(138,84,45,0.10), transparent 24%),
        linear-gradient(180deg, #faf6ef 0%, var(--bg) 100%);
    }}
    header {{
      background: rgba(255,253,248,0.94);
      border-bottom: 1px solid var(--line);
      padding: 14px 18px;
    }}
    h1 {{ margin: 0 0 6px; font-size: 26px; }}
    .sub {{ margin: 0; color: var(--muted); font-size: 13px; }}
    .layout {{
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      min-height: calc(100vh - 76px);
    }}
    aside {{
      position: sticky;
      top: 0;
      align-self: start;
      height: calc(100vh - 1px);
      overflow: auto;
      border-right: 1px solid var(--line);
      background: rgba(255, 251, 244, 0.95);
      backdrop-filter: blur(10px);
      padding: 14px 12px 18px;
    }}
    .sidebar-search {{
      width: 100%;
      margin: 0 0 10px;
      padding: 9px 10px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fffdf8;
      color: var(--ink);
    }}
    .tree-root, .tree-children, .tree-list {{
      display: grid;
      gap: 8px;
    }}
    .tree-root {{ gap: 10px; }}
    .branch {{
      border: 1px solid #e7d8c5;
      border-radius: 12px;
      background: #fffaf3;
      overflow: hidden;
    }}
    .branch-toggle {{
      width: 100%;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      padding: 10px 11px;
      border: 0;
      background: #fff4e7;
      color: var(--ink);
      cursor: pointer;
      font-weight: 600;
      text-transform: none;
    }}
    .branch-toggle:hover {{ background: #ffedd8; }}
    .tree-children {{
      padding: 8px;
    }}
    .tree-list {{
      list-style: none;
      margin: 0;
      padding: 8px;
    }}
    .branch-count, .leaf-count {{
      min-width: 26px;
      text-align: center;
      padding: 2px 8px;
      border-radius: 999px;
      background: rgba(138,84,45,0.12);
      font-size: 12px;
      color: var(--accent);
    }}
    .leaf {{
      width: 100%;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      text-align: left;
      padding: 9px 10px;
      border-radius: 10px;
      border: 1px solid #eadcca;
      background: #fffdf8;
      color: var(--ink);
      cursor: pointer;
    }}
    .leaf.active {{
      border-color: var(--accent);
      background: #fcefe0;
    }}
    .leaf-name {{ word-break: break-word; line-height: 1.25; }}
    .is-collapsed > .tree-children,
    .is-collapsed > .tree-list {{ display: none; }}
    main {{
      padding: 16px 18px 28px;
      min-width: 0;
    }}
    .group {{ display: none; }}
    .group.active {{ display: block; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      box-shadow: 0 8px 20px rgba(45, 30, 12, 0.04);
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: start;
    }}
    .card h3 {{
      margin: 0;
      font-size: 15px;
      line-height: 1.25;
      flex: 1;
      min-width: 0;
    }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid #ead8c7;
      padding: 2px 8px;
      font-size: 11px;
      white-space: nowrap;
    }}
    .caption, .detail, .path {{
      margin: 8px 0;
      font-size: 12px;
      line-height: 1.45;
      color: var(--muted);
      word-break: break-word;
    }}
    .striker-note {{
      margin: 8px 0;
      font-size: 12px;
      line-height: 1.45;
      color: var(--accent);
      font-weight: 600;
    }}
    .detail-toggle {{
      margin-top: 4px;
      background: none;
      border: 0;
      color: var(--accent);
      cursor: pointer;
      padding: 0;
      font-size: 12px;
    }}
    .detail-text.is-collapsed {{
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }}
    .media-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      margin-top: 8px;
    }}
    .media-block {{
      background: #faf5ee;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 8px;
    }}
    .media-label {{
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    video, img {{
      width: 100%;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #111;
    }}
    .actions {{
      margin-top: 10px;
      display: flex;
      justify-content: flex-end;
    }}
    .detail-link {{
      color: var(--accent);
      font-weight: 700;
      text-decoration: none;
    }}
    .empty {{ color: var(--muted); font-size: 12px; }}
    @media (max-width: 1100px) {{
      .layout {{ grid-template-columns: 1fr; }}
      aside {{
        position: static;
        height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }}
      .cards {{ grid-template-columns: 1fr; }}
      .media-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(PORTAL_TITLE)}</h1>
    <p class="sub">主页面保留 RGB 入口，每个样本新增详情页，按样本实际记录的数据形式展示。</p>
  </header>
  <div class="layout">
    <aside>
      <input id="search" class="sidebar-search" placeholder="Search sample / caption / slug">
      <div id="list" class="tree-list"></div>
    </aside>
    <main>
      <div id="groups"></div>
    </main>
  </div>
  <script>
    const GROUPS = {payload_json};
    const NAV_TREE = {nav_tree_json};
    const listEl = document.getElementById('list');
    const groupsEl = document.getElementById('groups');
    const searchEl = document.getElementById('search');

    function mediaHtml(item) {{
      const media = Array.isArray(item.media) ? item.media : [];
      if (!media.length) return '<p class="empty">No exported media.</p>';
      let selected = media.find((m) => m.label === 'RGB Video');
      if (!selected) {{
        selected = media.find((m) => m.label === 'Current Sample Video')
          || media.find((m) => m.kind === 'image')
          || media.find((m) => m.kind === 'video')
          || media[0];
      }}
      const src = selected.portal_src || selected.path;
      if (selected.kind === 'video') {{
        return `<div class="media-grid"><div class="media-block"><div class="media-label">${{selected.label}}</div><video src="${{src}}" controls preload="metadata"></video></div></div>`;
      }}
      return `<div class="media-grid"><div class="media-block"><div class="media-label">${{selected.label}}</div><img src="${{src}}" alt="${{selected.label}}"></div></div>`;
    }}

    function filterTree(nodes, allowedSlugs) {{
      return nodes.map((node) => {{
        if (node.slug) {{
          return allowedSlugs.has(node.slug) ? node : null;
        }}
        const children = filterTree(node.children || [], allowedSlugs).filter(Boolean);
        if (!children.length) return null;
        return {{...node, children}};
      }}).filter(Boolean);
    }}

    function countLeafSlugs(node) {{
      if (node.slug) return 1;
      return (node.children || []).reduce((acc, child) => acc + countLeafSlugs(child), 0);
    }}

    function renderTree(nodes, activeSlug, level=0) {{
      if (!nodes.length) return '';
      const containerClass = level === 0 ? 'tree-root' : (level < 3 ? 'tree-children' : 'tree-list');
      const blocks = nodes.map((node) => {{
        if (node.slug) {{
          const active = node.slug === activeSlug ? 'active' : '';
          return `
            <button class="leaf ${{active}}" data-target="${{node.slug}}">
              <div class="leaf-name">${{node.name}}</div>
              <div class="leaf-count">${{node.shown}} / ${{node.total}}</div>
            </button>
          `;
        }}
        const childCount = countLeafSlugs(node);
        const branchClass = level > 0 ? 'branch is-collapsed' : 'branch';
        return `
          <section class="${{branchClass}}">
            <button class="branch-toggle" type="button">
              <span>${{node.name}}</span>
              <span class="branch-count">${{childCount}}</span>
            </button>
            ${{renderTree(node.children || [], activeSlug, level + 1)}}
          </section>
        `;
      }}).join('');
      return `<div class="${{containerClass}}">${{blocks}}</div>`;
    }}

    function render(filterText='') {{
      const query = filterText.trim().toLowerCase();
      const filtered = GROUPS.map((group) => {{
        const items = group.items.filter((item) => {{
          if (!query) return true;
          const hay = [
            group.slug, group.title, group.subtitle,
            item.sample_name, item.case_name, item.caption, item.detail_caption, item.dataset, item.view_type, item.sample_dir
          ].join(' ').toLowerCase();
          return hay.includes(query);
        }});
        return {{...group, filteredItems: items}};
      }}).filter((group) => group.filteredItems.length > 0);

      const activeSlug = filtered.length ? filtered[0].slug : '';
      const allowedSlugs = new Set(filtered.map((group) => group.slug));
      const filteredTree = filterTree(NAV_TREE, allowedSlugs);
      listEl.innerHTML = renderTree(filteredTree, activeSlug);

      groupsEl.innerHTML = filtered.map((group, idx) => `
        <section class="group ${{idx===0 ? 'active' : ''}}" id="group-${{group.slug}}">
          <div class="card" style="margin-bottom:12px;">
            <h2 style="margin:0 0 6px;">${{group.title}}</h2>
            <p class="caption">${{group.subtitle}}</p>
          </div>
          <div class="cards">
            ${{group.filteredItems.map((item) => `
              <article class="card">
                <div class="card-head">
                  <h3>${{item.sample_name}}</h3>
                  <span class="pill">${{item.view_type}}</span>
                </div>
                <p class="caption"><strong>Caption:</strong> ${{item.caption || 'n/a'}}</p>
                ${{item.has_striker && item.striker_init_speed != null ? `<p class="striker-note">Striker Initial Speed: ${{Number(item.striker_init_speed).toFixed(4)}} m/s</p>` : ''}}
                <div class="detail">
                  <strong>Detail:</strong>
                  <div class="detail-text is-collapsed">${{item.detail_caption || 'n/a'}}</div>
                  <button class="detail-toggle" type="button">Expand</button>
                </div>
                <p class="path">${{item.sample_dir}}</p>
                ${{mediaHtml(item)}}
                <div class="actions">
                  <a class="detail-link" href="${{item.detail_page}}">查看详情页</a>
                </div>
              </article>
            `).join('')}}
          </div>
        </section>
      `).join('');

      bindInteractions();
    }}

    function bindInteractions() {{
      document.querySelectorAll('.leaf').forEach((btn) => {{
        btn.addEventListener('click', () => {{
          document.querySelectorAll('.leaf').forEach((x) => x.classList.remove('active'));
          document.querySelectorAll('.group').forEach((x) => x.classList.remove('active'));
          btn.classList.add('active');
          const target = document.getElementById('group-' + btn.dataset.target);
          if (target) target.classList.add('active');
        }});
      }});
      document.querySelectorAll('.branch-toggle').forEach((btn) => {{
        btn.addEventListener('click', () => {{
          const branch = btn.parentElement;
          branch.classList.toggle('is-collapsed');
        }});
      }});
      document.querySelectorAll('.detail-toggle').forEach((btn) => {{
        btn.addEventListener('click', () => {{
          const text = btn.previousElementSibling;
          const collapsed = text.classList.toggle('is-collapsed');
          btn.textContent = collapsed ? 'Expand' : 'Collapse';
        }});
      }});
    }}

    searchEl.addEventListener('input', () => render(searchEl.value));
    render('');
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    global ROOT, SUMMARY_ROOT, PORTAL_TITLE, PREFER_GIF, MANIFEST_PATH, INDEX_ASSET_ROOT
    ROOT = args.output_root.resolve()
    SUMMARY_ROOT = args.summary_root.resolve()
    PORTAL_TITLE = str(args.portal_title)
    PREFER_GIF = bool(args.prefer_gif)
    MANIFEST_PATH = ROOT / "manifest.json"
    INDEX_ASSET_ROOT = ROOT / "index_assets"
    ROOT.mkdir(parents=True, exist_ok=True)
    groups = load_groups_from_summary(
        sample_substring=args.sample_substring,
        collision_bucket_filter=args.collision_bucket,
    )
    if args.index_only:
        groups = attach_placeholder_detail_pages(groups)
    else:
        groups = build_group_cards(groups)
    (ROOT / "index.html").write_text(build_index(groups), encoding="utf-8")
    (ROOT / "manifest.json").write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
    print(ROOT / "index.html")


if __name__ == "__main__":
    main()
