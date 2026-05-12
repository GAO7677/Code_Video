#!/usr/bin/env python3
"""Build a portal with per-sample detail pages for TDW Genesis-style exports."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_INPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_genesis_format_exports")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_genesis_format_portal")
PORTAL_TITLE = "TDW Genesis Export Portal"
PREFER_GIF = False
ROOT = DEFAULT_OUTPUT_ROOT
INPUT_ROOT = DEFAULT_INPUT_ROOT
MANIFEST_PATH = ROOT / "manifest.json"
INDEX_ASSET_ROOT = ROOT / "index_assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a portal with per-sample detail pages for TDW Genesis-style exports.")
    parser.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT, help="TDW export root.")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Portal output root.")
    parser.add_argument("--portal_title", type=str, default=PORTAL_TITLE, help="Portal title.")
    parser.add_argument("--prefer_gif", action="store_true", help="Use GIF preview where possible.")
    parser.add_argument("--sample_substring", type=str, default="", help="Only include samples whose absolute path contains this substring.")
    parser.add_argument("--index_only", action="store_true", help="Only rebuild the index and manifest without regenerating detail pages.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_maybe(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return load_json(path)


def relpath(path: Path, start: Path) -> str:
    return os.path.relpath(path, start)


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


def video_preview_src(video_path: Path, page_dir: Path) -> tuple[str, str]:
    video_src = asset_src(video_path, page_dir)
    gif_path = page_dir / "_assets" / f"{hashlib.md5(str(video_path).encode('utf-8')).hexdigest()[:12]}_{video_path.stem}.gif"
    if (not gif_path.exists()) and video_path.exists():
        make_gif_from_video(video_path, gif_path)
    if gif_path.exists():
        return "image", relpath(gif_path, page_dir)
    return "video", video_src


def explain_card(title: str) -> str:
    mapping = {
        "Objects": "含义：列出当前样本中的物体实例、分割编号、角色和运动类型。",
        "Recorded Events": "含义：列出当前样本记录到的接触事件起止帧和事件类型。",
        "Collision Event GIFs": "含义：把每个碰撞或接触事件截成短 GIF，便于快速查看发生过程。",
        "Frame Phases": "含义：统计每个阶段标签占多少帧，用来快速判断样本主要处于什么物理阶段。",
        "Depth Video": "含义：展示深度视频或深度可视化视频，用来检查几何和远近变化。",
        "Trajectory Overview": "含义：把各物体的 2D 轨迹画在首帧背景上，便于查看整体运动路径。",
        "State Curves And Collision Timeline": "含义：展示中心速度、深度和可见性随时间的变化，并叠加事件时间轴。",
        "Segmentation And Depth Overview": "含义：抽取若干关键帧，同时展示 RGB+分割和深度图。",
        "Contact Heatmaps": "含义：把接触图和接触冲量整理成热力图，观察哪些物体在何时发生接触。",
        "Energy Curves": "含义：展示导出的能量相关曲线，用来检查动力学趋势是否合理。",
        "First Frame": "含义：展示样本第一帧，快速确认初始视角和物体布局。",
        "Recorded Files": "含义：列出当前样本已经导出的文件路径，方便核对数据完整性。",
        "RGB Video": "含义：展示当前样本的 RGB 视频。",
        "Depth Visualization Video": "含义：展示当前样本的深度伪彩视频。",
    }
    return mapping.get(title, "含义：这张卡片展示当前样本的一类记录结果。")


def build_sample_record(group: dict[str, Any], sample_dir: Path) -> dict[str, Any]:
    meta_path = sample_dir / "meta.json"
    physics_dir = sample_dir / "physics"
    meta = load_json(meta_path)
    first_frame_path = sample_dir / "rgb" / "frame_000.png"
    media = infer_media_for_sample(sample_dir)
    return {
        "group_slug": group["slug"],
        "group_title": group["title"],
        "sample_dir": sample_dir,
        "sample_name": str(meta.get("scene_id") or sample_dir.name),
        "case_name": str(meta.get("case_name") or ""),
        "dataset": str((meta.get("simulation") or {}).get("engine") or "TDW"),
        "view_type": "raw",
        "media": media,
        "meta_path": meta_path,
        "anchor_targets_path": (physics_dir / "anchor_targets.npz") if (physics_dir / "anchor_targets.npz").exists() else None,
        "rigid_kinematics_path": (physics_dir / "rigid_kinematics.npz") if (physics_dir / "rigid_kinematics.npz").exists() else None,
        "seg_path": (physics_dir / "seg.npy") if (physics_dir / "seg.npy").exists() else None,
        "depth_metric_path": (physics_dir / "depth_metric.npy") if (physics_dir / "depth_metric.npy").exists() else None,
        "event_windows_path": (physics_dir / "event_windows.json") if (physics_dir / "event_windows.json").exists() else None,
        "contact_graph_path": (physics_dir / "contact_graph.npy") if (physics_dir / "contact_graph.npy").exists() else None,
        "contact_impulse_path": (physics_dir / "contact_impulse.npy") if (physics_dir / "contact_impulse.npy").exists() else None,
        "frame_phase_path": (physics_dir / "frame_phase.npy") if (physics_dir / "frame_phase.npy").exists() else None,
        "energy_path": (physics_dir / "energy.npz") if (physics_dir / "energy.npz").exists() else None,
        "properties_path": (physics_dir / "properties.json") if (physics_dir / "properties.json").exists() else None,
        "scene_input_path": sample_dir / "scene_input.json",
        "first_frame_path": first_frame_path if first_frame_path.exists() else None,
    }


def infer_media_for_sample(sample_dir: Path) -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    candidates = [
        ("RGB Video", sample_dir / "videos" / "rgb.mp4", "video"),
        ("Depth Video", sample_dir / "videos" / "depth.mp4", "video"),
        ("Depth Visualization Video", sample_dir / "visualizations" / "depth_vis.mp4", "video"),
        ("First Frame", sample_dir / "rgb" / "frame_000.png", "image"),
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


def load_state_bundle(record: dict[str, Any]) -> dict[str, Any] | None:
    meta = load_json(record["meta_path"])
    kin_path = record.get("rigid_kinematics_path")
    if kin_path is None or not Path(kin_path).exists():
        return None
    payload = np.load(kin_path, allow_pickle=True)
    object_ids = np.asarray(payload["object_ids"], dtype=np.int32)
    seg_ids = np.asarray(payload["seg_ids"], dtype=np.int32)
    com_uv = np.asarray(payload["com_uv"], dtype=np.float32)
    visibility_mask = np.asarray(payload["visibility_mask"], dtype=np.uint8)
    bbox_xyxy = np.asarray(payload["bbox_xyxy"], dtype=np.float32)
    center_depth = None
    anchor_path = record.get("anchor_targets_path")
    if anchor_path is not None and Path(anchor_path).exists():
        anchor_payload = np.load(anchor_path, allow_pickle=True)
        center_depth = np.asarray(anchor_payload["center_depth"], dtype=np.float32)
    if center_depth is None:
        center_depth = np.zeros(com_uv.shape[:2], dtype=np.float32)
    x1 = bbox_xyxy[..., 0]
    y1 = bbox_xyxy[..., 1]
    x2 = bbox_xyxy[..., 2]
    y2 = bbox_xyxy[..., 3]
    width = np.maximum(0.0, x2 - x1).astype(np.float32)
    height = np.maximum(0.0, y2 - y1).astype(np.float32)
    u = com_uv[..., 0]
    v = com_uv[..., 1]
    d = center_depth.astype(np.float32)
    fps = float(((meta.get("simulation") or {}).get("video_fps")) or 24.0)
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
    path = record.get("event_windows_path")
    if path is None or not Path(path).exists():
        return []
    payload = load_json(Path(path))
    return payload if isinstance(payload, list) else []


def phase_name(value: int) -> str:
    mapping = {
        0: "unknown",
        1: "contact_or_motion",
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
        seg_id = int(seg_ids[idx]) if idx < seg_ids.shape[0] else idx + 1
        obj = by_object_id.get(object_id, {})
        name = str(obj.get("source_object_id") or obj.get("name") or f"obj{object_id}")
        labels.append(f"{name} (obj={object_id}, seg={seg_id})")
    return labels


def object_name_map(record: dict[str, Any]) -> dict[int, str]:
    meta = load_json(record["meta_path"])
    objects = meta.get("objects") if isinstance(meta.get("objects"), list) else []
    mapping: dict[int, str] = {}
    for obj in objects:
        if not isinstance(obj, dict) or obj.get("object_id") is None:
            continue
        obj_id = int(obj["object_id"])
        name = str(obj.get("source_object_id") or obj.get("name") or f"obj{obj_id}")
        role = str(obj.get("role", "")).strip()
        mapping[obj_id] = f"{role}:{name}" if role else name
    return mapping


def event_display_label(event: dict[str, Any], record: dict[str, Any]) -> str:
    obj_names = object_name_map(record)
    objects = load_json(record["meta_path"]).get("objects") or []
    ordered_object_ids = [int(obj["object_id"]) for obj in objects if isinstance(obj, dict) and obj.get("object_id") is not None]
    participants = event.get("participants", [])
    labels: list[str] = []
    for raw in participants:
        obj_id = int(raw)
        if obj_id < 0:
            labels.append(str(event.get("environment_name") or "environment"))
        elif obj_id in obj_names:
            labels.append(obj_names[obj_id])
        elif obj_id >= 0 and obj_id < len(ordered_object_ids):
            resolved = ordered_object_ids[obj_id]
            labels.append(obj_names.get(resolved, f"obj{resolved}"))
        else:
            labels.append(f"obj{obj_id}")
    return " <-> ".join(labels) if labels else "event"


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
    panels: list[tuple[str, np.ndarray]] = []
    if graph is not None:
        panels.append(("Contact Graph", graph.reshape(graph.shape[0], -1)))
    if impulse is not None:
        panels.append(("Contact Impulse", impulse.reshape(impulse.shape[0], -1)))
    if not panels:
        return None
    if all(float(np.nanmax(arr)) <= 0.0 for _, arr in panels):
        return None
    out_path = page_dir / "contact_heatmaps.png"
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
    if energy_path is None or not Path(energy_path).exists():
        return None
    payload = np.load(energy_path, allow_pickle=True)
    key_order = ["kinetic_trans", "kinetic_rot", "potential_gravity", "mechanical_total"]
    series: list[tuple[str, np.ndarray]] = []
    for key in key_order:
        if key in payload:
            values = np.asarray(payload[key], dtype=np.float32)
            if values.ndim == 1 and values.size > 0:
                series.append((key, values))
    if not series:
        return None

    out_path = page_dir / "energy_curves.png"
    fig, ax = plt.subplots(figsize=(8.6, 3.8), dpi=150)
    t = np.arange(series[0][1].shape[0], dtype=np.int32)
    for label, values in series:
        ax.plot(t, values, linewidth=2.0, label=label)
    ax.set_title("Energy / Motion Curves", loc="left", fontsize=10)
    ax.set_xlabel("frame index")
    ax.set_ylabel("value")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_object_table_html(record: dict[str, Any]) -> str:
    meta = load_json(record["meta_path"])
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
            f"<td>{html.escape(str(obj.get('source_object_id') or obj.get('name') or 'n/a'))}</td>"
            f"<td>{html.escape(str(obj.get('role', 'n/a')))}</td>"
            f"<td>{html.escape(str(obj.get('motion_type', obj.get('motion_group', 'n/a'))))}</td>"
            "</tr>"
        )
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
            f"<td>{html.escape(event_display_label(event, record))}</td>"
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
        gif_path = page_dir / "_assets" / "event_gifs" / f"event_{idx:02d}_{clip_start:03d}_{clip_end:03d}.gif"
        gif = save_gif_from_frame_paths([frame_paths[i] for i in range(clip_start, clip_end + 1)], gif_path, duration_ms=140)
        if gif is None:
            continue
        detail = event_display_label(event, record)
        cards.append(
            f"""
<section class="media-card">
  <h3>Event {idx}</h3>
  <p class="event-note">{html.escape(detail)} | frames {start}-{end} | gif {clip_start}-{clip_end}</p>
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


def file_list_html(record: dict[str, Any]) -> str:
    path_rows = [
        ("meta", record["meta_path"]),
        ("scene_input", record["scene_input_path"] if record["scene_input_path"].exists() else None),
        ("anchor_targets", record["anchor_targets_path"]),
        ("rigid_kinematics", record["rigid_kinematics_path"]),
        ("seg", record["seg_path"]),
        ("depth_metric", record["depth_metric_path"]),
        ("event_windows", record["event_windows_path"]),
        ("contact_graph", record["contact_graph_path"]),
        ("contact_impulse", record["contact_impulse_path"]),
        ("frame_phase", record["frame_phase_path"]),
        ("energy", record["energy_path"]),
        ("properties", record["properties_path"]),
        ("first_frame", record["first_frame_path"]),
    ]
    rows = []
    for label, path in path_rows:
        if path is None:
            rows.append(f"<tr><td>{html.escape(label)}</td><td>missing</td></tr>")
        else:
            rows.append(f"<tr><td>{html.escape(label)}</td><td><code>{html.escape(str(path))}</code></td></tr>")
    return (
        "<table><thead><tr><th>field</th><th>path</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def media_html(media: list[dict[str, Any]], page_dir: Path) -> str:
    blocks = []
    for item in media:
        path = Path(str(item["path"]))
        label = str(item.get("label", "media"))
        kind = str(item.get("kind", "video"))
        src = asset_src(path, page_dir)
        if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"} and PREFER_GIF:
            preview_kind, preview_src = index_asset_preview_src(path)
            kind = preview_kind
            src = relpath(ROOT / preview_src, page_dir)
        if kind == "video":
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


def render_physics_summary_html(record: dict[str, Any], page_dir: Path) -> str:
    blocks: list[str] = []
    depth_video = record["sample_dir"] / "videos" / "depth.mp4"
    if not depth_video.exists():
        depth_video = record["sample_dir"] / "visualizations" / "depth_vis.mp4"
    if depth_video.exists():
        kind, src = video_preview_src(depth_video, page_dir)
        tag = f'<img src="{html.escape(src)}" alt="depth video gif preview">' if kind == "image" else f'<video src="{html.escape(src)}" controls preload="metadata"></video>'
        blocks.append(
            f"""
<section class="media-card">
  <h3>Depth Video</h3>
  <p class="card-note">{html.escape(explain_card('Depth Video'))}</p>
  {tag}
</section>
"""
        )

    for title, path in [
        ("Trajectory Overview", render_trajectory_overview(record, page_dir)),
        ("State Curves And Collision Timeline", render_state_curves(record, page_dir)),
        ("Segmentation And Depth Overview", render_segmentation_depth_overview(record, page_dir)),
        ("Contact Heatmaps", render_contact_heatmaps(record, page_dir)),
        ("Energy Curves", render_energy_plot(record, page_dir)),
    ]:
        if path is None or not path.exists():
            continue
        src = asset_src(path, page_dir)
        wide = " wide" if "Timeline" in title or "Overview" in title or "Heatmaps" in title else ""
        blocks.append(
            f"""
<section class="media-card{wide}">
  <h3>{html.escape(title)}</h3>
  <p class="card-note">{html.escape(explain_card(title))}</p>
  <img src="{html.escape(src)}" alt="{html.escape(title)}">
</section>
"""
        )
    return "".join(blocks)


def build_sample_page(record: dict[str, Any]) -> str:
    page_dir = ROOT / "samples" / record["group_slug"] / record["sample_name"]
    page_dir.mkdir(parents=True, exist_ok=True)
    meta = load_json(record["meta_path"])
    simulation = meta.get("simulation") or {}
    media_blocks = media_html(record["media"], page_dir)
    physics_blocks = render_physics_summary_html(record, page_dir)
    object_table = render_object_table_html(record)
    event_table = render_event_table_html(record)
    event_gifs = render_event_gifs_html(record, page_dir)
    frame_phase = render_frame_phase_html(record)
    file_table = file_list_html(record)
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
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .event-gif-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <p><a href="../../../index.html">Back to {html.escape(PORTAL_TITLE)}</a></p>
      <h1>{html.escape(record['sample_name'])}</h1>
      <p>{html.escape(record['group_title'])}</p>
      <p><strong>Case:</strong> {html.escape(record['case_name'])} | <strong>Dataset:</strong> {html.escape(record['dataset'])} | <strong>View:</strong> raw</p>
      <p><strong>Frames:</strong> {html.escape(str(meta.get('frames', 'n/a')))} | <strong>Objects:</strong> {html.escape(str(meta.get('num_objects', 'n/a')))} | <strong>FPS:</strong> {html.escape(str(simulation.get('video_fps', 'n/a')))}</p>
      <p><strong>Sample Dir:</strong> <code>{html.escape(str(record['sample_dir']))}</code></p>
    </section>
    <section class="grid">
      {media_blocks}
      {event_gifs}
      {physics_blocks}
      {first_frame_block}
      {frame_phase}
      {object_table}
      {event_table}
      <section class="card wide">
        <h2>Recorded Files</h2>
        <p class="card-note">{html.escape(explain_card('Recorded Files'))}</p>
        {file_table}
      </section>
    </section>
  </div>
</body>
</html>
"""
    (page_dir / "index.html").write_text(html_text, encoding="utf-8")
    return relpath(page_dir / "index.html", ROOT)


def existing_detail_page_relpath(group: dict[str, Any], item: dict[str, Any]) -> str:
    page_path = ROOT / "samples" / str(group["slug"]) / str(item["sample_name"]) / "index.html"
    return relpath(page_path, ROOT) if page_path.exists() else "#"


def build_group_cards(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for group in groups:
        for item in group["items"]:
            record = build_sample_record(group, Path(item["sample_dir"]))
            item["detail_page"] = build_sample_page(record)
        out.append(group)
    return out


def attach_placeholder_detail_pages(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for group in groups:
        for item in group["items"]:
            item["detail_page"] = existing_detail_page_relpath(group, item)
        out.append(group)
    return out


def load_groups(sample_substring: str = "") -> list[dict[str, Any]]:
    groups_map: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for meta_path in sorted(INPUT_ROOT.glob("train/rigid/*/*/*/meta.json")):
        sample_dir = meta_path.parent
        sample_dir_str = str(sample_dir)
        if sample_substring and sample_substring not in sample_dir_str:
            continue
        meta = load_json(meta_path)
        split = str(meta.get("split") or "train")
        simulator_type = str(meta.get("simulator_type") or "unknown")
        scene_composition = str(meta.get("scene_composition") or "unknown_scene")
        object_count_bucket = str(meta.get("object_count_bucket") or "unknown_count")
        key = (split, simulator_type, scene_composition, object_count_bucket)
        groups_map.setdefault(key, []).append(
            {
                "sample_dir": sample_dir_str,
                "sample_name": str(meta.get("scene_id") or sample_dir.name),
                "case_name": str(meta.get("case_name") or ""),
                "caption": str(meta.get("interaction_pattern") or ""),
                "detail_caption": f"scene={scene_composition} | category={meta.get('motion_category', 'n/a')} | objects={meta.get('num_objects', 'n/a')}",
                "dataset": str((meta.get("simulation") or {}).get("engine") or "TDW"),
                "view_type": "raw",
                "media": infer_media_for_sample(sample_dir),
            }
        )
    groups: list[dict[str, Any]] = []
    for split, simulator_type, scene_composition, object_count_bucket in sorted(groups_map.keys()):
        items = sorted(groups_map[(split, simulator_type, scene_composition, object_count_bucket)], key=lambda x: x["sample_name"])
        slug = f"{split}__{simulator_type}__{scene_composition}__{object_count_bucket}"
        title = f"{split} / {simulator_type} / {scene_composition} / {object_count_bucket}"
        groups.append(
            {
                "slug": slug,
                "title": title,
                "subtitle": f"{len(items)} sample(s) | scene={scene_composition} | kind={simulator_type}",
                "split": split,
                "simulator_type": simulator_type,
                "scene_composition": scene_composition,
                "object_count_bucket": object_count_bucket,
                "items": items,
                "total": len(items),
            }
        )
    return groups


def build_nav_tree(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tree: dict[str, Any] = {}
    for group in groups:
        split = str(group["split"])
        simulator = str(group["simulator_type"])
        scene = str(group["scene_composition"])
        count_bucket = str(group["object_count_bucket"])
        split_node = tree.setdefault(split, {})
        sim_node = split_node.setdefault(simulator, {})
        scene_node = sim_node.setdefault(scene, {})
        scene_node[count_bucket] = {
            "slug": group["slug"],
            "title": group["title"],
            "shown": int(len(group["items"])),
            "total": int(group["total"]),
        }
    result = []
    for split, split_node in sorted(tree.items()):
        split_entry = {"name": split, "children": []}
        for simulator, sim_node in sorted(split_node.items()):
            sim_entry = {"name": simulator, "children": []}
            for scene, scene_node in sorted(sim_node.items()):
                scene_entry = {"name": scene, "children": []}
                for count_bucket, leaf in sorted(scene_node.items()):
                    scene_entry["children"].append(
                        {
                            "name": count_bucket,
                            "slug": leaf["slug"],
                            "title": leaf["title"],
                            "shown": leaf["shown"],
                            "total": leaf["total"],
                        }
                    )
                sim_entry["children"].append(scene_entry)
            split_entry["children"].append(sim_entry)
        result.append(split_entry)
    return result


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
    }}
    .tree-children {{ padding: 8px; }}
    .tree-list {{ list-style: none; margin: 0; padding: 8px; }}
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
    .leaf.active {{ border-color: var(--accent); background: #fcefe0; }}
    .leaf-name {{ word-break: break-word; line-height: 1.25; }}
    .is-collapsed > .tree-children,
    .is-collapsed > .tree-list {{ display: none; }}
    main {{ padding: 16px 18px 28px; min-width: 0; }}
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
    .actions {{ margin-top: 10px; display: flex; justify-content: flex-end; }}
    .detail-link {{ color: var(--accent); font-weight: 700; text-decoration: none; }}
    .empty {{ color: var(--muted); font-size: 12px; }}
    @media (max-width: 1100px) {{
      .layout {{ grid-template-columns: 1fr; }}
      aside {{ position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--line); }}
      .cards {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(PORTAL_TITLE)}</h1>
    <p class="sub">主页面展示每个 TDW Genesis 风格导出样本的 RGB 入口，并给每条样本生成详情页。</p>
  </header>
  <div class="layout">
    <aside>
      <input id="search" class="sidebar-search" placeholder="Search sample / case / slug">
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
      let selected = media.find((m) => m.label === 'RGB Video') || media[0];
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
          return `<button class="leaf ${{active}}" data-target="${{node.slug}}"><div class="leaf-name">${{node.name}}</div><div class="leaf-count">${{node.shown}} / ${{node.total}}</div></button>`;
        }}
        const childCount = countLeafSlugs(node);
        const branchClass = level > 0 ? 'branch is-collapsed' : 'branch';
        return `<section class="${{branchClass}}"><button class="branch-toggle" type="button"><span>${{node.name}}</span><span class="branch-count">${{childCount}}</span></button>${{renderTree(node.children || [], activeSlug, level + 1)}}</section>`;
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
                <p class="caption"><strong>Case:</strong> ${{item.case_name || 'n/a'}}</p>
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
          btn.parentElement.classList.toggle('is-collapsed');
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
    global ROOT, INPUT_ROOT, PORTAL_TITLE, PREFER_GIF, MANIFEST_PATH, INDEX_ASSET_ROOT
    ROOT = args.output_root.resolve()
    INPUT_ROOT = args.input_root.resolve()
    PORTAL_TITLE = str(args.portal_title)
    PREFER_GIF = bool(args.prefer_gif)
    MANIFEST_PATH = ROOT / "manifest.json"
    INDEX_ASSET_ROOT = ROOT / "index_assets"
    ROOT.mkdir(parents=True, exist_ok=True)
    groups = load_groups(sample_substring=args.sample_substring)
    if args.index_only:
        groups = attach_placeholder_detail_pages(groups)
    else:
        groups = build_group_cards(groups)
    (ROOT / "index.html").write_text(build_index(groups), encoding="utf-8")
    MANIFEST_PATH.write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
    print(ROOT / "index.html")


if __name__ == "__main__":
    main()
