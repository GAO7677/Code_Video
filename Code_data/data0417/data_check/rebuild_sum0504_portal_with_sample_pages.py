#!/usr/bin/env python3
"""Rebuild sum0504 portal with per-sample detail pages."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path("/home/gaoya/portal_hub_sim/sum0504_portal")
MANIFEST_PATH = ROOT / "manifest.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def relpath(path: Path, start: Path) -> str:
    return os.path.relpath(path, start)


def fmt_bool(v: bool) -> str:
    return "yes" if v else "no"


def pick_existing(sample_dir: Path, names: list[str]) -> list[Path]:
    result = []
    for name in names:
        p = sample_dir / name
        if p.exists():
            result.append(p)
    return result


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
    for key in ("collision_events_path", "event_windows_path"):
        path = record.get(key)
        if path is not None and Path(path).exists():
            payload = load_json(Path(path))
            if isinstance(payload, list):
                return payload
    return []


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


def render_trajectory_overview(record: dict[str, Any], page_dir: Path) -> Path | None:
    bundle = load_state_bundle(record)
    if bundle is None:
        return None
    state = np.asarray(bundle["state_raw"], dtype=np.float32)
    labels = object_labels(bundle)
    meta = bundle.get("meta") or {}
    width = float((meta.get("resolution") or [0, 0])[0] or max(1.0, float(np.nanmax(state[..., 0]) + 20.0)))
    height = float((meta.get("resolution") or [0, 0])[1] or max(1.0, float(np.nanmax(state[..., 1]) + 20.0)))
    out_path = page_dir / "trajectory_overview.png"

    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=150)
    ax.set_title("Trajectory Overview")
    frame0 = record["sample_dir"] / "rgb" / "frame_000.png"
    if frame0.exists():
        with Image.open(frame0) as img:
            bg = np.asarray(img.convert("RGB"))
        ax.imshow(bg, extent=(0, width, height, 0), alpha=0.42)
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


def render_physics_summary_html(record: dict[str, Any], page_dir: Path) -> str:
    blocks: list[str] = []
    depth_video = record["sample_dir"] / "videos" / "depth.mp4"
    if not depth_video.exists():
        depth_video = record["sample_dir"] / "visualizations" / "depth_vis.mp4"
    if depth_video.exists():
        src = relpath(depth_video, page_dir)
        blocks.append(
            f"""
<section class="media-card">
  <h3>Depth Video</h3>
  <video src="{html.escape(src)}" controls preload="metadata"></video>
</section>
"""
        )

    trajectory_path = render_trajectory_overview(record, page_dir)
    if trajectory_path is not None and trajectory_path.exists():
        src = relpath(trajectory_path, page_dir)
        blocks.append(
            f"""
<section class="media-card">
  <h3>Trajectory Overview</h3>
  <img src="{html.escape(src)}" alt="trajectory overview">
</section>
"""
        )

    curves_path = render_state_curves(record, page_dir)
    if curves_path is not None and curves_path.exists():
        src = relpath(curves_path, page_dir)
        blocks.append(
            f"""
<section class="media-card wide">
  <h3>State Curves And Collision Timeline</h3>
  <img src="{html.escape(src)}" alt="state curves">
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
                    "collision_events.json",
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
        "collision_events_path": (physics_dir / "collision_events.json") if (physics_dir / "collision_events.json").exists() else None,
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
        src = relpath(path, page_dir)
        if kind == "video":
            blocks.append(
                f"""
<section class="media-card">
  <h3>{html.escape(label)}</h3>
  <video src="{html.escape(src)}" controls preload="metadata"></video>
</section>
"""
            )
        else:
            blocks.append(
                f"""
<section class="media-card">
  <h3>{html.escape(label)}</h3>
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
        ("collision_events", record["collision_events_path"]),
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
    table_html = file_list_html(record, page_dir)
    first_frame_block = ""
    if record["first_frame_path"] is not None:
        src = relpath(record["first_frame_path"], page_dir)
        first_frame_block = f"""
<section class="media-card">
  <h3>First Frame</h3>
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
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <p><a href="../../../index.html">Back to sum0504 portal</a></p>
      <h1>{html.escape(record['sample_name'])}</h1>
      <p>{html.escape(record['group_title'])}</p>
      <p><strong>Dataset:</strong> {html.escape(record['dataset'])} | <strong>View:</strong> {html.escape(record['view_type'])}</p>
      <p><strong>Caption:</strong> {html.escape(record['caption'] or 'n/a')}</p>
      <p><strong>Detail Caption:</strong> {html.escape(record['detail_caption'] or 'n/a')}</p>
      <p><strong>Sample Dir:</strong> <code>{html.escape(str(record['sample_dir']))}</code></p>
    </section>
    <section class="grid">
      {media_blocks}
      {physics_blocks}
      {first_frame_block}
      <section class="card wide">
        <h2>Recorded Files</h2>
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
  <title>sum0504 Portal</title>
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
      grid-template-columns: repeat(3, minmax(0, 1fr));
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
    <h1>sum0504 Portal</h1>
    <p class="sub">主页面保留当前入口，每个样本新增详情页，按样本实际记录的数据形式展示。</p>
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
      return '<div class="media-grid">' + media.map((m) => {{
        const src = m.path.replace('/home/gaoya/portal_hub_sim/sum0504_portal/', '');
        if (m.kind === 'video') {{
          return `<div class="media-block"><div class="media-label">${{m.label}}</div><video src="${{src}}" controls preload="metadata"></video></div>`;
        }}
        return `<div class="media-block"><div class="media-label">${{m.label}}</div><img src="${{src}}" alt="${{m.label}}"></div>`;
      }}).join('') + '</div>';
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
                <div class="detail">
                  <strong>Detail:</strong>
                  <div class="detail-text is-collapsed">${{item.detail_caption || 'n/a'}}</div>
                  <button class="detail-toggle" type="button">Expand</button>
                </div>
                <p class="path">${{item.sample_dir}}</p>
                ${{mediaHtml(item)}}
                <div class="actions">
                  <a class="detail-link" href="${{item.detail_page}}">Open Sample Page</a>
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
    groups = load_json(MANIFEST_PATH)
    groups = build_group_cards(groups)
    (ROOT / "index.html").write_text(build_index(groups), encoding="utf-8")
    (ROOT / "manifest.json").write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
    print(ROOT / "index.html")


if __name__ == "__main__":
    main()
