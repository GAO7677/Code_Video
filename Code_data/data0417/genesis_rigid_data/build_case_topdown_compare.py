#!/usr/bin/env python3
# 用途：按 case 生成俯视对比图与轨迹可视化。
"""Build a local portal that compares the original RGB render with orthographic diagnostic views."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import matplotlib
import numpy as np
import trimesh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw

matplotlib.use("Agg")


CASE_DIR_DEFAULT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid/"
    "interaction_pair_plus_dynamic/count_02/10054__case000_static_center_v2"
)
OUTPUT_ROOT_DEFAULT = Path("/data/gaoya/AAA_test_video/portal_hub")
STRIKER_RADIUS_M = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a top-down orthographic comparison page for one saved case.")
    parser.add_argument("--case-dir", type=Path, default=CASE_DIR_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--gif-max-side", type=int, default=560)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sorted_frame_paths(frame_dir: Path) -> list[Path]:
    return sorted(frame_dir.glob("frame_*.png"))


def load_rgb_frames(case_dir: Path) -> list[Image.Image]:
    return [Image.open(path).convert("RGB") for path in sorted_frame_paths(case_dir / "rgb")]


def resize_for_gif(frame: Image.Image, max_side: int) -> Image.Image:
    scale = min(max_side / float(frame.width), max_side / float(frame.height), 1.0)
    size = (max(1, int(round(frame.width * scale))), max(1, int(round(frame.height * scale))))
    return frame.resize(size, Image.Resampling.BILINEAR)


def save_gif(frames: Sequence[Image.Image], dst: Path, max_side: int, duration_ms: int = 140) -> None:
    processed = [resize_for_gif(frame, max_side=max_side) for frame in frames]
    processed[0].save(
        dst,
        save_all=True,
        append_images=processed[1:],
        duration=duration_ms,
        loop=0,
    )


def save_mp4(frames: Sequence[Image.Image], dst: Path, max_side: int, fps: int = 7) -> None:
    processed = [np.asarray(resize_for_gif(frame, max_side=max_side).convert("RGB"), dtype=np.uint8) for frame in frames]
    imageio.mimsave(dst, processed, fps=fps)


def quat_xyzw_to_matrix(quat_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def load_part_meshes(summary: dict) -> tuple[list[dict], np.ndarray, np.ndarray]:
    parts: list[dict] = []
    all_vertices = []
    for part in summary.get("rigid_part_links", []):
        mesh_path = Path(str(part["mesh_path"]))
        mesh = trimesh.load_mesh(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        color_rgba = tuple(float(x) for x in part.get("color_rgba", [0.3, 0.6, 0.9, 1.0]))
        parts.append(
            {
                "vertices": vertices,
                "faces": faces,
                "color_rgba": color_rgba,
            }
        )
        all_vertices.append(vertices)
    if not all_vertices:
        raise RuntimeError("No rigid part meshes found in asset summary.")
    merged_vertices = np.concatenate(all_vertices, axis=0)
    center = merged_vertices.mean(axis=0)
    extents = merged_vertices.max(axis=0) - merged_vertices.min(axis=0)
    for part in parts:
        part["vertices_centered"] = part["vertices"] - center[None, :]
    return parts, center.astype(np.float32), extents.astype(np.float32)


def find_asset_summary(case_dir: Path) -> dict:
    meta = load_json(case_dir / "metadata.json")
    target_source_object_id = str(meta["objects"][0]["source_object_id"])
    dataset_root = None
    for parent in case_dir.parents:
        if parent.name == "version_1_genesis_rigid_data_all_cases":
            dataset_root = parent
            break
    if dataset_root is None:
        raise FileNotFoundError(f"Could not infer dataset root from case dir: {case_dir}")
    summary_path = dataset_root / "_asset_cache" / "physxnet_objects" / target_source_object_id / f"{target_source_object_id}_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Target asset summary not found: {summary_path}")
    return load_json(summary_path)


def draw_ground(ax, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    xs = np.linspace(xlim[0], xlim[1], 2)
    ys = np.linspace(ylim[0], ylim[1], 2)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color=(0.86, 0.88, 0.90), alpha=0.55, linewidth=0, shade=False)
    for tick in np.linspace(xlim[0], xlim[1], 9):
        ax.plot([tick, tick], [ylim[0], ylim[1]], [0.0, 0.0], color=(0.55, 0.58, 0.62), linewidth=0.5, alpha=0.45)
    for tick in np.linspace(ylim[0], ylim[1], 9):
        ax.plot([xlim[0], xlim[1]], [tick, tick], [0.0, 0.0], color=(0.55, 0.58, 0.62), linewidth=0.5, alpha=0.45)


def draw_yoz_grid(ax, ylim: tuple[float, float], zlim: tuple[float, float], x_plane: float) -> None:
    ys = np.linspace(ylim[0], ylim[1], 2)
    zs = np.linspace(zlim[0], zlim[1], 2)
    yy, zz = np.meshgrid(ys, zs)
    xx = np.full_like(yy, x_plane)
    ax.plot_surface(xx, yy, zz, color=(0.88, 0.90, 0.92), alpha=0.18, linewidth=0, shade=False)
    for tick in np.linspace(ylim[0], ylim[1], 9):
        ax.plot([x_plane, x_plane], [tick, tick], [zlim[0], zlim[1]], color=(0.55, 0.58, 0.62), linewidth=0.5, alpha=0.35)
    for tick in np.linspace(zlim[0], zlim[1], 9):
        ax.plot([x_plane, x_plane], [ylim[0], ylim[1]], [tick, tick], color=(0.55, 0.58, 0.62), linewidth=0.5, alpha=0.35)


def add_mesh_parts(
    ax,
    parts: Sequence[dict],
    quat_xyzw: Sequence[float],
    translation_xyz: Sequence[float],
) -> None:
    rot = quat_xyzw_to_matrix(quat_xyzw)
    trans = np.asarray(translation_xyz, dtype=np.float32)
    for part in parts:
        verts = np.asarray(part["vertices_centered"], dtype=np.float32)
        verts_world = verts @ rot.T + trans[None, :]
        faces = np.asarray(part["faces"], dtype=np.int32)
        tris = verts_world[faces]
        color = part["color_rgba"]
        poly = Poly3DCollection(
            tris,
            facecolors=[color] * len(tris),
            edgecolors=(0.05, 0.06, 0.08, 0.10),
            linewidths=0.1,
            alpha=0.98,
        )
        ax.add_collection3d(poly)


def add_striker_sphere(ax, center_xyz: Sequence[float], radius: float, color_rgb=(0.96, 0.81, 0.24)) -> None:
    u = np.linspace(0.0, 2.0 * math.pi, 36)
    v = np.linspace(0.0, math.pi, 18)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    center = np.asarray(center_xyz, dtype=np.float32)
    ax.plot_surface(
        x + center[0],
        y + center[1],
        z + center[2],
        color=color_rgb,
        linewidth=0,
        antialiased=True,
        shade=True,
    )


def render_orthographic_frames(
    case_dir: Path,
    output_dir: Path,
    gif_max_side: int,
    view_name: str,
    gif_name: str,
    frame_dir_name: str,
    title: str,
) -> list[Image.Image]:
    meta = load_json(case_dir / "metadata.json")
    kin = np.load(case_dir / "physics" / "rigid_kinematics.npz", allow_pickle=True)
    com_pos = np.asarray(kin["com_pos"], dtype=np.float32)
    orientation_quat = np.asarray(kin["orientation_quat"], dtype=np.float32)
    frame_count = int(com_pos.shape[0])

    summary = find_asset_summary(case_dir)
    parts, _mesh_center, extents = load_part_meshes(summary)

    all_xy = com_pos[:, :, :2].reshape(-1, 2)
    target_xy_pad = float(max(extents[0], extents[1]) * 0.8)
    x_min = float(all_xy[:, 0].min() - target_xy_pad)
    x_max = float(all_xy[:, 0].max() + target_xy_pad)
    y_min = float(all_xy[:, 1].min() - target_xy_pad)
    y_max = float(all_xy[:, 1].max() + target_xy_pad)
    span = max(x_max - x_min, y_max - y_min)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half = 0.5 * span
    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)
    zmax = float(max(com_pos[:, :, 2].max() + 0.12, 0.4))

    event_windows = load_json(case_dir / "physics" / "event_windows.json")
    contact_frames = sorted(
        {
            int(item["start_frame"])
            for item in event_windows
            if item.get("participants") == [0, 1] or item.get("participants") == [1, 0]
        }
    )
    contact_frame_set = set(contact_frames)

    all_xyz = com_pos.reshape(-1, 3)
    xlim_full = (float(all_xyz[:, 0].min() - target_xy_pad), float(all_xyz[:, 0].max() + target_xy_pad))
    ylim_full = (float(all_xyz[:, 1].min() - target_xy_pad), float(all_xyz[:, 1].max() + target_xy_pad))
    zlim_full = (0.0, zmax)

    frames: list[Image.Image] = []
    frame_dir = output_dir / frame_dir_name
    ensure_dir(frame_dir)
    for frame_idx in range(frame_count):
        fig = plt.figure(figsize=(6.5, 6.5), dpi=140)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_proj_type("ortho")
        if view_name == "xoy":
            ax.view_init(elev=90, azim=-90)
            draw_ground(ax, xlim=xlim, ylim=ylim)
            ax.set_box_aspect((1.0, 1.0, 0.25))
        elif view_name == "yoz":
            ax.view_init(elev=0, azim=0)
            draw_ground(ax, xlim=xlim, ylim=ylim)
            draw_yoz_grid(ax, ylim=ylim_full, zlim=zlim_full, x_plane=xlim_full[0])
            ax.set_box_aspect((0.28, 1.0, 0.75))
        else:
            raise ValueError(f"Unsupported view_name: {view_name}")
        ax.set_xlim(*xlim_full)
        ax.set_ylim(*ylim_full)
        ax.set_zlim(*zlim_full)
        ax.set_facecolor((0.95, 0.95, 0.95))
        fig.patch.set_facecolor((0.95, 0.95, 0.95))
        add_mesh_parts(ax, parts=parts, quat_xyzw=orientation_quat[frame_idx, 0], translation_xyz=com_pos[frame_idx, 0])
        add_striker_sphere(ax, center_xyz=com_pos[frame_idx, 1], radius=STRIKER_RADIUS_M)
        ax.scatter(
            [float(com_pos[frame_idx, 0, 0]), float(com_pos[frame_idx, 1, 0])],
            [float(com_pos[frame_idx, 0, 1]), float(com_pos[frame_idx, 1, 1])],
            [float(com_pos[frame_idx, 0, 2]), float(com_pos[frame_idx, 1, 2])],
            s=[32, 28],
            c=[(0.15, 0.45, 0.85), (0.98, 0.72, 0.14)],
            depthshade=False,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.set_title(title, fontsize=12, pad=14)
        badge_text = f"frame {frame_idx}"
        if frame_idx in contact_frame_set:
            badge_text += " | object contact"
        fig.text(
            0.03,
            0.95,
            badge_text,
            fontsize=10,
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=(1.0, 1.0, 1.0, 0.9), edgecolor=(0.4, 0.4, 0.4, 0.4)),
        )
        fig.tight_layout(pad=0.5)
        frame_path = frame_dir / f"frame_{frame_idx:03d}.png"
        fig.savefig(frame_path, dpi=140)
        plt.close(fig)
        frames.append(Image.open(frame_path).convert("RGB"))
    save_gif(frames, output_dir / gif_name, max_side=gif_max_side, duration_ms=140)
    return frames


def annotate_frame(frame: Image.Image, text: str) -> Image.Image:
    image = frame.copy()
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((10, 10, 220, 44), radius=10, fill=(10, 10, 10, 180))
    draw.text((18, 18), text, fill=(255, 255, 255))
    return image


def build_side_by_side_frames(
    frame_groups: Sequence[Sequence[Image.Image]],
    titles: Sequence[str],
) -> list[Image.Image]:
    assert frame_groups
    frame_count = len(frame_groups[0])
    assert all(len(group) == frame_count for group in frame_groups)
    assert len(frame_groups) == len(titles)
    result: list[Image.Image] = []
    for idx in range(frame_count):
        resized = [resize_for_gif(group[idx], max_side=420) for group in frame_groups]
        target_h = max(img.height for img in resized)
        resized = [
            img.resize((int(round(img.width * target_h / float(img.height))), target_h), Image.Resampling.BILINEAR)
            for img in resized
        ]
        pad = 18
        title_h = 46
        total_w = sum(img.width for img in resized) + pad * (len(resized) + 1)
        canvas = Image.new("RGB", (total_w, target_h + title_h + pad * 2), color=(242, 238, 231))
        draw = ImageDraw.Draw(canvas)
        cursor_x = pad
        for img, title in zip(resized, titles):
            canvas.paste(img, (cursor_x, title_h + pad))
            draw.text((cursor_x, 14), f"{title} | frame {idx}", fill=(25, 24, 22))
            cursor_x += img.width + pad
        result.append(canvas)
    return result


def build_html(case_dir: Path, metadata: dict, contact_frames: Sequence[int]) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Case Camera Compare</title>
  <style>
    :root {{
      --bg: #efe8df;
      --panel: #fffaf3;
      --ink: #1f1a16;
      --muted: #6c645b;
      --line: #d9c8b3;
      --shadow: rgba(34, 26, 18, 0.12);
      --accent: #8a3e1b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Iowan Old Style", "Palatino Linotype", "Noto Serif SC", serif;
      background:
        radial-gradient(circle at top left, rgba(138, 62, 27, 0.10), transparent 28rem),
        linear-gradient(180deg, #fbf7f1 0%, #efe8df 100%);
    }}
    .page {{ width: min(1480px, calc(100vw - 24px)); margin: 18px auto 40px; }}
    .hero, .panel {{
      background: rgba(255, 250, 243, 0.94);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 16px 40px var(--shadow);
    }}
    .hero {{ padding: 20px 22px; margin-bottom: 16px; }}
    .hero h1 {{ margin: 0; font-size: 30px; line-height: 1.1; }}
    .hero p {{ margin: 10px 0 0; color: var(--muted); line-height: 1.55; }}
    .pill-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      background: white;
      padding: 6px 10px;
      font-size: 13px;
    }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }}
    .panel {{ padding: 14px; }}
    .panel h2 {{ margin: 0 0 10px; font-size: 18px; }}
    .panel img, .panel video {{
      width: 100%;
      display: block;
      border-radius: 12px;
      background: #101215;
    }}
    .meta {{
      margin-top: 10px;
      color: var(--muted);
      line-height: 1.55;
      word-break: break-word;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Camera Compare: {metadata["scene_id"]}</h1>
      <p>当前页同时给出原始数据集相机视角、XOY 平面正交俯视图、以及 YOZ 平面正交侧视图。后两者都基于保存的刚体位姿轨迹与资产 mesh 重建。接触记录帧: {", ".join(str(x) for x in contact_frames) if contact_frames else "none"}。</p>
      <div class="pill-row">
        <span class="pill">scene_composition: {metadata["scene_composition"]}</span>
        <span class="pill">interaction_pattern: {metadata["interaction_pattern"]}</span>
        <span class="pill">frames: {metadata["frames"]}</span>
        <span class="pill">camera_res: {metadata["resolution"][0]}x{metadata["resolution"][1]}</span>
      </div>
      <div class="meta">
        case dir: <code>{case_dir}</code>
      </div>
    </section>
    <section class="grid">
      <article class="panel">
        <h2>Original RGB Video</h2>
        <video controls loop playsinline preload="metadata" src="original_rgb.mp4"></video>
        <div class="meta">fallback gif: <code>original_rgb.gif</code></div>
      </article>
      <article class="panel">
        <h2>XOY Orthographic Video</h2>
        <video controls loop playsinline preload="metadata" src="topdown_xoy_ortho.mp4"></video>
        <div class="meta">fallback gif: <code>topdown_xoy_ortho.gif</code></div>
      </article>
      <article class="panel">
        <h2>YOZ Orthographic Video</h2>
        <video controls loop playsinline preload="metadata" src="side_yoz_ortho.mp4"></video>
        <div class="meta">fallback gif: <code>side_yoz_ortho.gif</code></div>
      </article>
      <article class="panel" style="grid-column: 1 / -1;">
        <h2>Three-View Comparison Video</h2>
        <video controls loop playsinline preload="metadata" src="compare_three_view.mp4"></video>
        <div class="meta">fallback gif: <code>compare_side_by_side.gif</code></div>
      </article>
      <article class="panel" style="grid-column: 1 / -1;">
        <h2>Original MP4</h2>
        <video controls preload="metadata" src="rgb.mp4"></video>
      </article>
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    scene_id = case_dir.name
    output_dir = args.output_root / f"camera_compare_{scene_id}"
    ensure_dir(output_dir)

    metadata = load_json(case_dir / "metadata.json")
    event_windows = load_json(case_dir / "physics" / "event_windows.json")
    contact_frames = sorted(
        {
            int(item["start_frame"])
            for item in event_windows
            if item.get("participants") == [0, 1] or item.get("participants") == [1, 0]
        }
    )

    rgb_frames = [annotate_frame(frame, f"Original RGB | frame {idx}") for idx, frame in enumerate(load_rgb_frames(case_dir))]
    save_gif(rgb_frames, output_dir / "original_rgb.gif", max_side=int(args.gif_max_side), duration_ms=140)
    save_mp4(rgb_frames, output_dir / "original_rgb.mp4", max_side=int(args.gif_max_side), fps=7)

    topdown_frames = render_orthographic_frames(
        case_dir=case_dir,
        output_dir=output_dir,
        gif_max_side=int(args.gif_max_side),
        view_name="xoy",
        gif_name="topdown_xoy_ortho.gif",
        frame_dir_name="topdown_frames",
        title="XOY Orthographic Top View",
    )
    save_mp4(topdown_frames, output_dir / "topdown_xoy_ortho.mp4", max_side=int(args.gif_max_side), fps=7)
    side_frames = render_orthographic_frames(
        case_dir=case_dir,
        output_dir=output_dir,
        gif_max_side=int(args.gif_max_side),
        view_name="yoz",
        gif_name="side_yoz_ortho.gif",
        frame_dir_name="side_frames",
        title="YOZ Orthographic Side View",
    )
    save_mp4(side_frames, output_dir / "side_yoz_ortho.mp4", max_side=int(args.gif_max_side), fps=7)
    compare_frames = build_side_by_side_frames(
        [rgb_frames, topdown_frames, side_frames],
        ["Original camera", "XOY top view", "YOZ side view"],
    )
    save_gif(compare_frames, output_dir / "compare_side_by_side.gif", max_side=int(args.gif_max_side * 2), duration_ms=140)
    save_mp4(compare_frames, output_dir / "compare_three_view.mp4", max_side=int(args.gif_max_side * 2), fps=7)
    symlink_or_copy(case_dir / "videos" / "rgb.mp4", output_dir / "rgb.mp4")

    html = build_html(case_dir=case_dir, metadata=metadata, contact_frames=contact_frames)
    (output_dir / "index.html").write_text(html, encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
