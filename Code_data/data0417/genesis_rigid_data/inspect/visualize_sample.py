# 用途：把单样本物理记录渲染成可视化视频或 GIF。
"""该脚本用于把单个 Genesis 样本的 depth/seg/flow/contact 数组渲染成可视化视频。

兼容两种目录结构：
1. 旧版导出：sample_dir 下直接包含 meta.json、depth.npy、seg.npy 等数组。
2. 当前数据集：sample_dir 下包含 metadata.json、physics/*.npy、rgb/frame_*.png、depth/frame_*.png。

输出统一写到 sample_dir/visualizations，便于进一步挂到本地端口浏览。
"""
import argparse
import colorsys
import json
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.utils_io import (
    colorize_label_map,
    ensure_dir,
    matrix_to_vis,
    save_video,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visualization videos for an exported Genesis rigid sample.")
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=None, help="Override fps. Defaults to meta.json fps.")
    parser.add_argument("--num_preview_frames", type=int, default=5, help="How many representative frames to place in stitched preview images.")
    return parser.parse_args()


def load_metadata(sample_dir: Path) -> tuple[dict, Path]:
    for name in ("metadata.json", "meta.json"):
        path = sample_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), path
    raise FileNotFoundError(f"No metadata file found under {sample_dir}")


def infer_fps(meta: dict, override_fps: int | None) -> int:
    if override_fps is not None:
        return int(override_fps)
    if "fps" in meta:
        return int(meta["fps"])

    simulation = meta.get("simulation", {})
    dt = float(simulation.get("dt", 0.0) or 0.0)
    steps_per_frame = float(simulation.get("steps_per_frame", 0.0) or 0.0)
    if dt > 0 and steps_per_frame > 0:
        fps = int(round(1.0 / (dt * steps_per_frame)))
        return max(1, fps)
    return 12


def get_camera_intrinsics(meta: dict) -> dict:
    if "camera_intrinsics" in meta:
        return dict(meta["camera_intrinsics"])
    camera = meta.get("camera", {})
    if "camera_intrinsics" in camera:
        return dict(camera["camera_intrinsics"])
    return {}


def load_array(sample_dir: Path, meta: dict, *, legacy_name: str | None, output_key: str | None, default_rel: str | None) -> np.ndarray | None:
    candidates: list[Path] = []
    outputs = meta.get("outputs", {})
    if legacy_name:
        candidates.append(sample_dir / legacy_name)
    if output_key:
        rel_path = outputs.get(output_key)
        if rel_path:
            candidates.append(sample_dir / rel_path)
    if default_rel:
        candidates.append(sample_dir / default_rel)

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return np.load(path)
    return None


def load_image_frames(frame_dir: Path) -> list[np.ndarray]:
    frame_paths = sorted(frame_dir.glob("frame_*.png"))
    frames: list[np.ndarray] = []
    for frame_path in frame_paths:
        frame = np.asarray(imageio.imread(frame_path))
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=-1)
        elif frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]
        frames.append(frame.astype(np.uint8, copy=False))
    return frames


def make_seg_overlay_frames(rgb_frames: list[np.ndarray], seg: np.ndarray, *, alpha: float = 0.42) -> list[np.ndarray]:
    if not rgb_frames or len(rgb_frames) != int(seg.shape[0]):
        return []

    overlay_frames: list[np.ndarray] = []
    for rgb_frame, seg_frame in zip(rgb_frames, seg):
        rgb_arr = np.asarray(rgb_frame, dtype=np.float32)
        seg_vis = colorize_label_map(seg_frame).astype(np.float32)
        valid = (np.asarray(seg_frame) > 0)[..., None]
        blended = np.where(valid, rgb_arr * (1.0 - alpha) + seg_vis * alpha, rgb_arr)
        overlay_frames.append(np.clip(blended, 0.0, 255.0).astype(np.uint8))
    return overlay_frames


def maybe_save_video(path: Path, frames: list[np.ndarray], fps: int) -> bool:
    if not frames:
        return False
    save_video(path, frames, fps=fps)
    return True


def pick_preview_indices(length: int, num_preview_frames: int) -> list[int]:
    if length <= 0:
        return []
    count = max(1, min(length, int(num_preview_frames)))
    if count == 1:
        return [0]

    indices = np.linspace(0, length - 1, num=count)
    rounded = [int(round(value)) for value in indices]

    deduped: list[int] = []
    seen: set[int] = set()
    for idx in rounded:
        idx = max(0, min(length - 1, idx))
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)

    if len(deduped) < count:
        for idx in range(length):
            if idx not in seen:
                deduped.append(idx)
                seen.add(idx)
            if len(deduped) >= count:
                break

    return sorted(deduped[:count])


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    frame_uint8 = np.asarray(frame, dtype=np.uint8)
    return cv2.resize(frame_uint8, (width, height), interpolation=cv2.INTER_AREA)


def make_text_panel(text: str, width: int, height: int, *, align: str = "left", bg=(250, 246, 240), fg=(33, 28, 23)) -> np.ndarray:
    panel = np.full((height, width, 3), bg, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.72
    thickness = 2
    margin = 12
    lines = text.split("\n")
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    line_height = max((size[1] for size in sizes), default=0) + 10
    total_height = line_height * len(lines)
    y = max(margin + 18, (height - total_height) // 2 + 18)
    for line, size in zip(lines, sizes):
        if align == "center":
            x = max(margin, (width - size[0]) // 2)
        else:
            x = margin
        cv2.putText(panel, line, (x, y), font, scale, fg, thickness, cv2.LINE_AA)
        y += line_height
    return panel


def hsv_color(hue: float, sat: float = 0.95, val: float = 1.0) -> tuple[int, int, int]:
    rgb = colorsys.hsv_to_rgb(hue, sat, val)
    return tuple(int(np.clip(channel * 255.0, 0.0, 255.0)) for channel in rgb)


def flow_to_hsv_white_vis(
    flow: np.ndarray,
    *,
    clip_percentile: float = 92.0,
) -> np.ndarray:
    flow = np.asarray(flow, dtype=np.float32)
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    valid = np.isfinite(mag)

    if not np.any(valid):
        return np.zeros(flow.shape[:2] + (3,), dtype=np.uint8)

    clip_mag = float(np.percentile(mag[valid], clip_percentile))
    clip_mag = max(clip_mag, 1e-6)

    hue = np.zeros_like(mag, dtype=np.float32)
    hue[valid] = (np.arctan2(fy[valid], fx[valid]) + np.pi) / (2.0 * np.pi)

    strength = np.zeros_like(mag, dtype=np.float32)
    strength[valid] = np.clip(mag[valid] / clip_mag, 0.0, 1.0)
    strength = np.sqrt(strength, out=np.zeros_like(strength), where=valid)

    sat = np.zeros_like(mag, dtype=np.float32)
    sat[valid] = 0.25 + 0.75 * strength[valid]
    val = np.zeros_like(mag, dtype=np.float32)
    val[valid] = 0.18 + 0.82 * strength[valid]

    h = hue * 6.0
    i = np.floor(h).astype(np.int32)
    f = h - i
    p = val * (1.0 - sat)
    q = val * (1.0 - sat * f)
    t = val * (1.0 - sat * (1.0 - f))

    r = np.zeros_like(val)
    g = np.zeros_like(val)
    b = np.zeros_like(val)

    i_mod = i % 6
    masks = [i_mod == k for k in range(6)]
    r[masks[0]], g[masks[0]], b[masks[0]] = val[masks[0]], t[masks[0]], p[masks[0]]
    r[masks[1]], g[masks[1]], b[masks[1]] = q[masks[1]], val[masks[1]], p[masks[1]]
    r[masks[2]], g[masks[2]], b[masks[2]] = p[masks[2]], val[masks[2]], t[masks[2]]
    r[masks[3]], g[masks[3]], b[masks[3]] = p[masks[3]], q[masks[3]], val[masks[3]]
    r[masks[4]], g[masks[4]], b[masks[4]] = t[masks[4]], p[masks[4]], val[masks[4]]
    r[masks[5]], g[masks[5]], b[masks[5]] = val[masks[5]], p[masks[5]], q[masks[5]]

    rgb = np.stack([r, g, b], axis=-1)
    vis = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    vis[~valid] = 0
    return vis


def compute_depth_display_range(
    depth: np.ndarray,
    *,
    near: float | None = None,
    far: float | None = None,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> tuple[float, float]:
    depth_arr = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth_arr) & (depth_arr > 0)
    if near is not None:
        valid &= depth_arr >= float(near)
    if far is not None:
        valid &= depth_arr <= float(far)

    if not np.any(valid):
        lo = 0.0 if near is None else float(near)
        hi = lo + 1.0 if far is None else max(float(far), lo + 1.0)
        return lo, hi

    values = depth_arr[valid]
    lo = float(np.percentile(values, low_percentile))
    hi = float(np.percentile(values, high_percentile))

    if near is not None:
        lo = max(lo, float(near))
    if far is not None:
        hi = min(hi, float(far))

    if hi <= lo + 1e-6:
        lo = float(np.min(values))
        hi = float(np.max(values))
        if hi <= lo + 1e-6:
            hi = lo + 1.0
    return lo, hi


def depth_to_gray_vis(depth_frame: np.ndarray, *, lo: float, hi: float, gamma: float = 0.75) -> np.ndarray:
    depth_arr = np.asarray(depth_frame, dtype=np.float32)
    valid = np.isfinite(depth_arr) & (depth_arr > 0)
    if not np.any(valid):
        return np.zeros(depth_arr.shape + (3,), dtype=np.uint8)

    denom = max(hi - lo, 1e-6)
    norm = np.zeros_like(depth_arr, dtype=np.float32)
    norm[valid] = np.clip((depth_arr[valid] - lo) / denom, 0.0, 1.0)

    inv = 1.0 - norm
    inv = np.power(inv, gamma, where=valid, out=np.zeros_like(inv))
    gray = np.clip(inv * 255.0, 0.0, 255.0).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    gray_local = clahe.apply(gray)

    grad_x = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    edge_valid = edge_mag[valid]
    edge_scale = float(np.percentile(edge_valid, 97.0)) if edge_valid.size else 1.0
    edge_scale = max(edge_scale, 1e-6)
    edge = np.clip(edge_mag / edge_scale, 0.0, 1.0)
    edge_u8 = np.clip(edge * 255.0, 0.0, 255.0).astype(np.uint8)

    enhanced = np.clip(gray_local.astype(np.float32) * 0.82 + edge_u8.astype(np.float32) * 0.55, 0.0, 255.0).astype(np.uint8)
    gray_rgb = np.repeat(enhanced[..., None], 3, axis=-1)
    gray_rgb[~valid] = 0
    return gray_rgb


def save_preview_strips(
    sample_dir: Path,
    vis_dir: Path,
    scene_id: str,
    preview_indices: list[int],
    rows: list[tuple[str, list[np.ndarray]]],
    *,
    tile_width: int = 220,
    tile_height: int = 165,
    label_width: int = 170,
    header_height: int = 56,
    row_gap: int = 8,
    col_gap: int = 8,
) -> dict[str, str]:
    preview_dir = vis_dir / "preview_strips"
    ensure_dir(preview_dir)
    for old_png in preview_dir.glob("*.png"):
        old_png.unlink()

    total_width = label_width + len(preview_indices) * tile_width + max(0, len(preview_indices) - 1) * col_gap
    header = np.full((header_height, total_width, 3), 244, dtype=np.uint8)
    header[:, :label_width] = make_text_panel(f"{scene_id}\npreview", label_width, header_height, align="left")
    for col_idx, frame_idx in enumerate(preview_indices):
        x0 = label_width + col_idx * (tile_width + col_gap)
        header[:, x0:x0 + tile_width] = make_text_panel(f"frame {frame_idx:03d}", tile_width, header_height, align="center")

    strip_paths: dict[str, str] = {}
    rendered_rows: list[np.ndarray] = [header]

    for row_name, row_frames in rows:
        row_canvas = np.full((tile_height, total_width, 3), 248, dtype=np.uint8)
        row_canvas[:, :label_width] = make_text_panel(row_name, label_width, tile_height, align="left")
        for col_idx, frame in enumerate(row_frames):
            x0 = label_width + col_idx * (tile_width + col_gap)
            row_canvas[:, x0:x0 + tile_width] = resize_frame(frame, tile_width, tile_height)

        strip_name = f"{row_name.lower().replace(' ', '_')}_strip.png"
        imageio.imwrite(preview_dir / strip_name, row_canvas)
        strip_paths[row_name] = f"visualizations/preview_strips/{strip_name}"
        rendered_rows.append(row_canvas)

    gap_row = np.full((row_gap, total_width, 3), 236, dtype=np.uint8)
    combined_rows = [rendered_rows[0]]
    for row_canvas in rendered_rows[1:]:
        combined_rows.append(gap_row.copy())
        combined_rows.append(row_canvas)

    combined = np.vstack(combined_rows)
    imageio.imwrite(vis_dir / "preview_grid.png", combined)
    strip_paths["Preview Grid"] = "visualizations/preview_grid.png"
    return strip_paths


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    meta, meta_path = load_metadata(sample_dir)
    fps = infer_fps(meta, args.fps)

    vis_dir = sample_dir / "visualizations"
    ensure_dir(vis_dir)

    camera_intr = get_camera_intrinsics(meta)
    near = float(camera_intr.get("near", 0.1))
    far = float(camera_intr.get("far", 20.0))

    rgb_frame_dir = sample_dir / "rgb"
    rgb_frames = load_image_frames(rgb_frame_dir) if rgb_frame_dir.exists() else []

    outputs_written: dict[str, str] = {}
    if maybe_save_video(vis_dir / "rgb_vis.mp4", rgb_frames, fps=fps):
        outputs_written["rgb_vis"] = "visualizations/rgb_vis.mp4"

    depth = load_array(
        sample_dir,
        meta,
        legacy_name="depth.npy",
        output_key="depth_metric",
        default_rel="physics/depth_metric.npy",
    )
    seg = load_array(
        sample_dir,
        meta,
        legacy_name="seg.npy",
        output_key="segmentation",
        default_rel="physics/seg.npy",
    )
    flow = load_array(
        sample_dir,
        meta,
        legacy_name="flow.npy",
        output_key="flow",
        default_rel="physics/flow.npy",
    )
    contact_graph = load_array(
        sample_dir,
        meta,
        legacy_name="contact_graph.npy",
        output_key="contact_graph",
        default_rel="physics/contact_graph.npy",
    )
    contact_impulse = load_array(
        sample_dir,
        meta,
        legacy_name="contact_impulse.npy",
        output_key="contact_impulse",
        default_rel="physics/contact_impulse.npy",
    )

    if depth is not None:
        depth_lo, depth_hi = compute_depth_display_range(depth, near=near, far=far)
        depth_frames = [depth_to_gray_vis(frame, lo=depth_lo, hi=depth_hi) for frame in depth]
        save_video(vis_dir / "depth_vis.mp4", depth_frames, fps=fps)
        outputs_written["depth_vis"] = "visualizations/depth_vis.mp4"
    else:
        depth_frames = []

    if seg is not None:
        seg_frames = [colorize_label_map(frame) for frame in seg]
        save_video(vis_dir / "seg_vis.mp4", seg_frames, fps=fps)
        save_video(vis_dir / "mask_vis.mp4", seg_frames, fps=fps)
        outputs_written["seg_vis"] = "visualizations/seg_vis.mp4"
        outputs_written["mask_vis"] = "visualizations/mask_vis.mp4"
        seg_overlay_frames = make_seg_overlay_frames(rgb_frames, seg)
        if seg_overlay_frames:
            save_video(vis_dir / "seg_overlay.mp4", seg_overlay_frames, fps=fps)
            outputs_written["seg_overlay"] = "visualizations/seg_overlay.mp4"
    else:
        seg_frames = []
        seg_overlay_frames = []

    if flow is not None and flow.shape[0] > 0:
        flow_frames = [flow_to_hsv_white_vis(frame) for frame in flow]
        flow_frames.append(flow_frames[-1].copy())
        save_video(vis_dir / "flow_vis.mp4", flow_frames, fps=fps)
        outputs_written["flow_vis"] = "visualizations/flow_vis.mp4"
    else:
        flow_frames = []

    if contact_graph is not None:
        graph_frames = [matrix_to_vis(frame) for frame in contact_graph]
        save_video(vis_dir / "contact_graph_vis.mp4", graph_frames, fps=fps)
        outputs_written["contact_graph_vis"] = "visualizations/contact_graph_vis.mp4"
    else:
        graph_frames = []

    if contact_impulse is not None:
        impulse_frames = [matrix_to_vis(frame) for frame in contact_impulse]
        save_video(vis_dir / "contact_impulse_vis.mp4", impulse_frames, fps=fps)
        outputs_written["contact_impulse_vis"] = "visualizations/contact_impulse_vis.mp4"
    else:
        impulse_frames = []

    sequence_lengths = [
        len(rgb_frames),
        len(depth_frames),
        len(seg_frames),
        len(flow_frames),
        len(graph_frames),
        len(impulse_frames),
    ]
    preview_len = max(sequence_lengths) if any(sequence_lengths) else 0
    preview_indices = pick_preview_indices(preview_len, args.num_preview_frames)

    preview_rows: list[tuple[str, list[np.ndarray]]] = []
    preview_sources = [
        ("RGB", rgb_frames),
        ("Depth", depth_frames),
        ("Mask", seg_frames),
        ("Mask Overlay", seg_overlay_frames),
        ("Flow", flow_frames),
        ("Contact Graph", graph_frames),
        ("Contact Impulse", impulse_frames),
    ]
    for row_name, frames in preview_sources:
        if not frames:
            continue
        preview_rows.append((row_name, [frames[min(idx, len(frames) - 1)] for idx in preview_indices]))

    if preview_rows and preview_indices:
        preview_outputs = save_preview_strips(
            sample_dir,
            vis_dir,
            str(meta.get("scene_id", sample_dir.name)),
            preview_indices,
            preview_rows,
        )
        outputs_written["preview_grid"] = preview_outputs["Preview Grid"]
        outputs_written["preview_strips"] = preview_outputs

    manifest = {
        "scene_id": str(meta.get("scene_id", sample_dir.name)),
        "sample_dir": str(sample_dir),
        "metadata_path": str(meta_path),
        "fps": int(fps),
        "outputs": outputs_written,
    }
    write_json(vis_dir / "visualization_manifest.json", manifest)

    print(f"[DONE] visualization videos written to: {vis_dir}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
