"""该模块用于提供 Genesis 数据导出的通用 I/O 工具；输入为图像、数组、路径等中间结果，输出为保存到磁盘的视频、可视化文件和标准化路径。"""
import json
from pathlib import Path
from typing import Any, Iterable, List

import imageio.v2 as imageio
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


DEFAULT_DATASET_PARENT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(payload), f, ensure_ascii=False, indent=2)


def to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0)
        return arr.astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_video(path: Path, frames: Iterable[np.ndarray], fps: int) -> None:
    ensure_dir(path.parent)
    frame_list: List[np.ndarray] = [to_uint8_rgb(frame) for frame in frames]
    if not frame_list:
        raise ValueError("Cannot save an empty video.")
    try:
        imageio.mimwrite(path, frame_list, fps=int(fps), quality=8)
        return
    except Exception:
        pass

    if cv2 is None:
        imageio.mimwrite(path, frame_list, fps=int(fps))
        return

    height, width = frame_list[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    try:
        for frame in frame_list:
            if frame.shape[:2] != (height, width):
                raise ValueError(f"All video frames must share the same size, got {frame.shape[:2]} vs {(height, width)}")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def colorize_label_map(label_map: np.ndarray, background_color=(0, 0, 0)) -> np.ndarray:
    arr = np.asarray(label_map, dtype=np.int64)
    h, w = arr.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[...] = np.asarray(background_color, dtype=np.uint8)
    palette = np.asarray(
        [
            [230, 72, 57],
            [51, 128, 225],
            [238, 184, 52],
            [63, 184, 132],
            [153, 107, 235],
            [232, 117, 181],
            [70, 192, 208],
            [238, 140, 60],
        ],
        dtype=np.uint8,
    )
    unique_vals = np.unique(arr)
    for value in unique_vals:
        if int(value) <= 0:
            continue
        out[arr == value] = palette[(int(value) - 1) % len(palette)]
    return out


def depth_to_vis(depth: np.ndarray, near: float | None = None, far: float | None = None) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    vis = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if not np.any(valid):
        return vis

    d = depth[valid]
    lo = float(np.min(d) if near is None else near)
    hi = float(np.max(d) if far is None else min(np.max(d), far))
    hi = max(hi, lo + 1e-6)
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0)
    gray = ((1.0 - norm) * 255.0).astype(np.uint8)
    vis[..., 0] = gray
    vis[..., 1] = gray
    vis[..., 2] = gray
    vis[~valid] = 0
    return vis


def flow_to_vis(flow: np.ndarray, clip_magnitude: float | None = None) -> np.ndarray:
    flow = np.asarray(flow, dtype=np.float32)
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    ang = np.arctan2(fy, fx)
    hue = (ang + np.pi) / (2.0 * np.pi)
    max_mag = float(np.max(mag)) if clip_magnitude is None else float(clip_magnitude)
    max_mag = max(max_mag, 1e-6)
    sat = np.clip(mag / max_mag, 0.0, 1.0)
    val = np.where(mag > 0, 1.0, 0.0)

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
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def matrix_to_vis(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={arr.shape}")
    if arr.size == 0:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    hi = max(hi, lo + 1e-6)
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    gray_rgb = np.repeat(gray[..., None], 3, axis=-1)

    scale = max(1, 256 // max(arr.shape))
    return np.kron(gray_rgb, np.ones((scale, scale, 1), dtype=np.uint8))


def resolve_output_root(out_dir: str) -> Path:
    out_path = Path(out_dir)
    if out_path.is_absolute():
        return out_path

    parts = out_path.parts
    if parts and parts[0] == "data":
        trimmed = Path(*parts[1:]) if len(parts) > 1 else Path()
        return DEFAULT_DATASET_PARENT / trimmed
    return DEFAULT_DATASET_PARENT / out_path


def sample_dir_for(split_dir: Path, sample_index: int) -> Path:
    return split_dir / f"sample_{sample_index:06d}"
