#!/usr/bin/env python3
"""Overlay representative exact ball-query heads on their generated videos."""

from __future__ import annotations

import argparse
import csv
import html
import itertools
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROLES = ("S", "T", "P", "C", "G")
ROLE_LABELS = {
    "S": "S intraframe spatial",
    "T": "T moving-ball trajectory",
    "P": "P fixed-position time alignment",
    "C": "C first-frame/history context",
    "G": "G global aggregation",
}
ROLE_DESCRIPTIONS = {
    "S": "主要保留 query 帧内的局部空间结构。",
    "T": "对跨帧运动球轨迹响应较强；modal 标注用于提示与 P/C 的功能混合。",
    "P": "跨时间关注与 query 近似相同的固定屏幕坐标。",
    "C": "偏向首帧、历史帧或物体先前所在位置。",
    "G": "注意力分散到物体、支撑物、地面和背景区域。",
}
ROLE_NAMES_ZH = {
    "S": "帧内空间",
    "T": "球轨迹传播",
    "P": "固定位置时间对齐",
    "C": "首帧/历史上下文",
    "G": "全局聚合",
}
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiblock-root", type=Path, required=True)
    parser.add_argument("--block17-root", type=Path, required=True)
    parser.add_argument("--roles-csv", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocks", default="0,5,11,17,19,29")
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.70)
    parser.add_argument("--panel-width", type=int, default=448)
    parser.add_argument("--panel-height", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=49)
    parser.add_argument("--color-percentile", type=float, default=99.5)
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path("/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg"),
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _block_root(
    block: int, multiblock_root: Path, block17_root: Path
) -> Path:
    if block == 17:
        return block17_root
    return multiblock_root / f"block{block:02d}"


def _load_role_rows(path: Path) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(int(row["block"]), []).append(row)
    return grouped


def _select_distinct_role_heads(
    rows: list[dict[str, str]], candidate_count: int = 8
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, list[dict[str, str]]] = {}
    for role in ROLES:
        score_key = f"{role.lower()}_score"
        candidates[role] = sorted(
            rows, key=lambda row: float(row[score_key]), reverse=True
        )[:candidate_count]

    best_score = -math.inf
    best_choice: tuple[dict[str, str], ...] | None = None
    for choice in itertools.product(*(candidates[role] for role in ROLES)):
        if len({int(row["head"]) for row in choice}) != len(ROLES):
            continue
        score = sum(
            float(row[f"{role.lower()}_score"])
            for role, row in zip(ROLES, choice)
        )
        if score > best_score:
            best_score = score
            best_choice = choice
    if best_choice is None:
        raise RuntimeError("cannot select five distinct representative heads")

    return {
        role: {
            "head": int(row["head"]),
            "target_role": role,
            "target_role_label": ROLE_LABELS[role],
            "target_score": float(row[f"{role.lower()}_score"]),
            "modal_role": row["role"],
            "modal_role_label": row["role_label"],
            "role_stability": float(row["role_stability"]),
            "role_margin": float(row["role_margin"]),
        }
        for role, row in zip(ROLES, best_choice)
    }


def _summary_path(block_root: Path, model: str, case: str) -> Path:
    return block_root / "matrices" / model / case / "summary.json"


def _matrix_for_step(
    block_root: Path, model: str, case: str, step: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    summary_path = _summary_path(block_root, model, case)
    summary = _read_json(summary_path)
    entries = [
        entry
        for entry in summary["steps"]
        if int(entry["step_number_one_based"]) == step
    ]
    if len(entries) != 1:
        raise RuntimeError(
            f"{summary_path}: expected one capture for step {step}, got {len(entries)}"
        )
    entry = entries[0]
    matrix_path = (
        summary_path.parent
        / str(entry["directory"])
        / str(entry["matrix_npz"])
    )
    with np.load(matrix_path) as arrays:
        attention = arrays["attention"].astype(np.float32)
        query_coords = arrays["query_coords"].astype(np.int64)
    grid = tuple(int(value) for value in summary["latent_grid"])
    if attention.shape[1:] != grid:
        raise RuntimeError(
            f"{matrix_path}: attention {attention.shape} does not match grid {grid}"
        )
    return attention, query_coords, grid


def _generated_video(block_root: Path, model: str, case: str) -> Path:
    matches = sorted(
        (block_root / "generated" / model).glob(f"**/{case}.mp4")
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one generated video for {model}/{case}, found {matches}"
        )
    return matches[0]


def _source_video(generated_video: Path) -> Path:
    sidecar = _read_json(generated_video.with_suffix(".json"))
    for key in ("source_video", "input_video_original"):
        value = sidecar.get(key)
        if value and Path(value).is_file():
            return Path(value)
    input_json = sidecar.get("input_json")
    if input_json and Path(input_json).is_file():
        value = _read_json(Path(input_json)).get("source_video")
        if value and Path(value).is_file():
            return Path(value)
    raise FileNotFoundError(
        f"cannot resolve source video from {generated_video.with_suffix('.json')}"
    )


def _read_video(path: Path, max_frames: int) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video contains no readable frames: {path}")
    return frames, fps


def _temporal_slice(
    grid: np.ndarray, frame_index: int, frame_count: int
) -> np.ndarray:
    if frame_count <= 1:
        return grid[0]
    position = frame_index * (grid.shape[0] - 1) / float(frame_count - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, grid.shape[0] - 1)
    weight = position - lower
    return grid[lower] * (1.0 - weight) + grid[upper] * weight


def _label(
    image: np.ndarray,
    line1: str,
    line2: str,
    *,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    for text, y in ((line1, 24), (line2, 47)):
        cv2.putText(
            image,
            text,
            (9, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            (9, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_query_patch(
    image: np.ndarray,
    query_coords: np.ndarray,
    grid: tuple[int, int, int],
    frame_index: int,
    frame_count: int,
) -> None:
    latent_t = frame_index * (grid[0] - 1) / max(frame_count - 1, 1)
    query_t = int(query_coords[0, 0])
    if abs(latent_t - query_t) > 0.55:
        return
    height, width = image.shape[:2]
    y0 = int(query_coords[:, 1].min()) * height // grid[1]
    y1 = (int(query_coords[:, 1].max()) + 1) * height // grid[1]
    x0 = int(query_coords[:, 2].min()) * width // grid[2]
    x1 = (int(query_coords[:, 2].max()) + 1) * width // grid[2]
    cv2.rectangle(image, (x0, y0), (x1, y1), (80, 255, 80), 2)


def _overlay_panel(
    frame: np.ndarray,
    heat_grid: np.ndarray,
    *,
    frame_index: int,
    frame_count: int,
    alpha: float,
    vmax: float,
    panel_size: tuple[int, int],
    query_coords: np.ndarray,
    grid: tuple[int, int, int],
    line1: str,
    line2: str,
) -> np.ndarray:
    panel_width, panel_height = panel_size
    panel = cv2.resize(
        frame, (panel_width, panel_height), interpolation=cv2.INTER_AREA
    )
    heat = _temporal_slice(heat_grid, frame_index, frame_count)
    heat = np.clip(heat / vmax, 0.0, 1.0)
    heat = cv2.resize(
        heat.astype(np.float32),
        (panel_width, panel_height),
        interpolation=cv2.INTER_CUBIC,
    )
    heat = np.clip(heat, 0.0, 1.0)
    color = cv2.applyColorMap(
        np.round(heat * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO
    )
    local_alpha = (alpha * np.sqrt(heat))[..., None]
    panel = np.clip(
        panel.astype(np.float32) * (1.0 - local_alpha)
        + color.astype(np.float32) * local_alpha,
        0,
        255,
    ).astype(np.uint8)
    _draw_query_patch(panel, query_coords, grid, frame_index, frame_count)
    _label(panel, line1, line2)
    return panel


def _draw_query_patch_at_latent(
    image: np.ndarray,
    query_coords: np.ndarray,
    grid: tuple[int, int, int],
    latent_t: int,
) -> None:
    if latent_t != int(query_coords[0, 0]):
        return
    height, width = image.shape[:2]
    y0 = int(query_coords[:, 1].min()) * height // grid[1]
    y1 = (int(query_coords[:, 1].max()) + 1) * height // grid[1]
    x0 = int(query_coords[:, 2].min()) * width // grid[2]
    x1 = (int(query_coords[:, 2].max()) + 1) * width // grid[2]
    cv2.rectangle(image, (x0, y0), (x1, y1), (80, 255, 80), 2)


def _overlay_exact_latent_panel(
    frame: np.ndarray,
    heat: np.ndarray,
    *,
    latent_t: int,
    alpha: float,
    vmax: float,
    panel_size: tuple[int, int],
    query_coords: np.ndarray,
    grid: tuple[int, int, int],
    line1: str,
    line2: str,
) -> np.ndarray:
    panel_width, panel_height = panel_size
    panel = cv2.resize(
        frame, (panel_width, panel_height), interpolation=cv2.INTER_AREA
    )
    normalized = np.clip(heat / vmax, 0.0, 1.0)
    normalized = cv2.resize(
        normalized.astype(np.float32),
        (panel_width, panel_height),
        interpolation=cv2.INTER_CUBIC,
    )
    normalized = np.clip(normalized, 0.0, 1.0)
    color = cv2.applyColorMap(
        np.round(normalized * 255.0).astype(np.uint8),
        cv2.COLORMAP_INFERNO,
    )
    local_alpha = (alpha * np.sqrt(normalized))[..., None]
    panel = np.clip(
        panel.astype(np.float32) * (1.0 - local_alpha)
        + color.astype(np.float32) * local_alpha,
        0,
        255,
    ).astype(np.uint8)
    _draw_query_patch_at_latent(panel, query_coords, grid, latent_t)
    _label(panel, line1, line2)
    return panel


def _latent_to_output_frame(latent_t: int) -> int:
    # Wan VAE maps t=0 to frame 0, then each latent step covers four frames.
    return 0 if latent_t == 0 else 4 * latent_t


def _latent_output_range(latent_t: int) -> tuple[int, int]:
    if latent_t == 0:
        return 0, 0
    return 4 * latent_t - 3, 4 * latent_t


def _write_block_frame_sequence(
    *,
    video_path: Path,
    output_dir: Path,
    block: int,
    model: str,
    video_kind: str,
    step: int,
    attention: np.ndarray,
    query_coords: np.ndarray,
    grid: tuple[int, int, int],
    selections: dict[str, dict[str, Any]],
    vmax: float,
    alpha: float,
    panel_size: tuple[int, int],
    max_frames: int,
) -> list[str]:
    frames, _ = _read_video(video_path, max_frames)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_count = math.prod(grid)
    names = []
    for latent_t in range(grid[0]):
        output_frame = min(_latent_to_output_frame(latent_t), len(frames) - 1)
        frame = frames[output_frame]
        range_start, range_stop = _latent_output_range(latent_t)
        original = cv2.resize(frame, panel_size, interpolation=cv2.INTER_AREA)
        _draw_query_patch_at_latent(original, query_coords, grid, latent_t)
        _label(
            original,
            f"{MODEL_LABELS[model]} | B{block:02d} | {video_kind}",
            (
                f"latent {latent_t:02d} | frames "
                f"{range_start:02d}-{range_stop:02d} | bg {output_frame:02d}"
            ),
            color=(80, 255, 80),
        )
        panels = [original]
        for role in ROLES:
            selection = selections[role]
            head = int(selection["head"])
            enrichment = attention[head, latent_t] * float(token_count)
            heat = np.maximum(
                np.log2(np.maximum(enrichment, 1.0)), 0.0
            )
            modal = str(selection["modal_role"])
            modal_note = "" if modal == role else f" | modal {modal}"
            panels.append(
                _overlay_exact_latent_panel(
                    frame,
                    heat,
                    latent_t=latent_t,
                    alpha=alpha,
                    vmax=vmax,
                    panel_size=panel_size,
                    query_coords=query_coords,
                    grid=grid,
                    line1=f"{ROLE_LABELS[role]} | Head {head:02d}",
                    line2=(
                        f"score {selection['target_score']:.2f}"
                        f" | stable {selection['role_stability']:.0%}"
                        f"{modal_note}"
                    ),
                )
            )
        canvas = np.vstack(
            [np.hstack(panels[:3]), np.hstack(panels[3:])]
        )
        name = f"latent_{latent_t:02d}.jpg"
        path = output_dir / name
        if not cv2.imwrite(
            str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]
        ):
            raise RuntimeError(f"failed to write {path}")
        names.append(name)
    return names


def _write_comparison_video(
    *,
    video_path: Path,
    output_path: Path,
    block: int,
    model: str,
    video_kind: str,
    step: int,
    attention: np.ndarray,
    query_coords: np.ndarray,
    grid: tuple[int, int, int],
    selections: dict[str, dict[str, Any]],
    vmax: float,
    alpha: float,
    panel_size: tuple[int, int],
    max_frames: int,
    ffmpeg: Path,
) -> int:
    frames, fps = _read_video(video_path, max_frames)
    panel_width, panel_height = panel_size
    output_width, output_height = panel_width * 3, panel_height * 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{output_width}x{output_height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    token_count = math.prod(grid)
    try:
        for frame_index, frame in enumerate(frames):
            original = cv2.resize(
                frame, panel_size, interpolation=cv2.INTER_AREA
            )
            _draw_query_patch(
                original, query_coords, grid, frame_index, len(frames)
            )
            _label(
                original,
                f"{MODEL_LABELS[model]} | Block {block:02d}",
                f"{video_kind} | denoise step {step:02d}",
                color=(80, 255, 80),
            )
            panels = [original]
            for role in ROLES:
                selection = selections[role]
                head = int(selection["head"])
                enrichment = attention[head] * float(token_count)
                heat_grid = np.maximum(
                    np.log2(np.maximum(enrichment, 1.0)), 0.0
                )
                modal = str(selection["modal_role"])
                modal_note = "" if modal == role else f" | modal {modal}"
                panels.append(
                    _overlay_panel(
                        frame,
                        heat_grid,
                        frame_index=frame_index,
                        frame_count=len(frames),
                        alpha=alpha,
                        vmax=vmax,
                        panel_size=panel_size,
                        query_coords=query_coords,
                        grid=grid,
                        line1=f"{ROLE_LABELS[role]} | Head {head:02d}",
                        line2=(
                            f"score {selection['target_score']:.2f}"
                            f" | stable {selection['role_stability']:.0%}"
                            f"{modal_note}"
                        ),
                    )
                )
            canvas = np.vstack(
                [np.hstack(panels[:3]), np.hstack(panels[3:])]
            )
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while writing {output_path}")
    return len(frames)


def _write_role_video(
    *,
    generated_video: Path,
    source_video: Path,
    output_path: Path,
    block: int,
    model: str,
    role: str,
    step: int,
    attention: np.ndarray,
    query_coords: np.ndarray,
    grid: tuple[int, int, int],
    selection: dict[str, Any],
    vmax: float,
    alpha: float,
    panel_size: tuple[int, int],
    max_frames: int,
    ffmpeg: Path,
) -> int:
    generated_frames, generated_fps = _read_video(generated_video, max_frames)
    source_frames, _ = _read_video(source_video, max_frames)
    frame_count = min(len(generated_frames), len(source_frames))
    panel_width, panel_height = panel_size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{panel_width * 2}x{panel_height}",
        "-r",
        f"{generated_fps:.8f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    head = int(selection["head"])
    enrichment = attention[head] * float(math.prod(grid))
    heat_grid = np.maximum(np.log2(np.maximum(enrichment, 1.0)), 0.0)
    modal = str(selection["modal_role"])
    modal_note = "" if modal == role else f" | modal {modal}"
    try:
        for frame_index in range(frame_count):
            common = {
                "heat_grid": heat_grid,
                "frame_index": frame_index,
                "frame_count": frame_count,
                "alpha": alpha,
                "vmax": vmax,
                "panel_size": panel_size,
                "query_coords": query_coords,
                "grid": grid,
                "line2": (
                    f"B{block:02d} H{head:02d} | score "
                    f"{selection['target_score']:.2f}{modal_note}"
                ),
            }
            generated_panel = _overlay_panel(
                generated_frames[frame_index],
                line1=f"{MODEL_LABELS[model]} | generated | {role}",
                **common,
            )
            source_panel = _overlay_panel(
                source_frames[frame_index],
                line1=f"{MODEL_LABELS[model]} | source/GT ref | {role}",
                **common,
            )
            process.stdin.write(
                np.hstack([generated_panel, source_panel]).tobytes()
            )
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while writing {output_path}")
    return frame_count


def _write_role_frame_sequence(
    *,
    generated_video: Path,
    source_video: Path,
    output_dir: Path,
    block: int,
    model: str,
    role: str,
    attention: np.ndarray,
    query_coords: np.ndarray,
    grid: tuple[int, int, int],
    selection: dict[str, Any],
    vmax: float,
    alpha: float,
    panel_size: tuple[int, int],
    max_frames: int,
) -> list[str]:
    generated_frames, _ = _read_video(generated_video, max_frames)
    source_frames, _ = _read_video(source_video, max_frames)
    frame_count = min(len(generated_frames), len(source_frames))
    output_dir.mkdir(parents=True, exist_ok=True)
    head = int(selection["head"])
    enrichment = attention[head] * float(math.prod(grid))
    heat_grid = np.maximum(np.log2(np.maximum(enrichment, 1.0)), 0.0)
    modal = str(selection["modal_role"])
    modal_note = "" if modal == role else f" | modal {modal}"
    names = []
    for latent_t in range(grid[0]):
        output_frame = min(
            _latent_to_output_frame(latent_t), frame_count - 1
        )
        common = {
            "heat": heat_grid[latent_t],
            "latent_t": latent_t,
            "alpha": alpha,
            "vmax": vmax,
            "panel_size": panel_size,
            "query_coords": query_coords,
            "grid": grid,
            "line2": (
                f"B{block:02d} H{head:02d} | latent {latent_t:02d}"
                f" -> frames {_latent_output_range(latent_t)[0]:02d}-"
                f"{_latent_output_range(latent_t)[1]:02d}{modal_note}"
            ),
        }
        generated_panel = _overlay_exact_latent_panel(
            generated_frames[output_frame],
            line1=f"{MODEL_LABELS[model]} | generated | {role}",
            **common,
        )
        source_panel = _overlay_exact_latent_panel(
            source_frames[output_frame],
            line1=f"{MODEL_LABELS[model]} | source/GT ref | {role}",
            **common,
        )
        name = f"latent_{latent_t:02d}.jpg"
        path = output_dir / name
        if not cv2.imwrite(
            str(path),
            np.hstack([generated_panel, source_panel]),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        ):
            raise RuntimeError(f"failed to write {path}")
        names.append(name)
    return names


def _write_block_gallery(
    output_dir: Path,
    records: list[dict[str, Any]],
    selections: dict[int, dict[str, dict[str, Any]]],
    *,
    case: str,
    step: int,
    vmax: float,
) -> None:
    by_block: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_block.setdefault(int(record["block"]), []).append(record)

    sections = []
    for block in sorted(by_block):
        selection_text = " | ".join(
            f"{role}=H{selections[block][role]['head']:02d}"
            for role in ROLES
        )
        cards = []
        for model in MODEL_LABELS:
            model_records = {
                record["variant"]: record
                for record in by_block[block]
                if record["model"] == model
            }
            generated = model_records["generated"]
            source_gt = model_records["source_gt"]
            cards.append(
                "<article>"
                f"<h3>{html.escape(MODEL_LABELS[model])}</h3>"
                "<figure>"
                f"<img class='latent-frame' data-prefix='"
                f"{html.escape(generated['frame_prefix'])}' src='"
                f"{html.escape(generated['frame_prefix'])}latent_02.jpg'>"
                "<figcaption>模型生成视频上的直接 attention 映射</figcaption>"
                "</figure><figure>"
                f"<img class='latent-frame' data-prefix='"
                f"{html.escape(source_gt['frame_prefix'])}' src='"
                f"{html.escape(source_gt['frame_prefix'])}latent_02.jpg'>"
                "<figcaption>source/GT 坐标参照</figcaption>"
                "</figure></article>"
            )
        sections.append(
            "<section>"
            f"<h2>Block {block:02d}</h2>"
            f"<p>{html.escape(selection_text)}</p>"
            f"<div class='models'>{''.join(cards)}</div>"
            "</section>"
        )

    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Moving-ball query head overlays</title>
<style>
body{{margin:0;background:#0d0f11;color:#f4f5f6;font:15px/1.45 system-ui,sans-serif}}
header,section{{padding:18px 22px;border-bottom:1px solid #30343a}}
h1,h2{{margin:0 0 8px;letter-spacing:0}}p,figcaption{{color:#b9c0c8}}
.controls{{position:sticky;top:0;z-index:2;display:flex;align-items:center;gap:12px;
padding:12px 22px;background:#16191ddd;border-bottom:1px solid #3b4047;
backdrop-filter:blur(8px)}}.controls input{{width:min(620px,65vw)}}
.controls button{{width:34px;height:34px;border:1px solid #535962;background:#23272c;
color:#fff;cursor:pointer}}.controls output{{min-width:190px;color:#f3ca52}}
.models{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}
h3{{margin:0 0 7px}}figure{{margin:0 0 12px}}
img{{width:100%;display:block;background:#000}}
figcaption{{padding-top:5px}}
code{{color:#f3ca52}}.legend{{display:flex;gap:14px;flex-wrap:wrap}}
@media(max-width:1000px){{.models{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>{html.escape(case)} · exact moving-ball query overlays</h1>
<p>去噪步 {step}。每张图直接使用对应 latent 时间步的二维 attention，
不进行时间插值。除 t=0 外，每个 latent slice 对齐一个 4 帧输出区间，
背景采用区间末帧。绿色框标记捕获 attention 的四个运动球 query patches。
热力图显示 <code>max(log2(attention / uniform), 0)</code>，全部 Block、Head
和模型共用 0–{vmax:.2f} 色标，因此亮度可以横向比较。</p>
<p class="legend">S=帧内空间；T=球轨迹传播；P=固定位置时间对齐；
C=首帧/历史上下文；G=全局聚合。每个 Block 的五个代表 Head 互不重复；
若标注 modal X，表示该 Head 的综合主角色为 X，当前只是对应指标最强代表。</p>
</header>
<div class="controls">
<button class="previous" title="上一个 latent 时间步" aria-label="上一个 latent 时间步">◀</button>
<input class="latent-slider" type="range" min="0" max="12" step="1" value="2">
<button class="next" title="下一个 latent 时间步" aria-label="下一个 latent 时间步">▶</button>
<output class="time-label">latent t=02 · output frames 05–08 · shown 08</output>
</div>
{''.join(sections)}
<script>
const slider=document.querySelector('.latent-slider');
const label=document.querySelector('.time-label');
const images=[...document.querySelectorAll('.latent-frame')];
function outputRange(t){{return t===0?[0,0]:[4*t-3,4*t]}}
function show(t){{
  t=Math.max(0,Math.min(12,Number(t)));slider.value=t;
  const key=String(t).padStart(2,'0');
  images.forEach(img=>img.src=img.dataset.prefix+'latent_'+key+'.jpg');
  const range=outputRange(t);
  label.textContent='latent t='+key+' · output frames '
    +String(range[0]).padStart(2,'0')+'–'+String(range[1]).padStart(2,'0')
    +' · shown '+String(range[1]).padStart(2,'0');
}}
slider.addEventListener('input',()=>show(slider.value));
document.querySelector('.previous').addEventListener('click',()=>show(Number(slider.value)-1));
document.querySelector('.next').addEventListener('click',()=>show(Number(slider.value)+1));
document.addEventListener('keydown',event=>{{
  if(event.key==='ArrowLeft')show(Number(slider.value)-1);
  if(event.key==='ArrowRight')show(Number(slider.value)+1);
}});
</script>
</body></html>"""
    (output_dir / "block_view.html").write_text(document, encoding="utf-8")


def _write_role_gallery(
    output_dir: Path,
    records: list[dict[str, Any]],
    *,
    case: str,
    step: int,
    vmax: float,
) -> None:
    sections = []
    for role in ROLES:
        role_records = [record for record in records if record["role"] == role]
        block_rows = []
        for block in sorted({int(record["block"]) for record in role_records}):
            cards = []
            for model in MODEL_LABELS:
                matches = [
                    record
                    for record in role_records
                    if int(record["block"]) == block
                    and record["model"] == model
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"expected one role video for {role}/B{block}/{model}"
                    )
                record = matches[0]
                modal_note = (
                    ""
                    if record["modal_role"] == role
                    else f"；综合主角色 {record['modal_role']}"
                )
                cards.append(
                    "<figure>"
                    f"<img class='latent-frame' data-prefix='"
                    f"{html.escape(record['frame_prefix'])}' src='"
                    f"{html.escape(record['frame_prefix'])}latent_02.jpg'>"
                    f"<figcaption>{html.escape(MODEL_LABELS[model])} · "
                    f"H{record['head']:02d} · score {record['score']:.2f} · "
                    f"稳定性 {record['stability']:.0%}"
                    f"{html.escape(modal_note)}</figcaption></figure>"
                )
            block_rows.append(
                "<article class='block-row'>"
                f"<h3>Block {block:02d}</h3>"
                f"<div class='models'>{''.join(cards)}</div></article>"
            )
        sections.append(
            f"<section id='role-{role.lower()}'>"
            f"<h2>{role} · {html.escape(ROLE_NAMES_ZH[role])}</h2>"
            f"<p>{html.escape(ROLE_DESCRIPTIONS[role])}</p>"
            f"{''.join(block_rows)}</section>"
        )

    nav = "".join(
        f"<a href='#role-{role.lower()}'>{role}</a>" for role in ROLES
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Head-type grouped moving-ball overlays</title>
<style>
body{{margin:0;background:#0d0f11;color:#f4f5f6;font:15px/1.45 system-ui,sans-serif}}
header,section{{padding:18px 22px;border-bottom:1px solid #30343a}}
h1,h2,h3{{margin:0 0 8px;letter-spacing:0}}p,figcaption{{color:#b9c0c8}}
nav{{display:flex;gap:18px;margin:12px 0}}nav a{{color:#8fc9ff;font-weight:700}}
.controls{{position:sticky;top:0;z-index:2;display:flex;align-items:center;gap:12px;
padding:12px 22px;background:#16191ddd;border-bottom:1px solid #3b4047;
backdrop-filter:blur(8px)}}.controls input{{width:min(620px,65vw)}}
.controls button{{width:34px;height:34px;border:1px solid #535962;background:#23272c;
color:#fff;cursor:pointer}}.controls output{{min-width:190px;color:#f3ca52}}
.block-row{{padding:14px 0 20px;border-top:1px solid #25292e}}
.models{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}
figure{{margin:0}}img{{width:100%;display:block;background:#000}}
figcaption{{padding-top:5px}}code{{color:#f3ca52}}
@media(max-width:1000px){{.models{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>{html.escape(case)} · 按 Head 类型聚合</h1>
<p>去噪步 {step}。每张图左侧为生成结果，右侧为 source/GT 坐标参照。
热力图直接取当前 latent 时间步，不进行时间插值；除 t=0 外，每个
latent slice 对齐一个 4 帧输出区间，背景采用区间末帧。
同一类别按 Block 纵向排列，三个模型横向并列。所有结果共用
<code>max(log2(attention/uniform),0)</code> 的 0–{vmax:.2f} 色标。</p>
<nav>{nav}<a href='block_view.html'>按 Block 查看</a></nav>
</header>
<div class="controls">
<button class="previous" title="上一个 latent 时间步" aria-label="上一个 latent 时间步">◀</button>
<input class="latent-slider" type="range" min="0" max="12" step="1" value="2">
<button class="next" title="下一个 latent 时间步" aria-label="下一个 latent 时间步">▶</button>
<output class="time-label">latent t=02 · output frames 05–08 · shown 08</output>
</div>
{''.join(sections)}
<script>
const slider=document.querySelector('.latent-slider');
const label=document.querySelector('.time-label');
const images=[...document.querySelectorAll('.latent-frame')];
function outputRange(t){{return t===0?[0,0]:[4*t-3,4*t]}}
function show(t){{
  t=Math.max(0,Math.min(12,Number(t)));slider.value=t;
  const key=String(t).padStart(2,'0');
  images.forEach(img=>img.src=img.dataset.prefix+'latent_'+key+'.jpg');
  const range=outputRange(t);
  label.textContent='latent t='+key+' · output frames '
    +String(range[0]).padStart(2,'0')+'–'+String(range[1]).padStart(2,'0')
    +' · shown '+String(range[1]).padStart(2,'0');
}}
slider.addEventListener('input',()=>show(slider.value));
document.querySelector('.previous').addEventListener('click',()=>show(Number(slider.value)-1));
document.querySelector('.next').addEventListener('click',()=>show(Number(slider.value)+1));
document.addEventListener('keydown',event=>{{
  if(event.key==='ArrowLeft')show(Number(slider.value)-1);
  if(event.key==='ArrowRight')show(Number(slider.value)+1);
}});
</script>
</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    multiblock_root = args.multiblock_root.expanduser().resolve()
    block17_root = args.block17_root.expanduser().resolve()
    roles_csv = args.roles_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.ffmpeg.is_file():
        raise FileNotFoundError(args.ffmpeg)

    blocks = [int(value) for value in args.blocks.split(",") if value.strip()]
    models = tuple(MODEL_LABELS)
    rows_by_block = _load_role_rows(roles_csv)
    selections = {
        block: _select_distinct_role_heads(rows_by_block[block])
        for block in blocks
    }

    captures: dict[tuple[int, str], dict[str, Any]] = {}
    positive_values = []
    for block in blocks:
        block_root = _block_root(block, multiblock_root, block17_root)
        selected_heads = [
            selections[block][role]["head"] for role in ROLES
        ]
        for model in models:
            attention, query_coords, grid = _matrix_for_step(
                block_root, model, args.case, args.step
            )
            captures[(block, model)] = {
                "attention": attention,
                "query_coords": query_coords,
                "grid": grid,
                "video": _generated_video(block_root, model, args.case),
            }
            token_count = math.prod(grid)
            selected = attention[selected_heads] * float(token_count)
            values = np.maximum(np.log2(np.maximum(selected, 1.0)), 0.0)
            positive_values.append(values[values > 0])
    all_positive = np.concatenate(positive_values)
    vmax = max(
        float(np.percentile(all_positive, args.color_percentile)), 1.0
    )

    original_dir = output_dir / "originals"
    original_dir.mkdir(exist_ok=True)
    reference_capture = captures[(blocks[0], models[0])]
    source_video = _source_video(reference_capture["video"])
    shutil.copy2(source_video, original_dir / "source_gt.mp4")
    for model in models:
        shutil.copy2(
            captures[(blocks[0], model)]["video"],
            original_dir / f"{model}.mp4",
        )

    records: list[dict[str, Any]] = []
    for block in blocks:
        for model in models:
            capture = captures[(block, model)]
            variants = {
                "generated": (
                    capture["video"],
                    "generated",
                ),
                "source_gt": (
                    source_video,
                    "source/GT",
                ),
            }
            for variant, (video_path, video_kind) in variants.items():
                relative_dir = (
                    Path("frames")
                    / "block_view"
                    / f"block{block:02d}"
                    / model
                    / variant
                )
                frame_names = _write_block_frame_sequence(
                    video_path=video_path,
                    output_dir=output_dir / relative_dir,
                    block=block,
                    model=model,
                    video_kind=video_kind,
                    step=args.step,
                    attention=capture["attention"],
                    query_coords=capture["query_coords"],
                    grid=capture["grid"],
                    selections=selections[block],
                    vmax=vmax,
                    alpha=args.alpha,
                    panel_size=(args.panel_width, args.panel_height),
                    max_frames=args.max_frames,
                )
                records.append(
                    {
                        "block": block,
                        "model": model,
                        "variant": variant,
                        "frame_prefix": relative_dir.as_posix() + "/",
                        "frames": frame_names,
                    }
                )

    role_records: list[dict[str, Any]] = []
    for role in ROLES:
        for block in blocks:
            selection = selections[block][role]
            for model in models:
                capture = captures[(block, model)]
                relative_dir = (
                    Path("frames")
                    / "role_view"
                    / role.lower()
                    / f"block{block:02d}"
                    / model
                )
                frame_names = _write_role_frame_sequence(
                    generated_video=capture["video"],
                    source_video=source_video,
                    output_dir=output_dir / relative_dir,
                    block=block,
                    model=model,
                    role=role,
                    attention=capture["attention"],
                    query_coords=capture["query_coords"],
                    grid=capture["grid"],
                    selection=selection,
                    vmax=vmax,
                    alpha=args.alpha,
                    panel_size=(args.panel_width, args.panel_height),
                    max_frames=args.max_frames,
                )
                role_records.append(
                    {
                        "role": role,
                        "block": block,
                        "model": model,
                        "head": int(selection["head"]),
                        "score": float(selection["target_score"]),
                        "modal_role": str(selection["modal_role"]),
                        "stability": float(selection["role_stability"]),
                        "frame_prefix": relative_dir.as_posix() + "/",
                        "frames": frame_names,
                    }
                )

    report = {
        "case": args.case,
        "step": args.step,
        "blocks": blocks,
        "models": list(models),
        "heat_definition": "max(log2(attention_probability / uniform_probability), 0)",
        "shared_vmax": vmax,
        "color_percentile": args.color_percentile,
        "query_note": (
            "Green rectangle marks the four exact moving-ball query patches "
            "at output frame 8 / latent time 2."
        ),
        "temporal_alignment": {
            "latent_frames": 13,
            "output_frames": 49,
            "output_frame_ranges": {
                str(latent_t): list(_latent_output_range(latent_t))
                for latent_t in range(13)
            },
            "background_frame": {
                str(latent_t): _latent_to_output_frame(latent_t)
                for latent_t in range(13)
            },
            "interpolation": "none",
            "heat_slice": "attention[head, latent_t, :, :]",
        },
        "selection_note": (
            "Five distinct heads per block maximize the five role scores jointly. "
            "A role representative can have another modal role when functions overlap."
        ),
        "selections": selections,
        "block_comparison_frames": records,
        "role_grouped_frames": role_records,
    }
    (output_dir / "overlay_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_block_gallery(
        output_dir,
        records,
        selections,
        case=args.case,
        step=args.step,
        vmax=vmax,
    )
    _write_role_gallery(
        output_dir,
        role_records,
        case=args.case,
        step=args.step,
        vmax=vmax,
    )
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "block_frame_sequences": len(records),
                "role_frame_sequences": len(role_records),
                "shared_vmax": vmax,
                "selections": selections,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
