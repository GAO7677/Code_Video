#!/usr/bin/env python3
"""Render denoising-step attention audits on a common source-frame geometry."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FFMPEG = Path("/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg")
PANEL_WIDTH = 420
INK = (11, 24, 39)
WHITE = (244, 248, 251)
MUTED = (167, 188, 203)
GT = (61, 238, 151)
AMBER = (255, 174, 66)
CYAN = (39, 211, 224)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _atomic_jpeg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.jpg")
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
        temporary,
        format="JPEG",
        quality=91,
        optimize=True,
        subsampling=0,
    )
    temporary.replace(path)


def _finite(value: float | np.floating[Any]) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _heat_rgb(values: np.ndarray) -> np.ndarray:
    """Small perceptual blue-cyan-yellow heat palette without external state."""
    x = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    stops = np.asarray(
        [
            [10, 25, 55],
            [22, 86, 142],
            [31, 191, 194],
            [246, 211, 73],
            [245, 82, 67],
        ],
        dtype=np.float32,
    )
    scaled = x * (len(stops) - 1)
    left = np.floor(scaled).astype(np.int64)
    right = np.minimum(left + 1, len(stops) - 1)
    weight = (scaled - left)[..., None]
    return np.clip(stops[left] * (1.0 - weight) + stops[right] * weight, 0, 255).astype(
        np.uint8
    )


def _resize_map(values: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    height, width = output_hw
    image = Image.fromarray(np.asarray(values, dtype=np.float32), mode="F")
    return np.asarray(
        image.resize((width, height), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def _overlay_heat(frame: np.ndarray, heat: np.ndarray, scale: float) -> np.ndarray:
    resized = _resize_map(heat, frame.shape[:2])
    normalized = np.clip(resized / max(float(scale), 1.0e-12), 0.0, 1.0)
    color = _heat_rgb(normalized)
    alpha = (0.72 * np.power(normalized, 0.55))[..., None]
    return np.clip(frame.astype(np.float32) * (1.0 - alpha) + color * alpha, 0, 255).astype(
        np.uint8
    )


def _overlay_difference(
    frame: np.ndarray, difference: np.ndarray, scale: float
) -> np.ndarray:
    resized = _resize_map(difference, frame.shape[:2])
    normalized = np.clip(resized / max(float(scale), 1.0e-12), -1.0, 1.0)
    positive = np.asarray([248, 82, 111], dtype=np.float32)
    negative = np.asarray([36, 183, 220], dtype=np.float32)
    color = np.where(normalized[..., None] >= 0, positive, negative)
    alpha = (0.78 * np.power(np.abs(normalized), 0.55))[..., None]
    return np.clip(frame.astype(np.float32) * (1.0 - alpha) + color * alpha, 0, 255).astype(
        np.uint8
    )


def _draw_points(
    image: Image.Image,
    points: np.ndarray,
    visibility: np.ndarray,
    original_hw: tuple[int, int],
) -> None:
    draw = ImageDraw.Draw(image)
    original_height, original_width = original_hw
    scale_x = image.width / float(original_width)
    scale_y = image.height / float(original_height)
    for point_index, ((x, y), visible) in enumerate(
        zip(points, visibility, strict=True)
    ):
        if not bool(visible) or not np.isfinite((x, y)).all():
            continue
        px, py = float(x) * scale_x, float(y) * scale_y
        radius = 5
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=INK,
            outline=GT,
            width=3,
        )
        draw.text((px + 7, py - 10), f"i{point_index}", fill=GT, font=_font(11, True))


def _panel(
    frame: np.ndarray,
    title: str,
    subtitle: str,
    points: np.ndarray,
    visibility: np.ndarray,
) -> np.ndarray:
    original_height, original_width = frame.shape[:2]
    panel_height = max(2, int(round(PANEL_WIDTH * original_height / original_width)))
    if panel_height % 2:
        panel_height += 1
    image = Image.fromarray(frame).resize(
        (PANEL_WIDTH, panel_height), Image.Resampling.LANCZOS
    )
    _draw_points(image, points, visibility, (original_height, original_width))
    canvas = Image.new("RGB", (PANEL_WIDTH, panel_height + 74), INK)
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, panel_height + 9), title, fill=WHITE, font=_font(16, True))
    draw.text((12, panel_height + 36), subtitle, fill=MUTED, font=_font(11))
    return np.asarray(canvas)


def _format_metric(value: float | np.floating[Any], digits: int = 4) -> str:
    return "N/A" if not np.isfinite(value) else f"{float(value):.{digits}f}"


def _read_video_frames(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"could not decode generated video: {path}")
    return np.stack(frames, axis=0)


def ensure_generated_frame_attention_overlays(
    generated_video: Path,
    step_directory: Path,
) -> Path:
    """Render 13 static Final-RGB | PRE | POST frame comparisons on demand.

    Attention is still measured at the denoising step stored in ``step_directory``.
    The final generated RGB frames are only the common visualization canvas.
    """
    output_directory = step_directory / "generated_frame_overlays"
    marker = output_directory / "complete.json"
    expected = [output_directory / f"R{latent_time:02d}.jpg" for latent_time in range(13)]
    if marker.is_file() and all(path.is_file() and path.stat().st_size > 0 for path in expected):
        return output_directory

    raw_path = step_directory / "raw_attention_maps.npz"
    metrics_path = step_directory / "metrics.json"
    if not generated_video.is_file() or not raw_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("generated video or attention audit inputs are incomplete")

    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    frame_metrics = report.get("frames", [])
    with np.load(raw_path) as payload:
        pre_heat = np.asarray(payload["pre_heatmap"], dtype=np.float32)
        post_heat = np.asarray(payload["post_heatmap"], dtype=np.float32)
        tracks = np.asarray(payload["tracks_tn2"], dtype=np.float32)
        visibility = np.asarray(payload["visibility_tn"], dtype=bool)
        source_indices = np.asarray(payload["source_frame_indices"], dtype=np.int64)
    if pre_heat.shape[0] != 13 or post_heat.shape != pre_heat.shape:
        raise RuntimeError(f"invalid PRE/POST attention shapes: {pre_heat.shape}, {post_heat.shape}")

    # Direct-attention interventions store the same auditable PRE/POST maps but
    # do not have latent-guidance-specific per-frame loss summaries.  Derive
    # the display-only frame mass fields so both mechanisms can use the exact
    # same static overlay renderer without inventing a second visual grammar.
    if len(frame_metrics) != 13:
        direction = str(report.get("summary", {}).get("direction", "direct"))
        frame_metrics = [
            {
                "role": f"{direction} key map",
                "pre_frame_mass": float(pre_heat[latent_time].sum()),
                "post_frame_mass": float(post_heat[latent_time].sum()),
                "pre_localized_mass": np.nan,
                "post_localized_mass": np.nan,
                "pre_peak_distance_tokens": np.nan,
                "post_peak_distance_tokens": np.nan,
            }
            for latent_time in range(13)
        ]

    video_frames = _read_video_frames(generated_video)
    if int(source_indices.max()) >= len(video_frames):
        raise RuntimeError(
            f"anchor F{int(source_indices.max())} exceeds generated video with "
            f"{len(video_frames)} frames"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    summary = report.get("summary", {})
    step_1based = int(summary.get("step_1based", int(step_directory.name.split("_")[-1])))

    for latent_time, frame_index in enumerate(source_indices.tolist()):
        generated = np.asarray(video_frames[int(frame_index)], dtype=np.uint8)
        common_scale = max(
            float(np.quantile(pre_heat[latent_time], 0.995)),
            float(np.quantile(post_heat[latent_time], 0.995)),
            1.0e-12,
        )
        row = frame_metrics[latent_time]
        role = str(row.get("role", "future")).upper()
        points = tracks[latent_time]
        visible = visibility[latent_time]
        panels = [
            _panel(
                generated,
                f"FINAL GENERATED · R{latent_time:02d} · {role}",
                f"RGB F{int(frame_index):02d} · green = target GT/pseudo-GT points",
                points,
                visible,
            ),
            _panel(
                _overlay_heat(generated, pre_heat[latent_time], common_scale),
                f"PRE ATTENTION · STEP {step_1based:02d}",
                "mass "
                f"{_format_metric(float(row.get('pre_frame_mass', np.nan)))} · local "
                f"{_format_metric(float(row.get('pre_localized_mass', np.nan)))} · peak d "
                f"{_format_metric(float(row.get('pre_peak_distance_tokens', np.nan)), 2)} tok",
                points,
                visible,
            ),
            _panel(
                _overlay_heat(generated, post_heat[latent_time], common_scale),
                f"POST ATTENTION · STEP {step_1based:02d}",
                "mass "
                f"{_format_metric(float(row.get('post_frame_mass', np.nan)))} · local "
                f"{_format_metric(float(row.get('post_localized_mass', np.nan)))} · peak d "
                f"{_format_metric(float(row.get('post_peak_distance_tokens', np.nan)), 2)} tok",
                points,
                visible,
            ),
        ]
        _atomic_jpeg(expected[latent_time], np.concatenate(panels, axis=1))

    _atomic_json(
        marker,
        {
            "layout": "final generated RGB | PRE attention | POST attention",
            "step_1based": step_1based,
            "latent_frames": 13,
            "generated_video": str(generated_video),
            "attention_semantics": (
                "attention was measured at the selected denoising step; final RGB is "
                "used only as a common visualization canvas"
            ),
        },
    )
    return output_directory


def _encode(frames: list[np.ndarray], output: Path, fps: int = 2) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.mp4")
    height, width = frames[0].shape[:2]
    command = [
        str(FFMPEG),
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    if return_code or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {output}")
    temporary.replace(output)


def write_step_attention_audit(
    output_directory: Path,
    step_1based: int,
    source_frames_thwc: np.ndarray,
    source_frame_indices: np.ndarray,
    tracks_tn2: np.ndarray,
    visibility_tn: np.ndarray,
    capture: dict[str, Any],
    predicted_x0_thwc: np.ndarray,
    context_latent_frames: int,
    pre_loss: float,
    post_loss: float,
) -> dict[str, Any]:
    """Persist raw maps, exact metrics and a five-panel 13-anchor audit video."""
    pre = capture["pre"]
    post = capture["post"]
    pre_heat = np.asarray(pre["heatmap"], dtype=np.float32)
    post_heat = np.asarray(post["heatmap"], dtype=np.float32)
    if pre_heat.shape != post_heat.shape or pre_heat.shape[0] != 13:
        raise RuntimeError(f"invalid attention heatmap shapes: {pre_heat.shape}, {post_heat.shape}")
    if source_frames_thwc.shape[0] != 13 or predicted_x0_thwc.shape[0] != 13:
        raise RuntimeError("source and predicted-x0 references must contain 13 anchors")
    step_directory = output_directory / f"step_{int(step_1based):02d}"
    _atomic_npz(
        step_directory / "raw_attention_maps.npz",
        pre_heatmap=pre_heat.astype(np.float16),
        post_heatmap=post_heat.astype(np.float16),
        pre_frame_mass=np.asarray(pre["frame_mass"], dtype=np.float32),
        post_frame_mass=np.asarray(post["frame_mass"], dtype=np.float32),
        pre_localized_mass=np.asarray(pre["localized_mass"], dtype=np.float32),
        post_localized_mass=np.asarray(post["localized_mass"], dtype=np.float32),
        pre_peak_distance_tokens=np.asarray(pre["peak_distance_tokens"], dtype=np.float32),
        post_peak_distance_tokens=np.asarray(post["peak_distance_tokens"], dtype=np.float32),
        pre_peak_hit_rate_2sigma=np.asarray(pre["peak_hit_rate_2sigma"], dtype=np.float32),
        post_peak_hit_rate_2sigma=np.asarray(post["peak_hit_rate_2sigma"], dtype=np.float32),
        tracks_tn2=np.asarray(tracks_tn2, dtype=np.float32),
        visibility_tn=np.asarray(visibility_tn, dtype=np.uint8),
        source_frame_indices=np.asarray(source_frame_indices, dtype=np.int64),
    )
    frames: list[np.ndarray] = []
    frame_rows: list[dict[str, Any]] = []
    for latent_time in range(13):
        source = np.asarray(source_frames_thwc[latent_time], dtype=np.uint8)
        predicted = np.asarray(predicted_x0_thwc[latent_time], dtype=np.uint8)
        common_scale = max(
            float(np.quantile(pre_heat[latent_time], 0.995)),
            float(np.quantile(post_heat[latent_time], 0.995)),
            1.0e-12,
        )
        difference = post_heat[latent_time] - pre_heat[latent_time]
        difference_scale = max(float(np.quantile(np.abs(difference), 0.995)), 1.0e-12)
        points = tracks_tn2[latent_time]
        visible = visibility_tn[latent_time]
        pre_frame_mass = float(pre["frame_mass"][latent_time])
        post_frame_mass = float(post["frame_mass"][latent_time])
        pre_local = float(pre["localized_mass"][latent_time])
        post_local = float(post["localized_mass"][latent_time])
        pre_distance = float(pre["peak_distance_tokens"][latent_time])
        post_distance = float(post["peak_distance_tokens"][latent_time])
        role = "CONTEXT" if latent_time < int(context_latent_frames) else "FUTURE"
        panels = [
            _panel(
                source,
                f"SOURCE GT · R{latent_time:02d} · {role}",
                f"source F{int(source_frame_indices[latent_time]):02d} · green = same-ID points",
                points,
                visible,
            ),
            _panel(
                _overlay_heat(source, pre_heat[latent_time], common_scale),
                "PRE · original attention",
                f"frame mass {_format_metric(pre_frame_mass)} · local {_format_metric(pre_local)} · peak d {_format_metric(pre_distance,2)} tok",
                points,
                visible,
            ),
            _panel(
                _overlay_heat(source, post_heat[latent_time], common_scale),
                "POST · after latent constraint",
                f"frame mass {_format_metric(post_frame_mass)} · local {_format_metric(post_local)} · peak d {_format_metric(post_distance,2)} tok",
                points,
                visible,
            ),
            _panel(
                _overlay_difference(source, difference, difference_scale),
                "POST - PRE attention",
                f"red=increase · blue=decrease · Δlocal {_format_metric(post_local-pre_local)}",
                points,
                visible,
            ),
            _panel(
                predicted,
                f"PREDICTED x0 · STEP {int(step_1based):02d}",
                f"FlowMatch x_s - sigma*v_CFG · loss {pre_loss:.4f}→{post_loss:.4f}",
                points,
                visible,
            ),
        ]
        frames.append(np.concatenate(panels, axis=1))
        frame_rows.append(
            {
                "latent_time": latent_time,
                "source_frame": int(source_frame_indices[latent_time]),
                "role": role.lower(),
                "pre_frame_mass": _finite(pre_frame_mass),
                "post_frame_mass": _finite(post_frame_mass),
                "delta_frame_mass": _finite(post_frame_mass - pre_frame_mass),
                "pre_localized_mass": _finite(pre_local),
                "post_localized_mass": _finite(post_local),
                "delta_localized_mass": _finite(post_local - pre_local),
                "pre_peak_distance_tokens": _finite(pre_distance),
                "post_peak_distance_tokens": _finite(post_distance),
                "delta_peak_distance_tokens": _finite(post_distance - pre_distance),
                "pre_peak_hit_rate_2sigma": _finite(pre["peak_hit_rate_2sigma"][latent_time]),
                "post_peak_hit_rate_2sigma": _finite(post["peak_hit_rate_2sigma"][latent_time]),
            }
        )
    _encode(frames, step_directory / "attention_comparison.mp4")
    future = slice(int(context_latent_frames), 13)
    summary = {
        "step_1based": int(step_1based),
        "step_index": int(step_1based - 1),
        "pre_loss": float(pre_loss),
        "post_loss": float(post_loss),
        "loss_change": float(post_loss - pre_loss),
        "future_mean_pre_frame_mass": _finite(np.nanmean(pre["frame_mass"][future])),
        "future_mean_post_frame_mass": _finite(np.nanmean(post["frame_mass"][future])),
        "future_mean_pre_localized_mass": _finite(np.nanmean(pre["localized_mass"][future])),
        "future_mean_post_localized_mass": _finite(np.nanmean(post["localized_mass"][future])),
        "future_mean_pre_peak_distance_tokens": _finite(
            np.nanmean(pre["peak_distance_tokens"][future])
        ),
        "future_mean_post_peak_distance_tokens": _finite(
            np.nanmean(post["peak_distance_tokens"][future])
        ),
        "future_mean_pre_peak_hit_rate_2sigma": _finite(
            np.nanmean(pre["peak_hit_rate_2sigma"][future])
        ),
        "future_mean_post_peak_hit_rate_2sigma": _finite(
            np.nanmean(post["peak_hit_rate_2sigma"][future])
        ),
    }
    summary.update(
        {
            "delta_future_frame_mass": _difference(
                summary["future_mean_post_frame_mass"],
                summary["future_mean_pre_frame_mass"],
            ),
            "delta_future_localized_mass": _difference(
                summary["future_mean_post_localized_mass"],
                summary["future_mean_pre_localized_mass"],
            ),
            "delta_future_peak_distance_tokens": _difference(
                summary["future_mean_post_peak_distance_tokens"],
                summary["future_mean_pre_peak_distance_tokens"],
            ),
            "delta_future_peak_hit_rate_2sigma": _difference(
                summary["future_mean_post_peak_hit_rate_2sigma"],
                summary["future_mean_pre_peak_hit_rate_2sigma"],
            ),
        }
    )
    report = {
        "protocol": "wan_forward_attention_overlay_audit_v1",
        "normalization": "global softmax over all 13*H*W Keys",
        "aggregation": "pair-weighted mean over selected heads and visible context point Queries",
        "overlay_scale": "shared Pre/Post p99.5 within each latent frame; raw temporal mass is printed",
        "difference_color": "red = Post increase; blue = Post decrease",
        "summary": summary,
        "frames": frame_rows,
    }
    _atomic_json(step_directory / "metrics.json", report)
    _atomic_json(step_directory / "complete.json", {"step_1based": int(step_1based)})
    return report
