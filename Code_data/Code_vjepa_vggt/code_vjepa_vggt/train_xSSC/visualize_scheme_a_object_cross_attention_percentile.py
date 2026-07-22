#!/usr/bin/env python3
"""Percentile mask visualization for Scheme A oracle-slot object cross-attention."""
from __future__ import annotations

import csv
import html
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
from matplotlib import pyplot as plt

_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)


def _pop_option(argv: list[str], name: str, default: str) -> str:
    values: list[str] = []
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == name:
            if index + 1 >= len(argv):
                raise SystemExit(f"{name} requires a value")
            values.append(argv[index + 1])
            del argv[index : index + 2]
            continue
        if token.startswith(f"{name}="):
            values.append(token.split("=", 1)[1])
            del argv[index]
            continue
        index += 1
    if not values:
        return default
    if len(set(values)) != 1:
        raise SystemExit(f"conflicting {name} values: {values}")
    return values[-1]


OBJECT_ATTN_QUERY_CHUNK = int(_pop_option(sys.argv, "--scheme-a-attn-query-chunk", "2048"))
OBJECT_ATTN_SLOT_COUNT = int(_pop_option(sys.argv, "--scheme-a-attn-slots", "7"))
OBJECT_ATTN_TEMPORAL_AGGS = [
    part.strip().lower()
    for part in _pop_option(sys.argv, "--scheme-a-attn-temporal-aggs", "aligned,sum,max").split(",")
    if part.strip()
]
OBJECT_ATTN_SELECTED_LAYERS = [
    int(part.strip())
    for part in _pop_option(sys.argv, "--scheme-a-attn-layers", "5,11,17,29").split(",")
    if part.strip()
]
OBJECT_ATTN_PERCENTILE = float(_pop_option(sys.argv, "--scheme-a-attn-percentile", "90"))
OBJECT_ATTN_BEST_STAGE = _pop_option(sys.argv, "--scheme-a-attn-best-stage", "all").strip().lower()

for mode in OBJECT_ATTN_TEMPORAL_AGGS:
    if mode not in {"aligned", "sum", "max"}:
        raise SystemExit("--scheme-a-attn-temporal-aggs supports aligned,sum,max")
if OBJECT_ATTN_BEST_STAGE not in {"early", "middle", "late", "all"}:
    raise SystemExit("--scheme-a-attn-best-stage must be early, middle, late, or all")

from code_vjepa_vggt.train_xSSC import batch_infer_xssc_allframe_oracle_slots as scheme_a_batch

FFMPEG = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg")
SLOT_COLORS_BGR = [
    (68, 68, 239),
    (246, 130, 59),
    (94, 197, 34),
    (21, 204, 250),
    (247, 85, 168),
    (212, 182, 6),
    (22, 115, 249),
]


def _read_video_bgr(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 8.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video has no frames: {path}")
    return frames, fps


def _temporal_resize_lowres(attn: np.ndarray, output_frames: int) -> np.ndarray:
    tensor = torch.from_numpy(attn.astype(np.float32))[None, None]
    resized = F.interpolate(
        tensor,
        size=(int(output_frames), int(attn.shape[1]), int(attn.shape[2])),
        mode="trilinear",
        align_corners=True,
    )
    return resized[0, 0].numpy()


def _threshold_overlay(
    frame: np.ndarray,
    heat_lowres: np.ndarray,
    threshold: float,
    *,
    slot_id: int,
) -> np.ndarray:
    heat = cv2.resize(
        heat_lowres.astype(np.float32),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    mask = heat > float(threshold)
    color = np.zeros_like(frame)
    color[:, :] = SLOT_COLORS_BGR[int(slot_id) % len(SLOT_COLORS_BGR)]
    out = frame.copy()
    if np.any(mask):
        blended = (
            frame[mask].astype(np.float32) * 0.42
            + color[mask].astype(np.float32) * 0.58
        )
        out[mask] = np.clip(blended, 0, 255).astype(np.uint8)
    boundary = cv2.Canny(mask.astype(np.uint8) * 255, 40, 120)
    out[boundary > 0] = (255, 255, 255)
    return out


def _label(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    label_height = 26 * len(lines) + 8
    out = cv2.copyMakeBorder(
        frame,
        label_height,
        0,
        0,
        0,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    for index, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (8, 22 + 26 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _write_jpeg(path: Path, frame: np.ndarray, *, quality: int = 84) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"failed to write image: {path}")


def _write_h264(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError(f"cannot write empty video: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".temporary.mp4")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {temporary}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(temporary),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
    )
    temporary.unlink()


def _centered_cosine_matrix(x: np.ndarray) -> np.ndarray:
    flat = x.astype(np.float64).reshape(x.shape[0], -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    flat = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1.0e-12)
    return flat @ flat.T


def _offdiag_mean(matrix: np.ndarray) -> float:
    if matrix.shape[0] <= 1:
        return 0.0
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(matrix[mask].mean())


class SchemeAObjectCrossAttentionRecorder:
    def __init__(
        self,
        *,
        total_steps: int,
        slot_count: int,
        temporal_aggs: list[str],
        query_chunk: int,
    ) -> None:
        self.total_steps = int(total_steps)
        self.slot_count = int(slot_count)
        self.temporal_aggs = list(temporal_aggs)
        self.query_chunk = int(query_chunk)
        self.active = False
        self.step_index = -1
        self.grid: tuple[int, int, int] | None = None
        self.layer_count = 0
        self.key_time_steps: int | None = None
        self.key_count: int | None = None
        self._original_forwards: list[tuple[Any, Any]] = []
        self.layer_sums: dict[str, dict[str, np.ndarray]] = {
            mode: {} for mode in self.temporal_aggs
        }
        self.layer_counts: dict[str, dict[str, np.ndarray]] = {
            mode: {} for mode in self.temporal_aggs
        }

    def install(self, dit) -> None:
        self.layer_count = len(dit.blocks)
        for layer_id, block in enumerate(dit.blocks):
            cross_attn = getattr(block, "object_cross_attn", None)
            attn = getattr(cross_attn, "attn", None)
            if attn is None:
                raise RuntimeError(f"Wan block {layer_id} has no object cross-attention module")
            original = attn.forward
            num_heads = int(getattr(attn, "num_heads", 24))

            def wrapped(q, k, v, *, _original=original, _layer_id=layer_id, _num_heads=num_heads):
                output = _original(q, k, v)
                if self.active:
                    self.capture(layer_id=int(_layer_id), q=q, k=k, num_heads=int(_num_heads))
                return output

            attn.forward = wrapped
            self._original_forwards.append((attn, original))

    def restore(self) -> None:
        for module, original in self._original_forwards:
            module.forward = original
        self._original_forwards.clear()

    def _stage_names(self, step_index: int) -> tuple[str, str]:
        one_third = max(1, self.total_steps // 3)
        two_thirds = max(one_third + 1, (2 * self.total_steps) // 3)
        if step_index < one_third:
            stage = "early"
        elif step_index < two_thirds:
            stage = "middle"
        else:
            stage = "late"
        return ("all", stage)

    def _ensure_layer_accumulator(self, mode: str, stage: str) -> np.ndarray:
        if self.grid is None:
            raise RuntimeError("attention grid is missing")
        frames, grid_h, grid_w = self.grid
        if stage not in self.layer_sums[mode]:
            self.layer_sums[mode][stage] = np.zeros(
                (self.layer_count, self.slot_count, frames, grid_h, grid_w),
                dtype=np.float64,
            )
            self.layer_counts[mode][stage] = np.zeros((self.layer_count,), dtype=np.int64)
        return self.layer_sums[mode][stage]

    def _query_frame_to_key_time(self, query_frames: torch.Tensor, key_time_steps: int) -> torch.Tensor:
        if self.grid is None:
            raise RuntimeError("attention grid is missing")
        frames = int(self.grid[0])
        if frames <= 1 or key_time_steps <= 1:
            return torch.zeros_like(query_frames)
        scaled = query_frames.float() * float(key_time_steps - 1) / float(frames - 1)
        return torch.round(scaled).long().clamp(0, key_time_steps - 1)

    @torch.no_grad()
    def capture(self, *, layer_id: int, q: torch.Tensor, k: torch.Tensor, num_heads: int) -> None:
        if self.grid is None:
            raise RuntimeError("attention grid was not configured before capture")
        frames, grid_h, grid_w = self.grid
        expected_queries = int(frames * grid_h * grid_w)
        if int(q.shape[0]) != 1 or int(q.shape[1]) != expected_queries:
            raise RuntimeError(f"object attention q={list(q.shape)} does not match grid={self.grid}")
        key_count = int(k.shape[1])
        if key_count % self.slot_count != 0:
            raise RuntimeError(f"object key count {key_count} is not divisible by slots={self.slot_count}")
        key_time_steps = key_count // self.slot_count
        self.key_count = key_count
        self.key_time_steps = key_time_steps
        if int(q.shape[-1]) % int(num_heads) != 0:
            raise RuntimeError(f"object attention dim={q.shape[-1]} is not divisible by heads={num_heads}")

        head_dim = int(q.shape[-1]) // int(num_heads)
        keys = k.view(1, key_count, int(num_heads), head_dim).permute(0, 2, 1, 3).float()
        key_t = keys.transpose(-1, -2)
        outputs = {mode: [] for mode in self.temporal_aggs}
        pixels_per_frame = int(grid_h * grid_w)
        for start in range(0, int(q.shape[1]), self.query_chunk):
            stop = min(start + self.query_chunk, int(q.shape[1]))
            queries = (
                q[:, start:stop]
                .view(1, stop - start, int(num_heads), head_dim)
                .permute(0, 2, 1, 3)
                .float()
            )
            scores = torch.matmul(queries, key_t) / math.sqrt(float(head_dim))
            probs = torch.softmax(scores, dim=-1)
            grouped = probs.view(1, int(num_heads), stop - start, key_time_steps, self.slot_count)
            if "sum" in outputs:
                outputs["sum"].append(grouped.sum(dim=3).mean(dim=(0, 1)).cpu())
            if "max" in outputs:
                outputs["max"].append(grouped.max(dim=3).values.mean(dim=(0, 1)).cpu())
            if "aligned" in outputs:
                query_ids = torch.arange(start, stop, device=grouped.device)
                query_frames = query_ids.div(pixels_per_frame, rounding_mode="floor")
                key_ids = self._query_frame_to_key_time(query_frames, key_time_steps)
                gather_index = key_ids.view(1, 1, stop - start, 1, 1).expand(
                    1, int(num_heads), stop - start, 1, self.slot_count
                )
                aligned = grouped.gather(dim=3, index=gather_index).squeeze(3)
                outputs["aligned"].append(aligned.mean(dim=(0, 1)).cpu())
            del scores, probs, grouped, queries

        for mode, chunks in outputs.items():
            if not chunks:
                continue
            flat = torch.cat(chunks, dim=0).transpose(0, 1)
            array = flat.reshape(self.slot_count, frames, grid_h, grid_w).numpy()
            for stage in self._stage_names(int(self.step_index)):
                self._ensure_layer_accumulator(mode, stage)[int(layer_id)] += array
                self.layer_counts[mode][stage][int(layer_id)] += 1
            del flat, array

    def averaged_by_layer(self) -> dict[str, dict[str, np.ndarray]]:
        output: dict[str, dict[str, np.ndarray]] = {}
        for mode, stages in self.layer_sums.items():
            output[mode] = {}
            for stage, value in stages.items():
                counts = np.maximum(self.layer_counts[mode][stage].astype(np.float64), 1.0)
                output[mode][stage] = (value / counts[:, None, None, None, None]).astype(np.float32)
        return output


class ModelFnObjectAttentionScope:
    def __init__(self, *, pipe, attention: SchemeAObjectCrossAttentionRecorder, cfg_scale: float) -> None:
        self.pipe = pipe
        self.attention = attention
        self.calls_per_step = 1 if abs(float(cfg_scale) - 1.0) < 1.0e-8 else 2
        self.call_index = 0
        self.original = pipe.model_fn

    def install(self) -> None:
        self.pipe.model_fn = self

    def restore(self) -> None:
        self.pipe.model_fn = self.original

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        step_index = self.call_index // self.calls_per_step
        phase_index = self.call_index % self.calls_per_step
        positive_branch = phase_index == 0
        if positive_branch and step_index < self.attention.total_steps:
            latents = kwargs.get("latents")
            dit = kwargs.get("dit")
            if latents is None or dit is None:
                raise RuntimeError("model_fn did not receive latents/dit for attention tracing")
            patch = tuple(int(value) for value in getattr(dit, "patch_size", (1, 2, 2)))
            self.attention.grid = (
                int(latents.shape[2]) // patch[0],
                int(latents.shape[3]) // patch[1],
                int(latents.shape[4]) // patch[2],
            )
            self.attention.step_index = int(step_index)
            self.attention.active = True
        else:
            self.attention.active = False
        try:
            return self.original(*args, **kwargs)
        finally:
            self.attention.active = False
            self.call_index += 1


def _layer_saliency_rows(layer_maps: dict[str, dict[str, np.ndarray]]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for mode, stages in layer_maps.items():
        for stage, maps in stages.items():
            for layer_id in range(int(maps.shape[0])):
                layer = maps[layer_id].astype(np.float32)
                mean_value = float(layer.mean())
                std_value = float(layer.std())
                peak_to_mean = float(layer.max() / max(mean_value, 1.0e-12))
                cv = float(std_value / max(mean_value, 1.0e-12))
                centered_cross = []
                for frame_id in range(int(layer.shape[1])):
                    centered_cross.append(_offdiag_mean(_centered_cosine_matrix(layer[:, frame_id])))
                cross_slot_centered_mean = float(np.mean(centered_cross))
                slot_separation = float(1.0 - cross_slot_centered_mean)
                score = float(cv * slot_separation)
                rows.append(
                    {
                        "temporal_agg": mode,
                        "stage": stage,
                        "layer": int(layer_id),
                        "score": score,
                        "coefficient_of_variation": cv,
                        "peak_to_mean": peak_to_mean,
                        "cross_slot_centered_mean": cross_slot_centered_mean,
                        "slot_separation": slot_separation,
                        "raw_mean": mean_value,
                        "raw_std": std_value,
                        "raw_min": float(layer.min()),
                        "raw_max": float(layer.max()),
                    }
                )
    return rows


def _write_layer_ranking_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    ordered = sorted(
        rows,
        key=lambda item: (str(item["temporal_agg"]), str(item["stage"]), -float(item["score"])),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0].keys()))
        writer.writeheader()
        writer.writerows(ordered)


def _best_layers(rows: list[dict[str, float | int | str]], stage: str) -> dict[str, dict[str, float | int | str]]:
    best: dict[str, dict[str, float | int | str]] = {}
    for row in rows:
        if str(row["stage"]) != stage:
            continue
        mode = str(row["temporal_agg"])
        previous = best.get(mode)
        if previous is None or float(row["score"]) > float(previous["score"]):
            best[mode] = row
    return best


def _render_layer_slot_frames(
    *,
    frames: list[np.ndarray],
    output_dir: Path,
    maps: np.ndarray,
    temporal_agg: str,
    stage: str,
    layer_id: int,
    percentile: float,
) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    layer_dir = output_dir / f"{stage}_{temporal_agg}_layer{layer_id:02d}_p{int(percentile):02d}"
    frame_dir = layer_dir / "frames"
    layer_dir.mkdir(parents=True, exist_ok=True)
    for slot_id in range(int(maps.shape[0])):
        lowres = maps[slot_id].astype(np.float32)
        aligned = _temporal_resize_lowres(lowres, len(frames))
        threshold = float(np.percentile(aligned, percentile))
        active_ratio = float((aligned > threshold).mean())
        for frame_id, frame in enumerate(frames):
            overlay = _threshold_overlay(frame, aligned[frame_id], threshold, slot_id=slot_id)
            image = _label(
                overlay,
                [
                    f"Scheme A object cross-attn | {stage} | {temporal_agg} | layer {layer_id:02d} | slot {slot_id:02d}",
                    f"generated frame={frame_id:02d} | threshold=p{percentile:g} | active={active_ratio*100:.2f}%",
                ],
            )
            _write_jpeg(
                frame_dir / f"slot{slot_id:02d}_frame{frame_id:03d}.jpg",
                image,
            )
        images.append(
            {
                "slot": int(slot_id),
                "layer": int(layer_id),
                "temporal_agg": temporal_agg,
                "stage": stage,
                "percentile": float(percentile),
                "threshold": threshold,
                "active_ratio": active_ratio,
                "frame_pattern": f"{layer_dir.name}/frames/slot{slot_id:02d}_frame{{frame}}.jpg",
            }
        )
    return images


@torch.no_grad()
def _extract_slots_for_pca(
    *,
    model,
    result: dict[str, Any],
    output_frames: int,
    video_height: int,
    video_width: int,
) -> dict[str, Any]:
    source_video = result.get("source_video")
    if not isinstance(source_video, str) or not source_video.strip():
        return {"error": "result has no source_video"}

    reference = torch.empty(
        (3, int(output_frames), int(video_height), int(video_width)),
        device="cpu",
        dtype=torch.float32,
    )
    oracle_video_single, oracle_video_debug = scheme_a_batch.oracle_infer._load_oracle_video_single(
        model=model,
        video_path=source_video,
        reference_video_single=reference,
    )
    oracle_video = oracle_video_single.unsqueeze(0).to(
        device=model.pipe.device,
        dtype=model.pipe.torch_dtype,
    )
    preprocess_mode = str(result.get("object_debug", {}).get("xssc_preprocess", {}).get("mode", "center_crop"))
    xssc_video, preprocess_debug = scheme_a_batch.oracle_infer._preprocess_xssc_with_mode(
        model,
        oracle_video,
        mode=preprocess_mode,
    )
    slots = model._extract_xssc_slots(xssc_video)
    perturb_debug = None
    object_debug = result.get("object_debug", {})
    perturb = object_debug.get("xssc_slot_perturbation") if isinstance(object_debug, dict) else None
    if isinstance(perturb, dict):
        perturb_debug = perturb
        slots, _ = scheme_a_batch.oracle_infer._apply_slot_perturbation(
            slots,
            mode=str(perturb.get("mode", "none")),
            seed=perturb.get("seed"),
            noise_std=float(perturb.get("noise_std", 1.0)),
            drop_prob=float(perturb.get("drop_prob", 0.5)),
        )
    latent_mode = str(object_debug.get("xssc_latent_slot_mode", {}).get("mode", "mean_latent_align")) if isinstance(object_debug, dict) else "mean_latent_align"
    latent_slots, latent_slot_debug = scheme_a_batch.oracle_infer._make_latent_slots(
        slots,
        stride=int(getattr(model, "xssc_vae_temporal_stride", 4)),
        mode=latent_mode,
    )
    return {
        "raw_slots": slots.detach().float().cpu().numpy()[0],
        "latent_slots": latent_slots.detach().float().cpu().numpy()[0],
        "oracle_video": oracle_video_debug,
        "xssc_preprocess": preprocess_debug,
        "xssc_slot_perturbation": perturb_debug,
        "xssc_latent_slot_mode": latent_slot_debug,
    }


def _pca_2d(slots: np.ndarray) -> tuple[np.ndarray, list[float]]:
    time_steps, slot_count, dim = slots.shape
    x = slots.reshape(time_steps * slot_count, dim).astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    coords = (x @ vt[:2].T).reshape(time_steps, slot_count, 2)
    denom = float(np.sum(singular_values**2))
    explained = [] if denom <= 0 else [float(v) for v in (singular_values[:2] ** 2 / denom)]
    return coords.astype(np.float32), explained


def _set_equal_xy_limits(ax, coords: np.ndarray, *, margin: float = 0.08) -> None:
    x_min, y_min = coords.reshape(-1, 2).min(axis=0)
    x_max, y_max = coords.reshape(-1, 2).max(axis=0)
    span = max(float(x_max - x_min), float(y_max - y_min), 1.0e-6)
    pad = span * float(margin)
    x_center = float(x_min + x_max) * 0.5
    y_center = float(y_min + y_max) * 0.5
    half = span * 0.5 + pad
    ax.set_xlim(x_center - half, x_center + half)
    ax.set_ylim(y_center - half, y_center + half)


def _write_pca_trajectory(
    *,
    coords: np.ndarray,
    explained: list[float],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time_steps, slot_count, _ = coords.shape
    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=150)
    colors = plt.cm.tab10(np.linspace(0, 1, slot_count))
    for slot_id in range(slot_count):
        xy = coords[:, slot_id]
        ax.plot(xy[:, 0], xy[:, 1], "-o", color=colors[slot_id], markersize=3.5, label=f"slot{slot_id:02d}")
        for time_id, point in enumerate(xy):
            if time_steps <= 16 or time_id in {0, time_steps - 1}:
                ax.text(float(point[0]), float(point[1]), str(time_id), fontsize=6, color=colors[slot_id])
    _set_equal_xy_limits(ax, coords)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)" if explained else "PC1")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)" if len(explained) > 1 else "PC2")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_pca_frame_images(
    *,
    coords: np.ndarray,
    output_dir: Path,
    output_frames: int,
    title: str,
    time_label: str,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    time_steps, slot_count, _ = coords.shape
    colors = plt.cm.tab10(np.linspace(0, 1, slot_count))
    paths: list[str] = []
    for frame_id in range(int(output_frames)):
        if output_frames <= 1 or time_steps <= 1:
            time_id = 0
        else:
            time_id = int(round(frame_id * float(time_steps - 1) / float(output_frames - 1)))
        fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=130)
        for slot_id in range(slot_count):
            xy = coords[:, slot_id]
            ax.plot(xy[:, 0], xy[:, 1], "-", color=colors[slot_id], alpha=0.28, linewidth=1.0)
            ax.scatter(
                [float(xy[time_id, 0])],
                [float(xy[time_id, 1])],
                color=[colors[slot_id]],
                s=48,
                edgecolors="white",
                linewidths=0.7,
                label=f"s{slot_id}",
            )
            ax.text(float(xy[time_id, 0]), float(xy[time_id, 1]), f"s{slot_id}", fontsize=7)
        _set_equal_xy_limits(ax, coords)
        ax.set_title(f"{title} | frame {frame_id:02d} -> {time_label} {time_id:02d}", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.22)
        ax.legend(ncol=4, fontsize=6, loc="upper right")
        fig.tight_layout()
        name = f"frame{frame_id:03d}.jpg"
        fig.savefig(output_dir / name)
        plt.close(fig)
        paths.append(f"{output_dir.name}/{name}")
    return paths


def _write_slot_pca_assets(
    *,
    model,
    result: dict[str, Any],
    output_dir: Path,
    output_frames: int,
    video_height: int,
    video_width: int,
) -> dict[str, Any]:
    pca_root = output_dir / "slot_pca"
    pca_root.mkdir(parents=True, exist_ok=True)
    extracted = _extract_slots_for_pca(
        model=model,
        result=result,
        output_frames=output_frames,
        video_height=video_height,
        video_width=video_width,
    )
    if "error" in extracted:
        return extracted
    raw_slots = extracted.pop("raw_slots")
    latent_slots = extracted.pop("latent_slots")
    np.savez_compressed(
        pca_root / "scheme_a_xssc_slots_for_pca_fp16.npz",
        raw_slots=raw_slots.astype(np.float16),
        latent_slots=latent_slots.astype(np.float16),
    )
    raw_coords, raw_explained = _pca_2d(raw_slots)
    latent_coords, latent_explained = _pca_2d(latent_slots)
    _write_pca_trajectory(
        coords=raw_coords,
        explained=raw_explained,
        output_path=pca_root / "raw_slot_pca_trajectory.png",
        title="Raw frozen xSSC slots PCA",
    )
    _write_pca_trajectory(
        coords=latent_coords,
        explained=latent_explained,
        output_path=pca_root / "latent_slot_pca_trajectory.png",
        title="Latent-aligned xSSC slots PCA",
    )
    _write_pca_frame_images(
        coords=raw_coords,
        output_dir=pca_root / "raw_frame_pca",
        output_frames=output_frames,
        title="Raw slot PCA",
        time_label="xSSC frame",
    )
    _write_pca_frame_images(
        coords=latent_coords,
        output_dir=pca_root / "latent_frame_pca",
        output_frames=output_frames,
        title="Latent slot PCA",
        time_label="latent time",
    )
    return {
        "raw_slots_shape": list(raw_slots.shape),
        "latent_slots_shape": list(latent_slots.shape),
        "raw_explained_variance_ratio": raw_explained,
        "latent_explained_variance_ratio": latent_explained,
        "slots_npz": "slot_pca/scheme_a_xssc_slots_for_pca_fp16.npz",
        "raw_trajectory": "slot_pca/raw_slot_pca_trajectory.png",
        "latent_trajectory": "slot_pca/latent_slot_pca_trajectory.png",
        "raw_frame_pattern": "slot_pca/raw_frame_pca/frame{frame}.jpg",
        "latent_frame_pattern": "slot_pca/latent_frame_pca/frame{frame}.jpg",
        **extracted,
    }


def _write_slot_overlays(
    *,
    model,
    output_dir: Path,
    output_video: Path,
    recorder: SchemeAObjectCrossAttentionRecorder,
    result: dict[str, Any],
    fps: int,
) -> dict[str, Any]:
    frames, measured_fps = _read_video_bgr(output_video)
    fps_out = int(fps) if int(fps) > 0 else int(round(measured_fps))
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_frame_dir = output_dir / "generated_frames"
    for frame_id, frame in enumerate(frames):
        _write_jpeg(
            generated_frame_dir / f"frame{frame_id:03d}.jpg",
            _label(frame, [f"generated video | frame={frame_id:02d}"]),
            quality=88,
        )
    pca_summary = _write_slot_pca_assets(
        model=model,
        result=result,
        output_dir=output_dir,
        output_frames=len(frames),
        video_height=int(frames[0].shape[0]),
        video_width=int(frames[0].shape[1]),
    )
    layer_maps = recorder.averaged_by_layer()
    if not layer_maps:
        raise RuntimeError("no object cross-attention maps were captured")

    layer_rows = _layer_saliency_rows(layer_maps)
    ranking_csv = output_dir / "layer_saliency_ranking.csv"
    _write_layer_ranking_csv(ranking_csv, layer_rows)
    best = _best_layers(layer_rows, OBJECT_ATTN_BEST_STAGE)
    rendered: dict[str, list[dict[str, Any]]] = {}
    page_sections: list[str] = []
    raw_maps: dict[str, np.ndarray] = {}

    for mode, stages in layer_maps.items():
        if OBJECT_ATTN_BEST_STAGE not in stages:
            continue
        best_layer = None if mode not in best else int(best[mode]["layer"])
        selected_layers = list(dict.fromkeys(([best_layer] if best_layer is not None else []) + OBJECT_ATTN_SELECTED_LAYERS))
        mode_sections: list[str] = []
        for layer_id in selected_layers:
            if layer_id is None or layer_id < 0 or layer_id >= int(stages[OBJECT_ATTN_BEST_STAGE].shape[0]):
                continue
            maps = stages[OBJECT_ATTN_BEST_STAGE][layer_id]
            for slot_id in range(int(maps.shape[0])):
                raw_maps[
                    f"{OBJECT_ATTN_BEST_STAGE}_{mode}_layer{layer_id:02d}_slot{slot_id:02d}"
                ] = maps[slot_id].astype(np.float16)
            images = _render_layer_slot_frames(
                frames=frames,
                output_dir=output_dir,
                maps=maps,
                temporal_agg=mode,
                stage=OBJECT_ATTN_BEST_STAGE,
                layer_id=layer_id,
                percentile=OBJECT_ATTN_PERCENTILE,
            )
            key = f"{OBJECT_ATTN_BEST_STAGE}_{mode}_layer{layer_id:02d}"
            rendered[key] = images
            best_label = " auto-best" if best_layer is not None and int(layer_id) == int(best_layer) else ""
            figures = "".join(
                "<figure>"
                f"<img class='frame-sync' data-pattern='{html.escape(str(item['frame_pattern']))}' "
                f"src='{html.escape(str(item['frame_pattern'])).replace('{frame}', '000')}' "
                f"alt='slot {int(item['slot']):02d} layer {layer_id:02d}'>"
                f"<figcaption>slot {int(item['slot']):02d}; active={float(item['active_ratio'])*100:.2f}%</figcaption>"
                "</figure>"
                for item in images
            )
            mode_sections.append(
                "<section>"
                f"<h3>{html.escape(mode)} | layer {layer_id:02d}{best_label}</h3>"
                f"<div class='grid'>{figures}</div>"
                "</section>"
            )
        page_sections.append(
            "<section class='mode-section'>"
            f"<h2>Temporal aggregation: {html.escape(mode)}</h2>"
            f"{''.join(mode_sections)}</section>"
        )

    npz_path = output_dir / "scheme_a_object_cross_attention_selected_maps_fp16.npz"
    np.savez_compressed(npz_path, **raw_maps)
    summary = {
        "case": Path(str(result.get("input_json", "case"))).stem,
        "input_json": result.get("input_json"),
        "source_video": result.get("source_video"),
        "generated_video": str(output_video),
        "latent_grid": None if recorder.grid is None else list(recorder.grid),
        "output_frames": len(frames),
        "video_height": int(frames[0].shape[0]),
        "video_width": int(frames[0].shape[1]),
        "slot_count": int(recorder.slot_count),
        "key_count": recorder.key_count,
        "key_time_steps": recorder.key_time_steps,
        "generated_frame_pattern": "generated_frames/frame{frame}.jpg",
        "temporal_aggs": recorder.temporal_aggs,
        "selected_layers": OBJECT_ATTN_SELECTED_LAYERS,
        "best_stage": OBJECT_ATTN_BEST_STAGE,
        "best_layers": best,
        "slot_pca": pca_summary,
        "percentile": OBJECT_ATTN_PERCENTILE,
        "total_steps": int(recorder.total_steps),
        "captured_layer_count": int(recorder.layer_count),
        "layer_saliency_ranking_csv": ranking_csv.name,
        "selected_maps_npz": npz_path.name,
        "rendered": rendered,
        "note": (
            "Maps are Wan DiT video-query attention to Scheme A xSSC object K/V tokens. "
            "aligned uses the slot token at the corresponding Wan latent time; sum/max pool "
            "all latent-time tokens for the same slot. Overlays show regions above the "
            "requested percentile threshold."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pca_html = ""
    if "error" in pca_summary:
        pca_html = f"<p>Slot PCA unavailable: {html.escape(str(pca_summary['error']))}</p>"
    else:
        pca_html = f"""
<section class='pca-section'>
<h2>Slot PCA</h2>
<p>Raw slots: {html.escape(str(pca_summary.get('raw_slots_shape')))}; latent-aligned slots:
{html.escape(str(pca_summary.get('latent_slots_shape')))}. The slider maps generated frame f to raw xSSC frame f and to nearest latent time.</p>
<div class='pca-grid'>
<figure><img src='{html.escape(str(pca_summary["raw_trajectory"]))}'><figcaption>raw frozen xSSC slot PCA trajectory</figcaption></figure>
<figure><img src='{html.escape(str(pca_summary["latent_trajectory"]))}'><figcaption>latent-aligned slot PCA trajectory</figcaption></figure>
<figure><img class='frame-sync' data-pattern='{html.escape(str(pca_summary["raw_frame_pattern"]))}' src='slot_pca/raw_frame_pca/frame000.jpg'><figcaption>raw PCA current frame</figcaption></figure>
<figure><img class='frame-sync' data-pattern='{html.escape(str(pca_summary["latent_frame_pattern"]))}' src='slot_pca/latent_frame_pca/frame000.jpg'><figcaption>latent PCA current frame</figcaption></figure>
</div>
</section>"""

    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>Scheme A object cross-attention percentile masks</title>
<style>
body{{margin:0;background:#101216;color:#edf1f7;font:14px Arial,sans-serif}}
main{{max-width:1900px;margin:auto;padding:22px}}
h1,h2,h3{{letter-spacing:0;margin:12px 0}}
section{{border-top:1px solid #2a3038;padding-top:16px;margin-top:18px}}
.toolbar{{position:sticky;top:0;z-index:10;background:#101216e8;backdrop-filter:blur(10px);border-bottom:1px solid #2a3038;padding:12px 0;margin-bottom:12px}}
.slider-row{{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center}}
input[type=range]{{width:100%}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(260px,1fr));gap:14px;align-items:start}}
.pca-grid{{display:grid;grid-template-columns:repeat(4,minmax(260px,1fr));gap:14px;align-items:start}}
figure{{margin:0;background:#171b22;border:1px solid #2a3038;border-radius:4px;padding:8px;min-width:0;overflow:hidden}}
img{{width:100%;max-width:100%;height:auto;background:#000;display:block}}
figcaption{{padding-top:6px;color:#c4ccd8}}
a{{color:#9cc8ff}}
code{{color:#dce7ff}}
@media(max-width:1100px){{.grid{{grid-template-columns:repeat(2,minmax(220px,1fr))}}}}
@media(max-width:1100px){{.pca-grid{{grid-template-columns:repeat(2,minmax(220px,1fr))}}}}
@media(max-width:650px){{.grid{{grid-template-columns:1fr}}}}
@media(max-width:650px){{.pca-grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Scheme A Slot K/V Object Cross-Attention Percentile Masks</h1>
<p>Overlay target: final generated video. Latent grid: {html.escape(str(summary['latent_grid']))};
object keys: {summary['key_count']} = {summary['key_time_steps']} latent steps x {summary['slot_count']} slots;
threshold: p{OBJECT_ATTN_PERCENTILE:g}; selected layers: {html.escape(str(OBJECT_ATTN_SELECTED_LAYERS))};
best stage: {html.escape(OBJECT_ATTN_BEST_STAGE)}.</p>
<p><a href='../{html.escape(Path(str(output_video)).name)}'>generated video</a> |
<a href='summary.json'>summary JSON</a> | <a href='{html.escape(ranking_csv.name)}'>layer ranking CSV</a> |
<a href='{html.escape(npz_path.name)}'>selected fp16 maps</a></p>
<div class='toolbar'>
<div class='slider-row'>
<label for='frameSlider'>Frame</label>
<input id='frameSlider' type='range' min='0' max='{len(frames)-1}' value='0' step='1'>
<strong id='frameLabel'>0 / {len(frames)-1}</strong>
</div>
</div>
<section>
<h2>Generated Frame</h2>
<figure><img class='frame-sync' data-pattern='generated_frames/frame{{frame}}.jpg' src='generated_frames/frame000.jpg'><figcaption>current generated frame</figcaption></figure>
</section>
{pca_html}
{''.join(page_sections)}
<script>
const slider = document.getElementById('frameSlider');
const label = document.getElementById('frameLabel');
const images = Array.from(document.querySelectorAll('img.frame-sync'));
function padFrame(value) {{
  return String(value).padStart(3, '0');
}}
function setFrame(value) {{
  const frame = Math.max(0, Math.min({len(frames)-1}, Number(value)));
  const padded = padFrame(frame);
  label.textContent = `${{frame}} / {len(frames)-1}`;
  for (const image of images) {{
    const pattern = image.dataset.pattern;
    if (pattern) image.src = pattern.replace('{{frame}}', padded);
  }}
}}
slider.addEventListener('input', event => setFrame(event.target.value));
document.addEventListener('keydown', event => {{
  if (event.key === 'ArrowLeft') {{
    slider.value = Math.max(0, Number(slider.value) - 1);
    setFrame(slider.value);
  }} else if (event.key === 'ArrowRight') {{
    slider.value = Math.min({len(frames)-1}, Number(slider.value) + 1);
    setFrame(slider.value);
  }}
}});
setFrame(0);
</script>
</main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    return summary


_ORIGINAL_RUN_SINGLE_CASE = scheme_a_batch._run_single_case_in_process


def _run_single_case_with_scheme_a_object_attention(*args, **kwargs):
    model = kwargs["model"]
    sampling_steps = int(kwargs.get("sampling_steps", 40))
    cfg_scale = float(kwargs.get("cfg_scale", 5.0))
    output_video = Path(kwargs["output_video"])
    recorder = SchemeAObjectCrossAttentionRecorder(
        total_steps=sampling_steps,
        slot_count=OBJECT_ATTN_SLOT_COUNT,
        temporal_aggs=OBJECT_ATTN_TEMPORAL_AGGS,
        query_chunk=OBJECT_ATTN_QUERY_CHUNK,
    )
    model_fn_scope = ModelFnObjectAttentionScope(
        pipe=model.pipe,
        attention=recorder,
        cfg_scale=cfg_scale,
    )
    recorder.install(model.pipe.dit)
    model_fn_scope.install()
    try:
        result, logs = _ORIGINAL_RUN_SINGLE_CASE(*args, **kwargs)
    finally:
        model_fn_scope.restore()
        recorder.restore()

    attention_dir = output_video.with_name(f"{output_video.stem}_scheme_a_object_cross_attention_percentile")
    summary = _write_slot_overlays(
        model=model,
        output_dir=attention_dir,
        output_video=output_video,
        recorder=recorder,
        result=result,
        fps=int(kwargs.get("fps") or 8),
    )
    result["scheme_a_object_cross_attention_percentile"] = {
        "output_dir": str(attention_dir),
        "index": str(attention_dir / "index.html"),
        "summary": str(attention_dir / "summary.json"),
        "latent_grid": summary.get("latent_grid"),
        "key_time_steps": summary.get("key_time_steps"),
        "temporal_aggs": summary.get("temporal_aggs"),
        "percentile": summary.get("percentile"),
    }
    logs.append(f"[scheme-a-object-attn] {attention_dir / 'index.html'}")
    return result, logs


def main() -> None:
    scheme_a_batch._run_single_case_in_process = _run_single_case_with_scheme_a_object_attention
    scheme_a_batch.main()


if __name__ == "__main__":
    main()
