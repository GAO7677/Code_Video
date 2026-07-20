#!/usr/bin/env python3
"""Visualize Wan DiT object cross-attention for xSSC slot K/V tokens.

The xSSC object context is ordered as [ctx_time, slot]. During inference this
script records the DiT video-query attention to those tokens, pools the ctx
tokens per slot, and overlays the resulting maps on the final generated video.
"""
from __future__ import annotations

import html
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

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


OBJECT_ATTN_QUERY_CHUNK = int(
    _pop_option(sys.argv, "--xssc-object-attn-query-chunk", "2048")
)
OBJECT_ATTN_SLOT_COUNT = int(_pop_option(sys.argv, "--xssc-object-attn-slots", "7"))
OBJECT_ATTN_CTX_STEPS = int(
    _pop_option(sys.argv, "--xssc-object-attn-ctx-steps", "8")
)
OBJECT_ATTN_CTX_POOL = _pop_option(
    sys.argv, "--xssc-object-attn-ctx-pool", "mean"
).strip().lower()
if OBJECT_ATTN_CTX_POOL not in {"mean", "sum"}:
    raise SystemExit("--xssc-object-attn-ctx-pool must be mean or sum")

from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as batch_base,
)
from code_vjepa_vggt.train_xSSC import infer_xssc_context_slots as xssc_infer

FFMPEG = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg")


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


def _heat_overlay(frame: np.ndarray, heat_lowres: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = np.clip((heat_lowres - low) / max(high - low, 1.0e-12), 0.0, 1.0)
    heat = cv2.resize(
        (normalized * 255.0).astype(np.uint8),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    return cv2.addWeighted(frame, 0.56, color, 0.44, 0.0)


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
        if frame.shape[:2] != (height, width):
            raise ValueError(f"inconsistent frame size in {path}: {frame.shape[:2]}")
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


class XSSCObjectCrossAttentionRecorder:
    def __init__(
        self,
        *,
        total_steps: int,
        slot_count: int,
        ctx_steps: int,
        ctx_pool: str,
        query_chunk: int,
    ) -> None:
        self.total_steps = int(total_steps)
        self.slot_count = int(slot_count)
        self.ctx_steps = int(ctx_steps)
        self.ctx_pool = str(ctx_pool)
        self.query_chunk = int(query_chunk)
        self.active = False
        self.step_index = -1
        self.grid: tuple[int, int, int] | None = None
        self.layer_count = 0
        self._original_forwards: list[tuple[Any, Any]] = []
        self.sums: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}
        self.layer_calls: dict[str, int] = {}
        self.layer_sums: dict[str, np.ndarray] = {}
        self.layer_counts: dict[str, np.ndarray] = {}
        self.key_count: int | None = None
        self.num_heads_by_layer: dict[int, int] = {}

    def install(self, dit) -> None:
        self.layer_count = len(dit.blocks)
        for layer_id, block in enumerate(dit.blocks):
            cross_attn = getattr(block, "object_cross_attn", None)
            attn = getattr(cross_attn, "attn", None)
            if attn is None:
                raise RuntimeError(
                    f"Wan block {layer_id} has no object cross-attention module"
                )
            original = attn.forward
            num_heads = int(getattr(attn, "num_heads", 24))
            self.num_heads_by_layer[int(layer_id)] = int(num_heads)

            def wrapped(
                q,
                k,
                v,
                *,
                _original=original,
                _layer_id=layer_id,
                _num_heads=num_heads,
            ):
                output = _original(q, k, v)
                if self.active:
                    self.capture(
                        layer_id=int(_layer_id),
                        q=q,
                        k=k,
                        num_heads=int(_num_heads),
                    )
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

    def _ensure_accumulator(self, stage: str) -> np.ndarray:
        if self.grid is None:
            raise RuntimeError("attention grid is missing")
        frames, grid_h, grid_w = self.grid
        if stage not in self.sums:
            self.sums[stage] = np.zeros(
                (self.slot_count, frames, grid_h, grid_w), dtype=np.float64
            )
            self.counts[stage] = 0
            self.layer_calls[stage] = 0
        return self.sums[stage]

    def _ensure_layer_accumulator(self, stage: str) -> np.ndarray:
        if self.grid is None:
            raise RuntimeError("attention grid is missing")
        frames, grid_h, grid_w = self.grid
        if stage not in self.layer_sums:
            self.layer_sums[stage] = np.zeros(
                (
                    self.layer_count,
                    self.slot_count,
                    frames,
                    grid_h,
                    grid_w,
                ),
                dtype=np.float64,
            )
            self.layer_counts[stage] = np.zeros((self.layer_count,), dtype=np.int64)
        return self.layer_sums[stage]

    @torch.no_grad()
    def capture(self, *, layer_id: int, q: torch.Tensor, k: torch.Tensor, num_heads: int) -> None:
        if self.grid is None:
            raise RuntimeError("attention grid was not configured before capture")
        frames, grid_h, grid_w = self.grid
        expected_queries = int(frames * grid_h * grid_w)
        if int(q.shape[0]) != 1 or int(q.shape[1]) != expected_queries:
            raise RuntimeError(
                f"object attention q={list(q.shape)} does not match grid={self.grid}"
            )
        key_count = int(k.shape[1])
        expected_keys = int(self.ctx_steps * self.slot_count)
        if key_count != expected_keys:
            raise RuntimeError(
                f"expected {expected_keys} xSSC object keys "
                f"({self.ctx_steps} ctx x {self.slot_count} slots), got {key_count}"
            )
        if int(q.shape[-1]) % int(num_heads) != 0:
            raise RuntimeError(
                f"object attention dim={q.shape[-1]} is not divisible by heads={num_heads}"
            )
        self.key_count = key_count
        head_dim = int(q.shape[-1]) // int(num_heads)
        keys = (
            k.view(1, key_count, int(num_heads), head_dim)
            .permute(0, 2, 1, 3)
            .float()
        )
        key_t = keys.transpose(-1, -2)
        chunks: list[torch.Tensor] = []
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
            grouped = probs.view(
                1,
                int(num_heads),
                stop - start,
                self.ctx_steps,
                self.slot_count,
            )
            if self.ctx_pool == "sum":
                slot_probs = grouped.sum(dim=3)
            else:
                slot_probs = grouped.mean(dim=3)
            chunks.append(slot_probs.mean(dim=(0, 1)).cpu())
            del scores, probs, grouped, slot_probs, queries
        flat = torch.cat(chunks, dim=0).transpose(0, 1)
        array = flat.reshape(self.slot_count, frames, grid_h, grid_w).numpy()
        for stage in self._stage_names(int(self.step_index)):
            self._ensure_accumulator(stage)[:] += array
            self.counts[stage] += 1
            self.layer_calls[stage] += 1
            self._ensure_layer_accumulator(stage)[int(layer_id)] += array
            self.layer_counts[stage][int(layer_id)] += 1
        del flat, array

    def averaged(self) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for stage, value in self.sums.items():
            count = max(1, int(self.counts.get(stage, 0)))
            output[stage] = (value / float(count)).astype(np.float32)
        return output

    def averaged_by_layer(self) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for stage, value in self.layer_sums.items():
            counts = np.maximum(self.layer_counts[stage].astype(np.float64), 1.0)
            output[stage] = (value / counts[:, None, None, None, None]).astype(np.float32)
        return output


class ModelFnObjectAttentionScope:
    def __init__(
        self,
        *,
        pipe,
        attention: XSSCObjectCrossAttentionRecorder,
        cfg_scale: float,
    ) -> None:
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


def _layer_saliency_rows(layer_maps: dict[str, np.ndarray]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for stage, maps in layer_maps.items():
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


def _best_layers_by_stage(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float | int | str]]:
    best: dict[str, dict[str, float | int | str]] = {}
    for row in rows:
        stage = str(row["stage"])
        previous = best.get(stage)
        if previous is None or float(row["score"]) > float(previous["score"]):
            best[stage] = row
    return best


def _write_layer_ranking_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    import csv

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda item: (str(item["stage"]), -float(item["score"])))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0].keys()))
        writer.writeheader()
        writer.writerows(ordered)


def _render_stage_slot_videos(
    *,
    frames: list[np.ndarray],
    fps: int,
    stage_dir: Path,
    maps: np.ndarray,
    stage_label: str,
    recorder: XSSCObjectCrossAttentionRecorder,
    layer_id: int | None = None,
) -> list[dict[str, str | int]]:
    stage_videos: list[dict[str, str | int]] = []
    stage_dir.mkdir(parents=True, exist_ok=True)
    for slot_id in range(int(maps.shape[0])):
        lowres = maps[slot_id].astype(np.float32)
        aligned = _temporal_resize_lowres(lowres, len(frames))
        low, high = np.percentile(aligned, [2.0, 99.0]).tolist()
        rendered: list[np.ndarray] = []
        for frame_id, frame in enumerate(frames):
            overlay = _heat_overlay(frame, aligned[frame_id], float(low), float(high))
            layer_text = "layer-mean" if layer_id is None else f"layer {layer_id:02d}"
            rendered.append(
                _label(
                    overlay,
                    [
                        f"xSSC object cross-attn | {stage_label} | {layer_text} | slot {slot_id:02d}",
                        f"generated frame={frame_id:02d} | ctx_pool={recorder.ctx_pool}",
                    ],
                )
            )
        layer_part = "mean" if layer_id is None else f"layer{int(layer_id):02d}"
        video_name = f"slot{slot_id:02d}_{stage_label}_{layer_part}_object_cross_attention.mp4"
        video_path = stage_dir / video_name
        _write_h264(video_path, rendered, fps=fps)
        stage_videos.append(
            {
                "slot": int(slot_id),
                "layer": -1 if layer_id is None else int(layer_id),
                "video": f"{stage_dir.name}/{video_name}",
            }
        )
    return stage_videos


def _write_slot_overlays(
    *,
    output_dir: Path,
    output_video: Path,
    recorder: XSSCObjectCrossAttentionRecorder,
    result: dict[str, Any],
    fps: int,
) -> dict[str, Any]:
    frames, measured_fps = _read_video_bgr(output_video)
    fps_out = int(fps) if int(fps) > 0 else int(round(measured_fps))
    output_dir.mkdir(parents=True, exist_ok=True)
    averaged = recorder.averaged()
    layer_maps = recorder.averaged_by_layer()
    if not averaged:
        raise RuntimeError("no object cross-attention maps were captured")

    raw_maps: dict[str, np.ndarray] = {}
    layer_raw_maps: dict[str, np.ndarray] = {}
    stage_cards: list[str] = []
    best_layer_cards: list[str] = []
    videos_by_stage: dict[str, list[dict[str, str | int]]] = {}
    best_layer_videos_by_stage: dict[str, list[dict[str, str | int]]] = {}
    layer_rows = _layer_saliency_rows(layer_maps)
    best_layers = _best_layers_by_stage(layer_rows)
    ranking_csv = output_dir / "layer_saliency_ranking.csv"
    _write_layer_ranking_csv(ranking_csv, layer_rows)

    for stage in ("early", "middle", "late", "all"):
        if stage not in averaged:
            continue
        maps = averaged[stage]
        for slot_id in range(int(maps.shape[0])):
            raw_maps[f"{stage}_slot{slot_id:02d}"] = maps[slot_id].astype(np.float16)
        if stage in layer_maps:
            for layer_id in range(int(layer_maps[stage].shape[0])):
                for slot_id in range(int(layer_maps[stage].shape[1])):
                    layer_raw_maps[
                        f"{stage}_layer{layer_id:02d}_slot{slot_id:02d}"
                    ] = layer_maps[stage][layer_id, slot_id].astype(np.float16)

        best_row = best_layers.get(stage)
        if best_row is not None:
            best_layer_id = int(best_row["layer"])
            best_stage_dir = output_dir / f"{stage}_best_layer{best_layer_id:02d}"
            best_stage_videos = _render_stage_slot_videos(
                frames=frames,
                fps=fps_out,
                stage_dir=best_stage_dir,
                maps=layer_maps[stage][best_layer_id],
                stage_label=f"{stage}_best",
                recorder=recorder,
                layer_id=best_layer_id,
            )
            best_layer_videos_by_stage[stage] = best_stage_videos
            best_figures = "".join(
                "<figure>"
                f"<video controls muted loop src='{html.escape(str(item['video']))}'></video>"
                f"<figcaption>slot {int(item['slot']):02d}</figcaption>"
                "</figure>"
                for item in best_stage_videos
            )
            best_layer_cards.append(
                "<section>"
                f"<h2>{html.escape(stage)} best layer {best_layer_id:02d}</h2>"
                f"<p>saliency score={float(best_row['score']):.6f}; "
                f"CV={float(best_row['coefficient_of_variation']):.6f}; "
                f"peak/mean={float(best_row['peak_to_mean']):.4f}; "
                f"cross-slot centered mean={float(best_row['cross_slot_centered_mean']):.4f}</p>"
                f"<div class='grid'>{best_figures}</div></section>"
            )

        stage_dir = output_dir / stage
        stage_videos = _render_stage_slot_videos(
            frames=frames,
            fps=fps_out,
            stage_dir=stage_dir,
            maps=maps,
            stage_label=stage,
            recorder=recorder,
            layer_id=None,
        )
        videos_by_stage[stage] = stage_videos
        figures = "".join(
            "<figure>"
            f"<video controls muted loop src='{html.escape(str(item['video']))}'></video>"
            f"<figcaption>slot {int(item['slot']):02d}</figcaption>"
            "</figure>"
            for item in stage_videos
        )
        stage_cards.append(f"<section><h2>{html.escape(stage)}</h2><div class='grid'>{figures}</div></section>")

    npz_path = output_dir / "xssc_object_cross_attention_maps_fp16.npz"
    np.savez_compressed(npz_path, **raw_maps)
    layer_npz_path = output_dir / "xssc_object_cross_attention_layer_maps_fp16.npz"
    np.savez_compressed(layer_npz_path, **layer_raw_maps)
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
        "ctx_steps_per_slot": int(recorder.ctx_steps),
        "ctx_pool": recorder.ctx_pool,
        "key_count": recorder.key_count,
        "total_steps": int(recorder.total_steps),
        "captured_layer_count": int(recorder.layer_count),
        "capture_counts": recorder.counts,
        "layer_calls": recorder.layer_calls,
        "maps_npz": npz_path.name,
        "layer_maps_npz": layer_npz_path.name,
        "layer_saliency_ranking_csv": ranking_csv.name,
        "layer_saliency_metric": (
            "score = coefficient_of_variation * (1 - mean cross-slot centered cosine); "
            "higher means spatially sharper and more slot-separated attention maps"
        ),
        "best_layers_by_stage": best_layers,
        "videos_by_stage": videos_by_stage,
        "best_layer_videos_by_stage": best_layer_videos_by_stage,
        "note": (
            "Maps are Wan DiT video-query attention to xSSC object K/V tokens; "
            "the ctx tokens for each slot are pooled before visualization."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>xSSC object cross-attention</title>
<style>
body{{margin:0;background:#f4f2ee;color:#1e211d;font:14px Arial,sans-serif}}
main{{max-width:1800px;margin:auto;padding:22px}}
h1,h2{{letter-spacing:0;margin:12px 0}}
section{{border-top:1px solid #c9c5bb;padding-top:16px;margin-top:18px}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(260px,1fr));gap:14px;align-items:start}}
.grid>*{{min-width:0}}
figure{{margin:0;background:#fff;border:1px solid #d7d2c8;border-radius:4px;padding:8px;min-width:0;overflow:hidden}}
video,img{{width:100%;max-width:100%;height:auto;background:#000;display:block}}
figcaption{{padding-top:6px}}
@media(max-width:1100px){{.grid{{grid-template-columns:repeat(2,minmax(220px,1fr))}}}}
@media(max-width:650px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>xSSC Slot K/V Object Cross-Attention</h1>
<p>Overlay target: final generated video. Latent grid: {html.escape(str(summary['latent_grid']))};
object keys: {summary['key_count']} = {recorder.ctx_steps} ctx steps x {recorder.slot_count} slots;
ctx pooling: {html.escape(recorder.ctx_pool)}.</p>
<p><a href='../{html.escape(Path(str(output_video)).name)}'>generated video</a> |
<a href='summary.json'>summary JSON</a> | <a href='{html.escape(ranking_csv.name)}'>layer ranking CSV</a> |
<a href='{html.escape(layer_npz_path.name)}'>layer fp16 maps</a> | <a href='{html.escape(npz_path.name)}'>layer-mean fp16 maps</a></p>
<section><h2>Selected Layers</h2><p>The videos below use the automatically selected layer for each denoising stage. The old layer-mean videos are kept below only as a reference.</p></section>
{''.join(best_layer_cards)}
<section><h2>Layer-Mean Reference</h2><p>These videos average all object cross-attention layers and can blur layer-specific slot routing.</p></section>
{''.join(stage_cards)}
</main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    return summary


_ORIGINAL_RUN_SINGLE_CASE = batch_base._run_single_case_in_process


def _run_single_case_with_xssc_object_attention(*args, **kwargs):
    model = kwargs["model"]
    sampling_steps = int(kwargs.get("sampling_steps", 40))
    cfg_scale = float(kwargs.get("cfg_scale", 5.0))
    output_video = Path(kwargs["output_video"])
    recorder = XSSCObjectCrossAttentionRecorder(
        total_steps=sampling_steps,
        slot_count=OBJECT_ATTN_SLOT_COUNT,
        ctx_steps=OBJECT_ATTN_CTX_STEPS,
        ctx_pool=OBJECT_ATTN_CTX_POOL,
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

    attention_dir = output_video.with_name(
        f"{output_video.stem}_xssc_object_cross_attention"
    )
    summary = _write_slot_overlays(
        output_dir=attention_dir,
        output_video=output_video,
        recorder=recorder,
        result=result,
        fps=int(kwargs.get("fps", 8)),
    )
    result["xssc_object_cross_attention"] = {
        "output_dir": str(attention_dir),
        "index": str(attention_dir / "index.html"),
        "summary": str(attention_dir / "summary.json"),
        "latent_grid": summary.get("latent_grid"),
        "capture_counts": summary.get("capture_counts"),
    }
    logs.append(f"[xssc-object-attn] {attention_dir / 'index.html'}")
    return result, logs


def main() -> None:
    batch_base._install_kubric_runtime_hooks = xssc_infer._install_runtime_hooks
    batch_base._run_single_case_in_process = _run_single_case_with_xssc_object_attention
    batch_base.main()


if __name__ == "__main__":
    main()
