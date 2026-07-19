#!/usr/bin/env python3
"""Analyze context-to-future correspondence during Kubric Stage1b generation.

The generation path is imported from the original Kubric inference wrapper. This
file only installs transient hooks and does not modify the training repository.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/stage1b_kubric_generation_analysis"
)
for repo_root in (CODE_ROOT, DIFFSYNTH_ROOT):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from diffsynth.utils.data import save_video
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    infer_stage1b_context_only_no_gt_box_v_newtrain_kubric as kubric_infer,
)


base = kubric_infer.base
trainmod = kubric_infer.trainmod


def build_parser() -> argparse.ArgumentParser:
    original_parse_args = argparse.ArgumentParser.parse_args
    try:
        argparse.ArgumentParser.parse_args = lambda parser, *args, **kwargs: parser
        parser = base.parse_args()
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args

    parser.description = (
        "Reproduce Kubric Stage1b generation and measure post-RoPE self-attention "
        "Q/K and block-hidden correspondence from the last clean context latent "
        "to future latent frames."
    )
    parser.set_defaults(output_dir=str(DEFAULT_OUTPUT_ROOT))
    for action in parser._actions:
        if action.dest == "output_dir":
            action.required = False
    parser.add_argument(
        "--analysis-layers",
        type=int,
        nargs="+",
        default=[0, 5, 11, 17, 23, 29],
        help="Zero-based DiT layers to analyze.",
    )
    parser.add_argument(
        "--analysis-step-indices",
        type=int,
        nargs="+",
        default=None,
        help="Zero-based denoising steps. Default: five evenly spaced steps.",
    )
    parser.add_argument(
        "--analysis-query-grid",
        type=int,
        nargs=2,
        metavar=("ROWS", "COLS"),
        default=(3, 5),
        help="Uniform query grid used when --analysis-query-points-json is absent.",
    )
    parser.add_argument(
        "--analysis-query-points-json",
        type=Path,
        default=None,
        help="JSON file containing [[x, y], ...] in generated-video pixels.",
    )
    parser.add_argument(
        "--analysis-matching-mode",
        choices=("q_to_k", "symmetric"),
        default="q_to_k",
        help="Use the requested query-to-key affinity or average it with K-to-Q.",
    )
    parser.add_argument(
        "--analysis-no-hidden",
        action="store_true",
        help="Disable the block-hidden cosine-similarity baseline.",
    )
    parser.add_argument(
        "--analysis-hidden-temperature",
        type=float,
        default=0.07,
        help="Softmax temperature for normalized hidden-feature similarity.",
    )
    parser.add_argument(
        "--analysis-visualize-layer",
        type=int,
        default=None,
        help="Layer used for videos and heatmaps. Default: middle requested layer.",
    )
    parser.add_argument(
        "--analysis-visualize-step-index",
        type=int,
        default=None,
        help="Step used for videos and heatmaps. Default: last requested step.",
    )
    parser.add_argument(
        "--analysis-heatmap-query-index",
        type=int,
        default=0,
        help="Query point visualized in attention heatmaps.",
    )
    parser.add_argument(
        "--analysis-device",
        default=None,
        help="Override launch device, for example cuda:0.",
    )
    parser.add_argument(
        "--analysis-no-cotracker",
        action="store_true",
        help="Skip pseudo-GT tracking and only save predicted correspondences.",
    )
    parser.add_argument(
        "--analysis-no-video",
        action="store_true",
        help="Skip generated and trajectory MP4 files.",
    )
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evenly_spaced_steps(count: int) -> list[int]:
    if count <= 0:
        raise ValueError("sampling steps must be positive")
    return sorted({int(round(value)) for value in np.linspace(0, count - 1, min(5, count))})


def load_query_points(args: argparse.Namespace, height: int, width: int) -> np.ndarray:
    if args.analysis_query_points_json is not None:
        payload = json.loads(args.analysis_query_points_json.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("query_points", payload.get("points"))
        points = np.asarray(payload, dtype=np.float32)
    else:
        rows, cols = (int(value) for value in args.analysis_query_grid)
        if rows <= 0 or cols <= 0:
            raise ValueError("analysis query-grid dimensions must be positive")
        xs = np.linspace(0.12 * width, 0.88 * width, cols, dtype=np.float32)
        ys = np.linspace(0.12 * height, 0.88 * height, rows, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
        points = np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=-1)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError(f"query points must have shape [N, 2], got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("query points contain non-finite values")
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return points


def tensor_video_to_uint8(video: Any) -> np.ndarray:
    if isinstance(video, list):
        frames = [np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame) for frame in video]
        array = np.stack(frames)
    elif isinstance(video, torch.Tensor):
        array = video.detach().float().cpu().numpy()
    else:
        array = np.asarray(video)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 4:
        raise ValueError(f"unsupported generated video shape: {array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 3, 0))
    elif array.shape[1] in (1, 3) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (0, 2, 3, 1))
    if array.dtype != np.uint8:
        minimum = float(np.nanmin(array))
        if minimum < -0.01:
            array = (array + 1.0) * 127.5
        elif float(np.nanmax(array)) <= 1.01:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return array


def token_indices(
    points: torch.Tensor,
    grid_height: int,
    grid_width: int,
    pixel_height: int,
    pixel_width: int,
    device: torch.device,
) -> torch.Tensor:
    points = points.to(device=device, dtype=torch.float32)
    x = torch.floor(points[:, 0] * grid_width / pixel_width).long().clamp(0, grid_width - 1)
    y = torch.floor(points[:, 1] * grid_height / pixel_height).long().clamp(0, grid_height - 1)
    return y * grid_width + x


@dataclass
class MatchRecord:
    method: str
    layer: int
    step_index: int
    timestep: float
    sigma: float | None
    grid: tuple[int, int, int]
    clean_prefix_latents: int
    query_latent_index: int
    predictions: np.ndarray
    probabilities: np.ndarray


class GenerationCapture:
    def __init__(
        self,
        pipe,
        layers: list[int],
        step_indices: list[int],
        query_points: np.ndarray,
        pixel_hw: tuple[int, int],
        matching_mode: str,
        capture_hidden: bool,
        hidden_temperature: float,
    ) -> None:
        self.pipe = pipe
        self.layers = set(layers)
        self.step_indices = set(step_indices)
        self.query_points = torch.from_numpy(query_points).float()
        self.pixel_height, self.pixel_width = pixel_hw
        self.matching_mode = matching_mode
        self.capture_hidden = capture_hidden
        self.hidden_temperature = float(hidden_temperature)
        self.records: dict[tuple[str, int, int], MatchRecord] = {}
        self.call_counts: dict[int, int] = {}
        self.active = False
        self.current_step = -1
        self.current_timestep = float("nan")
        self.current_sigma: float | None = None
        self.current_grid: tuple[int, int, int] | None = None
        self.current_prefix = 0
        self._handles: list[Any] = []
        self._original_model_fn = None

    def _scheduler_step(self, timestep: torch.Tensor) -> int:
        value = float(timestep.detach().flatten()[0].float().cpu().item())
        timesteps = self.pipe.scheduler.timesteps.detach().float().cpu()
        return int(torch.argmin((timesteps - value).abs()).item())

    def _scheduler_sigma(self, step_index: int) -> float | None:
        sigmas = getattr(self.pipe.scheduler, "sigmas", None)
        if sigmas is None or step_index >= len(sigmas):
            return None
        value = sigmas[step_index]
        return float(value.detach().float().cpu().item() if isinstance(value, torch.Tensor) else value)

    def _wrap_model_fn(self, original):
        def wrapped_model_fn(*args, **kwargs):
            timestep = kwargs.get("timestep")
            latents = kwargs.get("latents")
            if timestep is None or latents is None:
                return original(*args, **kwargs)
            step_index = self._scheduler_step(timestep)
            call_index = self.call_counts.get(step_index, 0)
            self.call_counts[step_index] = call_index + 1
            patch_size = tuple(int(value) for value in kwargs["dit"].patch_size)
            self.current_grid = (
                int(latents.shape[2] // patch_size[0]),
                int(latents.shape[3] // patch_size[1]),
                int(latents.shape[4] // patch_size[2]),
            )
            clean_prefix = kwargs.get("clean_prefix_latents")
            self.current_prefix = (
                int(clean_prefix.shape[2])
                if clean_prefix is not None
                else int(kwargs.get("num_clean_prefix_latents") or 0)
            )
            self.current_step = step_index
            self.current_timestep = float(timestep.detach().flatten()[0].float().cpu().item())
            self.current_sigma = self._scheduler_sigma(step_index)
            self.active = call_index == 0 and step_index in self.step_indices
            try:
                return original(*args, **kwargs)
            finally:
                self.active = False

        return wrapped_model_fn

    def install(self) -> None:
        self._original_model_fn = self.pipe.model_fn
        self.pipe.model_fn = self._wrap_model_fn(self.pipe.model_fn)
        model_candidates = [self.pipe.dit]
        if getattr(self.pipe, "dit2", None) is not None and self.pipe.dit2 is not self.pipe.dit:
            model_candidates.append(self.pipe.dit2)
        for model in model_candidates:
            for layer in self.layers:
                if layer >= len(model.blocks):
                    raise ValueError(f"layer {layer} is outside model range [0, {len(model.blocks) - 1}]")
                attention_handle = model.blocks[layer].self_attn.attn.register_forward_pre_hook(
                    self._make_attention_hook(layer)
                )
                self._handles.append(attention_handle)
                if self.capture_hidden:
                    hidden_handle = model.blocks[layer].register_forward_hook(
                        self._make_hidden_hook(layer)
                    )
                    self._handles.append(hidden_handle)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._original_model_fn is not None:
            self.pipe.model_fn = self._original_model_fn
            self._original_model_fn = None

    def _make_attention_hook(self, layer: int):
        def hook(module, inputs):
            if not self.active:
                return
            q, k = inputs[:2]
            batch, sequence, channels = q.shape
            heads = int(module.num_heads)
            q = q.view(batch, sequence, heads, channels // heads)
            k = k.view(batch, sequence, heads, channels // heads)
            self._consume_qk(layer, q, k)

        return hook

    def _make_hidden_hook(self, layer: int):
        def hook(module, inputs, output):
            if self.active:
                self._consume_hidden(layer, output)

        return hook

    def _geometry(self, sequence: int) -> tuple[int, int, int, int]:
        if self.current_grid is None:
            raise RuntimeError("capture hook fired without a current latent grid")
        time, height, width = self.current_grid
        expected = time * height * width
        if expected != sequence:
            raise RuntimeError(f"token geometry mismatch: sequence={sequence}, grid={self.current_grid}")
        if not 0 < self.current_prefix < time:
            raise RuntimeError(
                f"clean-prefix boundary must leave future tokens: prefix={self.current_prefix}, time={time}"
            )
        return time, height, width, self.current_prefix - 1

    def _consume_qk(self, layer: int, q: torch.Tensor, k: torch.Tensor) -> None:
        key = ("qk", layer, self.current_step)
        if key in self.records:
            return
        if q.shape[0] != 1 or q.shape != k.shape:
            raise RuntimeError(f"Q/K capture expects equal batch-one tensors, got {q.shape} and {k.shape}")
        time, height, width, query_time = self._geometry(q.shape[1])
        spatial = height * width
        q_frames = q[0].view(time, spatial, q.shape[2], q.shape[3])
        k_frames = k[0].view(time, spatial, k.shape[2], k.shape[3])
        source_indices = token_indices(
            self.query_points, height, width, self.pixel_height, self.pixel_width, q.device
        )
        source_q = q_frames[query_time, source_indices].float()
        source_k = k_frames[query_time, source_indices].float()
        predictions = np.full((time, len(source_indices), 2), np.nan, dtype=np.float32)
        probabilities = np.full((time, len(source_indices), spatial), np.nan, dtype=np.float32)
        predictions[query_time] = self.query_points.numpy()
        scale = math.sqrt(q.shape[-1])
        for target_time in range(self.current_prefix, time):
            target_q = q_frames[target_time].float()
            target_k = k_frames[target_time].float()
            scores = torch.einsum("phd,shd->hps", source_q, target_k) / scale
            probability = scores.softmax(dim=-1).mean(dim=0)
            if self.matching_mode == "symmetric":
                reverse = torch.einsum("phd,shd->hps", source_k, target_q) / scale
                probability = 0.5 * (probability + reverse.softmax(dim=-1).mean(dim=0))
            best_index = probability.argmax(dim=-1)
            best_y = torch.div(best_index, width, rounding_mode="floor")
            best_x = best_index % width
            predictions[target_time, :, 0] = (
                (best_x.float() + 0.5) * self.pixel_width / width
            ).cpu().numpy()
            predictions[target_time, :, 1] = (
                (best_y.float() + 0.5) * self.pixel_height / height
            ).cpu().numpy()
            probabilities[target_time] = probability.cpu().numpy()
        self.records[key] = MatchRecord(
            method="qk",
            layer=layer,
            step_index=self.current_step,
            timestep=self.current_timestep,
            sigma=self.current_sigma,
            grid=(time, height, width),
            clean_prefix_latents=self.current_prefix,
            query_latent_index=query_time,
            predictions=predictions,
            probabilities=probabilities,
        )

    def _consume_hidden(self, layer: int, hidden: torch.Tensor) -> None:
        key = ("hidden", layer, self.current_step)
        if key in self.records:
            return
        if hidden.shape[0] != 1:
            raise RuntimeError(f"hidden capture expects batch one, got {hidden.shape}")
        time, height, width, query_time = self._geometry(hidden.shape[1])
        spatial = height * width
        frames = torch.nn.functional.normalize(hidden[0].float(), dim=-1).view(
            time, spatial, hidden.shape[-1]
        )
        source_indices = token_indices(
            self.query_points, height, width, self.pixel_height, self.pixel_width, hidden.device
        )
        source = frames[query_time, source_indices]
        predictions = np.full((time, len(source_indices), 2), np.nan, dtype=np.float32)
        probabilities = np.full((time, len(source_indices), spatial), np.nan, dtype=np.float32)
        predictions[query_time] = self.query_points.numpy()
        for target_time in range(self.current_prefix, time):
            scores = torch.einsum("pd,sd->ps", source, frames[target_time])
            probability = (scores / self.hidden_temperature).softmax(dim=-1)
            best_index = probability.argmax(dim=-1)
            best_y = torch.div(best_index, width, rounding_mode="floor")
            best_x = best_index % width
            predictions[target_time, :, 0] = (
                (best_x.float() + 0.5) * self.pixel_width / width
            ).cpu().numpy()
            predictions[target_time, :, 1] = (
                (best_y.float() + 0.5) * self.pixel_height / height
            ).cpu().numpy()
            probabilities[target_time] = probability.cpu().numpy()
        self.records[key] = MatchRecord(
            method="hidden",
            layer=layer,
            step_index=self.current_step,
            timestep=self.current_timestep,
            sigma=self.current_sigma,
            grid=(time, height, width),
            clean_prefix_latents=self.current_prefix,
            query_latent_index=query_time,
            predictions=predictions,
            probabilities=probabilities,
        )


def latent_anchor_frames(latent_time: int, pixel_frames: int) -> np.ndarray:
    if latent_time <= 1:
        return np.zeros((latent_time,), dtype=np.int64)
    expected = np.arange(latent_time, dtype=np.int64) * 4
    if expected[-1] < pixel_frames:
        return expected
    return np.rint(np.linspace(0, pixel_frames - 1, latent_time)).astype(np.int64)


def run_cotracker(
    model,
    frames: np.ndarray,
    query_points: np.ndarray,
    query_pixel_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device(model.pipe.device)
    frames_tensor = torch.from_numpy(frames).to(device=device, dtype=torch.float32).div(255.0)
    frames_bthwc = frames_tensor.unsqueeze(0)
    points = torch.from_numpy(query_points).to(device=device, dtype=model.pipe.torch_dtype).unsqueeze(0)
    frame_ids = torch.full(
        (1, len(query_points), 1),
        float(query_pixel_frame),
        device=device,
        dtype=model.pipe.torch_dtype,
    )
    output = model._run_cotracker(
        frames_bthwc,
        query_points_prior=points,
        query_frame_ids=frame_ids,
        query_image_hw=frames.shape[1:3],
    )
    return (
        output.tracks[0].detach().float().cpu().numpy(),
        output.visibility[0].detach().float().cpu().numpy() > 0.5,
    )


def evaluate_record(
    record: MatchRecord,
    gt_tracks: np.ndarray | None,
    gt_visibility: np.ndarray | None,
    anchors: np.ndarray,
    pixel_hw: tuple[int, int],
) -> dict[str, Any]:
    time, grid_height, grid_width = record.grid
    row: dict[str, Any] = {
        "method": record.method,
        "layer": record.layer,
        "step_index": record.step_index,
        "timestep": record.timestep,
        "sigma": record.sigma,
        "grid": list(record.grid),
        "clean_prefix_latents": record.clean_prefix_latents,
        "query_latent_index": record.query_latent_index,
        "future_latent_indices": list(range(record.clean_prefix_latents, time)),
    }
    if gt_tracks is None or gt_visibility is None:
        return row
    gt = gt_tracks[anchors]
    visibility = gt_visibility[anchors].copy()
    valid = visibility & visibility[record.query_latent_index : record.query_latent_index + 1]
    valid[: record.clean_prefix_latents] = False
    if not valid.any():
        row["comparisons"] = 0
        return row
    error = np.linalg.norm(record.predictions - gt, axis=-1)
    values = error[valid]
    height, width = pixel_hw
    stride_x = width / grid_width
    stride_y = height / grid_height
    token_radius = math.sqrt(stride_x * stride_y)
    gt_x = np.floor(gt[..., 0] * grid_width / width).astype(np.int64).clip(0, grid_width - 1)
    gt_y = np.floor(gt[..., 1] * grid_height / height).astype(np.int64).clip(0, grid_height - 1)
    gt_index = gt_y * grid_width + gt_x
    target_times, point_ids = np.nonzero(valid)
    selected_probabilities = record.probabilities[target_times, point_ids]
    selected_gt_indices = gt_index[target_times, point_ids]
    gt_probability = selected_probabilities[np.arange(len(point_ids)), selected_gt_indices]
    gt_rank = (selected_probabilities > gt_probability[:, None]).sum(axis=1) + 1
    sorted_probability = np.sort(selected_probabilities, axis=1)
    best_margin = sorted_probability[:, -1] - sorted_probability[:, -2]
    entropy = -np.sum(
        selected_probabilities * np.log(np.clip(selected_probabilities, 1e-12, None)), axis=1
    )
    non_gt_mean = (selected_probabilities.sum(axis=1) - gt_probability) / max(
        selected_probabilities.shape[1] - 1, 1
    )
    row.update(
        {
            "comparisons": int(values.size),
            "mean_error_px": float(values.mean()),
            "median_error_px": float(np.median(values)),
            "pck8": float((values <= 8).mean() * 100),
            "pck16": float((values <= 16).mean() * 100),
            "pck32": float((values <= 32).mean() * 100),
            "pck_one_token": float((values <= token_radius).mean() * 100),
            "mean_gt_probability": float(gt_probability.mean()),
            "median_gt_rank": float(np.median(gt_rank)),
            "mean_gt_rank": float(gt_rank.mean()),
            "top1_gt_rate": float((gt_rank == 1).mean() * 100),
            "mean_gt_minus_non_gt": float((gt_probability - non_gt_mean).mean()),
            "mean_gt_to_non_gt_ratio": float(
                np.mean(gt_probability / np.clip(non_gt_mean, 1e-12, None))
            ),
            "mean_top1_margin": float(best_margin.mean()),
            "mean_entropy": float(entropy.mean()),
            "normalized_entropy": float(entropy.mean() / math.log(selected_probabilities.shape[1])),
        }
    )
    return row


def point_colors(count: int) -> list[tuple[int, int, int]]:
    colors = []
    for index in range(count):
        hue = int(round(179 * index / max(count, 1)))
        bgr = cv2.cvtColor(np.uint8([[[hue, 210, 245]]]), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(value) for value in bgr))
    return colors


def draw_query_points(frame: np.ndarray, points: np.ndarray, output_path: Path) -> None:
    canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for index, (point, color) in enumerate(zip(points, point_colors(len(points)))):
        center = tuple(np.rint(point).astype(int))
        cv2.circle(canvas, center, 5, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(index), (center[0] + 6, center[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imwrite(str(output_path), canvas)


def draw_track_video(
    frames: np.ndarray,
    anchors: np.ndarray,
    record: MatchRecord,
    gt_tracks: np.ndarray | None,
    gt_visibility: np.ndarray | None,
    output_path: Path,
    fps: int,
) -> None:
    colors = point_colors(record.predictions.shape[1])
    gt = gt_tracks[anchors] if gt_tracks is not None else None
    visibility = gt_visibility[anchors] if gt_visibility is not None else None
    selected_times = range(record.query_latent_index, record.grid[0])
    with imageio.get_writer(output_path, fps=max(1, fps // 4), codec="libx264", quality=8) as writer:
        for latent_time in selected_times:
            canvas = cv2.cvtColor(frames[int(anchors[latent_time])], cv2.COLOR_RGB2BGR)
            for point_index, color in enumerate(colors):
                start = max(record.query_latent_index, latent_time - 6)
                for previous in range(start, latent_time):
                    if np.isfinite(record.predictions[previous : previous + 2, point_index]).all():
                        p0 = tuple(np.rint(record.predictions[previous, point_index]).astype(int))
                        p1 = tuple(np.rint(record.predictions[previous + 1, point_index]).astype(int))
                        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
                if np.isfinite(record.predictions[latent_time, point_index]).all():
                    point = tuple(np.rint(record.predictions[latent_time, point_index]).astype(int))
                    cv2.rectangle(
                        canvas, (point[0] - 4, point[1] - 4), (point[0] + 4, point[1] + 4), color, 2
                    )
                if gt is not None and visibility is not None and visibility[latent_time, point_index]:
                    point = tuple(np.rint(gt[latent_time, point_index]).astype(int))
                    cv2.circle(canvas, point, 4, color, -1, cv2.LINE_AA)
            label = (
                f"{record.method} L{record.layer} S{record.step_index} | "
                f"latent {latent_time} / pixel {int(anchors[latent_time])} | circle=CoTracker square=match"
            )
            cv2.putText(canvas, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3)
            cv2.putText(canvas, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
            writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def save_heatmap_montage(
    frames: np.ndarray,
    anchors: np.ndarray,
    record: MatchRecord,
    query_index: int,
    output_path: Path,
) -> None:
    if not 0 <= query_index < record.predictions.shape[1]:
        raise ValueError(f"heatmap query index {query_index} is out of range")
    panels = []
    _, grid_height, grid_width = record.grid
    for latent_time in range(record.clean_prefix_latents, record.grid[0]):
        probability = record.probabilities[latent_time, query_index].reshape(grid_height, grid_width)
        normalized = probability - probability.min()
        normalized /= max(float(normalized.max()), 1e-12)
        heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        frame = cv2.cvtColor(frames[int(anchors[latent_time])], cv2.COLOR_RGB2BGR)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        panel = cv2.addWeighted(frame, 0.55, heatmap, 0.45, 0)
        cv2.putText(
            panel,
            f"latent {latent_time} | pixel {int(anchors[latent_time])}",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        panels.append(panel)
    cv2.imwrite(str(output_path), np.concatenate(panels, axis=1))


def save_records(output_dir: Path, records: list[MatchRecord]) -> None:
    arrays = {}
    for record in records:
        prefix = f"{record.method}_layer{record.layer:02d}_step{record.step_index:03d}"
        arrays[f"{prefix}_predictions"] = record.predictions
    np.savez_compressed(output_dir / "predicted_tracks.npz", **arrays)


def write_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    visual_files: list[str],
) -> None:
    ranked = [row for row in rows if row.get("comparisons", 0) > 0]
    ranked.sort(key=lambda row: (row["method"] != "qk", row.get("mean_error_px", float("inf"))))
    lines = [
        "# Stage1b Kubric 生成过程对应分析",
        "",
        "query 固定在最后一个 clean-context latent frame；target 仅包含 future latent frames。Q/K 是视频 self-attention 在 RMSNorm 和 3D RoPE 之后、FlashAttention 之前的 conditional 分支张量。",
        "",
        f"- Context pixel frames: {manifest['context_pixel_frames']}",
        f"- Clean context latent frames: {manifest['clean_prefix_latents']}",
        f"- Full DiT token grid: {manifest['token_grid']}",
        f"- Query latent index: {manifest['query_latent_index']}",
        f"- Future latent indices: {manifest['future_latent_indices']}",
        f"- Matching mode: `{manifest['matching_mode']}`",
        "",
        "## 指标",
        "",
        "| feature | layer | step | timestep | sigma | mean error | PCK@32 | GT top-1 | mean GT rank | GT probability |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        formatted_sigma = "-" if row["sigma"] is None else f"{row['sigma']:.4f}"
        lines.append(
            "| {method} | {layer} | {step_index} | {timestep:.1f} | {sigma} | "
            "{mean_error_px:.2f} | {pck32:.2f}% | {top1_gt_rate:.2f}% | "
            "{mean_gt_rank:.2f} | {mean_gt_probability:.6f} |".format(
                **{**row, "sigma": formatted_sigma},
            )
        )
    lines.extend(["", "## 可视化", ""])
    lines.extend(f"- [{name}]({name})" for name in visual_files)
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = json.dumps({"manifest": manifest, "rows": ranked}, ensure_ascii=False).replace("</", "<\\/")
    video_tags = "".join(
        f'<section><h2>{name}</h2><video controls muted loop src="{name}"></video></section>'
        for name in visual_files
        if name.endswith(".mp4")
    )
    image_tags = "".join(
        f'<section><h2>{name}</h2><img src="{name}" alt="{name}"></section>'
        for name in visual_files
        if name.endswith(".png")
    )
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage1b Q/K Analysis</title>
<style>:root{{--ink:#19201f;--paper:#eee8da;--card:#fffdf7;--accent:#b9432f}}*{{box-sizing:border-box}}
body{{margin:0;background:radial-gradient(circle at 10% 0,#e9b99a66,transparent 35rem),var(--paper);color:var(--ink);font-family:Georgia,"Noto Serif CJK SC",serif}}
main{{width:min(1200px,calc(100% - 28px));margin:auto;padding:36px 0 80px}}h1{{font-size:clamp(38px,6vw,72px);line-height:.95;margin:0 0 18px}}p{{font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif;line-height:1.7}}
section{{background:var(--card);border:1px solid #d4cab6;padding:18px;margin:18px 0;border-radius:3px 22px 3px 3px}}video,img{{width:100%;display:block;background:#111}}a{{color:var(--accent)}}pre{{overflow:auto;font-size:12px}}</style></head>
<body><main><h1>Context → Future<br>Correspondence</h1><p>最后一个 clean-context latent query；conditional post-RoPE self-attention Q/K。</p>
<p><a href="report.md">实验报告</a> · <a href="metrics.json">完整指标</a> · <a href="manifest.json">运行清单</a></p>
{video_tags}{image_tags}<section><h2>Run payload</h2><pre id="payload"></pre></section>
<script type="application/json" id="data">{payload}</script><script>document.getElementById('payload').textContent=JSON.stringify(JSON.parse(document.getElementById('data').textContent),null,2)</script>
</main></body></html>'''
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    base.apply_vjepa_preset_if_requested(args)
    if args.enable_vjepa_guidance:
        raise ValueError(
            "V-JEPA inference guidance adds extra model_fn calls and is intentionally disabled for the "
            "first correspondence analysis. Run the exact unguided Stage1b process."
        )
    args.device = args.analysis_device or base._resolve_launch_device()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(args.seed))

    step_indices = args.analysis_step_indices or evenly_spaced_steps(int(args.sampling_steps))
    invalid_steps = [step for step in step_indices if not 0 <= step < int(args.sampling_steps)]
    if invalid_steps:
        raise ValueError(f"analysis step indices outside sampling range: {invalid_steps}")
    layers = sorted(set(int(layer) for layer in args.analysis_layers))
    step_indices = sorted(set(int(step) for step in step_indices))

    kubric_infer.base.t0705 = trainmod
    kubric_infer.base._build_object_context = kubric_infer._build_object_context
    kubric_infer.base._build_model_args = kubric_infer._build_model_args
    model, model_args, load_info = base._build_runtime_model(args)
    pipe = model.pipe
    pipe.dit.eval()

    context_path = Path(args.context_video).expanduser().resolve()
    frames, frame_indices = base._load_context_video(
        video_path=context_path,
        target_context_frames=int(args.context_frames),
    )
    context_video = base.preprocess_video_rgb_uint8(frames, (int(args.height), int(args.width)))
    context_pil = base._tensor_video_to_pil_list(context_video)
    query_points = load_query_points(args, int(args.height), int(args.width))

    with torch.inference_mode():
        object_context_raw, object_debug = kubric_infer._build_object_context(
            model,
            context_video_single=context_video,
            prompt=str(args.prompt),
            video_path=str(context_path),
        )
        object_context, ablation_debug = base._apply_object_context_ablation(
            object_context_raw,
            mode=str(args.object_context_ablation),
            random_seed=args.object_context_random_seed,
            random_scale=float(args.object_context_random_scale),
            scale_factor=float(args.object_context_scale_factor),
            token_norm_max=args.object_context_token_norm_max,
        )
        object_debug["object_context_ablation"] = ablation_debug
        capture = GenerationCapture(
            pipe=pipe,
            layers=layers,
            step_indices=step_indices,
            query_points=query_points,
            pixel_hw=(int(args.height), int(args.width)),
            matching_mode=str(args.analysis_matching_mode),
            capture_hidden=not bool(args.analysis_no_hidden),
            hidden_temperature=float(args.analysis_hidden_temperature),
        )
        capture.install()
        try:
            pipe_kwargs = dict(
                prompt=str(args.prompt),
                negative_prompt="",
                context_video=context_pil,
                seed=int(args.seed),
                tiled=True,
                height=int(args.height),
                width=int(args.width),
                num_frames=int(args.num_frames),
                num_inference_steps=int(args.sampling_steps),
                cfg_scale=float(args.cfg_scale),
            )
            if bool(getattr(model, "enable_object_branch", False)):
                pipe_kwargs["object_context"] = object_context
            video = pipe(**pipe_kwargs)
        finally:
            capture.remove()

    records = sorted(capture.records.values(), key=lambda item: (item.method, item.layer, item.step_index))
    expected = len(layers) * len(step_indices) * (1 if args.analysis_no_hidden else 2)
    if len(records) != expected:
        observed = [(record.method, record.layer, record.step_index) for record in records]
        raise RuntimeError(f"captured {len(records)}/{expected} requested records: {observed}")
    reference_record = records[0]
    generated_frames = tensor_video_to_uint8(video)
    anchors = latent_anchor_frames(reference_record.grid[0], len(generated_frames))
    query_pixel_frame = int(anchors[reference_record.query_latent_index])

    generated_video_name = "generated.mp4"
    if not args.analysis_no_video:
        save_video(video, str(output_dir / generated_video_name), fps=int(args.fps), quality=int(args.quality))
    draw_query_points(generated_frames[query_pixel_frame], query_points, output_dir / "query_points.png")

    gt_tracks = None
    gt_visibility = None
    if not args.analysis_no_cotracker:
        gt_tracks, gt_visibility = run_cotracker(
            model, generated_frames, query_points, query_pixel_frame
        )
        np.savez_compressed(
            output_dir / "cotracker_pseudo_gt.npz",
            tracks=gt_tracks,
            visibility=gt_visibility,
            query_points=query_points,
            latent_anchor_frames=anchors,
        )

    rows = [
        evaluate_record(
            record,
            gt_tracks,
            gt_visibility,
            anchors,
            (int(args.height), int(args.width)),
        )
        for record in records
    ]
    save_records(output_dir, records)
    (output_dir / "metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    visualize_layer = (
        int(args.analysis_visualize_layer)
        if args.analysis_visualize_layer is not None
        else layers[len(layers) // 2]
    )
    visualize_step = (
        int(args.analysis_visualize_step_index)
        if args.analysis_visualize_step_index is not None
        else step_indices[-1]
    )
    visual_files = ["query_points.png"]
    if not args.analysis_no_video:
        visual_files.insert(0, generated_video_name)
    for method in ("qk", "hidden"):
        match = capture.records.get((method, visualize_layer, visualize_step))
        if match is None:
            continue
        heatmap_name = f"heatmap_{method}_L{visualize_layer:02d}_S{visualize_step:03d}.png"
        save_heatmap_montage(
            generated_frames,
            anchors,
            match,
            int(args.analysis_heatmap_query_index),
            output_dir / heatmap_name,
        )
        visual_files.append(heatmap_name)
        if not args.analysis_no_video:
            track_name = f"tracks_{method}_L{visualize_layer:02d}_S{visualize_step:03d}.mp4"
            draw_track_video(
                generated_frames,
                anchors,
                match,
                gt_tracks,
                gt_visibility,
                output_dir / track_name,
                int(args.fps),
            )
            visual_files.append(track_name)

    checkpoint_path = Path(base.tvn._resolve_checkpoint_file(args.checkpoint)).resolve()
    manifest = {
        "analysis_protocol": "last_clean_context_latent_to_future_latents",
        "capture_location": "video_self_attention_post_rmsnorm_post_3d_rope_pre_flash_attention",
        "cfg_branch": "positive_conditional_first_call_only",
        "matching_mode": str(args.analysis_matching_mode),
        "checkpoint": str(checkpoint_path),
        "context_video": str(context_path),
        "prompt": str(args.prompt),
        "seed": int(args.seed),
        "sampling_steps": int(args.sampling_steps),
        "layers": layers,
        "step_indices": step_indices,
        "scheduler_timesteps": [float(value) for value in pipe.scheduler.timesteps.detach().float().cpu()],
        "scheduler_sigmas": [float(value) for value in pipe.scheduler.sigmas.detach().float().cpu()]
        if getattr(pipe.scheduler, "sigmas", None) is not None
        else None,
        "requested_num_frames": int(args.num_frames),
        "generated_pixel_frames": int(len(generated_frames)),
        "context_pixel_frames": int(context_video.shape[1]),
        "context_source_frame_indices": frame_indices.tolist(),
        "clean_prefix_latents": int(reference_record.clean_prefix_latents),
        "token_grid": list(reference_record.grid),
        "query_latent_index": int(reference_record.query_latent_index),
        "query_pixel_frame": query_pixel_frame,
        "future_latent_indices": list(
            range(reference_record.clean_prefix_latents, reference_record.grid[0])
        ),
        "latent_anchor_pixel_frames": anchors.tolist(),
        "query_points": query_points.tolist(),
        "height": int(args.height),
        "width": int(args.width),
        "cfg_scale": float(args.cfg_scale),
        "object_branch_enabled": bool(getattr(model, "enable_object_branch", False)),
        "object_debug": object_debug,
        "model_args": {
            "height": int(model_args.height),
            "width": int(model_args.width),
            "num_frames": int(model_args.num_frames),
            "fixed_num_context_frames": int(model_args.fixed_num_context_frames),
        },
        "load_info": base._summarize_load_info(load_info),
        "files": visual_files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(output_dir, rows, manifest, visual_files)
    print(f"Analysis complete: {output_dir}", flush=True)
    print(f"Dashboard: {output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
