#!/usr/bin/env python3
"""Compact all-step/all-block/all-head PCK@32 capture for legacy Wan2.2 TI2V."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
COTRACKER_CHECKPOINT = Path("/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")
for path in (ROOT, CODE_ROOT, COTRACKER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime import (
    WanTI2VArgs,
    build_wan_ti2v_pipeline,
    ensure_firstframe_image,
    run_single_case,
)
from AAA_my_test.legacy_ti2v_firstlatent_common import (
    LEGACY_VIDEO_ROOT,
    OUTPUT_ROOT,
    REGION_CACHE_ROOT,
    SEEDS_FILE,
    WAN_ROOT,
    all_tasks,
    read_payload,
    run_dir,
)
from AAA_my_test.sam2_region_query_utils import load_region_cache, region_metadata


class CompactFirstLatentCapture:
    def __init__(self, pipe, query_points: np.ndarray, pixel_hw: tuple[int, int]) -> None:
        self.pipe = pipe
        self.query_points = torch.from_numpy(query_points).float()
        self.pixel_height, self.pixel_width = pixel_hw
        self.layers = set(range(30))
        self.steps = set(range(40))
        self.records: dict[tuple[int, int], np.ndarray] = {}
        self.call_counts: dict[int, int] = {}
        self.current_step = -1
        self.current_grid: tuple[int, int, int] | None = None
        self.active = False
        self._handles: list[Any] = []
        self._original_model_fn = None

    def _scheduler_step(self, timestep: torch.Tensor) -> int:
        value = float(timestep.detach().flatten()[0].float().cpu())
        timesteps = self.pipe.scheduler.timesteps.detach().float().cpu()
        return int(torch.argmin((timesteps - value).abs()).item())

    def _wrap_model_fn(self, original):
        def wrapped(*args, **kwargs):
            timestep, latents = kwargs.get("timestep"), kwargs.get("latents")
            if timestep is None or latents is None:
                return original(*args, **kwargs)
            step = self._scheduler_step(timestep)
            call = self.call_counts.get(step, 0)
            self.call_counts[step] = call + 1
            patch = tuple(int(value) for value in kwargs["dit"].patch_size)
            self.current_grid = (
                int(latents.shape[2] // patch[0]),
                int(latents.shape[3] // patch[1]),
                int(latents.shape[4] // patch[2]),
            )
            self.current_step = step
            self.active = call == 0 and step in self.steps
            try:
                return original(*args, **kwargs)
            finally:
                self.active = False

        return wrapped

    def install(self) -> None:
        self._original_model_fn = self.pipe.model_fn
        self.pipe.model_fn = self._wrap_model_fn(self.pipe.model_fn)
        models = [self.pipe.dit]
        if getattr(self.pipe, "dit2", None) is not None and self.pipe.dit2 is not self.pipe.dit:
            models.append(self.pipe.dit2)
        for model in models:
            for layer in sorted(self.layers):
                self._handles.append(
                    model.blocks[layer].self_attn.attn.register_forward_pre_hook(
                        self._make_hook(layer)
                    )
                )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._original_model_fn is not None:
            self.pipe.model_fn = self._original_model_fn
            self._original_model_fn = None

    def _make_hook(self, layer: int):
        def hook(module, inputs):
            key = (self.current_step, layer)
            if not self.active or key in self.records:
                return
            q, k = inputs[:2]
            if q.shape[0] != 1 or q.shape != k.shape or self.current_grid is None:
                raise RuntimeError(f"unexpected Q/K shapes: {tuple(q.shape)}, {tuple(k.shape)}")
            time, height, width = self.current_grid
            if time * height * width != q.shape[1]:
                raise RuntimeError(
                    f"token geometry mismatch: sequence={q.shape[1]}, grid={self.current_grid}"
                )
            heads = int(module.num_heads)
            dim = q.shape[-1] // heads
            q_frames = q[0].view(time, height * width, heads, dim)
            k_frames = k[0].view(time, height * width, heads, dim)
            points = self.query_points.to(q.device)
            x = torch.floor(points[:, 0] * width / self.pixel_width).long().clamp(0, width - 1)
            y = torch.floor(points[:, 1] * height / self.pixel_height).long().clamp(0, height - 1)
            source = q_frames[0, y * width + x].float()
            predictions = np.full((heads, time, len(points), 2), np.nan, dtype=np.float32)
            predictions[:, 0] = self.query_points.numpy()[None]
            scale = math.sqrt(dim)
            for target_time in range(1, time):
                scores = torch.einsum(
                    "phd,shd->hps", source, k_frames[target_time].float()
                ) / scale
                best = scores.argmax(dim=-1)
                best_y = torch.div(best, width, rounding_mode="floor")
                best_x = best % width
                predictions[:, target_time, :, 0] = (
                    (best_x.float() + 0.5) * self.pixel_width / width
                ).cpu().numpy()
                predictions[:, target_time, :, 1] = (
                    (best_y.float() + 0.5) * self.pixel_height / height
                ).cpu().numpy()
            self.records[key] = predictions

        return hook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_cotracker(device: str):
    from cotracker.predictor import CoTrackerPredictor

    return CoTrackerPredictor(
        checkpoint=str(COTRACKER_CHECKPOINT), offline=True, v2=False, window_len=60
    ).to(device).eval().requires_grad_(False)


def run_cotracker(model, frames: np.ndarray, points: np.ndarray, device: str):
    input_height, input_width = 384, 512
    native_height, native_width = frames.shape[1:3]
    video = torch.from_numpy(frames).float().div(255.0).permute(0, 3, 1, 2)
    video = torch.nn.functional.interpolate(
        video, size=(input_height, input_width), mode="bilinear", align_corners=True
    ).unsqueeze(0).to(device)
    query = torch.from_numpy(points).float().to(device)
    query[:, 0] *= input_width / native_width
    query[:, 1] *= input_height / native_height
    frame_ids = torch.zeros((len(query), 1), device=device)
    queries = torch.cat((frame_ids, query), dim=-1).unsqueeze(0)
    with torch.inference_mode():
        tracks, visibility = model(video, queries=queries, backward_tracking=False)
    tracks = tracks[0].float().cpu().numpy()
    tracks[..., 0] *= max(native_width - 1, 1) / max(input_width - 1, 1)
    tracks[..., 1] *= max(native_height - 1, 1) / max(input_height - 1, 1)
    return tracks, visibility[0].float().cpu().numpy() > 0.5


def build_args(seed: int) -> WanTI2VArgs:
    manifest_path = LEGACY_VIDEO_ROOT / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return WanTI2VArgs(
        input_list=SEEDS_FILE,
        output_root=OUTPUT_ROOT,
        model_name="wan_ti2v_5b_legacy_firstlatent_pck50",
        wan_root=WAN_ROOT,
        backend="legacy",
        size="704*1280",
        frame_num=49,
        fps=30,
        seed=int(seed),
        sample_solver="unipc",
        sampling_steps=40,
        sample_shift=5.0,
        cfg_scale=5.0,
        negative_prompt=str(manifest.get("negative_prompt", "")),
        offload_model=False,
        t5_cpu=False,
        convert_model_dtype=False,
        force=True,
    )


def object_queries(cache):
    points, regions, start = [], [], 0
    for region in cache.regions:
        if region.region_type != "object":
            continue
        selected = cache.query_points[region.point_start : region.point_end]
        points.append(selected)
        end = start + len(selected)
        regions.append((region, slice(start, end)))
        start = end
    if not points:
        raise RuntimeError("region cache contains no object query points")
    return np.concatenate(points).astype(np.float32), regions


def atomic_npz(path: Path, **arrays) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def process_task(pipe, cotracker, case, seed: int, device: str, overwrite: bool) -> None:
    output = run_dir(case.key, seed)
    if (output / "complete.json").is_file() and not overwrite:
        print(f"skip {case.key} seed={seed}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    cache = load_region_cache(REGION_CACHE_ROOT, case.key)
    if int(cache.metadata.get("query_context_frame", -1)) != 0:
        raise RuntimeError(f"{case.key}: expected query frame 0 cache")
    query_points, query_regions = object_queries(cache)
    payload = read_payload(case)
    payload, firstframe = ensure_firstframe_image(case.json_path, payload)
    args = build_args(seed)
    capture = CompactFirstLatentCapture(pipe.pipe, query_points, (704, 1280))
    capture.install()
    try:
        result, logs = run_single_case(
            pipe=pipe,
            args=args,
            input_json_path=case.json_path,
            payload=payload,
            firstframe_path=firstframe,
            output_video=output / "generated.mp4",
        )
    finally:
        capture.remove()
    frames = iio.imread(output / "generated.mp4")
    gt_tracks, gt_visibility = run_cotracker(cotracker, frames, query_points, device)
    if not capture.records:
        raise RuntimeError("no self-attention records captured")
    reference = next(iter(capture.records.values()))
    latent_time = int(reference.shape[1])
    anchors = np.arange(latent_time, dtype=np.int64) * 4
    if int(anchors[-1]) >= len(frames):
        anchors = np.rint(np.linspace(0, len(frames) - 1, latent_time)).astype(np.int64)
    gt = gt_tracks[anchors]
    visibility = gt_visibility[anchors] & gt_visibility[0][None]
    visibility[0] = False
    correct32 = np.zeros((40, 30, 24), dtype=np.int32)
    comparisons = np.zeros((40, 30, 24), dtype=np.int32)
    error_sum = np.zeros((40, 30, 24), dtype=np.float64)
    per_object_correct = np.zeros((len(query_regions), 40, 30, 24), dtype=np.int32)
    per_object_comparisons = np.zeros_like(per_object_correct)
    for (step, layer), predictions in capture.records.items():
        for region_index, (_, point_slice) in enumerate(query_regions):
            valid = visibility[:, point_slice]
            error = np.linalg.norm(
                predictions[:, :, point_slice] - gt[None, :, point_slice], axis=-1
            )
            finite = np.isfinite(error)
            mask = valid[None] & finite
            count = mask.sum(axis=(1, 2)).astype(np.int32)
            correct = ((error <= 32.0) & mask).sum(axis=(1, 2)).astype(np.int32)
            summed_error = np.where(mask, error, 0.0).sum(axis=(1, 2), dtype=np.float64)
            correct32[step, layer, : len(correct)] += correct
            comparisons[step, layer, : len(count)] += count
            error_sum[step, layer, : len(summed_error)] += summed_error
            per_object_correct[region_index, step, layer, : len(correct)] = correct
            per_object_comparisons[region_index, step, layer, : len(count)] = count
    atomic_npz(
        output / "metrics.npz",
        correct32=correct32,
        comparisons=comparisons,
        error_sum=error_sum,
        per_object_correct32=per_object_correct,
        per_object_comparisons=per_object_comparisons,
    )
    manifest = {
        **result,
        "case_key": case.key,
        "source_json": str(case.json_path),
        "analysis_protocol": "first_latent_frame_query_to_future_latent_frames",
        "capture_location": "self_attention_post_rmsnorm_post_3d_rope_pre_flash_attention",
        "cfg_branch": "positive_conditional_first_call_only",
        "query_latent_index": 0,
        "query_pixel_frame": 0,
        "latent_anchor_pixel_frames": anchors.tolist(),
        "token_grid": [latent_time, int(reference.shape[-3] * 0 + pipe.pipe.dit.patch_size[1]), 0],
        "query_points": query_points.tolist(),
        "query_regions": [region_metadata(region) for region, _ in query_regions],
        "layers": list(range(30)),
        "step_indices": list(range(40)),
        "heads": 24,
        "matching": "per-target-frame Q-to-K argmax; no head averaging",
        "logs": logs,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "complete.json").write_text(
        json.dumps(
            {
                "case": case.key,
                "seed": int(seed),
                "captured_combinations": int(len(capture.records) * 24),
                "expected_combinations": 40 * 30 * 24,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0, num-workers)")
    tasks = all_tasks()[args.worker_id :: args.num_workers]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pipe = build_wan_ti2v_pipeline(build_args(tasks[0][1]))
    cotracker = load_cotracker(str(args.device))
    for index, (case, seed) in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] start {case.key} seed={seed}", flush=True)
        try:
            process_task(pipe, cotracker, case, seed, str(args.device), bool(args.overwrite))
        except Exception:
            output = run_dir(case.key, seed)
            output.mkdir(parents=True, exist_ok=True)
            (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(traceback.format_exc(), flush=True)
            raise
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{index}/{len(tasks)}] complete {case.key} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
