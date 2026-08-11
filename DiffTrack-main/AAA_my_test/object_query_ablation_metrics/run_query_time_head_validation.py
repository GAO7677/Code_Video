#!/usr/bin/env python3
"""Validate S039 head ranking for object queries at every latent anchor.

This script performs a baseline Wan forward pass and captures only the first
(conditional) CFG call at denoising step 39.  For each object point and each
latent query time, it matches Q_t to every other latent K_t' independently.
It writes compact counts and argmax locations, but no generated MP4.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
for path in (ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime import (  # noqa: E402
    build_wan_ti2v_pipeline,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations import (  # noqa: E402
    build_args,
    generate_video,
    generation_inputs,
)


VISUAL_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326"
)
DEFAULT_MANIFESTS = [
    VISUAL_ROOT / "cases.json",
    VISUAL_ROOT / "cases_other10_6seeds_latest3350.json",
]
DEFAULT_TRACK_ROOT = VISUAL_ROOT / "attention_matrix_ablations_temporal_tube_v1"
DEFAULT_SPEC = Path(__file__).with_name("experiment_spec_latest3350.json")
DEFAULT_HEAD_SCOPES = VISUAL_ROOT / "pck_head_scopes_s039_latest3350.json"
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/stage1_query_time_validation/runs"
)
PIXEL_HW = (704, 1280)
LATENT_FRAMES = 13
LAYERS = 30
HEADS = 24
STEP = 39


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--manifest-paths", type=Path, nargs="+", default=DEFAULT_MANIFESTS)
    parser.add_argument("--track-root", type=Path, default=DEFAULT_TRACK_ROOT)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--head-scopes", type=Path, default=DEFAULT_HEAD_SCOPES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, str | None]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        diff = subprocess.check_output(
            ["git", "-C", str(ROOT), "diff", "--binary", "--", "AAA_my_test"],
        )
        return {"commit": commit, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tracked_diff_sha256": None}


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def sample_templates(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for sample in payload.get("samples", []):
            result.setdefault(str(sample["case"]), dict(sample))
    return result


def tasks_from_spec(spec: dict[str, Any], templates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    validation = spec["stage1_query_time_validation"]
    tasks = []
    for case in validation["external_exploratory_cases"]:
        if case not in templates:
            raise KeyError(f"no sample template for {case}")
        for seed in validation["seeds"]:
            sample = dict(templates[case])
            sample["seed"] = int(seed)
            tasks.append(sample)
    return tasks


class AllLatentQueryCapture:
    def __init__(
        self,
        pipe,
        tracks: np.ndarray,
        visibility: np.ndarray,
        anchors: np.ndarray,
        region_slices: list[tuple[int, int]],
    ) -> None:
        self.pipe = pipe
        self.tracks = torch.as_tensor(tracks[anchors], dtype=torch.float32)
        self.visibility = np.asarray(visibility[anchors], dtype=bool)
        self.anchors = np.asarray(anchors, dtype=np.int64)
        self.region_slices = region_slices
        self.current_step = -1
        self.current_grid: tuple[int, int, int] | None = None
        self.active = False
        self.call_counts: dict[int, int] = {}
        self.records: set[tuple[int, int]] = set()
        point_count = int(self.tracks.shape[1])
        region_count = len(region_slices)
        self.correct32 = np.zeros((LATENT_FRAMES, LAYERS, HEADS), dtype=np.int32)
        self.comparisons = np.zeros_like(self.correct32)
        self.error_sum = np.zeros((LATENT_FRAMES, LAYERS, HEADS), dtype=np.float64)
        self.pair_correct32 = np.zeros(
            (LATENT_FRAMES, LATENT_FRAMES, LAYERS, HEADS), dtype=np.int32
        )
        self.pair_comparisons = np.zeros_like(self.pair_correct32)
        self.pair_error_sum = np.zeros_like(self.pair_correct32, dtype=np.float64)
        self.per_object_correct32 = np.zeros(
            (region_count, LATENT_FRAMES, LAYERS, HEADS), dtype=np.int32
        )
        self.per_object_comparisons = np.zeros_like(self.per_object_correct32)
        self.per_object_error_sum = np.zeros_like(
            self.per_object_correct32, dtype=np.float64
        )
        self.predictions = np.full(
            (LAYERS, HEADS, LATENT_FRAMES, LATENT_FRAMES, point_count, 2),
            np.nan,
            dtype=np.float16,
        )
        self._handles: list[Any] = []
        self._original_model_fn = None

    def scheduler_step(self, timestep: torch.Tensor) -> int:
        value = float(timestep.detach().flatten()[0].float().cpu())
        schedule = self.pipe.scheduler.timesteps.detach().float().cpu()
        return int(torch.argmin((schedule - value).abs()).item())

    def wrapped_model_fn(self, original):
        def wrapped(*args, **kwargs):
            timestep, latents = kwargs.get("timestep"), kwargs.get("latents")
            if timestep is None or latents is None:
                return original(*args, **kwargs)
            step = self.scheduler_step(timestep)
            call = self.call_counts.get(step, 0)
            self.call_counts[step] = call + 1
            patch = tuple(int(value) for value in kwargs["dit"].patch_size)
            self.current_grid = (
                int(latents.shape[2] // patch[0]),
                int(latents.shape[3] // patch[1]),
                int(latents.shape[4] // patch[2]),
            )
            self.current_step = step
            self.active = step == STEP and call == 0
            try:
                return original(*args, **kwargs)
            finally:
                self.active = False
                self.current_step = -1

        return wrapped

    def install(self) -> None:
        self._original_model_fn = self.pipe.model_fn
        self.pipe.model_fn = self.wrapped_model_fn(self.pipe.model_fn)
        models = [self.pipe.dit]
        if getattr(self.pipe, "dit2", None) is not None and self.pipe.dit2 is not self.pipe.dit:
            models.append(self.pipe.dit2)
        for model in models:
            for layer in range(LAYERS):
                self._handles.append(
                    model.blocks[layer].self_attn.attn.register_forward_pre_hook(
                        self.make_hook(layer)
                    )
                )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._original_model_fn is not None:
            self.pipe.model_fn = self._original_model_fn
            self._original_model_fn = None

    def make_hook(self, layer: int):
        def hook(module, inputs):
            key = (self.current_step, layer)
            if not self.active or key in self.records:
                return
            q, k = inputs[:2]
            if q.shape[0] != 1 or q.shape != k.shape or self.current_grid is None:
                raise RuntimeError(f"unexpected Q/K shapes: {tuple(q.shape)}, {tuple(k.shape)}")
            time_count, height, width = self.current_grid
            if time_count != LATENT_FRAMES or time_count * height * width != q.shape[1]:
                raise RuntimeError(
                    f"token geometry mismatch: q={tuple(q.shape)} grid={self.current_grid}"
                )
            num_heads = int(module.num_heads)
            if num_heads != HEADS:
                raise RuntimeError(f"expected {HEADS} heads, got {num_heads}")
            dim = q.shape[-1] // num_heads
            spatial_count = height * width
            q_frames = q[0].view(time_count, spatial_count, num_heads, dim)
            k_frames = k[0].view(time_count, spatial_count, num_heads, dim)
            points = self.tracks.to(q.device)
            x = torch.floor(points[..., 0] * width / PIXEL_HW[1]).long().clamp(0, width - 1)
            y = torch.floor(points[..., 1] * height / PIXEL_HW[0]).long().clamp(0, height - 1)
            time_indices = torch.arange(time_count, device=q.device)[:, None]
            q_source = q_frames[time_indices, y * width + x]
            layer_predictions = np.full(
                (HEADS, time_count, time_count, points.shape[1], 2),
                np.nan,
                dtype=np.float32,
            )
            scale = math.sqrt(dim)
            for head_start in range(0, num_heads, 4):
                head_end = min(num_heads, head_start + 4)
                scores = torch.einsum(
                    "tphd,ushd->htpus",
                    q_source[:, :, head_start:head_end].float(),
                    k_frames[:, :, head_start:head_end].float(),
                ) / scale
                best = scores.argmax(dim=-1)
                del scores
                best_y = torch.div(best, width, rounding_mode="floor")
                best_x = best % width
                pred_x = (best_x.float() + 0.5) * PIXEL_HW[1] / width
                pred_y = (best_y.float() + 0.5) * PIXEL_HW[0] / height
                prediction = torch.stack((pred_x, pred_y), dim=-1)
                # h, tq, point, tk, xy -> h, tq, tk, point, xy
                layer_predictions[head_start:head_end] = (
                    prediction.permute(0, 1, 3, 2, 4).cpu().numpy()
                )
                del best, best_x, best_y, pred_x, pred_y, prediction

            self.predictions[layer] = layer_predictions.astype(np.float16)
            gt = self.tracks.cpu().numpy()
            error = np.linalg.norm(
                layer_predictions - gt[None, None, :, :, :], axis=-1
            )
            # error: head, tq, tk, point
            valid = self.visibility[:, None, :] & self.visibility[None, :, :]
            valid &= ~np.eye(time_count, dtype=bool)[:, :, None]
            mask = valid[None] & np.isfinite(error)
            for query_time in range(time_count):
                query_mask = mask[:, query_time]
                self.correct32[query_time, layer] = (
                    ((error[:, query_time] <= 32.0) & query_mask)
                    .sum(axis=(1, 2))
                    .astype(np.int32)
                )
                self.comparisons[query_time, layer] = query_mask.sum(axis=(1, 2)).astype(
                    np.int32
                )
                self.error_sum[query_time, layer] = np.where(
                    query_mask, error[:, query_time], 0.0
                ).sum(axis=(1, 2), dtype=np.float64)
                for target_time in range(time_count):
                    pair_mask = mask[:, query_time, target_time]
                    self.pair_correct32[query_time, target_time, layer] = (
                        (
                            (error[:, query_time, target_time] <= 32.0)
                            & pair_mask
                        )
                        .sum(axis=1)
                        .astype(np.int32)
                    )
                    self.pair_comparisons[query_time, target_time, layer] = pair_mask.sum(
                        axis=1
                    ).astype(np.int32)
                    self.pair_error_sum[query_time, target_time, layer] = np.where(
                        pair_mask, error[:, query_time, target_time], 0.0
                    ).sum(axis=1, dtype=np.float64)
                for region_index, (start, end) in enumerate(self.region_slices):
                    region_mask = query_mask[:, :, start:end]
                    region_error = error[:, query_time, :, start:end]
                    self.per_object_correct32[region_index, query_time, layer] = (
                        ((region_error <= 32.0) & region_mask)
                        .sum(axis=(1, 2))
                        .astype(np.int32)
                    )
                    self.per_object_comparisons[region_index, query_time, layer] = (
                        region_mask.sum(axis=(1, 2)).astype(np.int32)
                    )
                    self.per_object_error_sum[region_index, query_time, layer] = np.where(
                        region_mask, region_error, 0.0
                    ).sum(axis=(1, 2), dtype=np.float64)
            self.records.add(key)

        return hook


def run_root(output_root: Path, case: str, seed: int) -> Path:
    return output_root / case / f"seed_{seed:05d}"


def process_task(
    pipe,
    sample: dict[str, Any],
    case_lookup: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    case, seed = str(sample["case"]), int(sample["seed"])
    output = run_root(args.output_root, case, seed)
    complete_path = output / "complete.json"
    if complete_path.is_file() and (output / "metrics.npz").is_file() and not args.overwrite:
        print(f"skip {case}/seed_{seed:05d}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    complete_path.unlink(missing_ok=True)
    (output / "error.txt").unlink(missing_ok=True)

    track_path = (
        args.track_root / case / f"seed_{seed:05d}" / "frozen_baseline_tracks" / "tracks.npz"
    )
    if not track_path.is_file():
        raise FileNotFoundError(track_path)
    with np.load(track_path, allow_pickle=False) as arrays:
        tracks = arrays["tracks"].astype(np.float32)
        visibility = arrays["visibility"].astype(bool)
        anchors = arrays["anchor_pixel_frames"].astype(np.int64)
        region_names = arrays["region_names"].astype(str)
        starts = arrays["point_starts"].astype(np.int64)
        ends = arrays["point_ends"].astype(np.int64)
    if len(anchors) != LATENT_FRAMES:
        raise RuntimeError(f"expected {LATENT_FRAMES} anchors, got {len(anchors)}")

    json_path, _, payload, generation_args, image = generation_inputs(
        sample, case_lookup, seed
    )
    capture = AllLatentQueryCapture(
        pipe.pipe,
        tracks,
        visibility,
        anchors,
        list(zip(starts.tolist(), ends.tolist())),
    )
    capture.install()
    try:
        video = generate_video(pipe, payload, generation_args, image, seed)
    finally:
        capture.remove()
    del video
    if capture.records != {(STEP, layer) for layer in range(LAYERS)}:
        raise RuntimeError(
            f"capture incomplete: {len(capture.records)}/{LAYERS} S{STEP:03d} layers"
        )
    if len(capture.call_counts) != 40 or any(value != 2 for value in capture.call_counts.values()):
        raise RuntimeError(f"unexpected CFG/step calls: {capture.call_counts}")

    atomic_npz(
        output / "metrics.npz",
        correct32=capture.correct32,
        comparisons=capture.comparisons,
        error_sum=capture.error_sum,
        pair_correct32=capture.pair_correct32,
        pair_comparisons=capture.pair_comparisons,
        pair_error_sum=capture.pair_error_sum,
        per_object_correct32=capture.per_object_correct32,
        per_object_comparisons=capture.per_object_comparisons,
        per_object_error_sum=capture.per_object_error_sum,
        predictions=capture.predictions,
        latent_anchor_pixel_frames=anchors,
        region_names=region_names,
        point_starts=starts,
        point_ends=ends,
    )
    scheduler = pipe.pipe.scheduler
    timestep_values = scheduler.timesteps.detach().float().cpu().tolist()
    sigma_values = (
        scheduler.sigmas.detach().float().cpu().tolist()
        if hasattr(scheduler, "sigmas") and torch.is_tensor(scheduler.sigmas)
        else None
    )
    provenance = git_state()
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": case,
        "seed": seed,
        "source_json": str(json_path),
        "track_path": str(track_path),
        "track_sha256": sha256_file(track_path),
        "experiment_spec": str(args.spec),
        "experiment_spec_sha256": sha256_file(args.spec),
        "head_scope_path": str(args.head_scopes),
        "head_scope_sha256": sha256_file(args.head_scopes),
        "script_sha256": sha256_file(Path(__file__)),
        "git": provenance,
        "protocol": "S039_all_latent_Qt_to_all_other_Kt_argmax_PCK32_v1",
        "capture_location": "self_attention_post_rmsnorm_post_3d_rope_pre_flash_attention",
        "cfg_branch": "conditional_first_call_only",
        "denoising_step": STEP,
        "latent_query_indices": list(range(LATENT_FRAMES)),
        "latent_target_policy": "all_tk_except_tq",
        "visibility_policy": "query_and_target_CoTracker_visibility_required",
        "matching": "per-target-frame per-head Q_t to K_t' argmax; no head averaging",
        "pck_threshold_px": 32,
        "pixel_hw": list(PIXEL_HW),
        "latent_anchor_pixel_frames": anchors.tolist(),
        "region_names": region_names.tolist(),
        "point_slices": [
            {"region": name, "start": int(start), "end": int(end)}
            for name, start, end in zip(region_names.tolist(), starts, ends)
        ],
        "scheduler_timesteps": timestep_values,
        "scheduler_sigmas": sigma_values,
        "model_call_counts": capture.call_counts,
        "captured_layer_count": len(capture.records),
        "predictions_dtype": "float16_pixel_coordinates",
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(
        complete_path,
        {
            "case": case,
            "seed": seed,
            "denoising_step": STEP,
            "captured_layers": len(capture.records),
            "query_anchors": LATENT_FRAMES,
        },
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0, num-workers)")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    templates = sample_templates(args.manifest_paths)
    tasks = tasks_from_spec(spec, templates)
    if args.task_index is not None:
        if not 0 <= args.task_index < len(tasks):
            raise ValueError(f"task-index must be in [0, {len(tasks)})")
        tasks = [tasks[args.task_index]]
    else:
        tasks = tasks[args.worker_id :: args.num_workers]
    if args.dry_run:
        print(
            json.dumps(
                [{"case": row["case"], "seed": row["seed"]} for row in tasks],
                indent=2,
            )
        )
        return
    if not tasks:
        return

    from AAA_my_test.legacy_ti2v_firstlatent_physiciq67_common import CASES

    case_lookup = {case.key: case for case in CASES}
    args.output_root.mkdir(parents=True, exist_ok=True)
    pipe = build_wan_ti2v_pipeline(build_args(int(tasks[0]["seed"])))
    for index, sample in enumerate(tasks, start=1):
        case, seed = str(sample["case"]), int(sample["seed"])
        print(f"[{index}/{len(tasks)}] start {case}/seed_{seed:05d}", flush=True)
        try:
            process_task(pipe, sample, case_lookup, args)
        except Exception:
            output = run_root(args.output_root, case, seed)
            output.mkdir(parents=True, exist_ok=True)
            (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(traceback.format_exc(), flush=True)
            raise
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{index}/{len(tasks)}] complete {case}/seed_{seed:05d}", flush=True)


if __name__ == "__main__":
    main()
