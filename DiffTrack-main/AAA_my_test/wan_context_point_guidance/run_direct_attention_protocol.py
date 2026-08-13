#!/usr/bin/env python3
"""Direct selected-head attention control along tracked object trajectories.

Unlike latent guidance, this runner never differentiates through or updates
the noisy latent.  At selected Wan self-attention heads it replaces only the
tracked point Query rows with an equal-total-variation interpolation toward
same-ID CoTracker point targets, applies the same operator to both CFG
branches, and then performs the ordinary FlowMatch scheduler step.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


DIFFTRACK_ROOT = Path(__file__).resolve().parents[2]
if str(DIFFTRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFTRACK_ROOT))

from AAA_my_test.wan_context_point_guidance import run_dual_protocol as base  # noqa: E402
from AAA_my_test.wan_context_point_guidance.direct_attention import (  # noqa: E402
    VALID_DIRECTIONS,
    intervene_attention_rows,
    point_attention_targets,
)


PROTOCOL = "wan_direct_point_attention_tv_v1"
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_context_point_guidance_head_compare/"
    "direct_attention_tv_v1"
)
FOCAL_CASE = "0613pybullet_sample_001460_w002"


class DirectAttentionController:
    """Patch selected attention outputs without changing Q/K/V or latents."""

    def __init__(
        self,
        pipe: Any,
        entries: list[dict[str, Any]],
        tracks_tn2: np.ndarray,
        visibility_tn: np.ndarray,
        pixel_hw: tuple[int, int],
        context_times: tuple[int, ...],
        future_times: tuple[int, ...],
        direction: str,
        sigma_tokens: float,
        tv_budget: float,
        capture_steps: tuple[int, ...],
    ) -> None:
        self.pipe = pipe
        self.entries = entries
        self.tracks_tn2 = np.asarray(tracks_tn2, dtype=np.float32)
        self.visibility_tn = torch.from_numpy(np.asarray(visibility_tn, dtype=bool))
        self.pixel_hw = tuple(int(value) for value in pixel_hw)
        self.context_times = tuple(int(value) for value in context_times)
        self.future_times = tuple(int(value) for value in future_times)
        self.direction = str(direction)
        self.sigma_tokens = float(sigma_tokens)
        self.tv_budget = float(tv_budget)
        self.capture_steps = {int(value) for value in capture_steps}
        self.by_block: dict[int, list[int]] = {}
        for row in entries:
            self.by_block.setdefault(int(row["block"]), []).append(int(row["head"]))
        self.current_grid: tuple[int, int, int] | None = None
        self.current_step = -1
        self.current_branch = ""
        self.active = False
        self._originals: list[tuple[Any, Any]] = []
        self._geometry_cache: dict[
            tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._metric_sums: dict[str, float] = {}
        self._metric_counts: dict[str, int] = {}
        self._events = {"positive": 0, "negative": 0}
        self._capture_sums: dict[str, np.ndarray] = {}
        self._capture_count = 0

    def _point_rows(self, token_hw: tuple[int, int]) -> torch.Tensor:
        return base.points_to_token_rows(self.tracks_tn2, self.pixel_hw, token_hw)

    def _geometry(
        self, token_hw: tuple[int, int], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(token_hw[0]), int(token_hw[1]), str(device))
        cached = self._geometry_cache.get(key)
        if cached is None:
            cached = point_attention_targets(
                self._point_rows(token_hw),
                self.visibility_tn,
                token_hw,
                self.context_times,
                self.future_times,
                self.direction,
                self.sigma_tokens,
                device,
            )
            self._geometry_cache[key] = cached
        return cached

    def begin_step(self, step: int, grid: tuple[int, int, int]) -> None:
        if int(grid[0]) != base.LATENT_FRAMES:
            raise RuntimeError(f"expected 13 latent frames, got {grid}")
        self.current_step = int(step)
        self.current_grid = tuple(int(value) for value in grid)
        self.current_branch = ""
        self._metric_sums.clear()
        self._metric_counts.clear()
        self._events = {"positive": 0, "negative": 0}
        self._capture_sums.clear()
        self._capture_count = 0

    def set_branch(self, branch: str) -> None:
        if branch not in self._events:
            raise ValueError(f"unknown CFG branch: {branch}")
        self.current_branch = branch

    def _accumulate(self, name: str, tensor: torch.Tensor) -> None:
        value = tensor.detach().float()
        self._metric_sums[name] = self._metric_sums.get(name, 0.0) + float(
            value.sum().cpu()
        )
        self._metric_counts[name] = self._metric_counts.get(name, 0) + value.numel()

    def _capture(self, before: torch.Tensor, after: torch.Tensor) -> None:
        if self.current_branch != "positive" or self.current_step + 1 not in self.capture_steps:
            return
        for name, tensor in (("pre", before), ("post", after)):
            value = tensor.detach().float().sum(dim=(0, 1, 2)).cpu().numpy()
            self._capture_sums[name] = self._capture_sums.get(name, 0.0) + value
        self._capture_count += int(before.shape[0] * before.shape[1] * before.shape[2])

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        original: Any,
        block: int,
    ) -> torch.Tensor:
        heads = self.by_block.get(block, ())
        if not self.active or not heads:
            return original(q, k, v)
        if self.current_grid is None or not self.current_branch:
            raise RuntimeError("direct attention controller step/branch is unset")
        time_count, token_height, token_width = self.current_grid
        sequence = time_count * token_height * token_width
        if q.shape[1] != sequence:
            raise RuntimeError(f"Q geometry mismatch: {q.shape} vs {self.current_grid}")
        num_heads = int(q.shape[-1] // base.HEAD_DIM)
        if num_heads != 24:
            raise RuntimeError(f"expected 24 Wan heads, got {num_heads}")

        fused = original(q, k, v)
        selected = torch.as_tensor(heads, device=q.device, dtype=torch.long)
        query_rows, targets = self._geometry((token_height, token_width), q.device)
        qh = q.view(q.shape[0], sequence, num_heads, base.HEAD_DIM)
        kh = k.view(k.shape[0], sequence, num_heads, base.HEAD_DIM)
        vh = v.view(v.shape[0], sequence, num_heads, base.HEAD_DIM)
        q_selected = qh[:, query_rows][:, :, selected].contiguous()
        k_selected = kh[:, :, selected].contiguous()
        v_selected = vh[:, :, selected].contiguous()
        replacement, metrics = intervene_attention_rows(
            q_selected, k_selected, v_selected, targets, self.tv_budget
        )

        output = fused.view(fused.shape[0], sequence, num_heads, base.HEAD_DIM).clone()
        selected_output = output[:, :, selected].clone()
        selected_output[:, query_rows] = replacement.to(selected_output.dtype)
        output[:, :, selected] = selected_output

        self._events[self.current_branch] += len(heads)
        self._accumulate("actual_tv", metrics["actual_tv"])
        self._accumulate("target_ce_before", metrics["target_ce_before"])
        self._accumulate("target_ce_after", metrics["target_ce_after"])
        self._accumulate("av_delta_rms", metrics["av_delta_rms"])
        self._accumulate("blend", metrics["blend"])
        self._capture(metrics["before"], metrics["after"])
        return output.reshape_as(fused)

    def end_step(self) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
        expected = len(self.entries)
        if self._events != {"positive": expected, "negative": expected}:
            raise RuntimeError(
                f"selected-head events mismatch: {self._events}, expected {expected}/branch"
            )
        means = {
            name: self._metric_sums[name] / max(self._metric_counts[name], 1)
            for name in self._metric_sums
        }
        report = {
            "step": int(self.current_step),
            "step_1based": int(self.current_step + 1),
            "direction": self.direction,
            "attention_tv_budget": self.tv_budget,
            "positive_head_events": int(self._events["positive"]),
            "negative_head_events": int(self._events["negative"]),
            "query_rows": int(self._geometry(self.current_grid[1:], torch.device("cpu"))[0].numel()),
            "mean_actual_tv": means["actual_tv"],
            "mean_target_ce_before": means["target_ce_before"],
            "mean_target_ce_after": means["target_ce_after"],
            "mean_target_ce_change": means["target_ce_after"] - means["target_ce_before"],
            "mean_av_delta_rms": means["av_delta_rms"],
            "mean_blend": means["blend"],
            "latent_update_rms": 0.0,
            "model_parameters_updated": False,
        }
        capture = None
        if self._capture_count:
            shape = self.current_grid
            capture = {
                "pre_heatmap": (
                    self._capture_sums["pre"] / float(self._capture_count)
                ).reshape(shape),
                "post_heatmap": (
                    self._capture_sums["post"] / float(self._capture_count)
                ).reshape(shape),
            }
        return report, capture

    def install(self) -> None:
        models = [self.pipe.dit]
        if getattr(self.pipe, "dit2", None) is not None and self.pipe.dit2 is not self.pipe.dit:
            models.append(self.pipe.dit2)
        for model in models:
            for block in self.by_block:
                module = model.blocks[block].self_attn.attn
                original = module.forward
                self._originals.append((module, original))

                def wrapped(q, k, v, *, _original=original, _block=block):
                    return self._attention(q, k, v, _original, _block)

                module.forward = wrapped

    def remove(self) -> None:
        for module, original in self._originals:
            module.forward = original
        self._originals.clear()


def build_controller(
    pipe: Any,
    spec: base.BackendSpec,
    tube: Any,
    target: Any,
    entries: list[dict[str, Any]],
    direction: str,
    args: argparse.Namespace,
) -> tuple[DirectAttentionController, dict[str, Any]]:
    tracks, visibility, geometry = base.target_point_arrays(tube, target, spec)
    return DirectAttentionController(
        pipe,
        entries,
        tracks,
        visibility,
        (spec.height, spec.width),
        spec.key_times,
        spec.query_times,
        direction,
        args.gaussian_sigma_tokens,
        args.attention_tv_budget,
        args.attention_capture_steps,
    ), geometry


def run_denoising(
    pipe: Any,
    spec: base.BackendSpec,
    inputs_shared: dict[str, Any],
    inputs_posi: dict[str, Any],
    inputs_nega: dict[str, Any],
    controller: DirectAttentionController | None,
    cfg_scale: float,
    guidance_start: int,
    guidance_end: int,
    stop_after_step: int | None = None,
) -> tuple[np.ndarray | None, list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    base.freeze_pipe(pipe)
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    audit: list[dict[str, Any]] = []
    captures: dict[int, dict[str, np.ndarray]] = {}
    for step, scheduler_timestep in enumerate(pipe.scheduler.timesteps):
        if (
            scheduler_timestep.item() < 0.875 * 1000
            and getattr(pipe, "dit2", None) is not None
            and models["dit"] is not pipe.dit2
        ):
            pipe.load_models_to_device(pipe.in_iteration_models_2)
            models["dit"] = pipe.dit2
            models["vace"] = pipe.vace2
        timestep = scheduler_timestep.unsqueeze(0).to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        latents = inputs_shared["latents"].detach()
        guided = controller is not None and guidance_start <= step <= guidance_end
        if guided:
            controller.begin_step(step, base.active_grid(latents, models["dit"]))
            controller.active = True
        try:
            with torch.no_grad():
                if controller is not None:
                    controller.set_branch("positive")
                noise_pos = base.model_forward(
                    pipe, models, inputs_shared, inputs_posi, timestep, False
                )
                if controller is not None:
                    controller.set_branch("negative")
                noise_neg = base.model_forward(
                    pipe, models, inputs_shared, inputs_nega, timestep, False
                )
        finally:
            if controller is not None:
                controller.active = False
        if guided:
            row, capture = controller.end_step()
            if capture is not None:
                captures[step + 1] = capture
        else:
            row = {
                "step": int(step),
                "step_1based": int(step + 1),
                "direction": None,
                "attention_tv_budget": 0.0,
                "mean_actual_tv": 0.0,
                "mean_target_ce_before": None,
                "mean_target_ce_after": None,
                "mean_target_ce_change": None,
                "mean_av_delta_rms": 0.0,
                "latent_update_rms": 0.0,
                "model_parameters_updated": False,
            }
        with torch.no_grad():
            noise_cfg = noise_neg + float(cfg_scale) * (noise_pos - noise_neg)
            inputs_shared["latents"] = pipe.scheduler.step(
                noise_cfg, scheduler_timestep, latents
            )
            base.restore_context(inputs_shared, spec.context_latent_frames)
        audit.append(
            {
                **row,
                "timestep": float(scheduler_timestep.detach().cpu()),
                "sigma": float(pipe.scheduler.sigmas[step]),
                "guided": bool(guided),
            }
        )
        print(
            f"[direct-attn] backend={spec.name} step={step:02d} guided={guided} "
            f"tv={row['mean_actual_tv']:.5f} ce_delta={row['mean_target_ce_change']}",
            flush=True,
        )
        del noise_pos, noise_neg, noise_cfg
        if stop_after_step is not None and step >= stop_after_step:
            return None, audit, captures

    with torch.no_grad():
        for unit in pipe.post_units:
            inputs_shared, _, _ = pipe.unit_runner(
                unit, pipe, inputs_shared, inputs_posi, inputs_nega
            )
        pipe.load_models_to_device(["vae"])
        decoded = pipe.vae.decode(
            inputs_shared["latents"],
            device=pipe.device,
            tiled=True,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )
        video = pipe.vae_output_to_video(decoded)
        pipe.load_models_to_device([])
    frames = np.stack(
        [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in video]
    )
    return frames, audit, captures


def write_capture(
    output: Path,
    step: int,
    capture: dict[str, np.ndarray],
    audit: dict[str, Any],
    tracks: np.ndarray,
    visibility: np.ndarray,
) -> None:
    directory = output / "attention_audit" / f"step_{step:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "raw_attention_maps.tmp.npz"
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            pre_heatmap=capture["pre_heatmap"].astype(np.float16),
            post_heatmap=capture["post_heatmap"].astype(np.float16),
            tracks_tn2=tracks.astype(np.float32),
            visibility_tn=visibility.astype(np.uint8),
            source_frame_indices=base.LATENT_ANCHORS,
        )
    temporary.replace(directory / "raw_attention_maps.npz")
    base.atomic_json(
        directory / "metrics.json",
        {
            "protocol": PROTOCOL,
            "normalization": "global softmax over all 13*H*W Keys",
            "intervention": "A' = A + lambda*(T-A), lambda chosen for matched row TV",
            "summary": audit,
        },
    )
    base.atomic_json(directory / "complete.json", {"step": int(step)})


def generation_dir(
    root: Path, spec: base.BackendSpec, case: str, seed: int, variant: str
) -> Path:
    return root / spec.name / "generations" / case / f"seed_{seed:05d}" / variant


def run_sanity(
    args: argparse.Namespace,
    spec: base.BackendSpec,
    pipe: Any,
    case_path: Path,
    target_map: dict[str, tuple[str, ...]],
    groups: dict[str, list[dict[str, Any]]],
) -> None:
    tube = base.legacy.load_frozen_tube(args.tube_root, case_path.stem)
    payload = base.legacy.load_payload(case_path)
    target = base.legacy.selected_target_specs(tube, target_map[tube.case])[0]
    group_name = next(iter(groups))
    direction = args.directions[0]
    controller, geometry = build_controller(
        pipe, spec, tube, target, groups[group_name], direction, args
    )
    controller.install()
    try:
        inputs = base.prepare_backend_inputs(
            pipe, spec, payload, tube, args.seed, args.cfg_scale
        )
        _, audit, _ = run_denoising(
            pipe, spec, *inputs, controller, args.cfg_scale, 0, 0, stop_after_step=0
        )
    finally:
        controller.remove()
    row = audit[0]
    report = {
        "protocol": PROTOCOL,
        "backend": spec.name,
        "case": tube.case,
        "target": target.name,
        "head_group": group_name,
        "direction": direction,
        "geometry": geometry,
        "step": row,
        "passed": bool(
            abs(row["mean_actual_tv"] - args.attention_tv_budget) < 1.0e-4
            and row["mean_target_ce_change"] < 0
            and row["positive_head_events"] == 100
            and row["negative_head_events"] == 100
            and row["latent_update_rms"] == 0.0
        ),
    }
    base.atomic_json(args.output_root / spec.name / "sanity.json", report)
    if not report["passed"]:
        raise RuntimeError(f"direct attention sanity failed: {report}")


def run_generate(
    args: argparse.Namespace,
    spec: base.BackendSpec,
    pipe: Any,
    case_paths: list[Path],
    target_map: dict[str, tuple[str, ...]],
    groups: dict[str, list[dict[str, Any]]],
    runtime_info: dict[str, Any],
) -> None:
    for case_path in case_paths:
        tube = base.legacy.load_frozen_tube(args.tube_root, case_path.stem)
        payload = base.legacy.load_payload(case_path)
        targets = base.legacy.selected_target_specs(tube, target_map[tube.case])
        tasks: list[tuple[str, Any | None, list[dict[str, Any]] | None, str | None]] = []
        if not args.no_baseline:
            tasks.append(("baseline", None, None, None))
        for target in targets:
            for direction in args.directions:
                tasks.extend(
                    (group_name, target, entries, direction)
                    for group_name, entries in groups.items()
                )
        for group_name, target, entries, direction in tasks:
            variant = (
                "baseline"
                if target is None
                else f"{group_name}__{direction}__{target.name}"
            )
            output = generation_dir(args.output_root, spec, tube.case, args.seed, variant)
            required = [output / "generated.mp4", output / "manifest.json", output / "complete.json"]
            if target is not None:
                required.extend(
                    output / "attention_audit" / f"step_{step:02d}" / "complete.json"
                    for step in args.attention_capture_steps
                )
            if all(path.is_file() for path in required) and not args.overwrite:
                print(f"[generate] skip {spec.name}/{tube.case}/{variant}", flush=True)
                continue
            output.mkdir(parents=True, exist_ok=True)
            (output / "complete.json").unlink(missing_ok=True)
            controller = None
            geometry = None
            if target is not None and entries is not None and direction is not None:
                controller, geometry = build_controller(
                    pipe, spec, tube, target, entries, direction, args
                )
                controller.install()
            try:
                inputs = base.prepare_backend_inputs(
                    pipe, spec, payload, tube, args.seed, args.cfg_scale
                )
                frames, audit, captures = run_denoising(
                    pipe,
                    spec,
                    *inputs,
                    controller,
                    args.cfg_scale,
                    args.guidance_start,
                    args.guidance_end,
                )
            finally:
                if controller is not None:
                    controller.remove()
            if frames is None:
                raise RuntimeError("direct attention generation returned no frames")
            temporary = output / "generated.tmp.mp4"
            base.save_video_np(frames, temporary, fps=30)
            temporary.replace(output / "generated.mp4")
            if target is not None:
                tracks, visibility, _ = base.target_point_arrays(tube, target, spec)
                missing = sorted(set(args.attention_capture_steps) - set(captures))
                if missing:
                    raise RuntimeError(f"missing direct attention captures: {missing}")
                for capture_step in args.attention_capture_steps:
                    write_capture(
                        output,
                        capture_step,
                        captures[capture_step],
                        audit[capture_step - 1],
                        tracks,
                        visibility,
                    )
            base.atomic_json(
                output / "manifest.json",
                {
                    "protocol": PROTOCOL,
                    "backend": base.serializable(spec.__dict__),
                    "case": tube.case,
                    "seed": int(args.seed),
                    "variant": variant,
                    "target": None if target is None else target.name,
                    "head_group": None if target is None else group_name,
                    "direction": direction,
                    "selected_heads": [] if entries is None else entries,
                    "source_json": str(case_path),
                    "source_video": str(tube.source_video),
                    "checkpoint": None if spec.name == "firstframe_ti2v" else str(args.checkpoint),
                    "runtime_info": runtime_info,
                    "model_parameters_updated": False,
                    "latent_update_rms": 0.0,
                    "direct_attention": {
                        "operator": "O'_q = O_q + (A'_q-A_q)V",
                        "probability_update": "A' = A + lambda*(T-A)",
                        "attention_tv_budget": float(args.attention_tv_budget),
                        "cfg_branches": ["positive", "negative"],
                        "guided_step_range_inclusive": [
                            int(args.guidance_start), int(args.guidance_end)
                        ],
                        "sigma_tokens": float(args.gaussian_sigma_tokens),
                    },
                    "geometry": geometry,
                    "audit": audit,
                    "attention_capture_steps": list(args.attention_capture_steps),
                },
            )
            base.atomic_json(output / "complete.json", {"variant": variant})
            print(f"[generate] complete {spec.name}/{tube.case}/{variant}", flush=True)
            del frames, captures
            gc.collect()
            torch.cuda.empty_cache()


def task_manifest(
    args: argparse.Namespace,
    spec: base.BackendSpec,
    case_paths: list[Path],
    target_map: dict[str, tuple[str, ...]],
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    target_count = sum(len(target_map[path.stem]) for path in case_paths)
    return {
        "protocol": PROTOCOL,
        "backend": base.serializable(spec.__dict__),
        "cases": [{"case": path.stem, "targets": list(target_map[path.stem])} for path in case_paths],
        "seed": int(args.seed),
        "directions": list(args.directions),
        "head_groups": list(groups),
        "baseline_videos": 0 if args.no_baseline else len(case_paths),
        "guided_videos": target_count * len(args.directions) * len(groups),
        "attention_tv_budget_per_selected_row": float(args.attention_tv_budget),
        "latent_update_rms": 0.0,
        "cfg_branches_intervened": ["positive", "negative"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--backend", choices=tuple(base.BACKENDS), default="firstframe_ti2v")
    parser.add_argument("--stage", choices=("dry-run", "sanity", "generate", "evaluate", "all"), default="all")
    parser.add_argument("--input-list", type=Path, default=base.DEFAULT_INPUT_LIST)
    parser.add_argument("--head-ranking", type=Path, default=base.DEFAULT_RANKING)
    parser.add_argument("--head-scopes", type=Path, default=base.DEFAULT_SCOPES)
    parser.add_argument("--tube-root", type=Path, default=base.DEFAULT_TUBE_ROOT)
    parser.add_argument("--target-map", type=Path, default=base.DEFAULT_TARGET_MAP)
    parser.add_argument("--checkpoint", type=Path, default=base.DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=47326)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--head-groups", nargs="+", choices=("top100", "bottom100", "random100"), default=("top100", "bottom100", "random100"))
    parser.add_argument("--directions", nargs="+", choices=VALID_DIRECTIONS, default=VALID_DIRECTIONS)
    parser.add_argument("--case-keys", nargs="*", default=(FOCAL_CASE,))
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--attention-tv-budget", type=float, default=0.10)
    parser.add_argument("--guidance-start", type=int, default=0)
    parser.add_argument("--guidance-end", type=int, default=39)
    parser.add_argument("--gaussian-sigma-tokens", type=float, default=1.5)
    parser.add_argument("--attention-capture-steps", nargs="*", type=int, default=base.DEFAULT_ATTENTION_CAPTURE_STEPS)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.device in {"cuda:4", "4"}:
        raise ValueError("GPU 4 is prohibited")
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0,num-workers)")
    if not 0 <= args.guidance_start <= args.guidance_end < 40:
        raise ValueError("guidance range must lie in [0,39]")
    if not 0.0 < args.attention_tv_budget <= 1.0:
        raise ValueError("attention TV budget must lie in (0,1]")
    if args.gaussian_sigma_tokens <= 0:
        raise ValueError("Gaussian sigma must be positive")
    if not args.attention_capture_steps:
        raise ValueError("at least one attention capture step is required")
    if min(args.attention_capture_steps) < 1 or max(args.attention_capture_steps) > 40:
        raise ValueError("attention capture steps must lie in 1..40")
    for path in (args.input_list, args.head_ranking, args.head_scopes, args.target_map):
        if not path.expanduser().is_file():
            raise FileNotFoundError(path)
    if args.backend == "context8_v2v" and not (args.checkpoint / "checkpoint.safetensors").is_file():
        raise FileNotFoundError(args.checkpoint / "checkpoint.safetensors")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args = argparse.Namespace(
        **{
            **vars(args),
            "input_list": args.input_list.expanduser().resolve(),
            "head_ranking": args.head_ranking.expanduser().resolve(),
            "head_scopes": args.head_scopes.expanduser().resolve(),
            "tube_root": args.tube_root.expanduser().resolve(),
            "target_map": args.target_map.expanduser().resolve(),
            "checkpoint": args.checkpoint.expanduser().resolve(),
            "output_root": args.output_root.expanduser().resolve(),
            "attention_capture_steps": tuple(sorted(set(args.attention_capture_steps))),
            "directions": tuple(dict.fromkeys(args.directions)),
        }
    )
    spec = base.BACKENDS[args.backend]
    case_paths, target_map = base.load_cases(args)
    groups = base.load_head_groups(args.head_ranking, args.head_scopes, args.head_groups)
    manifest = task_manifest(args, spec, case_paths, target_map, groups)
    args.output_root.mkdir(parents=True, exist_ok=True)
    base.atomic_json(args.output_root / spec.name / "task_manifest.json", manifest)
    base.atomic_json(
        args.output_root / spec.name / "run_config.json",
        {"arguments": base.serializable(vars(args)), **manifest},
    )
    if args.stage == "dry-run":
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if args.stage == "evaluate":
        base.run_evaluate(args, spec, case_paths, target_map)
        return

    error_path = args.output_root / spec.name / "run_error.txt"
    error_path.unlink(missing_ok=True)
    owner = None
    try:
        owner, pipe, runtime_info = base.build_backend(args, spec)
        if args.stage in {"sanity", "all"}:
            run_sanity(args, spec, pipe, case_paths[0], target_map, groups)
        if args.stage in {"generate", "all"}:
            run_generate(args, spec, pipe, case_paths, target_map, groups, runtime_info)
    except Exception:
        error = traceback.format_exc()
        error_path.write_text(error, encoding="utf-8")
        print(error, flush=True)
        raise
    finally:
        base.release_backend(owner)
    if args.stage == "all":
        base.run_evaluate(args, spec, case_paths, target_map)


if __name__ == "__main__":
    main()
