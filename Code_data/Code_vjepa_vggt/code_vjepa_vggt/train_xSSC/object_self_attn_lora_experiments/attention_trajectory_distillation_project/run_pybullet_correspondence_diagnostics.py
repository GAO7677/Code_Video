#!/usr/bin/env python3
"""Run noise-gated point-correspondence diagnostics on PyBullet train cases."""

from __future__ import annotations

import argparse
import gc
import html
import json
from pathlib import Path
import sys
from types import MethodType
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
DIFFTRACK_ROOT = Path("/home/gaoya/Code_Video/DiffTrack-main")
COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
for _path in (EXPERIMENT_ROOT, CODE_ROOT, DIFFTRACK_ROOT, COTRACKER_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from code_vjepa_vggt.train0419_reference.batch_eval_lora import build_pipeline
from code_vjepa_vggt.utils.object_priors import sample_points_from_mask
from AAA_my_test import analyze_wan_gt_toy_worker as wan_tools

import train_xssc_object_self_attn_lora as train_core
from frozen_motion_probe import load_pck_head_weights, ordered_head_pairs
from noise_gated_correspondence import (
    coordinate_loss_sensitivity,
    cross_frame_point_terms,
    noise_reliability_gate,
    points_to_token_coordinates,
)
from run_training_case_diagnostics import add_label, atomic_json, write_video


DEFAULT_SOURCE_CACHE = Path(
    "/data/gaoya/agent-data/cache/frozen_motion_probe_training_diagnostics"
)
DEFAULT_SOURCE_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/frozen_motion_probe_training_diagnostics"
)
DEFAULT_CACHE_ROOT = Path(
    "/data/gaoya/agent-data/cache/noise_gated_correspondence_diagnostics"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/noise_gated_correspondence_diagnostics"
)
DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_COTRACKER_CHECKPOINT = Path(
    "/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth"
)
DEFAULT_HEAD_CONFIG = (
    EXPERIMENT_ROOT
    / "configs/physiciq67_pck32_s039_latest3350_top100_heads.json"
)
DEFAULT_HEAD_SUBSET = "T_physiciq67_pck32_s039_latest3350_top100"
DEFAULT_HEAD_SUBTYPE = "physiciq67_pck32_s039_latest3350"
DEFAULT_TIMESTEPS = (100.0, 300.0, 500.0, 700.0, 900.0)
ANCHOR_PIXEL_FRAMES = np.arange(0, 49, 4, dtype=np.int64)
TOKEN_HW = (16, 28)
PALETTE_RGB = np.asarray(
    [
        [255, 222, 51],
        [38, 220, 255],
        [255, 84, 114],
        [54, 232, 138],
        [240, 132, 42],
        [196, 112, 255],
        [255, 255, 255],
        [44, 146, 255],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Noise-gated Gaussian point-correspondence case diagnostic."
    )
    parser.add_argument("mode", choices=("prepare-tracks", "forward", "render", "all"))
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--source-output", type=Path, default=DEFAULT_SOURCE_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument(
        "--cotracker-checkpoint", type=Path, default=DEFAULT_COTRACKER_CHECKPOINT
    )
    parser.add_argument("--head-config", type=Path, default=DEFAULT_HEAD_CONFIG)
    parser.add_argument("--head-subset", default=DEFAULT_HEAD_SUBSET)
    parser.add_argument("--head-subtype", default=DEFAULT_HEAD_SUBTYPE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-timesteps", type=float, nargs="+", default=DEFAULT_TIMESTEPS)
    parser.add_argument("--num-points", type=int, default=8)
    parser.add_argument("--source-pixel-frame", type=int, default=4)
    parser.add_argument("--source-latent-frame", type=int, default=1)
    parser.add_argument("--label-sigma-tokens", type=float, default=1.0)
    parser.add_argument("--gate-gamma", type=float, default=1.0)
    parser.add_argument("--gate-cutoff", type=float, default=0.75)
    parser.add_argument("--coordinate-huber-beta", type=float, default=0.5)
    parser.add_argument("--coordinate-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=4200)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if str(args.device).startswith("cuda:4"):
        raise ValueError("GPU 4 is prohibited by workspace rules")
    if int(args.num_points) <= 0:
        raise ValueError("num-points must be positive")
    if int(args.source_pixel_frame) != 4 or int(args.source_latent_frame) != 1:
        raise ValueError("this diagnostic is fixed to F04 / latent-1")
    if float(args.label_sigma_tokens) <= 0.0:
        raise ValueError("label-sigma-tokens must be positive")
    if float(args.coordinate_huber_beta) <= 0.0:
        raise ValueError("coordinate-huber-beta must be positive")
    if float(args.coordinate_weight) < 0.0:
        raise ValueError("coordinate-weight must be non-negative")
    for timestep in args.training_timesteps:
        if float(timestep) not in DEFAULT_TIMESTEPS:
            raise ValueError(
                f"timestep={timestep} has no controlled cached x_t; expected {DEFAULT_TIMESTEPS}"
            )
    manifest = args.source_cache.resolve() / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing source case manifest: {manifest}")


def stage_id(timestep: float) -> str:
    return f"train_{int(round(float(timestep))):04d}"


def load_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return json.loads(
        (args.source_cache.resolve() / "manifest.json").read_text(encoding="utf-8")
    )


def run_cotracker_from_anchor(
    model: torch.nn.Module,
    frames_thwc: np.ndarray,
    points_n2: np.ndarray,
    *,
    anchor_frame: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    input_height, input_width = 384, 512
    native_height, native_width = frames_thwc.shape[1:3]
    video = torch.from_numpy(frames_thwc).float().div(255.0).permute(0, 3, 1, 2)
    video = torch.nn.functional.interpolate(
        video,
        size=(input_height, input_width),
        mode="bilinear",
        align_corners=True,
    ).unsqueeze(0).to(device)
    query = torch.from_numpy(points_n2).float().to(device)
    query[:, 0] *= input_width / native_width
    query[:, 1] *= input_height / native_height
    frame_ids = torch.full(
        (len(query), 1), float(anchor_frame), device=device, dtype=query.dtype
    )
    queries = torch.cat((frame_ids, query), dim=-1).unsqueeze(0)
    with torch.inference_mode():
        tracks, visibility = model(
            video,
            queries=queries,
            backward_tracking=True,
        )
    tracks_np = tracks[0].float().cpu().numpy()
    tracks_np[..., 0] *= max(native_width - 1, 1) / max(input_width - 1, 1)
    tracks_np[..., 1] *= max(native_height - 1, 1) / max(input_height - 1, 1)
    visible_np = visibility[0].float().cpu().numpy() > 0.5
    in_bounds = (
        (tracks_np[..., 0] >= 0.0)
        & (tracks_np[..., 0] <= native_width - 1)
        & (tracks_np[..., 1] >= 0.0)
        & (tracks_np[..., 1] <= native_height - 1)
    )
    return tracks_np.astype(np.float32), np.logical_and(visible_np, in_bounds)


def run_prepare_tracks(args: argparse.Namespace) -> None:
    from cotracker.predictor import CoTrackerPredictor

    manifest = load_manifest(args)
    cache_root = args.cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    print("[tracks] loading CoTracker", flush=True)
    model = CoTrackerPredictor(
        checkpoint=str(args.cotracker_checkpoint.resolve()),
        offline=True,
        v2=False,
        window_len=60,
    ).to(args.device).eval().requires_grad_(False)
    summaries = []
    try:
        for index, case in enumerate(manifest["cases"], start=1):
            case_key = str(case["case_key"])
            source_case = Path(case["cache_dir"])
            case_cache = cache_root / "cases" / case_key
            complete = case_cache / "tracks_complete.json"
            if complete.is_file() and not args.overwrite:
                summaries.append(json.loads(complete.read_text(encoding="utf-8")))
                print(f"[tracks {index}/3] skip {case_key}", flush=True)
                continue
            case_cache.mkdir(parents=True, exist_ok=True)
            with np.load(source_case / "source_frames.npz") as arrays:
                frames = arrays["frames"].astype(np.uint8)
            with np.load(source_case / "sam2_masks.npz") as arrays:
                anchor_mask = arrays["selected_identity_mask"].astype(np.uint8)
            points = sample_points_from_mask(
                anchor_mask,
                int(args.num_points),
                avoid_edges=True,
            )
            if points.shape != (int(args.num_points), 2):
                raise RuntimeError(
                    f"{case_key}: sampled points have shape {points.shape}"
                )
            print(f"[tracks {index}/3] {case_key}: {len(points)} points", flush=True)
            tracks, visibility = run_cotracker_from_anchor(
                model,
                frames,
                points,
                anchor_frame=int(args.source_pixel_frame),
                device=str(args.device),
            )
            anchor_tracks = tracks[ANCHOR_PIXEL_FRAMES]
            anchor_visibility = visibility[ANCHOR_PIXEL_FRAMES]
            if not bool(anchor_visibility[int(args.source_latent_frame)].all()):
                raise RuntimeError(f"{case_key}: source points are not all visible at F04")
            np.savez_compressed(
                case_cache / "point_tracks.npz",
                query_points_n2=points,
                tracks_tn2=anchor_tracks,
                visibility_tn=anchor_visibility.astype(np.uint8),
                anchor_pixel_frames=ANCHOR_PIXEL_FRAMES,
                pixel_hw=np.asarray(frames.shape[1:3], dtype=np.int32),
            )
            summary = {
                "case_key": case_key,
                "supervision": "CoTracker pseudo-GT initialized from the F04 SAM2 identity mask",
                "point_count": int(len(points)),
                "anchor_pixel_frames": ANCHOR_PIXEL_FRAMES.tolist(),
                "source_pixel_frame": int(args.source_pixel_frame),
                "source_latent_frame": int(args.source_latent_frame),
                "visibility_rate": float(anchor_visibility.mean()),
                "future_pair_count": int(
                    anchor_visibility[int(args.source_latent_frame) + 1 :].sum()
                ),
            }
            atomic_json(complete, summary)
            summaries.append(summary)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(
        cache_root / "tracks_status.json",
        {"state": "complete", "case_count": len(summaries), "cases": summaries},
    )


class PCKCorrespondenceCollector:
    def __init__(
        self,
        dit: torch.nn.Module,
        selected_heads_by_block: Mapping[int, Sequence[int]],
        pck_weights: torch.Tensor,
        *,
        point_coordinates_tn2: torch.Tensor,
        point_visibility_tn: torch.Tensor,
        source_frame: int,
        label_sigma_tokens: float,
        coordinate_huber_beta: float,
    ) -> None:
        self.dit = dit
        self.selected_heads_by_block = {
            int(block): tuple(sorted(map(int, heads)))
            for block, heads in selected_heads_by_block.items()
        }
        pairs = ordered_head_pairs(self.selected_heads_by_block)
        if len(pairs) != int(pck_weights.numel()):
            raise ValueError(f"head/weight mismatch: {len(pairs)}/{pck_weights.numel()}")
        weights_by_pair = {
            pair: float(weight)
            for pair, weight in zip(pairs, pck_weights.detach().cpu().tolist())
        }
        self.weights_by_block = {
            block: torch.tensor(
                [weights_by_pair[(block, head)] for head in heads],
                dtype=torch.float32,
            )
            for block, heads in self.selected_heads_by_block.items()
        }
        self.point_coordinates_tn2 = point_coordinates_tn2
        self.point_visibility_tn = point_visibility_tn
        self.source_frame = int(source_frame)
        self.label_sigma_tokens = float(label_sigma_tokens)
        self.coordinate_huber_beta = float(coordinate_huber_beta)
        self.attention: torch.Tensor | None = None
        self.ce_contribution: torch.Tensor | None = None
        self.target: torch.Tensor | None = None
        self.valid: torch.Tensor | None = None
        self.head_count = 0
        self.originals: list[tuple[torch.nn.Module, Any]] = []

    def _capture(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        block_id: int,
    ) -> None:
        heads = self.selected_heads_by_block[block_id]
        num_heads = int(q.shape[-1] // 128)
        if num_heads != 24 or q.shape[1] != 13 * TOKEN_HW[0] * TOKEN_HW[1]:
            raise RuntimeError(
                f"unexpected Wan Q/K geometry: q={tuple(q.shape)}, heads={num_heads}"
            )
        selected = torch.as_tensor(heads, device=q.device, dtype=torch.long)
        q_heads = q.reshape(q.shape[0], q.shape[1], num_heads, 128).index_select(2, selected)
        k_heads = k.reshape(k.shape[0], k.shape[1], num_heads, 128).index_select(2, selected)
        terms = cross_frame_point_terms(
            q_heads,
            k_heads,
            point_coordinates_tn2=self.point_coordinates_tn2,
            point_visibility_tn=self.point_visibility_tn,
            token_hw=TOKEN_HW,
            source_frame=self.source_frame,
            sigma_tokens=self.label_sigma_tokens,
            coordinate_huber_beta=self.coordinate_huber_beta,
            future_only=True,
        )
        weights = self.weights_by_block[block_id].to(
            device=q.device, dtype=terms["attention"].dtype
        )
        weighted_attention = (
            terms["attention"] * weights.reshape(1, 1, -1, 1, 1)
        ).sum(dim=2)
        weighted_ce = (
            terms["ce_contribution"] * weights.reshape(1, 1, -1, 1, 1)
        ).sum(dim=2)
        self.attention = (
            weighted_attention
            if self.attention is None
            else self.attention + weighted_attention
        )
        self.ce_contribution = (
            weighted_ce
            if self.ce_contribution is None
            else self.ce_contribution + weighted_ce
        )
        self.target = terms["target"]
        self.valid = terms["valid"]
        self.head_count += len(heads)

    def install(self) -> None:
        for block_id in self.selected_heads_by_block:
            attention = self.dit.blocks[block_id].self_attn.attn
            original = attention.forward

            def wrapped(
                module,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                *,
                _block_id: int = block_id,
                _original=original,
            ):
                self._capture(q, k, _block_id)
                return _original(q, k, v)

            self.originals.append((attention, original))
            attention.forward = MethodType(wrapped, attention)

    def remove(self) -> None:
        for attention, original in self.originals:
            attention.forward = original
        self.originals.clear()

    def finalize(
        self,
        *,
        scheduler_sigma: float,
        gate_gamma: float,
        gate_cutoff: float,
        coordinate_weight: float,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self.head_count != 100:
            raise RuntimeError(f"captured {self.head_count} heads, expected 100")
        if self.attention is None or self.ce_contribution is None:
            raise RuntimeError("collector captured no correspondence maps")
        attention = self.attention[0].float()
        ce_contribution = self.ce_contribution[0].float()
        target = self.target.float()
        valid = self.valid.bool()
        probability_error = float((attention.sum(dim=-1) - 1.0).abs().max().item())
        if probability_error > 2.0e-5:
            raise RuntimeError(f"PCK aggregate probability error: {probability_error}")
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(TOKEN_HW[0], device=attention.device, dtype=torch.float32),
                torch.arange(TOKEN_HW[1], device=attention.device, dtype=torch.float32),
                indexing="ij",
            )[::-1],
            dim=-1,
        ).reshape(-1, 2)
        predicted = torch.einsum("tns,sc->tnc", attention, grid)
        coordinate_target = self.point_coordinates_tn2.to(predicted)
        coordinate_huber = torch.nn.functional.smooth_l1_loss(
            predicted,
            coordinate_target,
            beta=self.coordinate_huber_beta,
            reduction="none",
        ).mean(dim=-1)
        ce = ce_contribution.sum(dim=-1)
        raw_ce = ce[valid].mean()
        raw_coordinate = coordinate_huber[valid].mean()
        gate = noise_reliability_gate(
            float(scheduler_sigma),
            gamma=float(gate_gamma),
            cutoff=float(gate_cutoff),
        ).to(attention)
        sensitivity = coordinate_loss_sensitivity(
            attention,
            predicted,
            coordinate_target,
            token_hw=TOKEN_HW,
            beta=self.coordinate_huber_beta,
        )
        top1 = attention.argmax(dim=-1)
        top1_coordinates = torch.stack(
            ((top1 % TOKEN_HW[1]).float(), (top1 // TOKEN_HW[1]).float()),
            dim=-1,
        )
        top1_error = torch.linalg.vector_norm(top1_coordinates - coordinate_target, dim=-1)
        soft_error = torch.linalg.vector_norm(predicted - coordinate_target, dim=-1)
        total = gate * (raw_ce + float(coordinate_weight) * raw_coordinate)
        maps = {
            "attention_tns": attention.cpu().numpy(),
            "target_tns": target.cpu().numpy(),
            "ce_contribution_tns": ce_contribution.cpu().numpy(),
            "coordinate_sensitivity_tns": sensitivity.cpu().numpy(),
            "predicted_coordinates_tn2": predicted.cpu().numpy(),
            "target_coordinates_tn2": coordinate_target.cpu().numpy(),
            "visibility_tn": self.point_visibility_tn.cpu().numpy().astype(np.uint8),
            "valid_tn": valid.cpu().numpy().astype(np.uint8),
            "ce_tn": ce.cpu().numpy(),
            "coordinate_huber_tn": coordinate_huber.cpu().numpy(),
            "top1_error_tn": top1_error.cpu().numpy(),
            "softargmax_error_tn": soft_error.cpu().numpy(),
        }
        metrics = {
            "scheduler_sigma": float(scheduler_sigma),
            "noise_gate": float(gate.item()),
            "raw_soft_ce": float(raw_ce.item()),
            "raw_coordinate_huber": float(raw_coordinate.item()),
            "gated_soft_ce": float((gate * raw_ce).item()),
            "gated_coordinate_huber": float((gate * raw_coordinate).item()),
            "coordinate_weight": float(coordinate_weight),
            "gated_total": float(total.item()),
            "valid_point_frame_pairs": int(valid.sum().item()),
            "mean_top1_error_tokens": float(top1_error[valid].mean().item()),
            "mean_softargmax_error_tokens": float(soft_error[valid].mean().item()),
            "pck_at_1_token": float((top1_error[valid] <= 1.0).float().mean().item()),
            "pck_at_2_tokens": float((top1_error[valid] <= 2.0).float().mean().item()),
            "aggregate_probability_max_error": probability_error,
        }
        return maps, metrics


def run_forward(args: argparse.Namespace) -> None:
    manifest = load_manifest(args)
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    print("[forward] loading official Wan2.2-TI2V-5B baseline", flush=True)
    pipe = build_pipeline(args.wan_root.resolve(), str(device), None)
    pipe.dit.requires_grad_(False).eval()
    selected_heads, head_metadata = train_core.load_head_selection_config(
        args.head_config,
        expected_subset_id=args.head_subset,
        expected_role="T",
        expected_feature_subtype=args.head_subtype,
        expected_num_heads=100,
        num_blocks=30,
        num_heads=24,
    )
    pck_weights, pck_audit = load_pck_head_weights(head_metadata, selected_heads)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    try:
        for case_index, case in enumerate(manifest["cases"], start=1):
            case_key = str(case["case_key"])
            case_output = output_root / "cases" / case_key
            case_output.mkdir(parents=True, exist_ok=True)
            track_path = args.cache_root.resolve() / "cases" / case_key / "point_tracks.npz"
            if not track_path.is_file():
                raise FileNotFoundError(f"run prepare-tracks first: {track_path}")
            with np.load(track_path) as arrays:
                tracks_tn2 = arrays["tracks_tn2"].astype(np.float32)
                visibility_tn = arrays["visibility_tn"].astype(bool)
                pixel_hw = tuple(map(int, arrays["pixel_hw"].tolist()))
            point_coordinates = points_to_token_coordinates(
                torch.from_numpy(tracks_tn2),
                pixel_hw=pixel_hw,
                token_hw=TOKEN_HW,
            ).to(device)
            point_visibility = torch.from_numpy(visibility_tn).to(device)
            source_frames = wan_tools.load_video_prefix(
                Path(case["source_video"]), 49, 512, 896, "cache"
            )
            context_frames = source_frames[:8]
            inputs_shared, inputs_positive = wan_tools.prepare_conditioning(
                pipe,
                prompt=case["caption"],
                context_video=context_frames,
                height=512,
                width=896,
                num_frames=49,
                sampling_steps=40,
                sigma_shift=5.0,
                cfg_scale=5.0,
                seed=int(args.seed) + case_index,
            )
            captured_inputs = dict(inputs_shared)
            captured_inputs.update(inputs_positive)
            pipe.scheduler.set_timesteps(1000, training=True)
            pipe.load_models_to_device(pipe.in_iteration_models)
            models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
            stage_metrics = []
            for timestep_value in args.training_timesteps:
                sid = stage_id(timestep_value)
                stage_output = case_output / sid
                complete = stage_output / "forward_complete.json"
                if complete.is_file() and not args.overwrite:
                    stage_metrics.append(
                        json.loads((stage_output / "metrics.json").read_text(encoding="utf-8"))
                    )
                    print(f"[forward {case_index}/3] skip {case_key} {sid}", flush=True)
                    continue
                stage_output.mkdir(parents=True, exist_ok=True)
                source_stage = (
                    args.source_output.resolve()
                    / "cases"
                    / case_key
                    / "noise_sweep"
                    / "stages"
                    / sid
                )
                latent_state = torch.load(
                    source_stage / "latents.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                source_stage_metrics = json.loads(
                    (source_stage / "metrics.json").read_text(encoding="utf-8")
                )
                latent_xt = latent_state["training_xt"].to(device=device, dtype=pipe.torch_dtype)
                timestep = torch.full(
                    (1,),
                    float(timestep_value),
                    device=device,
                    dtype=pipe.torch_dtype,
                )
                sigma = float(source_stage_metrics["scheduler_sigma"])
                collector = PCKCorrespondenceCollector(
                    pipe.dit,
                    selected_heads,
                    pck_weights,
                    point_coordinates_tn2=point_coordinates,
                    point_visibility_tn=point_visibility,
                    source_frame=int(args.source_latent_frame),
                    label_sigma_tokens=float(args.label_sigma_tokens),
                    coordinate_huber_beta=float(args.coordinate_huber_beta),
                )
                print(
                    f"[forward {case_index}/3] {case_key} {sid} "
                    f"sigma={sigma:.4f}",
                    flush=True,
                )
                torch.cuda.reset_peak_memory_stats(device)
                main_inputs = dict(captured_inputs)
                main_inputs["latents"] = latent_xt
                collector.install()
                try:
                    with torch.no_grad():
                        velocity = pipe.model_fn(
                            **models,
                            **main_inputs,
                            timestep=timestep,
                        )
                finally:
                    collector.remove()
                del velocity, latent_xt, latent_state
                maps, metrics = collector.finalize(
                    scheduler_sigma=sigma,
                    gate_gamma=float(args.gate_gamma),
                    gate_cutoff=float(args.gate_cutoff),
                    coordinate_weight=float(args.coordinate_weight),
                )
                np.savez_compressed(stage_output / "loss_maps.npz", **maps)
                metrics.update(
                    {
                        "schema_version": 1,
                        "case_key": case_key,
                        "family": case["family"],
                        "caption": case["caption"],
                        "dataset_split": "train",
                        "dataset_index": int(case["dataset_index"]),
                        "training_timestep": float(timestep_value),
                        "stage_id": sid,
                        "model": "official Wan2.2-TI2V-5B step-0 baseline forward",
                        "execution_scope": "forward-only loss diagnostic; no optimizer step",
                        "supervision": "CoTracker pseudo-GT initialized from F04 SAM2 identity mask",
                        "source_frame": "L01/F04",
                        "target_frames": "future L02/F08 through L12/F48",
                        "point_count": int(args.num_points),
                        "label_sigma_tokens": float(args.label_sigma_tokens),
                        "gate_gamma": float(args.gate_gamma),
                        "gate_cutoff": float(args.gate_cutoff),
                        "coordinate_huber_beta": float(args.coordinate_huber_beta),
                        "head_weighting": "Top100 normalized pck32",
                        "pck_score_min": float(pck_audit["score_min"]),
                        "pck_score_max": float(pck_audit["score_max"]),
                        "peak_gpu_memory_mib": float(
                            torch.cuda.max_memory_allocated(device) / (1024**2)
                        ),
                    }
                )
                atomic_json(stage_output / "metrics.json", metrics)
                atomic_json(complete, {"state": "complete", "stage_id": sid})
                stage_metrics.append(metrics)
                del collector, maps
                gc.collect()
                torch.cuda.empty_cache()
            case_summary = {
                "case_key": case_key,
                "family": case["family"],
                "caption": case["caption"],
                "stages": stage_metrics,
            }
            atomic_json(case_output / "metrics.json", case_summary)
            summaries.append(case_summary)
    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(
        output_root / "forward_status.json",
        {"state": "complete", "case_count": len(summaries), "cases": summaries},
    )


def token_to_pixel(coordinates_n2: np.ndarray, pixel_hw: tuple[int, int]) -> np.ndarray:
    pixel_h, pixel_w = pixel_hw
    output = np.asarray(coordinates_n2, dtype=np.float32).copy()
    output[..., 0] *= float(pixel_w - 1) / float(TOKEN_HW[1] - 1)
    output[..., 1] *= float(pixel_h - 1) / float(TOKEN_HW[0] - 1)
    return output


def draw_points(
    frame: np.ndarray,
    points_n2: np.ndarray,
    visibility_n: np.ndarray,
    *,
    predicted_n2: np.ndarray | None = None,
) -> np.ndarray:
    output = np.asarray(frame, dtype=np.uint8).copy()
    for index, (point, visible) in enumerate(zip(points_n2, visibility_n)):
        if not bool(visible):
            continue
        x, y = map(lambda value: int(round(float(value))), point)
        color = tuple(map(int, PALETTE_RGB[index % len(PALETTE_RGB)]))
        cv2.circle(output, (x, y), 6, color, -1, cv2.LINE_AA)
        cv2.circle(output, (x, y), 8, (15, 20, 18), 2, cv2.LINE_AA)
        if predicted_n2 is not None:
            px, py = map(
                lambda value: int(round(float(value))), predicted_n2[index]
            )
            cv2.drawMarker(
                output,
                (px, py),
                (255, 255, 255),
                cv2.MARKER_TILTED_CROSS,
                13,
                2,
                cv2.LINE_AA,
            )
    return output


def average_visible_maps(maps_tns: np.ndarray, valid_tn: np.ndarray) -> np.ndarray:
    output = np.zeros((maps_tns.shape[0], maps_tns.shape[2]), dtype=np.float32)
    for time_index in range(maps_tns.shape[0]):
        selected = np.asarray(valid_tn[time_index], dtype=bool)
        if selected.any():
            output[time_index] = maps_tns[time_index, selected].mean(axis=0)
    return output.reshape(maps_tns.shape[0], *TOKEN_HW)


def overlay_heatmap(
    frame: np.ndarray,
    heatmap_hw: np.ndarray,
    *,
    vmax: float,
) -> np.ndarray:
    normalized = np.asarray(heatmap_hw, dtype=np.float32).clip(min=0.0)
    normalized = np.clip(normalized / max(float(vmax), 1.0e-12), 0.0, 1.0)
    resized = cv2.resize(
        normalized,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    ).clip(0.0, 1.0)
    colored_bgr = cv2.applyColorMap(
        np.round(resized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    colored = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    strength = (0.72 * np.power(resized, 0.55))[..., None]
    output = frame.astype(np.float32) * (1.0 - strength) + colored * strength
    return output.round().clip(0, 255).astype(np.uint8)


def resize_panel(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (448, 256), interpolation=cv2.INTER_AREA)


def render_stage_video(
    stage_output: Path,
    frames_thwc: np.ndarray,
    tracks_tn2: np.ndarray,
    visibility_tn: np.ndarray,
    metrics: dict[str, Any],
    *,
    fps: float,
    coordinate_weight: float,
) -> None:
    with np.load(stage_output / "loss_maps.npz") as arrays:
        attention = arrays["attention_tns"].astype(np.float32)
        target = arrays["target_tns"].astype(np.float32)
        ce_contribution = arrays["ce_contribution_tns"].astype(np.float32)
        coordinate_sensitivity_map = arrays["coordinate_sensitivity_tns"].astype(
            np.float32
        )
        predicted_token = arrays["predicted_coordinates_tn2"].astype(np.float32)
        valid = arrays["valid_tn"].astype(bool)
        coordinate_huber = arrays["coordinate_huber_tn"].astype(np.float32)
        top1_error = arrays["top1_error_tn"].astype(np.float32)
    gate = float(metrics["noise_gate"])
    attention_map = average_visible_maps(attention, visibility_tn)
    target_map = average_visible_maps(target, visibility_tn)
    ce_map = average_visible_maps(ce_contribution * gate, valid)
    coordinate_map = average_visible_maps(
        coordinate_sensitivity_map * gate * float(coordinate_weight), valid
    )
    predicted_pixel = token_to_pixel(predicted_token, frames_thwc.shape[1:3])

    def scale(values: np.ndarray, percentile: float = 99.5) -> float:
        positive = values[values > 0]
        return float(np.percentile(positive, percentile)) if positive.size else 1.0

    scales = {
        "attention": scale(attention_map),
        "target": scale(target_map),
        "ce": scale(ce_map),
        "coordinate": scale(coordinate_map),
    }
    rendered = []
    timestep = float(metrics["training_timestep"])
    for latent_index, pixel_index in enumerate(ANCHOR_PIXEL_FRAMES):
        frame = frames_thwc[int(pixel_index)]
        gt_points = tracks_tn2[latent_index]
        visible = visibility_tn[latent_index]
        is_valid_target = latent_index > 1 and bool(valid[latent_index].any())
        trajectory_panel = draw_points(frame, gt_points, visible)
        trajectory_panel = add_label(
            trajectory_panel,
            f"points | L{latent_index:02d}/F{pixel_index:02d} | visible {int(visible.sum())}",
        )
        target_panel = overlay_heatmap(
            frame, target_map[latent_index], vmax=scales["target"]
        )
        target_panel = draw_points(target_panel, gt_points, visible)
        target_panel = add_label(target_panel, "Gaussian soft target")
        attention_panel = overlay_heatmap(
            frame, attention_map[latent_index], vmax=scales["attention"]
        )
        attention_panel = draw_points(
            attention_panel,
            gt_points,
            visible,
            predicted_n2=predicted_pixel[latent_index],
        )
        error_value = (
            float(top1_error[latent_index][valid[latent_index]].mean())
            if is_valid_target
            else float("nan")
        )
        attention_panel = add_label(
            attention_panel,
            f"PCK QK attention | top1 err {error_value:.2f} token",
        )
        ce_panel = overlay_heatmap(frame, ce_map[latent_index], vmax=scales["ce"])
        ce_panel = draw_points(ce_panel, gt_points, visible)
        ce_panel = add_label(
            ce_panel,
            f"gated soft-CE contribution | gate {gate:.3f}",
        )
        coordinate_panel = overlay_heatmap(
            frame,
            coordinate_map[latent_index],
            vmax=scales["coordinate"],
        )
        coordinate_panel = draw_points(
            coordinate_panel,
            gt_points,
            visible,
            predicted_n2=predicted_pixel[latent_index],
        )
        coordinate_value = (
            float(coordinate_huber[latent_index][valid[latent_index]].mean())
            if is_valid_target
            else float("nan")
        )
        coordinate_panel = add_label(
            coordinate_panel,
            f"coord-loss sensitivity | Huber {coordinate_value:.3f}",
        )
        rendered.append(
            np.concatenate(
                [
                    resize_panel(trajectory_panel),
                    resize_panel(target_panel),
                    resize_panel(attention_panel),
                    resize_panel(ce_panel),
                    resize_panel(coordinate_panel),
                ],
                axis=1,
            )
        )
    write_video(
        stage_output / "correspondence_loss_overlay.mp4",
        rendered,
        fps=float(fps),
    )
    atomic_json(
        stage_output / "render_complete.json",
        {"state": "complete", "training_timestep": timestep},
    )


def build_report(output_root: Path, cases: list[dict[str, Any]]) -> None:
    sections = []
    for case in cases:
        rows = []
        videos = []
        for stage in case["stages"]:
            sid = stage["stage_id"]
            case_key = case["case_key"]
            video_path = f"cases/{case_key}/{sid}/correspondence_loss_overlay.mp4"
            rows.append(
                "<tr>"
                f"<td>{stage['training_timestep']:.0f}</td>"
                f"<td>{stage['scheduler_sigma']:.4f}</td>"
                f"<td>{stage['noise_gate']:.4f}</td>"
                f"<td>{stage['raw_soft_ce']:.4f}</td>"
                f"<td>{stage['raw_coordinate_huber']:.4f}</td>"
                f"<td>{stage['gated_total']:.4f}</td>"
                f"<td>{stage['mean_top1_error_tokens']:.3f}</td>"
                f"<td>{stage['pck_at_1_token']:.3f}</td>"
                "</tr>"
            )
            videos.append(
                f"""<article><div class="stage-title"><h3>t={stage['training_timestep']:.0f}</h3><span>sigma {stage['scheduler_sigma']:.4f} · gate {stage['noise_gate']:.4f}</span></div><video controls muted loop preload="metadata" src="{html.escape(video_path)}"></video><div class="links"><a href="cases/{html.escape(case_key)}/{sid}/metrics.json">metrics.json</a><a href="cases/{html.escape(case_key)}/{sid}/loss_maps.npz">loss_maps.npz</a></div></article>"""
            )
        sections.append(
            f"""<section><div class="case-heading"><span>{html.escape(case['family'])}</span><div><h2>{html.escape(case['case_key'])}</h2><p>{html.escape(case['caption'])}</p></div></div><table><thead><tr><th>t</th><th>sigma</th><th>gate</th><th>soft CE</th><th>coord Huber</th><th>gated total</th><th>top1 err</th><th>PCK@1</th></tr></thead><tbody>{''.join(rows)}</tbody></table><div class="stages">{''.join(videos)}</div></section>"""
        )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Noise-gated correspondence diagnostics</title><style>
:root{{--bg:#eef1ef;--paper:#fff;--ink:#17201d;--muted:#64706b;--line:#c7cfcb;--accent:#086b5f;--warm:#a6640b}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Noto Sans SC","Source Han Sans SC",sans-serif;letter-spacing:0}}header{{background:#17201d;color:#fff;padding:28px max(24px,calc((100vw - 1500px)/2))}}header h1{{font-size:28px;margin:0 0 8px}}header p{{margin:0;color:#c8d1cd;max-width:1000px;line-height:1.6}}main{{max-width:1500px;margin:0 auto;padding:24px}}section{{background:var(--paper);border:1px solid var(--line);margin-bottom:24px;padding:20px}}.case-heading{{display:flex;gap:14px;align-items:start;border-bottom:1px solid var(--line);padding-bottom:14px}}.case-heading>span{{background:var(--accent);color:#fff;padding:4px 8px;font-weight:700}}h2{{font-size:20px;margin:0}}.case-heading p{{margin:5px 0 0;color:var(--muted)}}table{{width:100%;border-collapse:collapse;margin:16px 0;font-variant-numeric:tabular-nums}}th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:right;font-size:13px}}th:first-child,td:first-child{{text-align:left}}.stages{{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:16px}}article{{border-top:3px solid var(--accent);background:#f7f9f8;padding:12px}}.stage-title{{display:flex;align-items:baseline;justify-content:space-between;gap:12px}}.stage-title h3{{font-size:16px;margin:0}}.stage-title span{{font-size:12px;color:var(--muted)}}video{{display:block;width:100%;margin-top:10px;background:#111}}.links{{display:flex;gap:14px;margin-top:8px}}a{{color:var(--accent);font-size:12px}}@media(max-width:700px){{main{{padding:10px}}section{{padding:12px}}.stages{{grid-template-columns:1fr}}table{{display:block;overflow-x:auto}}}}
</style></head><body><header><h1>Noise-gated point correspondence</h1><p>PyBullet train split · F04 SAM2 points tracked by CoTracker · source L01/F04 · future-frame Gaussian labels · Top100 PCK-weighted Main Student QK · forward-only diagnostic</p></header><main>{''.join(sections)}</main></body></html>"""
    (output_root / "index.html").write_text(page, encoding="utf-8")


def run_render(args: argparse.Namespace) -> None:
    manifest = load_manifest(args)
    output_root = args.output_root.resolve()
    cases = []
    for case_index, case in enumerate(manifest["cases"], start=1):
        case_key = str(case["case_key"])
        case_output = output_root / "cases" / case_key
        case_metrics = json.loads((case_output / "metrics.json").read_text(encoding="utf-8"))
        source_case = Path(case["cache_dir"])
        with np.load(source_case / "source_frames.npz") as arrays:
            frames = arrays["frames"].astype(np.uint8)
        track_path = args.cache_root.resolve() / "cases" / case_key / "point_tracks.npz"
        with np.load(track_path) as arrays:
            tracks = arrays["tracks_tn2"].astype(np.float32)
            visibility = arrays["visibility_tn"].astype(bool)
        print(f"[render {case_index}/3] {case_key}", flush=True)
        for stage in case_metrics["stages"]:
            stage_output = case_output / stage["stage_id"]
            complete = stage_output / "render_complete.json"
            if complete.is_file() and not args.overwrite:
                continue
            render_stage_video(
                stage_output,
                frames,
                tracks,
                visibility,
                stage,
                fps=float(args.fps),
                coordinate_weight=float(args.coordinate_weight),
            )
        cases.append(case_metrics)
    build_report(output_root, cases)
    atomic_json(
        output_root / "render_status.json",
        {"state": "complete", "case_count": len(cases)},
    )


def main() -> None:
    args = parse_args()
    check_args(args)
    if args.mode in {"prepare-tracks", "all"}:
        run_prepare_tracks(args)
    if args.mode in {"forward", "all"}:
        run_forward(args)
    if args.mode in {"render", "all"}:
        run_render(args)


if __name__ == "__main__":
    main()
