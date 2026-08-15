#!/usr/bin/env python3
"""Visualize uniform object-equal Q/K correspondence on PyBullet cases."""

from __future__ import annotations

import argparse
import gc
import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

import run_pybullet_correspondence_diagnostics as single
import train_xssc_object_self_attn_lora as train_core
from frozen_motion_probe import load_pck_head_weights
from noise_gated_correspondence import (
    points_to_token_coordinates,
    uniform_object_correspondence_objective,
)
from run_training_case_diagnostics import add_label, atomic_json, write_video
from AAA_my_test import analyze_wan_gt_toy_worker as wan_tools
from AAA_my_test.precompute_toydataset_sam2_regions import (
    build_provider,
    detect_and_track_objects,
)
from code_vjepa_vggt.train0419_reference.batch_eval_lora import build_pipeline
from code_vjepa_vggt.utils.object_priors import sample_points_from_mask


DEFAULT_SOURCE_CACHE = Path(
    "/data/gaoya/agent-data/cache/frozen_motion_probe_training_diagnostics"
)
DEFAULT_SOURCE_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/frozen_motion_probe_training_diagnostics"
)
DEFAULT_CACHE_ROOT = Path(
    "/data/gaoya/agent-data/cache/uniform_multiobject_correspondence_diagnostics"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/uniform_multiobject_correspondence_diagnostics"
)
DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_OPENVID_LORA = Path(
    "/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/"
    "openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors"
)
DEFAULT_COTRACKER_CHECKPOINT = Path(
    "/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth"
)
DEFAULT_HEAD_CONFIG = (
    single.EXPERIMENT_ROOT
    / "configs/physiciq67_pck32_s039_latest3350_top100_heads.json"
)
DEFAULT_TIMESTEPS = (100.0, 300.0, 500.0, 700.0, 900.0)
OBJECT_COLORS = np.asarray(
    [[238, 75, 71], [17, 150, 141], [45, 107, 185], [226, 167, 36]],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uniform multi-object correspondence visualization."
    )
    parser.add_argument(
        "mode",
        choices=("prepare-objects", "prepare-tracks", "forward", "render", "all"),
    )
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--source-output", type=Path, default=DEFAULT_SOURCE_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--openvid-lora", type=Path, default=DEFAULT_OPENVID_LORA)
    parser.add_argument(
        "--cotracker-checkpoint", type=Path, default=DEFAULT_COTRACKER_CHECKPOINT
    )
    parser.add_argument("--head-config", type=Path, default=DEFAULT_HEAD_CONFIG)
    parser.add_argument(
        "--head-subset", default="T_physiciq67_pck32_s039_latest3350_top100"
    )
    parser.add_argument(
        "--head-subtype", default="physiciq67_pck32_s039_latest3350"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--training-timesteps", type=float, nargs="+", default=DEFAULT_TIMESTEPS
    )
    parser.add_argument("--num-points", type=int, default=8)
    parser.add_argument("--source-pixel-frame", type=int, default=4)
    parser.add_argument("--source-latent-frame", type=int, default=1)
    parser.add_argument("--label-sigma-tokens", type=float, default=0.75)
    parser.add_argument("--lambda-corr", type=float, default=0.01)
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
    if float(args.label_sigma_tokens) <= 0.0 or float(args.lambda_corr) <= 0.0:
        raise ValueError("label sigma and lambda-corr must be positive")
    for timestep in args.training_timesteps:
        if float(timestep) not in DEFAULT_TIMESTEPS:
            raise ValueError(
                f"timestep={timestep} has no controlled cached x_t; expected {DEFAULT_TIMESTEPS}"
            )
    for path in (
        args.source_cache / "manifest.json",
        args.openvid_lora,
        args.head_config,
    ):
        if not path.expanduser().resolve().is_file():
            raise FileNotFoundError(path)


def load_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return json.loads(
        (args.source_cache.resolve() / "manifest.json").read_text(encoding="utf-8")
    )


def case_manifest_path(case: dict[str, Any]) -> Path:
    return Path(case["source_video"]).resolve().parent.parent / "case_manifest.json"


def dynamic_object_phrases(case: dict[str, Any]) -> list[str]:
    path = case_manifest_path(case)
    payload = json.loads(path.read_text(encoding="utf-8"))
    phrases = [
        str(value).strip()
        for value in payload.get("dynamic_object_phrases", [])
        if str(value).strip()
    ]
    if not phrases:
        raise RuntimeError(f"{case['case_key']}: metadata has no dynamic object phrases")
    return phrases


def load_source_frames(case: dict[str, Any]) -> np.ndarray:
    with np.load(Path(case["cache_dir"]) / "source_frames.npz") as arrays:
        return arrays["frames"].astype(np.uint8)


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    output = np.asarray(frame, dtype=np.uint8).astype(np.float32)
    selected = np.asarray(mask, dtype=bool)
    output[selected] = 0.68 * output[selected] + 0.32 * color.astype(np.float32)
    result = output.round().clip(0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, tuple(map(int, color)), 2, cv2.LINE_AA)
    return result


def render_identity_image(
    frame: np.ndarray,
    masks_ohw: np.ndarray,
    phrases: list[str],
) -> np.ndarray:
    output = np.asarray(frame, dtype=np.uint8).copy()
    for object_index, (mask, phrase) in enumerate(zip(masks_ohw, phrases)):
        color = OBJECT_COLORS[object_index % len(OBJECT_COLORS)]
        output = overlay_mask(output, mask, color)
        ys, xs = np.where(mask > 0)
        if xs.size:
            cv2.putText(
                output,
                f"O{object_index + 1}: {phrase[:34]}",
                (int(xs.min()), max(46, int(ys.min()) - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                tuple(map(int, color)),
                2,
                cv2.LINE_AA,
            )
    return add_label(output, f"F04 identity masks | {len(phrases)} dynamic objects")


def run_prepare_objects(args: argparse.Namespace) -> None:
    manifest = load_manifest(args)
    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    print("[objects] loading GroundingDINO + SAM2", flush=True)
    provider = build_provider(str(args.device), int(args.num_points))
    summaries = []
    try:
        for case_index, case in enumerate(manifest["cases"], start=1):
            case_key = str(case["case_key"])
            case_cache = cache_root / "cases" / case_key
            case_cache.mkdir(parents=True, exist_ok=True)
            complete = case_cache / "objects_complete.json"
            if complete.is_file() and not args.overwrite:
                summaries.append(json.loads(complete.read_text(encoding="utf-8")))
                print(f"[objects {case_index}/3] skip {case_key}", flush=True)
                continue
            phrases = dynamic_object_phrases(case)
            frames = load_source_frames(case)
            frames_tchw = frames.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
            print(
                f"[objects {case_index}/3] {case_key}: {len(phrases)} objects",
                flush=True,
            )
            tracked = detect_and_track_objects(provider, frames_tchw, phrases)
            if len(tracked.object_tracks) != len(phrases):
                raise RuntimeError(
                    f"{case_key}: tracked {len(tracked.object_tracks)}/{len(phrases)} objects"
                )
            masks_othw = np.stack(
                [np.asarray(track.masks_thw, dtype=np.uint8) for track in tracked.object_tracks]
            )
            boxes_ot4 = np.stack(
                [np.asarray(track.boxes_t4, dtype=np.float32) for track in tracked.object_tracks]
            )
            if masks_othw.shape[1:] != frames.shape[:3]:
                raise RuntimeError(
                    f"{case_key}: mask/video shape mismatch {masks_othw.shape}/{frames.shape}"
                )
            f04_masks = masks_othw[:, int(args.source_pixel_frame)]
            empty = [index for index, mask in enumerate(f04_masks) if not bool(mask.any())]
            if empty:
                raise RuntimeError(f"{case_key}: empty F04 object masks {empty}")
            np.savez_compressed(
                case_cache / "object_masks.npz",
                masks_othw=masks_othw,
                boxes_ot4=boxes_ot4,
                f04_rgb=frames[int(args.source_pixel_frame)],
            )
            preview = render_identity_image(
                frames[int(args.source_pixel_frame)], f04_masks, phrases
            )
            preview_path = output_root / "cases" / case_key / "f04_multiobject_identity.png"
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            summary = {
                "case_key": case_key,
                "object_count": len(phrases),
                "object_phrases": phrases,
                "object_source": "PyBullet case_manifest.dynamic_object_phrases",
                "assignment": tracked.debug,
                "source_frame": "F04/L01",
            }
            atomic_json(complete, summary)
            summaries.append(summary)
    finally:
        del provider
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(
        cache_root / "objects_status.json",
        {"state": "complete", "case_count": len(summaries), "cases": summaries},
    )


def run_prepare_tracks(args: argparse.Namespace) -> None:
    from cotracker.predictor import CoTrackerPredictor

    manifest = load_manifest(args)
    cache_root = args.cache_root.resolve()
    print("[tracks] loading CoTracker", flush=True)
    model = CoTrackerPredictor(
        checkpoint=str(args.cotracker_checkpoint.resolve()),
        offline=True,
        v2=False,
        window_len=60,
    ).to(args.device).eval().requires_grad_(False)
    summaries = []
    try:
        for case_index, case in enumerate(manifest["cases"], start=1):
            case_key = str(case["case_key"])
            case_cache = cache_root / "cases" / case_key
            complete = case_cache / "tracks_complete.json"
            if complete.is_file() and not args.overwrite:
                summaries.append(json.loads(complete.read_text(encoding="utf-8")))
                print(f"[tracks {case_index}/3] skip {case_key}", flush=True)
                continue
            object_meta = json.loads(
                (case_cache / "objects_complete.json").read_text(encoding="utf-8")
            )
            phrases = list(object_meta["object_phrases"])
            with np.load(case_cache / "object_masks.npz") as arrays:
                masks_othw = arrays["masks_othw"].astype(np.uint8)
            frames = load_source_frames(case)
            points_on2 = np.stack(
                [
                    sample_points_from_mask(
                        masks_othw[index, int(args.source_pixel_frame)],
                        int(args.num_points),
                        avoid_edges=True,
                    )
                    for index in range(len(phrases))
                ]
            )
            if points_on2.shape != (len(phrases), int(args.num_points), 2):
                raise RuntimeError(f"{case_key}: sampled point shape {points_on2.shape}")
            print(
                f"[tracks {case_index}/3] {case_key}: "
                f"{len(phrases)} x {args.num_points} points",
                flush=True,
            )
            tracks, visibility = single.run_cotracker_from_anchor(
                model,
                frames,
                points_on2.reshape(-1, 2),
                anchor_frame=int(args.source_pixel_frame),
                device=str(args.device),
            )
            tracks_ton2 = tracks[single.ANCHOR_PIXEL_FRAMES].reshape(
                len(single.ANCHOR_PIXEL_FRAMES), len(phrases), int(args.num_points), 2
            )
            visibility_ton = visibility[single.ANCHOR_PIXEL_FRAMES].reshape(
                len(single.ANCHOR_PIXEL_FRAMES), len(phrases), int(args.num_points)
            )
            source_visible = visibility_ton[int(args.source_latent_frame)]
            if not bool(source_visible.all()):
                raise RuntimeError(f"{case_key}: not all object points are visible at F04")
            future_counts = visibility_ton[int(args.source_latent_frame) + 1 :].sum(
                axis=(0, 2)
            )
            if bool((future_counts == 0).any()):
                raise RuntimeError(f"{case_key}: object without future visible tracks")
            np.savez_compressed(
                case_cache / "point_tracks.npz",
                query_points_on2=points_on2,
                tracks_ton2=tracks_ton2,
                visibility_ton=visibility_ton.astype(np.uint8),
                anchor_pixel_frames=single.ANCHOR_PIXEL_FRAMES,
                pixel_hw=np.asarray(frames.shape[1:3], dtype=np.int32),
            )
            per_object = [
                {
                    "object_index": index,
                    "phrase": phrase,
                    "visibility_rate": float(visibility_ton[:, index].mean()),
                    "future_pair_count": int(future_counts[index]),
                }
                for index, phrase in enumerate(phrases)
            ]
            summary = {
                "case_key": case_key,
                "object_count": len(phrases),
                "point_count_per_object": int(args.num_points),
                "source_frame": "F04/L01",
                "supervision": "per-object CoTracker pseudo-GT from SAM2 identity masks",
                "objects": per_object,
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


class MultiObjectPCKCollector(single.PCKCorrespondenceCollector):
    def __init__(
        self,
        *args,
        point_coordinates_ton2: torch.Tensor,
        point_visibility_ton: torch.Tensor,
        **kwargs,
    ) -> None:
        if point_coordinates_ton2.ndim != 4 or point_coordinates_ton2.shape[-1] != 2:
            raise ValueError(
                f"expected [T,O,N,2] coordinates, got {point_coordinates_ton2.shape}"
            )
        if point_visibility_ton.shape != point_coordinates_ton2.shape[:-1]:
            raise ValueError("multi-object visibility shape mismatch")
        self.object_count = int(point_coordinates_ton2.shape[1])
        self.points_per_object = int(point_coordinates_ton2.shape[2])
        self.point_coordinates_ton2 = point_coordinates_ton2
        self.point_visibility_ton = point_visibility_ton
        time_count = int(point_coordinates_ton2.shape[0])
        super().__init__(
            *args,
            point_coordinates_tn2=point_coordinates_ton2.reshape(time_count, -1, 2),
            point_visibility_tn=point_visibility_ton.reshape(time_count, -1),
            **kwargs,
        )

    def finalize_uniform(
        self,
        *,
        scheduler_sigma: float,
        lambda_corr: float,
        object_phrases: list[str],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self.head_count != 100:
            raise RuntimeError(f"captured {self.head_count} heads, expected 100")
        if self.attention is None or self.target is None or self.valid is None:
            raise RuntimeError("collector captured no correspondence maps")
        if len(object_phrases) != self.object_count:
            raise ValueError("object phrase/count mismatch")

        attention_tps = self.attention[0].float()
        target_tps = self.target.float()
        valid_tp = self.valid.bool()
        time_count, _, spatial_count = attention_tps.shape
        object_count = self.object_count
        point_count = self.points_per_object
        attention = attention_tps.reshape(
            time_count, object_count, point_count, spatial_count
        ).permute(1, 0, 2, 3)
        target = target_tps.reshape(
            time_count, object_count, point_count, spatial_count
        ).permute(1, 0, 2, 3)
        valid = valid_tp.reshape(time_count, object_count, point_count).permute(1, 0, 2)
        probability_error = float((attention.sum(dim=-1) - 1.0).abs().max().item())
        if probability_error > 2.0e-5:
            raise RuntimeError(f"PCK aggregate probability error: {probability_error}")

        grid = torch.stack(
            torch.meshgrid(
                torch.arange(
                    single.TOKEN_HW[0], device=attention.device, dtype=torch.float32
                ),
                torch.arange(
                    single.TOKEN_HW[1], device=attention.device, dtype=torch.float32
                ),
                indexing="ij",
            )[::-1],
            dim=-1,
        ).reshape(-1, 2)
        predicted = torch.einsum("otns,sc->otnc", attention, grid)
        coordinate_target = self.point_coordinates_ton2.permute(1, 0, 2, 3).to(
            predicted
        )
        objective = uniform_object_correspondence_objective(
            attention,
            target,
            valid,
            lambda_corr=float(lambda_corr),
        )
        top1 = attention.argmax(dim=-1)
        top1_coordinates = torch.stack(
            ((top1 % single.TOKEN_HW[1]).float(), (top1 // single.TOKEN_HW[1]).float()),
            dim=-1,
        )
        top1_error = torch.linalg.vector_norm(
            top1_coordinates - coordinate_target, dim=-1
        )
        soft_error = torch.linalg.vector_norm(predicted - coordinate_target, dim=-1)

        object_metrics = []
        for object_index, phrase in enumerate(object_phrases):
            selected = valid[object_index]
            object_metrics.append(
                {
                    "object_index": object_index,
                    "phrase": phrase,
                    "valid_point_frame_pairs": int(selected.sum().item()),
                    "raw_soft_ce": float(
                        objective["raw_soft_ce_per_object"][object_index].item()
                    ),
                    "correspondence_loss": float(
                        lambda_corr
                        * objective["raw_soft_ce_per_object"][object_index].item()
                    ),
                    "mean_top1_error_tokens": float(
                        top1_error[object_index][selected].mean().item()
                    ),
                    "mean_softargmax_error_tokens": float(
                        soft_error[object_index][selected].mean().item()
                    ),
                    "pck_at_1_token": float(
                        (top1_error[object_index][selected] <= 1.0).float().mean().item()
                    ),
                    "pck_at_2_tokens": float(
                        (top1_error[object_index][selected] <= 2.0).float().mean().item()
                    ),
                }
            )
        maps = {
            "attention_otns": attention.cpu().numpy(),
            "target_otns": target.cpu().numpy(),
            "ce_contribution_otns": objective["ce_contribution"].cpu().numpy(),
            "predicted_coordinates_otn2": predicted.cpu().numpy(),
            "target_coordinates_otn2": coordinate_target.cpu().numpy(),
            "visibility_otn": self.point_visibility_ton.permute(1, 0, 2)
            .cpu()
            .numpy()
            .astype(np.uint8),
            "valid_otn": valid.cpu().numpy().astype(np.uint8),
            "ce_otn": objective["ce"].cpu().numpy(),
            "top1_error_otn": top1_error.cpu().numpy(),
            "softargmax_error_otn": soft_error.cpu().numpy(),
        }
        metrics = {
            "scheduler_sigma": float(scheduler_sigma),
            "raw_soft_ce": float(objective["raw_soft_ce"].item()),
            "lambda_corr": float(lambda_corr),
            "correspondence_loss": float(objective["loss"].item()),
            "object_count": object_count,
            "point_count_per_object": point_count,
            "valid_point_frame_pairs": int(valid.sum().item()),
            "mean_top1_error_tokens": float(
                np.mean([row["mean_top1_error_tokens"] for row in object_metrics])
            ),
            "mean_softargmax_error_tokens": float(
                np.mean([row["mean_softargmax_error_tokens"] for row in object_metrics])
            ),
            "pck_at_1_token": float(
                np.mean([row["pck_at_1_token"] for row in object_metrics])
            ),
            "pck_at_2_tokens": float(
                np.mean([row["pck_at_2_tokens"] for row in object_metrics])
            ),
            "aggregate_probability_max_error": probability_error,
            "object_reduction": "mean valid pairs per object, then equal mean over objects",
            "objects": object_metrics,
        }
        return maps, metrics


def run_forward(args: argparse.Namespace) -> None:
    manifest = load_manifest(args)
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    print("[forward] loading Wan2.2 + OpenVid LoRA step-10000", flush=True)
    pipe = build_pipeline(
        args.wan_root.resolve(), str(device), args.openvid_lora.resolve()
    )
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
    summaries = []
    try:
        for case_index, case in enumerate(manifest["cases"], start=1):
            case_key = str(case["case_key"])
            case_cache = args.cache_root.resolve() / "cases" / case_key
            object_meta = json.loads(
                (case_cache / "objects_complete.json").read_text(encoding="utf-8")
            )
            phrases = list(object_meta["object_phrases"])
            with np.load(case_cache / "point_tracks.npz") as arrays:
                tracks_ton2 = arrays["tracks_ton2"].astype(np.float32)
                visibility_ton = arrays["visibility_ton"].astype(bool)
                pixel_hw = tuple(map(int, arrays["pixel_hw"].tolist()))
            point_coordinates = points_to_token_coordinates(
                torch.from_numpy(tracks_ton2),
                pixel_hw=pixel_hw,
                token_hw=single.TOKEN_HW,
            ).to(device)
            point_visibility = torch.from_numpy(visibility_ton).to(device)
            source_frames = wan_tools.load_video_prefix(
                Path(case["source_video"]), 49, 512, 896, "cache"
            )
            inputs_shared, inputs_positive = wan_tools.prepare_conditioning(
                pipe,
                prompt=case["caption"],
                context_video=source_frames[:8],
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
            case_output = output_root / "cases" / case_key
            case_output.mkdir(parents=True, exist_ok=True)
            stage_metrics = []
            for timestep_value in args.training_timesteps:
                sid = single.stage_id(timestep_value)
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
                    source_stage / "latents.pt", map_location="cpu", weights_only=True
                )
                source_metrics = json.loads(
                    (source_stage / "metrics.json").read_text(encoding="utf-8")
                )
                latent_xt = latent_state["training_xt"].to(
                    device=device, dtype=pipe.torch_dtype
                )
                timestep = torch.full(
                    (1,), float(timestep_value), device=device, dtype=pipe.torch_dtype
                )
                sigma = float(source_metrics["scheduler_sigma"])
                collector = MultiObjectPCKCollector(
                    pipe.dit,
                    selected_heads,
                    pck_weights,
                    point_coordinates_ton2=point_coordinates,
                    point_visibility_ton=point_visibility,
                    source_frame=int(args.source_latent_frame),
                    label_sigma_tokens=float(args.label_sigma_tokens),
                )
                print(
                    f"[forward {case_index}/3] {case_key} {sid} "
                    f"objects={len(phrases)} sigma={sigma:.4f}",
                    flush=True,
                )
                torch.cuda.reset_peak_memory_stats(device)
                main_inputs = dict(captured_inputs)
                main_inputs["latents"] = latent_xt
                collector.install()
                try:
                    with torch.no_grad():
                        velocity = pipe.model_fn(
                            **models, **main_inputs, timestep=timestep
                        )
                finally:
                    collector.remove()
                del velocity, latent_xt, latent_state
                maps, metrics = collector.finalize_uniform(
                    scheduler_sigma=sigma,
                    lambda_corr=float(args.lambda_corr),
                    object_phrases=phrases,
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
                        "model": (
                            "Wan2.2-TI2V-5B + OpenVid LoRA step-10000; "
                            "equivalent to fresh zero-delta Full-SA initialization"
                        ),
                        "execution_scope": "one forward per case/stage; no optimizer step",
                        "supervision": "per-object CoTracker pseudo-GT from F04 SAM2 masks",
                        "source_frame": "L01/F04",
                        "target_frames": "future L02/F08 through L12/F48",
                        "label_sigma_tokens": float(args.label_sigma_tokens),
                        "loss_formula": "0.01 * object_equal_mean(soft_label_CE)",
                        "noise_weighting": "uniform; no SNR gate and no hard cutoff",
                        "head_weighting": "Top100 normalized linear PCK32 mixture",
                        "head_reduction": "CE(target, sum_h weight_h * attention_h)",
                        "source_query_sampling": "bilinear continuous token coordinates",
                        "pixel_token_mapping": "cell-center aligned",
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
                "object_count": len(phrases),
                "object_phrases": phrases,
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


def draw_object_points(
    frame: np.ndarray,
    points_on2: np.ndarray,
    visible_on: np.ndarray,
    *,
    predicted_on2: np.ndarray | None = None,
) -> np.ndarray:
    output = np.asarray(frame, dtype=np.uint8).copy()
    for object_index, (points, visibility) in enumerate(zip(points_on2, visible_on)):
        color = tuple(map(int, OBJECT_COLORS[object_index % len(OBJECT_COLORS)]))
        for point_index, (point, visible) in enumerate(zip(points, visibility)):
            if not bool(visible):
                continue
            x, y = (int(round(float(value))) for value in point)
            cv2.circle(output, (x, y), 5, color, -1, cv2.LINE_AA)
            cv2.circle(output, (x, y), 7, (17, 21, 19), 1, cv2.LINE_AA)
            if predicted_on2 is not None:
                px, py = (
                    int(round(float(value)))
                    for value in predicted_on2[object_index, point_index]
                )
                cv2.drawMarker(
                    output,
                    (px, py),
                    (255, 255, 255),
                    cv2.MARKER_TILTED_CROSS,
                    11,
                    2,
                    cv2.LINE_AA,
                )
    return output


def positive_scale(values: np.ndarray, percentile: float = 99.5) -> float:
    positive = np.asarray(values)[np.asarray(values) > 0]
    return float(np.percentile(positive, percentile)) if positive.size else 1.0


def per_object_frame_maps(values_otns: np.ndarray, valid_otn: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            single.average_visible_maps(values_otns[index], valid_otn[index])
            for index in range(values_otns.shape[0])
        ]
    )


def equal_object_frame_mean(maps_othw: np.ndarray, valid_otn: np.ndarray) -> np.ndarray:
    output = np.zeros_like(maps_othw[0])
    for time_index in range(maps_othw.shape[1]):
        active = np.asarray(valid_otn[:, time_index].any(axis=1), dtype=bool)
        if active.any():
            output[time_index] = maps_othw[active, time_index].mean(axis=0)
    return output


def render_stage(
    stage_output: Path,
    frames: np.ndarray,
    masks_othw: np.ndarray,
    tracks_ton2: np.ndarray,
    visibility_ton: np.ndarray,
    metrics: dict[str, Any],
    phrases: list[str],
    *,
    fps: float,
) -> None:
    with np.load(stage_output / "loss_maps.npz") as arrays:
        attention = arrays["attention_otns"].astype(np.float32)
        target = arrays["target_otns"].astype(np.float32)
        ce_contribution = arrays["ce_contribution_otns"].astype(np.float32)
        predicted_token = arrays["predicted_coordinates_otn2"].astype(np.float32)
        valid = arrays["valid_otn"].astype(bool)
        top1_error = arrays["top1_error_otn"].astype(np.float32)
    tracks = tracks_ton2.transpose(1, 0, 2, 3)
    visibility = visibility_ton.transpose(1, 0, 2)
    predicted_pixel = single.token_to_pixel(predicted_token, frames.shape[1:3])
    lambda_corr = float(metrics["lambda_corr"])
    attention_maps = per_object_frame_maps(attention, visibility)
    target_maps = per_object_frame_maps(target, visibility)
    ce_maps = per_object_frame_maps(ce_contribution * lambda_corr, valid)

    object_dir = stage_output / "objects"
    object_dir.mkdir(parents=True, exist_ok=True)
    for object_index, phrase in enumerate(phrases):
        scales = {
            "attention": positive_scale(attention_maps[object_index]),
            "target": positive_scale(target_maps[object_index]),
            "ce": positive_scale(ce_maps[object_index]),
        }
        rendered = []
        for latent_index, pixel_index in enumerate(single.ANCHOR_PIXEL_FRAMES):
            base = overlay_mask(
                frames[int(pixel_index)],
                masks_othw[object_index, int(pixel_index)],
                OBJECT_COLORS[object_index % len(OBJECT_COLORS)],
            )
            points = tracks[object_index, latent_index]
            visible = visibility[object_index, latent_index]
            trajectory_panel = single.draw_points(base, points, visible)
            trajectory_panel = add_label(
                trajectory_panel,
                f"O{object_index + 1} trajectory | L{latent_index:02d}/F{pixel_index:02d}",
            )
            target_panel = single.overlay_heatmap(
                base, target_maps[object_index, latent_index], vmax=scales["target"]
            )
            target_panel = single.draw_points(target_panel, points, visible)
            target_panel = add_label(target_panel, "Gaussian trajectory target")
            attention_panel = single.overlay_heatmap(
                base,
                attention_maps[object_index, latent_index],
                vmax=scales["attention"],
            )
            attention_panel = single.draw_points(
                attention_panel,
                points,
                visible,
                predicted_n2=predicted_pixel[object_index, latent_index],
            )
            selected = valid[object_index, latent_index]
            error = (
                float(top1_error[object_index, latent_index][selected].mean())
                if selected.any()
                else float("nan")
            )
            attention_panel = add_label(
                attention_panel, f"PCK QK mixture | top1 err {error:.2f} token"
            )
            ce_panel = single.overlay_heatmap(
                base, ce_maps[object_index, latent_index], vmax=scales["ce"]
            )
            ce_panel = single.draw_points(ce_panel, points, visible)
            ce_panel = add_label(ce_panel, "uniform CE contribution | lambda 0.01")
            rendered.append(
                np.concatenate(
                    [
                        single.resize_panel(trajectory_panel),
                        single.resize_panel(target_panel),
                        single.resize_panel(attention_panel),
                        single.resize_panel(ce_panel),
                    ],
                    axis=1,
                )
            )
        write_video(
            object_dir / f"object_{object_index:02d}_loss_overlay.mp4",
            rendered,
            fps=float(fps),
        )

    combined_attention = equal_object_frame_mean(attention_maps, valid)
    combined_target = equal_object_frame_mean(target_maps, valid)
    combined_ce = equal_object_frame_mean(ce_maps, valid)
    combined_scales = {
        "attention": positive_scale(combined_attention),
        "target": positive_scale(combined_target),
        "ce": positive_scale(combined_ce),
    }
    combined_frames = []
    for latent_index, pixel_index in enumerate(single.ANCHOR_PIXEL_FRAMES):
        base = frames[int(pixel_index)].copy()
        for object_index in range(len(phrases)):
            base = overlay_mask(
                base,
                masks_othw[object_index, int(pixel_index)],
                OBJECT_COLORS[object_index % len(OBJECT_COLORS)],
            )
        trajectory_panel = draw_object_points(
            base,
            tracks[:, latent_index],
            visibility[:, latent_index],
        )
        trajectory_panel = add_label(
            trajectory_panel,
            f"all {len(phrases)} objects | L{latent_index:02d}/F{pixel_index:02d}",
        )
        target_panel = single.overlay_heatmap(
            base, combined_target[latent_index], vmax=combined_scales["target"]
        )
        target_panel = draw_object_points(
            target_panel, tracks[:, latent_index], visibility[:, latent_index]
        )
        target_panel = add_label(target_panel, "equal-object Gaussian target")
        attention_panel = single.overlay_heatmap(
            base,
            combined_attention[latent_index],
            vmax=combined_scales["attention"],
        )
        attention_panel = draw_object_points(
            attention_panel,
            tracks[:, latent_index],
            visibility[:, latent_index],
            predicted_on2=predicted_pixel[:, latent_index],
        )
        attention_panel = add_label(attention_panel, "equal-object PCK QK mixture")
        ce_panel = single.overlay_heatmap(
            base, combined_ce[latent_index], vmax=combined_scales["ce"]
        )
        ce_panel = draw_object_points(
            ce_panel, tracks[:, latent_index], visibility[:, latent_index]
        )
        ce_panel = add_label(ce_panel, "equal-object CE | no noise gate")
        combined_frames.append(
            np.concatenate(
                [
                    single.resize_panel(trajectory_panel),
                    single.resize_panel(target_panel),
                    single.resize_panel(attention_panel),
                    single.resize_panel(ce_panel),
                ],
                axis=1,
            )
        )
    write_video(
        stage_output / "object_equal_loss_overlay.mp4",
        combined_frames,
        fps=float(fps),
    )
    atomic_json(
        stage_output / "render_complete.json",
        {"state": "complete", "training_timestep": metrics["training_timestep"]},
    )


def build_report(output_root: Path, cases: list[dict[str, Any]]) -> None:
    sections = []
    for case in cases:
        case_key = str(case["case_key"])
        rows = []
        stages = []
        for stage_index, stage in enumerate(case["stages"]):
            object_losses = " / ".join(
                f"O{row['object_index'] + 1} {row['correspondence_loss']:.5f}"
                for row in stage["objects"]
            )
            rows.append(
                "<tr>"
                f"<td>{stage['training_timestep']:.0f}</td>"
                f"<td>{stage['scheduler_sigma']:.4f}</td>"
                f"<td>{stage['raw_soft_ce']:.4f}</td>"
                f"<td>{stage['correspondence_loss']:.6f}</td>"
                f"<td>{stage['mean_top1_error_tokens']:.3f}</td>"
                f"<td>{stage['pck_at_1_token']:.3f}</td>"
                f"<td>{html.escape(object_losses)}</td>"
                "</tr>"
            )
            sid = stage["stage_id"]
            object_figures = []
            for object_row in stage["objects"]:
                object_index = int(object_row["object_index"])
                source = (
                    f"cases/{case_key}/{sid}/objects/"
                    f"object_{object_index:02d}_loss_overlay.mp4"
                )
                object_figures.append(
                    "<figure>"
                    f"<figcaption><strong>O{object_index + 1}: "
                    f"{html.escape(object_row['phrase'])}</strong>"
                    f"<span>loss {object_row['correspondence_loss']:.6f} | "
                    f"PCK@1 {object_row['pck_at_1_token']:.3f}</span></figcaption>"
                    f"<video controls muted loop playsinline preload='metadata' src='{html.escape(source)}'></video>"
                    "</figure>"
                )
            combined_source = f"cases/{case_key}/{sid}/object_equal_loss_overlay.mp4"
            stages.append(
                f"<details {'open' if stage_index == 0 else ''}>"
                f"<summary><strong>t={stage['training_timestep']:.0f}</strong>"
                f"<span>sigma {stage['scheduler_sigma']:.4f} | "
                f"object-equal loss {stage['correspondence_loss']:.6f}</span></summary>"
                "<div class='stage-body'>"
                "<figure class='combined'><figcaption><strong>All objects, equal reduction</strong>"
                "<span>one DiT forward; per-object CE then equal mean</span></figcaption>"
                f"<video controls muted loop playsinline preload='metadata' src='{html.escape(combined_source)}'></video>"
                "</figure>"
                f"<div class='objects'>{''.join(object_figures)}</div>"
                f"<div class='links'><a href='cases/{html.escape(case_key)}/{sid}/metrics.json'>metrics.json</a>"
                f"<a href='cases/{html.escape(case_key)}/{sid}/loss_maps.npz'>loss_maps.npz</a></div>"
                "</div></details>"
            )
        phrases = " | ".join(
            f"O{index + 1}: {phrase}" for index, phrase in enumerate(case["object_phrases"])
        )
        preview = f"cases/{case_key}/f04_multiobject_identity.png"
        sections.append(
            "<section>"
            f"<div class='case-heading'><span>{case['object_count']} objects</span><div>"
            f"<h2>{html.escape(case_key)}</h2><p>{html.escape(case['caption'])}</p>"
            f"<small>{html.escape(phrases)}</small></div>"
            f"<img class='identity' src='{html.escape(preview)}' alt='F04 object masks'>"
            "</div>"
            "<div class='table-wrap'><table><thead><tr><th>t</th><th>sigma</th>"
            "<th>object-equal CE</th><th>loss</th><th>top1 err</th><th>PCK@1</th>"
            f"<th>per-object loss</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            f"{''.join(stages)}</section>"
        )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Multi-object correspondence loss</title><style>
:root{{--bg:#edf0ee;--paper:#fff;--ink:#17201d;--muted:#64706b;--line:#c6ceca;--accent:#08756a;--warm:#a04b37}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Noto Sans SC","Source Han Sans SC",sans-serif;letter-spacing:0}}header{{padding:28px max(22px,calc((100vw - 1580px)/2));background:#17201d;color:#fff}}h1{{font-size:28px;margin:0 0 9px}}header p{{max-width:1120px;margin:0;color:#c8d1cd;line-height:1.6}}main{{max-width:1580px;margin:auto;padding:22px}}section{{margin-bottom:24px;padding:18px;background:var(--paper);border:1px solid var(--line)}}.case-heading{{display:grid;grid-template-columns:auto minmax(0,1fr) minmax(320px,520px);gap:14px;align-items:start;padding-bottom:15px;border-bottom:1px solid var(--line)}}.case-heading>span{{padding:5px 9px;background:var(--accent);color:#fff;font-weight:800}}h2{{font-size:20px;margin:0 0 4px}}.case-heading p{{margin:0 0 5px;color:var(--muted)}}.case-heading small{{display:block;line-height:1.5}}.identity{{display:block;width:100%;height:auto}}.table-wrap{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;margin:15px 0;font-variant-numeric:tabular-nums}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right;font-size:12px;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}details{{border-top:3px solid var(--accent);background:#f7f9f8;margin-top:12px}}summary{{display:flex;justify-content:space-between;gap:15px;padding:11px 13px;cursor:pointer}}summary span{{font-size:12px;color:var(--muted)}}.stage-body{{padding:0 12px 13px}}figure{{margin:0;border:1px solid var(--line);background:#fff;padding:7px;min-width:0}}figcaption{{display:flex;justify-content:space-between;gap:12px;min-height:24px;margin-bottom:5px}}figcaption strong{{font-size:12px}}figcaption span{{font-size:11px;color:var(--muted);text-align:right}}video{{display:block;width:100%;height:auto;background:#111}}.combined{{margin-bottom:10px}}.objects{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:10px}}.links{{display:flex;gap:14px;margin-top:9px}}a{{color:var(--accent);font-size:12px}}@media(max-width:800px){{main{{padding:9px}}section{{padding:10px}}.case-heading{{grid-template-columns:1fr}}.identity{{max-width:none}}summary,figcaption{{display:block}}summary span,figcaption span{{display:block;text-align:left;margin-top:3px}}.objects{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Uniform multi-object Q/K correspondence</h1><p>PyBullet train cases with 1, 2, and 3 dynamic objects. Wan2.2-TI2V-5B + merged OpenVid LoRA step-10000 initialization; one DiT forward per case and timestep; F04/L01 queries; per-object CoTracker pseudo-GT; Top100 linear PCK32 attention mixture; Gaussian sigma 0.75 token; Lcorr = 0.01 x equal mean of per-object CE; no SNR gate.</p></header><main>{''.join(sections)}</main></body></html>"""
    (output_root / "index.html").write_text(page, encoding="utf-8")


def run_render(args: argparse.Namespace) -> None:
    manifest = load_manifest(args)
    output_root = args.output_root.resolve()
    cases = []
    for case_index, case in enumerate(manifest["cases"], start=1):
        case_key = str(case["case_key"])
        case_cache = args.cache_root.resolve() / "cases" / case_key
        case_output = output_root / "cases" / case_key
        case_metrics = json.loads(
            (case_output / "metrics.json").read_text(encoding="utf-8")
        )
        frames = load_source_frames(case)
        with np.load(case_cache / "object_masks.npz") as arrays:
            masks_othw = arrays["masks_othw"].astype(np.uint8)
        with np.load(case_cache / "point_tracks.npz") as arrays:
            tracks_ton2 = arrays["tracks_ton2"].astype(np.float32)
            visibility_ton = arrays["visibility_ton"].astype(bool)
        print(
            f"[render {case_index}/3] {case_key}: {case_metrics['object_count']} objects",
            flush=True,
        )
        for stage in case_metrics["stages"]:
            stage_output = case_output / stage["stage_id"]
            complete = stage_output / "render_complete.json"
            if complete.is_file() and not args.overwrite:
                continue
            render_stage(
                stage_output,
                frames,
                masks_othw,
                tracks_ton2,
                visibility_ton,
                stage,
                case_metrics["object_phrases"],
                fps=float(args.fps),
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
    if args.mode in {"prepare-objects", "all"}:
        run_prepare_objects(args)
    if args.mode in {"prepare-tracks", "all"}:
        run_prepare_tracks(args)
    if args.mode in {"forward", "all"}:
        run_forward(args)
    if args.mode in {"render", "all"}:
        run_render(args)


if __name__ == "__main__":
    main()
