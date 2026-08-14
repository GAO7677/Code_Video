#!/usr/bin/env python3
"""Run object-focused CoTracker trajectory-loss diagnostics on PyBullet cases."""

from __future__ import annotations

import argparse
import gc
import html
import json
from fractions import Fraction
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
TRAIN_XSSC_ROOT = EXPERIMENT_ROOT.parent
PACKAGE_ROOT = EXPERIMENT_ROOT.parents[2]
DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
DIFFTRACK_ROOT = Path("/home/gaoya/Code_Video/DiffTrack-main")
for _path in (
    HERE,
    EXPERIMENT_ROOT,
    TRAIN_XSSC_ROOT,
    PACKAGE_ROOT,
    DIFFSYNTH_ROOT,
    COTRACKER_ROOT,
    DIFFTRACK_ROOT,
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import av
import torch
import torch.nn.functional as F

import code_vjepa_vggt.context_wan_v_newtrain as context_wan
from code_vjepa_vggt.utils.object_priors import sample_points_from_mask
from AAA_my_test import analyze_wan_gt_toy_worker as wan_tools

import launch_from_config as launcher
import train_xssc_object_self_attn_lora as core
from xssc_loss_project import train_xssc_object_self_attn_lora_xssc_loss as xssc_trainer
from attention_trajectory_distillation_project.run_training_case_diagnostics import (
    restore_clean_conditioning,
)
from object_trajectory_loss import object_trajectory_loss
from vjepa_loss_project.train_xssc_object_self_attn_lora_vjepa_loss import (
    _load_tiny_vae,
)


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT
    / "xssc_loss_project/configs/"
    "full_sa_no_object_xssc_loss_dinov3_movic_step50000.json"
)
DEFAULT_SOURCE_CACHE = Path(
    "/data/gaoya/agent-data/cache/frozen_motion_probe_training_diagnostics"
)
DEFAULT_SOURCE_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/frozen_motion_probe_training_diagnostics"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_cotracker_trajectory_diagnostics"
)
DEFAULT_COTRACKER_CHECKPOINT = Path(
    "/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth"
)
TRACK_HEIGHT = 256
TRACK_WIDTH = 448
PALETTE = np.asarray(
    [
        (255, 214, 51),
        (38, 220, 255),
        (255, 84, 114),
        (54, 232, 138),
        (240, 132, 42),
        (196, 112, 255),
        (255, 255, 255),
        (44, 146, 255),
    ],
    dtype=np.uint8,
)


class _Accelerator:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.is_main_process = True

    @staticmethod
    def print(*args, **kwargs) -> None:
        print(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("forward", "tracks", "render", "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--source-output", type=Path, default=DEFAULT_SOURCE_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--cotracker-checkpoint",
        type=Path,
        default=DEFAULT_COTRACKER_CHECKPOINT,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-timestep", type=float, default=500.0)
    parser.add_argument("--num-points", type=int, default=24)
    parser.add_argument("--anchor-frame", type=int, default=4)
    parser.add_argument("--future-start-frame", type=int, default=8)
    parser.add_argument("--huber-delta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=5200)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--gradient-audit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if str(args.device).startswith("cuda:4"):
        raise ValueError("GPU 4 is prohibited by workspace rules")
    if int(args.num_points) <= 0:
        raise ValueError("num-points must be positive")
    if int(args.anchor_frame) != 4:
        raise ValueError("the cached identity masks are fixed to F04")
    if int(args.future_start_frame) <= int(args.anchor_frame):
        raise ValueError("future supervision must start after the anchor frame")
    if float(args.huber_delta) <= 0.0:
        raise ValueError("huber-delta must be positive")
    for path in (
        args.config,
        args.source_cache / "manifest.json",
        args.cotracker_checkpoint,
    ):
        if not path.expanduser().resolve().is_file():
            raise FileNotFoundError(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return json.loads(
        (args.source_cache.resolve() / "manifest.json").read_text(encoding="utf-8")
    )


def parse_train_args(args: argparse.Namespace):
    config_path = args.config.expanduser().resolve()
    raw, _ = launcher.load_config(config_path)
    config = launcher.validate_config(raw, config_path.parent)
    command = launcher.build_command(config, args.output_root.resolve() / "unused")
    script_index = command.index(str(launcher.XSSC_LOSS_TRAIN_SCRIPT))
    train_args = xssc_trainer.build_parser().parse_args(command[script_index + 1 :])
    train_args.height = 512
    train_args.width = 896
    train_args.train_batch_size = 1
    train_args.mixture_pybullet_ratio = 1.0
    train_args.mixture_kubric_ratio = 0.0
    train_args.mixture_openvid_ratio = 0.0
    # Avoid the transient GPU peak from loading DiT, T5, and the full Wan VAE
    # together. Conditioning stays on CPU until the DiT forward boundary.
    train_args.initialize_model_on_cpu = True
    train_args.xssc_loss_gradient_diagnostics_every_n_forwards = 1_000_000
    return core.tvn.prepare_args(train_args), config


def decode_tiny_vae(
    tiny_vae,
    tiny_vae_apply,
    latents: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    latent_ntchw = latents.permute(0, 2, 1, 3, 4).contiguous()
    with torch.autocast(
        device_type=latent_ntchw.device.type,
        dtype=dtype,
        enabled=latent_ntchw.device.type == "cuda",
    ):
        video = tiny_vae_apply(tiny_vae.decoder, latent_ntchw, False, False)
        if tiny_vae.patch_size > 1:
            video = F.pixel_shuffle(video, tiny_vae.patch_size)
    skip_trim = tiny_vae.is_cogvideox and latent_ntchw.shape[1] % 2 == 0
    if not skip_trim:
        video = video[:, tiny_vae.frames_to_trim :]
    return video


def video_to_uint8(video: torch.Tensor) -> np.ndarray:
    item = video[0].detach().float().clamp(0.0, 1.0)
    return (
        (item * 255.0)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def write_mp4(path: Path, frames: np.ndarray, fps: float) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f"frames must be uint8 [T,H,W,3], got {frames.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=Fraction(str(float(fps))))
    stream.width = int(frames.shape[2])
    stream.height = int(frames.shape[1])
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}
    for image in frames:
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def load_source_frames(case: dict[str, Any]) -> np.ndarray:
    cache_dir = Path(case["cache_dir"])
    with np.load(cache_dir / "source_frames.npz") as archive:
        frames = archive["frames"].astype(np.uint8)
    if frames.shape != (49, 512, 896, 3):
        raise ValueError(f"unexpected source-frame shape: {frames.shape}")
    return frames


def run_forward(args: argparse.Namespace) -> None:
    manifest = load_manifest(args)
    pending = []
    for case in manifest["cases"]:
        complete = args.output_root / "cases" / case["case_key"] / "forward_complete.json"
        if args.overwrite or not complete.is_file():
            pending.append(case)
    if not pending:
        print("[forward] all cases already complete", flush=True)
        return

    set_seed(args.seed)
    train_args, config = parse_train_args(args)
    device = torch.device(args.device)
    accelerator = _Accelerator(device)
    print("[forward] loading merged-OpenVid Full-SA No-Object Wan", flush=True)
    model = core.build_model(train_args, accelerator)
    model.train()
    if model.enable_object_branch:
        raise RuntimeError("diagnostic requires the object branch to be disabled")
    if model.self_attn_adaptation_mode != "full_sa":
        raise RuntimeError("diagnostic requires full_sa adaptation")
    if len(model.merged_pretrained_lora_modules) != 300:
        raise RuntimeError(
            f"expected 300 merged OpenVid modules, got {len(model.merged_pretrained_lora_modules)}"
        )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters != 23_592_960:
        raise RuntimeError(f"unexpected trainable parameter count: {trainable_parameters}")

    pipe = model.pipe
    tiny_vae, tiny_vae_apply = _load_tiny_vae(
        train_args.tiny_vae_root,
        train_args.tiny_vae_checkpoint,
        device,
        pipe.torch_dtype,
    )
    tiny_vae.eval().requires_grad_(False)
    case_metrics = []
    try:
        for position, case in enumerate(manifest["cases"], start=1):
            case_root = args.output_root / "cases" / case["case_key"]
            complete = case_root / "forward_complete.json"
            if complete.is_file() and not args.overwrite:
                case_metrics.append(json.loads((case_root / "forward_metrics.json").read_text()))
                continue
            case_root.mkdir(parents=True, exist_ok=True)
            print(
                f"[forward {position}/{len(manifest['cases'])}] {case['case_key']}",
                flush=True,
            )
            source_frames_for_pipe = wan_tools.load_video_prefix(
                Path(case["source_video"]),
                49,
                512,
                896,
                "cache",
            )
            context_frames = source_frames_for_pipe[:8]
            latent_bundle = torch.load(
                args.source_output / "cases" / case["case_key"] / "latents.pt",
                map_location="cpu",
                weights_only=True,
            )
            target_x0 = latent_bundle["target_x0"].to(
                device=device,
                dtype=pipe.torch_dtype,
            )
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
                seed=int(args.seed) + position,
            )
            captured_inputs = dict(inputs_shared)
            captured_inputs.update(inputs_positive)
            captured_inputs = model.transfer_data_to_device(
                captured_inputs,
                device,
                pipe.torch_dtype,
            )
            pipe.scheduler.set_timesteps(1000, training=True)
            timestep = torch.full(
                (1,),
                float(args.training_timestep),
                device=device,
                dtype=pipe.torch_dtype,
            )
            generator = torch.Generator(device=device)
            generator.manual_seed(int(args.seed) + 100 * position)
            training_noise = torch.randn(
                target_x0.shape,
                generator=generator,
                device=device,
                dtype=target_x0.dtype,
            )
            latent_xt = pipe.scheduler.add_noise(target_x0, training_noise, timestep)
            latent_xt = restore_clean_conditioning(
                latent_xt,
                target_x0,
                captured_inputs,
            )
            pipe.load_models_to_device(pipe.in_iteration_models)
            # Conditioning is prepared on CPU to avoid loading DiT, T5, and VAE
            # onto CUDA together. Move the complete augmented DiT only here.
            pipe.dit.to(device)
            if captured_inputs["context"].device != device:
                raise RuntimeError(
                    f"conditioning stayed on {captured_inputs['context'].device}"
                )
            if next(pipe.dit.text_embedding.parameters()).device != device:
                raise RuntimeError("DiT text embedding did not move to the forward device")
            models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
            main_inputs = dict(captured_inputs)
            main_inputs["latents"] = latent_xt
            with torch.no_grad():
                velocity = pipe.model_fn(**models, **main_inputs, timestep=timestep)
                sigma = context_wan._diffsynth_sigma_for_timestep(
                    pipe.scheduler,
                    timestep,
                ).to(device=device, dtype=target_x0.dtype)
                while sigma.ndim < target_x0.ndim:
                    sigma = sigma.unsqueeze(-1)
                pred_x0 = restore_clean_conditioning(
                    latent_xt - sigma * velocity,
                    target_x0,
                    captured_inputs,
                )
                pred_video = decode_tiny_vae(
                    tiny_vae,
                    tiny_vae_apply,
                    pred_x0,
                    dtype=pipe.torch_dtype,
                )
                target_velocity = pipe.scheduler.training_target(
                    target_x0,
                    training_noise,
                    timestep,
                )
                clean_prefix = context_wan.resolve_num_clean_prefix_latents(
                    clean_prefix_latents=captured_inputs.get("clean_prefix_latents"),
                    num_clean_prefix_latents=captured_inputs.get(
                        "num_clean_prefix_latents"
                    ),
                )
                flow_mse = F.mse_loss(
                    velocity[:, :, clean_prefix:].float(),
                    target_velocity[:, :, clean_prefix:].float(),
                )
                flow_loss = flow_mse * pipe.scheduler.training_weight(timestep).to(
                    flow_mse
                )
            pred_frames = video_to_uint8(pred_video)
            if pred_frames.shape != (49, 512, 896, 3):
                raise RuntimeError(f"unexpected Tiny-VAE output: {pred_frames.shape}")
            np.savez_compressed(case_root / "pred_x0_frames.npz", frames=pred_frames)
            write_mp4(case_root / "pred_x0.mp4", pred_frames, args.fps)
            source_frames = load_source_frames(case)
            write_mp4(case_root / "gt_source.mp4", source_frames, args.fps)
            metrics = {
                "case_key": case["case_key"],
                "caption": case["caption"],
                "model": "Wan2.2-TI2V-5B + merged OpenVid rank-32 LoRA + zero-init Full-SA adapter",
                "pretrained_lora_checkpoint": config["paths"][
                    "pretrained_lora_checkpoint"
                ],
                "merged_pretrained_lora_modules": len(
                    model.merged_pretrained_lora_modules
                ),
                "trainable_full_sa_parameters": trainable_parameters,
                "object_branch": False,
                "dataset_mixture": {"pybullet": 1.0, "kubric": 0.0, "openvid": 0.0},
                "training_timestep": float(args.training_timestep),
                "scheduler_sigma": float(sigma.detach().flatten()[0].item()),
                "flow_mse": float(flow_mse.item()),
                "flow_loss": float(flow_loss.item()),
                "tiny_vae": str(train_args.tiny_vae_checkpoint),
                "pred_frames": int(pred_frames.shape[0]),
                "pred_resolution": [int(pred_frames.shape[1]), int(pred_frames.shape[2])],
            }
            atomic_json(case_root / "forward_metrics.json", metrics)
            atomic_json(complete, {"state": "complete", **metrics})
            case_metrics.append(metrics)
            del (
                source_frames_for_pipe,
                context_frames,
                latent_bundle,
                target_x0,
                inputs_shared,
                inputs_positive,
                captured_inputs,
                training_noise,
                latent_xt,
                velocity,
                pred_x0,
                pred_video,
                pred_frames,
                target_velocity,
            )
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del tiny_vae, tiny_vae_apply, model
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(
        args.output_root / "forward_status.json",
        {"state": "complete", "case_count": len(case_metrics), "cases": case_metrics},
    )


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def prepare_tracker_inputs(
    frames: np.ndarray,
    points_xy_native: np.ndarray,
    *,
    anchor_frame: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    native_height, native_width = frames.shape[1:3]
    video = (
        torch.from_numpy(frames)
        .to(device=device, dtype=torch.float32)
        .permute(0, 3, 1, 2)
    )
    video = F.interpolate(
        video,
        size=(TRACK_HEIGHT, TRACK_WIDTH),
        mode="bilinear",
        align_corners=True,
    ).unsqueeze(0)
    points = torch.from_numpy(points_xy_native).to(device=device, dtype=torch.float32)
    points[:, 0] *= float(TRACK_WIDTH - 1) / float(native_width - 1)
    points[:, 1] *= float(TRACK_HEIGHT - 1) / float(native_height - 1)
    frame_ids = torch.full(
        (points.shape[0], 1),
        float(anchor_frame),
        device=device,
        dtype=points.dtype,
    )
    return video, torch.cat((frame_ids, points), dim=-1).unsqueeze(0)


@torch.no_grad()
def track_video(predictor, video: torch.Tensor, queries: torch.Tensor):
    return predictor(video, queries=queries, backward_tracking=False)


def tracker_input_gradient_audit(
    predictor,
    pred_frames: np.ndarray,
    points_xy: np.ndarray,
    gt_tracks: torch.Tensor,
    gt_visibility: torch.Tensor,
    args: argparse.Namespace,
) -> float:
    device = torch.device(args.device)
    pred_video, queries = prepare_tracker_inputs(
        pred_frames,
        points_xy,
        anchor_frame=args.anchor_frame,
        device=device,
    )
    pred_video.requires_grad_(True)
    pred_tracks, _ = predictor._compute_sparse_tracks(
        video=pred_video,
        queries=queries,
        add_support_grid=True,
        backward_tracking=False,
    )
    loss, _ = object_trajectory_loss(
        pred_tracks,
        gt_tracks,
        gt_visibility,
        height=TRACK_HEIGHT,
        width=TRACK_WIDTH,
        anchor_frame=args.anchor_frame,
        future_start_frame=args.future_start_frame,
        huber_delta=args.huber_delta,
    )
    gradient = torch.autograd.grad(loss, pred_video, retain_graph=False)[0]
    norm = float(gradient.detach().float().norm().item())
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(f"CoTracker loss has invalid input-gradient norm: {norm}")
    return norm


def run_tracks(args: argparse.Namespace) -> None:
    from cotracker.predictor import CoTrackerPredictor

    manifest = load_manifest(args)
    device = torch.device(args.device)
    print("[tracks] loading frozen CoTracker3", flush=True)
    predictor = (
        CoTrackerPredictor(
            checkpoint=str(args.cotracker_checkpoint.resolve()),
            offline=True,
            v2=False,
            window_len=60,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    summaries = []
    gradient_audited = False
    try:
        for position, case in enumerate(manifest["cases"], start=1):
            case_root = args.output_root / "cases" / case["case_key"]
            complete = case_root / "tracks_complete.json"
            if complete.is_file() and not args.overwrite:
                summaries.append(json.loads((case_root / "metrics.json").read_text()))
                continue
            if not (case_root / "forward_complete.json").is_file():
                raise FileNotFoundError(f"run forward first: {case_root}")
            print(
                f"[tracks {position}/{len(manifest['cases'])}] {case['case_key']}",
                flush=True,
            )
            source_frames = load_source_frames(case)
            with np.load(case_root / "pred_x0_frames.npz") as archive:
                pred_frames = archive["frames"].astype(np.uint8)
            with np.load(Path(case["cache_dir"]) / "sam2_masks.npz") as archive:
                identity_mask = archive["selected_identity_mask"].astype(np.uint8)
            points_xy = sample_points_from_mask(
                identity_mask,
                int(args.num_points),
                avoid_edges=True,
            )
            if points_xy.shape != (int(args.num_points), 2):
                raise RuntimeError(f"invalid object query shape: {points_xy.shape}")
            gt_video, queries = prepare_tracker_inputs(
                source_frames,
                points_xy,
                anchor_frame=args.anchor_frame,
                device=device,
            )
            pred_video, pred_queries = prepare_tracker_inputs(
                pred_frames,
                points_xy,
                anchor_frame=args.anchor_frame,
                device=device,
            )
            with torch.inference_mode():
                gt_tracks, gt_visibility = track_video(predictor, gt_video, queries)
                pred_tracks, pred_visibility = track_video(
                    predictor,
                    pred_video,
                    pred_queries,
                )
            visible_loss, visible_diagnostics = object_trajectory_loss(
                pred_tracks,
                gt_tracks,
                gt_visibility,
                height=TRACK_HEIGHT,
                width=TRACK_WIDTH,
                anchor_frame=args.anchor_frame,
                future_start_frame=args.future_start_frame,
                huber_delta=args.huber_delta,
            )
            all_point_visibility = torch.ones_like(gt_visibility, dtype=torch.bool)
            loss, diagnostics = object_trajectory_loss(
                pred_tracks,
                gt_tracks,
                all_point_visibility,
                height=TRACK_HEIGHT,
                width=TRACK_WIDTH,
                anchor_frame=args.anchor_frame,
                future_start_frame=args.future_start_frame,
                huber_delta=args.huber_delta,
            )
            gradient_norm = None
            if args.gradient_audit and not gradient_audited:
                print("[tracks] auditing dL/d(predicted RGB video)", flush=True)
                gradient_norm = tracker_input_gradient_audit(
                    predictor,
                    pred_frames,
                    points_xy,
                    gt_tracks.detach(),
                    all_point_visibility,
                    args,
                )
                gradient_audited = True

            valid = visible_diagnostics["valid_future"]
            gt_norm = gt_tracks.float() / gt_tracks.new_tensor(
                (TRACK_WIDTH - 1, TRACK_HEIGHT - 1)
            )
            pred_norm = pred_tracks.float() / pred_tracks.new_tensor(
                (TRACK_WIDTH - 1, TRACK_HEIGHT - 1)
            )
            gt_disp = gt_norm - gt_norm[:, args.anchor_frame : args.anchor_frame + 1]
            pred_disp = pred_norm - pred_norm[:, args.anchor_frame : args.anchor_frame + 1]
            error = torch.linalg.vector_norm(
                pred_disp[:, args.future_start_frame :]
                - gt_disp[:, args.future_start_frame :],
                dim=-1,
            )
            pred_visible_future = pred_visibility[:, args.future_start_frame :]
            joint_visible = valid & pred_visible_future
            forward_metrics = json.loads(
                (case_root / "forward_metrics.json").read_text(encoding="utf-8")
            )
            metrics = {
                **forward_metrics,
                "trajectory_extractor": "frozen CoTracker3 scaled_offline",
                "trajectory_checkpoint": str(args.cotracker_checkpoint.resolve()),
                "object_query_source": "cached GT F04 SAM2 identity mask",
                "object_query_points": int(points_xy.shape[0]),
                "anchor_frame": int(args.anchor_frame),
                "future_frames": [int(args.future_start_frame), 48],
                "trajectory_definition": "normalized displacement relative to F04",
                "loss_primary_validity": "all selected object-query tracks",
                "loss_audit_validity": "GT CoTracker visibility > 0.9",
                "trajectory_loss_type": "smooth_l1",
                "trajectory_loss_beta": float(args.huber_delta),
                "huber_delta": float(args.huber_delta),
                "trajectory_loss": float(loss.item()),
                "trajectory_loss_gt_visible": float(visible_loss.item()),
                "trajectory_smooth_l1": float(loss.item()),
                "trajectory_smooth_l1_gt_visible": float(visible_loss.item()),
                "trajectory_huber": float(diagnostics["raw_huber"].item()),
                "trajectory_huber_gt_visible": float(
                    visible_diagnostics["raw_huber"].item()
                ),
                "normalized_ade": float(diagnostics["normalized_ade"].item()),
                "normalized_ade_gt_visible": float(
                    visible_diagnostics["normalized_ade"].item()
                ),
                "normalized_rmse": float(diagnostics["normalized_rmse"].item()),
                "normalized_rmse_gt_visible": float(
                    visible_diagnostics["normalized_rmse"].item()
                ),
                "normalized_gt_motion": float(
                    diagnostics["normalized_gt_motion"].item()
                ),
                "gt_visible_fraction": float(
                    visible_diagnostics["valid_fraction"].item()
                ),
                "pred_visible_on_gt_valid_fraction": float(
                    pred_visible_future[valid].float().mean().item()
                ),
                "joint_visible_fraction": float(joint_visible.float().mean().item()),
                "all_future_point_frames": int(diagnostics["valid_count"].item()),
                "gt_visible_future_point_frames": int(
                    visible_diagnostics["valid_count"].item()
                ),
                "tracker_input_gradient_norm": gradient_norm,
            }
            np.savez_compressed(
                case_root / "trajectories.npz",
                query_points_xy_native=points_xy.astype(np.float32),
                identity_mask=identity_mask.astype(np.uint8),
                gt_tracks_trackres=gt_tracks[0].float().cpu().numpy(),
                pred_tracks_trackres=pred_tracks[0].float().cpu().numpy(),
                gt_visibility=gt_visibility[0].cpu().numpy().astype(np.uint8),
                pred_visibility=pred_visibility[0].cpu().numpy().astype(np.uint8),
                per_frame_trajectory_loss=diagnostics["per_frame_loss"][0]
                .cpu()
                .numpy(),
                per_frame_huber_raw=diagnostics["per_frame_raw_huber"][0]
                .cpu()
                .numpy(),
                per_frame_ade=diagnostics["per_frame_ade"][0].cpu().numpy(),
                per_frame_ade_gt_visible=visible_diagnostics["per_frame_ade"][0]
                .cpu()
                .numpy(),
                per_point_future_error=error[0].cpu().numpy(),
                valid_future=valid[0].cpu().numpy().astype(np.uint8),
            )
            atomic_json(case_root / "metrics.json", metrics)
            atomic_json(complete, {"state": "complete", **metrics})
            summaries.append(metrics)
            del (
                source_frames,
                pred_frames,
                identity_mask,
                gt_video,
                pred_video,
                queries,
                pred_queries,
                gt_tracks,
                pred_tracks,
                gt_visibility,
                pred_visibility,
                loss,
                diagnostics,
                visible_loss,
                visible_diagnostics,
            )
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(
        args.output_root / "tracks_status.json",
        {"state": "complete", "case_count": len(summaries), "cases": summaries},
    )


def trackres_to_native(tracks: np.ndarray, height: int, width: int) -> np.ndarray:
    result = tracks.astype(np.float32).copy()
    result[..., 0] *= float(width - 1) / float(TRACK_WIDTH - 1)
    result[..., 1] *= float(height - 1) / float(TRACK_HEIGHT - 1)
    return result


def draw_track_history(
    frame: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    frame_id: int,
    *,
    anchor_frame: int,
    colors: np.ndarray,
    ring: bool = False,
) -> None:
    start = min(int(anchor_frame), int(frame_id))
    stop = max(int(anchor_frame), int(frame_id))
    for point_id in range(tracks.shape[1]):
        color = tuple(int(v) for v in colors[point_id % len(colors)])
        history = []
        for index in range(start, stop + 1):
            if bool(visibility[index, point_id]):
                history.append(tuple(np.rint(tracks[index, point_id]).astype(int)))
            elif len(history) >= 2:
                cv2.polylines(frame, [np.asarray(history)], False, color, 2, cv2.LINE_AA)
                history = []
        if len(history) >= 2:
            cv2.polylines(frame, [np.asarray(history)], False, color, 2, cv2.LINE_AA)
        if bool(visibility[frame_id, point_id]):
            point = tuple(np.rint(tracks[frame_id, point_id]).astype(int))
            if ring:
                cv2.circle(frame, point, 7, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, point, 5, color, -1, cv2.LINE_AA)
                cv2.circle(frame, point, 7, (10, 10, 10), 1, cv2.LINE_AA)


def add_panel_label(frame: np.ndarray, title: str, detail: str) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 56), (8, 12, 16), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0.0, frame)
    cv2.putText(
        frame,
        title,
        (14, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        detail,
        (14, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (190, 207, 218),
        1,
        cv2.LINE_AA,
    )


def render_case(args: argparse.Namespace, case: dict[str, Any]) -> None:
    case_root = args.output_root / "cases" / case["case_key"]
    metrics = json.loads((case_root / "metrics.json").read_text(encoding="utf-8"))
    source_frames = load_source_frames(case)
    with np.load(case_root / "pred_x0_frames.npz") as archive:
        pred_frames = archive["frames"].astype(np.uint8)
    with np.load(case_root / "trajectories.npz") as archive:
        gt_tracks = archive["gt_tracks_trackres"].astype(np.float32)
        pred_tracks = archive["pred_tracks_trackres"].astype(np.float32)
        gt_visibility = archive["gt_visibility"].astype(bool)
        pred_visibility = archive["pred_visibility"].astype(bool)
        per_frame_loss = archive["per_frame_trajectory_loss"].astype(np.float32)
        per_frame_huber_raw = archive["per_frame_huber_raw"].astype(np.float32)
        per_frame_ade = archive["per_frame_ade"].astype(np.float32)
        identity_mask = archive["identity_mask"].astype(bool)
    height, width = source_frames.shape[1:3]
    gt_native = trackres_to_native(gt_tracks, height, width)
    pred_native = trackres_to_native(pred_tracks, height, width)
    frames = []
    for frame_id in range(source_frames.shape[0]):
        gt_panel = source_frames[frame_id].copy()
        pred_panel = pred_frames[frame_id].copy()
        compare_panel = pred_frames[frame_id].copy()
        if frame_id == int(args.anchor_frame):
            tint = np.asarray((45, 210, 128), dtype=np.float32)
            gt_panel[identity_mask] = np.clip(
                0.60 * gt_panel[identity_mask].astype(np.float32) + 0.40 * tint,
                0,
                255,
            ).astype(np.uint8)
        draw_track_history(
            gt_panel,
            gt_native,
            gt_visibility,
            frame_id,
            anchor_frame=args.anchor_frame,
            colors=PALETTE,
        )
        draw_track_history(
            pred_panel,
            pred_native,
            pred_visibility,
            frame_id,
            anchor_frame=args.anchor_frame,
            colors=PALETTE,
        )
        draw_track_history(
            compare_panel,
            gt_native,
            gt_visibility,
            frame_id,
            anchor_frame=args.anchor_frame,
            colors=PALETTE,
            ring=True,
        )
        draw_track_history(
            compare_panel,
            pred_native,
            pred_visibility,
            frame_id,
            anchor_frame=args.anchor_frame,
            colors=PALETTE,
        )
        if frame_id >= int(args.future_start_frame):
            for point_id in range(gt_native.shape[1]):
                if not bool(gt_visibility[frame_id, point_id]):
                    continue
                start = tuple(np.rint(gt_native[frame_id, point_id]).astype(int))
                stop = tuple(np.rint(pred_native[frame_id, point_id]).astype(int))
                cv2.line(compare_panel, start, stop, (255, 72, 72), 1, cv2.LINE_AA)
        frame_metrics = (
            f"F{frame_id:02d} | context (not supervised) | "
            f"clip L={metrics['trajectory_loss']:.5f} | "
            f"raw={metrics['trajectory_huber']:.2e}"
        )
        if frame_id >= int(args.future_start_frame):
            future_id = frame_id - int(args.future_start_frame)
            loss_value = per_frame_loss[future_id]
            raw_value = per_frame_huber_raw[future_id]
            ade_value = per_frame_ade[future_id]
            frame_metrics = (
                f"F{frame_id:02d} | Lframe={loss_value:.5f} | "
                f"Lclip={metrics['trajectory_loss']:.5f} | "
                f"raw={raw_value:.2e} | ADE={ade_value:.4f}"
            )
        add_panel_label(gt_panel, "GT RGB + GT CoTracker", f"F{frame_id:02d} | object points")
        add_panel_label(
            pred_panel,
            "Tiny-VAE x0_pred + Pred CoTracker",
            f"F{frame_id:02d} | colored predicted tracks",
        )
        add_panel_label(
            compare_panel,
            "Trajectory loss overlay",
            frame_metrics,
        )
        frames.append(np.concatenate((gt_panel, pred_panel, compare_panel), axis=1))
    rendered = np.stack(frames).astype(np.uint8)
    write_mp4(case_root / "object_trajectory_overlay.mp4", rendered, args.fps)
    anchor_preview = rendered[int(args.anchor_frame)]
    cv2.imwrite(
        str(case_root / "object_query_preview.jpg"),
        cv2.cvtColor(anchor_preview, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )
    future_preview_index = min(32, int(rendered.shape[0]) - 1)
    cv2.imwrite(
        str(case_root / "trajectory_future_preview.jpg"),
        cv2.cvtColor(rendered[future_preview_index], cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )
    atomic_json(
        case_root / "render_complete.json",
        {
            "state": "complete",
            "case_key": case["case_key"],
            "frames": int(rendered.shape[0]),
            "resolution": [int(rendered.shape[1]), int(rendered.shape[2])],
            "trajectory_loss": metrics["trajectory_loss"],
            "trajectory_huber": metrics["trajectory_huber"],
        },
    )


def render_report(args: argparse.Namespace) -> Path:
    manifest = load_manifest(args)
    rows = []
    summary_metrics = []
    for position, case in enumerate(manifest["cases"], start=1):
        case_root = args.output_root / "cases" / case["case_key"]
        render_case(args, case)
        metrics = json.loads((case_root / "metrics.json").read_text(encoding="utf-8"))
        summary_metrics.append(metrics)
        relative = case_root.relative_to(args.output_root)
        gradient = metrics.get("tracker_input_gradient_norm")
        gradient_text = "not run" if gradient is None else f"{float(gradient):.3e}"
        rows.append(
            f"""
<section class="case">
  <div class="case-head"><div><span class="case-index">CASE {position:02d}</span>
  <h2>{html.escape(case['case_key'])}</h2><p>{html.escape(case['caption'])}</p></div>
  <div class="loss"><small>ALL-POINT SMOOTH L1</small><strong>{metrics['trajectory_loss']:.6f}</strong><small>visible {metrics['trajectory_loss_gt_visible']:.6f} · raw Huber {metrics['trajectory_huber']:.2e}</small></div></div>
  <div class="metrics">
    <span><b>{metrics['normalized_ade']:.4f}</b> all-point ADE</span>
    <span><b>{metrics['normalized_ade_gt_visible']:.4f}</b> visible ADE</span>
    <span><b>{metrics['normalized_rmse']:.4f}</b> normalized RMSE</span>
    <span><b>{metrics['normalized_gt_motion']:.4f}</b> GT motion</span>
    <span><b>{metrics['gt_visible_fraction']:.1%}</b> GT visible</span>
    <span><b>{metrics['trajectory_huber']:.2e}</b> raw Huber · grad {gradient_text}</span>
  </div>
  <video controls muted loop playsinline preload="metadata" poster="{relative}/trajectory_future_preview.jpg" src="{relative}/object_trajectory_overlay.mp4"></video>
  <div class="media-row">
    <figure><img src="{relative}/object_query_preview.jpg"><figcaption>F04 object queries</figcaption></figure>
    <figure><video controls muted loop playsinline preload="metadata" src="{relative}/gt_source.mp4"></video><figcaption>Original PyBullet GT</figcaption></figure>
    <figure><video controls muted loop playsinline preload="metadata" src="{relative}/pred_x0.mp4"></video><figcaption>Tiny-VAE decoded x0_pred</figcaption></figure>
  </div>
  <div class="links"><a href="{relative}/metrics.json">metrics.json</a><a href="{relative}/trajectories.npz">trajectories.npz</a></div>
</section>"""
        )
    mean_loss = float(np.mean([item["trajectory_loss"] for item in summary_metrics]))
    mean_visible_loss = float(
        np.mean([item["trajectory_loss_gt_visible"] for item in summary_metrics])
    )
    mean_raw_huber = float(
        np.mean([item["trajectory_huber"] for item in summary_metrics])
    )
    mean_ade = float(np.mean([item["normalized_ade"] for item in summary_metrics]))
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Object trajectory loss diagnostics</title><style>
:root{{--bg:#eef2f3;--ink:#142026;--muted:#607078;--line:#c7d0d3;--paper:#fff;--red:#c73532;--green:#16745a;--blue:#26648b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 "IBM Plex Sans","Noto Sans SC",sans-serif;letter-spacing:0}}
header{{background:#172329;color:#f6fafb;border-bottom:5px solid var(--green)}}.mast{{max-width:1500px;margin:auto;padding:28px 28px 24px;display:grid;grid-template-columns:1fr auto;gap:30px;align-items:end}}
h1{{font:700 34px/1.1 "IBM Plex Sans Condensed","Noto Sans SC",sans-serif;margin:0 0 8px;letter-spacing:0}}header p{{margin:0;color:#b9c7cb}}.summary{{display:flex;gap:28px}}.summary small,.loss small{{display:block;color:#91a3a8;font-size:11px}}.summary strong{{font-size:25px}}
main{{max-width:1500px;margin:auto;padding:22px 28px 80px}}.case{{padding:25px 0 34px;border-bottom:1px solid var(--line)}}.case-head{{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:14px}}.case-index{{font-size:11px;color:var(--green);font-weight:800}}h2{{margin:2px 0 2px;font-size:23px}}.case-head p{{margin:0;color:var(--muted)}}.loss{{text-align:right}}.loss strong{{display:block;color:var(--red);font-size:27px}}
.metrics{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));border:1px solid var(--line);background:var(--paper);margin-bottom:14px}}.metrics span{{padding:10px 12px;border-right:1px solid var(--line);color:var(--muted)}}.metrics span:last-child{{border:0}}.metrics b{{display:block;color:var(--ink);font-size:17px}}
video,img{{display:block;width:100%;background:#050708}}.case>video{{border:1px solid #29383f;aspect-ratio:21/4;object-fit:contain}}.media-row{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:14px}}figure{{margin:0;background:var(--paper);border:1px solid var(--line)}}figure video,figure img{{aspect-ratio:16/9;object-fit:contain}}figcaption{{padding:8px 10px;color:var(--muted)}}.links{{display:flex;gap:16px;margin-top:10px}}a{{color:var(--blue);font-weight:700}}
#replay{{position:fixed;right:22px;bottom:20px;border:1px solid #0e4d3d;background:var(--green);color:white;padding:11px 17px;font-weight:750;cursor:pointer}}
@media(max-width:900px){{.mast{{grid-template-columns:1fr}}.summary{{justify-content:flex-start}}.metrics{{grid-template-columns:repeat(2,1fr)}}.metrics span{{border-bottom:1px solid var(--line)}}.media-row{{grid-template-columns:1fr}}.case-head{{display:block}}.loss{{text-align:left;margin-top:10px}}.case>video{{aspect-ratio:16/9}}}}
</style></head><body><header><div class="mast"><div><h1>PyBullet Object Trajectory Loss</h1><p>Full-SA No-Object · merged OpenVid initialization · t={args.training_timestep:g} · F04 SAM2 object points n={args.num_points} · F08–F48 displacement · Smooth L1 β={args.huber_delta:g} · all points vs GT-visible audit</p></div><div class="summary"><div><small>MEAN SMOOTH L1</small><strong>{mean_loss:.6f}</strong></div><div><small>MEAN RAW HUBER</small><strong>{mean_raw_huber:.2e}</strong></div><div><small>MEAN ADE</small><strong>{mean_ade:.4f}</strong></div></div></div></header><main>{''.join(rows)}</main><button id="replay">Replay all</button><script>
document.getElementById('replay').onclick=()=>document.querySelectorAll('video').forEach(v=>{{v.currentTime=0;v.play().catch(()=>{{}})}});
</script></body></html>"""
    args.output_root.mkdir(parents=True, exist_ok=True)
    index_path = args.output_root / "index.html"
    index_path.write_text(document, encoding="utf-8")
    atomic_json(
        args.output_root / "report_status.json",
        {
            "state": "complete",
            "case_count": len(summary_metrics),
            "mean_trajectory_loss": mean_loss,
            "mean_trajectory_loss_gt_visible": mean_visible_loss,
            "mean_trajectory_huber": mean_raw_huber,
            "mean_trajectory_huber_gt_visible": float(
                np.mean(
                    [item["trajectory_huber_gt_visible"] for item in summary_metrics]
                )
            ),
            "mean_normalized_ade": mean_ade,
            "index": str(index_path),
        },
    )
    return index_path


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_root = args.output_root.expanduser().resolve()
    args.source_cache = args.source_cache.expanduser().resolve()
    args.source_output = args.source_output.expanduser().resolve()
    if args.mode in {"forward", "all"}:
        run_forward(args)
    if args.mode in {"tracks", "all"}:
        run_tracks(args)
    if args.mode in {"render", "all"}:
        index_path = render_report(args)
        print(json.dumps({"index": str(index_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
