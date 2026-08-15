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
from object_trajectory_loss import (
    object_trajectory_loss,
    visibility_aware_trajectory_loss,
)
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
DEFAULT_MULTI_OBJECT_CACHE = Path(
    "/data/gaoya/agent-data/cache/uniform_multiobject_correspondence_diagnostics"
)
TRACK_HEIGHT = 256
TRACK_WIDTH = 448
OBJECT_COLORS = np.asarray(
    [
        (238, 75, 71),
        (17, 150, 141),
        (45, 107, 185),
        (226, 167, 36),
        (154, 83, 170),
        (80, 170, 80),
    ],
    dtype=np.uint8,
)
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
    parser.add_argument(
        "--multiobject-cache",
        type=Path,
        default=None,
        help="Prepared per-object F04 masks; when set, aggregate loss per object.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-timestep", type=float, default=500.0)
    parser.add_argument("--num-points", type=int, default=24)
    parser.add_argument("--anchor-frame", type=int, default=4)
    parser.add_argument("--future-start-frame", type=int, default=8)
    parser.add_argument("--huber-delta", type=float, default=0.01)
    parser.add_argument("--visibility-threshold", type=float, default=0.9)
    parser.add_argument("--visibility-loss-weight", type=float, default=0.05)
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
    if not 0.0 < float(args.visibility_threshold) < 1.0:
        raise ValueError("visibility-threshold must be in (0, 1)")
    if float(args.visibility_loss_weight) < 0.0:
        raise ValueError("visibility-loss-weight must be non-negative")
    for path in (
        args.config,
        args.source_cache / "manifest.json",
        args.cotracker_checkpoint,
    ):
        if not path.expanduser().resolve().is_file():
            raise FileNotFoundError(path)
    if args.multiobject_cache is not None:
        cache_root = args.multiobject_cache.expanduser().resolve()
        if not (cache_root / "objects_status.json").is_file():
            raise FileNotFoundError(cache_root / "objects_status.json")


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


def track_video_with_scores(
    predictor,
    video: torch.Tensor,
    queries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror sparse CoTracker inference while retaining raw visibility/confidence."""
    from cotracker.models.core.model_utils import get_points_on_a_grid

    batch, frames, channels, height, width = video.shape
    resized = F.interpolate(
        video.reshape(batch * frames, channels, height, width),
        tuple(predictor.interp_shape),
        mode="bilinear",
        align_corners=True,
    ).reshape(
        batch,
        frames,
        channels,
        predictor.interp_shape[0],
        predictor.interp_shape[1],
    )
    scaled_queries = queries.clone()
    scaled_queries[:, :, 1:] *= scaled_queries.new_tensor(
        (
            (predictor.interp_shape[1] - 1) / (width - 1),
            (predictor.interp_shape[0] - 1) / (height - 1),
        )
    )
    support = get_points_on_a_grid(
        predictor.support_grid_size,
        predictor.interp_shape,
        device=video.device,
    )
    support = torch.cat((torch.zeros_like(support[:, :, :1]), support), dim=-1)
    support = support.repeat(batch, 1, 1)
    model_queries = torch.cat((scaled_queries, support), dim=1)
    tracks, visibility, confidence, _ = predictor.model.forward(
        video=resized,
        queries=model_queries,
        iters=6,
    )
    query_count = int(queries.shape[1])
    tracks = tracks[:, :, :query_count]
    visibility = visibility[:, :, :query_count]
    confidence = confidence[:, :, :query_count]
    tracks, visibility, confidence = replace_query_predictions(
        tracks,
        visibility,
        confidence,
        scaled_queries,
    )
    tracks = tracks * tracks.new_tensor(
        (
            (width - 1) / (predictor.interp_shape[1] - 1),
            (height - 1) / (predictor.interp_shape[0] - 1),
        )
    )
    return tracks, visibility, confidence


def replace_query_predictions(
    tracks: torch.Tensor,
    visibility: torch.Tensor,
    confidence: torch.Tensor,
    scaled_queries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Set query-frame predictions without mutating differentiable model outputs."""
    anchor_mask = torch.zeros_like(visibility, dtype=torch.bool)
    anchor_tracks = torch.zeros_like(tracks, requires_grad=False)
    for batch_index in range(tracks.shape[0]):
        query_frames = scaled_queries[batch_index, :, 0].long()
        point_ids = torch.arange(tracks.shape[2], device=tracks.device)
        anchor_mask[batch_index, query_frames, point_ids] = True
        anchor_tracks[batch_index, query_frames, point_ids] = scaled_queries[
            batch_index, :, 1:
        ]
    return (
        torch.where(anchor_mask.unsqueeze(-1), anchor_tracks, tracks),
        torch.where(anchor_mask, torch.ones_like(visibility), visibility),
        torch.where(anchor_mask, torch.ones_like(confidence), confidence),
    )


def load_object_specs(
    args: argparse.Namespace,
    case: dict[str, Any],
    identity_mask: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray]:
    """Return F04 per-object masks and stable captions for query grouping."""
    if args.multiobject_cache is None:
        if identity_mask is None:
            raise ValueError("identity_mask is required without multiobject-cache")
        return ["selected object"], identity_mask[None].astype(np.uint8)

    case_cache = args.multiobject_cache / "cases" / case["case_key"]
    metadata = json.loads(
        (case_cache / "objects_complete.json").read_text(encoding="utf-8")
    )
    with np.load(case_cache / "object_masks.npz") as archive:
        masks_othw = archive["masks_othw"].astype(np.uint8)
    f04_masks = masks_othw[:, int(args.anchor_frame)]
    phrases = [str(value) for value in metadata["object_phrases"]]
    if f04_masks.shape[0] != len(phrases):
        raise RuntimeError(
            f"object mask/phrase mismatch for {case['case_key']}: "
            f"{f04_masks.shape[0]} masks/{len(phrases)} phrases"
        )
    if not bool(f04_masks.any(axis=(1, 2)).all()):
        raise RuntimeError(f"empty F04 object mask in {case['case_key']}")
    return phrases, f04_masks


def aggregate_object_trajectory_losses(
    pred_tracks: torch.Tensor,
    gt_tracks: torch.Tensor,
    gt_visibility: torch.Tensor,
    *,
    object_count: int,
    points_per_object: int,
    height: int,
    width: int,
    anchor_frame: int,
    future_start_frame: int,
    huber_delta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Compute point means inside each object, then an equal object mean."""
    if pred_tracks.shape[2] != int(object_count) * int(points_per_object):
        raise ValueError(
            "track query count does not match object grouping: "
            f"{pred_tracks.shape[2]} vs {object_count}x{points_per_object}"
        )
    all_losses = []
    visible_losses = []
    all_diagnostics = []
    visible_diagnostics = []
    object_rows = []
    for object_index in range(int(object_count)):
        start = object_index * int(points_per_object)
        stop = start + int(points_per_object)
        pred_object = pred_tracks[:, :, start:stop]
        gt_object = gt_tracks[:, :, start:stop]
        visibility_object = gt_visibility[:, :, start:stop]
        object_loss, object_diag = object_trajectory_loss(
            pred_object,
            gt_object,
            torch.ones_like(visibility_object, dtype=torch.bool),
            height=height,
            width=width,
            anchor_frame=anchor_frame,
            future_start_frame=future_start_frame,
            huber_delta=huber_delta,
        )
        visible_loss, visible_diag = object_trajectory_loss(
            pred_object,
            gt_object,
            visibility_object,
            height=height,
            width=width,
            anchor_frame=anchor_frame,
            future_start_frame=future_start_frame,
            huber_delta=huber_delta,
        )
        all_losses.append(object_loss)
        visible_losses.append(visible_loss)
        all_diagnostics.append(object_diag)
        visible_diagnostics.append(visible_diag)
        object_rows.append(
            {
                "object_index": torch.tensor(object_index),
                "loss": object_loss,
                "loss_gt_visible": visible_loss,
                "raw_huber": object_diag["raw_huber"],
                "raw_huber_gt_visible": visible_diag["raw_huber"],
                "normalized_ade": object_diag["normalized_ade"],
                "normalized_ade_gt_visible": visible_diag["normalized_ade"],
                "normalized_rmse": object_diag["normalized_rmse"],
                "normalized_gt_motion": object_diag["normalized_gt_motion"],
                "gt_visible_fraction": visible_diag["valid_fraction"],
                "all_future_point_frames": object_diag["valid_count"],
                "gt_visible_future_point_frames": visible_diag["valid_count"],
            }
        )

    def mean_diagnostic(name: str) -> torch.Tensor:
        return torch.stack([item[name] for item in all_diagnostics]).mean(dim=0)

    def mean_visible_diagnostic(name: str) -> torch.Tensor:
        return torch.stack([item[name] for item in visible_diagnostics]).mean(dim=0)

    diagnostics = {
        "normalized_ade": mean_diagnostic("normalized_ade"),
        "normalized_rmse": mean_diagnostic("normalized_rmse"),
        "normalized_gt_motion": mean_diagnostic("normalized_gt_motion"),
        "raw_huber": torch.stack(
            [item["raw_huber"] for item in all_diagnostics]
        ).mean(),
        "valid_fraction": torch.stack(
            [item["valid_fraction"] for item in all_diagnostics]
        ).mean(),
        "valid_count": torch.stack(
            [item["valid_count"] for item in all_diagnostics]
        ).sum(),
        "per_frame_loss": torch.stack(
            [item["per_frame_loss"] for item in all_diagnostics]
        ).mean(dim=0),
        "per_frame_raw_huber": torch.stack(
            [item["per_frame_raw_huber"] for item in all_diagnostics]
        ).mean(dim=0),
        "per_frame_ade": torch.stack(
            [item["per_frame_ade"] for item in all_diagnostics]
        ).mean(dim=0),
        "per_point_distance": torch.cat(
            [item["per_point_distance"] for item in all_diagnostics], dim=-1
        ),
        "valid_future": torch.cat(
            [item["valid_future"] for item in all_diagnostics], dim=-1
        ),
    }
    visible_loss = torch.stack(visible_losses).mean()
    visible_aggregate = {
        "normalized_ade": mean_visible_diagnostic("normalized_ade"),
        "normalized_rmse": mean_visible_diagnostic("normalized_rmse"),
        "normalized_gt_motion": mean_visible_diagnostic("normalized_gt_motion"),
        "raw_huber": torch.stack(
            [item["raw_huber"] for item in visible_diagnostics]
        ).mean(),
        "valid_fraction": torch.stack(
            [item["valid_fraction"] for item in visible_diagnostics]
        ).mean(),
        "valid_count": torch.stack(
            [item["valid_count"] for item in visible_diagnostics]
        ).sum(),
        "per_frame_loss": torch.stack(
            [item["per_frame_loss"] for item in visible_diagnostics]
        ).mean(dim=0),
        "per_frame_raw_huber": torch.stack(
            [item["per_frame_raw_huber"] for item in visible_diagnostics]
        ).mean(dim=0),
        "per_frame_ade": torch.stack(
            [item["per_frame_ade"] for item in visible_diagnostics]
        ).mean(dim=0),
        "valid_future": torch.cat(
            [item["valid_future"] for item in visible_diagnostics], dim=-1
        ),
    }
    diagnostics["visible_aggregate"] = visible_aggregate
    return torch.stack(all_losses).mean(), diagnostics, object_rows


def aggregate_visibility_aware_object_losses(
    pred_tracks: torch.Tensor,
    gt_tracks: torch.Tensor,
    gt_visibility_probability: torch.Tensor,
    gt_confidence_probability: torch.Tensor,
    pred_visibility_probability: torch.Tensor,
    *,
    object_count: int,
    points_per_object: int,
    height: int,
    width: int,
    anchor_frame: int,
    future_start_frame: int,
    huber_delta: float,
    visibility_threshold: float,
    visibility_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    totals = []
    rows = []
    for object_index in range(int(object_count)):
        start = object_index * int(points_per_object)
        stop = start + int(points_per_object)
        total, diagnostics = visibility_aware_trajectory_loss(
            pred_tracks[:, :, start:stop],
            gt_tracks[:, :, start:stop],
            gt_visibility_probability[:, :, start:stop],
            gt_confidence_probability[:, :, start:stop],
            pred_visibility_probability[:, :, start:stop],
            height=height,
            width=width,
            anchor_frame=anchor_frame,
            future_start_frame=future_start_frame,
            huber_delta=huber_delta,
            visibility_threshold=visibility_threshold,
            visibility_loss_weight=visibility_loss_weight,
        )
        totals.append(total)
        rows.append({"object_index": object_index, **diagnostics})

    def object_mean(name: str) -> torch.Tensor:
        return torch.stack([row[name] for row in rows]).mean(dim=0)

    def frame_mean(name: str) -> torch.Tensor:
        return torch.nanmean(torch.stack([row[name] for row in rows]), dim=0)

    aggregate = {
        "coordinate_loss": object_mean("coordinate_loss"),
        "visibility_loss": object_mean("visibility_loss"),
        "total_loss": object_mean("total_loss"),
        "raw_huber": object_mean("raw_huber"),
        "normalized_ade": object_mean("normalized_ade"),
        "normalized_rmse": object_mean("normalized_rmse"),
        "normalized_gt_motion": object_mean("normalized_gt_motion"),
        "valid_fraction": object_mean("valid_fraction"),
        "valid_count": torch.stack([row["valid_count"] for row in rows]).sum(),
        "effective_weight_sum": torch.stack(
            [row["effective_weight_sum"] for row in rows]
        ).sum(),
        "effective_weight_fraction": object_mean("effective_weight_fraction"),
        "mean_gt_visibility_probability": object_mean(
            "mean_gt_visibility_probability"
        ),
        "mean_gt_confidence_probability": object_mean(
            "mean_gt_confidence_probability"
        ),
        "mean_pred_visibility_probability": object_mean(
            "mean_pred_visibility_probability"
        ),
        "per_frame_coordinate_loss": frame_mean("per_frame_coordinate_loss"),
        "per_frame_visibility_loss": frame_mean("per_frame_visibility_loss"),
        "per_frame_total_loss": frame_mean("per_frame_total_loss"),
        "per_frame_ade": frame_mean("per_frame_ade"),
        "weights": torch.cat([row["weights"] for row in rows], dim=-1),
        "valid_future": torch.cat(
            [row["valid_future"] for row in rows], dim=-1
        ),
    }
    return torch.stack(totals).mean(), aggregate, rows


def tracker_input_gradient_audit(
    predictor,
    pred_frames: np.ndarray,
    points_xy: np.ndarray,
    gt_tracks: torch.Tensor,
    gt_visibility_probability: torch.Tensor,
    gt_confidence_probability: torch.Tensor,
    object_count: int,
    points_per_object: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    device = torch.device(args.device)
    pred_video, queries = prepare_tracker_inputs(
        pred_frames,
        points_xy,
        anchor_frame=args.anchor_frame,
        device=device,
    )
    pred_video.requires_grad_(True)
    pred_tracks, pred_visibility_probability, _ = track_video_with_scores(
        predictor, pred_video, queries
    )
    old_loss, _, _ = aggregate_object_trajectory_losses(
        pred_tracks,
        gt_tracks,
        torch.ones_like(gt_visibility_probability, dtype=torch.bool),
        object_count=object_count,
        points_per_object=points_per_object,
        height=TRACK_HEIGHT,
        width=TRACK_WIDTH,
        anchor_frame=args.anchor_frame,
        future_start_frame=args.future_start_frame,
        huber_delta=args.huber_delta,
    )
    new_loss, _, _ = aggregate_visibility_aware_object_losses(
        pred_tracks,
        gt_tracks,
        gt_visibility_probability,
        gt_confidence_probability,
        pred_visibility_probability,
        object_count=object_count,
        points_per_object=points_per_object,
        height=TRACK_HEIGHT,
        width=TRACK_WIDTH,
        anchor_frame=args.anchor_frame,
        future_start_frame=args.future_start_frame,
        huber_delta=args.huber_delta,
        visibility_threshold=args.visibility_threshold,
        visibility_loss_weight=args.visibility_loss_weight,
    )
    old_gradient = torch.autograd.grad(
        old_loss, pred_video, retain_graph=True, create_graph=False
    )[0]
    new_gradient = torch.autograd.grad(
        new_loss, pred_video, retain_graph=False, create_graph=False
    )[0]
    result = {
        "old_all_point": float(old_gradient.detach().float().norm().item()),
        "visibility_aware": float(new_gradient.detach().float().norm().item()),
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in result.values()):
        raise RuntimeError(f"CoTracker loss has invalid input-gradient norms: {result}")
    return result


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
            object_phrases, object_masks_f04 = load_object_specs(
                args,
                case,
                identity_mask,
            )
            points_on2 = np.stack(
                [
                    sample_points_from_mask(
                        object_mask,
                        int(args.num_points),
                        avoid_edges=True,
                    )
                    for object_mask in object_masks_f04
                ]
            ).astype(np.float32)
            object_count = int(len(object_phrases))
            if points_on2.shape != (object_count, int(args.num_points), 2):
                raise RuntimeError(f"invalid object query shape: {points_on2.shape}")
            points_xy = points_on2.reshape(-1, 2)
            print(
                f"[tracks {position}/{len(manifest['cases'])}] "
                f"{object_count} objects x {args.num_points} points",
                flush=True,
            )
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
                (
                    gt_tracks,
                    gt_visibility_probability,
                    gt_confidence_probability,
                ) = track_video_with_scores(predictor, gt_video, queries)
                (
                    pred_tracks,
                    pred_visibility_probability,
                    pred_confidence_probability,
                ) = track_video_with_scores(
                    predictor, pred_video, pred_queries
                )
            gt_visibility = (
                gt_visibility_probability > float(args.visibility_threshold)
            )
            pred_visibility = (
                pred_visibility_probability > float(args.visibility_threshold)
            )
            loss, diagnostics, object_rows = aggregate_object_trajectory_losses(
                pred_tracks,
                gt_tracks,
                gt_visibility,
                object_count=object_count,
                points_per_object=int(args.num_points),
                height=TRACK_HEIGHT,
                width=TRACK_WIDTH,
                anchor_frame=args.anchor_frame,
                future_start_frame=args.future_start_frame,
                huber_delta=args.huber_delta,
            )
            visible_loss = torch.stack(
                [row["loss_gt_visible"] for row in object_rows]
            ).mean()
            visible_diagnostics = diagnostics["visible_aggregate"]
            (
                visibility_aware_total,
                visibility_aware_diagnostics,
                visibility_aware_rows,
            ) = aggregate_visibility_aware_object_losses(
                pred_tracks,
                gt_tracks,
                gt_visibility_probability,
                gt_confidence_probability,
                pred_visibility_probability,
                object_count=object_count,
                points_per_object=int(args.num_points),
                height=TRACK_HEIGHT,
                width=TRACK_WIDTH,
                anchor_frame=args.anchor_frame,
                future_start_frame=args.future_start_frame,
                huber_delta=args.huber_delta,
                visibility_threshold=args.visibility_threshold,
                visibility_loss_weight=args.visibility_loss_weight,
            )
            all_point_visibility = torch.ones_like(gt_visibility, dtype=torch.bool)
            gradient_norms = None
            if args.gradient_audit and not gradient_audited:
                print("[tracks] auditing dL/d(predicted RGB video)", flush=True)
                gradient_norms = tracker_input_gradient_audit(
                    predictor,
                    pred_frames,
                    points_xy,
                    gt_tracks.detach(),
                    gt_visibility_probability.detach(),
                    gt_confidence_probability.detach(),
                    object_count,
                    int(args.num_points),
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
            object_metrics = []
            for object_index, (phrase, row, visibility_row) in enumerate(
                zip(object_phrases, object_rows, visibility_aware_rows)
            ):
                start = object_index * int(args.num_points)
                stop = start + int(args.num_points)
                object_valid = valid[:, :, start:stop]
                object_pred_visible = pred_visible_future[:, :, start:stop]
                object_joint = object_valid & object_pred_visible
                object_metrics.append(
                    {
                        "object_index": object_index,
                        "phrase": phrase,
                        "query_points": int(args.num_points),
                        "trajectory_loss": float(row["loss"].item()),
                        "trajectory_loss_gt_visible": float(
                            row["loss_gt_visible"].item()
                        ),
                        "trajectory_huber": float(row["raw_huber"].item()),
                        "visibility_aware_coordinate_loss": float(
                            visibility_row["coordinate_loss"].item()
                        ),
                        "visibility_preservation_loss": float(
                            visibility_row["visibility_loss"].item()
                        ),
                        "visibility_aware_total_loss": float(
                            visibility_row["total_loss"].item()
                        ),
                        "weighted_visibility_preservation_loss": float(
                            args.visibility_loss_weight
                            * visibility_row["visibility_loss"].item()
                        ),
                        "effective_gt_weight_sum": float(
                            visibility_row["effective_weight_sum"].item()
                        ),
                        "gt_reliable_weight_fraction": float(
                            visibility_row["effective_weight_fraction"].item()
                        ),
                        "mean_gt_visibility_probability": float(
                            visibility_row["mean_gt_visibility_probability"].item()
                        ),
                        "mean_gt_confidence_probability": float(
                            visibility_row["mean_gt_confidence_probability"].item()
                        ),
                        "mean_pred_visibility_probability": float(
                            visibility_row["mean_pred_visibility_probability"].item()
                        ),
                        "normalized_ade": float(row["normalized_ade"].item()),
                        "normalized_ade_gt_visible": float(
                            row["normalized_ade_gt_visible"].item()
                        ),
                        "normalized_rmse": float(row["normalized_rmse"].item()),
                        "normalized_gt_motion": float(
                            row["normalized_gt_motion"].item()
                        ),
                        "gt_visible_fraction": float(
                            row["gt_visible_fraction"].item()
                        ),
                        "pred_visible_fraction": float(
                            object_pred_visible.float().mean().item()
                        ),
                        "joint_visible_fraction": float(
                            object_joint.float().mean().item()
                        ),
                        "all_future_point_frames": int(
                            row["all_future_point_frames"].item()
                        ),
                        "gt_visible_future_point_frames": int(
                            row["gt_visible_future_point_frames"].item()
                        ),
                    }
                )
            forward_metrics = json.loads(
                (case_root / "forward_metrics.json").read_text(encoding="utf-8")
            )
            metrics = {
                **forward_metrics,
                "trajectory_extractor": "frozen CoTracker3 scaled_offline",
                "trajectory_checkpoint": str(args.cotracker_checkpoint.resolve()),
                "object_query_source": (
                    "PyBullet dynamic phrases -> GroundingDINO -> SAM2 F04 identity masks"
                    if args.multiobject_cache is not None
                    else "cached GT F04 SAM2 identity mask"
                ),
                "object_count": object_count,
                "object_query_points_per_object": int(args.num_points),
                "object_query_points": int(points_xy.shape[0]),
                "loss_aggregation": "point/time mean per object -> equal object mean",
                "objects": object_metrics,
                "anchor_frame": int(args.anchor_frame),
                "future_frames": [int(args.future_start_frame), 48],
                "trajectory_definition": "normalized displacement relative to F04",
                "old_loss_definition": "all future point frames, no visibility gating",
                "new_loss_definition": (
                    "GT visibility > threshold x GT confidence coordinate weighting "
                    "+ weighted prediction visibility preservation"
                ),
                "new_coordinate_validity": (
                    "GT raw visibility/confidence only; predicted visibility never "
                    "masks coordinate loss"
                ),
                "loss_primary_validity": (
                    "GT visibility > threshold x GT confidence; predicted visibility "
                    "is a separate preservation penalty"
                ),
                "loss_audit_validity": "GT CoTracker visibility > 0.9",
                "trajectory_loss_type": "smooth_l1",
                "trajectory_loss_beta": float(args.huber_delta),
                "visibility_threshold": float(args.visibility_threshold),
                "visibility_loss_weight": float(args.visibility_loss_weight),
                "huber_delta": float(args.huber_delta),
                "old_all_point_trajectory_loss": float(loss.item()),
                "trajectory_loss": float(loss.item()),
                "trajectory_loss_gt_visible": float(visible_loss.item()),
                "trajectory_smooth_l1": float(loss.item()),
                "trajectory_smooth_l1_gt_visible": float(visible_loss.item()),
                "trajectory_huber": float(diagnostics["raw_huber"].item()),
                "trajectory_huber_gt_visible": float(
                    visible_diagnostics["raw_huber"].item()
                ),
                "visibility_aware_coordinate_loss": float(
                    visibility_aware_diagnostics["coordinate_loss"].item()
                ),
                "visibility_preservation_loss": float(
                    visibility_aware_diagnostics["visibility_loss"].item()
                ),
                "weighted_visibility_preservation_loss": float(
                    args.visibility_loss_weight
                    * visibility_aware_diagnostics["visibility_loss"].item()
                ),
                "visibility_aware_total_loss": float(
                    visibility_aware_total.item()
                ),
                "visibility_aware_raw_huber": float(
                    visibility_aware_diagnostics["raw_huber"].item()
                ),
                "effective_gt_weight_sum": float(
                    visibility_aware_diagnostics["effective_weight_sum"].item()
                ),
                "gt_reliable_weight_fraction": float(
                    visibility_aware_diagnostics[
                        "effective_weight_fraction"
                    ].item()
                ),
                "mean_gt_visibility_probability": float(
                    visibility_aware_diagnostics[
                        "mean_gt_visibility_probability"
                    ].item()
                ),
                "mean_gt_confidence_probability": float(
                    visibility_aware_diagnostics[
                        "mean_gt_confidence_probability"
                    ].item()
                ),
                "mean_pred_visibility_probability": float(
                    visibility_aware_diagnostics[
                        "mean_pred_visibility_probability"
                    ].item()
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
                "tracker_input_gradient_norm": (
                    None
                    if gradient_norms is None
                    else gradient_norms["visibility_aware"]
                ),
                "tracker_input_gradient_norm_old_all_point": (
                    None
                    if gradient_norms is None
                    else gradient_norms["old_all_point"]
                ),
                "tracker_input_gradient_norm_visibility_aware": (
                    None
                    if gradient_norms is None
                    else gradient_norms["visibility_aware"]
                ),
            }
            np.savez_compressed(
                case_root / "trajectories.npz",
                query_points_xy_native=points_xy.astype(np.float32),
                query_points_on2=points_on2.astype(np.float32),
                object_ids_per_point=np.repeat(
                    np.arange(object_count, dtype=np.int32), int(args.num_points)
                ),
                object_masks_f04=object_masks_f04.astype(np.uint8),
                object_phrases=np.asarray(object_phrases),
                identity_mask=identity_mask.astype(np.uint8),
                gt_tracks_trackres=gt_tracks[0].float().cpu().numpy(),
                pred_tracks_trackres=pred_tracks[0].float().cpu().numpy(),
                gt_visibility=gt_visibility[0].cpu().numpy().astype(np.uint8),
                pred_visibility=pred_visibility[0].cpu().numpy().astype(np.uint8),
                gt_visibility_probability=gt_visibility_probability[0]
                .float()
                .cpu()
                .numpy(),
                gt_confidence_probability=gt_confidence_probability[0]
                .float()
                .cpu()
                .numpy(),
                pred_visibility_probability=pred_visibility_probability[0]
                .float()
                .cpu()
                .numpy(),
                pred_confidence_probability=pred_confidence_probability[0]
                .float()
                .cpu()
                .numpy(),
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
                per_frame_visibility_aware_coordinate_loss=
                visibility_aware_diagnostics["per_frame_coordinate_loss"][0]
                .cpu()
                .numpy(),
                per_frame_visibility_preservation_loss=
                visibility_aware_diagnostics["per_frame_visibility_loss"][0]
                .cpu()
                .numpy(),
                per_frame_visibility_aware_total_loss=
                visibility_aware_diagnostics["per_frame_total_loss"][0]
                .cpu()
                .numpy(),
                per_frame_visibility_aware_ade=
                visibility_aware_diagnostics["per_frame_ade"][0]
                .cpu()
                .numpy(),
                visibility_aware_weights=visibility_aware_diagnostics["weights"][0]
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
                gt_visibility_probability,
                gt_confidence_probability,
                pred_visibility_probability,
                pred_confidence_probability,
                loss,
                diagnostics,
                visible_loss,
                visible_diagnostics,
                visibility_aware_total,
                visibility_aware_diagnostics,
                visibility_aware_rows,
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


def overlay_object_masks(
    frame: np.ndarray,
    masks: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    output = frame.copy()
    for object_index, mask in enumerate(masks.astype(bool)):
        color = colors[object_index % len(colors)].astype(np.float32)
        output[mask] = np.clip(
            0.60 * output[mask].astype(np.float32) + 0.40 * color,
            0,
            255,
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            output,
            contours,
            -1,
            tuple(int(value) for value in color),
            2,
            cv2.LINE_AA,
        )
    return output


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
        gt_visibility_probability = archive[
            "gt_visibility_probability"
        ].astype(np.float32)
        gt_confidence_probability = archive[
            "gt_confidence_probability"
        ].astype(np.float32)
        pred_visibility_probability = archive[
            "pred_visibility_probability"
        ].astype(np.float32)
        object_masks_f04 = archive["object_masks_f04"].astype(bool)
        object_ids_per_point = archive["object_ids_per_point"].astype(np.int32)
        per_frame_loss = archive["per_frame_trajectory_loss"].astype(np.float32)
        per_frame_new_coordinate = archive[
            "per_frame_visibility_aware_coordinate_loss"
        ].astype(np.float32)
        per_frame_visibility = archive[
            "per_frame_visibility_preservation_loss"
        ].astype(np.float32)
        per_frame_new_total = archive[
            "per_frame_visibility_aware_total_loss"
        ].astype(np.float32)
        identity_mask = archive["identity_mask"].astype(bool)
    height, width = source_frames.shape[1:3]
    gt_native = trackres_to_native(gt_tracks, height, width)
    pred_native = trackres_to_native(pred_tracks, height, width)
    point_colors = OBJECT_COLORS[object_ids_per_point % len(OBJECT_COLORS)]
    object_count = int(object_masks_f04.shape[0])
    frames = []
    for frame_id in range(source_frames.shape[0]):
        gt_panel = source_frames[frame_id].copy()
        pred_panel = pred_frames[frame_id].copy()
        compare_panel = pred_frames[frame_id].copy()
        if frame_id == int(args.anchor_frame):
            gt_panel = overlay_object_masks(gt_panel, object_masks_f04, OBJECT_COLORS)
        draw_track_history(
            gt_panel,
            gt_native,
            gt_visibility,
            frame_id,
            anchor_frame=args.anchor_frame,
            colors=point_colors,
        )
        draw_track_history(
            pred_panel,
            pred_native,
            pred_visibility,
            frame_id,
            anchor_frame=args.anchor_frame,
            colors=point_colors,
        )
        draw_track_history(
            compare_panel,
            gt_native,
            gt_visibility,
            frame_id,
            anchor_frame=args.anchor_frame,
            colors=point_colors,
            ring=True,
        )
        draw_track_history(
            compare_panel,
            pred_native,
            pred_visibility,
            frame_id,
            anchor_frame=args.anchor_frame,
            colors=point_colors,
        )
        if frame_id >= int(args.future_start_frame):
            for point_id in range(gt_native.shape[1]):
                if not bool(gt_visibility[frame_id, point_id]):
                    continue
                start = tuple(np.rint(gt_native[frame_id, point_id]).astype(int))
                stop = tuple(np.rint(pred_native[frame_id, point_id]).astype(int))
                cv2.line(compare_panel, start, stop, (255, 72, 72), 1, cv2.LINE_AA)
        frame_metrics = (
            f"F{frame_id:02d} | context | "
            f"clip old={metrics['old_all_point_trajectory_loss']:.5f} | "
            f"new={metrics['visibility_aware_total_loss']:.5f}"
        )
        if frame_id >= int(args.future_start_frame):
            future_id = frame_id - int(args.future_start_frame)
            old_value = per_frame_loss[future_id]
            coordinate_value = per_frame_new_coordinate[future_id]
            weighted_visibility_value = (
                float(args.visibility_loss_weight) * per_frame_visibility[future_id]
            )
            new_total_value = per_frame_new_total[future_id]
            frame_metrics = (
                f"F{frame_id:02d} | old={old_value:.5f} | "
                f"coord={coordinate_value:.5f} | "
                f"{args.visibility_loss_weight:g}*vis="
                f"{weighted_visibility_value:.5f} | new={new_total_value:.5f}"
            )
        gt_reliable = int(
            (gt_visibility_probability[frame_id] > args.visibility_threshold).sum()
        )
        add_panel_label(
            gt_panel,
            "GT RGB + reliability-gated CoTracker",
            f"F{frame_id:02d} | reliable {gt_reliable}/{gt_tracks.shape[1]} | "
            f"mean vis={gt_visibility_probability[frame_id].mean():.2f} | "
            f"conf={gt_confidence_probability[frame_id].mean():.2f}",
        )
        add_panel_label(
            pred_panel,
            "Tiny-VAE x0_pred + Pred CoTracker",
            f"F{frame_id:02d} | visible {int(pred_visibility[frame_id].sum())}/"
            f"{pred_tracks.shape[1]} | mean vis="
            f"{pred_visibility_probability[frame_id].mean():.2f}",
        )
        add_panel_label(
            compare_panel,
            "Old all-point vs new visibility-aware loss",
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
            "old_all_point_trajectory_loss": metrics[
                "old_all_point_trajectory_loss"
            ],
            "visibility_aware_total_loss": metrics[
                "visibility_aware_total_loss"
            ],
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
        old_gradient = metrics.get("tracker_input_gradient_norm_old_all_point")
        new_gradient = metrics.get("tracker_input_gradient_norm_visibility_aware")
        old_gradient_text = (
            "not run" if old_gradient is None else f"{float(old_gradient):.3e}"
        )
        new_gradient_text = (
            "not run" if new_gradient is None else f"{float(new_gradient):.3e}"
        )
        object_cards = "".join(
            f"<article><h3>O{item['object_index'] + 1} · "
            f"{html.escape(item['phrase'])}</h3><div class=object-metrics>"
            f"<span><b>{item['trajectory_loss']:.6f}</b>old all-point</span>"
            f"<span><b>{item['visibility_aware_coordinate_loss']:.6f}</b>new coord</span>"
            f"<span><b>{item['weighted_visibility_preservation_loss']:.6f}</b>"
            f"{args.visibility_loss_weight:g} × visibility</span>"
            f"<span><b>{item['visibility_aware_total_loss']:.6f}</b>new total</span>"
            f"<span><b>{item['gt_visible_fraction']:.1%}</b>GT visible</span>"
            f"<span><b>{item['gt_reliable_weight_fraction']:.1%}</b>GT weighted coverage</span>"
            f"<span><b>{item['mean_pred_visibility_probability']:.1%}</b>pred visibility</span>"
            f"</div></article>"
            for item in metrics.get("objects", [])
        )
        rows.append(
            f"""
<section class="case">
  <div class="case-head"><div><span class="case-index">CASE {position:02d}</span>
  <h2>{html.escape(case['case_key'])} · {metrics.get('object_count', 1)} objects</h2><p>{html.escape(case['caption'])}</p></div>
  <div class="loss-pair"><div class=old><small>OLD · ALL POINTS</small><strong>{metrics['old_all_point_trajectory_loss']:.6f}</strong><small>grad {old_gradient_text}</small></div>
  <div class=new><small>NEW · VISIBILITY AWARE</small><strong>{metrics['visibility_aware_total_loss']:.6f}</strong><small>grad {new_gradient_text}</small></div></div></div>
  <div class="metrics comparison-metrics">
    <span><b>{metrics['old_all_point_trajectory_loss']:.6f}</b>old all-point SmoothL1</span>
    <span><b>{metrics['visibility_aware_coordinate_loss']:.6f}</b>GT-weighted coordinate</span>
    <span><b>{metrics['weighted_visibility_preservation_loss']:.6f}</b>{args.visibility_loss_weight:g} × pred visibility</span>
    <span><b>{metrics['visibility_aware_total_loss']:.6f}</b>new total</span>
    <span><b>{metrics['gt_visible_fraction']:.1%}</b>GT visible</span>
    <span><b>{metrics['gt_reliable_weight_fraction']:.1%}</b>GT weighted coverage</span>
    <span><b>{metrics['mean_pred_visibility_probability']:.1%}</b>pred visibility probability</span>
    <span><b>{metrics['normalized_ade_gt_visible']:.4f}</b>GT-visible ADE</span>
  </div>
  <div class="object-grid">{object_cards}</div>
  <video controls muted loop playsinline preload="metadata" poster="{relative}/trajectory_future_preview.jpg" src="{relative}/object_trajectory_overlay.mp4"></video>
  <div class="media-row">
    <figure><img src="{relative}/object_query_preview.jpg"><figcaption>F04 object queries</figcaption></figure>
    <figure><video controls muted loop playsinline preload="metadata" src="{relative}/gt_source.mp4"></video><figcaption>Original PyBullet GT</figcaption></figure>
    <figure><video controls muted loop playsinline preload="metadata" src="{relative}/pred_x0.mp4"></video><figcaption>Tiny-VAE decoded x0_pred</figcaption></figure>
  </div>
  <div class="links"><a href="{relative}/metrics.json">metrics.json</a><a href="{relative}/trajectories.npz">trajectories.npz</a></div>
</section>"""
        )
    mean_loss = float(
        np.mean([item["old_all_point_trajectory_loss"] for item in summary_metrics])
    )
    mean_new_coordinate = float(
        np.mean(
            [item["visibility_aware_coordinate_loss"] for item in summary_metrics]
        )
    )
    mean_visibility_penalty = float(
        np.mean(
            [
                item["weighted_visibility_preservation_loss"]
                for item in summary_metrics
            ]
        )
    )
    mean_new_total = float(
        np.mean([item["visibility_aware_total_loss"] for item in summary_metrics])
    )
    mean_visible_loss = float(
        np.mean([item["trajectory_loss_gt_visible"] for item in summary_metrics])
    )
    mean_raw_huber = float(
        np.mean([item["trajectory_huber"] for item in summary_metrics])
    )
    mean_ade = float(np.mean([item["normalized_ade"] for item in summary_metrics]))
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visibility-aware trajectory loss comparison</title><style>
:root{{--bg:#eef2f3;--ink:#142026;--muted:#607078;--line:#c7d0d3;--paper:#fff;--old:#b83d35;--green:#16745a;--blue:#26648b;--gold:#ad7411}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 "IBM Plex Sans","Noto Sans SC",sans-serif;letter-spacing:0}}
header{{background:#172329;color:#f6fafb;border-bottom:5px solid var(--green)}}.mast{{max-width:1500px;margin:auto;padding:28px 28px 24px;display:grid;grid-template-columns:1fr auto;gap:30px;align-items:end}}
h1{{font:700 34px/1.1 "IBM Plex Sans Condensed","Noto Sans SC",sans-serif;margin:0 0 8px;letter-spacing:0}}header p{{margin:0;color:#b9c7cb}}.summary{{display:flex;gap:24px}}.summary small,.loss-pair small{{display:block;color:#91a3a8;font-size:11px}}.summary strong{{font-size:24px}}
main{{max-width:1500px;margin:auto;padding:22px 28px 80px}}.case{{padding:25px 0 34px;border-bottom:1px solid var(--line)}}.case-head{{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:14px}}.case-index{{font-size:11px;color:var(--green);font-weight:800}}h2{{margin:2px 0 2px;font-size:23px}}.case-head p{{margin:0;color:var(--muted)}}.loss-pair{{display:flex;text-align:right;border-bottom:1px solid var(--line)}}.loss-pair>div{{min-width:190px;padding:4px 12px 7px;border-left:4px solid}}.loss-pair .old{{border-color:var(--old)}}.loss-pair .new{{border-color:var(--green)}}.loss-pair strong{{display:block;font-size:25px}}.loss-pair .old strong{{color:#f07970}}.loss-pair .new strong{{color:#56d0aa}}
.metrics{{display:grid;border:1px solid var(--line);background:var(--paper);margin-bottom:14px}}.comparison-metrics{{grid-template-columns:repeat(8,minmax(0,1fr))}}.metrics span{{padding:10px 12px;border-right:1px solid var(--line);color:var(--muted)}}.metrics span:last-child{{border:0}}.metrics b{{display:block;color:var(--ink);font-size:17px}}
.object-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:8px;margin:0 0 14px}}.object-grid article{{background:var(--paper);border-top:4px solid var(--green);padding:10px 12px;color:var(--ink)}}.object-grid article:nth-child(2n){{border-color:var(--blue)}}.object-grid article:nth-child(3n){{border-color:var(--gold)}}.object-grid h3{{font-size:15px;margin:0 0 8px}}.object-metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px 10px}}.object-metrics span{{color:var(--muted);font-size:11px}}.object-metrics b{{display:block;color:var(--ink);font-size:14px}}
video,img{{display:block;width:100%;background:#050708}}.case>video{{border:1px solid #29383f;aspect-ratio:21/4;object-fit:contain}}.media-row{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:14px}}figure{{margin:0;background:var(--paper);border:1px solid var(--line)}}figure video,figure img{{aspect-ratio:16/9;object-fit:contain}}figcaption{{padding:8px 10px;color:var(--muted)}}.links{{display:flex;gap:16px;margin-top:10px}}a{{color:var(--blue);font-weight:700}}
#replay{{position:fixed;right:22px;bottom:20px;border:1px solid #0e4d3d;background:var(--green);color:white;padding:11px 17px;font-weight:750;cursor:pointer}}
@media(max-width:1100px){{.comparison-metrics{{grid-template-columns:repeat(4,1fr)}}}}@media(max-width:900px){{.mast{{grid-template-columns:1fr}}.summary{{display:grid;grid-template-columns:repeat(2,1fr)}}.metrics{{grid-template-columns:repeat(2,1fr)}}.metrics span{{border-bottom:1px solid var(--line)}}.media-row{{grid-template-columns:1fr}}.case-head{{display:block}}.loss-pair{{text-align:left;margin-top:12px}}.loss-pair>div{{min-width:0;flex:1}}.case>video{{aspect-ratio:16/9}}.object-metrics{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><header><div class="mast"><div><h1>Visibility-aware Object Trajectory Loss</h1><p>PyBullet 100% · high-noise x_t t={args.training_timestep:g} · F04 object queries n={args.num_points}/object · F08–F48 displacement · GT visibility &gt; {args.visibility_threshold:g} × confidence · Lnew = Lcoord + {args.visibility_loss_weight:g} Lvis · equal-object mean</p></div><div class="summary"><div><small>MEAN OLD</small><strong>{mean_loss:.6f}</strong></div><div><small>MEAN NEW COORD</small><strong>{mean_new_coordinate:.6f}</strong></div><div><small>MEAN {args.visibility_loss_weight:g} × VIS</small><strong>{mean_visibility_penalty:.6f}</strong></div><div><small>MEAN NEW TOTAL</small><strong>{mean_new_total:.6f}</strong></div></div></div></header><main>{''.join(rows)}</main><button id="replay">Replay all</button><script>
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
            "mean_old_all_point_loss": mean_loss,
            "mean_visibility_aware_coordinate_loss": mean_new_coordinate,
            "mean_weighted_visibility_preservation_loss": mean_visibility_penalty,
            "mean_visibility_preservation_loss": float(
                np.mean(
                    [item["visibility_preservation_loss"] for item in summary_metrics]
                )
            ),
            "mean_visibility_aware_total_loss": mean_new_total,
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
