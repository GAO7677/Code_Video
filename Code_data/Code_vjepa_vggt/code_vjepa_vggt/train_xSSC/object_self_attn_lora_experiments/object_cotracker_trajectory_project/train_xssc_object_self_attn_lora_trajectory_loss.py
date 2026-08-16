#!/usr/bin/env python3
"""Train Wan Full-SA LoRA with frozen CoTracker object-trajectory loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
TRAIN_XSSC_ROOT = EXPERIMENT_ROOT.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[2]
COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
for _path in (HERE, EXPERIMENT_ROOT, TRAIN_XSSC_ROOT, REPOSITORY_ROOT, COTRACKER_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import code_vjepa_vggt.context_wan_v_newtrain as context_wan
import train_xssc_context_slots as official_xssc
import train_xssc_object_self_attn_lora as core
from object_trajectory_loss import object_equal_visibility_aware_trajectory_loss
from trajectory_cache import PyBulletTrajectoryCache, TrajectoryCachedDataset
from vjepa_loss_project.train_xssc_object_self_attn_lora_vjepa_loss import (
    _load_tiny_vae,
)


TRACK_HEIGHT = 256
TRACK_WIDTH = 448


def replace_query_predictions(
    tracks: torch.Tensor,
    visibility: torch.Tensor,
    confidence: torch.Tensor,
    scaled_queries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchor_mask = torch.zeros_like(visibility, dtype=torch.bool)
    anchor_tracks = torch.zeros_like(tracks)
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


def differentiable_track_video_with_scores(
    predictor,
    video: torch.Tensor,
    queries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run sparse CoTracker with checkpointed activations and score outputs."""
    from cotracker.models.core.model_utils import get_points_on_a_grid

    batch, frames, channels, height, width = video.shape
    resized = F.interpolate(
        video.reshape(batch * frames, channels, height, width),
        tuple(predictor.interp_shape),
        mode="bilinear",
        align_corners=True,
    ).reshape(batch, frames, channels, *predictor.interp_shape)
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
    model_queries = torch.cat(
        (scaled_queries, support.repeat(batch, 1, 1)), dim=1
    )

    def run_tracker(current_video: torch.Tensor, current_queries: torch.Tensor):
        tracks, visibility, confidence, _ = predictor.model.forward(
            video=current_video,
            queries=current_queries,
            iters=6,
        )
        return tracks, visibility, confidence

    device_type = resized.device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=device_type == "cuda",
    ):
        if torch.is_grad_enabled() and resized.requires_grad:
            tracks, visibility, confidence = checkpoint(
                run_tracker,
                resized,
                model_queries,
                use_reentrant=False,
            )
        else:
            tracks, visibility, confidence = run_tracker(resized, model_queries)
    query_count = int(queries.shape[1])
    tracks, visibility, confidence = replace_query_predictions(
        tracks[:, :, :query_count],
        visibility[:, :, :query_count],
        confidence[:, :, :query_count],
        scaled_queries,
    )
    tracks = tracks * tracks.new_tensor(
        (
            (width - 1) / (predictor.interp_shape[1] - 1),
            (height - 1) / (predictor.interp_shape[0] - 1),
        )
    )
    return tracks, visibility, confidence


class TrajectoryLossWanModule(core.DINOv3XSSCContextSlotsWanModule):
    """No-object self-attention module plus frozen CoTracker supervision."""

    def __init__(
        self,
        *args,
        trajectory_loss_weight: float,
        trajectory_anchor_frame: int,
        trajectory_future_start_frame: int,
        trajectory_points_per_object: int,
        trajectory_huber_delta: float,
        trajectory_visibility_threshold: float,
        trajectory_visibility_loss_weight: float,
        trajectory_gradient_diagnostics_every_n_forwards: int,
        cotracker_checkpoint: str,
        tiny_vae_root: str,
        tiny_vae_checkpoint: str,
        tiny_vae_parallel: bool,
        **kwargs,
    ) -> None:
        kwargs["load_tokenizer"] = False
        super().__init__(*args, **kwargs)
        if self.enable_object_branch:
            raise ValueError("trajectory-loss training requires no-object mode")
        if self.self_attn_adaptation_mode != "full_sa":
            raise ValueError("trajectory-loss training currently requires full_sa")
        if any(
            getattr(self.pipe, name, None) is not None
            for name in ("vae", "text_encoder", "tokenizer")
        ):
            raise RuntimeError(
                "precomputed-only training must not construct Wan VAE, UMT5, or tokenizer"
            )

        self.trajectory_loss_weight = float(trajectory_loss_weight)
        self.trajectory_anchor_frame = int(trajectory_anchor_frame)
        self.trajectory_future_start_frame = int(trajectory_future_start_frame)
        self.trajectory_points_per_object = int(trajectory_points_per_object)
        self.trajectory_huber_delta = float(trajectory_huber_delta)
        self.trajectory_visibility_threshold = float(trajectory_visibility_threshold)
        self.trajectory_visibility_loss_weight = float(
            trajectory_visibility_loss_weight
        )
        self.trajectory_gradient_diagnostics_every_n_forwards = int(
            trajectory_gradient_diagnostics_every_n_forwards
        )
        self.tiny_vae_parallel = bool(tiny_vae_parallel)
        self._trajectory_forward_count = 0
        self._trajectory_batch: list[dict] | None = None
        if self.trajectory_loss_weight <= 0.0:
            raise ValueError("trajectory_loss_weight must be positive")
        if self.trajectory_anchor_frame != 4:
            raise ValueError("trajectory anchor must be F04")
        if self.trajectory_future_start_frame != 8:
            raise ValueError("trajectory future supervision must start at F08")
        if self.trajectory_points_per_object <= 0:
            raise ValueError("trajectory_points_per_object must be positive")
        if self.trajectory_huber_delta <= 0.0:
            raise ValueError("trajectory_huber_delta must be positive")
        if self.trajectory_gradient_diagnostics_every_n_forwards <= 0:
            raise ValueError(
                "trajectory_gradient_diagnostics_every_n_forwards must be positive"
            )

        model_device = self.pipe.dit.patch_embedding.weight.device
        tiny_vae, tiny_vae_apply = _load_tiny_vae(
            tiny_vae_root,
            tiny_vae_checkpoint,
            model_device,
            self.pipe.torch_dtype,
        )
        tiny_vae.eval().requires_grad_(False)
        object.__setattr__(self, "_tiny_vae", tiny_vae)
        object.__setattr__(self, "_tiny_vae_apply", tiny_vae_apply)

        from cotracker.predictor import CoTrackerPredictor

        tracker = (
            CoTrackerPredictor(
                checkpoint=str(Path(cotracker_checkpoint).expanduser().resolve()),
                offline=True,
                v2=False,
                window_len=60,
            )
            .to(model_device)
            .eval()
            .requires_grad_(False)
        )
        object.__setattr__(self, "_trajectory_tracker", tracker)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "_tiny_vae"):
            self._tiny_vae.eval()
        if hasattr(self, "_trajectory_tracker"):
            self._trajectory_tracker.eval()
        return self

    def get_pipeline_inputs(self, data):
        inputs = super().get_pipeline_inputs(data)
        raw_sample = inputs[0].get("raw_sample")
        if isinstance(raw_sample, dict):
            raw_sample.pop("trajectory_cache", None)
        return inputs

    def _prepare_pipeline_sample(self, sample):
        inputs = self.get_pipeline_inputs(sample)
        inputs = self.transfer_data_to_device(
            inputs, self.pipe.device, self.pipe.torch_dtype
        )
        for unit in self.pipe.units:
            unit_name = unit.__class__.__name__
            if unit_name == "WanVideoUnit_PromptEmbedder" and "context" in inputs[1]:
                continue
            if unit_name == "WanVideoUnit_ContextVideoEmbedder":
                if not isinstance(inputs[0].get("input_latents"), torch.Tensor):
                    raise RuntimeError(
                        "DiT-only context conditioning requires cached input_latents"
                    )
                # flow_match_context_sft_loss restores the selected context latent
                # indices directly from input_latents, so no VAE prefix encode is needed.
                continue
            if (
                "vae" in tuple(getattr(unit, "onload_model_names", None) or ())
                and unit_name != "WanVideoUnit_InputVideoEmbedder"
            ):
                # Cached input_latents already contain every training frame. The
                # context-aware flow loss restores clean condition indices from
                # that tensor, so online VAE condition encoders are redundant.
                continue
            if unit_name == "WanVideoUnit_NoiseInitializer":
                input_latents = inputs[0].get("input_latents")
                if not isinstance(input_latents, torch.Tensor):
                    raise RuntimeError(
                        "DiT-only trajectory training requires precomputed input_latents"
                    )
                inputs[0]["noise"] = self.pipe.generate_noise(
                    tuple(input_latents.shape),
                    seed=inputs[0].get("seed"),
                    rand_device=inputs[0].get("rand_device", self.pipe.device),
                ).to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
                continue
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return inputs

    def forward(self, data, inputs=None):
        if inputs is not None:
            raise ValueError("trajectory-loss training requires raw cached samples")
        samples = data if isinstance(data, list) else [data]
        self._trajectory_batch = [sample["trajectory_cache"] for sample in samples]
        try:
            return self._forward_sample_batch(samples)
        finally:
            self._trajectory_batch = None

    @staticmethod
    def _restore_condition_latents(pred_x0, target_x0, captured_inputs):
        context_indices = context_wan.resolve_context_latent_indices_from_frames(
            raw_frame_indices=captured_inputs.get("context_frame_indices"),
            raw_num_frames=captured_inputs.get("num_frames"),
            latent_length=int(target_x0.shape[2]),
        )
        if context_indices:
            return context_wan.apply_clean_latents_at_indices(
                pred_x0, target_x0, context_indices
            )
        prefix_length = context_wan.resolve_num_clean_prefix_latents(
            clean_prefix_latents=captured_inputs.get("clean_prefix_latents"),
            num_clean_prefix_latents=captured_inputs.get("num_clean_prefix_latents"),
        )
        if prefix_length > 0:
            pred_x0 = pred_x0.clone()
            pred_x0[:, :, :prefix_length] = target_x0[:, :, :prefix_length]
        elif "first_frame_latents" in captured_inputs:
            pred_x0 = pred_x0.clone()
            pred_x0[:, :, 0:1] = target_x0[:, :, 0:1]
        return pred_x0

    def _decode_tiny_vae_raw(self, latents: torch.Tensor) -> torch.Tensor:
        latent_ntchw = latents.permute(0, 2, 1, 3, 4).contiguous()
        device_type = latent_ntchw.device.type
        with torch.autocast(
            device_type=device_type,
            dtype=self.pipe.torch_dtype,
            enabled=device_type == "cuda",
        ):
            video = self._tiny_vae_apply(
                self._tiny_vae.decoder,
                latent_ntchw,
                self.tiny_vae_parallel,
                False,
            )
            if self._tiny_vae.patch_size > 1:
                video = F.pixel_shuffle(video, self._tiny_vae.patch_size)
        if not (self._tiny_vae.is_cogvideox and latent_ntchw.shape[1] % 2 == 0):
            video = video[:, self._tiny_vae.frames_to_trim :]
        return video

    def _trajectory_loss_for_sample(
        self,
        pred_video: torch.Tensor,
        cache: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        object_count = int(cache["object_count"])
        points_per_object = int(cache["points_per_object"])
        if points_per_object != self.trajectory_points_per_object:
            raise ValueError(
                f"trajectory point-count mismatch: cache={points_per_object}, "
                f"training={self.trajectory_points_per_object}"
            )
        device = pred_video.device
        query_points = cache["query_points"].to(device=device, dtype=torch.float32)
        query_points = query_points.reshape(-1, 2)
        frame_ids = torch.full(
            (query_points.shape[0], 1),
            float(self.trajectory_anchor_frame),
            device=device,
            dtype=query_points.dtype,
        )
        queries = torch.cat((frame_ids, query_points), dim=-1).unsqueeze(0)
        tracker_video = F.interpolate(
            pred_video.float() * 255.0,
            size=(TRACK_HEIGHT, TRACK_WIDTH),
            mode="bilinear",
            align_corners=True,
        ).unsqueeze(0)
        pred_tracks, pred_visibility, _ = differentiable_track_video_with_scores(
            self._trajectory_tracker,
            tracker_video,
            queries,
        )

        time_steps = int(pred_tracks.shape[1])
        point_count = object_count * points_per_object
        gt_tracks = cache["gt_tracks"].to(device=device, dtype=torch.float32).reshape(
            1, time_steps, point_count, 2
        )
        gt_visibility = cache["gt_visibility_probability"].to(
            device=device, dtype=torch.float32
        ).reshape(1, time_steps, point_count)
        gt_confidence = cache["gt_confidence_probability"].to(
            device=device, dtype=torch.float32
        ).reshape(1, time_steps, point_count)
        gt_geometric = cache["gt_geometric_visibility"].to(device=device).bool().reshape(
            1, time_steps, point_count
        )
        return object_equal_visibility_aware_trajectory_loss(
            pred_tracks.float(),
            gt_tracks,
            gt_visibility,
            gt_confidence,
            pred_visibility.float(),
            gt_geometric,
            object_count=object_count,
            points_per_object=points_per_object,
            height=TRACK_HEIGHT,
            width=TRACK_WIDTH,
            anchor_frame=self.trajectory_anchor_frame,
            future_start_frame=self.trajectory_future_start_frame,
            huber_delta=self.trajectory_huber_delta,
            visibility_threshold=self.trajectory_visibility_threshold,
            visibility_loss_weight=self.trajectory_visibility_loss_weight,
        )

    @staticmethod
    def _normalized_timestep_weight(pipe, timestep):
        raw_weight = pipe.scheduler.training_weight(timestep).detach().float()
        normalizer = pipe.scheduler.linear_timesteps_weights.detach().float().mean()
        normalized = raw_weight / normalizer.clamp_min(1e-12)
        return (
            normalized.to(device=pipe.device, dtype=torch.float32),
            float(raw_weight.item()),
            float(normalizer.item()),
        )

    @staticmethod
    def _output_gradient_diagnostics(main_loss, weighted_aux_loss, model_output):
        main_grad = torch.autograd.grad(
            main_loss, model_output, retain_graph=True, create_graph=False
        )[0].detach().float()
        aux_grad = torch.autograd.grad(
            weighted_aux_loss, model_output, retain_graph=True, create_graph=False
        )[0].detach().float()
        main_norm = torch.linalg.vector_norm(main_grad)
        aux_norm = torch.linalg.vector_norm(aux_grad)
        cosine = (main_grad * aux_grad).sum() / (
            main_norm * aux_norm
        ).clamp_min(1e-20)
        return {
            "train/trajectory_grad_diag_applied": 1.0,
            "train/grad_v_main_norm": float(main_norm.item()),
            "train/grad_v_trajectory_norm": float(aux_norm.item()),
            "train/grad_v_trajectory_to_main_ratio": float(
                (aux_norm / main_norm.clamp_min(1e-20)).item()
            ),
            "train/grad_v_main_trajectory_cosine": float(cosine.item()),
        }

    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        if self._trajectory_batch is None:
            raise RuntimeError("trajectory batch was not attached before forward")
        captured = []
        original_model_fn = pipe.model_fn

        def capture_model_fn(*args, **kwargs):
            output = original_model_fn(*args, **kwargs)
            captured.append(
                {
                    "model_output": output,
                    "latents": kwargs.get("latents"),
                    "timestep": kwargs.get("timestep"),
                    "inputs": kwargs,
                }
            )
            return output

        pipe.model_fn = capture_model_fn
        try:
            total, metrics = super()._compute_object_losses(
                pipe, inputs_shared, inputs_posi
            )
        finally:
            pipe.model_fn = original_model_fn
        if len(captured) != 1:
            raise RuntimeError(
                f"expected exactly one DiT forward, captured {len(captured)}"
            )
        record = captured[0]
        sigma = context_wan._diffsynth_sigma_for_timestep(
            pipe.scheduler, record["timestep"]
        ).to(device=record["latents"].device, dtype=record["latents"].dtype)
        while sigma.ndim < record["latents"].ndim:
            sigma = sigma.unsqueeze(-1)
        pred_x0 = record["latents"] - sigma * record["model_output"]
        target_x0 = inputs_shared["input_latents"]
        pred_x0 = self._restore_condition_latents(
            pred_x0, target_x0, record["inputs"]
        )
        pred_raw = self._decode_tiny_vae_raw(pred_x0)
        pred_clipped = pred_raw.clamp(0.0, 1.0)
        pred_video = pred_raw + (pred_clipped - pred_raw).detach()
        if int(pred_video.shape[0]) != len(self._trajectory_batch):
            raise RuntimeError(
                f"trajectory batch mismatch: video={pred_video.shape[0]}, "
                f"cache={len(self._trajectory_batch)}"
            )

        sample_losses = []
        sample_diagnostics = []
        for index, cache in enumerate(self._trajectory_batch):
            sample_loss, diagnostics = self._trajectory_loss_for_sample(
                pred_video[index], cache
            )
            sample_losses.append(sample_loss)
            sample_diagnostics.append(diagnostics)
        trajectory_loss = torch.stack(sample_losses).mean()

        def batch_metric(name: str) -> float:
            return float(
                torch.stack([item[name].detach().float() for item in sample_diagnostics])
                .mean()
                .item()
            )

        metrics.update(
            {
                "train/loss_trajectory": float(trajectory_loss.detach().item()),
                "train/trajectory_coordinate_loss": batch_metric("coordinate_loss"),
                "train/trajectory_visibility_loss": batch_metric("visibility_loss"),
                "train/trajectory_normalized_ade": batch_metric("normalized_ade"),
                "train/trajectory_normalized_rmse": batch_metric("normalized_rmse"),
                "train/trajectory_gt_motion": batch_metric("normalized_gt_motion"),
                "train/trajectory_valid_fraction": batch_metric("valid_fraction"),
                "train/trajectory_effective_weight_fraction": batch_metric(
                    "effective_weight_fraction"
                ),
                "train/trajectory_valid_object_fraction": batch_metric(
                    "valid_object_fraction"
                ),
                "train/trajectory_skipped_object_count": batch_metric(
                    "skipped_object_count"
                ),
                "train/trajectory_pred_visibility": batch_metric(
                    "mean_pred_visibility_probability"
                ),
                "train/trajectory_pred_below_zero_fraction": float(
                    (pred_raw.detach() < 0.0).float().mean().item()
                ),
                "train/trajectory_pred_above_one_fraction": float(
                    (pred_raw.detach() > 1.0).float().mean().item()
                ),
            }
        )
        timestep_weight, raw_weight, normalizer = self._normalized_timestep_weight(
            pipe, record["timestep"]
        )
        weighted_aux = (
            self.trajectory_loss_weight
            * timestep_weight.to(device=trajectory_loss.device)
            * trajectory_loss
        )
        self._trajectory_forward_count += 1
        metrics["train/trajectory_grad_diag_applied"] = 0.0
        if (
            self._trajectory_forward_count
            % self.trajectory_gradient_diagnostics_every_n_forwards
            == 0
        ):
            metrics.update(
                self._output_gradient_diagnostics(
                    total, weighted_aux, record["model_output"]
                )
            )
        total = total + weighted_aux
        metrics.update(
            {
                "train/trajectory_loss_weight": self.trajectory_loss_weight,
                "train/trajectory_timestep_weight_raw": raw_weight,
                "train/trajectory_timestep_weight_normalizer": normalizer,
                "train/trajectory_timestep_weight": float(
                    timestep_weight.detach().item()
                ),
                "train/trajectory_weighted_contribution": float(
                    weighted_aux.detach().item()
                ),
                "train/trajectory_sigma": float(sigma.detach().flatten()[0].item()),
                "train/loss_total": float(total.detach().item()),
            }
        )
        if self._trajectory_forward_count == 1:
            grad_summary = ""
            if metrics["train/trajectory_grad_diag_applied"]:
                grad_summary = (
                    f", grad_flow={metrics['train/grad_v_main_norm']:.6g}, "
                    f"grad_trajectory={metrics['train/grad_v_trajectory_norm']:.6g}, "
                    "grad_ratio="
                    f"{metrics['train/grad_v_trajectory_to_main_ratio']:.6g}"
                )
            print(
                "[trajectory-loss] first forward: "
                f"flow={metrics['train/loss_main']:.6g}, "
                f"trajectory={metrics['train/loss_trajectory']:.6g}, "
                f"timestep_weight={metrics['train/trajectory_timestep_weight']:.6g}, "
                "weighted_contribution="
                f"{metrics['train/trajectory_weighted_contribution']:.6g}"
                f"{grad_summary}",
                flush=True,
            )
        return total, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = core.build_parser()
    parser.description += " Adds frozen CoTracker object-trajectory loss."
    group = parser.add_argument_group("object_trajectory_loss")
    group.add_argument("--trajectory_cache_dir", required=True)
    group.add_argument("--trajectory_loss_weight", type=float, default=0.1)
    group.add_argument("--trajectory_anchor_frame", type=int, default=4)
    group.add_argument("--trajectory_future_start_frame", type=int, default=8)
    group.add_argument("--trajectory_points_per_object", type=int, default=24)
    group.add_argument("--trajectory_huber_delta", type=float, default=0.01)
    group.add_argument("--trajectory_visibility_threshold", type=float, default=0.9)
    group.add_argument("--trajectory_visibility_loss_weight", type=float, default=0.05)
    group.add_argument(
        "--trajectory_gradient_diagnostics_every_n_forwards", type=int, default=400
    )
    group.add_argument("--trajectory_cotracker_checkpoint", required=True)
    group.add_argument("--tiny_vae_root", required=True)
    group.add_argument("--tiny_vae_checkpoint", required=True)
    group.add_argument("--tiny_vae_parallel", action="store_true")
    return parser


def build_dit_only_model_paths(wan_root: str | Path) -> str:
    root = Path(wan_root).expanduser().resolve()
    shards = [
        root / f"diffusion_pytorch_model-{index:05d}-of-00003.safetensors"
        for index in range(1, 4)
    ]
    if all(path.is_file() for path in shards):
        return json.dumps([[str(path) for path in shards]])
    single = root / "diffusion_pytorch_model.safetensors"
    if single.is_file():
        return json.dumps([str(single)])
    raise FileNotFoundError(f"Wan DiT checkpoint not found under {root}")


def build_dataset(args: argparse.Namespace):
    dataset = official_xssc.build_dataset(args)
    cache = PyBulletTrajectoryCache(
        args.trajectory_cache_dir,
        num_frames=args.num_frames,
        anchor_frame=args.trajectory_anchor_frame,
        points_per_object=args.trajectory_points_per_object,
        track_height=TRACK_HEIGHT,
        track_width=TRACK_WIDTH,
    )
    return TrajectoryCachedDataset(dataset, cache)


def build_model(args: argparse.Namespace, accelerator):
    args.model_paths = build_dit_only_model_paths(args.wan_root)
    args.model_id_with_origin_paths = None
    return core.build_model(
        args,
        accelerator,
        model_class=TrajectoryLossWanModule,
        extra_model_kwargs={
            "trajectory_loss_weight": args.trajectory_loss_weight,
            "trajectory_anchor_frame": args.trajectory_anchor_frame,
            "trajectory_future_start_frame": args.trajectory_future_start_frame,
            "trajectory_points_per_object": args.trajectory_points_per_object,
            "trajectory_huber_delta": args.trajectory_huber_delta,
            "trajectory_visibility_threshold": args.trajectory_visibility_threshold,
            "trajectory_visibility_loss_weight": (
                args.trajectory_visibility_loss_weight
            ),
            "trajectory_gradient_diagnostics_every_n_forwards": (
                args.trajectory_gradient_diagnostics_every_n_forwards
            ),
            "cotracker_checkpoint": args.trajectory_cotracker_checkpoint,
            "tiny_vae_root": args.tiny_vae_root,
            "tiny_vae_checkpoint": args.tiny_vae_checkpoint,
            "tiny_vae_parallel": args.tiny_vae_parallel,
        },
    )


def log_stage_summary(accelerator, model, args: argparse.Namespace) -> None:
    core._log_stage_summary(accelerator, model, args)
    if accelerator.is_main_process:
        accelerator.print(
            "Frozen CoTracker trajectory loss: "
            f"weight={args.trajectory_loss_weight:g}, anchor=F{args.trajectory_anchor_frame:02d}, "
            f"future=F{args.trajectory_future_start_frame:02d}-F{args.num_frames - 1:02d}, "
            f"points/object={args.trajectory_points_per_object}, "
            f"beta={args.trajectory_huber_delta:g}, "
            f"visibility_weight={args.trajectory_visibility_loss_weight:g}, "
            "aggregation=point/time per object -> object equal -> batch mean, "
            "Wan VAE/UMT5/tokenizer=not constructed"
        )


def main() -> None:
    core.main(
        build_parser_fn=build_parser,
        build_model_fn=build_model,
        build_dataset_fn=build_dataset,
        log_stage_summary_fn=log_stage_summary,
    )


if __name__ == "__main__":
    main()
