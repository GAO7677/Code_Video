"""Stage1B replay fine-tuning with null-object dropout and frozen-Wan preservation."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import code_vjepa_vggt.train0705_kubric_no_gt_box.train_stage1b_context_only_no_gt_box_v_newtrain_kubric as base
import code_vjepa_vggt.train_v_newtrain as tvn
from code_vjepa_vggt.context_wan_v_newtrain import (
    apply_clean_latents_at_indices,
    apply_clean_prefix_to_latents,
    resolve_context_latent_indices_from_frames,
    resolve_num_clean_prefix_latents,
    slice_non_context_latents,
)
from code_vjepa_vggt.data.mixed_replay_no_gt_box_dataset import (
    KubricReplayNoGTBoxDataset,
    OpenVidNoGTBoxDataset,
    WeightedNoGTBoxMixture,
)
from code_vjepa_vggt.data.pybullet0713_no_gt_box_dataset import PyBullet0713NoGTBoxDataset
from code_vjepa_vggt.data.pybullet_raw_no_gt_box_dataset import PyBulletRawNoGTBoxDataset
from code_vjepa_vggt.headonly_val_loss import HeadOnlyValConfig

from diffsynth.diffusion import ModelLogger


def _shared_flow_match_predictions(
    pipe,
    inputs_shared: dict[str, Any],
    inputs_posi: dict[str, Any],
    *,
    object_context: torch.Tensor | None,
    run_teacher: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Run student and no-object teacher at the exact same noise and timestep."""
    max_boundary = int(inputs_shared.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_boundary = int(inputs_shared.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_boundary, max_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    input_latents = inputs_shared["input_latents"]
    noise = torch.randn_like(input_latents)
    training_target = pipe.scheduler.training_target(input_latents, noise, timestep)
    clean_prefix_latents = inputs_shared.get("clean_prefix_latents")
    num_clean_prefix_latents = resolve_num_clean_prefix_latents(
        clean_prefix_latents=clean_prefix_latents,
        num_clean_prefix_latents=inputs_shared.get("num_clean_prefix_latents"),
    )
    context_latent_indices = resolve_context_latent_indices_from_frames(
        raw_frame_indices=inputs_shared.get("context_frame_indices"),
        raw_num_frames=inputs_shared.get("num_frames"),
        latent_length=input_latents.shape[2],
    )
    if num_clean_prefix_latents < 0 or num_clean_prefix_latents >= input_latents.shape[2]:
        raise ValueError(
            "num_clean_prefix_latents must leave at least one supervised latent; "
            f"got {num_clean_prefix_latents}/{input_latents.shape[2]}"
        )
    if context_latent_indices and len(context_latent_indices) >= input_latents.shape[2]:
        raise ValueError("context_latent_indices must leave at least one supervised latent")

    if context_latent_indices:
        noisy_latents = pipe.scheduler.add_noise(input_latents, noise, timestep)
        noisy_latents = apply_clean_latents_at_indices(
            noisy_latents, input_latents, context_latent_indices
        )
    elif num_clean_prefix_latents > 0:
        noisy_latents = input_latents.clone()
        noisy_latents[:, :, num_clean_prefix_latents:] = pipe.scheduler.add_noise(
            input_latents[:, :, num_clean_prefix_latents:],
            noise[:, :, num_clean_prefix_latents:],
            timestep,
        )
        noisy_latents = apply_clean_prefix_to_latents(noisy_latents, clean_prefix_latents)
    else:
        noisy_latents = pipe.scheduler.add_noise(input_latents, noise, timestep)
        if "first_frame_latents" in inputs_shared:
            noisy_latents[:, :, 0:1] = inputs_shared["first_frame_latents"]

    model_inputs = dict(inputs_shared)
    model_inputs.update(inputs_posi)
    model_inputs["latents"] = noisy_latents
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    teacher_prediction = None
    if run_teacher:
        teacher_inputs = dict(model_inputs)
        teacher_inputs["object_context"] = None
        with torch.no_grad():
            teacher_prediction = pipe.model_fn(**models, **teacher_inputs, timestep=timestep)

    student_inputs = dict(model_inputs)
    student_inputs["object_context"] = object_context
    student_prediction = pipe.model_fn(**models, **student_inputs, timestep=timestep)

    def supervised_slice(tensor: torch.Tensor) -> torch.Tensor:
        if context_latent_indices:
            return slice_non_context_latents(
                tensor,
                latent_length=input_latents.shape[2],
                context_latent_indices=context_latent_indices,
            )
        if num_clean_prefix_latents > 0:
            return tensor[:, :, num_clean_prefix_latents:]
        if "first_frame_latents" in inputs_shared:
            return tensor[:, :, 1:]
        return tensor

    student_supervised = supervised_slice(student_prediction)
    target_supervised = supervised_slice(training_target)
    training_weight = pipe.scheduler.training_weight(timestep)
    main_loss = F.mse_loss(student_supervised.float(), target_supervised.float()) * training_weight

    preservation_loss = student_supervised.new_zeros((), dtype=torch.float32)
    teacher_rms = 0.0
    delta_rms = 0.0
    delta_ratio = 0.0
    if teacher_prediction is not None:
        teacher_supervised = supervised_slice(teacher_prediction).detach()
        preservation_loss = (
            F.mse_loss(student_supervised.float(), teacher_supervised.float()) * training_weight
        )
        with torch.no_grad():
            teacher_rms = float(teacher_supervised.float().square().mean().sqrt().item())
            delta_rms = float(
                (student_supervised.detach().float() - teacher_supervised.float())
                .square()
                .mean()
                .sqrt()
                .item()
            )
            delta_ratio = delta_rms / max(teacher_rms, 1.0e-8)

    with torch.no_grad():
        student_rms = float(student_supervised.detach().float().square().mean().sqrt().item())
    diagnostics = {
        "train/teacher_preservation_active": float(run_teacher),
        "train/teacher_prediction_rms": teacher_rms,
        "train/student_prediction_rms": student_rms,
        "train/student_teacher_delta_rms": delta_rms,
        "train/student_teacher_delta_to_teacher_ratio": delta_ratio,
        "train/flow_timestep_id": float(timestep_id.item()),
    }
    return main_loss, preservation_loss, diagnostics


class ReplayPreserveNoGTBoxWanModule(base.ContextOnlyNoGTBoxWanModule):
    def __init__(self, *args, **kwargs) -> None:
        self.object_branch_dropout_prob = float(kwargs.pop("object_branch_dropout_prob", 0.0))
        openvid_dropout_prob = kwargs.pop("openvid_object_branch_dropout_prob", None)
        self.openvid_object_branch_dropout_prob = float(
            self.object_branch_dropout_prob
            if openvid_dropout_prob is None
            else openvid_dropout_prob
        )
        self.lambda_teacher_preservation = float(kwargs.pop("lambda_teacher_preservation", 0.0))
        pybullet_teacher_lambda = kwargs.pop(
            "pybullet_teacher_preservation_lambda", None
        )
        kubric_teacher_lambda = kwargs.pop(
            "kubric_teacher_preservation_lambda", None
        )
        openvid_teacher_lambda = kwargs.pop(
            "openvid_teacher_preservation_lambda", None
        )
        self.teacher_preservation_lambdas = {
            "pybullet": float(
                self.lambda_teacher_preservation
                if pybullet_teacher_lambda is None
                else pybullet_teacher_lambda
            ),
            "kubric": float(
                self.lambda_teacher_preservation
                if kubric_teacher_lambda is None
                else kubric_teacher_lambda
            ),
            "openvid": float(
                self.lambda_teacher_preservation
                if openvid_teacher_lambda is None
                else openvid_teacher_lambda
            ),
        }
        self.teacher_preservation_every_n_steps = int(
            kwargs.pop("teacher_preservation_every_n_steps", 1)
        )
        openvid_teacher_every = kwargs.pop(
            "openvid_teacher_preservation_every_n_steps", None
        )
        self.openvid_teacher_preservation_every_n_steps = int(
            self.teacher_preservation_every_n_steps
            if openvid_teacher_every is None
            else openvid_teacher_every
        )
        self.teacher_preservation_unbiased_interval_scale = bool(
            kwargs.pop("teacher_preservation_unbiased_interval_scale", False)
        )
        replay_fixed_context_frames = kwargs.pop("replay_fixed_context_frames", None)
        self.replay_fixed_context_frames = (
            None
            if replay_fixed_context_frames is None
            else int(replay_fixed_context_frames)
        )
        if not 0.0 <= self.object_branch_dropout_prob <= 1.0:
            raise ValueError("object_branch_dropout_prob must be in [0, 1]")
        if not 0.0 <= self.openvid_object_branch_dropout_prob <= 1.0:
            raise ValueError("openvid_object_branch_dropout_prob must be in [0, 1]")
        if self.lambda_teacher_preservation < 0.0:
            raise ValueError("lambda_teacher_preservation must be non-negative")
        for source_name, source_lambda in self.teacher_preservation_lambdas.items():
            if source_lambda < 0.0:
                raise ValueError(
                    f"{source_name}_teacher_preservation_lambda must be non-negative"
                )
        if self.teacher_preservation_every_n_steps <= 0:
            raise ValueError("teacher_preservation_every_n_steps must be positive")
        if self.openvid_teacher_preservation_every_n_steps <= 0:
            raise ValueError(
                "openvid_teacher_preservation_every_n_steps must be positive"
            )
        if (
            self.replay_fixed_context_frames is not None
            and self.replay_fixed_context_frames <= 0
        ):
            raise ValueError("replay_fixed_context_frames must be positive")
        self._preservation_forward_counts: dict[str, int] = {}
        self._full_dropout_count = 0
        self._full_dropout_total = 0
        self._full_dropout_counts_by_source: dict[str, int] = {}
        self._full_dropout_totals_by_source: dict[str, int] = {}
        self._last_preservation_metrics: dict[str, float] = {}
        super().__init__(*args, **kwargs)

    @staticmethod
    def _dataset_source(inputs_shared: dict[str, Any]) -> str:
        raw_sample = inputs_shared.get("raw_sample", {})
        metadata = raw_sample.get("metadata", {}) if isinstance(raw_sample, dict) else {}
        return str(metadata.get("dataset_source", "unknown")).strip().lower()

    def sample_context_spec(self, video, raw_sample=None):
        if self.replay_fixed_context_frames is None:
            return super().sample_context_spec(video, raw_sample=raw_sample)

        total_frames = len(video)
        if raw_sample is not None:
            raw_video = raw_sample.get("video")
            if isinstance(raw_video, torch.Tensor) and raw_video.ndim >= 2:
                total_frames = int(raw_video.shape[1])
        context_frames = int(self.replay_fixed_context_frames)
        if context_frames >= total_frames:
            raise ValueError(
                "replay_fixed_context_frames must leave at least one target frame; "
                f"got context={context_frames}, total={total_frames}"
            )
        return self._finalize_context_spec(
            "fixed_prefix",
            list(range(context_frames)),
            ctx_max_length=context_frames - 1,
        )

    def _run_main_loss_with_trace(self, pipe, inputs_shared, inputs_posi, object_context):
        source = self._dataset_source(inputs_shared)
        source_teacher_lambda = float(
            self.teacher_preservation_lambdas.get(
                source, self.lambda_teacher_preservation
            )
        )
        source_count = self._preservation_forward_counts.get(source, 0) + 1
        self._preservation_forward_counts[source] = source_count
        teacher_interval = (
            self.openvid_teacher_preservation_every_n_steps
            if source == "openvid"
            else self.teacher_preservation_every_n_steps
        )
        run_teacher = (
            source_teacher_lambda > 0.0
            and (source_count - 1) % teacher_interval == 0
        )
        preservation_interval_scale = (
            float(teacher_interval)
            if run_teacher and self.teacher_preservation_unbiased_interval_scale
            else 1.0
        )
        active_dit = getattr(pipe, "dit", None)
        trace_layers = None
        trace_enabled = bool(
            self.object_branch_train_trace
            or float(self.object_branch_ratio_guard_max_ratio) > 0.0
        )
        if active_dit is not None and hasattr(active_dit, "_object_branch_trace_collect") and trace_enabled:
            active_dit._object_branch_trace_collect = True
            active_dit._object_branch_trace_buffer = []
        try:
            main_loss, preservation_loss, diagnostics = _shared_flow_match_predictions(
                pipe,
                inputs_shared,
                inputs_posi,
                object_context=object_context,
                run_teacher=run_teacher,
            )
            if active_dit is not None and hasattr(active_dit, "_object_branch_trace_buffer"):
                trace_layers = getattr(active_dit, "_object_branch_trace_buffer", None)
        finally:
            if active_dit is not None and hasattr(active_dit, "_object_branch_trace_collect"):
                active_dit._object_branch_trace_collect = False
                active_dit._object_branch_trace_buffer = None

        diagnostics.update(
            {
                "train/loss_main_unregularized": float(main_loss.detach().item()),
                "train/loss_teacher_preservation": float(preservation_loss.detach().item()),
                "train/loss_teacher_preservation_weighted": float(
                    source_teacher_lambda
                    * preservation_interval_scale
                    * preservation_loss.detach().item()
                ),
                "train/teacher_preservation_source_lambda": source_teacher_lambda,
                "train/teacher_preservation_interval": float(teacher_interval),
                "train/teacher_preservation_interval_scale": preservation_interval_scale,
                "train/teacher_preservation_effective_coefficient": float(
                    source_teacher_lambda * preservation_interval_scale
                    if run_teacher
                    else 0.0
                ),
            }
        )
        self._last_preservation_metrics = diagnostics
        return (
            main_loss
            + source_teacher_lambda
            * preservation_interval_scale
            * preservation_loss,
            trace_layers,
        )

    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        source = self._dataset_source(inputs_shared)
        dropout_prob = (
            self.openvid_object_branch_dropout_prob
            if source == "openvid"
            else self.object_branch_dropout_prob
        )
        full_dropout = bool(
            dropout_prob > 0.0
            and torch.rand((), device=pipe.device) < dropout_prob
        )
        self._full_dropout_total += 1
        self._full_dropout_count += int(full_dropout)
        self._full_dropout_totals_by_source[source] = (
            self._full_dropout_totals_by_source.get(source, 0) + 1
        )
        self._full_dropout_counts_by_source[source] = (
            self._full_dropout_counts_by_source.get(source, 0) + int(full_dropout)
        )
        effective_inputs = inputs_shared
        if full_dropout:
            effective_inputs = dict(inputs_shared)
            raw_sample = dict(inputs_shared["raw_sample"])
            raw_sample["num_context_frames"] = 0
            raw_sample["context_frame_indices"] = torch.empty(0, dtype=torch.long)
            effective_inputs["raw_sample"] = raw_sample

        total, metrics = super()._compute_object_losses(pipe, effective_inputs, inputs_posi)
        combined_main = float(metrics.get("train/loss_main", 0.0))
        metrics.update(self._last_preservation_metrics)
        metrics["train/loss_main_with_preservation"] = combined_main
        metrics["train/loss_main"] = float(
            self._last_preservation_metrics.get("train/loss_main_unregularized", combined_main)
        )
        metrics["train/object_branch_full_dropout"] = float(full_dropout)
        metrics["train/object_branch_full_dropout_running_fraction"] = (
            self._full_dropout_count / max(self._full_dropout_total, 1)
        )
        metrics["train/object_branch_source_dropout_probability"] = float(dropout_prob)
        metrics["train/object_branch_source_dropout_running_fraction"] = (
            self._full_dropout_counts_by_source[source]
            / max(self._full_dropout_totals_by_source[source], 1)
        )
        metrics["train/openvid_detected_object_mode"] = float(
            source == "openvid" and not full_dropout
        )
        metrics["train/openvid_null_object_mode"] = float(
            source == "openvid" and full_dropout
        )
        metrics["train/object_condition_num_context_frames"] = float(
            0 if full_dropout else inputs_shared["raw_sample"].get("num_context_frames", 0)
        )
        metrics["train/sampled_ctx_num_frames"] = float(
            inputs_shared["raw_sample"].get("num_context_frames", 0)
        )

        source_ids = {"pybullet": 0, "kubric": 1, "openvid": 2}
        metrics["train/dataset_source_id"] = float(source_ids.get(source, -1))
        metrics["train/replay_fixed_context_frames"] = float(
            self.replay_fixed_context_frames
            if self.replay_fixed_context_frames is not None
            else -1
        )
        for source_name in source_ids:
            metrics[f"train/dataset_source_{source_name}"] = float(source == source_name)
        return total, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = "Stage1B replay-preservation training on PyBullet, Kubric, and OpenVid."
    for action in parser._actions:
        if action.dest == "dataset_type" and "replay_preserve_mix" not in action.choices:
            action.choices = [*action.choices, "replay_preserve_mix"]
            break

    regularization = parser.add_argument_group("replay_preservation")
    regularization.add_argument("--stage2_init_from", default=None)
    regularization.add_argument("--object_branch_dropout_prob", type=float, default=0.20)
    regularization.add_argument(
        "--openvid_object_branch_dropout_prob", type=float, default=None
    )
    regularization.add_argument("--lambda_teacher_preservation", type=float, default=0.05)
    regularization.add_argument(
        "--pybullet_teacher_preservation_lambda", type=float, default=None
    )
    regularization.add_argument(
        "--kubric_teacher_preservation_lambda", type=float, default=None
    )
    regularization.add_argument(
        "--openvid_teacher_preservation_lambda", type=float, default=None
    )
    regularization.add_argument("--teacher_preservation_every_n_steps", type=int, default=4)
    regularization.add_argument(
        "--openvid_teacher_preservation_every_n_steps", type=int, default=None
    )
    regularization.add_argument(
        "--teacher_preservation_unbiased_interval_scale", action="store_true"
    )
    regularization.add_argument("--replay_fixed_context_frames", type=int, default=None)

    mixture = parser.add_argument_group("replay_mixture")
    mixture.add_argument("--openvid_root", type=str, default=None)
    mixture.add_argument("--openvid_max_samples", type=int, default=None)
    mixture.add_argument("--kubric_replay_index_num_frames", type=int, default=69)
    mixture.add_argument("--kubric_replay_index_num_context_frames", type=int, default=20)
    mixture.add_argument("--replay_pybullet_dataset", choices=["raw", "0713"], default="raw")
    mixture.add_argument("--mixture_pybullet_ratio", type=float, default=0.30)
    mixture.add_argument("--mixture_kubric_ratio", type=float, default=0.30)
    mixture.add_argument("--mixture_openvid_ratio", type=float, default=0.40)
    return parser


def build_dataset(args: argparse.Namespace):
    if args.dataset_type != "replay_preserve_mix":
        return base.build_dataset(args)
    if not args.kubric_root or not args.openvid_root:
        raise ValueError("replay_preserve_mix requires kubric_root and openvid_root")

    resolution = (args.height, args.width)
    if args.replay_pybullet_dataset == "0713":
        if not args.pybullet0713_root:
            raise ValueError(
                "replay_preserve_mix with --replay_pybullet_dataset 0713 requires "
                "--pybullet0713_root"
            )
        pybullet = PyBullet0713NoGTBoxDataset(
            root=args.pybullet0713_root,
            split=args.pybullet0713_split,
            resolution=resolution,
            num_frames=args.num_frames,
            num_context_frames=args.fixed_num_context_frames,
            sampling_strategy=args.pybullet0713_sampling_strategy,
            families=args.pybullet0713_family,
            init_scan_limit=args.pybullet0713_init_scan_limit,
            split_train_ratio=args.pybullet0713_split_train_ratio,
            split_val_ratio=args.pybullet0713_split_val_ratio,
            max_retry_samples=args.pybullet0713_max_retry_samples,
        )
    else:
        if not args.pybullet_raw_root:
            raise ValueError(
                "replay_preserve_mix with --replay_pybullet_dataset raw requires "
                "--pybullet_raw_root"
            )
        pybullet = PyBulletRawNoGTBoxDataset(
            root=args.pybullet_raw_root,
            split=args.pybullet_raw_split,
            resolution=resolution,
            num_frames=args.num_frames,
            num_context_frames=args.fixed_num_context_frames,
            sampling_strategy=args.pybullet_raw_sampling_strategy,
            window_starts=tuple(
                int(value.strip())
                for value in args.pybullet_raw_window_starts.split(",")
                if value.strip()
            ),
            init_scan_limit=args.pybullet_raw_init_scan_limit,
        )
    kubric = KubricReplayNoGTBoxDataset(
        root=args.kubric_root,
        split=args.kubric_split,
        resolution=resolution,
        num_frames=args.num_frames,
        num_context_frames=args.fixed_num_context_frames,
        index_num_frames=args.kubric_replay_index_num_frames,
        index_num_context_frames=args.kubric_replay_index_num_context_frames,
        sampling_strategy=args.kubric_sampling_strategy,
        seed=42,
        scenarios=args.kubric_scenario,
        init_scan_limit=args.kubric_init_scan_limit,
        cache_root=args.kubric_cache_root,
        split_train_ratio=args.kubric_split_train_ratio,
        split_val_ratio=args.kubric_split_val_ratio,
        max_retry_samples=args.kubric_max_retry_samples,
    )
    openvid = OpenVidNoGTBoxDataset(
        root=args.openvid_root,
        resolution=resolution,
        num_frames=args.num_frames,
        num_context_frames=args.fixed_num_context_frames,
        max_samples=args.openvid_max_samples,
    )
    return WeightedNoGTBoxMixture(
        datasets=(pybullet, kubric, openvid),
        source_names=("pybullet", "kubric", "openvid"),
        source_probabilities=(
            args.mixture_pybullet_ratio,
            args.mixture_kubric_ratio,
            args.mixture_openvid_ratio,
        ),
    )


def build_model(args: argparse.Namespace, accelerator) -> ReplayPreserveNoGTBoxWanModule:
    original_class = base.ContextOnlyNoGTBoxWanModule

    def factory(*model_args, **model_kwargs):
        return ReplayPreserveNoGTBoxWanModule(
            *model_args,
            **model_kwargs,
            object_branch_dropout_prob=args.object_branch_dropout_prob,
            openvid_object_branch_dropout_prob=args.openvid_object_branch_dropout_prob,
            lambda_teacher_preservation=args.lambda_teacher_preservation,
            pybullet_teacher_preservation_lambda=(
                args.pybullet_teacher_preservation_lambda
            ),
            kubric_teacher_preservation_lambda=args.kubric_teacher_preservation_lambda,
            openvid_teacher_preservation_lambda=args.openvid_teacher_preservation_lambda,
            teacher_preservation_every_n_steps=args.teacher_preservation_every_n_steps,
            openvid_teacher_preservation_every_n_steps=(
                args.openvid_teacher_preservation_every_n_steps
            ),
            teacher_preservation_unbiased_interval_scale=(
                args.teacher_preservation_unbiased_interval_scale
            ),
            replay_fixed_context_frames=args.replay_fixed_context_frames,
        )

    base.ContextOnlyNoGTBoxWanModule = factory
    try:
        return base.build_model(args, accelerator)
    finally:
        base.ContextOnlyNoGTBoxWanModule = original_class


def _load_stage2_trainables(model, checkpoint: str) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "checkpoint.safetensors"
    elif checkpoint_path.name == "training_state.pt":
        checkpoint_path = Path(tvn.resolve_lora_checkpoint_for_resume(str(checkpoint_path)))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Stage1B model checkpoint not found: {checkpoint_path}")
    return tvn._load_filtered_checkpoint_into_model(
        model,
        str(checkpoint_path),
        include_prefixes=("object_adapter.",),
        include_substrings=(
            "object_embedding",
            ".object_cross_attn.",
            ".object_gate",
            ".norm4.",
        ),
    )


def main() -> None:
    args = tvn.prepare_args(build_parser().parse_args())
    if args.stage2_init_from is not None and args.stage2_resume_from is not None:
        raise ValueError("--stage2_init_from and --stage2_resume_from are mutually exclusive")
    previous_handlers = tvn.install_interrupt_handlers()
    accelerator = tvn.build_accelerator(args)
    tvn.init_trackers(accelerator, args)

    dataset = build_dataset(args)
    if accelerator.is_main_process and hasattr(dataset, "dataset_stats"):
        accelerator.print(f"Replay mixture: {dataset.dataset_stats}")
    disabled_val = HeadOnlyValConfig(
        enabled=False,
        split="val",
        every_steps=None,
        num_batches=1,
    )
    model = build_model(args, accelerator)

    if args.stage1a_init_from is not None:
        info = tvn._load_filtered_checkpoint_into_model(
            model,
            args.stage1a_init_from,
            include_prefixes=("object_pooler.", "object_aux_heads."),
        )
        accelerator.print(
            "Loaded Stage1A token builder: "
            f"loaded={info['loaded_count']} shape_mismatch={len(info['skipped_shape_mismatch'])}"
        )
    stage2_source = args.stage2_init_from or args.stage2_resume_from
    if stage2_source is not None:
        info = _load_stage2_trainables(model, stage2_source)
        mode = "model-only initialization (fresh optimizer)" if args.stage2_init_from else "resume"
        accelerator.print(
            f"Loaded Stage1B {mode}: source={stage2_source} "
            f"loaded={info['loaded_count']} shape_mismatch={len(info['skipped_shape_mismatch'])}"
        )

    base._log_stage_summary(accelerator, model, args)
    accelerator.print(
        "Replay preservation: "
        f"physics_dropout={args.object_branch_dropout_prob:.3f}, "
        f"openvid_dropout={args.openvid_object_branch_dropout_prob}, "
        f"legacy_teacher_lambda={args.lambda_teacher_preservation:.4f}, "
        f"pybullet_teacher_lambda={args.pybullet_teacher_preservation_lambda}, "
        f"kubric_teacher_lambda={args.kubric_teacher_preservation_lambda}, "
        f"openvid_teacher_lambda={args.openvid_teacher_preservation_lambda}, "
        f"teacher_every={args.teacher_preservation_every_n_steps}, "
        f"openvid_teacher_every={args.openvid_teacher_preservation_every_n_steps}, "
        f"unbiased_interval_scale={args.teacher_preservation_unbiased_interval_scale}, "
        f"fixed_context_frames={args.replay_fixed_context_frames}"
    )
    model_logger = ModelLogger(
        tvn.get_checkpoint_dir(args),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    runtime_state: dict[str, Any] = {}
    try:
        tvn.train_loop(
            accelerator,
            dataset,
            model,
            model_logger,
            args,
            runtime_state=runtime_state,
            headonly_val_dataloader=None,
            headonly_val_config=disabled_val,
        )
    except (KeyboardInterrupt, tvn.TrainingInterrupted) as exc:
        checkpoint_root = tvn.get_checkpoint_dir(args)
        model_logger.save_model(
            accelerator,
            model,
            tvn.training_checkpoint_file(checkpoint_root, "interrupted-latest"),
        )
        optimizer = runtime_state.get("optimizer")
        scheduler = runtime_state.get("scheduler")
        progress = runtime_state.get("progress", {})
        if optimizer is not None and scheduler is not None:
            tvn.save_training_state(
                accelerator=accelerator,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=progress.get("global_step", 0),
                epoch_id=progress.get("epoch_id", 0),
                batch_in_epoch=progress.get("batch_in_epoch", 0),
                model_logger=model_logger,
                state_path=tvn.training_state_file(checkpoint_root, "interrupted-latest"),
            )
        accelerator.end_training()
        tvn.restore_interrupt_handlers(previous_handlers)
        raise SystemExit(130) from exc

    accelerator.end_training()
    tvn.restore_interrupt_handlers(previous_handlers)


if __name__ == "__main__":
    main()
