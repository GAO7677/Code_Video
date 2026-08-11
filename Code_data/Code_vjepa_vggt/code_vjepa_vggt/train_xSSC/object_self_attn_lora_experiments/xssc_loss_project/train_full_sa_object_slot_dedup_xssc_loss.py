#!/usr/bin/env python3
"""Train Full-SA + Object + Slot-Dedup with a frozen xSSC feature loss.

This combines the already-used slot-track de-duplication implementation from
``train_xssc_object_self_attn_lora_slot_dedup.py`` with the differentiable
Tiny-VAE -> DINOv3 MOVi-C xSSC loss implementation in this package.  All
dataset, Wan/OpenVid initialization, optimizer, checkpoint, and DDP behavior
continues to use the shared training entry point.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_XSSC_ROOT = EXPERIMENT_ROOT.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[2]
for _path in (EXPERIMENT_ROOT, TRAIN_XSSC_ROOT, REPOSITORY_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch

import train_xssc_object_self_attn_lora as core
import train_xssc_object_self_attn_lora_slot_dedup as slot_dedup
from xssc_loss_project import train_xssc_object_self_attn_lora_xssc_loss as xssc_loss


class SlotDedupXSSCFeatureLossWanModule(xssc_loss.XSSCFeatureLossWanModule):
    """Object-conditioned Full-SA module with slot de-duplication and xSSC loss."""

    require_object_branch_disabled = False

    def __init__(
        self,
        *args,
        xssc_slot_dedup_mode: str,
        xssc_slot_dedup_similarity_threshold: float,
        xssc_slot_dedup_similarity_metric: str,
        xssc_slot_dedup_min_keep: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not self.enable_object_branch:
            raise ValueError("Slot-Dedup xSSC-loss training requires the object branch")
        if self.self_attn_adaptation_mode != "full_sa":
            raise ValueError("Slot-Dedup xSSC-loss training requires Full-SA adaptation")

        self.xssc_slot_dedup_mode = str(xssc_slot_dedup_mode)
        self.xssc_slot_dedup_similarity_threshold = float(
            xssc_slot_dedup_similarity_threshold
        )
        self.xssc_slot_dedup_similarity_metric = str(
            xssc_slot_dedup_similarity_metric
        )
        self.xssc_slot_dedup_min_keep = int(xssc_slot_dedup_min_keep)
        if self.xssc_slot_dedup_mode == "none":
            raise ValueError("This training variant requires Slot-Dedup to be enabled")

        self._last_slot_dedup_stats: dict[str, float] = {
            "enabled": 1.0,
            "threshold": self.xssc_slot_dedup_similarity_threshold,
            "retained_slots_mean": float(self.xssc_num_slots),
            "duplicate_fraction_mean": 0.0,
            "groups_per_sample_mean": float(self.xssc_num_slots),
            "max_group_size_mean": 1.0,
            "mean_offdiag_similarity": 0.0,
            "mean_duplicate_pair_similarity": 0.0,
        }

    def _build_xssc_boxes(self, xssc_video: torch.Tensor) -> torch.Tensor:
        # The xSSC-loss path offloads the shared SAM2 AMG model to CPU after
        # building its target query.  Move that already-created model back to
        # the current rank's device before the object branch reuses it on the
        # next micro-step; AMGBoxBuilder._get_generator() otherwise returns
        # the cached CPU instance unchanged.
        generator = getattr(self.xssc_box_builder, "_generator", None)
        if generator is not None:
            generator.predictor.model.to(device=xssc_video.device)
        return super()._build_xssc_boxes(xssc_video)

    def _build_object_context(
        self,
        context_video: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return slot_dedup._build_object_context_with_dedup(self, context_video)

    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        total, metrics = super()._compute_object_losses(
            pipe,
            inputs_shared,
            inputs_posi,
        )
        prefix = "train/xssc_slot_dedup_"
        for key, value in self._last_slot_dedup_stats.items():
            metrics[f"{prefix}{key}"] = float(value)
        metrics[f"{prefix}mode_merge"] = float(
            self.xssc_slot_dedup_mode == "merge"
        )
        metrics[f"{prefix}mode_mask"] = float(
            self.xssc_slot_dedup_mode == "mask"
        )
        return total, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = xssc_loss.build_parser()
    parser.description += " Enables the existing xSSC slot-track de-duplication path."
    group = parser.add_argument_group("xssc_slot_track_dedup")
    group.add_argument(
        "--xssc_slot_dedup_mode",
        choices=slot_dedup.DEDUP_MODES,
        required=True,
    )
    group.add_argument(
        "--xssc_slot_dedup_similarity_threshold",
        type=float,
        default=0.94,
    )
    group.add_argument(
        "--xssc_slot_dedup_similarity_metric",
        choices=slot_dedup.DEDUP_SIMILARITY_METRICS,
        default="mean_frame_cosine",
    )
    group.add_argument("--xssc_slot_dedup_min_keep", type=int, default=3)
    return parser


def build_model(args: argparse.Namespace, accelerator):
    return core.build_model(
        args,
        accelerator,
        model_class=SlotDedupXSSCFeatureLossWanModule,
        extra_model_kwargs={
            "xssc_loss_backend": args.xssc_loss_backend,
            "xssc_loss_weight": args.xssc_loss_weight,
            "xssc_loss_future_start_frame": args.xssc_loss_future_start_frame,
            "xssc_loss_backbone_chunk_size": args.xssc_loss_backbone_chunk_size,
            "xssc_loss_gradient_diagnostics_every_n_forwards": (
                args.xssc_loss_gradient_diagnostics_every_n_forwards
            ),
            "tiny_vae_root": args.tiny_vae_root,
            "tiny_vae_checkpoint": args.tiny_vae_checkpoint,
            "tiny_vae_parallel": args.tiny_vae_parallel,
            "xssc_slot_dedup_mode": args.xssc_slot_dedup_mode,
            "xssc_slot_dedup_similarity_threshold": (
                args.xssc_slot_dedup_similarity_threshold
            ),
            "xssc_slot_dedup_similarity_metric": (
                args.xssc_slot_dedup_similarity_metric
            ),
            "xssc_slot_dedup_min_keep": args.xssc_slot_dedup_min_keep,
        },
    )


def log_stage_summary(accelerator, model, args: argparse.Namespace) -> None:
    xssc_loss.log_stage_summary(accelerator, model, args)
    if not accelerator.is_main_process:
        return
    accelerator.print(
        "\n".join(
            [
                "xSSC slot-track de-duplication:",
                f"  mode={model.xssc_slot_dedup_mode}",
                f"  similarity_metric={model.xssc_slot_dedup_similarity_metric}",
                f"  similarity_threshold={model.xssc_slot_dedup_similarity_threshold:g}",
                f"  min_keep={model.xssc_slot_dedup_min_keep}",
                "  object encoder and xSSC-loss encoder share the frozen xSSC-50k model",
            ]
        )
    )


def main() -> None:
    core.main(
        build_parser_fn=build_parser,
        build_model_fn=build_model,
        log_stage_summary_fn=log_stage_summary,
    )


if __name__ == "__main__":
    main()
