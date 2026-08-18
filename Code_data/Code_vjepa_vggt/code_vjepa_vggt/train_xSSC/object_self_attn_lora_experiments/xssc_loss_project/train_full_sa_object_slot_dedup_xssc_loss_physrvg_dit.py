#!/usr/bin/env python3
"""Train PhysRVG DiT with Full-SA, Object, Slot-Dedup, and xSSC loss.

This is the PhysRVG-initialized counterpart of
``train_full_sa_object_slot_dedup_xssc_loss.py``.  The object branch, slot
de-duplication, and frozen DINOv3 MOVi-C xSSC loss are shared with that
implementation; only the frozen DiT initialization is replaced by the full
PhysRVG Diffusers checkpoint loaded through the DiffSynth converter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from safetensors.torch import load_file

import train_full_sa_object_slot_dedup_xssc_loss as base
import train_xssc_object_self_attn_lora as core
from diffsynth.utils.state_dict_converters.wan_video_dit import (
    WanVideoDiTFromDiffusers,
)


class PhysRVGSlotDedupXSSCFeatureLossWanModule(
    base.SlotDedupXSSCFeatureLossWanModule
):
    """Slot-Dedup xSSC-loss module initialized from full PhysRVG DiT."""

    def __init__(self, *args, physrvg_dit_checkpoint: str, **kwargs) -> None:
        checkpoint = Path(physrvg_dit_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"PhysRVG DiT checkpoint does not exist: {checkpoint}")
        self.physrvg_dit_checkpoint = str(checkpoint)
        self.physrvg_dit_load_info: dict[str, int] = {}
        super().__init__(*args, **kwargs)

    def _initialize_frozen_dit(self, dit):
        # The shared module creates the expected zero-delta pretrained adapter
        # before this hook. Remove it, then strictly replace the full DiT.
        removed_modules = core.merge_and_unload_pretrained_lora(
            dit,
            expected_module_count=self.pretrained_lora_expected_modules,
        )
        source_state = load_file(self.physrvg_dit_checkpoint, device="cpu")
        converted_state = WanVideoDiTFromDiffusers(source_state)
        target_state = dit.state_dict()

        missing = sorted(set(target_state) - set(converted_state))
        unexpected = sorted(set(converted_state) - set(target_state))
        shape_mismatch = sorted(
            key
            for key in set(target_state) & set(converted_state)
            if tuple(target_state[key].shape) != tuple(converted_state[key].shape)
        )
        dropped_source_count = len(source_state) - len(converted_state)
        if missing or unexpected or shape_mismatch or dropped_source_count:
            raise RuntimeError(
                "PhysRVG Diffusers-to-DiffSynth conversion is incomplete: "
                f"source={len(source_state)}, converted={len(converted_state)}, "
                f"target={len(target_state)}, dropped_source={dropped_source_count}, "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
                f"shape_mismatch={shape_mismatch[:8]}"
            )

        dit.load_state_dict(converted_state, strict=True)
        self.physrvg_dit_load_info = {
            "source_tensors": len(source_state),
            "converted_tensors": len(converted_state),
            "target_tensors": len(target_state),
        }
        self.base_dit_initialization = (
            f"PhysRVG full fine-tuned DiT: {self.physrvg_dit_checkpoint} "
            f"({len(converted_state)} tensors, strict load)"
        )
        return removed_modules


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = (
        "PhysRVG Full-SA + Object + Slot-Dedup with frozen DINOv3 MOVi-C xSSC loss"
    )
    parser.add_argument("--physrvg_dit_checkpoint", required=True)
    return parser


def build_model(args, accelerator):
    if args.lora_checkpoint not in (None, ""):
        raise ValueError(
            "PhysRVG initialization must not load a pretrained OpenVid LoRA"
        )
    args.lora_checkpoint = None
    return core.build_model(
        args,
        accelerator,
        model_class=PhysRVGSlotDedupXSSCFeatureLossWanModule,
        extra_model_kwargs={
            "physrvg_dit_checkpoint": args.physrvg_dit_checkpoint,
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
            "xssc_slot_dedup_similarity_threshold": args.xssc_slot_dedup_similarity_threshold,
            "xssc_slot_dedup_similarity_metric": args.xssc_slot_dedup_similarity_metric,
            "xssc_slot_dedup_min_keep": args.xssc_slot_dedup_min_keep,
        },
    )


if __name__ == "__main__":
    base.core.main(
        build_parser_fn=build_parser,
        build_model_fn=build_model,
        log_stage_summary_fn=base.log_stage_summary,
        require_pretrained_lora=False,
    )
