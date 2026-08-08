#!/usr/bin/env python3
"""Train Full-SA/Object adapters from the frozen PhysRVG fine-tuned DiT."""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn
from safetensors.torch import load_file

import train_xssc_object_self_attn_lora as train
from diffsynth.utils.state_dict_converters.wan_video_dit import (
    WanVideoDiTFromDiffusers,
)


class PhysRVGDiTDINOv3XSSCContextSlotsWanModule(
    train.DINOv3XSSCContextSlotsWanModule
):
    """Replace the stock-Wan/OpenVid initialization with the full PhysRVG DiT."""

    def __init__(self, *args, physrvg_dit_checkpoint: str, **kwargs) -> None:
        checkpoint = Path(physrvg_dit_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"PhysRVG DiT checkpoint does not exist: {checkpoint}")
        self.physrvg_dit_checkpoint = str(checkpoint)
        self.physrvg_dit_load_info: dict[str, int] = {}
        super().__init__(*args, **kwargs)

    def _initialize_frozen_dit(self, dit: nn.Module) -> list[str]:
        # DiffSynth injects a fresh, zero-delta LoRA training wrapper before this
        # hook. Remove it without changing the Wan weights, then replace every
        # DiT tensor with the PhysRVG full-finetune checkpoint.
        removed_modules = train.merge_and_unload_pretrained_lora(
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
                "PhysRVG Diffusers-to-DiffSynth DiT conversion is incomplete: "
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


def build_parser():
    parser = train.build_parser()
    parser.add_argument("--physrvg_dit_checkpoint", required=True)
    return parser


def build_model(args, accelerator):
    if args.lora_checkpoint not in (None, ""):
        raise ValueError(
            "PhysRVG DiT initialization must not also load an OpenVid LoRA checkpoint"
        )
    args.lora_checkpoint = None
    return train.build_model(
        args,
        accelerator,
        model_class=PhysRVGDiTDINOv3XSSCContextSlotsWanModule,
        extra_model_kwargs={
            "physrvg_dit_checkpoint": args.physrvg_dit_checkpoint,
        },
    )


if __name__ == "__main__":
    train.main(
        build_parser_fn=build_parser,
        build_model_fn=build_model,
        require_pretrained_lora=False,
    )
