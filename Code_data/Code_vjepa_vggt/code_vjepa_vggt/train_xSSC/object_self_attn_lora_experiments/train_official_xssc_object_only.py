#!/usr/bin/env python3
"""Validated Object-only training wrapper for the official DINOv2 xSSC."""

from __future__ import annotations

import argparse

import code_vjepa_vggt.train_v_newtrain as tvn
from code_vjepa_vggt.train_xSSC import train_xssc_context_slots as official


def checkpoint_saver_only_on_sync(save_fn):
    """Suppress checkpoint writes during gradient-accumulation micro-steps."""

    def wrapped(*args, **kwargs):
        accelerator = kwargs.get("accelerator")
        if accelerator is None:
            raise TypeError("Checkpoint saver requires accelerator as a keyword")
        if not bool(accelerator.sync_gradients):
            return None
        return save_fn(*args, **kwargs)

    return wrapped


def install_parser_validation() -> None:
    original_build_parser = official.build_parser

    def build_parser() -> argparse.ArgumentParser:
        parser = original_build_parser()
        parser.add_argument("--expected_trainable_params", type=int, required=True)
        return parser

    official.build_parser = build_parser


def install_model_validation() -> None:
    original_build_model = official.build_model

    def build_model(args, accelerator):
        model = original_build_model(args, accelerator)
        trainable = sum(param.numel() for param in model.trainable_modules())
        if trainable != int(args.expected_trainable_params):
            raise RuntimeError(
                "Official xSSC Object-only trainable parameter mismatch: "
                f"expected={args.expected_trainable_params:,}, found={trainable:,}"
            )
        return model

    official.build_model = build_model


def install_synced_checkpoint_saves() -> None:
    original_train_loop = tvn.train_loop

    def train_loop(*args, **kwargs):
        original_save = tvn.save_training_checkpoint_bundle
        tvn.save_training_checkpoint_bundle = checkpoint_saver_only_on_sync(
            original_save
        )
        try:
            return original_train_loop(*args, **kwargs)
        finally:
            tvn.save_training_checkpoint_bundle = original_save

    tvn.train_loop = train_loop


def main() -> None:
    install_parser_validation()
    install_model_validation()
    install_synced_checkpoint_saves()
    official.main()


if __name__ == "__main__":
    main()
