#!/usr/bin/env python3
"""Inference entry point restricted to Full-SA checkpoints without object modules."""

from __future__ import annotations

import infer_xssc_object_self_attn_lora as common

_COMMON_LOAD_RESOLVED_CONFIG = common._load_resolved_config


def _load_resolved_config(checkpoint):
    config, manifest_path = _COMMON_LOAD_RESOLVED_CONFIG(checkpoint)
    if config["adaptation"]["mode"] != "full_sa":
        raise ValueError("Full-SA-only inference requires adaptation.mode='full_sa'")
    if bool(config["adaptation"].get("enable_object_branch", True)):
        raise ValueError(
            "Full-SA-only inference refuses a checkpoint with object branch enabled"
        )
    return config, manifest_path


def _install_runtime_hooks() -> None:
    common._load_resolved_config = _load_resolved_config
    common._install_runtime_hooks()


def main() -> None:
    common.batch_base._install_kubric_runtime_hooks = _install_runtime_hooks
    common.batch_base.main()


if __name__ == "__main__":
    main()
