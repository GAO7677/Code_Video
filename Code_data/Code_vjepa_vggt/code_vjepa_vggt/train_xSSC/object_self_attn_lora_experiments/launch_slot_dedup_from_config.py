#!/usr/bin/env python3
"""Launch xSSC slot-dedup training from an experiment JSON."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import launch_from_config as base


DEDUP_TRAIN_SCRIPT = ROOT / "train_xssc_object_self_attn_lora_slot_dedup.py"
DEDUP_MODES = {"none", "mask", "merge"}
DEDUP_SIMILARITY_METRICS = {"mean_frame_cosine", "pooled_cosine"}
_ORIGINAL_VALIDATE_CONFIG = base.validate_config
_ORIGINAL_BUILD_COMMAND = base.build_command


def validate_config(config: dict, config_dir: Path) -> dict:
    normalized = _ORIGINAL_VALIDATE_CONFIG(config, config_dir)
    dedup = dict(normalized.get("conditioning", {}).get("slot_dedup", {}))
    dedup.setdefault("mode", "none")
    dedup.setdefault("similarity_threshold", 0.94)
    dedup.setdefault("similarity_metric", "mean_frame_cosine")
    dedup.setdefault("min_keep", 1)
    if str(dedup["mode"]) not in DEDUP_MODES:
        raise ValueError(f"conditioning.slot_dedup.mode must be one of {sorted(DEDUP_MODES)}")
    threshold = float(dedup["similarity_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("conditioning.slot_dedup.similarity_threshold must be in [0,1]")
    if str(dedup["similarity_metric"]) not in DEDUP_SIMILARITY_METRICS:
        raise ValueError(
            "conditioning.slot_dedup.similarity_metric must be one of "
            f"{sorted(DEDUP_SIMILARITY_METRICS)}"
        )
    if int(dedup["min_keep"]) <= 0:
        raise ValueError("conditioning.slot_dedup.min_keep must be positive")
    normalized["conditioning"]["slot_dedup"] = dedup
    return normalized


def build_command(config: dict, output_dir: Path) -> list[str]:
    original_train_script = base.TRAIN_SCRIPT
    base.TRAIN_SCRIPT = DEDUP_TRAIN_SCRIPT
    try:
        command = _ORIGINAL_BUILD_COMMAND(config, output_dir)
    finally:
        base.TRAIN_SCRIPT = original_train_script
    dedup = config["conditioning"]["slot_dedup"]
    command.extend(
        [
            "--xssc_slot_dedup_mode",
            str(dedup["mode"]),
            "--xssc_slot_dedup_similarity_threshold",
            str(dedup["similarity_threshold"]),
            "--xssc_slot_dedup_similarity_metric",
            str(dedup["similarity_metric"]),
            "--xssc_slot_dedup_min_keep",
            str(dedup["min_keep"]),
        ]
    )
    return command


def main() -> None:
    base.validate_config = validate_config
    base.build_command = build_command
    base.main()


if __name__ == "__main__":
    main()
