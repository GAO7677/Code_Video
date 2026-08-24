#!/usr/bin/env python3
"""Register complete Scene-Enabled test roots for the existing metric queues.

This is intentionally a one-shot bridge.  It does not run inference, refresh
the large dashboard, or alter existing metric markers; it only creates the
test_5 checkpoint manifests and PhysicIQ inference manifests required by the
two existing parallel metric runners.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


WATCHER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WATCHER_DIR))

from xssc_lora_checkpoint_watch import (  # noqa: E402
    atomic_write_json,
    load_json,
    method_config,
    state_paths,
    timestamp,
    validate_result_root,
)
from xssc_lora_physiciq_watch import phys_manifest_path  # noqa: E402


CONFIG_PATH = WATCHER_DIR / "xssc_lora_three_train_watch_config_with_t_head.json"
METHOD_KEY = "full_sa_physrvg_vjepa_utonia_scene_hardmask_v1_enabled"
OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "physrvg_full_sa_vjepa_utonia_scene_enabled_eval"
)
CHECKPOINT_ROOTS = (
    Path(
        "/data/gaoya/agent-data/checkpoints/physrvg_full_sa_vjepa_utonia_scene/"
        "full-sa-pybullet-physrvg-vjepa-utonia-scene-hardmask-v1-formal-"
        "b2-gacc2-bf16-restart-20260821/checkpoints"
    ),
    Path(
        "/data/gaoya/agent-data/checkpoints/physrvg_full_sa_vjepa_utonia_scene/"
        "full-sa-pybullet-physrvg-vjepa-utonia-scene-hardmask-v1-formal-"
        "b4-gacc1-resume1000-20260822/checkpoints"
    ),
)
TEST5_LIST = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
PHYSICIQ_LIST = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt"
)
STEPS = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500)


def checkpoint_for_step(step: int) -> Path:
    for root in CHECKPOINT_ROOTS:
        candidate = root / f"step-{step:06d}"
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"checkpoint step not found: {step}")


def output_root(step: int, dataset: str) -> Path:
    inference_steps = 8 if dataset == "test5" else 40
    return OUTPUT_ROOT / dataset / (
        f"full_sa_physrvg_vjepa_utonia_scene_enabled_step-{step:06d}_"
        f"steps{inference_steps}_512x896_ctx08_49f"
    )


def register(config: dict) -> list[dict[str, object]]:
    paths = state_paths(config)
    method = method_config(config, METHOD_KEY)
    method_index = next(
        index for index, item in enumerate(config["methods"])
        if item["key"] == METHOD_KEY
    )
    records: list[dict[str, object]] = []
    for step in STEPS:
        checkpoint = checkpoint_for_step(step)
        test5_root = output_root(step, "test5")
        phys_root = output_root(step, "physiciq")
        test5_validation = validate_result_root(
            config,
            test5_root,
            input_list=TEST5_LIST,
            expected_cases=20,
        )
        phys_validation = validate_result_root(
            config,
            phys_root,
            input_list=PHYSICIQ_LIST,
            expected_cases=67,
        )
        common = {
            "method_key": METHOD_KEY,
            "method_label": method["label"],
            "method_index": method_index,
            "step": step,
            "checkpoint_dir": str(checkpoint),
            "origin": "scene-enabled-online-queue",
            "condition": "utonia_scene_enabled_online_vggt_frame7",
            "utonia_scene_source": "online",
            "utonia_scene_token_count": 1792,
            "utonia_scene_feature_dim": 1386,
        }
        test5_payload = {
            **common,
            "result_root": str(test5_root),
            "inference_completed_utc": timestamp(),
            "validation": test5_validation,
        }
        phys_payload = {
            **common,
            "result_root": str(phys_root),
            "input_list": str(PHYSICIQ_LIST),
            "num_inference_steps": 40,
            "completed_utc": timestamp(),
            "validation": phys_validation,
        }
        test5_path = paths["checkpoints"] / METHOD_KEY / f"step-{step:06d}.json"
        phys_path = phys_manifest_path(config, METHOD_KEY, step)
        atomic_write_json(test5_path, test5_payload)
        atomic_write_json(phys_path, phys_payload)
        records.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "test5_manifest": str(test5_path),
                "physiciq_manifest": str(phys_path),
                "test5_cases": test5_validation["num_cases"],
                "physiciq_cases": phys_validation["num_cases"],
            }
        )
    return records


def main() -> None:
    config = load_json(CONFIG_PATH)
    config["_config_path"] = str(CONFIG_PATH.resolve())
    print(json.dumps(register(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
