#!/usr/bin/env python3
"""Freeze the 18 step-500 checkpoints into a paired 40-step A/B watcher config."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "xssc_lora_three_train_watch_config_with_t_head.json"
DEFAULT_OUTPUT = ROOT / "test5_step40_object_identity_count_ab_config.json"
OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/test5_step40_object_identity_count_ab"
)
SOURCE_MANIFEST_ROOTS = (
    Path(
        "/data/gaoya/agent-data/outputs/"
        "xssc_object_self_attn_lora_checkpoint_watch/state/checkpoints"
    ),
    Path(
        "/data/gaoya/agent-data/outputs/"
        "xssc_object_self_attn_lora_three_run_watch/state/checkpoints"
    ),
)
PROMPT_SUFFIX = "Maintain consistent object identity and count throughout the video."
REUSABLE_ORIGINAL_ROOTS = {
    "object_only": (
        "/data/gaoya/agent-data/outputs/test5_step40_object_count_ab/watch/"
        "results/object_only__original/step-000500_steps40_512x896_ctx08_49f"
    ),
    "full_sa": (
        "/data/gaoya/agent-data/outputs/test5_step40_object_count_ab/watch/"
        "results/full_sa__original/step-000500_steps40_512x896_ctx08_49f"
    ),
}
METHOD_ORDER = (
    "object_only",
    "full_sa",
    "full_sa_physrvg_dit",
    "full_sa_no_object",
    "full_sa_no_object_vjepa_loss",
    "full_sa_no_object_xssc_loss_dinov3_movic_step50000",
    "full_sa_object_slot_dedup_xssc50k_xssc_loss_dinov3_movic_step50000",
    "t_head_pck32_s039_latest3350_top100_no_object_xssc_loss_dinov3_movic_step50000",
    "s_head59",
    "t_head70",
    "t_head70_no_object",
    "t_head100_lora_pck32_no_object",
    "full_sa_no_object_pybullet100",
    "full_sa_no_object_kubric100",
    "t_head70_slot_dedup_merge",
    "slot_dedup_merge",
    "slot_dedup_merge_xssc_step050000",
    "t_head70_slot_dedup_merge_xssc_step050000",
)
METHOD_COLORS = {
    "object_only": "#4D4D4D",
    "full_sa": "#D62728",
    "full_sa_physrvg_dit": "#17BECF",
    "full_sa_no_object": "#FF7F0E",
    "full_sa_no_object_vjepa_loss": "#009E73",
    "full_sa_no_object_xssc_loss_dinov3_movic_step50000": "#6F4EAD",
    "full_sa_object_slot_dedup_xssc50k_xssc_loss_dinov3_movic_step50000": "#D55E00",
    "t_head_pck32_s039_latest3350_top100_no_object_xssc_loss_dinov3_movic_step50000": "#CC79A7",
    "s_head59": "#2CA02C",
    "t_head70": "#9467BD",
    "t_head70_no_object": "#E377C2",
    "t_head100_lora_pck32_no_object": "#0072B2",
    "full_sa_no_object_pybullet100": "#00A6A6",
    "full_sa_no_object_kubric100": "#F28E2B",
    "t_head70_slot_dedup_merge": "#17BECF",
    "slot_dedup_merge": "#1F77B4",
    "slot_dedup_merge_xssc_step050000": "#8C564B",
    "t_head70_slot_dedup_merge_xssc_step050000": "#00B894",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_source_manifests() -> dict[str, dict]:
    manifests: dict[str, dict] = {}
    for root in SOURCE_MANIFEST_ROOTS:
        for path in sorted(root.glob("*/step-000500.json")):
            payload = load_json(path)
            key = str(payload["method_key"])
            if key in manifests:
                raise ValueError(f"Duplicate step-500 manifest for {key}: {path}")
            manifests[key] = payload
    expected = set(METHOD_ORDER)
    actual = set(manifests)
    if actual != expected:
        raise ValueError(
            f"Step-500 method mismatch; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return manifests


def validate_checkpoint(path: Path) -> None:
    for name in ("checkpoint.safetensors", "training_state.pt"):
        candidate = path / name
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            raise FileNotFoundError(f"Incomplete checkpoint file: {candidate}")


def build_methods(manifests: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    methods: list[dict] = []
    inventory: list[dict] = []
    for pair_index, key in enumerate(METHOD_ORDER):
        source = manifests[key]
        checkpoint = Path(source["checkpoint_dir"]).resolve()
        validate_checkpoint(checkpoint)
        label = str(source["method_label"])
        inventory.append(
            {
                "pair_index": pair_index,
                "method_key": key,
                "method_label": label,
                "checkpoint_step": 500,
                "checkpoint_dir": str(checkpoint),
                "source_result_root": source.get("result_root"),
            }
        )
        checkpoint_spec = [{"step": 500, "path": str(checkpoint)}]
        methods.extend(
            [
                {
                    "key": f"{key}__original",
                    "label": f"{label} · A original prompt",
                    "scheme_key": key,
                    "scheme_label": label,
                    "color": METHOD_COLORS[key],
                    "condition": "control_original_prompt",
                    "generation_prompt_suffix": "",
                    "evaluation_caption_policy": "original_input_caption",
                    "bootstrap_result_roots": (
                        {"500": REUSABLE_ORIGINAL_ROOTS[key]}
                        if key in REUSABLE_ORIGINAL_ROOTS
                        else {}
                    ),
                    "static_checkpoints": checkpoint_spec,
                    "watch_roots": [],
                },
                {
                    "key": f"{key}__identity_count",
                    "label": f"{label} · B + identity/count prompt",
                    "scheme_key": key,
                    "scheme_label": label,
                    "color": "#167A52",
                    "condition": "treatment_identity_count_prompt",
                    "generation_prompt_suffix": PROMPT_SUFFIX,
                    "evaluation_caption_policy": "original_input_caption",
                    "static_checkpoints": checkpoint_spec,
                    "watch_roots": [],
                },
            ]
        )
    return methods, inventory


def main() -> None:
    args = parse_args()
    config = load_json(BASE_CONFIG)
    manifests = load_source_manifests()
    methods, inventory = build_methods(manifests)

    watch_root = OUTPUT_ROOT / "watch"
    dedicated_hub = OUTPUT_ROOT / "hub"
    config["paths"].update(
        {
            "watch_root": str(watch_root),
            "master_hub_root": str(dedicated_hub),
            "physrvg_physiciq_lora_portal_root": None,
            "legacy_watch_root": None,
            "legacy_physiciq_metrics_root": None,
        }
    )
    config["runtime"].update(
        {
            "poll_seconds": 10,
            "retry_seconds": 60,
            "checkpoint_stability_seconds": 0,
            "gpu_poll_seconds": 30,
            "gpu_ids": [7],
            "gpu_ready_max_used_mib": 8000,
            "expected_cases": 20,
            "num_inference_steps": 40,
            "height": 512,
            "width": 896,
            "num_frames": 49,
            "fps": 30,
            "context_frames": 8,
            "seed": 42,
            "force_inference": False,
        }
    )
    config["physiciq"] = {"enabled": False}
    config["methods"] = methods
    config["site_titles"] = {
        "test5": "test_5 · 40-step Object Identity/Count Prompt A/B",
        "test5_average_metrics": (
            "test_5 · 40-step Object Identity/Count Prompt A/B · 全 case 平均指标"
        ),
    }
    config["ab_experiment"] = {
        "checkpoint_step": 500,
        "num_base_methods": len(METHOD_ORDER),
        "num_conditions": 2,
        "num_cases": 20,
        "expected_videos": len(METHOD_ORDER) * 2 * 20,
        "num_inference_steps": 40,
        "seed": 42,
        "prompt_suffix": PROMPT_SUFFIX,
        "evaluation_caption_policy": "original_input_caption",
        "gpu_id": 7,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    inventory_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.output.resolve()),
        "prompt_suffix": PROMPT_SUFFIX,
        "evaluation_caption_policy": "original_input_caption",
        "checkpoints": inventory,
    }
    (OUTPUT_ROOT / "checkpoint_inventory.json").write_text(
        json.dumps(inventory_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "config": str(args.output.resolve()),
                "methods": len(methods),
                "checkpoints": len(inventory),
                "expected_videos": config["ab_experiment"]["expected_videos"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
