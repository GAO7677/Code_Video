#!/usr/bin/env python3
"""Hard-gate validation for the TF-0 M1 soft-scaling smoke runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np


ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/training_free_m1_control_v1"
)
CASE = "0613pybullet_sample_001460_w002"
SEED_DIR = "seed_47326"
BASELINE = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_pck50/"
    "runs/0613pybullet_sample_001460_w002/seed_47326/generated.mp4"
)
STAGE3_KNOCKOUT = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/"
    "visual_samples/attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1/"
    "0613pybullet_sample_001460_w002/seed_47326/"
    "single_object__object_A__self_only__top100_s039r3350/generated.mp4"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_videos(left: Path, right: Path) -> dict[str, Any]:
    a = iio.imread(left)
    b = iio.imread(right)
    if a.shape != b.shape:
        raise RuntimeError(f"video shape mismatch: {a.shape} != {b.shape}")
    delta = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return {
        "left": str(left),
        "right": str(right),
        "shape": list(a.shape),
        "array_equal": bool(np.array_equal(a, b)),
        "mae_uint8": float(delta.mean()),
        "max_abs_uint8": float(delta.max()),
        "mae_unit_range": float(delta.mean() / 255.0),
    }


def main() -> None:
    args = parse_args()
    soft_root = args.root / "soft_scaling" / CASE / SEED_DIR
    alpha0 = soft_root / "single_object__object_A__m1_all_time__top100__alpha_0"
    alpha_minus1 = soft_root / "single_object__object_A__m1_all_time__top100__alpha_m1"
    clean_reference = soft_root / "tf0_reference__clean_runtime_baseline"
    stage3_reference = (
        soft_root
        / "tf0_reference__single_object__object_A__stage3_self_only__top100"
    )
    required = [
        BASELINE,
        STAGE3_KNOCKOUT,
        clean_reference / "generated.mp4",
        clean_reference / "manifest.json",
        stage3_reference / "generated.mp4",
        stage3_reference / "manifest.json",
        alpha0 / "generated.mp4",
        alpha0 / "manifest.json",
        alpha_minus1 / "generated.mp4",
        alpha_minus1 / "manifest.json",
        alpha_minus1 / "dose_metrics.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"TF-0 required artifacts missing: {missing}")

    alpha0_manifest = read_json(alpha0 / "manifest.json")
    alpha_minus1_manifest = read_json(alpha_minus1 / "manifest.json")
    clean_manifest = read_json(clean_reference / "manifest.json")
    stage3_reference_manifest = read_json(stage3_reference / "manifest.json")
    stage3_manifest = read_json(STAGE3_KNOCKOUT.parent / "manifest.json")
    no_op = compare_videos(alpha0 / "generated.mp4", clean_reference / "generated.mp4")
    knockout = compare_videos(
        alpha_minus1 / "generated.mp4", stage3_reference / "generated.mp4"
    )
    archived_no_op = compare_videos(alpha0 / "generated.mp4", BASELINE)
    archived_knockout = compare_videos(
        alpha_minus1 / "generated.mp4", STAGE3_KNOCKOUT
    )

    with np.load(alpha_minus1 / "dose_metrics.npz") as arrays:
        dose_finite_events = int(np.isfinite(arrays["attention_mass"]).sum())
        applied_finite_events = int(np.isfinite(arrays["applied_delta_norm"]).sum())

    checks = {
        "alpha0_decoded_rgb_exact": no_op["array_equal"],
        "alpha0_hook_exact": (
            int(alpha0_manifest["audit"]["noop_mismatch_count"]) == 0
        ),
        "alpha_minus1_video_mae_le_1_over_255": (
            knockout["mae_unit_range"] <= 1.0 / 255.0
        ),
        "clean_reference_declared": bool(clean_manifest["audit"]["reference_clean"]),
        "stage3_reference_head_events_8000": (
            int(stage3_reference_manifest["audit"]["modified_head_events"]) == 8000
        ),
        "selected_heads_equal_stage3": (
            alpha_minus1_manifest["selected_entries"]
            == stage3_manifest["selected_entries"]
        ),
        "tube_tokens_equal_stage3": (
            alpha_minus1_manifest["audit"]["query_token_indices_by_latent_frame"]
            == stage3_manifest["audit"]["query_token_indices_by_latent_frame"]
        ),
        "soft_scaling_head_events_8000": (
            int(alpha_minus1_manifest["audit"]["modified_head_events"]) == 8000
        ),
        "dose_events_8000": dose_finite_events == 8000,
        "applied_dose_events_8000": applied_finite_events == 8000,
        "latent_anchor_count_13": (
            len(alpha_minus1_manifest["audit"]["query_token_indices_by_latent_frame"])
            == 13
        ),
        "fp32_decomposition_hard_gate": (
            bool(
                alpha_minus1_manifest[
                    "fp32_attention_decomposition_audit"
                ]["passed"]
            )
        ),
        "bf16_decomposition_residual_finite": (
            bool(alpha_minus1_manifest["audit"]["decomposition_audited"])
            and int(
                alpha_minus1_manifest["audit"]["decomposition_nonfinite_count"]
            )
            == 0
        ),
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "alpha0_vs_baseline": no_op,
        "alpha_minus1_vs_stage3_knockout": knockout,
        "archived_alpha0_vs_baseline_diagnostic_only": archived_no_op,
        "archived_alpha_minus1_vs_stage3_diagnostic_only": archived_knockout,
        "bf16_decomposition_diagnostic": {
            key: alpha_minus1_manifest["audit"][key]
            for key in (
                "decomposition_mismatch_count",
                "decomposition_max_abs_error",
                "decomposition_max_call_relative_l2_error",
                "decomposition_global_relative_l2_error",
            )
        },
        "dose_finite_events": dose_finite_events,
        "applied_dose_finite_events": applied_finite_events,
    }
    output = args.root / "tf0"
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / ("PASS.json" if passed else "FAIL.json")
    stale_path = output / ("FAIL.json" if passed else "PASS.json")
    stale_path.unlink(missing_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
