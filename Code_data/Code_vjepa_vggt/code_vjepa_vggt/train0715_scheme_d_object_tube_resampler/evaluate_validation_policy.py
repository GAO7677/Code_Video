#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_policy(
    payload: dict[str, object],
    *,
    min_object_effect_mae: float = 0.002,
    high_scale_to_zero_ratio: float = 1.05,
    high_scale_changed_fraction: float = 0.20,
) -> dict[str, object]:
    cases = payload.get("cases")
    if not isinstance(cases, dict) or not cases:
        raise ValueError("variant metrics contain no cases")
    warnings: list[str] = []
    case_reports: dict[str, object] = {}
    for case_name, raw_variants in cases.items():
        if not isinstance(raw_variants, dict):
            raise ValueError(f"invalid variants for {case_name}")
        required = (
            "no_object_context",
            "object_residual_0p75x",
            "object_residual_1p5x",
        )
        if any(name not in raw_variants for name in required):
            raise ValueError(f"missing validation variant for {case_name}")

        def future_metric(variant: str, key: str) -> float:
            return float(raw_variants[variant]["future_frames"][key])

        zero_mae = future_metric("no_object_context", "mae_0_1")
        low_mae = future_metric("object_residual_0p75x", "mae_0_1")
        high_mae = future_metric("object_residual_1p5x", "mae_0_1")
        high_changed = future_metric(
            "object_residual_1p5x", "changed_pixel_fraction_gt_5_255"
        )
        high_to_zero = high_mae / max(zero_mae, 1.0e-12)
        object_effect_detected = zero_mae >= float(min_object_effect_mae)
        high_scale_nonmonotonic = high_to_zero > float(high_scale_to_zero_ratio)
        high_scale_large_change = high_changed > float(high_scale_changed_fraction)
        if not object_effect_detected:
            warnings.append(
                f"{case_name}: zero-object effect is weak (future MAE={zero_mae:.6f})"
            )
        if high_scale_nonmonotonic:
            warnings.append(
                f"{case_name}: 1.5x deviates more than zero-object "
                f"(ratio={high_to_zero:.3f})"
            )
        if high_scale_large_change:
            warnings.append(
                f"{case_name}: 1.5x changed-pixel fraction is high "
                f"({high_changed:.3f})"
            )
        case_reports[str(case_name)] = {
            "zero_object_future_mae_0_1": zero_mae,
            "scale_0p75_future_mae_0_1": low_mae,
            "scale_1p5_future_mae_0_1": high_mae,
            "scale_1p5_to_zero_mae_ratio": high_to_zero,
            "scale_1p5_changed_pixel_fraction_gt_5_255": high_changed,
            "object_effect_detected": object_effect_detected,
            "scale_1p5_nonmonotonic_risk": high_scale_nonmonotonic,
            "scale_1p5_large_change_risk": high_scale_large_change,
        }
    return {
        "status": "warning" if warnings else "healthy",
        "thresholds": {
            "min_object_effect_mae": float(min_object_effect_mae),
            "high_scale_to_zero_ratio": float(high_scale_to_zero_ratio),
            "high_scale_changed_fraction": float(high_scale_changed_fraction),
        },
        "warnings": warnings,
        "cases": case_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    report = evaluate_policy(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "warnings": report["warnings"]}))


if __name__ == "__main__":
    main()
