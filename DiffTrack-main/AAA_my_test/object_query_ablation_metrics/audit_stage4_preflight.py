#!/usr/bin/env python3
"""Audit exact Stage-4 pilot cells and directional-dose reusability."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_SPEC = HERE / "experiment_spec_stage4_temporal_v1.json"
DEFAULT_ROOTS = (
    Path(
        "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
        "latest3350_v1/stage3_discovery_videos"
    ),
    Path(
        "/data/gaoya/agent-data/outputs/"
        "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
        "attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1"
    ),
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/stage4_preflight_inventory"
)
FLOW_MODES = {
    "M1": "self",
    "M2": "incoming",
    "M3": "outgoing",
}
DIRECTIONS = ("same", "future", "past")
FULL_SCOPES = ("top100", "bottom100", "random100_layer_matched_draw0")
CURRENT_DIRECTIONAL_PROTOCOL = "attention_matrix_ablation_temporal_direction_v2_dose"
BASE_DOSE_KEYS = (
    "attention_mass",
    "removed_value_norm",
    "original_output_norm",
    "removed_to_output_ratio",
    "target_query_count",
)
QUERY_SUM_KEYS = (
    "attention_mass_query_sum",
    "removed_value_norm_query_sum",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--roots", type=Path, nargs="+", default=list(DEFAULT_ROOTS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def scope_pairs(payload: dict[str, Any]) -> dict[str, set[tuple[int, int]]]:
    entries = list(payload["entries"])
    result: dict[str, set[tuple[int, int]]] = {}
    for name, definition in payload["head_scopes"].items():
        if "pairs" in definition:
            pairs = definition["pairs"]
        else:
            start = int(definition["rank_start"]) - 1
            end = int(definition["rank_end"])
            pairs = [
                [int(row["block"]), int(row["head"])] for row in entries[start:end]
            ]
        result[str(name)] = {(int(pair[0]), int(pair[1])) for pair in pairs}
    return result


def targets(case: dict[str, Any]) -> list[tuple[str, str | None]]:
    count = int(case["object_count"])
    values = [
        ("single_object", f"object_{chr(ord('A') + index)}")
        for index in range(count)
    ]
    if count > 1:
        values.append(("all_objects", None))
    if len(values) != int(case["targets_per_seed"]):
        raise RuntimeError(f"target-count mismatch in Stage-4 spec: {case}")
    return values


def cell_key(
    case: str,
    seed: int,
    target_scope: str,
    region: str | None,
    flow: str,
    time_mode: str,
    head_scope: str,
) -> tuple[Any, ...]:
    return (case, seed, target_scope, region, flow, time_mode, head_scope)


def expected_cells(spec: dict[str, Any]) -> dict[tuple[Any, ...], str]:
    cells: dict[tuple[Any, ...], str] = {}
    cases = spec["stage4a_pilot"]["cases"]
    seeds = [int(value) for value in spec["stage4a_pilot"]["seeds"]]
    for case_row in cases:
        case = str(case_row["case"])
        for seed in seeds:
            for target_scope, region in targets(case_row):
                for scope in FULL_SCOPES:
                    for flow in FLOW_MODES:
                        cells[
                            cell_key(
                                case,
                                seed,
                                target_scope,
                                region,
                                flow,
                                "all_time",
                                scope,
                            )
                        ] = "all_time_reference"
                        for direction in DIRECTIONS:
                            cells[
                                cell_key(
                                    case,
                                    seed,
                                    target_scope,
                                    region,
                                    flow,
                                    direction,
                                    scope,
                                )
                            ] = "directional_primary"

    sentinel = spec["head_scopes"]["all720_sentinel"]
    sentinel_case = next(
        row for row in cases if row["case"] == sentinel["case"]
    )
    for target_scope, region in targets(sentinel_case):
        for flow in sentinel["flows"]:
            for direction in sentinel["time_modes"]:
                cells[
                    cell_key(
                        str(sentinel["case"]),
                        int(sentinel["seed"]),
                        target_scope,
                        region,
                        str(flow),
                        str(direction),
                        "all720",
                    )
                ] = "all720_sentinel"
    return cells


def manifest_flow_time(mask_mode: str) -> tuple[str | None, str | None]:
    flow = next(
        (flow for flow, prefix in FLOW_MODES.items() if mask_mode.startswith(prefix)),
        None,
    )
    if mask_mode.endswith("_only"):
        time_mode = "all_time"
    else:
        time_mode = next(
            (direction for direction in DIRECTIONS if mask_mode.endswith(direction)),
            None,
        )
    return flow, time_mode


def dose_audit(path: Path, selected_head_count: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "base_complete": False,
        "query_sums_present": False,
        "derived_query_sums_repairable": False,
        "finite_events": 0,
        "expected_finite_events": selected_head_count * 80,
        "reason": None,
    }
    if not path.is_file():
        result["reason"] = "missing_dose_metrics"
        return result
    try:
        with np.load(path) as arrays:
            missing = [key for key in BASE_DOSE_KEYS if key not in arrays]
            if missing:
                result["reason"] = f"missing_keys:{','.join(missing)}"
                return result
            expected_shape = (40, 2, 30, 24)
            if any(tuple(arrays[key].shape) != expected_shape for key in BASE_DOSE_KEYS):
                result["reason"] = "invalid_dose_shape"
                return result
            finite_masks = [
                np.isfinite(arrays[key])
                for key in BASE_DOSE_KEYS
                if key != "target_query_count"
            ]
            common = np.logical_and.reduce(finite_masks)
            finite_events = int(common.sum())
            positive_query_events = int((arrays["target_query_count"] > 0).sum())
            result["finite_events"] = finite_events
            if (
                finite_events != result["expected_finite_events"]
                or positive_query_events != result["expected_finite_events"]
            ):
                result["reason"] = (
                    f"coverage:{finite_events}/{positive_query_events}/"
                    f"{result['expected_finite_events']}"
                )
                return result
            result["base_complete"] = True
            result["query_sums_present"] = all(key in arrays for key in QUERY_SUM_KEYS)
            result["derived_query_sums_repairable"] = not result["query_sums_present"]
    except Exception as exc:  # inventory must record malformed legacy files
        result["reason"] = f"dose_read_error:{type(exc).__name__}:{exc}"
        return result
    return result


def main() -> None:
    args = parse_args()
    spec = read_json(args.spec)
    head_path = Path(spec["head_scopes"]["definition_file"])
    head_payload = read_json(head_path)
    expected_pairs = scope_pairs(head_payload)
    expected = expected_cells(spec)

    candidates: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    parse_errors: list[dict[str, str]] = []
    for root in args.roots:
        if not root.is_dir():
            continue
        for manifest_path in sorted(root.rglob("manifest.json")):
            try:
                manifest = read_json(manifest_path)
                flow, time_mode = manifest_flow_time(str(manifest.get("mask_mode", "")))
                key = cell_key(
                    str(manifest.get("case", "")),
                    int(manifest.get("seed", -1)),
                    str(manifest.get("target_scope", "")),
                    manifest.get("region"),
                    str(flow),
                    str(time_mode),
                    str(manifest.get("head_scope", "")),
                )
                if key not in expected:
                    continue
                selected = {
                    (int(row["block"]), int(row["head"]))
                    for row in manifest.get("selected_entries", [])
                }
                scope = str(manifest.get("head_scope", ""))
                head_match = selected == expected_pairs.get(scope, set())
                audit = manifest.get("audit") or {}
                counts = audit.get("model_call_counts") or {}
                full_40x2 = len(counts) == 40 and all(
                    int(value) == 2 for value in counts.values()
                )
                video = manifest_path.parent / "generated.mp4"
                complete = manifest_path.parent / "complete.json"
                complete_video = (
                    video.is_file()
                    and video.stat().st_size > 0
                    and complete.is_file()
                )
                dose = dose_audit(
                    manifest_path.parent / "dose_metrics.npz", len(selected)
                )
                provenance = manifest.get("implementation_provenance") or {}
                provenance_files = provenance.get("files_sha256") or {}
                provenance_valid = (
                    provenance.get("combined_sha256") == manifest.get("code_hash")
                    and len(provenance_files) >= 2
                    and all(bool(value) for value in provenance_files.values())
                )
                current_directional = (
                    time_mode in DIRECTIONS
                    and manifest.get("protocol") == CURRENT_DIRECTIONAL_PROTOCOL
                    and provenance_valid
                )
                base_valid = complete_video and head_match and full_40x2
                dose_valid = bool(dose["base_complete"])
                if expected[key] == "all_time_reference":
                    primary_reusable = base_valid and dose_valid
                else:
                    primary_reusable = base_valid and dose_valid and current_directional
                candidates[key].append(
                    {
                        "manifest_path": str(manifest_path),
                        "root": str(root),
                        "protocol": manifest.get("protocol"),
                        "code_hash": manifest.get("code_hash"),
                        "implementation_provenance_valid": provenance_valid,
                        "complete_video": complete_video,
                        "head_scope_match": head_match,
                        "full_40x2": full_40x2,
                        "dose": dose,
                        "current_directional_protocol": current_directional,
                        "primary_reusable": primary_reusable,
                    }
                )
            except Exception as exc:
                parse_errors.append(
                    {"path": str(manifest_path), "error": f"{type(exc).__name__}: {exc}"}
                )

    records = []
    for key, category in sorted(expected.items(), key=lambda item: tuple(map(str, item[0]))):
        options = candidates.get(key, [])
        reusable = [row for row in options if row["primary_reusable"]]
        visual_only = [
            row
            for row in options
            if row["complete_video"] and not row["primary_reusable"]
        ]
        records.append(
            {
                "key": list(key),
                "category": category,
                "primary_reusable": bool(reusable),
                "candidate_count": len(options),
                "visual_only_candidate_count": len(visual_only),
                "best_candidate": (reusable or visual_only or options or [None])[0],
            }
        )

    category_counts = {}
    for category in sorted(set(expected.values())):
        subset = [row for row in records if row["category"] == category]
        category_counts[category] = {
            "expected": len(subset),
            "primary_reusable": sum(row["primary_reusable"] for row in subset),
            "must_generate_or_rerun": sum(not row["primary_reusable"] for row in subset),
            "visual_only_existing": sum(
                row["visual_only_candidate_count"] > 0 and not row["primary_reusable"]
                for row in subset
            ),
        }
    summary = {
        "expected_cells": len(records),
        "primary_reusable": sum(row["primary_reusable"] for row in records),
        "must_generate_or_rerun": sum(not row["primary_reusable"] for row in records),
        "categories": category_counts,
        "parse_error_count": len(parse_errors),
        "cells_with_multiple_candidates": sum(row["candidate_count"] > 1 for row in records),
        "cells_with_multiple_primary_reusable_candidates": sum(
            sum(option["primary_reusable"] for option in candidates.get(tuple(row["key"]), []))
            > 1
            for row in records
        ),
        "legacy_visual_only_cells": sum(
            row["visual_only_candidate_count"] > 0 and not row["primary_reusable"]
            for row in records
        ),
    }
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec": str(args.spec),
        "spec_sha256": sha256_file(args.spec),
        "head_scopes": str(head_path),
        "head_scopes_sha256": sha256_file(head_path),
        "roots": [str(root) for root in args.roots],
        "summary": summary,
        "parse_errors": parse_errors,
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "inventory.json", payload)

    lines = [
        "# Stage 4 Preflight Inventory",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "| Category | Expected | Reusable for primary | Must generate/rerun | Existing visual-only |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "directional_primary": "Top/Bottom/Random directional",
        "all_time_reference": "All-time reference",
        "all720_sentinel": "All720 sentinel directional",
    }
    for category, values in category_counts.items():
        lines.append(
            f"| {labels.get(category, category)} | {values['expected']} | "
            f"{values['primary_reusable']} | {values['must_generate_or_rerun']} | "
            f"{values['visual_only_existing']} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "- A directional cell is primary-reusable only when video/complete/head/40x2 checks pass, dose coverage is complete, and the manifest uses the v2 directional-dose protocol with a joint fingerprint of the temporal runner and inherited base ablator.",
        "- Legacy directional videos without valid dose remain visual-only and must not enter Stage 4 mechanism statistics.",
        "- Older All-time Stage 3 dose files may omit the two query-sum arrays; those sums are algebraically derivable from the stored mean and target-query count and do not require regeneration.",
        f"- Parse errors: `{len(parse_errors)}`; expected cells with multiple candidate paths: `{summary['cells_with_multiple_candidates']}`.",
        f"- Cells with more than one primary-reusable candidate: `{summary['cells_with_multiple_primary_reusable_candidates']}`; any nonzero value is a launch blocker.",
        "",
        "See `inventory.json` for every exact required cell and selected candidate path.",
    ]
    (args.output_dir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
