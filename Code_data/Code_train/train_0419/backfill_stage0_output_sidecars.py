#!/usr/bin/env python3
"""Backfill stage0_V2V output sidecars with normalized paths and input_path."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import batch_eval_lora as bel
import batch_eval_vace as bev


STAGE0_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V")
OUTPUT_ROOT = STAGE0_ROOT / "output"
TOOLS_META_ROOT = STAGE0_ROOT / "tools" / "meta"
WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
VACE_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B")
LORA_8K = Path(
    "/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/"
    "checkpoints/step-008000/checkpoint.safetensors"
)
LORA_10K = Path(
    "/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/"
    "checkpoints/step-010000/checkpoint.safetensors"
)


def load_existing_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def collect_case_index() -> dict[str, tuple[int, dict[str, Any]]]:
    meta_paths: list[Path] = []
    seen: set[str] = set()
    for list_name in (
        "benchmark_meta_json_paths_full_sample300.txt",
        "common_case_meta_json_paths.txt",
    ):
        list_path = TOOLS_META_ROOT / list_name
        if not list_path.is_file():
            continue
        for meta_path in bel.load_meta_paths(list_path):
            key = str(meta_path)
            if key in seen:
                continue
            seen.add(key)
            meta_paths.append(meta_path)

    cases = bel.collect_cases(meta_paths, limit=None)
    return {
        Path(case["output_name"]).stem: (index, case)
        for index, case in enumerate(cases)
    }


def preserve_existing_fields(new_payload: dict[str, Any], existing_payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("evaluation", "metrics"):
        if key in existing_payload:
            new_payload[key] = existing_payload[key]
    if isinstance(existing_payload.get("runtime"), dict):
        new_payload["runtime"] = existing_payload["runtime"]
    return new_payload


def build_wan_args(
    *,
    model_name: str,
    lora_path: Path | None,
    context_frames: int,
    conditioning_mode: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=model_name,
        lora_path=lora_path,
        height=384,
        width=672,
        fps=8,
        num_inference_steps=50,
        cfg_scale=5.0,
        requested_output_frames=49,
        num_frames=49,
        context_frames=context_frames,
        negative_prompt="",
        conditioning_mode=conditioning_mode,
        shard_id=0,
        num_shards=1,
    )


def build_vace_args(
    *,
    model_name: str,
    mode: str,
    context_frames: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=model_name,
        vace_root=VACE_ROOT,
        seed=42,
        height=544,
        width=720,
        fps=16,
        requested_output_frames=49,
        num_frames=49,
        num_inference_steps=50,
        cfg_scale=5.0,
        negative_prompt="",
        context_frames=context_frames,
        mode=mode,
    )


def backfill_leaf_dir(
    *,
    output_dir: Path,
    case_index: dict[str, tuple[int, dict[str, Any]]],
    payload_builder: Any,
) -> tuple[int, list[str]]:
    updated = 0
    missing: list[str] = []
    for output_path in sorted(output_dir.glob("*.mp4")):
        stem = output_path.stem
        matched = case_index.get(stem)
        if matched is None:
            missing.append(stem)
            continue
        index, case = matched
        payload = payload_builder(index=index, case=case, output_path=output_path)
        sidecar_path = output_path.with_suffix(".json")
        payload = preserve_existing_fields(payload, load_existing_json(sidecar_path))
        bel.write_json(sidecar_path, payload)
        updated += 1
    return updated, missing


def main() -> None:
    case_index = collect_case_index()
    total_updated = 0
    missing_cases: list[str] = []

    wan_specs = [
        (
            OUTPUT_ROOT / "wan2_2_5B_baseline_TI2V",
            build_wan_args(
                model_name="base-ti2v-5b",
                lora_path=None,
                context_frames=16,
                conditioning_mode="context_aware",
            ),
            16,
        ),
        (
            OUTPUT_ROOT / "wan2.25B_lora_sample300_full49" / "step-008000",
            build_wan_args(
                model_name="step-008000",
                lora_path=LORA_8K,
                context_frames=16,
                conditioning_mode="context_aware",
            ),
            16,
        ),
        (
            OUTPUT_ROOT / "wan2.25B_lora_sample300_full49" / "step-010000",
            build_wan_args(
                model_name="step-010000",
                lora_path=LORA_10K,
                context_frames=16,
                conditioning_mode="context_aware",
            ),
            16,
        ),
        (
            OUTPUT_ROOT / "Wan2_2_5B_pure_TI2V",
            build_wan_args(
                model_name="wan_pure_ti2v_5b",
                lora_path=None,
                context_frames=1,
                conditioning_mode="input_image_only",
            ),
            1,
        ),
    ]

    for output_dir, args, used_context_frames in wan_specs:
        updated, missing = backfill_leaf_dir(
            output_dir=output_dir,
            case_index=case_index,
            payload_builder=lambda *, index, case, output_path, args=args, used_context_frames=used_context_frames: bel.build_case_metadata(
                args=args,
                row=case,
                index=index,
                seed=42,
                output_path=output_path,
                used_context_frames=used_context_frames,
                status="generated",
            ),
        )
        total_updated += updated
        missing_cases.extend(missing)

    vace_specs = [
        (
            OUTPUT_ROOT / "VACE_1_3B_TI2V",
            build_vace_args(
                model_name="vace_ti2v_firstframe",
                mode="ti2v_firstframe",
                context_frames=1,
            ),
            1,
        ),
        (
            OUTPUT_ROOT / "VACE_1_3B_V2V" / "context_01f",
            build_vace_args(
                model_name="vace_v2v_ctx01f",
                mode="v2v_clipref",
                context_frames=1,
            ),
            1,
        ),
        (
            OUTPUT_ROOT / "VACE_1_3B_V2V" / "context_02f",
            build_vace_args(
                model_name="vace_v2v_ctx02f",
                mode="v2v_clipref",
                context_frames=2,
            ),
            2,
        ),
        (
            OUTPUT_ROOT / "VACE_1_3B_V2V" / "context_04f",
            build_vace_args(
                model_name="vace_v2v_ctx04f",
                mode="v2v_clipref",
                context_frames=4,
            ),
            4,
        ),
        (
            OUTPUT_ROOT / "VACE_1_3B_V2V" / "context_08f",
            build_vace_args(
                model_name="vace_v2v_ctx08f",
                mode="v2v_clipref",
                context_frames=8,
            ),
            8,
        ),
    ]

    for output_dir, args, used_context_frames in vace_specs:
        updated, missing = backfill_leaf_dir(
            output_dir=output_dir,
            case_index=case_index,
            payload_builder=lambda *, index, case, output_path, args=args, used_context_frames=used_context_frames: bev.build_case_payload(
                args=args,
                case=case,
                index=index,
                output_path=output_path,
                used_context_frames=used_context_frames,
                status="generated",
            ),
        )
        total_updated += updated
        missing_cases.extend(missing)

    print(f"updated_sidecars={total_updated}")
    if missing_cases:
        print("missing_cases=" + ",".join(sorted(set(missing_cases))))


if __name__ == "__main__":
    main()
