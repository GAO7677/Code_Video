#!/usr/bin/env python3
"""Add the audited evidence-caption pipeline to the existing PhysV comparison page."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any


OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/physv_qwen3vl")
DEFAULT_COMPARISON = OUTPUT_ROOT / "0613_phyco_frame_compare_chinese_prompt_fps15_maxpixels6500000.jsonl"
DEFAULT_EVIDENCE_RESULTS = (
    OUTPUT_ROOT / "0613_phyco_evidence_pipeline_final_compact_fps15_maxpixels6500000.jsonl"
)
DEFAULT_OUTPUT = OUTPUT_ROOT / "0613_phyco_frame_compare_evidence_pipeline_final.jsonl"
DEFAULT_BASE_VARIANT = "full_fps15_6_5m"
DEFAULT_EVIDENCE_VARIANT = "full_fps15_6_5m_evidence"

VIDEO_AUDIT_FIELDS = (
    "source_fps",
    "source_frame_indices",
    "source_total_frames",
    "source_video_backend",
    "stage_one_shape",
    "stage_one_tensor_sha256",
    "visual_replay_shape",
    "vllm_grid_thw",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--evidence-results", type=Path, default=DEFAULT_EVIDENCE_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-variant", default=DEFAULT_BASE_VARIANT)
    parser.add_argument("--evidence-variant", default=DEFAULT_EVIDENCE_VARIANT)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomically(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def index_by_case(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{label} has an invalid case_id")
        if case_id in indexed:
            raise ValueError(f"{label} has duplicate case_id: {case_id}")
        indexed[case_id] = row
    return indexed


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"Mismatch for {label}: {actual!r} != {expected!r}")


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected object for {label}")
    return value


def build_evidence_variant(
    base: dict[str, Any], evidence: dict[str, Any], base_key: str
) -> dict[str, Any]:
    if evidence.get("status") != "ok":
        raise ValueError(f"Evidence result failed for {evidence.get('case_id')}")
    require_equal(base.get("video"), evidence.get("video"), f"{evidence['case_id']} video")
    require_equal(
        base.get("video_params"), evidence.get("video_params"), f"{evidence['case_id']} video parameters"
    )
    base_audit = require_mapping(base.get("vlm_input_audit"), "base VLM audit")
    final_input = require_mapping(evidence.get("final_input"), "evidence final input")
    evidence_input_audit = require_mapping(final_input.get("input_audit"), "evidence input audit")
    evidence_video_audit = require_mapping(evidence_input_audit.get("video"), "evidence video audit")
    for field in VIDEO_AUDIT_FIELDS:
        require_equal(
            base_audit.get(field),
            evidence_video_audit.get(field),
            f"{evidence['case_id']} full-video audit {field}",
        )
    storyboard = require_mapping(evidence.get("event_window"), "event window").get("storyboard", {})
    storyboard = require_mapping(storyboard, "event storyboard")
    storyboard_path = Path(str(storyboard.get("path", "")))
    if not storyboard_path.is_file():
        raise FileNotFoundError(storyboard_path)
    if not base.get("vlm_input_video"):
        raise ValueError(f"Base replay missing for {evidence['case_id']} / {base_key}")

    return {
        "label": "完整视频 + 6帧事件证据 / FPS 15 / 6.5M px",
        "frame_count": None,
        "video": evidence["video"],
        "video_params": copy.deepcopy(evidence["video_params"]),
        "video_info": copy.deepcopy(base.get("video_info")),
        "response_raw": evidence.get("response_raw"),
        "response_before_compaction": evidence.get("response_before_compaction"),
        "response_final": evidence.get("response_final"),
        "status": "ok",
        "elapsed_seconds": evidence.get("elapsed_seconds"),
        "thinking_disabled": True,
        "source": "full_video_evidence_pipeline",
        "vlm_input_video": base["vlm_input_video"],
        "vlm_input_video_fps": base.get("vlm_input_video_fps"),
        "vlm_input_audit": copy.deepcopy(base_audit),
        "evidence_input_audit": copy.deepcopy(evidence_input_audit),
        "evidence_storyboard": str(storyboard_path),
        "evidence_states": copy.deepcopy(evidence.get("event_probe", {}).get("states")),
        "evidence_window": copy.deepcopy(evidence.get("event_window")),
        "evidence_params": copy.deepcopy(evidence.get("evidence_params")),
        "caption_compaction": copy.deepcopy(evidence.get("caption_compaction")),
        "caption_prompt": evidence.get("caption_prompt"),
        "video_replay_equivalence": {
            "verified": True,
            "base_variant": base_key,
            "matched_fields": list(VIDEO_AUDIT_FIELDS),
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.resolve() in {args.comparison.resolve(), args.evidence_results.resolve()}:
        raise ValueError("Output must not overwrite an input")
    comparison_rows = load_jsonl(args.comparison)
    evidence_rows = load_jsonl(args.evidence_results)
    comparison_by_case = index_by_case(comparison_rows, "comparison")
    evidence_by_case = index_by_case(evidence_rows, "evidence results")
    require_equal(set(evidence_by_case), set(comparison_by_case), "evidence case IDs")

    merged_rows = copy.deepcopy(comparison_rows)
    for row in merged_rows:
        case_id = row["case_id"]
        variants = require_mapping(row.get("variants"), f"{case_id} variants")
        base = require_mapping(variants.get(args.base_variant), f"{case_id} base variant")
        variants[args.evidence_variant] = build_evidence_variant(
            base, evidence_by_case[case_id], args.base_variant
        )
        row["evidence_pipeline"] = {
            "variant": args.evidence_variant,
            "base_video_variant": args.base_variant,
            "description": "Full-video VLM replay plus six pixel-selected local evidence frames.",
        }
    write_jsonl_atomically(args.output, merged_rows)
    print(f"cases={len(merged_rows)}")
    print(f"base_variant={args.base_variant}")
    print(f"evidence_variant={args.evidence_variant}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
