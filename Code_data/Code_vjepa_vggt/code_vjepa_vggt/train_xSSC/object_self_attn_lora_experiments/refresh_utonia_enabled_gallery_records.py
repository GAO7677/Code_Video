#!/usr/bin/env python3
"""Refresh Scene Enabled video references in the existing test5/PhysicIQ pages."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re


HUB_ROOT = Path("/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub")
RAW_ROOT = Path("/data/gaoya/agent-data/outputs/physrvg_full_sa_vjepa_utonia_scene_enabled_eval")
METHOD_KEY = "full_sa_physrvg_vjepa_utonia_scene_hardmask_v1_enabled"
PAGE_SPECS = {
    "test5": {"steps": 8, "media_root": "gallery"},
    "physiciq": {"steps": 40, "media_root": "physiciq-gallery"},
}

METRIC_PATHS = {
    "videophy2_pc_raw": ("videophy2", "pc_raw_score"),
    "cosmos_reason1": ("cosmos_reason1", "score"),
    "physics_iq_with_context": ("physics_iq_with_context", "score"),
    "physics_iq_without_context": ("physics_iq_without_context", "score"),
    "videophy2": ("videophy2", "score"),
    "videophy2_sa": ("videophy2", "sa_score"),
    "videophy2_pc": ("videophy2", "pc_score"),
    "videophy2_joint_rate": ("videophy2", "joint_rate"),
    "pmf_with_context": ("pmf_with_context", "score"),
    "pmf_without_context": ("pmf_without_context", "score"),
    "wmreward": ("wmreward", "surprise"),
    "vbench_subject_consistency": ("vbench_subject_consistency", "score"),
    "vbench_background_consistency": ("vbench_background_consistency", "score"),
    "vbench_temporal_flickering": ("vbench_temporal_flickering", "score"),
    "vbench_motion_smoothness": ("vbench_motion_smoothness", "score"),
    "vbench_dynamic_degree": ("vbench_dynamic_degree", "score"),
    "vbench_aesthetic_quality": ("vbench_aesthetic_quality", "score"),
    "vbench_imaging_quality": ("vbench_imaging_quality", "score"),
}


def load_case_metrics(result_path: Path) -> dict[str, float]:
    if not result_path.is_file():
        return {}
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    values: dict[str, float] = {}
    for metric, path in METRIC_PATHS.items():
        value: object = payload
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            values[metric] = number
    return values


def refresh_page(page_name: str, inference_steps: int, media_root: str) -> dict[str, object]:
    page_path = HUB_ROOT / page_name / "index.html"
    text = page_path.read_text(encoding="utf-8")
    match = re.search(r"const D=(\{.*?\});\n    const caseSelect", text, re.DOTALL)
    if match is None:
        raise RuntimeError(f"embedded dashboard data not found: {page_path}")
    data = json.loads(match.group(1))
    case_stems = {str(case["stem"]) for case in data["cases"]}
    counts: dict[str, int] = {}

    for record in data["records"]:
        if record.get("method_key") != METHOD_KEY:
            continue
        step = int(record["step"])
        output_dir = RAW_ROOT / page_name / (
            f"full_sa_physrvg_vjepa_utonia_scene_enabled_step-{step:06d}_"
            f"steps{inference_steps}_512x896_ctx08_49f"
        )
        videos = {
            video.stem: (
                f"../{media_root}/media/{METHOD_KEY}/step-{step:06d}/{video.name}"
            )
            for video in sorted(output_dir.glob("*.mp4"))
            if video.stem in case_stems
        }
        record["videos"] = videos
        record["metrics"] = {
            stem: metrics
            for stem in sorted(case_stems)
            if (metrics := load_case_metrics(output_dir / f"{stem}.json"))
        }
        counts[str(step)] = len(videos)

    replacement = (
        "const D="
        + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        + ";\n    const caseSelect"
    )
    page_path.write_text(
        text[: match.start()] + replacement + text[match.end() :],
        encoding="utf-8",
    )
    return {"page": str(page_path), "video_counts": counts}


def main() -> None:
    results = [
        refresh_page(name, int(spec["steps"]), str(spec["media_root"]))
        for name, spec in PAGE_SPECS.items()
    ]
    print(json.dumps({"refreshed": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
