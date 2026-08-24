#!/usr/bin/env python3
"""Prepare resumable bench.py result roots for the local P0 submissions.

The official metric runner expects one result JSON next to each candidate MP4.
The strict submissions already contain the final 120-frame videos, so this
adapter only creates small JSON sidecars and symlinks; it never copies media.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physicsiq-verified-strict-metrics"
)
INPUT_ROOT = Path(
    "/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/inputs/bpp/jsons"
)
METHODS = {
    "xssc_step2000": {
        "label": "xSSC Full-SA no-object",
        "video_root": Path(
            "/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/"
            "generated_videos_5s/"
            "full_sa_no_object_gpu67_resume_step6-step-002000-36890878a58d-bpp-run_01"
        ),
    },
    "physrvg_72f": {
        "label": "PhysRVG-72f-adapted",
        "video_root": Path(
            "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub/"
            "physicsiq-verified-standard/assets/physrvg"
        ),
    },
    "physrvg_full_sa": {
        "label": "PhysRVG Full-SA VJEPA",
        "video_root": Path(
            "/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/"
            "generated_videos_5s/physrvg-full-sa-vjepa-step000500-bpp-run_01"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def ensure_video_link(destination: Path, target: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() == target.resolve():
            return
        destination.unlink()
    elif destination.exists():
        if destination.resolve() != target.resolve():
            raise RuntimeError(f"Refusing to replace existing file: {destination}")
        return
    destination.symlink_to(target)


def load_inputs() -> dict[str, Path]:
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"BPP input JSON directory is unavailable: {INPUT_ROOT}")
    by_video_name: dict[str, Path] = {}
    for input_json in sorted(INPUT_ROOT.glob("*.json")):
        payload = json.loads(input_json.read_text(encoding="utf-8"))
        video_name = payload.get("generated_video_name")
        source_video = payload.get("source_video")
        caption = payload.get("input_caption")
        if not isinstance(video_name, str) or not video_name:
            raise ValueError(f"Missing generated_video_name: {input_json}")
        if not isinstance(source_video, str) or not Path(source_video).is_file():
            raise FileNotFoundError(f"Invalid source_video in {input_json}: {source_video}")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Missing input_caption: {input_json}")
        if video_name in by_video_name:
            raise ValueError(f"Duplicate generated video name: {video_name}")
        by_video_name[video_name] = input_json.resolve()
    if len(by_video_name) != 198:
        raise ValueError(f"Expected 198 BPP inputs, found {len(by_video_name)}")
    return by_video_name


def prepare_method(
    *,
    output_root: Path,
    method_key: str,
    spec: dict[str, Any],
    input_by_video: dict[str, Path],
) -> dict[str, Any]:
    video_root = Path(spec["video_root"]).expanduser().resolve()
    if not video_root.is_dir():
        raise FileNotFoundError(f"Video root is unavailable: {video_root}")
    result_root = output_root / "results" / method_key
    result_root.mkdir(parents=True, exist_ok=True)
    video_names = {path.name for path in video_root.glob("*.mp4")}
    expected_names = set(input_by_video)
    missing = sorted(expected_names - video_names)
    extra = sorted(video_names - expected_names)
    if missing or extra:
        raise ValueError(
            f"{method_key} video mismatch: missing={len(missing)} extra={len(extra)}"
        )

    for video_name in sorted(expected_names):
        input_json = input_by_video[video_name]
        source_video = json.loads(input_json.read_text(encoding="utf-8"))["source_video"]
        caption = json.loads(input_json.read_text(encoding="utf-8"))["input_caption"]
        source_video_path = Path(source_video).expanduser().resolve()
        candidate_source = (video_root / video_name).resolve()
        candidate_link = result_root / f"{Path(video_name).stem}.mp4"
        result_json = result_root / f"{Path(video_name).stem}.json"
        ensure_video_link(candidate_link, candidate_source)
        payload = {
            "schema_version": 1,
            "input_json": str(input_json),
            "output_video": str(candidate_link.resolve()),
            "input_caption": caption,
            "source_video": str(source_video_path),
            "strict_metric_protocol": {
                "benchmark": "Physics-IQ-Verified",
                "submission_frames": 120,
                "submission_fps": 24,
                "context_frames_removed_before_submission": 69,
                "metric_context_frames_removed": 0,
                "metric_input": "final 120-frame submission video",
            },
            # bench.py's generated-only VideoPhy2 path requires an explicit
            # override; zero means the submission is already context-free.
            "context_frames": 0,
            "effective_context_frames": 0,
        }
        if result_json.is_file():
            existing = json.loads(result_json.read_text(encoding="utf-8"))
            for key in ("videophy2", "cosmos_reason1"):
                if key in existing:
                    payload[key] = existing[key]
            for key in (
                "vbench_subject_consistency",
                "vbench_background_consistency",
                "vbench_temporal_flickering",
                "vbench_motion_smoothness",
                "vbench_dynamic_degree",
                "vbench_aesthetic_quality",
                "vbench_imaging_quality",
            ):
                if key in existing:
                    payload[key] = existing[key]
        atomic_write(result_json, payload)

    return {
        "key": method_key,
        "label": str(spec["label"]),
        "video_root": str(video_root),
        "result_root": str(result_root),
        "num_cases": len(expected_names),
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    input_by_video = load_inputs()
    methods = [
        prepare_method(
            output_root=output_root,
            method_key=method_key,
            spec=spec,
            input_by_video=input_by_video,
        )
        for method_key, spec in METHODS.items()
    ]
    allowlist = output_root / "strict_input_allowlist.txt"
    allowlist.parent.mkdir(parents=True, exist_ok=True)
    allowlist.write_text(
        "\n".join(str(path) for path in sorted(input_by_video.values())) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_json_root": str(INPUT_ROOT),
        "allowlist": str(allowlist),
        "expected_cases": 198,
        "methods": methods,
        "metrics": [
            "vbench_subject_consistency",
            "vbench_background_consistency",
            "vbench_temporal_flickering",
            "vbench_motion_smoothness",
            "vbench_dynamic_degree",
            "vbench_aesthetic_quality",
            "vbench_imaging_quality",
            "videophy2",
            "cosmos_reason1",
        ],
    }
    atomic_write(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
