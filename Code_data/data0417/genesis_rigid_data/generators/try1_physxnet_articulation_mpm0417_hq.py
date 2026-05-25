#!/usr/bin/env python3
# 用途：高质量 rigid 数据生成入口；复用原 try1 后端，但把更稳的相机、采样、裁尾和 QA 默认值收口到单独脚本里，避免继续改动原脚本。
"""High-quality wrapper for Genesis rigid sample generation.

This script intentionally does not modify the original
`try1_physxnet_articulation_mpm0417.py`. It imports that generator as a backend
and applies a stricter set of defaults geared toward cleaner rigid videos:

- safer camera presets
- denser temporal sampling without fake slow-motion
- stronger first-frame visibility / scale defaults
- trailing-still trimming
- more retry budget for randomized motion cases

Typical usage:

python3 try1_physxnet_articulation_mpm0417_hq.py \
  --object_id 10032 \
  --output_root /data/gaoya/AAA_test_video/Dataset_physV/0417data/version0515zoom_genesis_rigid \
  --run_genesis \
  --generate_all_count_motion_cases \
  --rigid_count_filter 1 2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


THIS_DIR = Path(__file__).resolve().parent
BACKEND_PATH = THIS_DIR / "try1_physxnet_articulation_mpm0417.py"


def _load_backend():
    spec = importlib.util.spec_from_file_location("try1_physxnet_articulation_mpm0417_backend", BACKEND_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load backend generator from {BACKEND_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BACKEND = _load_backend()
BACKEND_DEFAULT_ARGS = BACKEND.build_argparser().parse_args(["--output_root", "/tmp/codex_hq_dummy"])


CAMERA_PRESETS: Dict[str, Dict[str, Any]] = {
    "single_clean": {
        "camera_pos_override": [0.0, -3.05, 1.72],
        "camera_lookat_override": [0.0, 0.0, 0.62],
        "camera_up_override": [0.0, 0.0, 1.0],
        "camera_fov_override": 43.0,
        "camera_tag": "single_clean",
    },
    "pair_wide": {
        "camera_pos_override": [0.0, -3.35, 1.82],
        "camera_lookat_override": [0.0, 0.0, 0.66],
        "camera_up_override": [0.0, 0.0, 1.0],
        "camera_fov_override": 46.0,
        "camera_tag": "pair_wide",
    },
    "pair_collision": {
        "camera_pos_override": [0.0, -3.15, 1.78],
        "camera_lookat_override": [0.0, 0.0, 0.64],
        "camera_up_override": [0.0, 0.0, 1.0],
        "camera_fov_override": 44.0,
        "camera_tag": "pair_collision",
    },
}


QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "dt": 0.003,
        "substeps": 40,
        "fps": 24,
        "duration_sec": 2.25,
        "sampling_fps_mult": 1.75,
        "trim_trailing_still": 1,
        "trim_visible_uv_disp_thresh_px": 1.0,
        "trim_seg_change_px_thresh": 32,
        "trim_post_active_sec": 0.25,
        "trim_min_frames": 18,
        "trim_min_duration_sec": 1.40,
        "trim_min_removed_frames": 6,
        "video_slowmo_prob": 0.0,
        "motion_case_max_retries": 10,
        "min_projected_bbox_area_px": 5000.0,
        "max_auto_scale_up_mult": 2.8,
        "max_projected_bbox_fill_ratio": 0.76,
        "camera_distance_mult": 1.0,
    },
    "dense": {
        "dt": 0.0025,
        "substeps": 48,
        "fps": 24,
        "duration_sec": 2.20,
        "sampling_fps_mult": 2.25,
        "trim_trailing_still": 1,
        "trim_visible_uv_disp_thresh_px": 0.85,
        "trim_seg_change_px_thresh": 24,
        "trim_post_active_sec": 0.22,
        "trim_min_frames": 20,
        "trim_min_duration_sec": 1.50,
        "trim_min_removed_frames": 6,
        "video_slowmo_prob": 0.0,
        "motion_case_max_retries": 12,
        "min_projected_bbox_area_px": 5600.0,
        "max_auto_scale_up_mult": 3.0,
        "max_projected_bbox_fill_ratio": 0.74,
        "camera_distance_mult": 1.0,
    },
    "cautious": {
        "dt": 0.003,
        "substeps": 40,
        "fps": 24,
        "duration_sec": 2.00,
        "sampling_fps_mult": 1.50,
        "trim_trailing_still": 1,
        "trim_visible_uv_disp_thresh_px": 1.1,
        "trim_seg_change_px_thresh": 40,
        "trim_post_active_sec": 0.20,
        "trim_min_frames": 16,
        "trim_min_duration_sec": 1.35,
        "trim_min_removed_frames": 6,
        "video_slowmo_prob": 0.0,
        "motion_case_max_retries": 8,
        "min_projected_bbox_area_px": 4600.0,
        "max_auto_scale_up_mult": 2.5,
        "max_projected_bbox_fill_ratio": 0.78,
        "camera_distance_mult": 1.02,
    },
}


def _append_hq_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--hq_profile",
        type=str,
        default="balanced",
        choices=sorted(QUALITY_PRESETS.keys()),
        help="High-quality default profile layered on top of the backend parser.",
    )
    parser.add_argument(
        "--hq_camera_preset",
        type=str,
        default="auto",
        choices=["auto", *sorted(CAMERA_PRESETS.keys())],
        help="Camera preset. auto chooses a preset from target object count and striker usage.",
    )
    parser.add_argument(
        "--hq_manifest_name",
        type=str,
        default="hq_run_manifest.json",
        help="Extra manifest written beside each object summary for reproducibility.",
    )
    parser.add_argument(
        "--hq_disable_manifest",
        action="store_true",
        help="Do not emit the extra HQ manifest file.",
    )
    return parser


def build_argparser() -> argparse.ArgumentParser:
    parser = BACKEND.build_argparser()
    return _append_hq_args(parser)


def _infer_target_count(args: argparse.Namespace) -> Optional[int]:
    count = getattr(args, "rigid_target_object_count", None)
    if count is not None:
        try:
            return int(count)
        except Exception:
            return None
    count_filter = getattr(args, "rigid_count_filter", None)
    if isinstance(count_filter, list) and len(count_filter) == 1:
        try:
            return int(count_filter[0])
        except Exception:
            return None
    return None


def _select_camera_preset(args: argparse.Namespace) -> Optional[str]:
    requested = str(getattr(args, "hq_camera_preset", "auto") or "auto").strip().lower()
    if requested != "auto":
        return requested

    target_count = _infer_target_count(args)
    disable_striker = bool(getattr(args, "disable_striker", False))
    if target_count is not None and target_count <= 1:
        return "single_clean"
    if disable_striker:
        return "pair_wide"
    return "pair_collision"


def _apply_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    profile_name = str(getattr(args, "hq_profile", "balanced"))
    profile_cfg = dict(QUALITY_PRESETS[profile_name])
    for key, value in profile_cfg.items():
        current = getattr(args, key, None)
        default = getattr(BACKEND_DEFAULT_ARGS, key, None)
        if current == default or current is None:
            setattr(args, key, value)

    camera_name = _select_camera_preset(args)
    camera_cfg: Dict[str, Any] = {}
    if camera_name:
        camera_cfg = dict(CAMERA_PRESETS[camera_name])
        for key, value in camera_cfg.items():
            current = getattr(args, key, None)
            default = getattr(BACKEND_DEFAULT_ARGS, key, None)
            if current == default or current is None:
                setattr(args, key, value)

    if getattr(args, "duration_sec", None) is None:
        args.duration_sec = float(profile_cfg["duration_sec"])

    if getattr(args, "video_slowmo_prob", None) == getattr(BACKEND_DEFAULT_ARGS, "video_slowmo_prob", None):
        args.video_slowmo_prob = 0.0
    args.prefer_existing_runtime_meshes = True

    return {
        "hq_profile": profile_name,
        "hq_camera_preset": camera_name,
        "quality_overrides": profile_cfg,
        "camera_overrides": camera_cfg,
    }


def _write_hq_manifest(prepared_summary: Dict[str, Any], args: argparse.Namespace, applied: Dict[str, Any]) -> None:
    if bool(getattr(args, "hq_disable_manifest", False)):
        return
    output_dir = prepared_summary.get("output_dir")
    if not output_dir:
        return
    manifest_path = Path(output_dir) / str(getattr(args, "hq_manifest_name", "hq_run_manifest.json"))
    payload = {
        "script": str(Path(__file__).resolve()),
        "backend_script": str(BACKEND_PATH.resolve()),
        "object_id": prepared_summary.get("object_id"),
        "simulator_mode": str(getattr(args, "simulator_mode", "rigid")),
        "hq_profile": applied.get("hq_profile"),
        "hq_camera_preset": applied.get("hq_camera_preset"),
        "quality_overrides": applied.get("quality_overrides", {}),
        "camera_overrides": applied.get("camera_overrides", {}),
        "run_args_subset": {
            "output_root": str(getattr(args, "output_root", "")),
            "rigid_target_object_count": getattr(args, "rigid_target_object_count", None),
            "rigid_count_filter": getattr(args, "rigid_count_filter", None),
            "case_index_filter": getattr(args, "case_index_filter", None),
            "generate_all_count_motion_cases": bool(getattr(args, "generate_all_count_motion_cases", False)),
            "disable_striker": bool(getattr(args, "disable_striker", False)),
            "object_scale_mult": float(getattr(args, "object_scale_mult", 1.0)),
            "min_projected_bbox_area_px": float(getattr(args, "min_projected_bbox_area_px", 0.0)),
            "max_auto_scale_up_mult": float(getattr(args, "max_auto_scale_up_mult", 0.0)),
            "duration_sec": float(getattr(args, "duration_sec", 0.0)),
            "sampling_fps_mult": float(getattr(args, "sampling_fps_mult", 1.0)),
            "dt": float(getattr(args, "dt", 0.0)),
            "substeps": int(getattr(args, "substeps", 0)),
            "fps": int(getattr(args, "fps", 0)),
        },
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_object_ids(args: argparse.Namespace) -> List[str]:
    if getattr(args, "json_override", None) and not getattr(args, "object_id", None):
        args.object_id = Path(str(args.json_override)).stem
        print(f"INFO inferred object_id={args.object_id} from --json_override")

    if getattr(args, "object_id", None):
        return [str(args.object_id)]

    return BACKEND._sample_random_object_ids(
        physx_root=Path(args.physx_root),
        version=str(args.version),
        num_objects=int(args.num_random_objects),
        seed=int(args.random_object_seed),
    )


def run_single_object(args: argparse.Namespace, object_id: str) -> Dict[str, Any]:
    applied = _apply_overrides(args)
    summary = BACKEND._run_single_object(args=args, object_id=str(object_id))
    _write_hq_manifest(summary, args, applied)
    return summary


def main() -> None:
    args = build_argparser().parse_args()
    with BACKEND._quiet_terminal_output(enabled=True):
        object_ids = _resolve_object_ids(args)
        print(f"INFO hq_profile={args.hq_profile} hq_camera_preset={_select_camera_preset(args)}")
        for idx, object_id in enumerate(object_ids, start=1):
            print(f"INFO object {idx}/{len(object_ids)} object_id={object_id}")
            run_single_object(args=args, object_id=str(object_id))


if __name__ == "__main__":
    main()
