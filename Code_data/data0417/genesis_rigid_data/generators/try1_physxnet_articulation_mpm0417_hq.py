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

import numpy as np


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
        "camera_pos_override": [0.0, -2.45, 1.35],
        "camera_lookat_override": [0.0, 0.0, 0.58],
        "camera_up_override": [0.0, 0.0, 1.0],
        "camera_fov_override": 38.0,
        "camera_tag": "single_clean",
    },
    "pair_wide": {
        "camera_pos_override": [0.0, -2.60, 1.42],
        "camera_lookat_override": [0.0, 0.0, 0.60],
        "camera_up_override": [0.0, 0.0, 1.0],
        "camera_fov_override": 39.5,
        "camera_tag": "pair_wide",
    },
    "pair_collision": {
        "camera_pos_override": [0.0, -2.50, 1.38],
        "camera_lookat_override": [0.0, 0.0, 0.59],
        "camera_up_override": [0.0, 0.0, 1.0],
        "camera_fov_override": 38.8,
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
        "min_projected_bbox_area_px": 7600.0,
        "max_auto_scale_up_mult": 3.4,
        "max_projected_bbox_fill_ratio": 0.76,
        "camera_distance_mult": 0.88,
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
        "min_projected_bbox_area_px": 8200.0,
        "max_auto_scale_up_mult": 3.6,
        "max_projected_bbox_fill_ratio": 0.74,
        "camera_distance_mult": 0.88,
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
        "min_projected_bbox_area_px": 7200.0,
        "max_auto_scale_up_mult": 3.2,
        "max_projected_bbox_fill_ratio": 0.78,
        "camera_distance_mult": 0.90,
    },
}

LARGE_OBJECT_SINGLE_OBJECT_CASE_IDS = {0, 1, 2, 3, 5, 6, 7, 100, 101, 102, 900, 901}
LARGE_OBJECT_DYNAMIC_CASE_IDS = {900, 901, 210, 211, 220, 221, 230, 231}


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
    parser.add_argument(
        "--hq_disable_dynamic_for_large_objects",
        type=int,
        choices=[0, 1],
        default=1,
        help="When enabled, large main objects will not generate free-motion cases such as random parabola / high-drop / multi free-motion.",
    )
    parser.add_argument(
        "--hq_large_object_volume_threshold_m3",
        type=float,
        default=None,
        help="Optional override for the large-object threshold. Defaults to --physxnet_volume_threshold_m3.",
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
    return "single_clean"


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
            setattr(args, key, value)

    if getattr(args, "duration_sec", None) is None:
        args.duration_sec = float(profile_cfg["duration_sec"])

    if getattr(args, "video_slowmo_prob", None) == getattr(BACKEND_DEFAULT_ARGS, "video_slowmo_prob", None):
        args.video_slowmo_prob = 0.0
    args.prefer_existing_runtime_meshes = True
    args.camera_distance_mult = float(profile_cfg.get("camera_distance_mult", 0.88))
    args.camera_fov_override = float(CAMERA_PRESETS["single_clean"]["camera_fov_override"])
    args.camera_tag = "single_clean"

    return {
        "hq_profile": profile_name,
        "hq_camera_preset": camera_name,
        "quality_overrides": profile_cfg,
        "camera_overrides": camera_cfg,
    }


def _hq_case_filter(args: argparse.Namespace, case_cfgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for cfg in case_cfgs:
        label = f"{str(cfg.get('case_name', '') or '')}::{str(cfg.get('scene_label', '') or '')}".lower()
        if "multi3_" in label:
            continue
        filtered.append(cfg)
    return filtered


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


def _prepared_bbox_volume_m3(prepared: Any) -> float:
    bbox_min = np.asarray(getattr(prepared, "object_bbox_min", [0.0, 0.0, 0.0]), dtype=np.float64)
    bbox_max = np.asarray(getattr(prepared, "object_bbox_max", [0.0, 0.0, 0.0]), dtype=np.float64)
    extent = np.maximum(bbox_max - bbox_min, 1e-6)
    return float(np.prod(extent))


def _large_object_threshold_m3(args: argparse.Namespace) -> float:
    override = getattr(args, "hq_large_object_volume_threshold_m3", None)
    if override is not None:
        return float(override)
    return float(getattr(args, "physxnet_volume_threshold_m3", 0.20) or 0.20)


def _filter_case_configs_for_large_object(
    case_cfgs: List[Dict[str, Any]],
    *,
    prepared: Any,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    if not bool(int(getattr(args, "hq_disable_dynamic_for_large_objects", 1) or 0)):
        return case_cfgs

    threshold_m3 = _large_object_threshold_m3(args)
    if threshold_m3 <= 0.0:
        return case_cfgs

    bbox_volume_m3 = _prepared_bbox_volume_m3(prepared)
    if bbox_volume_m3 < threshold_m3:
        return case_cfgs

    filtered: List[Dict[str, Any]] = []
    removed_names: List[str] = []
    for cfg in case_cfgs:
        case_idx = int(cfg.get("case_index", -1))
        case_name = str(cfg.get("case_name", "") or "")
        scene_label = str(cfg.get("scene_label", "") or "")
        use_entry_motion = bool(cfg.get("use_entry_motion", False))
        object_fixed = bool(cfg.get("object_fixed", False))
        placed_pos_offset = np.asarray(cfg.get("placed_pos_offset", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
        z_offset = float(placed_pos_offset[2]) if placed_pos_offset.size >= 3 else 0.0
        label = f"{case_name}::{scene_label}".lower()
        is_single_object_case = (
            case_idx in LARGE_OBJECT_SINGLE_OBJECT_CASE_IDS
            or "single_object_preview" in label
            or "static_center" in label
            or "static_left" in label
            or "static_right" in label
            or "static_highdrop" in label
            or "entry_left" in label
            or "entry_right" in label
            or "entry_fast_center" in label
            or "random_parabola" in label
            or "high_drop" in label
            or "highdrop" in label
        )
        is_large_dynamic_case = (
            use_entry_motion
            or ((not object_fixed) and z_offset > 1e-4)
            or case_idx in LARGE_OBJECT_DYNAMIC_CASE_IDS
            or "random_parabola" in label
            or "high_drop" in label
            or "highdrop" in label
            or ("multi" in label and ("projectile" in label or "drop" in label))
        )
        if is_single_object_case or is_large_dynamic_case:
            removed_names.append(case_name or f"case_{case_idx}")
            continue
        filtered.append(cfg)

    if removed_names:
        preview = ", ".join(removed_names[:8])
        if len(removed_names) > 8:
            preview += ", ..."
        print(
            "INFO large_object_dynamic_filter "
            f"object_id={getattr(prepared, 'object_id', 'unknown')} "
            f"bbox_volume_m3={bbox_volume_m3:.4f} "
            f"threshold_m3={threshold_m3:.4f} "
            f"removed={len(removed_names)} "
            f"cases=[{preview}]"
        )
    return filtered


def _install_large_object_case_filter() -> None:
    if bool(getattr(BACKEND, "_hq_large_object_case_filter_installed", False)):
        return

    original_fn = BACKEND.build_preview_case_configs

    def wrapped_build_preview_case_configs(*, prepared: Any, output_root: Path, object_fixed: bool, args: argparse.Namespace):
        case_cfgs = original_fn(prepared=prepared, output_root=output_root, object_fixed=object_fixed, args=args)
        case_cfgs = _filter_case_configs_for_large_object(case_cfgs, prepared=prepared, args=args)
        return _hq_case_filter(args, case_cfgs)

    BACKEND.build_preview_case_configs = wrapped_build_preview_case_configs
    BACKEND._hq_large_object_case_filter_installed = True


def run_single_object(args: argparse.Namespace, object_id: str) -> Dict[str, Any]:
    applied = _apply_overrides(args)
    _install_large_object_case_filter()
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
