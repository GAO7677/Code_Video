#!/usr/bin/env python3
"""Validate and optionally quarantine single-object motion preview cases.

该脚本用于筛选单物体 case900/case901 运动样本中的异常样本；输入为样本根目录及可选 quarantine/report 参数，输出为 qa_metrics.json、JSON 报告以及可选隔离后的无效样本目录。

The filter is intentionally based on exported training targets, not on rendered
video only. It catches cases where Genesis dynamics or object geometry explode
and the object leaves the camera very early.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


CASE_NAMES = ("case900_random_parabola", "case901_high_drop")


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_ratio(uv: np.ndarray, vis: np.ndarray, width: int, height: int, margin: float) -> float:
    finite = np.isfinite(uv).all(axis=1)
    inside = (
        finite
        & vis
        & (uv[:, 0] >= margin)
        & (uv[:, 0] < float(width) - margin)
        & (uv[:, 1] >= margin)
        & (uv[:, 1] < float(height) - margin)
    )
    return float(np.mean(inside)) if inside.size else 0.0


def evaluate_sample(sample_dir: Path, *, margin: float = 24.0) -> Dict:
    metadata_path = sample_dir / "metadata.json"
    scene_path = sample_dir / "scene_input.json"
    kin_path = sample_dir / "physics" / "rigid_kinematics.npz"
    anchor_path = sample_dir / "physics" / "anchor_targets.npz"
    required = [metadata_path, scene_path, kin_path, anchor_path]
    missing = [str(p.relative_to(sample_dir)) for p in required if not p.exists()]
    if missing:
        return {"sample_dir": str(sample_dir), "valid": False, "reasons": ["missing_files"], "missing": missing}

    meta = _load_json(metadata_path)
    scene = _load_json(scene_path)
    width, height = [int(v) for v in meta.get("resolution", [960, 720])]
    kin = np.load(kin_path)
    anchor = np.load(anchor_path)

    pos = np.asarray(kin["com_pos"], dtype=np.float64)[:, 0, :]
    vel = np.asarray(kin["linear_vel"], dtype=np.float64)[:, 0, :]
    uv = np.asarray(anchor["com_uv"], dtype=np.float64)[:, 0, :]
    vis = np.asarray(anchor["visibility_mask"])[:, 0].astype(bool)
    bbox = np.asarray(anchor["bbox_xyxy"], dtype=np.float64)[:, 0, :]

    finite_uv = np.isfinite(uv).all(axis=1)
    bbox_w = np.maximum(0.0, bbox[:, 2] - bbox[:, 0])
    bbox_h = np.maximum(0.0, bbox[:, 3] - bbox[:, 1])
    bbox_area = bbox_w * bbox_h
    speed = np.linalg.norm(vel, axis=1)
    xy_radius = np.linalg.norm(pos[:, :2], axis=1)

    metrics = {
        "sample_dir": str(sample_dir),
        "case_name": str(scene.get("case_name", sample_dir.name)),
        "frames": int(pos.shape[0]),
        "visible_ratio": float(np.mean(vis)) if vis.size else 0.0,
        "finite_uv_ratio": float(np.mean(finite_uv)) if finite_uv.size else 0.0,
        "safe_uv_ratio": _safe_ratio(uv, vis, width, height, margin),
        "first_visible_frames": int(np.sum(vis[: min(8, len(vis))])),
        "last_visible_frames": int(np.sum(vis[max(0, len(vis) - 8) :])),
        "median_bbox_area_visible": float(np.median(bbox_area[vis])) if np.any(vis) else 0.0,
        "max_speed_mps": float(np.nanmax(speed)) if speed.size else 0.0,
        "max_xy_radius_m": float(np.nanmax(xy_radius)) if xy_radius.size else 0.0,
        "max_abs_z_m": float(np.nanmax(np.abs(pos[:, 2]))) if pos.size else 0.0,
        "initial_com_pos": pos[0].tolist() if pos.shape[0] else [],
        "final_com_pos": pos[-1].tolist() if pos.shape[0] else [],
        "entry_linear_velocity": scene.get("entry_linear_velocity"),
        "resolution": [width, height],
        "margin_px": float(margin),
    }

    reasons: List[str] = []
    if metrics["visible_ratio"] < 0.70:
        reasons.append("low_visible_ratio")
    if metrics["safe_uv_ratio"] < 0.55:
        reasons.append("low_safe_uv_ratio")
    if metrics["first_visible_frames"] < 6:
        reasons.append("not_visible_in_context")
    if metrics["last_visible_frames"] < 4:
        reasons.append("not_visible_near_end")
    if metrics["median_bbox_area_visible"] < 64.0:
        reasons.append("tiny_or_missing_bbox")
    if metrics["max_speed_mps"] > 20.0:
        reasons.append("speed_explosion")
    if metrics["max_xy_radius_m"] > 6.0:
        reasons.append("xy_position_explosion")
    if metrics["max_abs_z_m"] > 8.0:
        reasons.append("z_position_explosion")

    metrics["valid"] = not reasons
    metrics["reasons"] = reasons
    return metrics


def iter_samples(root: Path, case_names: Iterable[str]) -> Iterable[Path]:
    for case in case_names:
        yield from sorted(root.rglob(f"*__{case}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter invalid single-object case900/901 samples")
    parser.add_argument("--root", type=Path, required=True, help="Root folder to scan, e.g. .../single_object_preview/count_01")
    parser.add_argument("--case", action="append", choices=CASE_NAMES, help="Case name to scan; defaults to both")
    parser.add_argument("--margin", type=float, default=24.0, help="Pixel margin for safe projected-center ratio")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--write_metrics", action="store_true", help="Write qa_metrics.json in each sample dir")
    parser.add_argument("--quarantine_dir", type=Path, default=None, help="Move invalid samples here instead of deleting")
    parser.add_argument("--delete_invalid", action="store_true", help="Delete invalid samples after writing/reporting metrics")
    args = parser.parse_args()

    cases = tuple(args.case) if args.case else CASE_NAMES
    samples = list(iter_samples(args.root, cases))
    results = [evaluate_sample(sample, margin=float(args.margin)) for sample in samples]

    for result in results:
        sample_dir = Path(result["sample_dir"])
        if args.write_metrics and sample_dir.exists():
            (sample_dir / "qa_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    invalid = [r for r in results if not r.get("valid", False)]
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        payload = {"total": len(results), "valid": len(results) - len(invalid), "invalid": len(invalid), "samples": results}
        args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.quarantine_dir is not None:
        args.quarantine_dir.mkdir(parents=True, exist_ok=True)
        for result in invalid:
            src = Path(result["sample_dir"])
            if not src.exists():
                continue
            dst = args.quarantine_dir / src.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            result["quarantined_to"] = str(dst)

    if args.delete_invalid:
        for result in invalid:
            src = Path(result["sample_dir"])
            if src.exists():
                shutil.rmtree(src)

    print(f"total={len(results)} valid={len(results) - len(invalid)} invalid={len(invalid)}")
    for result in invalid[:50]:
        print(f"INVALID {Path(result['sample_dir']).name} reasons={','.join(result['reasons'])} "
              f"vis={result.get('visible_ratio', 0):.2f} safe={result.get('safe_uv_ratio', 0):.2f} "
              f"speed={result.get('max_speed_mps', 0):.2f} xy={result.get('max_xy_radius_m', 0):.2f} z={result.get('max_abs_z_m', 0):.2f}")


if __name__ == "__main__":
    main()
