"""Build a static, frame-accurate overlay viewer for the RGB-to-Bullet pilot."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from context_rgb_pybullet_common import BALL_RADIUS_M, dump_json, load_json


PRIMARY_MODES = {
    "A": "A_gt_state_gt_geometry__omega_gt_observed",
    "B": "B_estimated_state_gt_geometry__omega_gt_observed",
    "C": "C_gt_state_estimated_geometry__omega_gt_observed",
    "D": "D_estimated_state_estimated_geometry__omega_gt_observed",
}
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def project_points(
    points: np.ndarray,
    calibration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    intrinsic = np.asarray(calibration["intrinsic_K"], dtype=np.float64)
    world_to_camera = np.asarray(calibration["world_to_camera_3x4"], dtype=np.float64)
    camera = np.column_stack((points, np.ones(len(points)))) @ world_to_camera.T
    projected = camera @ intrinsic.T
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    visible = camera[:, 2] > 1e-5
    uv[visible] = projected[visible, :2] / projected[visible, 2:3]
    return uv, camera[:, 2]


def json_points(values: np.ndarray) -> list[list[float | None]]:
    output = []
    for row in np.asarray(values, dtype=np.float64):
        output.append([float(value) if np.isfinite(value) else None for value in row])
    return output


def primitive_corners(primitive: dict[str, Any], *, estimated: bool) -> np.ndarray:
    center = np.asarray(primitive["position_m"], dtype=np.float64)
    half = np.asarray(
        primitive["size"]["half_extents_m"]
        if estimated else primitive["half_extents_m"],
        dtype=np.float64,
    )
    quaternion = np.asarray(primitive["orientation_xyzw"], dtype=np.float64)
    qx, qy, qz, qw = quaternion
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    rotation = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    signs = np.asarray(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    return signs * half @ rotation.T + center


def primitive_wireframe(
    primitive: dict[str, Any],
    calibration: dict[str, Any],
    *,
    estimated: bool,
) -> list[list[list[float]]]:
    uv, depth = project_points(primitive_corners(primitive, estimated=estimated), calibration)
    segments = []
    for left, right in BOX_EDGES:
        if depth[left] <= 0 or depth[right] <= 0 or not np.isfinite(uv[[left, right]]).all():
            continue
        segments.append(uv[[left, right]].astype(float).tolist())
    return segments


def tint_mask(image: np.ndarray, mask: np.ndarray, bgr: tuple[int, int, int], alpha: float) -> None:
    color = np.asarray(bgr, dtype=np.float32)
    current = image[mask].astype(np.float32)
    image[mask] = np.clip((1.0 - alpha) * current + alpha * color, 0, 255).astype(np.uint8)


def draw_mask_frame(
    base: np.ndarray,
    sam_mask: np.ndarray,
    sphere_mask: np.ndarray,
) -> np.ndarray:
    output = base.copy()
    tint_mask(output, sam_mask, (211, 149, 36), 0.36)
    tint_mask(output, sphere_mask, (66, 209, 255), 0.38)
    sam_contours, _ = cv2.findContours(
        sam_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    sphere_contours, _ = cv2.findContours(
        sphere_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, sam_contours, -1, (230, 173, 62), 2, cv2.LINE_AA)
    cv2.drawContours(output, sphere_contours, -1, (45, 245, 255), 2, cv2.LINE_AA)
    return output


def depth_limits(depth: np.ndarray, metric_scale: float) -> tuple[float, float]:
    values = np.asarray(depth, dtype=np.float32).reshape(-1) * float(metric_scale)
    values = values[np.isfinite(values) & (values > 0.05) & (values < 20.0)]
    if len(values) < 100:
        raise ValueError("too few finite depth values")
    low, high = np.quantile(values, [0.02, 0.98])
    if high - low < 0.05:
        high = low + 0.05
    return float(low), float(high)


def draw_depth_frame(
    base: np.ndarray,
    depth: np.ndarray,
    confidence: np.ndarray,
    metric_scale: float,
    limits: tuple[float, float],
    sphere_mask: np.ndarray,
) -> np.ndarray:
    height, width = base.shape[:2]
    depth_m = cv2.resize(
        np.asarray(depth, dtype=np.float32) * float(metric_scale),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    confidence = cv2.resize(
        np.asarray(confidence, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    low, high = limits
    normalized = np.clip((depth_m - low) / (high - low), 0.0, 1.0)
    heat = cv2.applyColorMap(
        np.rint((1.0 - normalized) * 255.0).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    conf_low, conf_high = np.quantile(confidence[np.isfinite(confidence)], [0.05, 0.95])
    confidence_alpha = np.clip((confidence - conf_low) / max(conf_high - conf_low, 1e-6), 0.0, 1.0)
    alpha = (0.25 + 0.35 * confidence_alpha)[..., None]
    output = np.clip((1.0 - alpha) * base + alpha * heat, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        sphere_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, (45, 245, 255), 2, cv2.LINE_AA)
    return output


def draw_pointmap_frame(
    base: np.ndarray,
    points: np.ndarray,
    confidence: np.ndarray,
    calibration: dict[str, Any],
) -> tuple[np.ndarray, int]:
    height, width = base.shape[:2]
    uv, camera_depth = project_points(points, calibration)
    visible = (
        np.isfinite(uv).all(axis=1)
        & (camera_depth > 0)
        & (uv[:, 0] >= 0) & (uv[:, 0] < width)
        & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    indices = np.flatnonzero(visible)
    if len(indices) > 6500:
        # Preserve the full image footprint while mildly preferring confident points.
        order = np.argsort(confidence[indices], kind="stable")
        ranked = indices[order]
        uniform = np.linspace(0, len(ranked) - 1, 6500, dtype=np.int64)
        indices = ranked[uniform]
    selected_uv = np.rint(uv[indices]).astype(np.int32)
    selected_z = points[indices, 2]
    z_low, z_high = np.quantile(selected_z, [0.02, 0.98])
    normalized = np.clip((selected_z - z_low) / max(z_high - z_low, 1e-6), 0.0, 1.0)
    colors = cv2.applyColorMap(
        np.rint(normalized * 255.0).astype(np.uint8).reshape(-1, 1),
        cv2.COLORMAP_VIRIDIS,
    ).reshape(-1, 3)
    overlay = base.copy()
    for (x, y), color in zip(selected_uv, colors):
        cv2.circle(overlay, (int(x), int(y)), 1, tuple(map(int, color)), -1, cv2.LINE_AA)
    output = cv2.addWeighted(base, 0.48, overlay, 0.72, 0.0)
    return output, int(len(indices))


def write_webp(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_WEBP_QUALITY, 86]):
        raise RuntimeError(f"failed to write {path}")


def rollout_payload(
    evaluation_root: Path,
    case: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    payload = {}
    for letter, mode_name in PRIMARY_MODES.items():
        mode = case["modes"][mode_name]
        with np.load(
            evaluation_root / "rollouts" / case["case_id"] / f"{mode_name}.npz",
            allow_pickle=False,
        ) as archive:
            future = archive["positions"].astype(np.float64)
        positions = np.vstack((np.asarray(mode["initial_position_m"], dtype=np.float64), future))
        uv, camera_depth = project_points(positions, calibration)
        focal = float(np.asarray(calibration["intrinsic_K"], dtype=np.float64)[0, 0])
        radius_px = np.where(
            camera_depth > 1e-5,
            np.clip(focal * BALL_RADIUS_M / camera_depth, 3.0, 30.0),
            7.0,
        )
        event = mode["event"]
        event_index = event["first_interaction_future_index_30hz"]
        payload[letter] = {
            "label": {
                "A": "GT state + GT geometry",
                "B": "Estimated state + GT geometry",
                "C": "GT state + estimated geometry",
                "D": "Estimated state + estimated geometry",
            }[letter],
            "points_px": json_points(uv),
            "radius_px": radius_px.astype(float).tolist(),
            "position_error_m": mode["position_error_by_future_frame_m"],
            "ADE_m": float(mode["ADE_m"]),
            "FDE_m": float(mode["FDE_m"]),
            "status": mode["trajectory_metric_status"],
            "included_in_valid_aggregate": bool(mode["included_in_valid_aggregate"]),
            "contact_outcome": event["category"],
            "contact_match": bool(mode["contact_outcome_match_oracle"]),
            "first_contact_time_s": event["first_interaction_time_after_rgb7_s"],
            "first_contact_path_index": None if event_index is None else int(event_index) + 1,
            "max_penetration_m": float(event["max_penetration_m"]),
            "post_contact_velocity_error": mode["post_contact_velocity_error"],
        }
    return payload


def build_case(
    case_id: str,
    input_root: Path,
    vision_source: Path,
    vision_root: Path,
    primitive_root: Path,
    evaluation_root: Path,
    output: Path,
) -> dict[str, Any]:
    case_input = input_root / case_id
    calibration = load_json(case_input / "calibration.json")
    input_record = load_json(case_input / "input.json")
    vision_report = load_json(vision_root / "estimates" / case_id / "report.json")
    primitive_report = load_json(primitive_root / "cases" / case_id / "primitives.json")
    case = load_json(evaluation_root / "cases" / case_id / "evaluation.json")
    with np.load(
        vision_root / "estimates" / case_id / "estimate.npz", allow_pickle=False
    ) as archive:
        estimated_centers = archive["estimated_centers"].astype(np.float64)
        estimated_p7 = archive["estimated_p7"].astype(np.float64)
        estimated_v7 = archive["estimated_v7"].astype(np.float64)
        circles = archive["sphere_circles_source"].astype(np.float64)
        sphere_masks = archive["masks_source"].astype(bool)
        static_points = archive["static_points"].astype(np.float64)
        static_confidence = archive["static_confidence"].astype(np.float64)
    with np.load(
        vision_source / "sam2_masks" / f"{case_id}.npz", allow_pickle=False
    ) as archive:
        sam_masks = archive["masks"].astype(bool)
    with np.load(
        vision_source / "raw_vggt" / f"{case_id}.npz", allow_pickle=False
    ) as archive:
        depth = archive["depth"].astype(np.float32)
        if depth.ndim == 4:
            depth = depth[..., 0]
        depth_confidence = archive["depth_conf"].astype(np.float32)

    context_paths = [f"../vision_inputs/{case_id}/{name}" for name in input_record["frames"]]
    base_frames = [cv2.imread(str(case_input / name), cv2.IMREAD_COLOR) for name in input_record["frames"]]
    if any(frame is None for frame in base_frames):
        raise FileNotFoundError(f"missing context frame for {case_id}")

    asset_root = output / "assets" / case_id
    mask_paths, depth_paths = [], []
    metric_scale = float(vision_report["vggt_depth_scale"]["metric_scale"])
    limits = depth_limits(depth, metric_scale)
    for frame_index, base in enumerate(base_frames):
        mask_path = asset_root / f"mask_{frame_index:02d}.webp"
        depth_path = asset_root / f"depth_{frame_index:02d}.webp"
        write_webp(mask_path, draw_mask_frame(base, sam_masks[frame_index], sphere_masks[frame_index]))
        write_webp(
            depth_path,
            draw_depth_frame(
                base,
                depth[frame_index],
                depth_confidence[frame_index],
                metric_scale,
                limits,
                sphere_masks[frame_index],
            ),
        )
        mask_paths.append(str(mask_path.relative_to(output)))
        depth_paths.append(str(depth_path.relative_to(output)))

    pointmap_path = asset_root / "pointmap_rgb7.webp"
    pointmap, rendered_point_count = draw_pointmap_frame(
        base_frames[-1], static_points, static_confidence, calibration
    )
    write_webp(pointmap_path, pointmap)

    centers_px, _ = project_points(estimated_centers, calibration)
    velocity_world = np.vstack((estimated_p7, estimated_p7 + 0.20 * estimated_v7))
    velocity_px, _ = project_points(velocity_world, calibration)
    primitives = []
    for row in case["geometry_evaluation"]["per_primitive"]:
        primitives.append({
            "name": row["name"],
            "estimated_segments": primitive_wireframe(
                row["estimated"], calibration, estimated=True
            ),
            "gt_segments": primitive_wireframe(
                row["gt_canonical"], calibration, estimated=False
            ),
            "center_error_m": float(row["center_error_m"]),
            "full_size_l2_error_m": float(row["full_size_l2_error_m"]),
            "orientation_error_deg": float(row["orientation_error_deg"]),
        })

    frame_mask_stats = vision_report["sphere_mask_regularization"]["frames"]
    payload = {
        "schema": "context_rgb_to_pybullet_overlay_case_v1",
        "case_id": case_id,
        "family": case["family"],
        "group_id": case["group_id"],
        "geometry_value": case["geometry_value"],
        "geometry_units": case["geometry_units"],
        "context": {
            "frames": context_paths,
            "times_s": input_record["time_s"],
            "resolution_wh": calibration["resolution_wh"],
            "allowed_inputs": input_record["allowed_estimator_inputs"],
        },
        "mask": {
            "frames": mask_paths,
            "sam2_area_pixels": [int(row["sam2_area_pixels"]) for row in frame_mask_stats],
            "sphere_area_pixels": [int(row["circle_area_pixels"]) for row in frame_mask_stats],
            "sphere_inside_sam_ratio": [float(row["circle_inside_sam2_ratio"]) for row in frame_mask_stats],
            "circles_xyr": circles.astype(float).tolist(),
            "method": vision_report["sphere_mask_regularization"]["method"],
        },
        "state": {
            "centers_world_m": estimated_centers.astype(float).tolist(),
            "centers_px": json_points(centers_px),
            "estimated_p7_m": estimated_p7.astype(float).tolist(),
            "estimated_v7_mps": estimated_v7.astype(float).tolist(),
            "velocity_arrow_px": json_points(velocity_px),
            "fit_degree": int(vision_report["velocity_fit"]["method"].split("degree_")[1].split("_")[0]),
            "inlier_frames": vision_report["velocity_fit"]["inlier_frames"],
            "excluded_frames": vision_report["velocity_fit"]["excluded_frames"],
            "evaluation": case["state_evaluation"],
        },
        "depth": {
            "frames": depth_paths,
            "metric_scale": metric_scale,
            "display_range_m": list(limits),
            "confidence_threshold": float(vision_report["static_cloud"]["confidence_threshold"]),
            "confidence_quantile": float(vision_report["static_cloud"]["confidence_quantile"]),
            "method": vision_report["static_cloud"]["method"],
        },
        "geometry": {
            "pointmap_frame": str(pointmap_path.relative_to(output)),
            "rendered_point_count": rendered_point_count,
            "primitives": primitives,
            "primitive_count": len(primitive_report["primitives"]),
            "fit": primitive_report["family_fit"],
            "ground_fit": primitive_report["ground_fit"],
            "evaluation": case["geometry_evaluation"],
        },
        "rollout": rollout_payload(evaluation_root, case, calibration),
        "isolation": {
            "future_rgb_used_for_estimation": False,
            "gt_state_used_for_estimation": False,
            "blueprint_used_for_estimation": False,
            "future_background": "RGB7 hold; projected trajectories only",
        },
    }
    case_json = output / "data" / "cases" / f"{case_id}.json"
    dump_json(case_json, payload)
    d_mode = case["modes"][PRIMARY_MODES["D"]]
    return {
        "case_id": case_id,
        "family": case["family"],
        "group_id": case["group_id"],
        "geometry_value": case["geometry_value"],
        "geometry_units": case["geometry_units"],
        "data": str(case_json.relative_to(output)),
        "D_ADE_m": float(d_mode["ADE_m"]),
        "D_FDE_m": float(d_mode["FDE_m"]),
        "D_contact_match": bool(d_mode["contact_outcome_match_oracle"]),
        "D_contact_outcome": d_mode["event"]["category"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    for name in ("index.html", "styles.css", "app.js"):
        shutil.copy2(args.web_root.resolve() / name, output / name)

    input_manifest = load_json(args.input_root.resolve() / "manifest.json")
    case_ids = sorted(str(record["case_id"]) for record in input_manifest["records"])
    if len(case_ids) != 36:
        raise ValueError(f"expected 36 cases, got {len(case_ids)}")
    records = []
    for index, case_id in enumerate(case_ids, start=1):
        record = build_case(
            case_id,
            args.input_root.resolve(),
            args.vision_source.resolve(),
            args.vision_root.resolve(),
            args.primitive_root.resolve(),
            args.evaluation_root.resolve(),
            output,
        )
        records.append(record)
        print(f"VIEWER_CASE_OK {index:02d}/36 {case_id}", flush=True)

    summary = load_json(args.delivery_root.resolve() / "summary.json")
    manifest = {
        "schema": "context_rgb_to_pybullet_overlay_manifest_v1",
        "status": "EXECUTED",
        "case_count": len(records),
        "families": ["aperture", "deflector", "support_edge"],
        "records": records,
        "default_case_id": "phase1_deflector_g03_v28000",
        "stages": ["context", "mask", "state", "depth", "geometry", "rollout"],
        "summary": {
            "D_ADE_m": summary["rollout"]["D"]["ADE_mean_valid_m"],
            "D_FDE_m": summary["rollout"]["D"]["FDE_mean_valid_m"],
            "D_contact_match": summary["rollout"]["D"]["contact_match_valid_count"],
            "state_position_error_m": summary["state"]["all"]["position_error_m"]["mean"],
            "state_velocity_error_mps": summary["state"]["all"]["velocity_vector_error_mps"]["mean"],
        },
        "source_versions": {
            "vision": str(args.vision_root.resolve()),
            "primitives": str(args.primitive_root.resolve()),
            "evaluation": str(args.evaluation_root.resolve()),
        },
    }
    dump_json(output / "data" / "manifest.json", manifest)
    dump_json(
        output / "build_report.json",
        {
            "status": "EXECUTED",
            "case_count": len(records),
            "mask_overlay_frame_count": len(records) * 8,
            "depth_overlay_frame_count": len(records) * 8,
            "pointmap_frame_count": len(records),
            "future_rgb_rendered": False,
            "future_background": "RGB7 hold",
            "gt_and_future_scope": "evaluation overlays only",
        },
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--vision-source", type=Path, required=True)
    parser.add_argument("--vision-root", type=Path, required=True)
    parser.add_argument("--primitive-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--delivery-root", type=Path, required=True)
    parser.add_argument("--web-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "case_count": result["case_count"]}, indent=2))
