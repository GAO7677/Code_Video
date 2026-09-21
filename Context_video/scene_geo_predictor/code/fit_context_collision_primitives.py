"""Fit sparse PyBullet collision primitives from RGB-derived point maps only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from context_rgb_pybullet_common import BALL_RADIUS_M, dump_json, load_json, yaw_quaternion


def histogram_mode(values: np.ndarray, bin_width: float) -> tuple[float, dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 50:
        raise ValueError(f"too few values for a mode: {len(values)}")
    low, high = np.quantile(values, [0.005, 0.995])
    if high - low < bin_width:
        return float(np.median(values)), {"count": int(len(values)), "bin_width": bin_width}
    edges = np.arange(low, high + 2.0 * bin_width, bin_width)
    counts, edges = np.histogram(values, bins=edges)
    index = int(np.argmax(counts))
    admitted = values[(values >= edges[index]) & (values < edges[index + 1])]
    return float(np.median(admitted)), {
        "count": int(len(values)),
        "winning_count": int(counts[index]),
        "bin_width": float(bin_width),
        "winning_interval": [float(edges[index]), float(edges[index + 1])],
    }


def box_primitive(
    name: str,
    position: np.ndarray | list[float],
    half_extents: np.ndarray | list[float],
    yaw_rad: float = 0.0,
    *,
    source: str,
) -> dict[str, Any]:
    half_extents = np.asarray(half_extents, dtype=np.float64)
    if half_extents.shape != (3,) or not np.isfinite(half_extents).all() or np.any(half_extents <= 0):
        raise ValueError(f"invalid half extents for {name}: {half_extents}")
    return {
        "name": name,
        "shape": "box",
        "position_m": np.asarray(position, dtype=np.float64).tolist(),
        "orientation_xyzw": yaw_quaternion(yaw_rad),
        "orientation_yaw_deg": float(math.degrees(yaw_rad)),
        "size": {
            "half_extents_m": half_extents.tolist(),
            "full_extents_m": (2.0 * half_extents).tolist(),
        },
        "material": {"lateral_friction": 0.35, "restitution": 0.25},
        "source": source,
    }


def fit_ground(points: np.ndarray, estimated_p7: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    z = points[:, 2]
    candidates = z[(z > -0.8) & (z < min(float(estimated_p7[2]) + 0.08, 0.7))]
    ground_z, mode_report = histogram_mode(candidates, 0.015)
    primitive = box_primitive(
        "estimated_ground",
        [0.0, 0.0, ground_z - 0.05],
        [6.0, 6.0, 0.05],
        source="dominant_low_horizontal_depth_mode",
    )
    return primitive, {"ground_top_z_m": ground_z, "mode": mode_report}


def fit_deflector(points: np.ndarray, colors: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rgb = colors.astype(np.float32)
    blue = (
        (rgb[:, 2] > 90)
        & (rgb[:, 2] - rgb[:, 0] > 22)
        & (rgb[:, 2] - rgb[:, 1] > 10)
    )
    selected = points[blue]
    if len(selected) < 150:
        raise ValueError(f"only {len(selected)} blue deflector points")
    xy = selected[:, :2]
    robust_center = np.median(xy, axis=0)
    distance = np.linalg.norm(xy - robust_center, axis=1)
    selected = selected[distance <= np.quantile(distance, 0.985)]
    xy = selected[:, :2]
    covariance = np.cov(xy - np.mean(xy, axis=0), rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    long_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if long_axis[1] < 0:
        long_axis = -long_axis
    normal_axis = np.asarray([long_axis[1], -long_axis[0]])
    along_long = xy @ long_axis
    along_normal = xy @ normal_axis
    long_low, long_high = np.quantile(along_long, [0.01, 0.99])
    normal_low, normal_high = np.quantile(along_normal, [0.02, 0.98])
    z_low, z_high = np.quantile(selected[:, 2], [0.01, 0.99])
    half_long_raw = 0.5 * float(long_high - long_low)
    half_short_raw = 0.5 * float(normal_high - normal_low)
    half_z_raw = 0.5 * float(z_high - z_low)
    half_long = float(np.clip(half_long_raw, 0.20, 1.30))
    half_short = float(np.clip(half_short_raw, 0.02, 0.20))
    half_z = float(np.clip(half_z_raw, 0.08, 0.55))
    center_xy = normal_axis * (0.5 * (normal_low + normal_high)) + long_axis * (0.5 * (long_low + long_high))
    center_z = 0.5 * float(z_low + z_high)
    yaw = math.atan2(float(normal_axis[1]), float(normal_axis[0]))
    primitive = box_primitive(
        "estimated_deflector",
        [center_xy[0], center_xy[1], center_z],
        [half_short, half_long, half_z],
        yaw,
        source="blue_static_points_oriented_PCA_quantile_box",
    )
    report = {
        "selected_point_count": int(len(selected)),
        "normal_xy": normal_axis.tolist(),
        "long_axis_xy": long_axis.tolist(),
        "raw_half_extents_m": [half_short_raw, half_long_raw, half_z_raw],
        "clipped_half_extents_m": [half_short, half_long, half_z],
        "extent_clipped": bool(
            not np.allclose(
                [half_short_raw, half_long_raw, half_z_raw],
                [half_short, half_long, half_z],
            )
        ),
    }
    return [primitive], report


def low_occupancy_runs(values: np.ndarray, low: float, high: float, bin_width: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    edges = np.arange(low, high + 2.0 * bin_width, bin_width)
    counts, edges = np.histogram(values, bins=edges)
    smooth = np.convolve(counts.astype(np.float64), np.ones(3), mode="same")
    positive = smooth[smooth > 0]
    if len(positive) == 0:
        raise ValueError("empty occupancy histogram")
    threshold = max(1.0, 0.12 * float(np.median(positive)))
    empty = smooth <= threshold
    runs = []
    start = None
    for index, is_empty in enumerate(np.r_[empty, False]):
        if is_empty and start is None:
            start = index
        elif not is_empty and start is not None:
            end = index
            if start > 2 and end < len(counts) - 2:
                runs.append(
                    {
                        "low": float(edges[start]),
                        "high": float(edges[end]),
                        "width": float(edges[end] - edges[start]),
                        "start_bin": int(start),
                        "end_bin_exclusive": int(end),
                    }
                )
            start = None
    return runs, {
        "range": [float(low), float(high)],
        "bin_width": float(bin_width),
        "threshold": threshold,
        "bin_count": int(len(counts)),
        "counts": counts.astype(int).tolist(),
    }


def choose_central_gap(runs: list[dict[str, Any]], low: float, high: float) -> dict[str, Any]:
    span = high - low
    center = 0.5 * (low + high)
    candidates = [
        run for run in runs
        if run["width"] >= 0.025
        and run["low"] > low + 0.18 * span
        and run["high"] < high - 0.18 * span
    ]
    if not candidates:
        raise ValueError(f"no bracketed central gap among {len(runs)} low-occupancy runs")
    return min(
        candidates,
        key=lambda run: (abs(0.5 * (run["low"] + run["high"]) - center), -run["width"]),
    )


def fit_support_edge(
    points: np.ndarray,
    estimated_p7: np.ndarray,
    ground_z: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x, y, z = points.T
    plausible = (
        (z > ground_z + 0.12)
        & (z < float(estimated_p7[2]) + 0.20)
        & (np.abs(y - float(estimated_p7[1])) < 1.25)
        & (x > float(estimated_p7[0]) - 1.0)
        & (x < float(estimated_p7[0]) + 4.0)
    )
    top_z, top_mode = histogram_mode(z[plausible], 0.012)
    top = plausible & (np.abs(z - top_z) < 0.035)
    top_points = points[top]
    if len(top_points) < 250:
        raise ValueError(f"only {len(top_points)} platform-top points")
    x_low, x_high = np.quantile(top_points[:, 0], [0.005, 0.995])
    y_low, y_high = np.quantile(top_points[:, 1], [0.01, 0.99])
    runs, occupancy = low_occupancy_runs(top_points[:, 0], float(x_low), float(x_high), 0.015)
    gap = choose_central_gap(runs, float(x_low), float(x_high))
    if gap["width"] > 0.9:
        raise ValueError(f"implausibly wide platform gap: {gap['width']:.3f} m")

    footprint = (
        (x >= x_low) & (x <= x_high)
        & (y >= y_low) & (y <= y_high)
        & (z > ground_z + 0.08) & (z <= top_z + 0.04)
    )
    lower_surface = float(np.quantile(z[footprint], 0.10)) if int(footprint.sum()) >= 100 else top_z - 0.12
    full_thickness_raw = top_z - lower_surface
    full_thickness = float(np.clip(full_thickness_raw, 0.04, 0.30))
    half_z = 0.5 * full_thickness
    center_z = top_z - half_z
    half_y = max(0.05, 0.5 * float(y_high - y_low))
    left_half_x = 0.5 * float(gap["low"] - x_low)
    right_half_x = 0.5 * float(x_high - gap["high"])
    if min(left_half_x, right_half_x) <= 0.05:
        raise ValueError("estimated platform side is too short")
    primitives = [
        box_primitive(
            "estimated_left_platform",
            [0.5 * (x_low + gap["low"]), 0.5 * (y_low + y_high), center_z],
            [left_half_x, half_y, half_z],
            source="elevated_horizontal_mode_left_of_depth_gap",
        ),
        box_primitive(
            "estimated_right_platform",
            [0.5 * (gap["high"] + x_high), 0.5 * (y_low + y_high), center_z],
            [right_half_x, half_y, half_z],
            source="elevated_horizontal_mode_right_of_depth_gap",
        ),
    ]
    report = {
        "top_z_m": top_z,
        "top_mode": top_mode,
        "top_point_count": int(len(top_points)),
        "x_outer_edges_m": [float(x_low), float(x_high)],
        "y_outer_edges_m": [float(y_low), float(y_high)],
        "gap_edges_m": [gap["low"], gap["high"]],
        "gap_width_m": gap["width"],
        "raw_thickness_m": full_thickness_raw,
        "used_thickness_m": full_thickness,
        "thickness_clipped": not math.isclose(full_thickness_raw, full_thickness),
        "occupancy": occupancy,
        "candidate_gaps": runs,
    }
    return primitives, report


def fit_aperture(
    points: np.ndarray,
    estimated_p7: np.ndarray,
    ground_z: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x, y, z = points.T
    ahead = (
        (x > float(estimated_p7[0]) + 0.35)
        & (x < float(estimated_p7[0]) + 3.5)
        & (z > ground_z + 0.05)
        & (z < ground_z + 2.5)
        & (np.abs(y - float(estimated_p7[1])) < 2.2)
    )
    front_x, plane_mode = histogram_mode(x[ahead], 0.015)
    plane = ahead & (np.abs(x - front_x) < 0.045)
    plane_points = points[plane]
    if len(plane_points) < 500:
        raise ValueError(f"only {len(plane_points)} aperture-plane points")
    y_low, y_high = np.quantile(plane_points[:, 1], [0.005, 0.995])
    z_high = float(np.quantile(plane_points[:, 2], 0.995))
    low_band = plane_points[
        (plane_points[:, 2] > ground_z + 0.08)
        & (plane_points[:, 2] < min(ground_z + 0.70, z_high - 0.10))
    ]
    runs, occupancy = low_occupancy_runs(low_band[:, 1], float(y_low), float(y_high), 0.012)
    gap = choose_central_gap(runs, float(y_low), float(y_high))
    if not 0.12 <= gap["width"] <= 1.20:
        raise ValueError(f"implausible aperture width: {gap['width']:.3f} m")
    inside = plane_points[
        (plane_points[:, 1] > gap["low"] + 0.15 * gap["width"])
        & (plane_points[:, 1] < gap["high"] - 0.15 * gap["width"])
        & (plane_points[:, 2] > ground_z + 0.25)
    ]
    if len(inside) < 50:
        raise ValueError("too few header points inside aperture span")
    opening_height = float(np.quantile(inside[:, 2], 0.02) - ground_z)
    if not 0.55 <= opening_height <= 1.50:
        raise ValueError(f"implausible opening height: {opening_height:.3f} m")

    near_structure = ahead & (np.abs(x - front_x) < 0.40)
    x_depth = x[near_structure]
    back_x = float(np.quantile(x_depth[x_depth >= front_x - 0.02], 0.90))
    full_thickness_raw = back_x - front_x
    full_thickness = float(np.clip(full_thickness_raw, 0.04, 0.50))
    half_x = 0.5 * full_thickness
    center_x = front_x + half_x
    pillar_half_z = 0.5 * opening_height
    pillar_center_z = ground_z + pillar_half_z
    top_z = max(z_high, ground_z + opening_height + 0.10)
    header_half_z = 0.5 * (top_z - (ground_z + opening_height))
    header_center_z = ground_z + opening_height + header_half_z
    left_half_y = 0.5 * (gap["low"] - y_low)
    right_half_y = 0.5 * (y_high - gap["high"])
    if min(left_half_y, right_half_y, header_half_z) <= 0.04:
        raise ValueError("degenerate aperture primitive extent")
    primitives = [
        box_primitive(
            "estimated_aperture_left",
            [center_x, 0.5 * (y_low + gap["low"]), pillar_center_z],
            [half_x, left_half_y, pillar_half_z],
            source="front_wall_plane_left_of_low_height_depth_gap",
        ),
        box_primitive(
            "estimated_aperture_right",
            [center_x, 0.5 * (gap["high"] + y_high), pillar_center_z],
            [half_x, right_half_y, pillar_half_z],
            source="front_wall_plane_right_of_low_height_depth_gap",
        ),
        box_primitive(
            "estimated_aperture_header",
            [center_x, 0.5 * (y_low + y_high), header_center_z],
            [half_x, 0.5 * (y_high - y_low), header_half_z],
            source="front_wall_plane_above_vertical_depth_gap",
        ),
    ]
    report = {
        "front_surface_x_m": front_x,
        "plane_mode": plane_mode,
        "plane_point_count": int(len(plane_points)),
        "outer_y_edges_m": [float(y_low), float(y_high)],
        "gap_edges_y_m": [gap["low"], gap["high"]],
        "opening_width_m": gap["width"],
        "opening_height_m": opening_height,
        "top_z_m": top_z,
        "raw_thickness_m": full_thickness_raw,
        "used_thickness_m": full_thickness,
        "thickness_clipped": not math.isclose(full_thickness_raw, full_thickness),
        "occupancy": occupancy,
        "candidate_gaps": runs,
    }
    return primitives, report


def fit_case(vision_output: Path, record: dict[str, Any], output: Path) -> dict[str, Any]:
    case_id = str(record["case_id"])
    family = str(record["family"])
    estimate_path = vision_output / "estimates" / case_id / "estimate.npz"
    with np.load(estimate_path, allow_pickle=False) as archive:
        points = archive["static_points"].astype(np.float64)
        colors = archive["static_colors"].astype(np.uint8)
        estimated_p7 = archive["estimated_p7"].astype(np.float64)
        estimated_v7 = archive["estimated_v7"].astype(np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError(f"invalid point/color arrays for {case_id}")
    if len(points) < 1000 or not np.isfinite(points).all():
        raise ValueError(f"invalid static cloud for {case_id}")
    ground, ground_report = fit_ground(points, estimated_p7)
    ground_z = float(ground_report["ground_top_z_m"])
    if family == "aperture":
        structures, family_report = fit_aperture(points, estimated_p7, ground_z)
    elif family == "deflector":
        structures, family_report = fit_deflector(points, colors)
    elif family == "support_edge":
        structures, family_report = fit_support_edge(points, estimated_p7, ground_z)
    else:
        raise ValueError(f"unsupported family: {family}")
    payload = {
        "schema": "context_rgb8_collision_primitives_v1",
        "status": "EXECUTED",
        "case_id": case_id,
        "family": family,
        "estimated_p7_m": estimated_p7.tolist(),
        "estimated_v7_mps": estimated_v7.tolist(),
        "primitives": [ground, *structures],
        "ground_fit": ground_report,
        "family_fit": family_report,
        "input_point_count": int(len(points)),
        "future_rgb_used": False,
        "gt_state_used": False,
        "blueprint_used": False,
    }
    case_dir = output / "cases" / case_id
    dump_json(case_dir / "primitives.json", payload)
    return {
        "case_id": case_id,
        "family": family,
        "status": "EXECUTED",
        "primitive_count": len(payload["primitives"]),
        "path": str((case_dir / "primitives.json").relative_to(output)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    vision_output = args.vision_output.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    inference_report = load_json(vision_output / "inference_report.json")
    if inference_report.get("status") != "EXECUTED" or inference_report.get("case_count") != 36:
        raise ValueError("vision inference is incomplete")
    records = inference_report["estimates"]["records"]
    rows = []
    for index, record in enumerate(records, start=1):
        try:
            row = fit_case(vision_output, record, output)
        except Exception as error:
            row = {
                "case_id": record["case_id"],
                "family": record["family"],
                "status": "FAILED",
                "error": f"{type(error).__name__}: {error}",
            }
        rows.append(row)
        print(f"PRIMITIVE_{row['status']} {index:02d}/36 {row['case_id']}", flush=True)
    report = {
        "schema": "context_rgb8_collision_primitive_fit_report_v1",
        "status": "EXECUTED" if all(row["status"] == "EXECUTED" for row in rows) else "PARTIAL",
        "case_count": len(rows),
        "executed_count": sum(row["status"] == "EXECUTED" for row in rows),
        "failed_count": sum(row["status"] == "FAILED" for row in rows),
        "records": rows,
        "estimator_inputs": ["RGB-derived metric point cloud", "RGB colors", "family", "estimated p7/v7"],
        "forbidden_inputs": ["GT blueprint", "GT state", "RGB8-RGB48", "future trajectory/contact"],
    }
    dump_json(output / "fit_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
