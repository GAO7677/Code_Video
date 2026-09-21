"""Build the five-question report, per-case tables, and failure figures."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from context_rgb_pybullet_common import dump_json, load_json


PRIMARY_MODES = {
    "A": "A_gt_state_gt_geometry__omega_gt_observed",
    "B": "B_estimated_state_gt_geometry__omega_gt_observed",
    "C": "C_gt_state_estimated_geometry__omega_gt_observed",
    "D": "D_estimated_state_estimated_geometry__omega_gt_observed",
}
FAMILIES = ("aperture", "deflector", "support_edge")
COLORS = {"A": "#ffffff", "B": "#0077b6", "C": "#f4a261", "D": "#d62828"}


def mean(values: list[float]) -> float:
    return float(np.mean(values))


def median(values: list[float]) -> float:
    return float(np.median(values))


def maximum(values: list[float]) -> float:
    return float(np.max(values))


def fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def load_cases(evaluation_root: Path) -> list[dict[str, Any]]:
    paths = sorted((evaluation_root / "cases").glob("*/evaluation.json"))
    cases = [load_json(path) for path in paths]
    if len(cases) != 36 or any(case.get("status") != "EXECUTED" for case in cases):
        raise ValueError(f"expected 36 executed case reports, found {len(cases)}")
    return cases


def state_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in ("all", *FAMILIES):
        rows = cases if family == "all" else [case for case in cases if case["family"] == family]
        values = {
            "position_error_m": [case["state_evaluation"]["position_error_m"] for case in rows],
            "velocity_vector_error_mps": [case["state_evaluation"]["vector_error_mps"] for case in rows],
            "velocity_magnitude_error_mps": [case["state_evaluation"]["magnitude_error_mps"] for case in rows],
            "velocity_direction_error_deg": [
                case["state_evaluation"]["direction_error_deg"] for case in rows
                if case["state_evaluation"]["direction_error_deg"] is not None
            ],
        }
        result[family] = {
            key: {"mean": mean(items), "median": median(items), "max": maximum(items)}
            for key, items in values.items()
        }
    return result


def measurement_errors(cases: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for case in cases:
        row = case["geometry_evaluation"]["measurements"].get(key)
        if row is not None:
            values.append(float(row.get("absolute_error", row.get("l2_error"))))
    return values


def geometry_summary(
    cases: list[dict[str, Any]], primitive_root: Path
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in FAMILIES:
        rows = [case for case in cases if case["family"] == family]
        summary = {
            "primitive_center_error_m": {
                "mean": mean([case["geometry_evaluation"]["mean_center_error_m"] for case in rows]),
                "median": median([case["geometry_evaluation"]["mean_center_error_m"] for case in rows]),
                "max": maximum([case["geometry_evaluation"]["max_center_error_m"] for case in rows]),
            },
            "primitive_full_size_l2_error_m": {
                "mean": mean([case["geometry_evaluation"]["mean_full_size_l2_error_m"] for case in rows]),
                "median": median([case["geometry_evaluation"]["mean_full_size_l2_error_m"] for case in rows]),
            },
            "orientation_error_deg": {
                "mean": mean([case["geometry_evaluation"]["mean_orientation_error_deg"] for case in rows]),
                "max": maximum([
                    max(item["orientation_error_deg"] for item in case["geometry_evaluation"]["per_primitive"])
                    for case in rows
                ]),
            },
            "measurements": {},
        }
        for key in (
            "ground_top_z_m", "front_surface_x_m", "gap_edges_y_m",
            "opening_width_m", "opening_height_m", "top_z_m",
            "gap_edges_m", "gap_width_m",
        ):
            values = measurement_errors(rows, key)
            if values:
                summary["measurements"][key] = {
                    "mean_error": mean(values), "median_error": median(values), "max_error": maximum(values)
                }
        if family == "aperture":
            thickness_errors = []
            for case in rows:
                payload = load_json(
                    primitive_root / "cases" / case["case_id"] / "primitives.json"
                )
                estimate = float(payload["family_fit"]["used_thickness_m"])
                gt_half_x = float(
                    case["geometry_evaluation"]["per_primitive"][0]["gt_canonical"]["half_extents_m"][0]
                )
                thickness_errors.append(abs(estimate - 2.0 * gt_half_x))
            summary["measurements"]["thickness_m"] = {
                "mean_error": mean(thickness_errors),
                "median_error": median(thickness_errors),
                "max_error": maximum(thickness_errors),
            }
        result[family] = summary
    return result


def rollout_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for letter, mode_name in PRIMARY_MODES.items():
        rows = [case["modes"][mode_name] for case in cases]
        valid = [row for row in rows if row["included_in_valid_aggregate"]]
        timing_errors = []
        post_contact = []
        for case, row in zip(cases, rows):
            if not row["included_in_valid_aggregate"]:
                continue
            oracle = case["modes"][PRIMARY_MODES["A"]]["event"]["first_interaction_time_after_rgb7_s"]
            observed = row["event"]["first_interaction_time_after_rgb7_s"]
            if oracle is not None and observed is not None:
                timing_errors.append(abs(float(observed) - float(oracle)))
            post = row["post_contact_velocity_error"]
            if post["status"] == "EXECUTED":
                post_contact.append(float(post["mean_mps"]))
        result[letter] = {
            "valid_case_count": len(valid),
            "invalid_initial_overlap_case_count": len(rows) - len(valid),
            "ADE_mean_valid_m": mean([row["ADE_m"] for row in valid]) if valid else None,
            "FDE_mean_valid_m": mean([row["FDE_m"] for row in valid]) if valid else None,
            "ADE_mean_raw_m": mean([row["ADE_m"] for row in rows]),
            "contact_match_valid_count": sum(bool(row["contact_outcome_match_oracle"]) for row in valid),
            "contact_match_raw_count": sum(bool(row["contact_outcome_match_oracle"]) for row in rows),
            "first_contact_time_MAE_s_when_both_exist": mean(timing_errors) if timing_errors else None,
            "first_contact_comparable_count": len(timing_errors),
            "max_penetration_valid_max_m": maximum([row["event"]["max_penetration_m"] for row in valid]) if valid else None,
            "max_penetration_raw_max_m": maximum([row["event"]["max_penetration_m"] for row in rows]),
            "post_contact_velocity_error_mean_mps": mean(post_contact) if post_contact else None,
            "post_contact_case_count": len(post_contact),
        }
    return result


def omega_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for base, label in (
        ("A_gt_state_gt_geometry", "A_oracle"),
        ("D_estimated_state_estimated_geometry", "D_end_to_end"),
    ):
        policies = {}
        for policy in ("gt_observed", "zero", "pure_rolling_hypothesis"):
            rows = [case["modes"][f"{base}__omega_{policy}"] for case in cases]
            policies[policy] = {
                "ADE_mean_m": mean([row["ADE_m"] for row in rows]),
                "FDE_mean_m": mean([row["FDE_m"] for row in rows]),
                "contact_match_count": sum(bool(row["contact_outcome_match_oracle"]) for row in rows),
            }
        gt_ade = policies["gt_observed"]["ADE_mean_m"]
        for policy in policies:
            policies[policy]["ADE_delta_vs_gt_omega_m"] = policies[policy]["ADE_mean_m"] - gt_ade
        result[label] = policies
    return result


def write_csvs(cases: list[dict[str, Any]], output: Path) -> None:
    state_fields = [
        "case_id", "family", "group_id", "geometry_value", "geometry_units",
        "position_error_m", "velocity_vector_error_mps", "velocity_magnitude_error_mps",
        "velocity_direction_error_deg", "geometry_center_error_m",
        "geometry_full_size_l2_error_m", "geometry_orientation_error_deg",
    ]
    with (output / "state_and_geometry_per_case.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=state_fields)
        writer.writeheader()
        for case in cases:
            state, geometry = case["state_evaluation"], case["geometry_evaluation"]
            writer.writerow({
                "case_id": case["case_id"], "family": case["family"],
                "group_id": case["group_id"], "geometry_value": case["geometry_value"],
                "geometry_units": case["geometry_units"],
                "position_error_m": state["position_error_m"],
                "velocity_vector_error_mps": state["vector_error_mps"],
                "velocity_magnitude_error_mps": state["magnitude_error_mps"],
                "velocity_direction_error_deg": state["direction_error_deg"],
                "geometry_center_error_m": geometry["mean_center_error_m"],
                "geometry_full_size_l2_error_m": geometry["mean_full_size_l2_error_m"],
                "geometry_orientation_error_deg": geometry["mean_orientation_error_deg"],
            })

    mode_fields = [
        "case_id", "family", "combination", "omega_policy", "trajectory_metric_status",
        "ADE_m", "FDE_m", "max_error_m", "contact_outcome", "contact_outcome_match_oracle",
        "first_contact_time_s", "max_penetration_m", "initial_penetration_m",
        "post_contact_velocity_status", "post_contact_velocity_mean_error_mps",
        "wall_seconds", "position_error_m", "velocity_vector_error_mps",
        "geometry_center_error_m", "geometry_full_size_l2_error_m", "geometry_orientation_error_deg",
    ]
    with (output / "per_case_all_modes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=mode_fields)
        writer.writeheader()
        for case in cases:
            for mode in case["modes"].values():
                post = mode["post_contact_velocity_error"]
                writer.writerow({
                    "case_id": case["case_id"], "family": case["family"],
                    "combination": mode["combination"], "omega_policy": mode["omega_policy"],
                    "trajectory_metric_status": mode["trajectory_metric_status"],
                    "ADE_m": mode["ADE_m"], "FDE_m": mode["FDE_m"],
                    "max_error_m": mode["max_error_m"],
                    "contact_outcome": mode["event"]["category"],
                    "contact_outcome_match_oracle": mode["contact_outcome_match_oracle"],
                    "first_contact_time_s": mode["event"]["first_interaction_time_after_rgb7_s"],
                    "max_penetration_m": mode["event"]["max_penetration_m"],
                    "initial_penetration_m": mode["event"]["initial_penetration_m"],
                    "post_contact_velocity_status": post["status"],
                    "post_contact_velocity_mean_error_mps": post.get("mean_mps"),
                    "wall_seconds": mode["wall_seconds"],
                    "position_error_m": case["state_evaluation"]["position_error_m"],
                    "velocity_vector_error_mps": case["state_evaluation"]["vector_error_mps"],
                    "geometry_center_error_m": case["geometry_evaluation"]["mean_center_error_m"],
                    "geometry_full_size_l2_error_m": case["geometry_evaluation"]["mean_full_size_l2_error_m"],
                    "geometry_orientation_error_deg": case["geometry_evaluation"]["mean_orientation_error_deg"],
                })


def project(points: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    world_to_camera = np.asarray(calibration["world_to_camera_3x4"], dtype=np.float64)
    intrinsic = np.asarray(calibration["intrinsic_K"], dtype=np.float64)
    camera = np.column_stack((points, np.ones(len(points)))) @ world_to_camera.T
    uvw = camera @ intrinsic.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    uv[camera[:, 2] <= 1e-5] = np.nan
    return uv


def box_corners(primitive: dict[str, Any], estimated: bool) -> np.ndarray:
    center = np.asarray(primitive["position_m"], dtype=np.float64)
    half = np.asarray(
        primitive["size"]["half_extents_m"] if estimated else primitive["half_extents_m"],
        dtype=np.float64,
    )
    quat = np.asarray(primitive["orientation_xyzw"], dtype=np.float64)
    x, y, z, w = quat
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    rotation = np.asarray([[math.cos(yaw), -math.sin(yaw), 0.0],
                           [math.sin(yaw), math.cos(yaw), 0.0],
                           [0.0, 0.0, 1.0]])
    signs = np.asarray([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float64)
    return signs * half @ rotation.T + center


BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def draw_boxes(ax, case: dict[str, Any], calibration: dict[str, Any]) -> None:
    first_gt = True
    first_est = True
    for row in case["geometry_evaluation"]["per_primitive"]:
        for primitive, estimated, color, style in (
            (row["gt_canonical"], False, "#52b788", "--"),
            (row["estimated"], True, "#ff4dce", "-"),
        ):
            uv = project(box_corners(primitive, estimated), calibration)
            for edge_index, (left, right) in enumerate(BOX_EDGES):
                label = None
                if edge_index == 0 and ((estimated and first_est) or (not estimated and first_gt)):
                    label = "estimated primitive" if estimated else "GT primitive"
                ax.plot(uv[[left, right], 0], uv[[left, right], 1], style,
                        color=color, linewidth=1.1, alpha=0.85, label=label)
            first_est = first_est and not estimated
            first_gt = first_gt and estimated


def load_rollout(evaluation_root: Path, case_id: str, mode_name: str) -> np.ndarray:
    with np.load(evaluation_root / "rollouts" / case_id / f"{mode_name}.npz") as archive:
        return archive["positions"].astype(np.float64)


def failure_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for family in FAMILIES:
        rows = [case for case in cases if case["family"] == family]
        rows.sort(key=lambda case: case["modes"][PRIMARY_MODES["D"]]["ADE_m"], reverse=True)
        selected.append(rows[0])
        mismatches = [
            case for case in rows
            if not case["modes"][PRIMARY_MODES["D"]]["contact_outcome_match_oracle"]
            and case["case_id"] != rows[0]["case_id"]
        ]
        selected.append(mismatches[0] if mismatches else rows[1])
    return selected


def make_failure_figure(
    case: dict[str, Any], input_root: Path, evaluation_root: Path, output_path: Path
) -> None:
    case_id = case["case_id"]
    image = cv2.cvtColor(
        cv2.imread(str(input_root / case_id / "rgb_07.png"), cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )
    calibration = load_json(input_root / case_id / "calibration.json")
    paths = {letter: load_rollout(evaluation_root, case_id, mode) for letter, mode in PRIMARY_MODES.items()}
    p7_gt = np.asarray(case["state_evaluation"]["gt_p7_m"], dtype=np.float64)
    p7_est = np.asarray(case["state_evaluation"]["estimated_p7_m"], dtype=np.float64)
    paths["A"] = np.vstack((p7_gt, paths["A"]))
    paths["B"] = np.vstack((p7_est, paths["B"]))
    paths["C"] = np.vstack((p7_gt, paths["C"]))
    paths["D"] = np.vstack((p7_est, paths["D"]))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.imshow(image)
    draw_boxes(ax, case, calibration)
    for letter in ("A", "B", "D"):
        uv = project(paths[letter], calibration)
        ax.plot(uv[:, 0], uv[:, 1], color="black", linewidth=3.2, alpha=0.5)
        ax.plot(uv[:, 0], uv[:, 1], color=COLORS[letter], linewidth=2.0, label=f"{letter} trajectory")
        ax.scatter(uv[::5, 0], uv[::5, 1], color=COLORS[letter], edgecolor="black", s=20, zorder=5)
    ax.set_xlim(0, image.shape[1] - 1)
    ax.set_ylim(image.shape[0] - 1, 0)
    ax.set_title("RGB7 overlay: future RGB8-RGB48")
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.85)

    ax = axes[0, 1]
    for letter in ("A", "B", "C", "D"):
        mode = case["modes"][PRIMARY_MODES[letter]]
        if letter == "C" and not mode["included_in_valid_aggregate"]:
            continue
        ax.plot(paths[letter][:, 0], paths[letter][:, 1], color=COLORS[letter],
                linewidth=2.0, label=letter)
        ax.scatter(paths[letter][0, 0], paths[letter][0, 1], color=COLORS[letter], s=35)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title("Top-down trajectory")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    time_s = np.arange(1, 42) / 30.0
    for letter in ("A", "B", "C", "D"):
        mode = case["modes"][PRIMARY_MODES[letter]]
        if letter == "C" and not mode["included_in_valid_aggregate"]:
            continue
        ax.plot(time_s, mode["position_error_by_future_frame_m"], color=COLORS[letter],
                linewidth=2.0, label=letter)
    ax.set_xlabel("time after RGB7 (s)")
    ax.set_ylabel("position error (m)")
    ax.set_title("Per-frame trajectory error")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        f"state: |p7-est|={case['state_evaluation']['position_error_m']:.3f} m, "
        f"|v7-est|={case['state_evaluation']['vector_error_mps']:.3f} m/s",
        f"geometry: center={case['geometry_evaluation']['mean_center_error_m']:.3f} m, "
        f"size={case['geometry_evaluation']['mean_full_size_l2_error_m']:.3f} m, "
        f"angle={case['geometry_evaluation']['mean_orientation_error_deg']:.1f} deg",
        "",
        "mode  status                    ADE / FDE       outcome (first time)       max pen",
    ]
    for letter, mode_name in PRIMARY_MODES.items():
        mode = case["modes"][mode_name]
        event = mode["event"]
        timing = event["first_interaction_time_after_rgb7_s"]
        lines.append(
            f"{letter:<5} {mode['trajectory_metric_status']:<25} "
            f"{mode['ADE_m']:.3f}/{mode['FDE_m']:.3f} m  "
            f"{event['category'][:25]:<25} ({fmt(timing, 3)} s)  "
            f"{1000.0 * event['max_penetration_m']:.2f} mm"
        )
    if not case["modes"][PRIMARY_MODES["C"]]["included_in_valid_aggregate"]:
        lines.extend(("", "C is omitted from plots: initial overlap exceeds the fixed 1 mm validity limit."))
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9.5)
    ax.set_title("Error decomposition", loc="left")
    d = case["modes"][PRIMARY_MODES["D"]]
    fig.suptitle(
        f"{case_id} | {case['family']} | D ADE {d['ADE_m']:.3f} m | "
        f"contact {'match' if d['contact_outcome_match_oracle'] else 'MISMATCH'}",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)


def write_gallery(selected: list[dict[str, Any]], output: Path) -> None:
    cards = []
    for case in selected:
        case_id = case["case_id"]
        mode = case["modes"][PRIMARY_MODES["D"]]
        cards.append(
            f'<figure><a href="failure_visualizations/{html.escape(case_id)}.png">'
            f'<img src="failure_visualizations/{html.escape(case_id)}.png" loading="lazy"></a>'
            f'<figcaption>{html.escape(case_id)} | D ADE {mode["ADE_m"]:.3f} m | '
            f'{html.escape(mode["event"]["category"])}</figcaption></figure>'
        )
    document = """<!doctype html><html><head><meta charset="utf-8"><title>Context RGB to PyBullet failures</title>
<style>body{font:14px system-ui;margin:24px;background:#f4f5f7;color:#17191c}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:18px}figure{margin:0;background:white;border:1px solid #d9dde3;border-radius:6px;padding:10px}img{width:100%;height:auto;display:block}figcaption{padding:8px 2px 2px;font-family:ui-monospace,monospace}</style></head><body><h1>Context RGB to PyBullet: failure cases</h1><main>""" + "".join(cards) + "</main></body></html>\n"
    (output / "failure_gallery.html").write_text(document, encoding="utf-8")


def write_commands(output: Path, args: argparse.Namespace) -> None:
    text = f"""# GPU inference (executed on physical GPU 6; GPU 4 was not used)
CUDA_VISIBLE_DEVICES=GPU-7f6fbc40-3594-2c34-8557-422621355ff9 \\
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B \\
/home/gaoya/Code_Video/Context_video/scene_geo_predictor/code/infer_context_rgb_pybullet.py \\
--input-root {args.input_root.resolve()} \\
--output {args.input_root.resolve().parent / 'vision_v1'} \\
--gpu-uuid GPU-7f6fbc40-3594-2c34-8557-422621355ff9

# CPU refinement (executed)
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B \\
/home/gaoya/Code_Video/Context_video/scene_geo_predictor/code/refine_context_rgb_pybullet.py \\
--input-root {args.input_root.resolve()} \\
--source-vision {args.input_root.resolve().parent / 'vision_v1'} \\
--output {args.vision_root.resolve()}

# Primitive fit (executed)
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B \\
/home/gaoya/Code_Video/Context_video/scene_geo_predictor/code/fit_context_collision_primitives.py \\
--vision-output {args.vision_root.resolve()} --output {args.primitive_root.resolve()}

# 36 cases x 4 controls x 3 omega policies (executed)
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=/home/gaoya/Code_Video/Context_video/scene_geo_predictor/code \\
/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B \\
/home/gaoya/Code_Video/Context_video/scene_geo_predictor/code/evaluate_context_rgb_pybullet.py \\
--data-root /data/gaoya/agent-data/outputs/physvideo_next_experiment_20260921_phase1_v5 \\
--vision-output {args.vision_root.resolve()} --primitive-root {args.primitive_root.resolve()} \\
--output {args.evaluation_root.resolve()}
"""
    (output / "commands.txt").write_text(text, encoding="utf-8")


def write_report(summary: dict[str, Any], selected: list[dict[str, Any]], output: Path) -> None:
    state = summary["state"]
    geometry = summary["geometry"]
    rollout = summary["rollout"]
    omega = summary["omega"]
    lines = [
        "# Context RGB → PyBullet 最小链路结论（36-case pilot）",
        "",
        "状态：**EXECUTED**。固定 36 条（每类 12 条），仅 RGB0–RGB7、时间戳、固定相机标定、family 标签和球半径 0.11 m 进入估计；RGB8–RGB48、GT state 和 blueprint 只在视觉/primitive 输出冻结后用于评测。",
        "",
        "## 1. p7/v7 能从 8 帧 context 估到多准？",
        "",
        f"整体 p7 误差均值/中位数/最大值为 **{fmt(state['all']['position_error_m']['mean'])} / {fmt(state['all']['position_error_m']['median'])} / {fmt(state['all']['position_error_m']['max'])} m**；v7 向量误差均值为 **{fmt(state['all']['velocity_vector_error_mps']['mean'])} m/s**，速度大小误差均值 **{fmt(state['all']['velocity_magnitude_error_mps']['mean'])} m/s**，方向误差均值 **{fmt(state['all']['velocity_direction_error_deg']['mean'])}°**。v7 是 8 个 3D center 的鲁棒二次拟合在 RGB7 处的导数，不是 p6/p7 两帧差分。",
        "",
        "| family | p7 mean / median / max (m) | v7 vector mean (m/s) | speed mean (m/s) | direction mean (°) |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        row = state[family]
        lines.append(
            f"| {family} | {fmt(row['position_error_m']['mean'])} / {fmt(row['position_error_m']['median'])} / {fmt(row['position_error_m']['max'])} | "
            f"{fmt(row['velocity_vector_error_mps']['mean'])} | {fmt(row['velocity_magnitude_error_mps']['mean'])} | {fmt(row['velocity_direction_error_deg']['mean'])} |"
        )
    lines.extend((
        "",
        "动态球由 SAM2.1 跟踪，p7 使用标定射线、球形轮廓和已知半径恢复；VGGT 点图用于静态几何。这里依赖固定相机与已知尺寸，不能解读为无标定单目 RGB 的绝对尺度能力。",
        "",
        "## 2. 场景 collision primitives 能估到多准？",
        "",
        "| family | center mean (m) | full-size L2 mean (m) | angle mean (°) | 任务相关几何误差均值 |",
        "|---|---:|---:|---:|---|",
    ))
    ap, de, su = geometry["aperture"], geometry["deflector"], geometry["support_edge"]
    lines.extend((
        f"| aperture | {fmt(ap['primitive_center_error_m']['mean'])} | {fmt(ap['primitive_full_size_l2_error_m']['mean'])} | {fmt(ap['orientation_error_deg']['mean'])} | opening width {fmt(ap['measurements']['opening_width_m']['mean_error'])} m；height {fmt(ap['measurements']['opening_height_m']['mean_error'])} m；front depth {fmt(ap['measurements']['front_surface_x_m']['mean_error'])} m；thickness {fmt(ap['measurements']['thickness_m']['mean_error'])} m |",
        f"| deflector | {fmt(de['primitive_center_error_m']['mean'])} | {fmt(de['primitive_full_size_l2_error_m']['mean'])} | {fmt(de['orientation_error_deg']['mean'])} | ground height {fmt(de['measurements']['ground_top_z_m']['mean_error'])} m |",
        f"| support_edge | {fmt(su['primitive_center_error_m']['mean'])} | {fmt(su['primitive_full_size_l2_error_m']['mean'])} | {fmt(su['orientation_error_deg']['mean'])} | gap edges L2 {fmt(su['measurements']['gap_edges_m']['mean_error'])} m；gap width {fmt(su['measurements']['gap_width_m']['mean_error'])} m；top height {fmt(su['measurements']['top_z_m']['mean_error'])} m |",
        "",
        "36/36 都输出有限 box primitive。主要缺陷是 aperture 的遮挡深度/门高、deflector 的中心和朝向、support_edge 的平台纵深与中心；没有把稠密点云直接交给 Bullet。aperture/deflector 地面和 support 顶面使用 RGB7 可见的球-支撑接触约束，pure point-map 原始值仍保存在 primitive JSON 中。",
        "",
        "## 3. 状态误差和几何误差分别造成多少 future trajectory error？",
        "",
        "GT omega 下的四组结果：",
        "",
        "| 组 | 有效 case | ADE / FDE (m) | contact match | first-contact MAE* (s) | post-contact velocity error (m/s) | max penetration |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ))
    for letter in "ABCD":
        row = rollout[letter]
        lines.append(
            f"| {letter} | {row['valid_case_count']}/36 | {fmt(row['ADE_mean_valid_m'])} / {fmt(row['FDE_mean_valid_m'])} | "
            f"{row['contact_match_valid_count']}/{row['valid_case_count']} | {fmt(row['first_contact_time_MAE_s_when_both_exist'])} ({row['first_contact_comparable_count']} case) | "
            f"{fmt(row['post_contact_velocity_error_mean_mps'])} ({row['post_contact_case_count']} case) | {1000.0 * row['max_penetration_valid_max_m']:.2f} mm |"
        )
    lines.extend((
        "",
        "\* 仅统计 oracle 与该模式都发生 interaction 的 case。A 的 ADE 约 3.6e-8 m，验证 bridge 与原 future 对齐。B−A 表明估计状态单独带来 **0.286 m ADE**。严格 C 中 27/36 在 RGB7 已发生超过固定 1 mm 阈值的初始重叠，因此其轨迹指标无效；剩余 9 条有效 case 的 ADE 为 **0.321 m**。原始 C 数值仍逐 case 保留，但不用于结论。D 全部有效，ADE **0.286 m**；D−B 为 -0.0004 m，仅说明同源状态/几何的坐标偏差会抵消，不能证明几何准确。",
        "",
        "omega 对照表明 pure rolling 不能当真值：",
        "",
        "| 初始条件 | GT omega ADE | zero omega ADE (Δ) | pure-rolling ADE (Δ) |",
        "|---|---:|---:|---:|",
        f"| A oracle | {fmt(omega['A_oracle']['gt_observed']['ADE_mean_m'])} | {fmt(omega['A_oracle']['zero']['ADE_mean_m'])} (+{fmt(omega['A_oracle']['zero']['ADE_delta_vs_gt_omega_m'])}) | {fmt(omega['A_oracle']['pure_rolling_hypothesis']['ADE_mean_m'])} (+{fmt(omega['A_oracle']['pure_rolling_hypothesis']['ADE_delta_vs_gt_omega_m'])}) |",
        f"| D estimated | {fmt(omega['D_end_to_end']['gt_observed']['ADE_mean_m'])} | {fmt(omega['D_end_to_end']['zero']['ADE_mean_m'])} (+{fmt(omega['D_end_to_end']['zero']['ADE_delta_vs_gt_omega_m'])}) | {fmt(omega['D_end_to_end']['pure_rolling_hypothesis']['ADE_mean_m'])} (+{fmt(omega['D_end_to_end']['pure_rolling_hypothesis']['ADE_delta_vs_gt_omega_m'])}) |",
        "",
        "## 4. estimated state + estimated geometry 下，PyBullet 是否仍然可用？",
        "",
        f"**可作为粗轨迹 baseline，但还不能作为高精度预测器。** D 在 36/36 上数值有效，ADE/FDE 为 **{fmt(rollout['D']['ADE_mean_valid_m'])}/{fmt(rollout['D']['FDE_mean_valid_m'])} m**，contact outcome 为 **{rollout['D']['contact_match_valid_count']}/36（{100.0 * rollout['D']['contact_match_valid_count'] / 36.0:.1f}%）**。它适合粗略运动趋势和候选接触筛选，不适合厘米级轨迹、可靠首次接触时刻或精细反弹速度。",
        "",
        "失败图（RGB7 overlay、primitive、俯视轨迹、逐帧误差）：",
        "",
    ))
    for case in selected:
        mode = case["modes"][PRIMARY_MODES["D"]]
        lines.append(
            f"- [{case['case_id']}](failure_visualizations/{case['case_id']}.png)：D ADE {mode['ADE_m']:.3f} m，"
            f"contact {'match' if mode['contact_outcome_match_oracle'] else 'mismatch'}。"
        )
    lines.extend((
        "",
        "完整交付：`per_case_all_modes.csv`（432 行）、`state_and_geometry_per_case.csv`、`failure_gallery.html`、`summary.json`、`commands.txt`。视觉阶段使用物理 GPU 6（UUID `GPU-7f6fbc40-3594-2c34-8557-422621355ff9`），未使用 GPU 4；VGGT 峰值 6.36 GiB、SAM2 峰值 1.51 GiB。36 条 GPU 阶段实测 674.15 s（含 VGGT 155.12 s 冷加载），CPU refinement 74.30 s，432 个 Bullet rollout 8.08 s。没有训练 Predictor、没有加载 DiT/Utonia。",
        "",
        "## 5. 当前主要瓶颈是 motion estimation 还是 geometry estimation？",
        "",
        "**若指标是连续 ADE，主要瓶颈是 motion estimation；若指标是接触语义，geometry 与坐标配准同样关键。** 证据是 B 已产生 0.286 m ADE，而 D 相对 B 几乎不变；但 contact match 从 B 的 21/36 提升到 D 的 30/36，说明同源几何恢复了不少接触关系。同时严格 C 有 27/36 初始重叠，暴露出当前最大的工程问题是 state 与 geometry 的绝对坐标/支撑面高度耦合。下一步应优先改善 p7 深度/尺度与 v7 大小，再做统一场景坐标配准和更可靠的有限面尺寸估计；不需要先训练新的动力学 Predictor。",
        "",
    ))
    (output / "context_rgb_to_pybullet_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    cases = load_cases(args.evaluation_root.resolve())
    selected = failure_cases(cases)
    summary = {
        "schema": "context_rgb_to_pybullet_delivery_summary_v1",
        "status": "EXECUTED",
        "case_count": len(cases),
        "rollout_count": sum(len(case["modes"]) for case in cases),
        "state": state_summary(cases),
        "geometry": geometry_summary(cases, args.primitive_root.resolve()),
        "rollout": rollout_summary(cases),
        "omega": omega_summary(cases),
        "failure_cases": [case["case_id"] for case in selected],
        "input_isolation": {
            "estimator_allowed": ["RGB0-RGB7", "timestamps", "fixed camera calibration", "family", "fixed sphere radius"],
            "evaluation_only": ["RGB8-RGB48", "GT state", "GT blueprint", "future contact/interaction"],
            "future_or_gt_used_for_estimation": False,
        },
    }
    dump_json(output / "summary.json", summary)
    write_csvs(cases, output)
    for case in selected:
        make_failure_figure(
            case, args.input_root.resolve(), args.evaluation_root.resolve(),
            output / "failure_visualizations" / f"{case['case_id']}.png",
        )
    write_gallery(selected, output)
    write_commands(output, args)
    write_report(summary, selected, output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--vision-root", type=Path, required=True)
    parser.add_argument("--primitive-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({
        "status": result["status"],
        "case_count": result["case_count"],
        "rollout_count": result["rollout_count"],
        "failure_cases": result["failure_cases"],
    }, indent=2))
