#!/usr/bin/env python3
"""Re-evaluate ball/block xSSC tracks with simulator-derived object masks."""
from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
DEFAULT_ANALYSIS = Path(
    "/data/gaoya/agent-data/outputs/"
    "xssc_object_slot_separation_cases_dinov3_latest/ball_block_parameter_pairs"
)
DEFAULT_SIMULATOR = Path(
    "/home/gaoya/Code_Video/Code_data/Code_try0526/physics_sim/simulate_ball_block.py"
)
ROLE_NAMES = ("ball", "block")
ROLE_COLORS_RGB = np.asarray([[239, 68, 68], [34, 211, 238]], dtype=np.uint8)
PAIR_METRICS = (
    "static_distance",
    "dynamic_track_distance",
    "d_adj_relative_l2",
    "frequency_js",
    "centroid_rmse",
    "area_curve_relative_l2",
)
DYNAMIC_METRICS = (
    "dynamic_track_distance",
    "d_adj_relative_l2",
    "frequency_js",
    "centroid_rmse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--simulator", type=Path, default=DEFAULT_SIMULATOR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force-masks", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def import_simulator(path: Path):
    spec = importlib.util.spec_from_file_location("ball_block_simulator", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def letterbox_mask(mask: np.ndarray, size: int = 256) -> np.ndarray:
    source_h, source_w = mask.shape
    scale = min(size / source_w, size / source_h)
    width = max(1, round(source_w * scale))
    height = max(1, round(source_h * scale))
    left = (size - width) // 2
    top = (size - height) // 2
    output = np.zeros((size, size), dtype=np.float32)
    output[top : top + height, left : left + width] = cv2.resize(
        mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA
    )
    return output


def downsample_mask(mask: np.ndarray, grid: int = 16) -> np.ndarray:
    return cv2.resize(mask, (grid, grid), interpolation=cv2.INTER_AREA).astype(np.float32)


def render_role_masks(sim, scenario, expected_frames: int = 150) -> np.ndarray:
    p = sim.p
    p.resetSimulation()
    p.setAdditionalSearchPath(sim.pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(
        fixedTimeStep=1.0 / sim.SIM_HZ,
        numSolverIterations=100,
        numSubSteps=1,
    )
    ball_r, ball_z = 0.18, 0.20
    block_h = (0.25, 0.20, 0.30)
    ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_r)
    ball_id = p.createMultiBody(
        scenario.ball_mass, ball_shape, basePosition=(-1.0, 0.0, ball_z)
    )
    block_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=block_h)
    block_id = p.createMultiBody(
        1.5, block_shape, basePosition=(0.3, 0.0, block_h[2])
    )
    p.changeDynamics(
        ball_id,
        -1,
        restitution=scenario.restitution,
        lateralFriction=scenario.lateral_friction,
        spinningFriction=0.003,
        linearDamping=0.03,
        angularDamping=0.03,
    )
    p.changeDynamics(
        block_id,
        -1,
        restitution=scenario.restitution,
        lateralFriction=scenario.lateral_friction,
        spinningFriction=0.008,
        linearDamping=0.06,
        angularDamping=0.06,
        activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
    )
    p.resetBaseVelocity(ball_id, linearVelocity=[3.5, 0.0, 1.8])
    for _ in range(10):
        p.stepSimulation()

    renderer = sim.SceneRenderer()
    masks = []
    record_every = sim.SIM_HZ // sim.DEFAULT_FPS
    for step in range(int(sim.DEFAULT_SIM_DURATION * sim.SIM_HZ)):
        p.stepSimulation()
        if step % record_every:
            continue
        ball_pose = p.getBasePositionAndOrientation(ball_id)
        block_pose = p.getBasePositionAndOrientation(block_id)
        renderer.set_ball(*ball_pose, ball_r)
        renderer.set_block(*block_pose, block_h)
        colors = {
            renderer.ball_node: np.asarray([255, 0, 0], dtype=np.uint8),
            renderer.block_node: np.asarray([0, 255, 0], dtype=np.uint8),
        }
        segmentation, _ = renderer.renderer.render(
            renderer.scene,
            flags=sim.RenderFlags.SEG,
            seg_node_map=colors,
        )
        role_masks = []
        for color in colors.values():
            full = np.all(segmentation == color[None, None, :], axis=-1)
            role_masks.append(downsample_mask(letterbox_mask(full)))
        masks.append(role_masks)
    renderer.cleanup()
    if len(masks) != expected_frames:
        raise RuntimeError(f"Expected {expected_frames} masks, rendered {len(masks)}")
    masks_array = np.asarray(masks, dtype=np.float32)
    visible_frames = (masks_array.sum(axis=(-1, -2)) > 0).sum(axis=0)
    if np.any(visible_frames == 0):
        raise RuntimeError(f"Empty role mask sequence: visible_frames={visible_frames.tolist()}")
    return masks_array


def prepare_role_masks(
    simulator_path: Path,
    cases: list[dict[str, Any]],
    output_dir: Path,
    force: bool,
) -> dict[str, np.ndarray]:
    sim = import_simulator(simulator_path)
    scenario_by_name = {scenario.name: scenario for scenario in sim.SCENARIOS}
    mask_dir = output_dir / "role_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    connected = False
    try:
        for case in cases:
            scenario = case["scenario"]
            path = mask_dir / f"{scenario}.npz"
            if path.is_file() and not force:
                result[scenario] = np.load(path)["masks"].astype(np.float32)
                continue
            if not connected:
                sim.p.connect(sim.p.DIRECT)
                connected = True
            print(f"[mask] {scenario}", flush=True)
            masks = render_role_masks(sim, scenario_by_name[scenario])
            np.savez_compressed(path, masks=masks.astype(np.float16))
            result[scenario] = masks
    finally:
        if connected:
            sim.p.disconnect()
    return result


def role_slot_assignment(attention: np.ndarray, role_masks: np.ndarray) -> dict[str, Any]:
    # attention: [T,S,H,W], role_masks: [T,2,H,W]. Scores are mean slot mass
    # on GT object pixels. Hungarian assignment keeps ball/block identities distinct.
    slot_mass = attention.sum(axis=(-1, -2)) + 1.0e-8
    raw_object_mass = role_masks.sum(axis=(-1, -2))
    object_mass = raw_object_mass + 1.0e-8
    visible = raw_object_mass > 0.25
    intersection = np.einsum("tshw,trhw->trs", attention, role_masks)
    recall_per_frame = intersection / object_mass[:, :, None]
    precision_per_frame = intersection / slot_mass[:, None, :]
    visible_count = np.maximum(visible.sum(axis=0), 1)
    recall = (recall_per_frame * visible[:, :, None]).sum(axis=0) / visible_count[:, None]
    precision = (precision_per_frame * visible[:, :, None]).sum(axis=0) / visible_count[:, None]
    f1 = 2.0 * recall * precision / np.maximum(recall + precision, 1.0e-8)
    rows, cols = linear_sum_assignment(-recall)
    selected = np.full(2, -1, dtype=np.int64)
    selected[rows] = cols
    details = []
    for role_index, slot_index in enumerate(selected):
        details.append(
            {
                "role": ROLE_NAMES[role_index],
                "slot": int(slot_index),
                "soft_recall": float(recall[role_index, slot_index]),
                "soft_precision": float(precision[role_index, slot_index]),
                "soft_f1": float(f1[role_index, slot_index]),
                "visible_frames": int(visible_count[role_index]),
                "best_unconstrained_slot": int(np.argmax(recall[role_index])),
            }
        )
    return {
        "selected": selected,
        "details": details,
        "recall_matrix": recall,
        "precision_matrix": precision,
        "f1_matrix": f1,
        "same_unconstrained_slot": bool(np.argmax(recall[0]) == np.argmax(recall[1])),
    }


def role_overlay(
    input_video: Path,
    attention: np.ndarray,
    selected: np.ndarray,
    role_masks: np.ndarray,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(input_video))
    frames = []
    labels = attention.argmax(axis=1)
    index = 0
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        for role_index, slot_index in enumerate(selected):
            slot_mask = cv2.resize(
                (labels[index] == slot_index).astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            frame[slot_mask] = (
                0.50 * frame[slot_mask] + 0.50 * ROLE_COLORS_RGB[role_index]
            ).astype(np.uint8)
            gt = cv2.resize(
                role_masks[index, role_index],
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            contour = cv2.morphologyEx(
                (gt >= 0.25).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
            ).astype(bool)
            frame[contour] = ROLE_COLORS_RGB[role_index]
        cv2.putText(
            frame,
            f"ball=S{selected[0]}  block=S{selected[1]}",
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        frames.append(frame)
        index += 1
    capture.release()
    return frames


def write_video(path: Path, frames: list[np.ndarray], fps: float = 60.0) -> None:
    if not frames:
        raise ValueError(f"No frames for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {path}")


def relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    numerator = float(np.linalg.norm(left - right))
    denominator = 0.5 * float(np.linalg.norm(left) + np.linalg.norm(right))
    return numerator / max(denominator, 1.0e-8)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return 0.0
    return float(1.0 - np.dot(left.reshape(-1), right.reshape(-1)) / denominator)


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64) / max(float(left.sum()), 1.0e-12)
    right = right.astype(np.float64) / max(float(right.sum()), 1.0e-12)
    middle = 0.5 * (left + right)
    kl_left = np.sum(np.where(left > 0, left * np.log((left + 1e-12) / (middle + 1e-12)), 0.0))
    kl_right = np.sum(np.where(right > 0, right * np.log((right + 1e-12) / (middle + 1e-12)), 0.0))
    return float(0.5 * (kl_left + kl_right))


def attention_centroids(attention: np.ndarray) -> np.ndarray:
    _, _, height, width = attention.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    mass = attention.sum(axis=(-1, -2)) + 1.0e-8
    x = (attention * xx[None, None]).sum(axis=(-1, -2)) / mass
    y = (attention * yy[None, None]).sum(axis=(-1, -2)) / mass
    return np.stack([x, y], axis=-1)


def load_role_bundle(arrays_path: Path, role_masks: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays = np.load(arrays_path)
    slots = arrays["slots"].astype(np.float32)
    attention = arrays["attention"].astype(np.float32)
    assignment = role_slot_assignment(attention, role_masks)
    selected = assignment["selected"]
    role_slots = slots[:, selected]
    static = role_slots.mean(axis=0)
    residual = role_slots - static[None]
    d_adj = np.linalg.norm(np.diff(role_slots, axis=0), axis=-1) / math.sqrt(slots.shape[-1])
    centered = residual * np.hanning(len(residual))[:, None, None]
    power = (np.abs(np.fft.rfft(centered, axis=0)) ** 2)[1:].sum(axis=-1).T
    labels = attention.argmax(axis=1)
    area = np.stack([(labels == slot).mean(axis=(1, 2)) for slot in selected], axis=1)
    centroids = attention_centroids(attention)[:, selected]
    bundle = {
        "selected": selected,
        "static": static,
        "residual": residual,
        "d_adj": d_adj,
        "power": power,
        "area": area,
        "centroids": centroids,
        "attention": attention,
    }
    return bundle, assignment


def compare_role_tracks(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    per_role = []
    for role_index, role in enumerate(ROLE_NAMES):
        left_static = left["static"][role_index]
        right_static = right["static"][role_index]
        row = {
            "role": role,
            "static_distance": cosine_distance(left_static, right_static),
            "dynamic_track_distance": cosine_distance(
                left["residual"][:, role_index], right["residual"][:, role_index]
            ),
            "d_adj_relative_l2": relative_l2(
                left["d_adj"][:, role_index], right["d_adj"][:, role_index]
            ),
            "frequency_js": js_divergence(left["power"][role_index], right["power"][role_index]),
            "centroid_rmse": float(
                np.sqrt(np.mean((left["centroids"][:, role_index] - right["centroids"][:, role_index]) ** 2)) / 16.0
            ),
            "area_curve_relative_l2": relative_l2(
                left["area"][:, role_index], right["area"][:, role_index]
            ),
        }
        per_role.append(row)
    return {
        **{metric: float(np.mean([row[metric] for row in per_role])) for metric in PAIR_METRICS},
        "per_role": per_role,
    }


def parameter_gaps(left: dict[str, float], right: dict[str, float]) -> dict[str, Any]:
    e_gap = abs(float(left["restitution"]) - float(right["restitution"]))
    mu_gap = abs(float(left["lateral_friction"]) - float(right["lateral_friction"]))
    mass_gap = abs(math.log10(float(left["ball_mass_kg"])) - math.log10(float(right["ball_mass_kg"])))
    changed = [name for name, value in (("restitution", e_gap), ("friction", mu_gap), ("mass", mass_gap)) if value > 1e-12]
    return {
        "restitution_gap": e_gap,
        "friction_gap": mu_gap,
        "log10_mass_gap": mass_gap,
        "normalized_parameter_distance": math.sqrt(
            (e_gap / 0.6) ** 2 + (mu_gap / 0.9) ** 2 + (mass_gap / math.log10(50.0)) ** 2
        ),
        "axis": changed[0] if len(changed) == 1 else "mixed",
    }


def rank01(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    if len(values_array) <= 1:
        return np.ones_like(values_array)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=np.float64)
    ranks[order] = np.arange(len(values_array), dtype=np.float64)
    return ranks / (len(values_array) - 1)


def correlation(left: list[float], right: list[float]) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def plot_heatmaps(path: Path, scenarios: list[str], pairs: list[dict[str, Any]], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (*DYNAMIC_METRICS, "dynamic_composite", "raft_relative_l2")
    index = {name: position for position, name in enumerate(scenarios)}
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axis_plot, metric in zip(axes.flat, metrics):
        matrix = np.zeros((len(scenarios), len(scenarios)), dtype=np.float64)
        for row in pairs:
            i, j = index[row["left"]], index[row["right"]]
            matrix[i, j] = matrix[j, i] = row[metric]
        image = axis_plot.imshow(matrix, cmap="viridis", aspect="equal")
        axis_plot.set_title(metric)
        axis_plot.set_xticks(range(len(scenarios)), scenarios, rotation=45, ha="right", fontsize=7)
        axis_plot.set_yticks(range(len(scenarios)), scenarios, fontsize=7)
        figure.colorbar(image, ax=axis_plot, fraction=0.046, pad=0.04)
    figure.suptitle(title)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def aggregate(
    analysis_dir: Path,
    output_dir: Path,
    source: dict[str, Any],
    role_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    cases = source["cases"]
    scenario_order = [case["scenario"] for case in cases]
    parameters = {case["scenario"]: case["parameters"] for case in cases}
    raft = {
        scenario: np.load(analysis_dir / "inputs" / scenario / "raft_flow_16.npz")["flow"].astype(np.float32)
        for scenario in scenario_order
    }
    models = []
    for model_source in source["models"]:
        model = model_source["model"]
        safe_name = model["safe_name"]
        bundles = {}
        case_records = []
        for case in model_source["cases"]:
            scenario = case["scenario"]
            arrays_path = analysis_dir / case["assets"]["arrays"]
            bundle, assignment = load_role_bundle(arrays_path, role_masks[scenario])
            bundles[scenario] = bundle
            case_dir = output_dir / "models" / safe_name / scenario
            video_path = case_dir / "gt_aligned_object_slots.mp4"
            frames = role_overlay(
                analysis_dir / "inputs" / scenario / "xssc_input_150f.mp4",
                bundle["attention"],
                bundle["selected"],
                role_masks[scenario],
            )
            write_video(video_path, frames)
            np.savez_compressed(
                case_dir / "role_assignment.npz",
                selected_slots=bundle["selected"],
                recall_matrix=assignment["recall_matrix"].astype(np.float32),
                precision_matrix=assignment["precision_matrix"].astype(np.float32),
                f1_matrix=assignment["f1_matrix"].astype(np.float32),
            )
            case_records.append(
                {
                    "scenario": scenario,
                    "parameters": case["parameters"],
                    "selected_slots": bundle["selected"].tolist(),
                    "assignment": assignment["details"],
                    "same_unconstrained_slot": assignment["same_unconstrained_slot"],
                    "assets": {
                        "video": str(video_path.relative_to(output_dir)),
                        "arrays": str((case_dir / "role_assignment.npz").relative_to(output_dir)),
                    },
                }
            )
        pairs = []
        for left_index, left in enumerate(scenario_order):
            for right in scenario_order[left_index + 1 :]:
                row = {
                    "left": left,
                    "right": right,
                    **parameter_gaps(parameters[left], parameters[right]),
                    **compare_role_tracks(bundles[left], bundles[right]),
                    "raft_relative_l2": relative_l2(raft[left], raft[right]),
                }
                pairs.append(row)
        ranks = np.stack([rank01([row[metric] for row in pairs]) for metric in DYNAMIC_METRICS], axis=1)
        for row, value in zip(pairs, ranks.mean(axis=1)):
            row["dynamic_composite"] = float(value)
        correlations = {}
        for scope in ("all", "restitution", "friction", "mass"):
            selected_pairs = pairs if scope == "all" else [row for row in pairs if row["axis"] == scope]
            correlations[scope] = {
                "pair_count": len(selected_pairs),
                **{
                    metric: {
                        "parameter_rho": correlation(
                            [row["normalized_parameter_distance"] for row in selected_pairs],
                            [row[metric] for row in selected_pairs],
                        ),
                        "raft_rho": correlation(
                            [row["raft_relative_l2"] for row in selected_pairs],
                            [row[metric] for row in selected_pairs],
                        ),
                    }
                    for metric in (*PAIR_METRICS, "dynamic_composite")
                },
            }
        heatmap = output_dir / "models" / safe_name / "pair_heatmaps.png"
        plot_heatmaps(heatmap, scenario_order, pairs, model["short_name"])
        models.append(
            {
                "model": model,
                "cases": case_records,
                "pairs": pairs,
                "correlations": correlations,
                "heatmap": str(heatmap.relative_to(output_dir)),
            }
        )
        print(f"[aggregate] {model['short_name']}", flush=True)
    result = {
        "dataset_root": source["dataset_root"],
        "cases": cases,
        "models": models,
        "method": {
            "frames": 150,
            "preprocess": source["preprocess"],
            "role_masks": "deterministic PyBullet replay plus Pyrender SEG at the original camera, letterboxed and area-downsampled to 16x16",
            "slot_binding": "Hungarian assignment maximizing 150-frame mean soft-attention recall on GT ball/block masks",
            "cross_video_matching": "identity aligned: ball-to-ball and block-to-block",
            "purpose_of_gt": "evaluation-only slot identity binding; xSSC inputs and forward passes are unchanged",
        },
    }
    write_json(output_dir / "metadata.json", result)
    build_html(output_dir, result)
    return result


def rho_text(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:+.3f}"


def load_selected_role_slots(
    analysis_dir: Path,
    model: dict[str, Any],
    case: dict[str, Any],
) -> np.ndarray:
    arrays_path = (
        analysis_dir
        / "models"
        / model["safe_name"]
        / case["scenario"]
        / "slots_attention.npz"
    )
    arrays = np.load(arrays_path)
    slots = arrays["slots"].astype(np.float32)
    return slots[:, np.asarray(case["selected_slots"], dtype=np.int64)]


def smooth_curve(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return values.copy()
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    result = np.empty_like(values, dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = (cumulative[index + 1] - cumulative[start]) / (index - start + 1)
    return result.astype(values.dtype, copy=False)


def plot_control_dynamic_curves(
    path: Path,
    title: str,
    labels: list[str],
    curves: np.ndarray,
    fps: float = 60.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#22d3ee", "#f59e0b", "#a78bfa", "#34d399")
    times = np.arange(curves.shape[1], dtype=np.float32) / fps
    figure, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, constrained_layout=True)
    for role_index, (axis_plot, role) in enumerate(zip(axes, ROLE_NAMES)):
        for case_index, label in enumerate(labels):
            curve = curves[case_index, :, role_index]
            color = colors[case_index % len(colors)]
            axis_plot.plot(times, curve, color=color, alpha=0.18, linewidth=0.8)
            axis_plot.plot(
                times,
                smooth_curve(curve),
                color=color,
                linewidth=1.8,
                label=label,
            )
        axis_plot.axhline(0.0, color="#94a3b8", linewidth=0.8, linestyle="--")
        axis_plot.set_ylabel(f"{role} dynamic deviation RMS")
        axis_plot.grid(alpha=0.18)
        axis_plot.legend(ncol=min(4, len(labels)), fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time (seconds)")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def checkpoint_tensor_by_suffix(state: dict[str, Any], suffix: str):
    matches = [(key, value) for key, value in state.items() if str(key).endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one checkpoint tensor ending in {suffix}, found {[key for key, _ in matches]}")
    return matches[0][1]


def decoder_dynamic_ratio(config_path: Path) -> float:
    text = config_path.read_text(encoding="utf-8")
    match = re.search(r"^decoder_dynamic_ratio\s*=\s*([0-9.]+)", text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"decoder_dynamic_ratio not found in {config_path}")
    ratio = float(match.group(1))
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"Invalid decoder_dynamic_ratio={ratio}")
    return ratio


def prepare_decoder_partition_features(
    output_dir: Path,
    model: dict[str, Any],
    role_slots: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    checkpoint = Path(model["xssc_checkpoint"]).expanduser().resolve()
    config = Path(model["xssc_config"]).expanduser().resolve()
    feature_dir = output_dir / "decoder_features"
    metadata_path = feature_dir / "metadata.json"
    ratio = decoder_dynamic_ratio(config)
    expected_metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_size": checkpoint.stat().st_size,
        "config": str(config),
        "dynamic_ratio": ratio,
        "projection": "decode.project2 Linear(no bias) + LayerNorm",
    }
    cached_metadata = None
    if metadata_path.is_file():
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache_valid = cached_metadata is not None and all(
        cached_metadata.get(key) == value for key, value in expected_metadata.items()
    )
    cached_paths = {scenario: feature_dir / f"{scenario}.npz" for scenario in role_slots}
    if cache_valid and all(path.is_file() for path in cached_paths.values()):
        features = {}
        for scenario, path in cached_paths.items():
            with np.load(path) as item:
                features[scenario] = {
                    "static": item["static"].astype(np.float32),
                    "dynamic": item["dynamic"].astype(np.float32),
                }
        return features, cached_metadata

    print(f"[decoder-project2] loading {checkpoint}", flush=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    linear_weight = checkpoint_tensor_by_suffix(state, "decode.project2.0.weight").float()
    norm_weight = checkpoint_tensor_by_suffix(state, "decode.project2.1.weight").float()
    norm_bias = checkpoint_tensor_by_suffix(state, "decode.project2.1.bias").float()
    decoder_dim = int(linear_weight.shape[0])
    slot_dim = int(linear_weight.shape[1])
    dynamic_dim = int(decoder_dim * ratio)
    static_dim = decoder_dim - dynamic_dim
    if norm_weight.shape != (decoder_dim,) or norm_bias.shape != (decoder_dim,):
        raise RuntimeError("Decoder project2 LayerNorm shape mismatch")
    features = {}
    feature_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for scenario, slots in role_slots.items():
            if slots.shape[-1] != slot_dim:
                raise RuntimeError(f"Slot dim mismatch for {scenario}: {slots.shape[-1]} != {slot_dim}")
            tensor = torch.from_numpy(slots.astype(np.float32, copy=False))
            projected = F.linear(tensor, linear_weight)
            projected = F.layer_norm(
                projected,
                (decoder_dim,),
                weight=norm_weight,
                bias=norm_bias,
                eps=1.0e-5,
            ).numpy()
            item = {
                "static": projected[..., :static_dim].astype(np.float32),
                "dynamic": projected[..., static_dim:].astype(np.float32),
            }
            features[scenario] = item
            np.savez_compressed(cached_paths[scenario], **item)
    del state
    metadata = {
        **expected_metadata,
        "slot_dim": slot_dim,
        "decoder_dim": decoder_dim,
        "static_dim": static_dim,
        "dynamic_dim": dynamic_dim,
        "role_order": list(ROLE_NAMES),
        "feature_shape": {
            "static": [150, 2, static_dim],
            "dynamic": [150, 2, dynamic_dim],
        },
    }
    write_json(metadata_path, metadata)
    return features, metadata


def plot_decoder_partition_curves(
    path: Path,
    title: str,
    labels: list[str],
    curves: np.ndarray,
    fps: float = 60.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#22d3ee", "#f59e0b", "#a78bfa", "#34d399")
    partitions = ("decoder-static", "decoder-dynamic")
    times = np.arange(curves.shape[1], dtype=np.float32) / fps
    figure, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, constrained_layout=True)
    for partition_index, partition in enumerate(partitions):
        for role_index, role in enumerate(ROLE_NAMES):
            axis_plot = axes[partition_index, role_index]
            for case_index, label in enumerate(labels):
                curve = curves[case_index, :, partition_index, role_index]
                color = colors[case_index % len(colors)]
                axis_plot.plot(times, curve, color=color, alpha=0.18, linewidth=0.8)
                axis_plot.plot(
                    times,
                    smooth_curve(curve),
                    color=color,
                    linewidth=1.8,
                    label=label,
                )
            axis_plot.axhline(0.0, color="#94a3b8", linewidth=0.8, linestyle="--")
            axis_plot.set_title(f"{partition} · {role}")
            axis_plot.set_ylabel("initial-aligned RMS / channel")
            axis_plot.grid(alpha=0.18)
            axis_plot.legend(ncol=1, fontsize=7, loc="upper right")
    for axis_plot in axes[-1]:
        axis_plot.set_xlabel("time (seconds)")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def build_html(output_dir: Path, result: dict[str, Any]) -> None:
    model_result = next(
        item for item in result["models"]
        if item["model"]["name"] == "dinov3_movic_step044000"
    )
    model = model_result["model"]
    baseline = "e07_mu05_m1"
    case_map = {case["scenario"]: case for case in model_result["cases"]}
    pair_map = {
        frozenset((pair["left"], pair["right"])): pair
        for pair in model_result["pairs"]
    }
    groups = (
        (
            "恢复系数 restitution",
            "只改变恢复系数 e；固定 μ=0.5、ball mass=1.0kg。",
            ("e03_mu05_m1", "e05_mu05_m1", baseline, "e09_mu05_m1"),
            "restitution",
        ),
        (
            "摩擦系数 friction",
            "只改变横向摩擦系数 μ；固定 e=0.7、ball mass=1.0kg。",
            ("e07_mu01_m1", baseline, "e07_mu10_m1"),
            "friction",
        ),
        (
            "球质量 mass",
            "只改变 ball mass；固定 e=0.7、μ=0.5。",
            ("e07_mu05_m01", baseline, "e07_mu05_m5"),
            "mass",
        ),
    )
    analysis_dir = output_dir.parent
    role_slots = {
        scenario: load_selected_role_slots(analysis_dir, model, case)
        for scenario, case in case_map.items()
    }
    baseline_role_slots = role_slots[baseline]
    baseline_static = baseline_role_slots.mean(axis=0)
    baseline_residual = baseline_role_slots - baseline_static[None]
    baseline_initial_aligned = baseline_role_slots - baseline_role_slots[:1]
    decoder_features, decoder_metadata = prepare_decoder_partition_features(
        output_dir,
        model,
        role_slots,
    )
    decoder_partitions = ("static", "dynamic")
    baseline_decoder_initial = {
        partition: decoder_features[baseline][partition] - decoder_features[baseline][partition][:1]
        for partition in decoder_partitions
    }

    def parameter_title(case: dict[str, Any]) -> str:
        values = case["parameters"]
        return (
            f"e={values['restitution']:.1f} · μ={values['lateral_friction']:.1f} · "
            f"m={values['ball_mass_kg']:g}kg"
        )

    group_sections = []
    for title, description, scenarios, axis in groups:
        videos = []
        rows = []
        dynamic_curves = []
        mean_centered_curves = []
        decoder_curves = []
        curve_labels = []
        for scenario in scenarios:
            case = case_map[scenario]
            assignment = {item["role"]: item for item in case["assignment"]}
            current_slots = role_slots[scenario]
            current_static = current_slots.mean(axis=0)
            current_residual = current_slots - current_static[None]
            mean_centered_curve = np.linalg.norm(
                current_residual - baseline_residual,
                axis=-1,
            ) / math.sqrt(current_slots.shape[-1])
            current_initial_aligned = current_slots - current_slots[:1]
            curve = np.linalg.norm(
                current_initial_aligned - baseline_initial_aligned,
                axis=-1,
            ) / math.sqrt(current_slots.shape[-1])
            dynamic_curves.append(curve.astype(np.float32))
            mean_centered_curves.append(mean_centered_curve.astype(np.float32))
            decoder_case_curves = []
            for partition in decoder_partitions:
                current_decoder = decoder_features[scenario][partition]
                current_decoder_initial = current_decoder - current_decoder[:1]
                decoder_case_curves.append(
                    (
                        np.linalg.norm(
                            current_decoder_initial - baseline_decoder_initial[partition],
                            axis=-1,
                        )
                        / math.sqrt(current_decoder.shape[-1])
                    ).astype(np.float32)
                )
            decoder_curves.append(np.stack(decoder_case_curves, axis=1))
            curve_labels.append(parameter_title(case))
            baseline_tag = "<span class='baseline-tag'>baseline</span>" if scenario == baseline else ""
            videos.append(
                f"<figure><h3>{parameter_title(case)} {baseline_tag}</h3>"
                f"<p class='scenario'>{html.escape(scenario)}</p>"
                f"<video src='{html.escape(case['assets']['video'])}' controls muted preload='none'></video>"
                f"<figcaption>ball=S{case['selected_slots'][0]} recall={assignment['ball']['soft_recall']:.3f}; "
                f"block=S{case['selected_slots'][1]} recall={assignment['block']['soft_recall']:.3f}</figcaption></figure>"
            )
            if scenario == baseline:
                values = {metric: 0.0 for metric in PAIR_METRICS}
                roles = {role: {"dynamic_track_distance": 0.0} for role in ROLE_NAMES}
                raft = 0.0
                static_by_role = np.zeros(2, dtype=np.float32)
            else:
                pair = pair_map[frozenset((baseline, scenario))]
                values = pair
                roles = {item["role"]: item for item in pair["per_role"]}
                raft = pair["raft_relative_l2"]
                static_by_role = np.asarray(
                    [
                        cosine_distance(current_static[role_index], baseline_static[role_index])
                        for role_index in range(2)
                    ],
                    dtype=np.float32,
                )
            rows.append(
                f"<tr{' class=baseline-row' if scenario == baseline else ''}>"
                f"<td>{parameter_title(case)}</td><td>{static_by_role.mean():.3f}</td>"
                f"<td>{static_by_role[0]:.3f}</td><td>{static_by_role[1]:.3f}</td>"
                f"<td>{values['dynamic_track_distance']:.3f}</td>"
                f"<td>{roles['ball']['dynamic_track_distance']:.3f}</td>"
                f"<td>{roles['block']['dynamic_track_distance']:.3f}</td>"
                f"<td>{values['d_adj_relative_l2']:.3f}</td><td>{values['frequency_js']:.3f}</td>"
                f"<td>{values['centroid_rmse']:.3f}</td><td>{raft:.3f}</td></tr>"
            )
        dynamic_curves_array = np.stack(dynamic_curves, axis=0)
        mean_centered_curves_array = np.stack(mean_centered_curves, axis=0)
        decoder_curves_array = np.stack(decoder_curves, axis=0)
        curve_dir = output_dir / "control_curves"
        curve_stem = f"{axis}_dynamic_initial_aligned_vs_baseline"
        curve_npz = curve_dir / f"{curve_stem}.npz"
        curve_png = curve_dir / f"{curve_stem}.png"
        curve_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            curve_npz,
            scenarios=np.asarray(scenarios),
            labels=np.asarray(curve_labels),
            frame_times=np.arange(dynamic_curves_array.shape[1], dtype=np.float32) / 60.0,
            role_names=np.asarray(ROLE_NAMES),
            dynamic_initial_aligned_rms=dynamic_curves_array,
            dynamic_mean_centered_rms=mean_centered_curves_array,
        )
        plot_control_dynamic_curves(
            curve_png,
            f"DINOv3 xSSC controlled {axis}: initial-aligned dynamic vs baseline",
            curve_labels,
            dynamic_curves_array,
        )
        decoder_curve_stem = f"{axis}_decoder_native_partitions_vs_baseline"
        decoder_curve_npz = curve_dir / f"{decoder_curve_stem}.npz"
        decoder_curve_png = curve_dir / f"{decoder_curve_stem}.png"
        np.savez_compressed(
            decoder_curve_npz,
            scenarios=np.asarray(scenarios),
            labels=np.asarray(curve_labels),
            frame_times=np.arange(decoder_curves_array.shape[1], dtype=np.float32) / 60.0,
            partition_names=np.asarray(decoder_partitions),
            role_names=np.asarray(ROLE_NAMES),
            decoder_initial_aligned_rms=decoder_curves_array,
        )
        plot_decoder_partition_curves(
            decoder_curve_png,
            f"DINOv3 xSSC controlled {axis}: native decoder partitions vs baseline",
            curve_labels,
            decoder_curves_array,
        )
        decoder_rows = []
        for case_index, label in enumerate(curve_labels):
            mean_values = decoder_curves_array[case_index].mean(axis=0)
            peak_values = decoder_curves_array[case_index].max(axis=0)
            decoder_rows.append(
                f"<tr{' class=baseline-row' if scenarios[case_index] == baseline else ''}>"
                f"<td>{html.escape(label)}</td>"
                f"<td>{mean_values[0, 0]:.4f}</td><td>{mean_values[0, 1]:.4f}</td>"
                f"<td>{mean_values[1, 0]:.4f}</td><td>{mean_values[1, 1]:.4f}</td>"
                f"<td>{peak_values[0, 0]:.4f}</td><td>{peak_values[0, 1]:.4f}</td>"
                f"<td>{peak_values[1, 0]:.4f}</td><td>{peak_values[1, 1]:.4f}</td></tr>"
            )
        group_sections.append(
            f"<section class='control-group'><h2>{title}</h2><p class='muted'>{description} "
            f"所有指标均相对同一 baseline：e=0.7、μ=0.5、m=1kg。</p>"
            f"<div class='control-videos cols-{len(scenarios)}'>{''.join(videos)}</div>"
            f"<figure class='curve'><img src='{html.escape(str(curve_png.relative_to(output_dir)))}'>"
            f"<figcaption>首帧对齐的动态分歧：细线=原始逐帧值，粗线=5帧平滑；原始数组："
            f"<a href='{html.escape(str(curve_npz.relative_to(output_dir)))}'>{curve_npz.name}</a></figcaption></figure>"
            f"<div class='scroll'><table><thead><tr><th>变量取值</th><th>static all</th>"
            f"<th>static ball</th><th>static block</th><th>dynamic all</th>"
            f"<th>ball dynamic</th><th>block dynamic</th><th>D_adj</th><th>frequency</th>"
            f"<th>centroid</th><th>RAFT</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
            f"<h3 class='subhead'>Decoder 原生 static/dynamic 通道</h3>"
            f"<p class='muted'>decode.project2 后按前 {decoder_metadata['static_dim']} 维 static / 后 "
            f"{decoder_metadata['dynamic_dim']} 维 dynamic 切分；曲线均按各自通道数归一化。</p>"
            f"<figure class='curve'><img src='{html.escape(str(decoder_curve_png.relative_to(output_dir)))}'>"
            f"<figcaption>四个面板分别为 decoder-static/dynamic × ball/block；原始曲线："
            f"<a href='{html.escape(str(decoder_curve_npz.relative_to(output_dir)))}'>{decoder_curve_npz.name}</a>；"
            f"逐 case 投影特征与维度说明：<a href='decoder_features/metadata.json'>decoder metadata</a></figcaption></figure>"
            f"<div class='scroll'><table><thead><tr><th>变量取值</th>"
            f"<th>S-ball mean</th><th>S-block mean</th><th>D-ball mean</th><th>D-block mean</th>"
            f"<th>S-ball peak</th><th>S-block peak</th><th>D-ball peak</th><th>D-block peak</th>"
            f"</tr></thead><tbody>{''.join(decoder_rows)}</tbody></table></div></section>"
        )

    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DINOv3 xSSC controlled-variable analysis</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#101316;color:#edf2f7;font:13px system-ui,sans-serif;letter-spacing:0}}header{{position:sticky;top:0;z-index:5;background:#171c21;border-bottom:1px solid #38424c;padding:12px 18px}}main{{max-width:1800px;margin:auto;padding:16px}}h1{{font-size:21px;margin:0 0 5px}}h2{{font-size:18px;margin:25px 0 5px}}h3{{font-size:13px;margin:0 0 2px}}.subhead{{font-size:16px;margin:20px 0 4px;color:#d8f6ff}}a{{color:#7dd3fc}}.muted,.scenario,figcaption{{color:#b5c0ca}}.scenario{{font-size:11px;margin:0 0 5px}}.note{{border-left:3px solid #22d3ee;padding:8px 11px;background:#162027}}.control-group{{padding-bottom:18px;border-bottom:1px solid #34404a}}.control-videos{{display:grid;gap:10px;margin:10px 0 12px}}.cols-4{{grid-template-columns:repeat(4,minmax(0,1fr))}}.cols-3{{grid-template-columns:repeat(3,minmax(0,1fr))}}figure{{margin:0;min-width:0}}video{{display:block;width:100%;aspect-ratio:1/1;object-fit:contain;max-height:340px;background:#000}}figcaption{{font-size:11px;padding-top:4px}}.curve{{margin:10px 0 12px}}.curve img{{display:block;width:100%;max-width:1500px;background:#fff}}.baseline-tag{{display:inline-block;padding:1px 5px;margin-left:4px;background:#075985;color:#e0f2fe;font-size:10px}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #37414a;padding:5px 7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#1d242b}}td{{background:#13181d}}.baseline-row td{{background:#13242c;color:#d8f6ff}}@media(max-width:900px){{.cols-4,.cols-3{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:560px){{.cols-4,.cols-3{{grid-template-columns:1fr}}}}</style></head><body><header><h1>DINOv3 MOVi-C step-044000 · 控制变量分析</h1><div class="muted">150 frames @ 60 FPS · slots [150,11,512]</div></header><main>
<p class="note">只分析 <b>{html.escape(model['short_name'])}</b>。红色 overlay=ball slot，青色 overlay=block slot，细轮廓=仿真 GT；GT 仅用于绑定匿名 slot 身份。</p>
{''.join(group_sections)}</main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    output_dir = (args.output_dir or (analysis_dir / "object_gt_aligned")).expanduser().resolve()
    simulator = args.simulator.expanduser().resolve()
    source = json.loads((analysis_dir / "metadata.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    masks = prepare_role_masks(simulator, source["cases"], output_dir, args.force_masks)
    result = aggregate(analysis_dir, output_dir, source, masks)
    print(f"[complete] models={len(result['models'])} cases={len(result['cases'])}")
    print(f"[viewer] {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
