#!/usr/bin/env python3
"""Compare full-video xSSC object slots across controlled ball-block simulations."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
TRAIN_XSSC_ROOT = ROOT.parent
PROJECT_ROOT = TRAIN_XSSC_ROOT.parent
PACKAGE_PARENT = PROJECT_ROOT.parent
PYTHON_BIN = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
DEFAULT_DATASET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/ball_block"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/"
    "xssc_object_slot_separation_cases_dinov3_latest/ball_block_parameter_pairs"
)
DEFAULT_DINOV3_CHECKPOINT = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC/"
    "restart_save1000_20260720T140029Z/"
    "movi_c_transfer15000_b64_acc3_20260721T134713Z/"
    "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/42/step-044000.pth"
)
IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)
PAIR_METRICS = (
    "static_distance",
    "dynamic_track_distance",
    "d_adj_relative_l2",
    "frequency_js",
    "centroid_rmse",
    "area_curve_relative_l2",
    "raft_relative_l2",
)
DYNAMIC_METRICS = (
    "dynamic_track_distance",
    "d_adj_relative_l2",
    "frequency_js",
    "centroid_rmse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dinov3-checkpoint", type=Path, default=DEFAULT_DINOV3_CHECKPOINT)
    parser.add_argument("--official-root", type=Path, default=Path("/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis"))
    parser.add_argument("--gpus", default="3,5")
    parser.add_argument("--raft-device", default="cuda:3")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--xssc-batch-size", type=int, default=16)
    parser.add_argument("--raft-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-raft", action="store_true")
    parser.add_argument("--worker-spec", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_gpus(text: str) -> list[int]:
    gpus = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError(f"Invalid GPU list: {text}")
    if 4 in gpus:
        raise ValueError("GPU 4 is disabled by workspace policy")
    return gpus


def load_cases(dataset_root: Path) -> list[dict[str, Any]]:
    cases = []
    for metadata_path in sorted(dataset_root.glob("*.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        video_path = Path(payload["video"])
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        cases.append(
            {
                "scenario": payload["scenario"],
                "video": str(video_path),
                "metadata": str(metadata_path),
                "parameters": payload["parameters"],
                "initial_conditions": payload["initial_conditions"],
                "rendering": payload["rendering"],
                "physics": payload["physics"],
            }
        )
    if len(cases) != 8:
        raise RuntimeError(f"Expected 8 ball-block cases, found {len(cases)}")
    return cases


def read_letterboxed_video(video_path: Path, input_size: int):
    import cv2
    import numpy as np
    import torch

    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if not frames:
        raise RuntimeError(f"No readable frames: {video_path}")
    source = np.stack(frames, axis=0)
    source_h, source_w = source.shape[1:3]
    scale = min(input_size / source_w, input_size / source_h)
    resized_w = max(1, round(source_w * scale))
    resized_h = max(1, round(source_h * scale))
    left = (input_size - resized_w) // 2
    top = (input_size - resized_h) // 2
    fill = np.asarray(IMAGENET_MEAN, dtype=np.uint8)
    output = np.empty((len(source), input_size, input_size, 3), dtype=np.uint8)
    output[...] = fill
    for index, frame in enumerate(source):
        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        output[index, top : top + resized_h, left : left + resized_w] = resized
    tensor = torch.from_numpy(output).permute(0, 3, 1, 2).float()
    mean = tensor.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    normalized = (tensor - mean) / std
    transform = {
        "mode": "aspect_ratio_resize_imagenet_mean_padding",
        "source_size": [source_h, source_w],
        "output_size": [input_size, input_size],
        "resized_size": [resized_h, resized_w],
        "padding_ltrb": [left, top, input_size - resized_w - left, input_size - resized_h - top],
    }
    return normalized, output, fps, transform


def build_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    vitl_root = TRAIN_XSSC_ROOT / "xssc_rsfq2_ytvis_dinov3_vitl16_256"
    config = (
        vitl_root
        / "upstream/config-randsfq/rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000.py"
    )
    specs = [
        {
            "name": "dinov3_movic_step044000",
            "short_name": "DINOv3 MOVi-C step-044000",
            "family": "dinov3",
            "variant": "vitl_movic_slot512_bbox_mlp",
            "xssc_root": str(vitl_root),
            "xssc_config": str(config),
            "xssc_checkpoint": str(args.dinov3_checkpoint.resolve()),
            "dinov3_root": str(vitl_root / "third_party/dinov3"),
            "dinov3_checkpoint": "/data/gaoya/ckpt/facebook-dinov3-vitl16-pretrain-lvd1689m/model.safetensors",
        }
    ]
    official_config = "/home/gaoya/Code_Video/xSSC-main/config-randsfq/rsfq2_r-ytvis.py"
    for checkpoint in sorted(args.official_root.resolve().glob("*.pth")):
        specs.append(
            {
                "name": f"official_dinov2_{checkpoint.stem}",
                "short_name": f"Official DINOv2 {checkpoint.stem}",
                "family": "dinov2",
                "variant": "official_r_ytvis",
                "xssc_root": "/home/gaoya/Code_Video/xSSC-main",
                "xssc_config": official_config,
                "xssc_checkpoint": str(checkpoint),
            }
        )
    if len(specs) != 4:
        raise RuntimeError(f"Expected DINOv3 plus 3 official weights, found {len(specs)} specs")
    for spec in specs:
        spec["safe_name"] = safe_name(spec["name"])
    return specs


def prepare_inputs(args: argparse.Namespace, cases: list[dict[str, Any]]) -> None:
    import cv2
    import numpy as np
    import torch

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import analyze_official_xssc_dynamics_raft as base

    input_root = args.output_dir / "inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    raft = None
    device = torch.device(args.raft_device)
    for position, case in enumerate(cases, start=1):
        case_dir = input_root / case["scenario"]
        flow_path = case_dir / "raft_flow_16.npz"
        video_path = case_dir / "xssc_input_150f.mp4"
        metadata_path = case_dir / "metadata.json"
        if metadata_path.is_file() and video_path.is_file() and (args.skip_raft or flow_path.is_file()) and not args.force:
            continue
        normalized, rgb, fps, transform = read_letterboxed_video(Path(case["video"]), args.input_size)
        if len(rgb) != 150:
            raise RuntimeError(f"Expected 150 frames for {case['scenario']}, found {len(rgb)}")
        case_dir.mkdir(parents=True, exist_ok=True)
        base.write_video(video_path, rgb, fps=fps)
        if not args.skip_raft:
            if raft is None:
                raft = base.build_raft(device, args.raft_iters)
            full_flow = base.compute_raft_flow(raft, rgb, device, args.raft_iters)
            low_flow = np.stack(
                [cv2.resize(frame, (16, 16), interpolation=cv2.INTER_AREA) for frame in full_flow],
                axis=0,
            ).astype(np.float16)
            np.savez_compressed(flow_path, flow=low_flow)
        write_json(
            metadata_path,
            {
                **case,
                "frames": len(rgb),
                "fps": fps,
                "preprocess": transform,
                "raft_flow": str(flow_path) if not args.skip_raft else None,
            },
        )
    del raft
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def selected_slot_video(rgb, labels, selected_slots):
    import cv2
    import numpy as np
    import analyze_xssc_object_slot_separation_cases as objsep

    selected = list(selected_slots[:2])
    output = []
    for frame_index in range(len(rgb)):
        panels = []
        for slot_id in selected:
            color = objsep.PALETTE[int(slot_id) % len(objsep.PALETTE)]
            panel = (rgb[frame_index].astype(np.float32) * 0.38).astype(np.uint8)
            mask = labels[frame_index] == int(slot_id)
            panel[mask] = (
                panel[mask].astype(np.float32) * 0.2 + color.astype(np.float32) * 0.8
            ).round().clip(0, 255).astype(np.uint8)
            cv2.putText(
                panel,
                f"S{slot_id}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                tuple(int(value) for value in color),
                2,
                cv2.LINE_AA,
            )
            panels.append(panel)
        output.append(np.concatenate(panels, axis=1))
    return np.stack(output, axis=0)


def run_worker(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    for path in (PACKAGE_PARENT, PROJECT_ROOT, TRAIN_XSSC_ROOT, ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import analyze_official_xssc_dynamics_raft as base
    import analyze_xssc_dinov3_object_slot_separation_cases as dinov3_eval
    import analyze_xssc_object_slot_separation_cases as objsep
    import run_xssc_slot_dedup_weight_compare as compare

    spec = json.loads(args.worker_spec.read_text(encoding="utf-8"))
    model_dir = args.output_dir / "models" / spec["safe_name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, slot_dim, num_slots, initializer = compare.load_xssc_variant(spec, device)
    cases = load_cases(args.dataset_root)
    records = []
    for case in cases:
        normalized, rgb, fps, transform = read_letterboxed_video(Path(case["video"]), args.input_size)
        flow_path = args.output_dir / "inputs" / case["scenario"] / "raft_flow_16.npz"
        flow = None if args.skip_raft else np.load(flow_path)["flow"].astype(np.float32)
        boxes = None
        selected_boxes = 0
        if initializer == "bbox_mlp":
            boxes, selected_boxes = dinov3_eval.build_amg_boxes(normalized[None].to(device), num_slots)
        slots, attention = dinov3_eval.extract_variant_slots(
            model,
            normalized,
            device=device,
            seed=args.seed,
            batch_size=args.xssc_batch_size,
            initializer=initializer,
            boxes=boxes,
        )
        slots_np = slots.numpy().astype(np.float32)
        attention_np = attention.numpy().astype(np.float32)
        labels_low, labels = objsep.attention_to_hard_labels(attention_np, args.input_size)
        analysis = objsep.analyze_slots(slots_np, attention_np, flow, labels_low)
        centroids = objsep.attention_centroids(attention_np)
        selected = list(analysis["selected_slots"][:2])
        if len(selected) < 2:
            ordered = sorted(analysis["slot_summary"], key=lambda item: item["object_score"], reverse=True)
            selected = [int(item["slot"]) for item in ordered[:2]]
        case_dir = model_dir / case["scenario"]
        case_dir.mkdir(parents=True, exist_ok=True)
        all_overlay = objsep.overlay_all_slots(rgb, labels)
        selected_overlay = selected_slot_video(rgb, labels, selected)
        base.write_video(case_dir / "all_slot_overlay.mp4", all_overlay, fps=fps)
        base.write_video(case_dir / "selected_object_slots.mp4", selected_overlay, fps=fps)
        objsep.plot_slot_curves(
            case_dir / "slot_dynamics_curves.png",
            {
                "d_adj": analysis["d_adj"],
                "slot_flow": analysis["slot_flow"],
                "centroid_speed": analysis["centroid_speed"],
            },
            f"{spec['short_name']} | {case['scenario']}",
            selected,
        )
        np.savez_compressed(
            case_dir / "slots_attention.npz",
            slots=slots_np.astype(np.float16),
            attention=attention_np.astype(np.float16),
            selected_slots=np.asarray(selected, dtype=np.int16),
            d_adj=analysis["d_adj"].astype(np.float32),
            slot_flow=analysis["slot_flow"].astype(np.float32),
            centroids=centroids.astype(np.float32),
            centroid_speed=analysis["centroid_speed"].astype(np.float32),
        )
        write_json(
            case_dir / "summary.json",
            {
                "scenario": case["scenario"],
                "parameters": case["parameters"],
                "frames": len(rgb),
                "fps": fps,
                "preprocess": transform,
                "selected_slots": selected,
                "selected_boxes": selected_boxes,
                "slot_summary": analysis["slot_summary"],
            },
        )
        records.append(
            {
                "scenario": case["scenario"],
                "parameters": case["parameters"],
                "selected_slots": selected,
                "selected_boxes": selected_boxes,
                "assets": {
                    "all_slot_overlay": f"models/{spec['safe_name']}/{case['scenario']}/all_slot_overlay.mp4",
                    "selected_object_slots": f"models/{spec['safe_name']}/{case['scenario']}/selected_object_slots.mp4",
                    "slot_dynamics_curves": f"models/{spec['safe_name']}/{case['scenario']}/slot_dynamics_curves.png",
                    "arrays": f"models/{spec['safe_name']}/{case['scenario']}/slots_attention.npz",
                },
            }
        )
        print(f"[{spec['short_name']}] {case['scenario']} selected={selected}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(
        model_dir / "metadata.json",
        {
            "model": {
                **spec,
                "slot_dim": slot_dim,
                "num_slots": num_slots,
                "initializer": initializer,
                "slot_shape": [150, num_slots, slot_dim],
            },
            "cases": records,
        },
    )


def relative_l2(left, right) -> float:
    import numpy as np

    numerator = float(np.linalg.norm(left - right))
    denominator = 0.5 * float(np.linalg.norm(left) + np.linalg.norm(right))
    return numerator / max(denominator, 1.0e-8)


def cosine_distance(left, right) -> float:
    import numpy as np

    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return 0.0
    return float(1.0 - np.dot(left.reshape(-1), right.reshape(-1)) / denominator)


def js_divergence(left, right) -> float:
    import numpy as np

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / max(left.sum(), 1.0e-12)
    right = right / max(right.sum(), 1.0e-12)
    middle = 0.5 * (left + right)
    kl_left = np.sum(np.where(left > 0, left * np.log((left + 1.0e-12) / (middle + 1.0e-12)), 0.0))
    kl_right = np.sum(np.where(right > 0, right * np.log((right + 1.0e-12) / (middle + 1.0e-12)), 0.0))
    return float(0.5 * (kl_left + kl_right))


def load_track_bundle(output_dir: Path, model: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    path = output_dir / case["assets"]["arrays"]
    arrays = np.load(path)
    slots = arrays["slots"].astype(np.float32)
    attention = arrays["attention"].astype(np.float32)
    selected = arrays["selected_slots"].astype(int)[:2]
    selected_slots = slots[:, selected]
    static = selected_slots.mean(axis=0)
    residual = selected_slots - static[None]
    d_adj = np.linalg.norm(np.diff(selected_slots, axis=0), axis=-1) / math.sqrt(slots.shape[-1])
    centered = residual * np.hanning(len(residual))[:, None, None]
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    power = power[1:].sum(axis=-1).T
    labels = attention.argmax(axis=1)
    area = np.stack([(labels == slot_id).mean(axis=(1, 2)) for slot_id in selected], axis=1)
    centroids = arrays["centroids"].astype(np.float32)[:, selected]
    return {
        "selected": selected,
        "static": static,
        "residual": residual,
        "d_adj": d_adj,
        "power": power,
        "area": area,
        "centroids": centroids,
    }


def compare_tracks(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    left_static = left["static"] / np.maximum(np.linalg.norm(left["static"], axis=1, keepdims=True), 1.0e-12)
    right_static = right["static"] / np.maximum(np.linalg.norm(right["static"], axis=1, keepdims=True), 1.0e-12)
    static_cosine = left_static @ right_static.T
    left_area = left["area"].mean(axis=0)
    right_area = right["area"].mean(axis=0)
    area_cost = np.abs(left_area[:, None] - right_area[None, :])
    row_ids, col_ids = linear_sum_assignment((1.0 - static_cosine) + 0.25 * area_cost)
    metrics = {key: [] for key in PAIR_METRICS if key != "raft_relative_l2"}
    matches = []
    for left_id, right_id in zip(row_ids, col_ids):
        metrics["static_distance"].append(float(1.0 - static_cosine[left_id, right_id]))
        metrics["dynamic_track_distance"].append(
            cosine_distance(left["residual"][:, left_id], right["residual"][:, right_id])
        )
        metrics["d_adj_relative_l2"].append(
            relative_l2(left["d_adj"][:, left_id], right["d_adj"][:, right_id])
        )
        metrics["frequency_js"].append(
            js_divergence(left["power"][left_id], right["power"][right_id])
        )
        metrics["centroid_rmse"].append(
            float(np.sqrt(np.mean((left["centroids"][:, left_id] - right["centroids"][:, right_id]) ** 2)) / 16.0)
        )
        metrics["area_curve_relative_l2"].append(
            relative_l2(left["area"][:, left_id], right["area"][:, right_id])
        )
        matches.append(
            {
                "left_slot": int(left["selected"][left_id]),
                "right_slot": int(right["selected"][right_id]),
                "static_cosine": float(static_cosine[left_id, right_id]),
            }
        )
    return {**{key: float(np.mean(value)) for key, value in metrics.items()}, "matches": matches}


def parameter_gaps(left: dict[str, float], right: dict[str, float]) -> dict[str, Any]:
    e_gap = abs(float(left["restitution"]) - float(right["restitution"]))
    mu_gap = abs(float(left["lateral_friction"]) - float(right["lateral_friction"]))
    mass_gap = abs(math.log10(float(left["ball_mass_kg"])) - math.log10(float(right["ball_mass_kg"])))
    changed = [name for name, value in (("restitution", e_gap), ("friction", mu_gap), ("mass", mass_gap)) if value > 1.0e-12]
    axis = changed[0] if len(changed) == 1 else "mixed"
    normalized = math.sqrt((e_gap / 0.6) ** 2 + (mu_gap / 0.9) ** 2 + (mass_gap / math.log10(50.0)) ** 2)
    return {
        "restitution_gap": e_gap,
        "friction_gap": mu_gap,
        "log10_mass_gap": mass_gap,
        "normalized_parameter_distance": normalized,
        "axis": axis,
    }


def rank01(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return np.ones_like(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / (len(values) - 1)


def correlation(x, y) -> float:
    import numpy as np
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def plot_pair_heatmaps(path: Path, scenarios: list[str], pair_rows: list[dict[str, Any]], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = (
        "static_distance",
        "dynamic_track_distance",
        "d_adj_relative_l2",
        "frequency_js",
        "centroid_rmse",
        "dynamic_composite",
    )
    index = {name: position for position, name in enumerate(scenarios)}
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axis_plot, metric in zip(axes.flat, metrics):
        matrix = np.zeros((len(scenarios), len(scenarios)), dtype=np.float64)
        for row in pair_rows:
            left = index[row["left"]]
            right = index[row["right"]]
            matrix[left, right] = matrix[right, left] = row[metric]
        image = axis_plot.imshow(matrix, cmap="viridis", aspect="equal")
        axis_plot.set_title(metric)
        axis_plot.set_xticks(range(len(scenarios)), scenarios, rotation=45, ha="right", fontsize=7)
        axis_plot.set_yticks(range(len(scenarios)), scenarios, fontsize=7)
        figure.colorbar(image, ax=axis_plot, fraction=0.046, pad=0.04)
    figure.suptitle(title)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def aggregate(args: argparse.Namespace, specs: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    model_results = []
    scenario_order = [case["scenario"] for case in cases]
    parameters = {case["scenario"]: case["parameters"] for case in cases}
    raft = {
        case["scenario"]: np.load(args.output_dir / "inputs" / case["scenario"] / "raft_flow_16.npz")["flow"].astype(np.float32)
        for case in cases
    }
    for spec in specs:
        metadata = json.loads(
            (args.output_dir / "models" / spec["safe_name"] / "metadata.json").read_text(encoding="utf-8")
        )
        case_by_scenario = {case["scenario"]: case for case in metadata["cases"]}
        bundles = {
            scenario: load_track_bundle(args.output_dir, metadata["model"], case_by_scenario[scenario])
            for scenario in scenario_order
        }
        pairs = []
        for left_index, left in enumerate(scenario_order):
            for right in scenario_order[left_index + 1 :]:
                row = {
                    "left": left,
                    "right": right,
                    **parameter_gaps(parameters[left], parameters[right]),
                    **compare_tracks(bundles[left], bundles[right]),
                    "raft_relative_l2": relative_l2(raft[left], raft[right]),
                }
                pairs.append(row)
        composite_parts = np.stack([rank01([row[metric] for row in pairs]) for metric in DYNAMIC_METRICS], axis=1)
        for row, value in zip(pairs, composite_parts.mean(axis=1)):
            row["dynamic_composite"] = float(value)
        correlations = {}
        for scope in ("all", "restitution", "friction", "mass"):
            selected = pairs if scope == "all" else [row for row in pairs if row["axis"] == scope]
            correlations[scope] = {
                "pair_count": len(selected),
                **{
                    metric: {
                        "parameter_rho": correlation(
                            [row["normalized_parameter_distance"] for row in selected],
                            [row[metric] for row in selected],
                        ),
                        "raft_rho": correlation(
                            [row["raft_relative_l2"] for row in selected],
                            [row[metric] for row in selected],
                        ),
                    }
                    for metric in (*PAIR_METRICS[:-1], "dynamic_composite")
                },
            }
        heatmap_path = args.output_dir / "models" / spec["safe_name"] / "pair_heatmaps.png"
        plot_pair_heatmaps(heatmap_path, scenario_order, pairs, spec["short_name"])
        model_results.append(
            {
                "model": metadata["model"],
                "cases": metadata["cases"],
                "pairs": pairs,
                "correlations": correlations,
                "heatmap": str(heatmap_path.relative_to(args.output_dir)),
            }
        )
    result = {
        "dataset_root": str(args.dataset_root),
        "preprocess": "all 150 frames; preserve aspect ratio; resize to 256x144; ImageNet-mean pad to 256x256",
        "cases": cases,
        "models": model_results,
        "method": {
            "slot_selection": "top-2 stable active slots by RAFT-weighted object_score",
            "cross_video_matching": "Hungarian assignment on static slot cosine plus mean hard-mask area",
            "dynamic_composite": "mean within-model percentile rank of dynamic-track, D_adj, frequency-JS, centroid distances",
        },
    }
    write_json(args.output_dir / "metadata.json", result)
    build_html(args.output_dir, result)
    return result


def format_rho(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:+.3f}"


def build_html(output_dir: Path, result: dict[str, Any]) -> None:
    case_rows = "".join(
        f"<tr><td>{html.escape(case['scenario'])}</td><td>{case['parameters']['restitution']}</td>"
        f"<td>{case['parameters']['lateral_friction']}</td><td>{case['parameters']['ball_mass_kg']}</td>"
        f"<td><video src='inputs/{html.escape(case['scenario'])}/xssc_input_150f.mp4' controls muted preload='metadata'></video></td></tr>"
        for case in result["cases"]
    )
    correlation_rows = []
    for model_result in result["models"]:
        model = model_result["model"]
        values = model_result["correlations"]["all"]
        for metric in ("static_distance", *DYNAMIC_METRICS, "dynamic_composite"):
            correlation_rows.append(
                f"<tr><td>{html.escape(model['short_name'])}</td><td>{metric}</td>"
                f"<td>{format_rho(values[metric]['parameter_rho'])}</td>"
                f"<td>{format_rho(values[metric]['raft_rho'])}</td></tr>"
            )
    model_sections = []
    for model_result in result["models"]:
        model = model_result["model"]
        videos = "".join(
            f"<figure><h4>{html.escape(case['scenario'])} | slots {case['selected_slots']}</h4>"
            f"<video src='{html.escape(case['assets']['all_slot_overlay'])}' controls muted preload='metadata'></video>"
            f"<figcaption><a href='{html.escape(case['assets']['selected_object_slots'])}'>selected object slots</a> | "
            f"<a href='{html.escape(case['assets']['slot_dynamics_curves'])}'>dynamics</a></figcaption></figure>"
            for case in model_result["cases"]
        )
        controlled_rows = []
        for pair in model_result["pairs"]:
            if pair["axis"] == "mixed":
                continue
            controlled_rows.append(
                f"<tr><td>{pair['axis']}</td><td>{pair['left']}</td><td>{pair['right']}</td>"
                f"<td>{pair['static_distance']:.3f}</td><td>{pair['dynamic_track_distance']:.3f}</td>"
                f"<td>{pair['d_adj_relative_l2']:.3f}</td><td>{pair['frequency_js']:.3f}</td>"
                f"<td>{pair['centroid_rmse']:.3f}</td><td>{pair['raft_relative_l2']:.3f}</td>"
                f"<td>{pair['dynamic_composite']:.3f}</td></tr>"
            )
        model_sections.append(
            f"<section><h2>{html.escape(model['short_name'])}</h2>"
            f"<p class='muted'>shape={model['slot_shape']}, initializer={html.escape(model['initializer'])}, checkpoint=<code>{html.escape(model['xssc_checkpoint'])}</code></p>"
            f"<img class='heatmap' src='{html.escape(model_result['heatmap'])}'>"
            f"<div class='videos'>{videos}</div><div class='scroll'><table><thead><tr><th>axis</th><th>left</th><th>right</th><th>static</th><th>dynamic</th><th>D_adj</th><th>frequency</th><th>centroid</th><th>RAFT</th><th>composite</th></tr></thead><tbody>{''.join(controlled_rows)}</tbody></table></div></section>"
        )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>xSSC ball-block parameter pairs</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#101316;color:#edf2f7;font:13px system-ui,sans-serif;letter-spacing:0}}header{{position:sticky;top:0;z-index:5;background:#171c21;border-bottom:1px solid #38424c;padding:12px 18px}}main{{max-width:1900px;margin:auto;padding:16px}}h1{{font-size:21px;margin:0 0 5px}}h2{{font-size:18px;margin:24px 0 7px}}h4{{font-size:12px;margin:0 0 5px}}.muted{{color:#b5c0ca}}code{{color:#c8f3ff}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #37414a;padding:5px 7px;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}}th{{background:#1d242b}}td{{background:#13181d}}video,img{{display:block;width:100%;background:#000}}.source video{{width:240px}}.videos{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}}figure{{margin:0;min-width:0}}figcaption{{font-size:11px;padding:4px 0}}a{{color:#8bd5ff}}.heatmap{{max-width:1500px;margin:8px 0 14px}}.scroll{{overflow:auto}}@media(max-width:1000px){{.videos{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}</style></head><body><header><h1>xSSC full-video ball-block parameter-pair analysis</h1><div class="muted">150 frames @ 60 FPS; aspect-preserving mean padding; 1 DINOv3 + 3 official xSSC checkpoints.</div></header><main>
<h2>Controlled simulation parameters</h2><table class="source"><thead><tr><th>scenario</th><th>restitution</th><th>friction</th><th>ball mass kg</th><th>input</th></tr></thead><tbody>{case_rows}</tbody></table>
<h2>Cross-video capture correlations</h2><p class="muted">Positive rho means larger physics/RAFT differences produce larger xSSC feature differences. Axis-specific values and all 28 raw pairs are retained in metadata.json.</p><table><thead><tr><th>model</th><th>feature</th><th>parameter rho</th><th>RAFT rho</th></tr></thead><tbody>{''.join(correlation_rows)}</tbody></table>
{''.join(model_sections)}</main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def run_model_queue(gpu: int, specs: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    failures = []
    for spec in specs:
        model_dir = args.output_dir / "models" / spec["safe_name"]
        if (model_dir / "metadata.json").is_file() and not args.force:
            print(f"[reuse][gpu{gpu}] {spec['short_name']}", flush=True)
            continue
        spec_path = args.output_dir / "specs" / f"{spec['safe_name']}.json"
        write_json(spec_path, spec)
        command = [
            str(PYTHON_BIN),
            str(Path(__file__).resolve()),
            "--worker-spec", str(spec_path),
            "--dataset-root", str(args.dataset_root),
            "--output-dir", str(args.output_dir),
            "--device", "cuda:0",
            "--input-size", str(args.input_size),
            "--xssc-batch-size", str(args.xssc_batch_size),
            "--seed", str(args.seed),
            *( ["--skip-raft"] if args.skip_raft else [] ),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(PACKAGE_PARENT), str(PROJECT_ROOT), str(TRAIN_XSSC_ROOT), str(ROOT), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        model_dir.mkdir(parents=True, exist_ok=True)
        log_path = model_dir / "worker.log"
        print(f"[run][gpu{gpu}] {spec['short_name']}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            failures.append(spec["name"])
            print(f"[failed][gpu{gpu}] {spec['short_name']} log={log_path}", flush=True)
        else:
            print(f"[complete][gpu{gpu}] {spec['short_name']}", flush=True)
    return failures


def run_orchestrator(args: argparse.Namespace) -> None:
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.dinov3_checkpoint = args.dinov3_checkpoint.expanduser().resolve()
    if not args.dinov3_checkpoint.is_file():
        raise FileNotFoundError(args.dinov3_checkpoint)
    gpus = parse_gpus(args.gpus)
    cases = load_cases(args.dataset_root)
    specs = build_specs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)
    (args.output_dir / "specs").mkdir(exist_ok=True)
    write_json(args.output_dir / "run_config.json", {"args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "specs": specs, "cases": cases})
    prepare_inputs(args, cases)
    queues = [[] for _ in gpus]
    for index, spec in enumerate(specs):
        queues[index % len(gpus)].append(spec)
    failures = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_model_queue, gpu, queue, args) for gpu, queue in zip(gpus, queues)]
        for future in as_completed(futures):
            failures.extend(future.result())
    if failures:
        raise RuntimeError(f"Model workers failed: {failures}")
    result = aggregate(args, specs, cases)
    print(f"[complete] models={len(result['models'])} cases={len(cases)}", flush=True)
    print(f"[viewer] {args.output_dir / 'index.html'}", flush=True)


def main() -> None:
    args = parse_args()
    if args.worker_spec is not None:
        args.dataset_root = args.dataset_root.expanduser().resolve()
        args.output_dir = args.output_dir.expanduser().resolve()
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
