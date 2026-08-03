#!/usr/bin/env python3
"""Build the descriptive Phase-1 physics-versus-appearance report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_ROOT = Path("/data/gaoya/agent-data/outputs/xssc_physics_representation/phase1")
ROLES = ("ball", "block")
REPRESENTATIONS = ("raw_slot", "decoder_static", "decoder_dynamic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--quality-recall", type=float, default=0.8)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def state_dict(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict) and isinstance(value.get("state_dict"), dict):
        value = value["state_dict"]
    if not isinstance(value, dict):
        raise TypeError(f"Invalid checkpoint payload: {path}")
    return value


def unique_suffix(state: dict[str, Any], suffix: str) -> torch.Tensor:
    values = [value for key, value in state.items() if str(key).endswith(suffix)]
    if len(values) != 1:
        raise RuntimeError(f"Expected one tensor ending in {suffix}, found {len(values)}")
    return values[0].float()


def dynamic_ratio(config_path: Path) -> float:
    text = config_path.read_text(encoding="utf-8")
    explicit = re.search(r"^decoder_dynamic_ratio\s*=\s*([0-9.]+)", text, re.MULTILINE)
    if explicit:
        return float(explicit.group(1))
    decoder = re.search(r"type=MarkovRarDecoder,[\s\S]{0,200}?\brd=([0-9.]+)", text)
    if decoder:
        return float(decoder.group(1))
    raise RuntimeError(f"Cannot find decoder dynamic ratio in {config_path}")


def projector(spec: dict[str, Any]):
    state = state_dict(Path(spec["xssc_checkpoint"]))
    weight = unique_suffix(state, "decode.project2.0.weight")
    norm_weight = unique_suffix(state, "decode.project2.1.weight")
    norm_bias = unique_suffix(state, "decode.project2.1.bias")
    ratio = dynamic_ratio(Path(spec["xssc_config"]))
    decoder_dim = int(weight.shape[0])
    dynamic_dim = int(decoder_dim * ratio)
    static_dim = decoder_dim - dynamic_dim

    def apply(slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            value = F.linear(torch.from_numpy(slots), weight)
            value = F.layer_norm(value, (decoder_dim,), norm_weight, norm_bias, eps=1.0e-5)
        array = value.numpy().astype(np.float32)
        return array[..., :static_dim], array[..., static_dim:]

    return apply, {
        "slot_dim": int(weight.shape[1]),
        "decoder_dim": decoder_dim,
        "static_dim": static_dim,
        "dynamic_dim": dynamic_dim,
        "dynamic_ratio": ratio,
    }


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return 0.0
    return float(1.0 - np.dot(left.reshape(-1), right.reshape(-1)) / denominator)


def initial_aligned_curve(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_delta = left - left[:1]
    right_delta = right - right[:1]
    return np.linalg.norm(left_delta - right_delta, axis=-1) / math.sqrt(left.shape[-1])


def trajectory_divergence_percent(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Bounded disagreement between two trajectories after frame-0 alignment."""
    left_delta = left - left[:1]
    right_delta = right - right[:1]
    numerator = np.linalg.norm(left_delta - right_delta, axis=-1)
    denominator = np.linalg.norm(left_delta, axis=-1) + np.linalg.norm(right_delta, axis=-1)
    output = np.zeros_like(numerator, dtype=np.float32)
    valid = denominator > 1.0e-8
    output[valid] = 100.0 * numerator[valid] / denominator[valid]
    return np.clip(output, 0.0, 100.0)


def adjacent_trajectory_divergence_percent(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Bounded disagreement between adjacent-frame feature changes."""
    left_step = np.diff(left, axis=0, prepend=left[:1])
    right_step = np.diff(right, axis=0, prepend=right[:1])
    numerator = np.linalg.norm(left_step - right_step, axis=-1)
    denominator = np.linalg.norm(left_step, axis=-1) + np.linalg.norm(right_step, axis=-1)
    output = np.zeros_like(numerator, dtype=np.float32)
    valid = denominator > 1.0e-8
    output[valid] = 100.0 * numerator[valid] / denominator[valid]
    return np.clip(output, 0.0, 100.0)


def adjacent_curve(features: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(features, axis=0), axis=-1) / math.sqrt(features.shape[-1])


def relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    numerator = float(np.linalg.norm(left - right))
    denominator = 0.5 * float(np.linalg.norm(left) + np.linalg.norm(right))
    return numerator / max(denominator, 1.0e-8)


def load_case(path: Path, project) -> dict[str, Any]:
    with np.load(path) as item:
        slots = item["slots"].astype(np.float32)
        attention = item["attention"].astype(np.float32)
        selected = item["selected_slots"].astype(np.int64)
        recall = item["recall_matrix"].astype(np.float32)
        precision = item["precision_matrix"].astype(np.float32)
        f1 = item["f1_matrix"].astype(np.float32)
    role_slots = slots[:, selected]
    decoder_static, decoder_dynamic = project(role_slots)
    role_attention = attention[:, selected]
    grid_y, grid_x = np.mgrid[0:16, 0:16].astype(np.float32)
    mass = np.maximum(role_attention.sum(axis=(2, 3)), 1.0e-8)
    centroid_x = (role_attention * grid_x[None, None]).sum(axis=(2, 3)) / mass
    centroid_y = (role_attention * grid_y[None, None]).sum(axis=(2, 3)) / mass
    centroids = np.stack((centroid_x, centroid_y), axis=-1)
    return {
        "selected": selected,
        "recall": recall[np.arange(2), selected],
        "precision": precision[np.arange(2), selected],
        "f1": f1[np.arange(2), selected],
        "representations": {
            "raw_slot": role_slots,
            "decoder_static": decoder_static,
            "decoder_dynamic": decoder_dynamic,
        },
        "centroids": centroids,
    }


def physics_pairs(cases: list[dict[str, Any]], baseline: str) -> list[tuple[str, str, str]]:
    return [(baseline, case["case_id"], "physics") for case in cases if case["family"] == "physics" and case["case_id"] != baseline]


def appearance_pairs(cases: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    by_base: dict[str, dict[str, str]] = {}
    for case in cases:
        if case["family"] != "appearance":
            continue
        by_base.setdefault(case["base_scenario"], {})[case["appearance_variant"]] = case["case_id"]
    pairs = []
    for variants in by_base.values():
        reference = variants["v1_default"]
        for variant in ("v2_dark_blue", "v3_warm_bright"):
            pairs.append((reference, variants[variant], "appearance"))
    return pairs


def summarize(root: Path, manifest: dict[str, Any], quality_recall: float) -> dict[str, Any]:
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    cases = manifest["cases"]
    case_by_id = {case["case_id"]: case for case in cases}
    baseline = next(case["case_id"] for case in cases if case["family"] == "physics" and case["case_id"] == "e07_mu05_m1")
    pair_specs = physics_pairs(cases, baseline) + appearance_pairs(cases)
    model_results = []
    quality_rows = []
    pair_rows = []
    temporal_rows = []
    stability_rows = []
    curves_dir = report_dir / "curves"
    curves_dir.mkdir(exist_ok=True)

    for spec in manifest["models"]:
        feature_root = root / "features" / spec["name"]
        missing = [case["case_id"] for case in cases if not (feature_root / f"{case['case_id']}.npz").is_file()]
        if missing:
            raise FileNotFoundError(f"{spec['name']} missing {len(missing)} cases; first={missing[:3]}")
        project, dimensions = projector(spec)
        loaded = {case["case_id"]: load_case(feature_root / f"{case['case_id']}.npz", project) for case in cases}
        for case in cases:
            item = loaded[case["case_id"]]
            for role_id, role in enumerate(ROLES):
                quality_pass = bool(item["recall"][role_id] >= quality_recall)
                quality_rows.append(
                    {
                        "model": spec["name"], "case_id": case["case_id"], "family": case["family"], "role": role,
                        "slot": int(item["selected"][role_id]), "recall": float(item["recall"][role_id]),
                        "precision": float(item["precision"][role_id]), "f1": float(item["f1"][role_id]),
                        "quality_pass": quality_pass,
                    }
                )
                for representation in REPRESENTATIONS:
                    features = item["representations"][representation][:, role_id]
                    curve = np.zeros(features.shape[0], dtype=np.float32)
                    curve[1:] = adjacent_curve(features).astype(np.float32)
                    stability_rows.append(
                        {
                            "model": spec["name"], "case_id": case["case_id"],
                            "family": case["family"], "role": role,
                            "representation": representation, "quality_pass": quality_pass,
                            "curve": curve,
                        }
                    )
        for left_id, right_id, intervention in pair_specs:
            left = loaded[left_id]
            right = loaded[right_id]
            for representation in REPRESENTATIONS:
                left_features = left["representations"][representation]
                right_features = right["representations"][representation]
                curves = initial_aligned_curve(left_features, right_features)
                divergence_curves = trajectory_divergence_percent(left_features, right_features)
                adjacent_divergence_curves = adjacent_trajectory_divergence_percent(left_features, right_features)
                for role_id, role in enumerate(ROLES):
                    left_pass = bool(left["recall"][role_id] >= quality_recall)
                    right_pass = bool(right["recall"][role_id] >= quality_recall)
                    pair_rows.append(
                        {
                            "model": spec["name"], "comparison_role": spec["comparison_role"],
                            "intervention": intervention, "left": left_id, "right": right_id,
                            "base_scenario": case_by_id[right_id]["base_scenario"], "role": role,
                            "representation": representation,
                            "static_cosine_distance": cosine_distance(left_features[:, role_id].mean(axis=0), right_features[:, role_id].mean(axis=0)),
                            "initial_aligned_mean": float(curves[:, role_id].mean()),
                            "initial_aligned_peak": float(curves[:, role_id].max()),
                            "adjacent_profile_relative_l2": relative_l2(adjacent_curve(left_features[:, role_id]), adjacent_curve(right_features[:, role_id])),
                            "left_quality_pass": left_pass,
                            "right_quality_pass": right_pass,
                        }
                    )
                    temporal_rows.append(
                        {
                            "model": spec["name"], "intervention": intervention,
                            "left": left_id, "right": right_id, "role": role,
                            "representation": representation,
                            "left_quality_pass": left_pass, "right_quality_pass": right_pass,
                            "curve": curves[:, role_id].astype(np.float32),
                            "divergence_curve": divergence_curves[:, role_id].astype(np.float32),
                            "adjacent_divergence_curve": adjacent_divergence_curves[:, role_id].astype(np.float32),
                        }
                    )
            centroid_curve = np.linalg.norm(
                (right["centroids"] - right["centroids"][:1]) - (left["centroids"] - left["centroids"][:1]), axis=-1
            ) / 16.0
            centroid_divergence = trajectory_divergence_percent(left["centroids"], right["centroids"])
            centroid_adjacent_divergence = adjacent_trajectory_divergence_percent(
                left["centroids"], right["centroids"]
            )
            for role_id, role in enumerate(ROLES):
                left_pass = bool(left["recall"][role_id] >= quality_recall)
                right_pass = bool(right["recall"][role_id] >= quality_recall)
                pair_rows.append(
                    {
                        "model": spec["name"], "comparison_role": spec["comparison_role"],
                        "intervention": intervention, "left": left_id, "right": right_id,
                        "base_scenario": case_by_id[right_id]["base_scenario"], "role": role,
                        "representation": "attention_centroid",
                        "static_cosine_distance": float("nan"),
                        "initial_aligned_mean": float(centroid_curve[:, role_id].mean()),
                        "initial_aligned_peak": float(centroid_curve[:, role_id].max()),
                        "adjacent_profile_relative_l2": relative_l2(
                            np.linalg.norm(np.diff(left["centroids"][:, role_id], axis=0), axis=-1),
                            np.linalg.norm(np.diff(right["centroids"][:, role_id], axis=0), axis=-1),
                        ),
                        "left_quality_pass": left_pass,
                        "right_quality_pass": right_pass,
                    }
                )
                temporal_rows.append(
                    {
                        "model": spec["name"], "intervention": intervention,
                        "left": left_id, "right": right_id, "role": role,
                        "representation": "attention_centroid",
                        "left_quality_pass": left_pass, "right_quality_pass": right_pass,
                        "curve": centroid_curve[:, role_id].astype(np.float32),
                        "divergence_curve": centroid_divergence[:, role_id].astype(np.float32),
                        "adjacent_divergence_curve": centroid_adjacent_divergence[:, role_id].astype(np.float32),
                    }
                )
        model_results.append({"model": spec, "dimensions": dimensions})

    aggregate_rows = []
    for spec in manifest["models"]:
        for intervention in ("physics", "appearance"):
            for representation in (*REPRESENTATIONS, "attention_centroid"):
                for role in ROLES:
                    selected = [
                        row for row in pair_rows
                        if row["model"] == spec["name"] and row["intervention"] == intervention
                        and row["representation"] == representation and row["role"] == role
                    ]
                    clean = [row for row in selected if row["left_quality_pass"] and row["right_quality_pass"]]
                    values = clean or selected
                    aggregate_rows.append(
                        {
                            "model": spec["name"], "intervention": intervention, "representation": representation, "role": role,
                            "pair_count": len(selected), "clean_pair_count": len(clean), "fallback_to_all": not bool(clean),
                            "initial_aligned_mean_median": float(np.median([row["initial_aligned_mean"] for row in values])),
                            "initial_aligned_peak_median": float(np.median([row["initial_aligned_peak"] for row in values])),
                            "adjacent_profile_relative_l2_median": float(np.median([row["adjacent_profile_relative_l2"] for row in values])),
                        }
                    )

    write_csv(report_dir / "slot_quality.csv", quality_rows)
    write_csv(report_dir / "pair_metrics.csv", pair_rows)
    write_csv(report_dir / "aggregate_metrics.csv", aggregate_rows)
    temporal_array = np.stack([row["curve"] for row in temporal_rows])
    divergence_array = np.stack([row["divergence_curve"] for row in temporal_rows])
    adjacent_divergence_array = np.stack([row["adjacent_divergence_curve"] for row in temporal_rows])
    stability_array = np.stack([row["curve"] for row in stability_rows])
    np.savez_compressed(report_dir / "temporal_curves.npz", curves=temporal_array)
    np.savez_compressed(report_dir / "trajectory_divergence_percent.npz", curves=divergence_array)
    np.savez_compressed(
        report_dir / "adjacent_trajectory_divergence_percent.npz",
        curves=adjacent_divergence_array,
    )
    np.savez_compressed(report_dir / "adjacent_feature_change_rms.npz", curves=stability_array)
    write_json(
        report_dir / "temporal_curves.json",
        {
            "shape": list(temporal_array.shape),
            "definition": "per-frame initial-aligned response; frame 0 is zero by construction",
            "records": [
                {
                    "curve_index": index,
                    **{
                        key: value for key, value in row.items()
                        if key not in ("curve", "divergence_curve", "adjacent_divergence_curve")
                    },
                }
                for index, row in enumerate(temporal_rows)
            ],
        },
    )
    write_json(
        report_dir / "trajectory_divergence_percent.json",
        {
            "shape": list(divergence_array.shape),
            "range": [0.0, 100.0],
            "definition": "100 * ||delta_left(t)-delta_right(t)|| / (||delta_left(t)||+||delta_right(t)||), where delta(t)=feature(t)-feature(0)",
            "interpretation": "0 means identical frame-0-aligned evolution; 100 means maximal disagreement",
            "records": [
                {
                    "curve_index": index,
                    **{
                        key: value for key, value in row.items()
                        if key not in ("curve", "divergence_curve", "adjacent_divergence_curve")
                    },
                }
                for index, row in enumerate(temporal_rows)
            ],
        },
    )
    write_json(
        report_dir / "adjacent_trajectory_divergence_percent.json",
        {
            "status": "deprecated_for_static_stability; retained only for audit",
            "shape": list(adjacent_divergence_array.shape),
            "range": [0.0, 100.0],
            "definition": "100 * ||step_left(t)-step_right(t)|| / (||step_left(t)||+||step_right(t)||), where step(t)=feature(t)-feature(t-1)",
            "interpretation": "0 means identical adjacent-frame change; 100 means maximal instantaneous disagreement",
            "records": [
                {
                    "curve_index": index,
                    **{
                        key: value for key, value in row.items()
                        if key not in ("curve", "divergence_curve", "adjacent_divergence_curve")
                    },
                }
                for index, row in enumerate(temporal_rows)
            ],
        },
    )
    write_json(
        report_dir / "adjacent_feature_change_rms.json",
        {
            "shape": list(stability_array.shape),
            "definition": "||feature(t)-feature(t-1)||_2 / sqrt(channel_dim)",
            "interpretation": "absolute per-channel adjacent change; lower means more temporally stable",
            "records": [
                {
                    "curve_index": index,
                    **{key: value for key, value in row.items() if key != "curve"},
                }
                for index, row in enumerate(stability_rows)
            ],
        },
    )
    plot_effects(report_dir / "physics_vs_appearance.png", aggregate_rows, manifest["models"])
    temporal_plots = plot_temporal_effects(
        report_dir / "temporal", temporal_rows, manifest["models"],
        curve_key="divergence_curve", y_label="trajectory divergence (%)",
    )
    stability_plots = plot_adjacent_stability(
        report_dir / "adjacent_stability", stability_rows, manifest["models"]
    )
    result = {
        "status": "descriptive_single-initial-condition pilot",
        "quality_recall_threshold": quality_recall,
        "models": model_results,
        "slot_quality": quality_rows,
        "pairs": pair_rows,
        "aggregate": aggregate_rows,
        "temporal_plots": temporal_plots,
        "stability_plots": stability_plots,
        "limitations": [
            "Only one base initial condition is available, so no inferential or generalization claim is made.",
            "MOVi-C frame-0 boxes are derived from simulator GT masks downsampled to the 16x16 xSSC patch grid in this pilot.",
            "YTVIS uses its official NormalShared initializer and is a secondary cross-domain control, not an architecture-only control.",
        ],
    }
    write_json(report_dir / "summary.json", result)
    build_html(report_dir, result)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_effects(path: Path, rows: list[dict[str, Any]], models: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(len(models), 2, figsize=(13, 4 * len(models)), constrained_layout=True)
    colors = {"physics": "#dc2626", "appearance": "#2563eb"}
    for model_id, spec in enumerate(models):
        for role_id, role in enumerate(ROLES):
            axis = axes[model_id, role_id]
            labels = list(REPRESENTATIONS) + ["attention_centroid"]
            x = np.arange(len(labels), dtype=np.float32)
            for offset, intervention in ((-0.18, "physics"), (0.18, "appearance")):
                values = [
                    next(
                        row["initial_aligned_mean_median"] for row in rows
                        if row["model"] == spec["name"] and row["role"] == role
                        and row["representation"] == representation and row["intervention"] == intervention
                    )
                    for representation in labels
                ]
                axis.bar(x + offset, values, width=0.34, color=colors[intervention], label=intervention)
            axis.set_xticks(x, labels, rotation=20, ha="right")
            axis.set_title(f"{spec['short_name']} | {role}")
            axis.set_ylabel("median initial-aligned response")
            axis.grid(axis="y", alpha=0.2)
            axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_temporal_effects(
    output_dir: Path,
    rows: list[dict[str, Any]],
    models: list[dict[str, Any]],
    *,
    curve_key: str,
    y_label: str,
) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"physics": "#dc2626", "appearance": "#2563eb"}
    labels = {"physics": "physics", "appearance": "appearance"}
    plots = []
    for spec in models:
        for role in ROLES:
            figure, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)
            for axis, representation in zip(axes.flat, (*REPRESENTATIONS, "attention_centroid")):
                subtitles = []
                for intervention in ("physics", "appearance"):
                    selected = [
                        row for row in rows
                        if row["model"] == spec["name"] and row["role"] == role
                        and row["representation"] == representation
                        and row["intervention"] == intervention
                    ]
                    clean = [row for row in selected if row["left_quality_pass"] and row["right_quality_pass"]]
                    used = clean or selected
                    curves = np.stack([row[curve_key] for row in used])
                    frames = np.arange(curves.shape[1])
                    for curve in curves:
                        axis.plot(frames, curve, color=colors[intervention], alpha=0.16, linewidth=0.8)
                    median = np.median(curves, axis=0)
                    lower, upper = np.quantile(curves, (0.25, 0.75), axis=0)
                    axis.fill_between(frames, lower, upper, color=colors[intervention], alpha=0.14)
                    axis.plot(frames, median, color=colors[intervention], linewidth=2.0, label=labels[intervention])
                    fallback = " fallback" if not clean else ""
                    subtitles.append(f"{intervention[0].upper()} {len(clean)}/{len(selected)}{fallback}")
                axis.set_title(f"{representation} | {'; '.join(subtitles)}")
                axis.set_xlabel("frame")
                axis.set_ylabel(y_label)
                axis.set_ylim(0.0, 100.0)
                axis.grid(alpha=0.2)
                axis.legend()
            figure.suptitle(f"{spec['short_name']} | {role}", fontsize=14)
            filename = f"{spec['name']}_{role}.png"
            figure.savefig(output_dir / filename, dpi=150)
            plt.close(figure)
            plots.append(
                {
                    "model": spec["name"],
                    "short_name": spec["short_name"],
                    "role": role,
                    "path": str(Path(output_dir.name) / filename),
                }
            )
    return plots


def plot_adjacent_stability(
    output_dir: Path,
    rows: list[dict[str, Any]],
    models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"raw_slot": "#64748b", "decoder_static": "#16a34a", "decoder_dynamic": "#ea580c"}
    plots = []
    for spec in models:
        for role in ROLES:
            selected_by_rep = {}
            fallback = False
            for representation in REPRESENTATIONS:
                selected = [
                    row for row in rows
                    if row["model"] == spec["name"] and row["role"] == role
                    and row["representation"] == representation
                ]
                clean = [row for row in selected if row["quality_pass"]]
                selected_by_rep[representation] = clean or selected
                fallback = fallback or not bool(clean)
            dynamic = np.stack([row["curve"] for row in selected_by_rep["decoder_dynamic"]])
            scale = max(float(np.median(dynamic[:, 1:])), 1.0e-8)
            figure, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
            medians = {}
            for representation in REPRESENTATIONS:
                curves = np.stack([row["curve"] for row in selected_by_rep[representation]]) / scale
                frames = np.arange(curves.shape[1])
                for curve in curves:
                    axis.plot(frames, curve, color=colors[representation], alpha=0.10, linewidth=0.7)
                median = np.median(curves, axis=0)
                lower, upper = np.quantile(curves, (0.25, 0.75), axis=0)
                axis.fill_between(frames, lower, upper, color=colors[representation], alpha=0.12)
                axis.plot(frames, median, color=colors[representation], linewidth=2.0, label=representation)
                medians[representation] = float(np.median(curves[:, 1:]))
            axis.axhline(1.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.65)
            status = "fallback: no clean slots" if fallback else f"clean cases: {len(selected_by_rep['decoder_dynamic'])}"
            ratio = medians["decoder_static"] / max(medians["decoder_dynamic"], 1.0e-8)
            axis.set_title(f"{spec['short_name']} | {role} | static/dynamic median={ratio:.3f} | {status}")
            axis.set_xlabel("frame (change from t-1 to t)")
            axis.set_ylabel("adjacent change / median dynamic change")
            axis.grid(alpha=0.2)
            axis.legend(ncol=3)
            filename = f"{spec['name']}_{role}.png"
            figure.savefig(output_dir / filename, dpi=150)
            plt.close(figure)
            plots.append(
                {
                    "model": spec["name"], "short_name": spec["short_name"],
                    "role": role, "path": str(Path(output_dir.name) / filename),
                    "static_dynamic_median_ratio": ratio, "fallback": fallback,
                }
            )
    return plots


def build_html(output_dir: Path, result: dict[str, Any]) -> None:
    temporal_cards = "".join(
        "<figure class='curve-card'>"
        f"<img src='{html.escape(item['path'])}' loading='lazy'>"
        f"<figcaption>{html.escape(item['short_name'])} | {item['role']}</figcaption>"
        "</figure>"
        for item in result.get("temporal_plots", [])
    )
    stability_cards = "".join(
        "<figure class='curve-card'>"
        f"<img src='{html.escape(item['path'])}' loading='lazy'>"
        f"<figcaption>{html.escape(item['short_name'])} | {item['role']} | static/dynamic={item['static_dynamic_median_ratio']:.3f}{' | fallback' if item['fallback'] else ''}</figcaption>"
        "</figure>"
        for item in result.get("stability_plots", [])
    )
    overlay_manifest_path = output_dir / "slot_overlays" / "manifest.json"
    overlay_section = ""
    if overlay_manifest_path.is_file():
        overlay_manifest = json.loads(overlay_manifest_path.read_text(encoding="utf-8"))
        by_case: dict[str, list[dict[str, Any]]] = {}
        for record in overlay_manifest["records"]:
            by_case.setdefault(record["case_id"], []).append(record)
        model_order = {
            item["model"]["name"]: index for index, item in enumerate(result["models"])
        }
        case_options = "".join(
            f"<option value='{html.escape(case_id)}'>{html.escape(case_id)}</option>"
            for case_id in by_case
        )
        case_sections = []
        for case_index, (case_id, records) in enumerate(by_case.items()):
            records.sort(key=lambda item: model_order[item["model"]])
            videos = "".join(
                "<figure>"
                f"<video controls muted playsinline preload='metadata' src='{html.escape(record['video'])}'></video>"
                f"<figcaption>{html.escape(record['short_name'])}</figcaption>"
                "</figure>"
                for record in records
            )
            case_sections.append(
                f"<section class='overlay-case{' active' if case_index == 0 else ''}' data-case='{html.escape(case_id)}'>"
                f"<h3>{html.escape(case_id)}</h3><div class='video-grid'>{videos}</div></section>"
            )
        legend = "".join(
            f"<span><i style='background:rgb({color[0]},{color[1]},{color[2]})'></i>S{slot}</span>"
            for slot, color in enumerate(overlay_manifest["palette_rgb"])
        )
        overlay_section = (
            "<h2>逐帧 All-slot Overlay</h2>"
            "<p class='note'>每个 patch 取所有 slot 中 attention 最大者；颜色在所有模型、case 和帧中固定。"
            "这里展示的是 slot 竞争结果，不等价于独立实例分割。</p>"
            f"<div class='overlay-toolbar'><label for='case-select'>case</label><select id='case-select'>{case_options}</select>"
            f"<div class='legend'>{legend}</div></div>{''.join(case_sections)}"
        )
    aggregate_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['model'])}</td><td>{row['intervention']}</td>"
        f"<td>{row['representation']}</td><td>{row['role']}</td>"
        f"<td>{row['clean_pair_count']}/{row['pair_count']}</td>"
        f"<td>{row['initial_aligned_mean_median']:.4f}</td>"
        f"<td>{row['initial_aligned_peak_median']:.4f}</td>"
        f"<td>{row['adjacent_profile_relative_l2_median']:.4f}</td></tr>"
        for row in result["aggregate"]
    )
    quality_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['model'])}</td><td>{row['case_id']}</td><td>{row['role']}</td>"
        f"<td>S{row['slot']}</td><td>{row['recall']:.3f}</td><td>{row['precision']:.3f}</td>"
        f"<td>{row['f1']:.3f}</td><td>{'pass' if row['quality_pass'] else 'low'}</td></tr>"
        for row in result["slot_quality"]
    )
    page = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>xSSC Physics Phase 1</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#111417;color:#eef2f7;font:13px system-ui,sans-serif;letter-spacing:0}}header{{padding:14px 18px;background:#191e23;border-bottom:1px solid #39424b}}main{{max-width:1800px;margin:auto;padding:16px}}h1{{font-size:21px;margin:0 0 5px}}h2{{font-size:17px;margin:24px 0 8px}}h3{{font-size:15px;margin:12px 0 8px}}.note{{border-left:3px solid #22d3ee;background:#172129;padding:9px 11px;color:#cbd5df}}img{{display:block;width:100%;max-width:1500px;background:#fff}}.scroll{{overflow:auto;max-height:600px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:5px 7px;border:1px solid #37414a;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{background:#20272e;position:sticky;top:0}}td{{background:#14191e}}.overlay-toolbar{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:10px 0}}select{{background:#20272e;color:#eef2f7;border:1px solid #53606c;padding:6px 8px}}.legend{{display:flex;gap:7px;flex-wrap:wrap}}.legend span{{display:flex;align-items:center;gap:3px;color:#cbd5df}}.legend i{{display:inline-block;width:11px;height:11px;border:1px solid #d5dbe1}}.overlay-case{{display:none}}.overlay-case.active{{display:block}}.video-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}.curve-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}figure{{margin:0}}video{{display:block;width:100%;background:#050607;aspect-ratio:1/1;object-fit:contain}}figcaption{{padding:6px 2px;color:#d8e0e7;font-weight:600}}.curve-card img{{border:1px solid #39424b}}@media(max-width:900px){{.video-grid,.curve-grid{{grid-template-columns:1fr}}}}</style></head><body><header><h1>xSSC 物理表征 Phase 1</h1><div>单初始条件描述性 pilot；红色=物理参数干预，蓝色=同轨迹外观干预</div></header><main>
<p class='note'>本页不提供显著性或跨 seed 泛化结论。clean pair 要求两端对应物体 slot recall 均达到 {result['quality_recall_threshold']:.2f}；没有 clean pair 时表格明确回退到全样本。</p>
{overlay_section}
<h2>逐帧轨迹分歧率</h2>
<p class='note'>纵轴范围 0–100%；0% 表示两段视频相对首帧的表征演化一致，100% 表示完全不一致。细线是单个视频 pair，粗线是逐帧中位数，阴影是 25%–75% 区间。红线高于蓝线表示物理干预比纯外观干预造成更强的表征分歧。P/A 是 clean/all；fallback 曲线只用于 slot 失败诊断。</p>
<div class='curve-grid'>{temporal_cards}</div>
<h2>相邻帧特征稳定性</h2>
<p class='note'>这里直接计算单个视频 ||f(t)-f(t-1)||/sqrt(D)，再除以该模型、该物体的 dynamic 变化中位数。虚线 1.0 是典型 dynamic 变化；绿色 static 低于 1 表示 static 比 dynamic 稳定。该图不使用近零 step 作分母，因而不会把微小噪声放大成高分歧率。</p>
<div class='curve-grid'>{stability_cards}</div>
<h2>物理与外观响应</h2><img src='physics_vs_appearance.png'>
<h2>聚合结果</h2><div class='scroll'><table><thead><tr><th>model</th><th>intervention</th><th>representation</th><th>role</th><th>clean/all</th><th>mean response</th><th>peak</th><th>adjacent profile</th></tr></thead><tbody>{aggregate_rows}</tbody></table></div>
<h2>Slot 质量</h2><div class='scroll'><table><thead><tr><th>model</th><th>case</th><th>role</th><th>slot</th><th>recall</th><th>precision</th><th>F1</th><th>status</th></tr></thead><tbody>{quality_rows}</tbody></table></div>
</main><script>const select=document.getElementById('case-select');if(select){{select.addEventListener('change',()=>{{document.querySelectorAll('.overlay-case').forEach(section=>section.classList.toggle('active',section.dataset.case===select.value));}});}}</script></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = summarize(root, manifest, args.quality_recall)
    print(f"[complete] models={len(result['models'])} report={root / 'report' / 'index.html'}")


if __name__ == "__main__":
    main()
