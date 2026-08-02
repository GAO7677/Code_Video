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
                quality_rows.append(
                    {
                        "model": spec["name"], "case_id": case["case_id"], "family": case["family"], "role": role,
                        "slot": int(item["selected"][role_id]), "recall": float(item["recall"][role_id]),
                        "precision": float(item["precision"][role_id]), "f1": float(item["f1"][role_id]),
                        "quality_pass": bool(item["recall"][role_id] >= quality_recall),
                    }
                )
        for left_id, right_id, intervention in pair_specs:
            left = loaded[left_id]
            right = loaded[right_id]
            for representation in REPRESENTATIONS:
                left_features = left["representations"][representation]
                right_features = right["representations"][representation]
                curves = initial_aligned_curve(left_features, right_features)
                for role_id, role in enumerate(ROLES):
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
                            "left_quality_pass": bool(left["recall"][role_id] >= quality_recall),
                            "right_quality_pass": bool(right["recall"][role_id] >= quality_recall),
                        }
                    )
            centroid_curve = np.linalg.norm(
                (right["centroids"] - right["centroids"][:1]) - (left["centroids"] - left["centroids"][:1]), axis=-1
            ) / 16.0
            for role_id, role in enumerate(ROLES):
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
                        "left_quality_pass": bool(left["recall"][role_id] >= quality_recall),
                        "right_quality_pass": bool(right["recall"][role_id] >= quality_recall),
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
    plot_effects(report_dir / "physics_vs_appearance.png", aggregate_rows, manifest["models"])
    result = {
        "status": "descriptive_single-initial-condition pilot",
        "quality_recall_threshold": quality_recall,
        "models": model_results,
        "slot_quality": quality_rows,
        "pairs": pair_rows,
        "aggregate": aggregate_rows,
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


def build_html(output_dir: Path, result: dict[str, Any]) -> None:
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
*{{box-sizing:border-box}}body{{margin:0;background:#111417;color:#eef2f7;font:13px system-ui,sans-serif;letter-spacing:0}}header{{padding:14px 18px;background:#191e23;border-bottom:1px solid #39424b}}main{{max-width:1800px;margin:auto;padding:16px}}h1{{font-size:21px;margin:0 0 5px}}h2{{font-size:17px;margin:24px 0 8px}}.note{{border-left:3px solid #22d3ee;background:#172129;padding:9px 11px;color:#cbd5df}}img{{display:block;width:100%;max-width:1500px;background:#fff}}.scroll{{overflow:auto;max-height:600px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:5px 7px;border:1px solid #37414a;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{background:#20272e;position:sticky;top:0}}td{{background:#14191e}}</style></head><body><header><h1>xSSC 物理表征 Phase 1</h1><div>单初始条件描述性 pilot；红色=物理参数干预，蓝色=同轨迹外观干预</div></header><main>
<p class='note'>本页不提供显著性或跨 seed 泛化结论。clean pair 要求两端对应物体 slot recall 均达到 {result['quality_recall_threshold']:.2f}；没有 clean pair 时表格明确回退到全样本。</p>
<h2>物理与外观响应</h2><img src='physics_vs_appearance.png'>
<h2>聚合结果</h2><div class='scroll'><table><thead><tr><th>model</th><th>intervention</th><th>representation</th><th>role</th><th>clean/all</th><th>mean response</th><th>peak</th><th>adjacent profile</th></tr></thead><tbody>{aggregate_rows}</tbody></table></div>
<h2>Slot 质量</h2><div class='scroll'><table><thead><tr><th>model</th><th>case</th><th>role</th><th>slot</th><th>recall</th><th>precision</th><th>F1</th><th>status</th></tr></thead><tbody>{quality_rows}</tbody></table></div>
</main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = summarize(root, manifest, args.quality_recall)
    print(f"[complete] models={len(result['models'])} report={root / 'report' / 'index.html'}")


if __name__ == "__main__":
    main()
