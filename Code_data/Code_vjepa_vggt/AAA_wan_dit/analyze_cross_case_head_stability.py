#!/usr/bin/env python3
"""Measure per-model head-role stability across test_5 cases."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from analyze_case_moving_ball_heads import _circle_tokens
from analyze_multiblock_ball_query_heads import (
    ROLE_LABELS,
    _feature_rows,
    _role_scores,
    _sample_labels,
)


MODELS = ("wan_lora", "xssc", "physrvg")
BLOCKS = tuple(range(30))
STEPS = (5, 15, 25, 35)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query-map", type=Path)
    query_group.add_argument(
        "--query-map-root",
        type=Path,
        help="Directory containing <model>/query_map.json for each model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _trajectory_tokens(item: dict[str, Any]) -> list[np.ndarray]:
    if "query_coords_per_time" in item:
        return [
            np.asarray(coords, dtype=np.int64).reshape(-1, 3)
            for coords in item["query_coords_per_time"]
        ]
    shape = tuple(int(value) for value in item["frame_shape"])
    output = []
    for time, point in enumerate(item["trajectory"]):
        if not point.get("valid", True):
            output.append(np.empty((0, 3), dtype=np.int64))
            continue
        output.append(
            _circle_tokens(
                latent_t=time,
                circle=np.asarray(
                    [
                        point["cx"],
                        point["cy"],
                        max(16.0, min(48.0, point["radius"])),
                    ],
                    dtype=np.float64,
                ),
                frame_shape=shape,
                grid=(13, 16, 28),
            )
        )
    return output


def _case_group(case: str) -> str:
    if case.startswith("0613pybullet"):
        return "pybullet"
    if case.startswith("phyco_kubric"):
        return "phyco"
    return "physiciq"


def _read_case_block(
    root: Path,
    model: str,
    case: str,
    block: int,
    trajectory: list[np.ndarray] | None,
) -> dict[str, Any]:
    summary_path = (
        root / f"block{block:02d}" / "matrices" / model / case / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    samples = []
    for entry in summary["steps"]:
        if "features_npz" in entry:
            matrix_path = (
                summary_path.parent / entry["directory"] / entry["features_npz"]
            )
            with np.load(matrix_path) as arrays:
                features = {
                    name: arrays[name]
                    for name in (
                        "entropy",
                        "same_frame_mass",
                        "local_mass",
                        "first_frame_mass",
                        "history_bias",
                        "mean_time_distance",
                        "aligned_enrichment",
                        "cross_ball_enrichment",
                    )
                }
            cosine = float("nan")
        else:
            if trajectory is None:
                raise ValueError(
                    f"{summary_path}: legacy matrix capture requires query trajectory"
                )
            matrix_path = summary_path.parent / entry["directory"] / entry["matrix_npz"]
            with np.load(matrix_path) as arrays:
                attention = arrays["attention"]
                query_coords = arrays["query_coords"]
            features, cosine = _feature_rows(
                attention, query_coords=query_coords, trajectory_tokens=trajectory
            )
        scores = _role_scores(features)
        samples.append(
            {
                "step": int(entry["step_number_one_based"]),
                "features": features,
                "scores": scores,
                "labels": _sample_labels(scores),
                "cosine": cosine,
            }
        )
    aggregate_features = {
        key: np.stack([sample["features"][key] for sample in samples]).mean(0)
        for key in samples[0]["features"]
    }
    aggregate_scores = _role_scores(aggregate_features)
    roles = list(ROLE_LABELS)
    matrix = np.stack([aggregate_scores[role] for role in roles], axis=1)
    order = np.argsort(matrix, axis=1)
    primary = [roles[int(index)] for index in order[:, -1]]
    secondary = [roles[int(index)] for index in order[:, -2]]
    margin = np.take_along_axis(matrix, order[:, -1:], axis=1)[:, 0] - np.take_along_axis(
        matrix, order[:, -2:-1], axis=1
    )[:, 0]
    labels = np.asarray([sample["labels"] for sample in samples])
    step_consistency = np.asarray(
        [np.mean(labels[:, head] == primary[head]) for head in range(24)]
    )
    return {
        "primary": primary,
        "secondary": secondary,
        "margin": margin,
        "step_consistency": step_consistency,
        "mean_head_cosine": float(np.mean([sample["cosine"] for sample in samples])),
    }


def _stability_rows(
    model: str,
    case_records: dict[tuple[str, int], dict[str, Any]],
    cases: list[str],
) -> list[dict[str, Any]]:
    rows = []
    synthetic = [case for case in cases if _case_group(case) != "physiciq"]
    for block in BLOCKS:
        for head in range(24):
            labels = [case_records[(case, block)]["primary"][head] for case in cases]
            counts = Counter(labels)
            role, count = counts.most_common(1)[0]
            ranked_counts = counts.most_common(2)
            second_role, second_count = (
                ranked_counts[1] if len(ranked_counts) > 1 else (role, 0)
            )
            synthetic_labels = [
                case_records[(case, block)]["primary"][head] for case in synthetic
            ]
            synthetic_counts = Counter(synthetic_labels)
            synthetic_role, synthetic_count = synthetic_counts.most_common(1)[0]
            margins = [
                float(case_records[(case, block)]["margin"][head]) for case in cases
            ]
            step_values = [
                float(case_records[(case, block)]["step_consistency"][head])
                for case in cases
            ]
            consistency = count / len(cases)
            synthetic_consistency = synthetic_count / len(synthetic)
            if consistency >= 0.70 and float(np.median(margins)) >= 0.10:
                verdict = "稳定"
            elif consistency >= 0.50:
                verdict = "中等"
            else:
                verdict = "不稳定"
            rows.append(
                {
                    "model": model,
                    "block": block,
                    "head": head,
                    "dominant_role": role,
                    "dominant_role_label": ROLE_LABELS[role],
                    "cross_case_consistency": consistency,
                    "synthetic_dominant_role": synthetic_role,
                    "synthetic_case_consistency": synthetic_consistency,
                    "secondary_role": second_role,
                    "secondary_case_fraction": second_count / len(cases),
                    "median_role_margin": float(np.median(margins)),
                    "mean_step_consistency": float(np.mean(step_values)),
                    "verdict": verdict,
                    **{
                        f"{name}_case_fraction": counts.get(name, 0) / len(cases)
                        for name in ROLE_LABELS
                    },
                }
            )
    return rows


def _plot(rows: list[dict[str, Any]], output: Path, model: str) -> None:
    matrix = np.zeros((30, 24), dtype=np.float64)
    for row in rows:
        matrix[int(row["block"]), int(row["head"])] = row["cross_case_consistency"]
    fig, axis = plt.subplots(figsize=(16, 9))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axis.set_title(f"{model}: head-role consistency across 20 unique test_5 cases")
    axis.set_xlabel("Head")
    axis.set_ylabel("Block")
    axis.set_xticks(range(24))
    axis.set_yticks(range(30))
    fig.colorbar(image, ax=axis, label="Cross-case consistency")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _block_rows(model: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for block in BLOCKS:
        current = [row for row in rows if int(row["block"]) == block]
        roles = Counter(row["dominant_role"] for row in current)
        output.append(
            {
                "model": model,
                "block": block,
                "mean_cross_case_consistency": float(
                    np.mean([row["cross_case_consistency"] for row in current])
                ),
                "median_cross_case_consistency": float(
                    np.median([row["cross_case_consistency"] for row in current])
                ),
                "stable_head_fraction": float(
                    np.mean([row["verdict"] == "稳定" for row in current])
                ),
                "mean_synthetic_case_consistency": float(
                    np.mean([row["synthetic_case_consistency"] for row in current])
                ),
                "mean_step_consistency": float(
                    np.mean([row["mean_step_consistency"] for row in current])
                ),
                "dominant_role": roles.most_common(1)[0][0],
                **{f"{role.lower()}_heads": roles.get(role, 0) for role in ROLE_LABELS},
            }
        )
    return output


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if args.query_map_root is not None:
        query_root = args.query_map_root.expanduser().resolve()
        query_payloads = {
            model: json.loads(
                (query_root / model / "query_map.json").read_text(encoding="utf-8")
            )
            for model in MODELS
        }
    else:
        query_payload = json.loads(
            args.query_map.expanduser().resolve().read_text(encoding="utf-8")
        )
        query_payloads = {model: query_payload for model in MODELS}
    cases = list(query_payloads[MODELS[0]]["cases"])
    expected_cases = set(cases)
    for model, payload in query_payloads.items():
        if set(payload["cases"]) != expected_cases:
            raise ValueError(f"{model} query map has a different case set")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_block_rows = []
    report_sections = []
    for model in MODELS:
        query_payload = query_payloads[model]
        records = {}
        for case in cases:
            trajectory = _trajectory_tokens(query_payload["cases"][case])
            for block in BLOCKS:
                records[(case, block)] = _read_case_block(
                    root, model, case, block, trajectory
                )
        rows = _stability_rows(model, records, cases)
        block_rows = _block_rows(model, rows)
        all_rows.extend(rows)
        all_block_rows.extend(block_rows)
        model_dir = output_dir / model
        model_dir.mkdir(exist_ok=True)
        with (model_dir / "head_stability.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (model_dir / "block_stability.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(block_rows[0]))
            writer.writeheader()
            writer.writerows(block_rows)
        _plot(rows, model_dir / "head_stability_heatmap.png", model)
        verdicts = Counter(row["verdict"] for row in rows)
        role_counts = Counter(row["dominant_role"] for row in rows)
        stable = sorted(
            rows,
            key=lambda row: (
                row["cross_case_consistency"],
                row["mean_step_consistency"],
                row["median_role_margin"],
            ),
            reverse=True,
        )
        unstable = sorted(rows, key=lambda row: row["cross_case_consistency"])
        stable_blocks = sorted(
            block_rows,
            key=lambda row: (
                row["mean_cross_case_consistency"],
                row["stable_head_fraction"],
            ),
            reverse=True,
        )
        unstable_blocks = list(reversed(stable_blocks))
        report_sections.append(
            f"""## {model}

- 稳定/中等/不稳定：{verdicts['稳定']}/{verdicts['中等']}/{verdicts['不稳定']}
- 主角色数量：{dict(sorted(role_counts.items()))}
- 跨 case 一致率均值/中位数：{np.mean([r['cross_case_consistency'] for r in rows]):.3f}/{np.median([r['cross_case_consistency'] for r in rows]):.3f}
- 合成 case 一致率均值：{np.mean([r['synthetic_case_consistency'] for r in rows]):.3f}
- 最稳定 blocks：{", ".join(f"B{r['block']:02d}({r['mean_cross_case_consistency']:.1%}, stable heads {r['stable_head_fraction']:.0%})" for r in stable_blocks[:6])}
- 最不稳定 blocks：{", ".join(f"B{r['block']:02d}({r['mean_cross_case_consistency']:.1%}, stable heads {r['stable_head_fraction']:.0%})" for r in unstable_blocks[:6])}
- 最稳定：{", ".join(f"B{r['block']:02d}H{r['head']:02d} {r['dominant_role']}({r['cross_case_consistency']:.0%})" for r in stable[:12])}
- 最不稳定：{", ".join(f"B{r['block']:02d}H{r['head']:02d}({r['cross_case_consistency']:.0%})" for r in unstable[:12])}
"""
        )
    with (output_dir / "all_models_head_stability.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (output_dir / "all_models_block_stability.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_block_rows[0]))
        writer.writeheader()
        writer.writerows(all_block_rows)
    report = f"""# test_5 全 Block/Head 跨 Case 稳定性

共 {len(cases)} 个唯一 case；每个模型独立统计 30 blocks × 24 heads。每个 case
先聚合去噪步 5/15/25/35 的角色得分，再统计同一 `(block, head)` 在不同 case
上的主角色一致率。稳定要求一致率至少 70%，且角色分数中位 margin 至少 0.10。

{chr(10).join(report_sections)}

注意：角色是基于运动目标 query 的描述性启发式标签，不是因果证明。PhysicIQ
自动目标轨迹比合成数据噪声更大，因此 CSV 同时给出全部 case 和仅合成 case
的一致率。
"""
    (output_dir / "cross_case_head_stability.md").write_text(
        report, encoding="utf-8"
    )
    (output_dir / "cross_case_head_stability.json").write_text(
        json.dumps(
            {"cases": cases, "role_labels": ROLE_LABELS, "heads": all_rows},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_dir / "cross_case_head_stability.md")


if __name__ == "__main__":
    main()
