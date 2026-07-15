#!/usr/bin/env python3
"""Aggregate context/future noun-attention boundary metrics across cases."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SHAPE_JS_THRESHOLD = 0.08
SHAPE_COSINE_THRESHOLD = 0.70
CENTROID_JUMP_THRESHOLD = 0.08
HIGH_RESPONSE_LAYER_COUNT = 10
TRACKED_LAYERS = (7, 8, 10, 11, 12, 25, 26)
REMAINING_STEPS = (40, 30, 20, 10, 1)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} OUTPUT_ROOT")
    root = Path(sys.argv[1]).expanduser().resolve()
    manifests = sorted(root.rglob("*_text_noun_attention/manifest.json"))
    if not manifests:
        raise SystemExit(f"no attention manifests found under {root}")

    combined: list[dict[str, object]] = []
    per_step: list[dict[str, object]] = []
    output_dirs: dict[str, Path] = {}
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        case = str(manifest["case"])
        output_dirs[case] = manifest_path.parent
        summary_rows = _read_csv(Path(manifest["boundary_metrics"]["summary_csv"]))
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in summary_rows:
            grouped[row["noun"]].append(row)
        for noun, rows in grouped.items():
            ranked = sorted(
                rows,
                key=lambda item: float(item["context_top1pct_mean"]),
                reverse=True,
            )
            response_ranks = {
                int(row["layer_id"]): rank for rank, row in enumerate(ranked, start=1)
            }
            for row in rows:
                layer = int(row["layer_id"])
                js_mean = float(row["js_mean"])
                cosine_mean = float(row["cosine_mean"])
                centroid_jump_mean = float(row["centroid_jump_mean"])
                high_response = response_ranks[layer] <= HIGH_RESPONSE_LAYER_COUNT
                shape_break = (
                    js_mean >= SHAPE_JS_THRESHOLD
                    and cosine_mean <= SHAPE_COSINE_THRESHOLD
                )
                large_centroid_jump = centroid_jump_mean >= CENTROID_JUMP_THRESHOLD
                confirmed_break = high_response and (shape_break or large_centroid_jump)
                combined.append(
                    {
                        "case": case,
                        "noun": noun,
                        "layer_id": layer,
                        "response_rank": response_ranks[layer],
                        "high_response_top10": int(high_response),
                        "context_top1pct_mean": float(row["context_top1pct_mean"]),
                        "js_mean": js_mean,
                        "js_max": float(row["js_max"]),
                        "cosine_mean": cosine_mean,
                        "cosine_min": float(row["cosine_min"]),
                        "centroid_jump_mean": centroid_jump_mean,
                        "centroid_jump_max": float(row["centroid_jump_max"]),
                        "mass_ratio_mean": float(row["mass_ratio_mean"]),
                        "shape_break": int(shape_break),
                        "large_centroid_jump": int(large_centroid_jump),
                        "confirmed_break": int(confirmed_break),
                    }
                )
        for row in _read_csv(Path(manifest["boundary_metrics"]["per_step_csv"])):
            per_step.append(
                {
                    "case": case,
                    "noun": row["noun"],
                    "layer_id": int(row["layer_id"]),
                    "remaining_steps": int(row["remaining_steps"]),
                    "js_divergence": float(row["js_divergence"]),
                    "cosine_similarity": float(row["cosine_similarity"]),
                    "centroid_jump": float(row["centroid_jump"]),
                }
            )

    combined_path = root / "combined_boundary_metrics_summary.csv"
    _write_csv(combined_path, combined)

    layer_step_rows: list[dict[str, object]] = []
    for layer in TRACKED_LAYERS:
        for remaining in REMAINING_STEPS:
            group = [
                row
                for row in per_step
                if row["layer_id"] == layer and row["remaining_steps"] == remaining
            ]
            layer_step_rows.append(
                {
                    "layer_id": layer,
                    "remaining_steps": remaining,
                    "num_noun_cases": len(group),
                    "js_mean": float(np.mean([row["js_divergence"] for row in group])),
                    "cosine_mean": float(np.mean([row["cosine_similarity"] for row in group])),
                    "centroid_jump_mean": float(np.mean([row["centroid_jump"] for row in group])),
                }
            )
    layer_step_path = root / "layer_boundary_progress.csv"
    _write_csv(layer_step_path, layer_step_rows)

    break_counts = Counter(
        int(row["layer_id"]) for row in combined if int(row["confirmed_break"])
    )
    noun_break_counts = Counter(
        str(row["noun"]) for row in combined if int(row["confirmed_break"])
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for layer in TRACKED_LAYERS:
        rows = [row for row in layer_step_rows if row["layer_id"] == layer]
        x = list(range(len(REMAINING_STEPS)))
        axes[0].plot(x, [row["js_mean"] for row in rows], marker="o", label=f"L{layer}")
        axes[1].plot(x, [row["cosine_mean"] for row in rows], marker="o", label=f"L{layer}")
        axes[2].plot(x, [row["centroid_jump_mean"] for row in rows], marker="o", label=f"L{layer}")
    for axis, title, ylabel in zip(
        axes,
        ("Boundary JS", "Boundary cosine", "Boundary centroid jump"),
        ("JS divergence", "cosine similarity", "normalized distance"),
    ):
        axis.set_xticks(range(len(REMAINING_STEPS)), [str(value) for value in REMAINING_STEPS])
        axis.set_xlabel("denoising steps remaining")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)
    progress_plot = root / "layer_boundary_progress.png"
    figure.savefig(progress_plot, dpi=180)
    plt.close(figure)

    layers = sorted(break_counts)
    figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    axis.bar([str(layer) for layer in layers], [break_counts[layer] for layer in layers])
    axis.set_xlabel("DiT layer")
    axis.set_ylabel("confirmed noun-case boundary breaks")
    axis.set_title("High-response boundary discontinuities")
    axis.grid(axis="y", alpha=0.25)
    break_plot = root / "confirmed_break_count_by_layer.png"
    figure.savefig(break_plot, dpi=180)
    plt.close(figure)

    report_lines = [
        "# Text Attention Boundary Analysis",
        "",
        "## Definition",
        "",
        "Metrics compare raw latent attention slice `t=1` (last pure context center) "
        "against `t=2` (first future center). Spatial maps are normalized before JS, "
        "cosine, and centroid calculations.",
        "",
        "A layer is marked as a confirmed boundary break only when it is among the "
        f"top {HIGH_RESPONSE_LAYER_COUNT} context-response layers for that noun and either:",
        "",
        f"- shape break: `JS >= {SHAPE_JS_THRESHOLD}` and `cosine <= {SHAPE_COSINE_THRESHOLD}`; or",
        f"- spatial jump: normalized centroid distance `>= {CENTROID_JUMP_THRESHOLD}`.",
        "",
        "This is a diagnostic threshold, not a training target or a claim that all "
        "motion-related centroid displacement is erroneous.",
        "",
        "## Confirmed Breaks",
        "",
    ]
    for case in sorted(output_dirs):
        report_lines.append(f"### {case}")
        report_lines.append("")
        case_rows = [
            row for row in combined if row["case"] == case and int(row["confirmed_break"])
        ]
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in case_rows:
            grouped[str(row["noun"])].append(row)
        for noun in sorted({str(row["noun"]) for row in combined if row["case"] == case}):
            rows = sorted(grouped.get(noun, []), key=lambda item: int(item["layer_id"]))
            if not rows:
                report_lines.append(f"- `{noun}`: no confirmed high-response break")
                continue
            formatted = ", ".join(
                f"L{int(row['layer_id'])} (JS={float(row['js_mean']):.3f}, "
                f"cos={float(row['cosine_mean']):.3f}, jump={float(row['centroid_jump_mean']):.3f})"
                for row in rows
            )
            report_lines.append(f"- `{noun}`: {formatted}")
        report_lines.append("")

    report_lines.extend(
        [
            "## Cross-Case Pattern",
            "",
            "Confirmed-break counts by layer:",
            "",
        ]
    )
    for layer, count in break_counts.most_common():
        report_lines.append(f"- Layer {layer}: {count}")
    if break_counts:
        layer_pattern = ", ".join(
            f"L{layer} ({count})" for layer, count in break_counts.most_common(5)
        )
        noun_pattern = ", ".join(
            f"`{noun}` ({count})" for noun, count in noun_break_counts.most_common(5)
        )
        report_lines.extend(
            [
                "",
                f"Most recurrent layers in this run: {layer_pattern}.",
                "",
                f"Most recurrent noun labels in this run: {noun_pattern}.",
                "",
            ]
        )
    else:
        report_lines.extend(
            [
                "",
                "No noun-layer pair met the confirmed-break thresholds in this run.",
                "",
            ]
        )
    report_lines.extend(
        [
            "These counts describe only the manifests under this output root. Compare "
            "the per-step JS, cosine, and centroid curves before assigning a semantic "
            "cause to a recurrent layer.",
            "",
            "## Artifacts",
            "",
            f"- Combined metrics: `{combined_path}`",
            f"- Layer progress CSV: `{layer_step_path}`",
            f"- Layer progress plot: `{progress_plot}`",
            f"- Confirmed break counts: `{break_plot}`",
            "- Per-case shared-scale overlays and boundary contact sheets are in each "
            "  `<case>_text_noun_attention/noun_<noun>` directory.",
            "",
        ]
    )
    report_path = root / "attention_boundary_analysis_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
