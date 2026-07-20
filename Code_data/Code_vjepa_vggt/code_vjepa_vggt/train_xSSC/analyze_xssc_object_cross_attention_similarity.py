#!/usr/bin/env python3
"""Analyze xSSC object cross-attention map similarity and update the viewer."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STAGES = ("early", "middle", "late", "all")


def _cosine_matrix(x: np.ndarray, y: np.ndarray, *, centered: bool) -> np.ndarray:
    x2 = x.astype(np.float64).reshape(x.shape[0], -1)
    y2 = y.astype(np.float64).reshape(y.shape[0], -1)
    if centered:
        x2 = x2 - x2.mean(axis=1, keepdims=True)
        y2 = y2 - y2.mean(axis=1, keepdims=True)
    x2 = x2 / np.maximum(np.linalg.norm(x2, axis=1, keepdims=True), 1.0e-12)
    y2 = y2 / np.maximum(np.linalg.norm(y2, axis=1, keepdims=True), 1.0e-12)
    return x2 @ y2.T


def _js_divergence(x: np.ndarray, y: np.ndarray) -> float:
    p = np.maximum(x.astype(np.float64).reshape(-1), 0.0)
    q = np.maximum(y.astype(np.float64).reshape(-1), 0.0)
    p = p / max(float(p.sum()), 1.0e-12)
    q = q / max(float(q.sum()), 1.0e-12)
    m = 0.5 * (p + q)
    return 0.5 * float(np.sum(p * np.log(np.maximum(p, 1.0e-12) / m))) + 0.5 * float(
        np.sum(q * np.log(np.maximum(q, 1.0e-12) / m))
    )


def _offdiag(matrix: np.ndarray) -> np.ndarray:
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def _plot_matrix(
    matrix: np.ndarray,
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    vmin: float,
    vmax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=160)
    im = ax.imshow(matrix, cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels([f"s{i}" for i in range(matrix.shape[1])])
    ax.set_yticklabels([f"s{i}" for i in range(matrix.shape[0])])
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_stage_bars(rows: list[dict[str, float | str]], path: Path) -> None:
    stages = [str(row["stage"]) for row in rows]
    adjacent = [float(row["same_slot_adjacent_centered_mean"]) for row in rows]
    cross = [float(row["cross_slot_same_frame_centered_mean"]) for row in rows]
    raw_adjacent = [float(row["same_slot_adjacent_raw_mean"]) for row in rows]
    x = np.arange(len(stages))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=160)
    ax.bar(x - width, raw_adjacent, width, label="same-slot adjacent raw")
    ax.bar(x, adjacent, width, label="same-slot adjacent centered")
    ax.bar(x + width, cross, width, label="cross-slot same-frame centered")
    ax.set_ylim(-0.1, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("cosine")
    ax.set_title("Object cross-attention similarity by denoising stage")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_slot_bars(slot_rows: list[dict[str, float | int | str]], path: Path, stage: str) -> None:
    rows = [row for row in slot_rows if str(row["stage"]) == stage]
    slots = [int(row["slot"]) for row in rows]
    raw = [float(row["adjacent_raw_mean"]) for row in rows]
    centered = [float(row["adjacent_centered_mean"]) for row in rows]
    js = [float(row["adjacent_js_mean"]) for row in rows]
    x = np.arange(len(slots))
    width = 0.36
    fig, ax1 = plt.subplots(figsize=(8.2, 4.8), dpi=160)
    ax1.bar(x - width / 2, raw, width, label="raw cosine")
    ax1.bar(x + width / 2, centered, width, label="centered cosine")
    ax1.set_ylim(-0.1, 1.02)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"s{i}" for i in slots])
    ax1.set_ylabel("adjacent-frame cosine")
    ax2 = ax1.twinx()
    ax2.plot(x, js, "-o", color="#8f4f24", label="JS divergence")
    ax2.set_ylabel("JS divergence")
    ax1.set_title(f"Per-slot temporal stability ({stage})")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_pca(maps: np.ndarray, path: Path, stage: str) -> None:
    slots, frames = maps.shape[:2]
    x = maps.reshape(slots * frames, -1).astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=160)
    colors = plt.cm.tab10(np.linspace(0, 1, slots))
    for slot in range(slots):
        idx = slot * frames + np.arange(frames)
        ax.plot(coords[idx, 0], coords[idx, 1], "-o", color=colors[slot], label=f"slot{slot:02d}")
        for frame_id, point_idx in enumerate(idx):
            ax.text(coords[point_idx, 0], coords[point_idx, 1], str(frame_id), fontsize=6)
    ax.set_title(f"PCA of centered object-attention maps ({stage})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _load_stage_maps(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as payload:
        stages: dict[str, np.ndarray] = {}
        for stage in STAGES:
            arrays = []
            slot_id = 0
            while f"{stage}_slot{slot_id:02d}" in payload:
                arrays.append(payload[f"{stage}_slot{slot_id:02d}"].astype(np.float32))
                slot_id += 1
            if arrays:
                stages[stage] = np.stack(arrays, axis=0)
    if "all" not in stages:
        raise RuntimeError(f"{npz_path} does not contain all_slotXX maps")
    return stages


def _analyze_stage(stage: str, maps: np.ndarray) -> tuple[dict[str, float | str], list[dict[str, float | int | str]], np.ndarray]:
    slots, frames = maps.shape[:2]
    slot_rows: list[dict[str, float | int | str]] = []
    raw_adjacent_all: list[float] = []
    centered_adjacent_all: list[float] = []
    js_adjacent_all: list[float] = []
    adjacent_matrices = []
    top1_hits = []

    for slot in range(slots):
        raw = []
        centered = []
        js_values = []
        for frame in range(frames - 1):
            raw.append(float(_cosine_matrix(maps[slot, frame : frame + 1], maps[slot, frame + 1 : frame + 2], centered=False)[0, 0]))
            centered.append(float(_cosine_matrix(maps[slot, frame : frame + 1], maps[slot, frame + 1 : frame + 2], centered=True)[0, 0]))
            js_values.append(_js_divergence(maps[slot, frame], maps[slot, frame + 1]))
        raw_adjacent_all.extend(raw)
        centered_adjacent_all.extend(centered)
        js_adjacent_all.extend(js_values)
        slot_rows.append(
            {
                "stage": stage,
                "slot": int(slot),
                "adjacent_raw_mean": float(np.mean(raw)),
                "adjacent_raw_min": float(np.min(raw)),
                "adjacent_centered_mean": float(np.mean(centered)),
                "adjacent_centered_min": float(np.min(centered)),
                "adjacent_js_mean": float(np.mean(js_values)),
                "adjacent_js_max": float(np.max(js_values)),
                "peak_to_mean": float(maps[slot].max() / max(float(maps[slot].mean()), 1.0e-12)),
                "coefficient_of_variation": float(maps[slot].std() / max(float(maps[slot].mean()), 1.0e-12)),
            }
        )

    cross_values = []
    raw_cross_values = []
    for frame in range(frames):
        raw_mat = _cosine_matrix(maps[:, frame], maps[:, frame], centered=False)
        centered_mat = _cosine_matrix(maps[:, frame], maps[:, frame], centered=True)
        raw_cross_values.extend(float(v) for v in _offdiag(raw_mat))
        cross_values.extend(float(v) for v in _offdiag(centered_mat))

    for frame in range(frames - 1):
        mat = _cosine_matrix(maps[:, frame], maps[:, frame + 1], centered=True)
        adjacent_matrices.append(mat)
        for slot in range(slots):
            top1_hits.append(int(np.argmax(mat[slot])) == slot)
    adjacent_mean_matrix = np.mean(np.stack(adjacent_matrices, axis=0), axis=0)
    summary = {
        "stage": stage,
        "same_slot_adjacent_raw_mean": float(np.mean(raw_adjacent_all)),
        "same_slot_adjacent_raw_min": float(np.min(raw_adjacent_all)),
        "same_slot_adjacent_centered_mean": float(np.mean(centered_adjacent_all)),
        "same_slot_adjacent_centered_min": float(np.min(centered_adjacent_all)),
        "same_slot_adjacent_js_mean": float(np.mean(js_adjacent_all)),
        "same_slot_adjacent_js_max": float(np.max(js_adjacent_all)),
        "cross_slot_same_frame_raw_mean": float(np.mean(raw_cross_values)),
        "cross_slot_same_frame_centered_mean": float(np.mean(cross_values)),
        "cross_slot_same_frame_centered_max": float(np.max(cross_values)),
        "same_slot_adjacent_centered_top1_rate": float(np.mean(top1_hits)),
        "mean_peak_to_mean": float(np.mean([row["peak_to_mean"] for row in slot_rows])),
        "mean_coefficient_of_variation": float(np.mean([row["coefficient_of_variation"] for row in slot_rows])),
    }
    return summary, slot_rows, adjacent_mean_matrix


def _write_html_section(attention_dir: Path, summary: dict[str, object]) -> None:
    index_path = attention_dir / "index.html"
    page = index_path.read_text(encoding="utf-8")
    marker_start = "<!-- object-attention-similarity:start -->"
    marker_end = "<!-- object-attention-similarity:end -->"
    if marker_start in page and marker_end in page:
        before = page.split(marker_start, 1)[0]
        after = page.split(marker_end, 1)[1]
    else:
        before = page.replace("<section><h2>early</h2>", marker_start + "\n<section><h2>early</h2>", 1).split(marker_start, 1)[0]
        after = page.split("<section><h2>early</h2>", 1)[1]
        after = "<section><h2>early</h2>" + after

    stage_rows = summary["stage_rows"]
    row_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['stage']))}</td>"
        f"<td>{float(row['same_slot_adjacent_raw_mean']):.4f}</td>"
        f"<td>{float(row['same_slot_adjacent_centered_mean']):.4f}</td>"
        f"<td>{float(row['cross_slot_same_frame_centered_mean']):.4f}</td>"
        f"<td>{float(row['same_slot_adjacent_js_mean']):.6f}</td>"
        f"<td>{float(row['same_slot_adjacent_centered_top1_rate']):.3f}</td>"
        "</tr>"
        for row in stage_rows
    )
    section = f"""{marker_start}
<section>
<h2>Slot Similarity Analysis</h2>
<p>Raw cosine measures overall attention-vector similarity; centered cosine subtracts each frame map mean and is more sensitive to spatial attention-shape differences.</p>
<div class='grid'>
<figure><img src='similarity/stage_similarity_bars.png'><figcaption>stage-level similarity summary</figcaption></figure>
<figure><img src='similarity/all_adjacent_slot_to_slot_centered_cosine.png'><figcaption>all-stage adjacent slot-to-slot centered cosine</figcaption></figure>
<figure><img src='similarity/all_slot_adjacent_stability.png'><figcaption>all-stage per-slot temporal stability</figcaption></figure>
<figure><img src='similarity/all_pca_centered_attention_maps.png'><figcaption>all-stage PCA of centered maps</figcaption></figure>
</div>
<table>
<thead><tr><th>stage</th><th>same-slot adjacent raw cosine</th><th>same-slot adjacent centered cosine</th><th>cross-slot same-frame centered cosine</th><th>adjacent JS</th><th>same-slot top1</th></tr></thead>
<tbody>{row_html}</tbody>
</table>
<p><a href='similarity/object_attention_similarity_summary.json'>similarity summary JSON</a> | <a href='similarity/stage_similarity.csv'>stage CSV</a> | <a href='similarity/slot_similarity.csv'>slot CSV</a></p>
</section>
{marker_end}
"""
    page = before + section + after
    if "table{" not in page:
        page = page.replace(
            "</style>",
            "table{width:100%;border-collapse:collapse;background:#fff;margin-top:12px}"
            "th,td{border:1px solid #d7d2c8;padding:7px;text-align:left}"
            "th{background:#ece7dc}</style>",
        )
    index_path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-dir", type=Path, required=True)
    parser.add_argument("--update-html", action="store_true")
    args = parser.parse_args()

    attention_dir = args.attention_dir.expanduser().resolve()
    npz_path = attention_dir / "xssc_object_cross_attention_maps_fp16.npz"
    output_dir = attention_dir / "similarity"
    output_dir.mkdir(parents=True, exist_ok=True)

    stages = _load_stage_maps(npz_path)
    stage_rows: list[dict[str, float | str]] = []
    slot_rows: list[dict[str, float | int | str]] = []
    matrix_paths: dict[str, str] = {}
    for stage in STAGES:
        if stage not in stages:
            continue
        stage_summary, stage_slot_rows, adjacent_matrix = _analyze_stage(stage, stages[stage])
        stage_rows.append(stage_summary)
        slot_rows.extend(stage_slot_rows)
        matrix_path = output_dir / f"{stage}_adjacent_slot_to_slot_centered_cosine.png"
        _plot_matrix(
            adjacent_matrix,
            matrix_path,
            title=f"Adjacent slot-to-slot centered cosine ({stage})",
            xlabel="slot at latent t+1",
            ylabel="slot at latent t",
            vmin=-1.0,
            vmax=1.0,
        )
        _plot_slot_bars(stage_slot_rows, output_dir / f"{stage}_slot_adjacent_stability.png", stage)
        _plot_pca(stages[stage], output_dir / f"{stage}_pca_centered_attention_maps.png", stage)
        matrix_paths[stage] = matrix_path.name

    _plot_stage_bars(stage_rows, output_dir / "stage_similarity_bars.png")

    stage_csv = output_dir / "stage_similarity.csv"
    with stage_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stage_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stage_rows)

    slot_csv = output_dir / "slot_similarity.csv"
    with slot_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(slot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(slot_rows)

    summary = {
        "attention_dir": str(attention_dir),
        "maps_npz": str(npz_path),
        "stage_rows": stage_rows,
        "slot_rows": slot_rows,
        "matrix_pngs": matrix_paths,
        "outputs": {
            "stage_csv": str(stage_csv),
            "slot_csv": str(slot_csv),
            "stage_bars": str(output_dir / "stage_similarity_bars.png"),
            "all_adjacent_matrix": str(output_dir / "all_adjacent_slot_to_slot_centered_cosine.png"),
            "all_slot_bars": str(output_dir / "all_slot_adjacent_stability.png"),
            "all_pca": str(output_dir / "all_pca_centered_attention_maps.png"),
        },
    }
    summary_path = output_dir / "object_attention_similarity_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.update_html:
        _write_html_section(attention_dir, summary)
    print(json.dumps({"summary": str(summary_path), "updated_html": bool(args.update_html)}, indent=2))


if __name__ == "__main__":
    main()
