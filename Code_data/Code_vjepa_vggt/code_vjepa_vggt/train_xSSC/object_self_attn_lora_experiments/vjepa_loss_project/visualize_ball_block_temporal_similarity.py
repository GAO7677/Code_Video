#!/usr/bin/env python3
"""Plot frame-aligned pairwise V-JEPA cosine similarity over time."""

from __future__ import annotations

import argparse
import csv
import html
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/vjepa_ball_block_pairwise/"
    "ball_block49_native_rect_vitl_with_raw_20260808"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize pairwise V-JEPA cosine similarity across 49 frames."
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--page-name", default="temporal_similarity.html")
    return parser.parse_args()


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def token_cosine_curve(feature_a: Path, feature_b: Path) -> tuple[np.ndarray, np.ndarray]:
    a = np.load(feature_a, mmap_mode="r")
    b = np.load(feature_b, mmap_mode="r")
    if a.shape != b.shape or a.ndim != 4:
        raise ValueError(f"Feature shape mismatch: {a.shape} vs {b.shape}")
    means = np.empty(int(a.shape[0]), dtype=np.float32)
    stds = np.empty(int(a.shape[0]), dtype=np.float32)
    for temporal_index in range(int(a.shape[0])):
        aa = np.asarray(a[temporal_index], dtype=np.float32)
        bb = np.asarray(b[temporal_index], dtype=np.float32)
        aa /= np.maximum(np.linalg.norm(aa, axis=-1, keepdims=True), 1e-8)
        bb /= np.maximum(np.linalg.norm(bb, axis=-1, keepdims=True), 1e-8)
        patch_cosine = np.einsum("hwd,hwd->hw", aa, bb, optimize=True)
        means[temporal_index] = float(patch_cosine.mean(dtype=np.float64))
        stds[temporal_index] = float(patch_cosine.std(dtype=np.float64))
    return means, stds


def expand_tubelets(values: np.ndarray, num_frames: int) -> np.ndarray:
    frame_values = np.empty(num_frames, dtype=np.float32)
    for frame_index in range(num_frames):
        frame_values[frame_index] = values[min(frame_index // 2, len(values) - 1)]
    return frame_values


def add_frame_axis(axis, fps: float) -> None:
    top = axis.secondary_xaxis(
        "top",
        functions=(lambda seconds: seconds * fps, lambda frames: frames / fps),
    )
    top.set_xlabel("Frame index")


def plot_all_pairs(
    records: list[dict[str, Any]],
    *,
    fps: float,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(18, 10), constrained_layout=True)
    colors = plt.cm.turbo(np.linspace(0.02, 0.98, len(records)))
    for color, record in zip(colors, records):
        axis.plot(
            record["time_seconds"],
            record["frame_similarity"],
            color=color,
            linewidth=1.8,
            alpha=0.82,
            label=record["pair_label"],
        )
    axis.set_title("Pairwise V-JEPA feature similarity over time", fontsize=20)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Mean patch cosine similarity")
    axis.grid(True, alpha=0.2)
    add_frame_axis(axis, fps)
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        frameon=False,
        ncol=1,
    )
    figure.savefig(output_path, dpi=150, facecolor="#f8f6ee")
    plt.close(figure)


def plot_small_multiples(
    records: list[dict[str, Any]],
    *,
    output_path: Path,
) -> None:
    all_values = np.concatenate(
        [np.asarray(record["frame_similarity"], dtype=np.float32) for record in records]
    )
    margin = max(0.005, float(all_values.max() - all_values.min()) * 0.08)
    ymin = float(all_values.min() - margin)
    ymax = min(1.0, float(all_values.max() + margin))
    figure, axes = plt.subplots(7, 4, figsize=(18, 23), sharex=True, sharey=True)
    for axis, record in zip(axes.flat, records):
        axis.plot(
            record["time_seconds"],
            record["frame_similarity"],
            color="#bc4b2d",
            linewidth=1.8,
        )
        axis.fill_between(
            record["time_seconds"],
            np.asarray(record["frame_similarity"]) - np.asarray(record["frame_patch_std"]),
            np.asarray(record["frame_similarity"]) + np.asarray(record["frame_patch_std"]),
            color="#bc4b2d",
            alpha=0.10,
            linewidth=0,
        )
        axis.set_title(record["pair_label"], fontsize=9)
        axis.set_ylim(ymin, ymax)
        axis.grid(True, alpha=0.18)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Cosine")
    figure.suptitle(
        "Each video pair: mean spatial-token similarity (band = patch std)",
        fontsize=19,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=150, facecolor="#f8f6ee")
    plt.close(figure)


def plot_aggregate(
    records: list[dict[str, Any]],
    *,
    fps: float,
    output_path: Path,
) -> None:
    curves = np.asarray([record["frame_similarity"] for record in records], dtype=np.float32)
    times = np.asarray(records[0]["time_seconds"], dtype=np.float32)
    mean = curves.mean(axis=0)
    std = curves.std(axis=0)
    minimum = curves.min(axis=0)
    maximum = curves.max(axis=0)
    figure, axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
    axis.fill_between(times, minimum, maximum, color="#315f4b", alpha=0.12, label="pair min-max")
    axis.fill_between(times, mean - std, mean + std, color="#315f4b", alpha=0.24, label="mean +/- std")
    axis.plot(times, mean, color="#173f32", linewidth=2.8, label="mean across 28 pairs")
    axis.set_title("Aggregate pairwise V-JEPA similarity", fontsize=20)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Mean patch cosine similarity")
    axis.grid(True, alpha=0.2)
    axis.legend(frameon=False)
    add_frame_axis(axis, fps)
    figure.savefig(output_path, dpi=160, facecolor="#f8f6ee")
    plt.close(figure)


def sparkline(values: list[float]) -> str:
    width, height, padding = 250.0, 58.0, 4.0
    array = np.asarray(values, dtype=np.float64)
    ymin, ymax = float(array.min()), float(array.max())
    span = max(ymax - ymin, 1e-8)
    points = []
    for index, value in enumerate(array):
        x = padding + (width - 2 * padding) * index / max(len(array) - 1, 1)
        y = height - padding - (height - 2 * padding) * (float(value) - ymin) / span
        points.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" aria-label="similarity curve">'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#bc4b2d" '
        f'stroke-width="2.2" vector-effect="non-scaling-stroke"/></svg>'
    )


def build_page(
    result_root: Path,
    records: list[dict[str, Any]],
    *,
    page_name: str,
) -> Path:
    rows = []
    for record in sorted(records, key=lambda item: item["mean_similarity"]):
        label = html.escape(record["pair_label"])
        rows.append(
            f'''<tr data-mean="{record['mean_similarity']:.12f}" data-min="{record['min_similarity']:.12f}">
<td>{label}</td><td>{record['mean_similarity']:.7f}</td><td>{record['min_similarity']:.7f}</td>
<td>{record['min_frame']}</td><td>{record['min_time_seconds']:.4f}</td><td>{sparkline(record['frame_similarity'])}</td></tr>'''
        )
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V-JEPA temporal similarity</title><style>
:root{{--ink:#13231d;--paper:#f5f1e6;--rust:#bc4b2d;--line:#c7c0ae;--moss:#315f4b}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(145deg,#ece5d5,#faf8f1 55%,#e6eee2);color:var(--ink);font-family:Georgia,serif}}
header{{padding:44px clamp(20px,5vw,72px) 30px;border-bottom:1px solid var(--line)}} h1{{font-size:clamp(38px,6vw,76px);line-height:.95;margin:0 0 16px}} header p{{max-width:980px;font:15px/1.6 ui-monospace,monospace}}
nav{{position:sticky;top:0;background:#f5f1e6e8;backdrop-filter:blur(10px);padding:12px clamp(20px,5vw,72px);border-bottom:1px solid var(--line);z-index:2}}
button,a.button{{display:inline-block;border:1px solid var(--ink);background:transparent;color:var(--ink);padding:9px 13px;margin-right:7px;text-decoration:none;font:600 13px ui-monospace,monospace;cursor:pointer}} button:hover,a.button:hover{{background:var(--ink);color:white}}
main{{padding:28px clamp(14px,4vw,60px) 70px}} .panel{{background:#fffdf7;border:1px solid var(--line);padding:18px;margin-bottom:24px;box-shadow:0 10px 28px #24352a14}} .panel h2{{margin:0 0 12px;font-size:29px}} img{{display:block;width:100%;border:1px solid var(--line)}}
.table-wrap{{overflow:auto}} table{{width:100%;border-collapse:collapse;font:13px/1.4 ui-monospace,monospace}} th,td{{padding:9px 11px;border-bottom:1px solid #ded8c9;text-align:left;white-space:nowrap}} th{{color:var(--moss)}} td svg{{width:250px;height:58px;display:block}}
</style></head><body><header><h1>Temporal feature<br>similarity</h1>
<p>28 video pairs. X-axis is time at 60 FPS; Y-axis is mean cosine similarity across corresponding 24x42 V-JEPA patches. One 2-frame tubelet produces one value, so adjacent frame pairs share the same score.</p></header>
<nav><a class="button" href="index.html">Back to heatmaps</a><button onclick="sortRows(true)">Lowest mean first</button><button onclick="sortRows(false)">Highest mean first</button><button onclick="location.reload()">Manual refresh</button></nav>
<main><section class="panel"><h2>All 28 pairs</h2><img src="temporal_similarity/all_pairs.png"></section>
<section class="panel"><h2>Aggregate trend</h2><img src="temporal_similarity/aggregate.png"></section>
<section class="panel"><h2>Pair-by-pair curves</h2><img src="temporal_similarity/small_multiples.png"></section>
<section class="panel"><h2>Statistics</h2><div class="table-wrap"><table><thead><tr><th>Video pair</th><th>Mean</th><th>Minimum</th><th>Min frame</th><th>Min time (s)</th><th>Curve</th></tr></thead><tbody id="rows">{''.join(rows)}</tbody></table></div></section></main>
<script>function sortRows(lowFirst){{const body=document.getElementById('rows');const rows=[...body.children];rows.sort((a,b)=>(Number(a.dataset.mean)-Number(b.dataset.mean))*(lowFirst?1:-1));rows.forEach(row=>body.appendChild(row))}}</script>
</body></html>'''
    page_path = result_root / page_name
    page_path.write_text(page, encoding="utf-8")
    index_path = result_root / "index.html"
    if index_path.is_file():
        index = index_path.read_text(encoding="utf-8")
        if page_name not in index:
            button = f'<button onclick="location.href=\'{page_name}\'">Temporal similarity</button>'
            index = index.replace("</nav>", button + "</nav>", 1)
            index_path.write_text(index, encoding="utf-8")
    return page_path


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    manifest_path = result_root / "results.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = manifest.get("features", {})
    if len(features) < 2:
        raise RuntimeError("results.json does not contain at least two feature records")
    names = sorted(features)
    fps_values = [float(features[name]["fps"]) for name in names]
    fps = float(np.median(fps_values))
    num_frames = int(manifest["source_frames_used"])
    frame_indices = np.arange(num_frames, dtype=np.int64)
    time_seconds = frame_indices.astype(np.float64) / fps
    output_dir = result_root / "temporal_similarity"
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for pair_index, (name_a, name_b) in enumerate(itertools.combinations(names, 2), start=1):
        tubelet_mean, tubelet_std = token_cosine_curve(
            Path(features[name_a]["feature_path"]),
            Path(features[name_b]["feature_path"]),
        )
        frame_mean = expand_tubelets(tubelet_mean, num_frames)
        frame_std = expand_tubelets(tubelet_std, num_frames)
        min_frame = int(np.argmin(frame_mean))
        record = {
            "pair_index": pair_index,
            "video_a": name_a,
            "video_b": name_b,
            "pair_label": f"{Path(name_a).stem} vs {Path(name_b).stem}",
            "fps": fps,
            "frame_indices": frame_indices.tolist(),
            "time_seconds": time_seconds.tolist(),
            "tubelet_similarity": tubelet_mean.tolist(),
            "tubelet_patch_std": tubelet_std.tolist(),
            "frame_similarity": frame_mean.tolist(),
            "frame_patch_std": frame_std.tolist(),
            "mean_similarity": float(frame_mean.mean(dtype=np.float64)),
            "min_similarity": float(frame_mean[min_frame]),
            "max_similarity": float(frame_mean.max()),
            "min_frame": min_frame,
            "min_time_seconds": float(time_seconds[min_frame]),
        }
        records.append(record)
        print(
            f"[{pair_index:02d}/28] {record['pair_label']}: "
            f"mean={record['mean_similarity']:.7f} min={record['min_similarity']:.7f} "
            f"at frame={min_frame}",
            flush=True,
        )

    plot_all_pairs(records, fps=fps, output_path=output_dir / "all_pairs.png")
    plot_small_multiples(records, output_path=output_dir / "small_multiples.png")
    plot_aggregate(records, fps=fps, output_path=output_dir / "aggregate.png")
    payload = {
        "schema_version": 1,
        "metric": "mean cosine similarity of corresponding normalized V-JEPA patch tokens",
        "fps": fps,
        "num_frames": num_frames,
        "num_pairs": len(records),
        "records": records,
    }
    json_dump(output_dir / "temporal_similarity.json", payload)
    with (output_dir / "temporal_similarity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_a", "video_b", "frame", "time_seconds", "cosine_similarity"])
        for record in records:
            for frame, seconds, similarity in zip(
                record["frame_indices"],
                record["time_seconds"],
                record["frame_similarity"],
            ):
                writer.writerow(
                    [record["video_a"], record["video_b"], frame, seconds, similarity]
                )
    page = build_page(result_root, records, page_name=args.page_name)
    print(f"Page: {page}", flush=True)


if __name__ == "__main__":
    main()
