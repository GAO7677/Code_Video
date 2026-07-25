#!/usr/bin/env python3
"""Find query-independent key-token columns and overlay them on video."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-heads-per-step", type=int, default=2)
    parser.add_argument("--top-bins-per-head", type=int, default=8)
    parser.add_argument("--coverage-threshold", type=float, default=2.0)
    parser.add_argument("--overlay-alpha", type=float, default=0.62)
    parser.add_argument("--max-frames", type=int, default=49)
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path("/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg"),
    )
    return parser.parse_args()


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    normalized = weights / weights.sum()
    return np.sum(values * normalized[None, :, None], axis=1)


def _column_metrics(
    key_mass: np.ndarray,
    query_counts: np.ndarray,
    key_counts: np.ndarray,
    token_count: int,
    coverage_threshold: float,
) -> dict[str, np.ndarray]:
    uniform_mass = key_counts.astype(np.float64) / float(token_count)
    enrichment = key_mass.astype(np.float64) / uniform_mass[None, None, :]
    geometric = np.exp(
        _weighted_mean(np.log(np.maximum(enrichment, 1.0e-30)), query_counts)
    )
    arithmetic = _weighted_mean(enrichment, query_counts)
    coverage = _weighted_mean(
        (enrichment >= coverage_threshold).astype(np.float64), query_counts
    )
    score = geometric * np.sqrt(np.maximum(coverage, 1.0e-12))
    return {
        "geometric_enrichment": geometric,
        "mean_enrichment": arithmetic,
        "coverage": coverage,
        "score": score,
    }


def _select_bins(scores: np.ndarray, count: int) -> list[int]:
    selected: list[int] = []
    for candidate in np.argsort(scores)[::-1]:
        value = int(candidate)
        if any(abs(value - previous) <= 1 for previous in selected):
            continue
        selected.append(value)
        if len(selected) == count:
            break
    return selected


def _bin_token_ids(bin_id: int, bins: int, token_count: int) -> np.ndarray:
    token_ids = np.arange(token_count, dtype=np.int64)
    token_bins = token_ids * bins // token_count
    return token_ids[token_bins == int(bin_id)]


def _token_regions(
    token_ids: np.ndarray, grid: tuple[int, int, int]
) -> list[dict[str, int | list[int]]]:
    _, grid_h, grid_w = grid
    per_frame = grid_h * grid_w
    regions: list[dict[str, int | list[int]]] = []
    for token_id in token_ids.tolist():
        latent_t, spatial = divmod(token_id, per_frame)
        y, x = divmod(spatial, grid_w)
        if (
            regions
            and regions[-1]["latent_t"] == latent_t
            and regions[-1]["y"] == y
            and regions[-1]["x"][1] + 1 == x
        ):
            regions[-1]["x"][1] = x
        else:
            frame_range = [0, 0] if latent_t == 0 else [4 * latent_t - 3, 4 * latent_t]
            regions.append(
                {
                    "latent_t": latent_t,
                    "output_frames_approx": frame_range,
                    "y": y,
                    "x": [x, x],
                }
            )
    return regions


def _expand_bins_to_grid(
    values: np.ndarray,
    *,
    bins: int,
    grid: tuple[int, int, int],
) -> np.ndarray:
    token_count = math.prod(grid)
    token_ids = np.arange(token_count, dtype=np.int64)
    token_bins = token_ids * bins // token_count
    return values[token_bins].reshape(grid)


def _find_generated_video(root: Path, model: str, case: str) -> Path:
    generated_root = root.parent / "generated" / model
    matches = sorted(generated_root.glob(f"**/{case}.mp4"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one generated video for {model}/{case}, found {matches}"
        )
    return matches[0]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_source_video(generated_video: Path) -> Path:
    sidecar = generated_video.with_suffix(".json")
    payload = _read_json(sidecar)
    direct = payload.get("source_video") or payload.get("input_video_original")
    if direct and Path(direct).is_file():
        return Path(direct)
    input_json = payload.get("input_json")
    if input_json and Path(input_json).is_file():
        source = _read_json(Path(input_json)).get("source_video")
        if source and Path(source).is_file():
            return Path(source)
    raise FileNotFoundError(f"cannot resolve source video from {sidecar}")


def _read_video(path: Path, max_frames: int) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video contains no readable frames: {path}")
    return frames, fps


def _temporal_slice(grid: np.ndarray, frame_index: int, frame_count: int) -> np.ndarray:
    if frame_count <= 1:
        return grid[0]
    position = frame_index * (grid.shape[0] - 1) / float(frame_count - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, grid.shape[0] - 1)
    weight = position - lower
    return grid[lower] * (1.0 - weight) + grid[upper] * weight


def _write_overlay(
    *,
    video_path: Path,
    output_path: Path,
    heat_grid: np.ndarray,
    selected_grid: np.ndarray,
    title: str,
    alpha: float,
    max_frames: int,
    ffmpeg: Path,
) -> None:
    frames, fps = _read_video(video_path, max_frames)
    height, width = frames[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        positive = heat_grid[heat_grid > 0]
        high = float(np.percentile(positive, 99.5)) if positive.size else 1.0
        high = max(high, 1.0e-6)
        frame_count = len(frames)
        for index, frame in enumerate(frames):
            heat = _temporal_slice(heat_grid, index, frame_count)
            heat = np.clip(heat / high, 0.0, 1.0)
            heat = cv2.resize(
                heat.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC
            )
            heat = np.clip(heat, 0.0, 1.0)
            color = cv2.applyColorMap(
                np.round(heat * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO
            )
            local_alpha = (alpha * np.sqrt(heat))[..., None]
            overlay = np.clip(
                frame.astype(np.float32) * (1.0 - local_alpha)
                + color.astype(np.float32) * local_alpha,
                0,
                255,
            ).astype(np.uint8)

            selected = _temporal_slice(selected_grid, index, frame_count)
            selected = cv2.resize(
                selected.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            contours, _ = cv2.findContours(
                selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
            label = f"{title} | frame {index:02d}"
            cv2.putText(
                overlay,
                label,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                label,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            process.stdin.write(overlay.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while writing {output_path}")


def _write_gallery(output: Path, rows: list[dict[str, Any]], case: str) -> None:
    cards = []
    for row in rows:
        bins = ", ".join(str(value) for value in row["selected_bins"])
        cards.append(
            "<article>"
            f"<h2>{html.escape(row['model'])} · step {row['step']:02d} · "
            f"head {row['head']:02d}</h2>"
            f"<p>vertical score {row['vertical_score']:.2f}; selected bins {bins}</p>"
            "<div class='grid'>"
            f"<figure><img src='{html.escape(row['matrix_image'])}'><figcaption>"
            "512×512 pooled attention matrix</figcaption></figure>"
            f"<figure><video controls loop muted src='{html.escape(row['generated_overlay'])}'>"
            "</video><figcaption>generated video overlay</figcaption></figure>"
            f"<figure><video controls loop muted src='{html.escape(row['source_overlay'])}'>"
            "</video><figcaption>source/GT video overlay</figcaption></figure>"
            "</div></article>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vertical key-token overlays · {html.escape(case)}</title>
<style>
body{{margin:0;background:#101214;color:#f2f4f5;font:15px/1.45 system-ui,sans-serif}}
header,article{{padding:18px 24px;border-bottom:1px solid #34383d}}
h1,h2{{margin:0 0 8px}}p,figcaption{{color:#b9c0c7}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}
figure{{margin:0}}img,video{{width:100%;background:#000;display:block}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>{html.escape(case)}</h1>
<p>Block 17 key bins whose attention remains high across query bins. Cyan outlines
mark the selected pooled-token regions; color shows log2 enrichment over uniform
attention. The source overlay is a spatial/temporal reference, not a claim of
pixel-exact semantic correspondence.</p></header>
{''.join(cards)}
</body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.ffmpeg.is_file():
        raise FileNotFoundError(args.ffmpeg)

    ranking: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob(f"*/{args.case}/summary.json")):
        summary = _read_json(summary_path)
        model = str(summary["model"])
        grid = tuple(int(value) for value in summary["latent_grid"])
        token_count = math.prod(grid)
        generated_video = _find_generated_video(root, model, args.case)
        source_video = _find_source_video(generated_video)
        for step_entry in summary["steps"]:
            step = int(step_entry["step_number_one_based"])
            matrix_path = (
                summary_path.parent
                / str(step_entry["directory"])
                / str(step_entry["matrix_npz"])
            )
            with np.load(matrix_path) as arrays:
                key_mass = arrays["key_mass"]
            metadata = step_entry["matrix_metadata"]
            bins = int(metadata["output_bins"])
            query_counts = np.asarray(metadata["query_bin_counts"], dtype=np.float64)
            key_counts = np.asarray(metadata["key_bin_counts"], dtype=np.float64)
            metrics = _column_metrics(
                key_mass,
                query_counts,
                key_counts,
                token_count,
                args.coverage_threshold,
            )
            head_scores = np.mean(
                np.sort(metrics["score"], axis=1)[:, -args.top_bins_per_head :],
                axis=1,
            )
            heads = np.argsort(head_scores)[::-1][: args.top_heads_per_step]
            ranking.append(
                {
                    "model": model,
                    "step": step,
                    "head_vertical_scores": head_scores.tolist(),
                }
            )
            for head_value in heads:
                head = int(head_value)
                selected_bins = _select_bins(
                    metrics["score"][head], args.top_bins_per_head
                )
                selected_mask = np.zeros(bins, dtype=np.uint8)
                selected_mask[selected_bins] = 1
                heat_values = np.maximum(
                    np.log2(np.maximum(metrics["geometric_enrichment"][head], 1.0)),
                    0.0,
                )
                heat_grid = _expand_bins_to_grid(
                    heat_values, bins=bins, grid=grid
                )
                selected_grid = _expand_bins_to_grid(
                    selected_mask, bins=bins, grid=grid
                )
                stem = f"{model}_step{step:02d}_head{head:02d}"
                model_dir = output / model
                generated_overlay = model_dir / f"{stem}_generated_overlay.mp4"
                source_overlay = model_dir / f"{stem}_source_overlay.mp4"
                title = f"{model} B17 step {step:02d} head {head:02d}"
                _write_overlay(
                    video_path=generated_video,
                    output_path=generated_overlay,
                    heat_grid=heat_grid,
                    selected_grid=selected_grid,
                    title=title,
                    alpha=args.overlay_alpha,
                    max_frames=args.max_frames,
                    ffmpeg=args.ffmpeg,
                )
                _write_overlay(
                    video_path=source_video,
                    output_path=source_overlay,
                    heat_grid=heat_grid,
                    selected_grid=selected_grid,
                    title=title + " source",
                    alpha=args.overlay_alpha,
                    max_frames=args.max_frames,
                    ffmpeg=args.ffmpeg,
                )
                matrix_source = (
                    matrix_path.parent / f"head_{head:02d}_token_attention_matrix.png"
                )
                matrix_target = model_dir / f"{stem}_matrix.png"
                shutil.copy2(matrix_source, matrix_target)
                selected_rows.append(
                    {
                        "model": model,
                        "step": step,
                        "head": head,
                        "vertical_score": float(head_scores[head]),
                        "selected_bins": selected_bins,
                        "matrix_image": str(matrix_target.relative_to(output)),
                        "generated_overlay": str(generated_overlay.relative_to(output)),
                        "source_overlay": str(source_overlay.relative_to(output)),
                    }
                )
                for bin_id in selected_bins:
                    token_ids = _bin_token_ids(bin_id, bins, token_count)
                    row = {
                        "model": model,
                        "step": step,
                        "head": head,
                        "bin": bin_id,
                        "token_start": int(token_ids[0]),
                        "token_stop": int(token_ids[-1]),
                        "token_count": int(token_ids.size),
                        "geometric_enrichment": float(
                            metrics["geometric_enrichment"][head, bin_id]
                        ),
                        "mean_enrichment": float(
                            metrics["mean_enrichment"][head, bin_id]
                        ),
                        "coverage_above_threshold": float(
                            metrics["coverage"][head, bin_id]
                        ),
                        "regions": _token_regions(token_ids, grid),
                    }
                    csv_rows.append(row)

    if not selected_rows:
        raise FileNotFoundError(f"no summaries for case {args.case} under {root}")
    report = {
        "case": args.case,
        "matrix_root": str(root),
        "selection": {
            "meaning": (
                "Per key bin, geometric-mean attention enrichment across all query "
                "bins, weighted by query-bin token count."
            ),
            "coverage_threshold_over_uniform": args.coverage_threshold,
            "top_heads_per_step": args.top_heads_per_step,
            "top_bins_per_head": args.top_bins_per_head,
            "limitation": (
                "The stored matrix has 512 pooled bins for 5824 tokens. Each "
                "selected bin therefore identifies a contiguous token range, not "
                "one exact original token."
            ),
        },
        "rankings": ranking,
        "selected": csv_rows,
    }
    (output / "vertical_key_token_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "vertical_key_tokens.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "model",
            "step",
            "head",
            "bin",
            "token_start",
            "token_stop",
            "token_count",
            "geometric_enrichment",
            "mean_enrichment",
            "coverage_above_threshold",
            "regions",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            serializable = dict(row)
            serializable["regions"] = json.dumps(
                serializable["regions"], ensure_ascii=False, separators=(",", ":")
            )
            writer.writerow(serializable)
    _write_gallery(output, selected_rows, args.case)
    print(
        json.dumps(
            {
                "gallery": str(output / "index.html"),
                "report": str(output / "vertical_key_token_report.json"),
                "selected_head_step_pairs": len(selected_rows),
                "selected_token_bins": len(csv_rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
