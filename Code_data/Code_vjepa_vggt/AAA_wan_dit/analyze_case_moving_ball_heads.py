#!/usr/bin/env python3
"""Analyze every Block-17 head for a tracked moving-ball case."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from analyze_self_attention_head_roles import _metrics, _role_name, _tags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _generated_video(root: Path, model: str, case: str) -> Path:
    matches = sorted((root.parent / "generated" / model).glob(f"**/{case}.mp4"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one generated video for {model}/{case}, found {matches}"
        )
    return matches[0]


def _circle_candidates(frame: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=80,
        param2=23,
        minRadius=12,
        maxRadius=45,
    )
    if circles is None:
        return []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yy, xx = np.ogrid[: frame.shape[0], : frame.shape[1]]
    candidates: list[np.ndarray] = []
    for cx, cy, radius in circles[0]:
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.8 * radius) ** 2
        pixels = hsv[disk]
        orange_fraction = np.mean(
            (pixels[:, 0] <= 25)
            & (pixels[:, 1] >= 55)
            & (pixels[:, 2] >= 80)
        )
        if orange_fraction > 0.75 and float(pixels[:, 1].mean()) > 105:
            candidates.append(np.asarray([cx, cy, radius], dtype=np.float64))
    return candidates


def _track_latent_ball(
    video_path: Path, temporal_tokens: int
) -> tuple[np.ndarray, tuple[int, int]]:
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"cannot read {video_path}")

    # Wan uses 4x temporal VAE compression for 4k+1 frame videos.
    indices = [
        min(4 * latent_t, len(frames) - 1)
        for latent_t in range(temporal_tokens)
    ]
    candidates = [_circle_candidates(frames[index]) for index in indices]
    anchor = np.asarray(
        [0.525 * frames[0].shape[1], 0.473 * frames[0].shape[0]],
        dtype=np.float64,
    )
    has_split_motion = any(
        any(np.linalg.norm(circle[:2] - anchor) > 100 for circle in options)
        for options in candidates[3:]
    )

    trajectory: list[np.ndarray | None] = []
    previous: np.ndarray | None = None
    for latent_t, options in enumerate(candidates):
        selected: np.ndarray | None = None
        if options:
            if has_split_motion and latent_t >= 3:
                moving = [
                    circle
                    for circle in options
                    if np.linalg.norm(circle[:2] - anchor) > 80
                ]
                if moving:
                    selected = max(
                        moving, key=lambda circle: np.linalg.norm(circle[:2] - anchor)
                    )
            if selected is None and (not has_split_motion or latent_t < 3):
                if previous is None:
                    selected = max(options, key=lambda circle: circle[2])
                else:
                    selected = min(
                        options,
                        key=lambda circle: np.linalg.norm(
                            circle[:2] - previous[:2]
                        ),
                    )
        trajectory.append(selected)
        if selected is not None:
            previous = selected

    known = [index for index, value in enumerate(trajectory) if value is not None]
    if not known:
        raise RuntimeError(f"ball detector found no trajectory in {video_path}")
    for index, value in enumerate(trajectory):
        if value is not None:
            continue
        lower = max((item for item in known if item < index), default=known[0])
        upper = min((item for item in known if item > index), default=known[-1])
        weight = 0.0 if upper == lower else (index - lower) / (upper - lower)
        trajectory[index] = (
            trajectory[lower] * (1.0 - weight) + trajectory[upper] * weight
        )
    return (
        np.stack([value for value in trajectory if value is not None]),
        frames[0].shape[:2],
    )


def _circle_tokens(
    *,
    latent_t: int,
    circle: np.ndarray,
    frame_shape: tuple[int, int],
    grid: tuple[int, int, int],
) -> np.ndarray:
    _, grid_h, grid_w = grid
    frame_h, frame_w = frame_shape
    cx, cy, radius = circle
    cell_margin = 0.30 * math.hypot(frame_w / grid_w, frame_h / grid_h)
    token_ids = []
    for y in range(grid_h):
        for x in range(grid_w):
            px = (x + 0.5) * frame_w / grid_w
            py = (y + 0.5) * frame_h / grid_h
            if (px - cx) ** 2 + (py - cy) ** 2 <= (radius + cell_margin) ** 2:
                token_ids.append(latent_t * grid_h * grid_w + y * grid_w + x)
    if not token_ids:
        raise RuntimeError(f"ball at latent frame {latent_t} maps to no tokens")
    return np.asarray(token_ids, dtype=np.int64)


def _ball_attention_metrics(
    attention: np.ndarray,
    *,
    trajectory_tokens: list[np.ndarray],
    token_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    heads, bins, _ = attention.shape
    token_bins = np.arange(token_count, dtype=np.int64) * bins // token_count
    key_counts = np.bincount(token_bins, minlength=bins)
    full_trajectory = np.unique(np.concatenate(trajectory_tokens))
    all_trajectory_scores = np.zeros(heads, dtype=np.float64)
    cross_time_scores = np.zeros(heads, dtype=np.float64)
    for head in range(heads):
        all_samples = []
        cross_samples = []
        for latent_t, query_ids in enumerate(trajectory_tokens):
            query_bins = token_bins[query_ids]
            key_bin_mass = attention[head, query_bins].mean(axis=0)
            per_token_mass = (key_bin_mass / key_counts)[token_bins]

            all_expected = len(full_trajectory) / token_count
            all_samples.append(per_token_mass[full_trajectory].sum() / all_expected)
            cross_ids = np.concatenate(
                [
                    ids
                    for other_t, ids in enumerate(trajectory_tokens)
                    if other_t != latent_t
                ]
            )
            cross_expected = len(cross_ids) / token_count
            cross_samples.append(per_token_mass[cross_ids].sum() / cross_expected)
        all_trajectory_scores[head] = float(np.mean(all_samples))
        cross_time_scores[head] = float(np.mean(cross_samples))
    return all_trajectory_scores, cross_time_scores


def _short_interpretation(role: str, row: dict[str, Any]) -> str:
    if row["cross_ball_step35"] >= 6.5:
        ball = "强跨帧球轨迹"
    elif row["cross_ball_step35"] >= 4.5:
        ball = "中等跨帧球轨迹"
    else:
        ball = "球轨迹响应较弱"
    if row["history_bias"] >= 0.25:
        direction = "；强历史偏置"
    elif row["history_bias"] <= -0.10:
        direction = "；未来帧偏置"
    else:
        direction = ""
    if "上下文" in role:
        return f"{ball}；同时偏首帧/历史上下文{direction}"
    if "全局" in role:
        return f"{ball}；高熵全局聚合，不宜解释为精确跟踪{direction}"
    if "帧内" in role:
        return f"{ball}；主要维持当前帧局部结构{direction}"
    if "时间" in role or "时空" in role:
        return f"{ball}；承担跨帧位置传播{direction}"
    return f"{ball}；相对位置或混合路由{direction}"


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_paths = sorted(root.glob(f"*/{args.case}/summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"no summaries for {args.case} under {root}")

    structural: dict[int, list[tuple[int, np.ndarray]]] = {}
    ball_scores: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    trajectories: dict[str, Any] = {}
    for summary_path in summary_paths:
        summary = _read_json(summary_path)
        model = str(summary["model"])
        grid = tuple(int(value) for value in summary["latent_grid"])
        temporal_tokens, _, _ = grid
        token_count = math.prod(grid)
        trajectory, frame_shape = _track_latent_ball(
            _generated_video(root, model, args.case), temporal_tokens
        )
        trajectory_tokens = [
            _circle_tokens(
                latent_t=latent_t,
                circle=circle,
                frame_shape=frame_shape,
                grid=grid,
            )
            for latent_t, circle in enumerate(trajectory)
        ]
        trajectories[model] = {
            "latent_circles_cx_cy_r": trajectory.round(2).tolist(),
            "tokens_per_latent_frame": [len(ids) for ids in trajectory_tokens],
        }
        for entry in summary["steps"]:
            step = int(entry["step_number_one_based"])
            matrix_path = (
                summary_path.parent
                / str(entry["directory"])
                / str(entry["matrix_npz"])
            )
            with np.load(matrix_path) as arrays:
                attention = arrays["key_mass"]
            per_head = _metrics(
                attention,
                token_count=token_count,
                temporal_tokens=temporal_tokens,
            )
            all_ball, cross_ball = _ball_attention_metrics(
                attention,
                trajectory_tokens=trajectory_tokens,
                token_count=token_count,
            )
            for head in range(attention.shape[0]):
                structural.setdefault(head, []).append((step, per_head[head]))
                ball_scores.setdefault(head, []).append(
                    (step, all_ball[head], cross_ball[head])
                )

    records: list[dict[str, Any]] = []
    for head in sorted(structural):
        step5 = np.stack([row for step, row in structural[head] if step == 5]).mean(0)
        step35 = np.stack(
            [row for step, row in structural[head] if step == 35]
        ).mean(0)
        overall = np.stack([row for _, row in structural[head]]).mean(0)
        tags = _tags(overall)
        role = _role_name(tags)
        ball5 = np.mean(
            [cross for step, _, cross in ball_scores[head] if step == 5]
        )
        ball35 = np.mean(
            [cross for step, _, cross in ball_scores[head] if step == 35]
        )
        all_ball = np.mean([all_value for _, all_value, _ in ball_scores[head]])
        record = {
            "head": head,
            "role": role,
            "tags": tags,
            "same_frame_step5": float(step5[1]),
            "same_frame_step35": float(step35[1]),
            "aligned_cross_time_step5": float(step5[4]),
            "aligned_cross_time_step35": float(step35[4]),
            "first_frame_mass": float(overall[2]),
            "history_bias": float(overall[8] - overall[9]),
            "entropy": float(overall[0]),
            "all_ball_trajectory_enrichment": float(all_ball),
            "cross_ball_step5": float(ball5),
            "cross_ball_step35": float(ball35),
        }
        record["interpretation"] = _short_interpretation(role, record)
        records.append(record)

    by_ball = sorted(
        records, key=lambda row: row["cross_ball_step35"], reverse=True
    )
    for rank, record in enumerate(by_ball, start=1):
        record["cross_ball_step35_rank"] = rank

    csv_path = output_dir / "all_head_moving_ball_analysis.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    payload = {
        "case": args.case,
        "matrix_root": str(root),
        "limitation": (
            "Attention uses the saved 512-bin pooled matrix. Each bin averages "
            "about 11-12 original tokens, so moving-ball query scores are "
            "approximate. Cross-ball metrics exclude the current latent frame."
        ),
        "trajectories": trajectories,
        "heads": records,
    }
    json_path = output_dir / "all_head_moving_ball_analysis.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    table_rows = []
    for row in records:
        table_rows.append(
            f"| {row['head']:02d} | {row['role']} | "
            f"{row['same_frame_step5']:.1%}→{row['same_frame_step35']:.1%} | "
            f"{row['aligned_cross_time_step5']:.1%}→"
            f"{row['aligned_cross_time_step35']:.1%} | "
            f"{row['first_frame_mass']:.1%} | {row['history_bias']:+.1%} | "
            f"{row['cross_ball_step5']:.2f}×→{row['cross_ball_step35']:.2f}× "
            f"(#{row['cross_ball_step35_rank']}) | {row['interpretation']} |"
        )
    markdown = f"""# Block 17 Moving-Ball Head Analysis

Case: `{args.case}`. Values are averaged across Wan+LoRA, xSSC, and PhysRVG.
Step arrows compare denoise step 5 with step 35. Ball-trajectory enrichment excludes
the current latent frame, so it does not reward a head merely for attending locally
to the query ball itself.

| Head | Role | Same-frame | Aligned cross-time | First-frame | History bias | Cross-time ball trajectory | Interpretation |
|---:|---|---:|---:|---:|---:|---:|---|
{chr(10).join(table_rows)}

## Reading Notes

- `Same-frame`: attention mass retained within the query latent frame.
- `Aligned cross-time`: attention sent to approximately the same spatial position
  in other latent frames.
- `History bias`: past-frame mass minus future-frame mass.
- `Cross-time ball trajectory`: attention on tracked ball tokens in all other
  latent frames divided by the uniform-attention expectation.
- The matrices contain 512 pooled bins for 5824 tokens. The ball-query results are
  suitable for comparing heads, but exact token-level claims require targeted Q/K
  recapture during inference.
"""
    markdown_path = output_dir / "all_head_moving_ball_analysis.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "markdown": str(markdown_path),
                "csv": str(csv_path),
                "json": str(json_path),
                "heads": len(records),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
