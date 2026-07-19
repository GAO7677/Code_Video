#!/usr/bin/env python3
"""Compute Q/K-to-CoTracker hit rates for the 49-frame latent-track experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physiciq_selected_three_model_qk_49f"
)
MODELS = ("gt", "stage1b", "lora", "baseline")
THRESHOLDS = (16, 32, 64)
EXPECTED_CAPTURE = "video_self_attention_post_rmsnorm_post_3d_rope_pre_flash_attention"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def evaluate(
    predictions: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    anchors: np.ndarray,
    query_latent: int,
    clean_prefix: int,
    point_slice: slice,
) -> dict[str, float | int]:
    predicted = predictions[:, point_slice]
    target = tracks[anchors, point_slice]
    visible = visibility[anchors, point_slice].copy()
    visible &= visibility[int(anchors[query_latent]), point_slice][None, :]
    valid = visible & np.isfinite(predicted).all(axis=-1)
    valid[:clean_prefix] = False
    distances = np.linalg.norm(predicted - target, axis=-1)[valid]
    if not distances.size:
        return {
            "comparisons": 0,
            "mean_error_px": None,
            "median_error_px": None,
            **{f"hit_rate_{threshold}px": None for threshold in THRESHOLDS},
        }
    result: dict[str, float | int] = {
        "comparisons": int(distances.size),
        "mean_error_px": float(distances.mean()),
        "median_error_px": float(np.median(distances)),
    }
    for threshold in THRESHOLDS:
        result[f"hit_rate_{threshold}px"] = float((distances <= threshold).mean())
    return result


def load_case(case_dir: Path, model: str) -> tuple[dict, list[dict]]:
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("capture_location") != EXPECTED_CAPTURE:
        raise RuntimeError(f"unexpected Q/K capture location: {case_dir}")
    if manifest.get("layers") != [23] or manifest.get("step_indices") != [39]:
        raise RuntimeError(f"unexpected layer/step: {case_dir}")
    predictions = np.load(case_dir / "predicted_tracks.npz")[
        "qk_layer23_step039_predictions"
    ]
    cotracker = np.load(case_dir / "cotracker_pseudo_gt.npz")
    tracks = cotracker["tracks"]
    visibility = cotracker["visibility"].astype(bool)
    anchors = np.asarray(manifest["latent_anchor_pixel_frames"], dtype=np.int64)
    query_latent = int(manifest["query_latent_index"])
    clean_prefix = int(manifest["clean_prefix_latents"])
    if predictions.shape[0] != len(anchors) or int(anchors[-1]) >= len(tracks):
        raise RuntimeError(f"Q/K, anchor, and CoTracker geometry mismatch: {case_dir}")

    regions = [
        ("all", slice(0, predictions.shape[1])),
        *[
            (
                region["region_name"],
                slice(int(region["point_start"]), int(region["point_end"])),
            )
            for region in manifest["query_regions"]
        ],
    ]
    rows = []
    for region_name, point_slice in regions:
        rows.append(
            {
                "model": model,
                "case_key": case_dir.name,
                "region": region_name,
                "pixel_frames": int(
                    manifest.get("generated_pixel_frames") or manifest["gt_pixel_frames"]
                ),
                "latent_frames": int(predictions.shape[0]),
                "query_pixel_frame": int(anchors[query_latent]),
                "target_latents": list(range(clean_prefix, len(anchors))),
                **evaluate(
                    predictions,
                    tracks,
                    visibility,
                    anchors,
                    query_latent,
                    clean_prefix,
                    point_slice,
                ),
            }
        )
    source_video = manifest.get("gt_video") if model == "gt" else str(case_dir / "generated.mp4")
    case = {
        "model": model,
        "case_key": case_dir.name,
        "video_used_by_cotracker": source_video,
        "qk_feature": "post-RMSNorm post-3D-RoPE self-attention Q/K",
        "layer": 23,
        "step": 39,
    }
    return case, rows


def pooled(rows: list[dict], model: str, region: str) -> dict:
    selected = [
        row
        for row in rows
        if row["model"] == model and row["region"] == region and row["comparisons"] > 0
    ]
    total = sum(int(row["comparisons"]) for row in selected)
    result = {"model": model, "region": region, "comparisons": total}
    for key in ("mean_error_px", *[f"hit_rate_{value}px" for value in THRESHOLDS]):
        result[key] = sum(float(row[key]) * int(row["comparisons"]) for row in selected) / total
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "model",
        "case_key",
        "region",
        "pixel_frames",
        "latent_frames",
        "comparisons",
        "mean_error_px",
        "median_error_px",
        "hit_rate_16px",
        "hit_rate_32px",
        "hit_rate_64px",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict], summary: list[dict]) -> None:
    overall = [row for row in rows if row["region"] == "all"]
    lines = [
        "# Q/K track 命中 CoTracker 范围的概率",
        "",
        "定义：仅统计 query 在 pixel frame 4 可见且目标 latent anchor 可见的点。命中半径 32 px 等于一个 Wan spatial token；同时报告 16 px 和 64 px。",
        "",
        "## 每个视频",
        "",
        "| model | case | frames | comparisons | <=16 px | <=32 px | <=64 px | mean error |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['model']} | {row['case_key']} | {row['pixel_frames']} | "
            f"{row['comparisons']} | {100 * row['hit_rate_16px']:.1f}% | "
            f"{100 * row['hit_rate_32px']:.1f}% | {100 * row['hit_rate_64px']:.1f}% | "
            f"{row['mean_error_px']:.1f} px |"
        )
    lines.extend(
        [
            "",
            "## 按模型汇总",
            "",
            "| model | region | comparisons | <=32 px | mean error |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['region']} | {row['comparisons']} | "
            f"{100 * row['hit_rate_32px']:.1f}% | {row['mean_error_px']:.1f} px |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.result_root.resolve()
    cases = []
    rows = []
    for model in MODELS:
        case_dirs = sorted((root / model / "cases").glob("case_*"))
        for case_dir in case_dirs:
            if not (case_dir / "complete.json").is_file():
                continue
            case, case_rows = load_case(case_dir, model)
            cases.append(case)
            rows.extend(case_rows)
    summary = [
        pooled(rows, model, region)
        for model in MODELS
        for region in ("all", "object_A", "object_B", "background")
    ]
    payload = {
        "protocol": {
            "qk": "layer 23, denoising step 39, post-RMSNorm/post-3D-RoPE/pre-FlashAttention",
            "reference": "CoTracker run independently on each corresponding generated or GT video",
            "visibility": "query and target must both be visible",
            "thresholds_px": list(THRESHOLDS),
            "primary_threshold_px": 32,
        },
        "cases": cases,
        "rows": rows,
        "summary": summary,
    }
    (root / "qk_cotracker_hit_rates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(root / "qk_cotracker_hit_rates.csv", rows)
    write_report(root / "qk_cotracker_hit_rates.md", rows, summary)
    print(f"Computed {len(rows)} rows for {len(cases)} videos")


if __name__ == "__main__":
    main()
