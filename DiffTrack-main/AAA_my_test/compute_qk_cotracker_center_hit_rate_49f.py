#!/usr/bin/env python3
"""Evaluate one SAM2-center query per region against CoTracker."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from center_query_utils import DEFAULT_CACHE, select_center_queries


DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physiciq_selected_three_model_qk_49f"
)
MODELS = ("gt", "stage1b", "lora", "baseline")
REGIONS = ("all_centers", "object_A", "object_B", "background")
THRESHOLDS = (16, 32, 64)
EXPECTED_CAPTURE = "video_self_attention_post_rmsnorm_post_3d_rope_pre_flash_attention"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--step-index", type=int, default=39)
    return parser.parse_args()


def evaluate(
    predictions: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    anchors: np.ndarray,
    query_latent: int,
    clean_prefix: int,
    point_indices: list[int],
) -> dict:
    predicted = predictions[:, point_indices]
    target = tracks[anchors][:, point_indices]
    visible = visibility[anchors][:, point_indices].copy()
    visible &= visibility[int(anchors[query_latent]), point_indices][None, :]
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
    result = {
        "comparisons": int(distances.size),
        "mean_error_px": float(distances.mean()),
        "median_error_px": float(np.median(distances)),
    }
    for threshold in THRESHOLDS:
        result[f"hit_rate_{threshold}px"] = float((distances <= threshold).mean())
    return result


def load_case(
    case_dir: Path,
    model: str,
    center_queries: dict[str, dict],
    layer: int,
    step_index: int,
) -> tuple[dict, list[dict]]:
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("capture_location") != EXPECTED_CAPTURE:
        raise RuntimeError(f"unexpected Q/K capture location: {case_dir}")
    if manifest.get("layers") != [layer] or manifest.get("step_indices") != [step_index]:
        raise RuntimeError(f"unexpected layer/step: {case_dir}")

    prediction_key = f"qk_layer{layer:02d}_step{step_index:03d}_predictions"
    predictions = np.load(case_dir / "predicted_tracks.npz")[prediction_key]
    cotracker = np.load(case_dir / "cotracker_pseudo_gt.npz")
    tracks = cotracker["tracks"]
    visibility = cotracker["visibility"].astype(bool)
    anchors = np.asarray(manifest["latent_anchor_pixel_frames"], dtype=np.int64)
    query_latent = int(manifest["query_latent_index"])
    clean_prefix = int(manifest["clean_prefix_latents"])
    if predictions.shape[0] != len(anchors) or int(anchors[-1]) >= len(tracks):
        raise RuntimeError(f"Q/K, anchor, and CoTracker geometry mismatch: {case_dir}")

    selected_indices = {
        name: int(center_queries[name]["global_index"])
        for name in ("object_A", "object_B", "background")
    }
    index_groups = {
        "all_centers": list(selected_indices.values()),
        **{name: [index] for name, index in selected_indices.items()},
    }
    rows = []
    for region, indices in index_groups.items():
        rows.append(
            {
                "model": model,
                "case_key": case_dir.name,
                "region": region,
                "selected_global_indices": indices,
                "pixel_frames": int(
                    manifest.get("generated_pixel_frames") or manifest["gt_pixel_frames"]
                ),
                "latent_frames": int(predictions.shape[0]),
                **evaluate(
                    predictions,
                    tracks,
                    visibility,
                    anchors,
                    query_latent,
                    clean_prefix,
                    indices,
                ),
            }
        )

    source_video = (
        manifest.get("gt_video") if model == "gt" else str(case_dir / "generated.mp4")
    )
    return {
        "model": model,
        "case_key": case_dir.name,
        "video_used_by_cotracker": source_video,
        "center_queries": center_queries,
        "layer": layer,
        "step_index": step_index,
    }, rows


def pooled(rows: list[dict], model: str, region: str) -> dict:
    selected = [
        row
        for row in rows
        if row["model"] == model and row["region"] == region and row["comparisons"] > 0
    ]
    total = sum(int(row["comparisons"]) for row in selected)
    result = {"model": model, "region": region, "comparisons": total}
    for key in ("mean_error_px", *[f"hit_rate_{value}px" for value in THRESHOLDS]):
        result[key] = (
            sum(float(row[key]) * int(row["comparisons"]) for row in selected) / total
            if total
            else None
        )
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "model",
        "case_key",
        "region",
        "selected_global_indices",
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


def percentage(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.1f}%"


def pixels(value: float | None) -> str:
    return "NA" if value is None else f"{value:.1f} px"


def write_report(path: Path, rows: list[dict], summary: list[dict]) -> None:
    per_video = [row for row in rows if row["region"] == "all_centers"]
    lines = [
        "# 中心点 Q/K track 对 CoTracker 的命中率",
        "",
        "每个视频只使用 object A、object B、background 各一个中心代表点。物体点是已有 query 中离 SAM2 mask 质心最近者；背景点是离画面中心最近的有效背景 query。仅统计 query 与目标均可见的 latent，主命中半径为 32 px。",
        "",
        "## 每个视频（三个中心点合并）",
        "",
        "| model | case | frames | comparisons | <=16 px | <=32 px | <=64 px | mean error |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_video:
        lines.append(
            f"| {row['model']} | {row['case_key']} | {row['pixel_frames']} | "
            f"{row['comparisons']} | {percentage(row['hit_rate_16px'])} | "
            f"{percentage(row['hit_rate_32px'])} | {percentage(row['hit_rate_64px'])} | "
            f"{pixels(row['mean_error_px'])} |"
        )
    lines.extend(
        [
            "",
            "## 按模型与区域汇总",
            "",
            "| model | region | comparisons | <=32 px | mean error |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['region']} | {row['comparisons']} | "
            f"{percentage(row['hit_rate_32px'])} | {pixels(row['mean_error_px'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.result_root.resolve()
    cache_root = args.cache_root.resolve()
    cases = []
    rows = []
    for model in MODELS:
        for case_dir in sorted((root / model / "cases").glob("case_*")):
            if not (case_dir / "complete.json").is_file():
                continue
            manifest = json.loads(
                (case_dir / "manifest.json").read_text(encoding="utf-8")
            )
            center_queries = select_center_queries(
                cache_root, case_dir.name, manifest["query_regions"]
            )
            case, case_rows = load_case(
                case_dir, model, center_queries, args.layer, args.step_index
            )
            cases.append(case)
            rows.extend(case_rows)

    summary = [
        pooled(rows, model, region) for model in MODELS for region in REGIONS
    ]
    payload = {
        "protocol": {
            "queries": "one existing center-representative query per SAM2 region",
            "qk": f"layer {args.layer}, denoising step {args.step_index}, post-RMSNorm/post-3D-RoPE/pre-FlashAttention",
            "reference": "CoTracker run independently on each corresponding generated or GT video",
            "visibility": "query and target must both be visible",
            "thresholds_px": list(THRESHOLDS),
            "primary_threshold_px": 32,
        },
        "cases": cases,
        "rows": rows,
        "summary": summary,
    }
    stem = root / "qk_cotracker_center_hit_rates"
    stem.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(stem.with_suffix(".csv"), rows)
    write_report(stem.with_suffix(".md"), rows, summary)
    print(f"Computed {len(rows)} center-only rows for {len(cases)} videos")


if __name__ == "__main__":
    main()
