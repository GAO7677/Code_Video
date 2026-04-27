#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchlib.manifest import BenchSample, load_manifest
from benchlib.staging import safe_stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter PhysV benchmark samples whose empty-scene time exceeds a threshold.")
    parser.add_argument("--prepared-root", required=True, help="Prepared benchmark root with dataset manifests and future_videos.")
    parser.add_argument("--output-root", required=True, help="Benchmark output root containing short_ctx8 and i2v_ctx8 results.")
    parser.add_argument("--archive-root", required=True, help="Archive directory for removed benchmark assets.")
    parser.add_argument("--visible-ratio-threshold", type=float, default=0.02, help="Frame is treated as empty if visible foreground ratio is below this value.")
    parser.add_argument("--pixel-diff-threshold", type=int, default=18, help="Per-pixel RGB max-diff threshold used for foreground extraction.")
    parser.add_argument("--max-empty-fraction", type=float, default=1.0 / 3.0, help="Drop sample when empty-frame fraction is strictly above this value.")
    parser.add_argument("--median-blur-ksize", type=int, default=5, help="Median blur kernel size for foreground mask cleanup.")
    return parser.parse_args()


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_manifest_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def backup_file(path: Path, tag: str) -> None:
    backup_path = path.with_name(f"{path.name}.{tag}.bak")
    if backup_path.exists():
        return
    shutil.copy2(path, backup_path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def video_background(video_path: Path) -> np.ndarray:
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames were collected from {video_path}")
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def foreground_ratios(
    video_path: Path,
    pixel_diff_threshold: int,
    median_blur_ksize: int,
) -> list[float]:
    background = video_background(video_path)
    ratios: list[float] = []
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        diff = np.abs(rgb.astype(np.int16) - background.astype(np.int16)).max(axis=2)
        mask = (diff > pixel_diff_threshold).astype(np.uint8) * 255
        if median_blur_ksize > 1:
            mask = cv2.medianBlur(mask, median_blur_ksize)
        ratios.append(float((mask > 0).mean()))
    cap.release()
    return ratios


def move_path(src: Path, dst: Path) -> None:
    if not src.exists() and not src.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(src), str(dst))


def remove_sample_assets(
    sample: BenchSample,
    sample_stem: str,
    dataset_root: Path,
    output_root: Path,
    archive_root: Path,
) -> None:
    future_video = Path(sample.video_path)
    context_dir = Path(sample.context_frames_dir) if sample.context_frames_dir else None
    future_rel = future_video.relative_to(dataset_root)
    move_path(future_video, archive_root / "prepared" / future_rel)
    if context_dir and context_dir.exists():
        context_rel = context_dir.relative_to(dataset_root)
        move_path(context_dir, archive_root / "prepared" / context_rel)

    for suite in ("short_ctx8", "i2v_ctx8"):
        staging_dir = output_root / suite / "staging"
        for staged_file in (staging_dir / "videos").glob(f"{sample_stem}.*"):
            move_path(staged_file, archive_root / suite / "staging" / "videos" / staged_file.name)
        images_dir = staging_dir / "images"
        if images_dir.exists():
            for staged_file in images_dir.glob(f"{sample_stem}.*"):
                move_path(staged_file, archive_root / suite / "staging" / "images" / staged_file.name)


def filter_eval_results(path: Path, removed_video_names: set[str]) -> None:
    data = read_json(path)
    changed = False
    for dimension, payload in data.items():
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            continue
        kept_items = [item for item in payload[1] if Path(item.get("video_path", "")).name not in removed_video_names]
        if len(kept_items) == len(payload[1]):
            continue
        changed = True
        values = [float(item["video_results"]) for item in kept_items if isinstance(item.get("video_results"), (int, float))]
        data[dimension] = [float(sum(values) / len(values)) if values else None, kept_items]
    if changed:
        write_json(path, data)


def filter_full_info(path: Path, removed_video_names: set[str]) -> None:
    data = read_json(path)
    if not isinstance(data, list):
        return
    kept = []
    for item in data:
        video_list = item.get("video_list", [])
        if any(Path(video_path).name in removed_video_names for video_path in video_list):
            continue
        kept.append(item)
    write_json(path, kept)


def filter_run_metadata(path: Path, removed_sample_ids: set[str]) -> None:
    data = read_json(path)
    if not isinstance(data, dict):
        return
    samples = data.get("samples")
    if isinstance(samples, list):
        data["samples"] = [sample_id for sample_id in samples if sample_id not in removed_sample_ids]
    write_json(path, data)


def main() -> None:
    args = parse_args()
    prepared_root = Path(args.prepared_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    archive_root = Path(args.archive_root).expanduser().resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    tag = f"before_empty_filter_{datetime.utcnow().strftime('%Y%m%d')}"

    summary_path = prepared_root / "summary.json"
    backup_file(summary_path, tag)
    summary_payload = read_json(summary_path)

    audit: dict[str, Any] = {
        "visible_ratio_threshold": args.visible_ratio_threshold,
        "pixel_diff_threshold": args.pixel_diff_threshold,
        "max_empty_fraction": args.max_empty_fraction,
        "median_blur_ksize": args.median_blur_ksize,
        "datasets": {},
    }

    for manifest_path in sorted(prepared_root.glob("*/manifest.jsonl")):
        dataset_name = manifest_path.parent.name
        dataset_root = manifest_path.parent
        if not (dataset_root / "future_videos").exists():
            continue

        backup_file(manifest_path, tag)
        samples = load_manifest(str(manifest_path))
        raw_rows = load_manifest_rows(manifest_path)
        raw_by_sample_id = {str(row.get("sample_id") or row.get("id")): row for row in raw_rows}

        kept_rows: list[dict[str, Any]] = []
        removed_sample_ids: set[str] = set()
        removed_video_names: set[str] = set()
        removed_details: list[dict[str, Any]] = []
        count_before = len(samples)

        for sample in samples:
            ratios = foreground_ratios(
                video_path=Path(sample.video_path),
                pixel_diff_threshold=args.pixel_diff_threshold,
                median_blur_ksize=args.median_blur_ksize,
            )
            empty_frames = sum(ratio < args.visible_ratio_threshold for ratio in ratios)
            total_frames = len(ratios)
            empty_fraction = (empty_frames / total_frames) if total_frames else 0.0
            drop = empty_fraction > args.max_empty_fraction

            if drop:
                removed_sample_ids.add(sample.sample_id)
                sample_stem = safe_stem(sample)
                removed_video_names.add(f"{sample_stem}{Path(sample.video_path).suffix.lower()}")
                removed_details.append(
                    {
                        "sample_id": sample.sample_id,
                        "video_name": Path(sample.video_path).name,
                        "frames": total_frames,
                        "empty_frames": empty_frames,
                        "empty_fraction": round(empty_fraction, 4),
                        "min_visible_ratio": round(min(ratios), 4) if ratios else None,
                        "max_visible_ratio": round(max(ratios), 4) if ratios else None,
                        "mean_visible_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
                    }
                )
                remove_sample_assets(
                    sample=sample,
                    sample_stem=sample_stem,
                    dataset_root=dataset_root,
                    output_root=output_root / dataset_name,
                    archive_root=archive_root / dataset_name / sample.sample_id,
                )
                continue

            row = raw_by_sample_id.get(sample.sample_id)
            if row is None:
                raise KeyError(f"Sample {sample.sample_id} was not found in {manifest_path}")
            kept_rows.append(row)

        write_manifest_rows(manifest_path, kept_rows)

        short_dir = output_root / dataset_name / "short_ctx8"
        i2v_dir = output_root / dataset_name / "i2v_ctx8"
        for path in (
            short_dir / "short_ctx8_eval_results.json",
            short_dir / "short_ctx8_full_info.json",
            short_dir / "run_metadata.json",
            i2v_dir / "i2v_ctx8_eval_results.json",
            i2v_dir / "i2v_ctx8_full_info.json",
            i2v_dir / "run_metadata.json",
        ):
            if path.exists():
                backup_file(path, tag)

        filter_eval_results(short_dir / "short_ctx8_eval_results.json", removed_video_names)
        filter_eval_results(i2v_dir / "i2v_ctx8_eval_results.json", removed_video_names)
        filter_full_info(short_dir / "short_ctx8_full_info.json", removed_video_names)
        filter_full_info(i2v_dir / "i2v_ctx8_full_info.json", removed_video_names)
        filter_run_metadata(short_dir / "run_metadata.json", removed_sample_ids)
        filter_run_metadata(i2v_dir / "run_metadata.json", removed_sample_ids)

        if dataset_name in summary_payload:
            summary_payload[dataset_name]["count"] = len(kept_rows)
            summary_payload[dataset_name]["removed_empty_gt_one_third"] = sorted(removed_sample_ids)
            summary_payload[dataset_name]["removed_empty_gt_one_third_count"] = len(removed_sample_ids)

        audit["datasets"][dataset_name] = {
            "count_before": count_before,
            "count_after": len(kept_rows),
            "removed_count": len(removed_sample_ids),
            "removed_samples": removed_details,
        }

    write_json(summary_path, summary_payload)
    audit_path = archive_root / "empty_clip_filter_audit.json"
    write_json(audit_path, audit)
    print(json.dumps({"audit_path": str(audit_path), "datasets": audit["datasets"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
