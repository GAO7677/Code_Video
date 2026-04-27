#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import imageio
import pyarrow.parquet as pq
import torch
from PIL import Image


DEFAULT_DATASET_ROOT = Path(
    "/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/val"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/mytest"
)
DEFAULT_DATASET_ID = "mvp-lab/OpenVidHD-0.4M-720p-48fps"
DEFAULT_SPLIT = "val"
DEFAULT_CONTEXT_FRAMES = 8
DEFAULT_NUM_WORKERS = min(8, os.cpu_count() or 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert OpenVid val parquet shards into a meta.json-driven mytest layout."
        )
    )
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--context_frames", type=int, default=DEFAULT_CONTEXT_FRAMES)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of worker processes used to prepare samples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples for a smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sample directories and manifest files.",
    )
    return parser.parse_args()


def decode_info_blob(blob: bytes) -> dict[str, Any]:
    info = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    if not isinstance(info, dict):
        raise TypeError(f"Expected dict from info blob, got {type(info).__name__}.")
    return info


def gather_parquet_paths(dataset_root: Path) -> list[Path]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    parquet_paths = sorted(dataset_root.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under: {dataset_root}")
    return parquet_paths


def iter_samples(
    dataset_root: Path,
) -> Iterable[tuple[int, Path, int, dict[str, Any], bytes]]:
    global_index = 0
    for parquet_path in gather_parquet_paths(dataset_root):
        table = pq.read_table(parquet_path, columns=["info", "raw_video"])
        info_column = table.column("info")
        raw_video_column = table.column("raw_video")
        for local_row_index in range(table.num_rows):
            info = decode_info_blob(info_column[local_row_index].as_py())
            raw_video = raw_video_column[local_row_index].as_py()
            yield global_index, parquet_path, local_row_index, info, raw_video
            global_index += 1


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def sanitize_filename(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("._")
    return safe or "sample"


def build_sample_id(parquet_path: Path, local_row_index: int, source_video_name: str) -> str:
    stem = Path(source_video_name).stem if source_video_name else ""
    if stem:
        return f"{parquet_path.stem}__{local_row_index:05d}__{sanitize_filename(stem)}"
    return f"{parquet_path.stem}__{local_row_index:05d}"


def prepare_sample_dir(sample_dir: Path, overwrite: bool) -> None:
    if sample_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Sample directory already exists: {sample_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)


def probe_video(full_video_path: Path) -> tuple[float, int]:
    reader = imageio.get_reader(full_video_path)
    try:
        meta = reader.get_meta_data()
        fps = float(meta.get("fps") or 0.0)
        frame_count = int(reader.count_frames())
    finally:
        reader.close()
    if fps <= 0:
        raise ValueError(f"Failed to resolve fps from {full_video_path}")
    if frame_count <= 0:
        raise ValueError(f"Failed to resolve frame count from {full_video_path}")
    return fps, frame_count


def export_split_media(
    *,
    full_video_path: Path,
    context_video_path: Path,
    future_gt_video_path: Path,
    first_frame_path: Path,
    context_frames: int,
    fps: float,
) -> None:
    reader = imageio.get_reader(full_video_path)
    try:
        with imageio.get_writer(
            context_video_path,
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            ffmpeg_log_level="error",
            quality=None,
            output_params=["-preset", "ultrafast", "-crf", "28"],
        ) as context_writer, imageio.get_writer(
            future_gt_video_path,
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            ffmpeg_log_level="error",
            quality=None,
            output_params=["-preset", "ultrafast", "-crf", "28"],
        ) as future_writer:
            for frame_index, frame in enumerate(reader):
                if frame_index < context_frames:
                    context_writer.append_data(frame)
                else:
                    future_writer.append_data(frame)
                if frame_index == context_frames - 1:
                    Image.fromarray(frame).save(first_frame_path)
    finally:
        reader.close()


def write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {path}. Pass --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_sample_json(
    *,
    sample_id: str,
    sample_dir: Path,
    full_video_path: Path,
    context_video_path: Path,
    future_gt_video_path: Path,
    first_frame_path: Path,
    dataset_id: str,
    split: str,
    global_index: int,
    parquet_path: Path,
    local_row_index: int,
    source_video_name: str | None,
    info: dict[str, Any],
    decoded_fps: float,
    decoded_frame_count: int,
    context_frames: int,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "caption": str(info.get("caption", "")).strip(),
        "dataset_id": dataset_id,
        "split": split,
        "source_video_name": source_video_name,
        "global_index": global_index,
        "parquet_file": parquet_path.name,
        "parquet_path": str(parquet_path.resolve()),
        "parquet_row_index": local_row_index,
        "fps": decoded_fps,
        "decoded_num_frames": decoded_frame_count,
        "context_frames": context_frames,
        "future_frames": decoded_frame_count - context_frames,
        "context_frame_range": [0, context_frames - 1],
        "future_frame_range": [context_frames, decoded_frame_count - 1],
        "first_frame_index_in_full_video": context_frames - 1,
        "duration_seconds": decoded_frame_count / decoded_fps,
        "camera_motion": info.get("camera motion"),
        "aesthetic_score": safe_float(info.get("aesthetic score")),
        "motion_score": safe_float(info.get("motion score")),
        "temporal_consistency_score": safe_float(info.get("temporal consistency score")),
        "info_fps": safe_float(info.get("fps")),
        "info_num_frames": safe_int(info.get("frame")),
        "info_duration_seconds": safe_float(info.get("seconds")),
        "paths": {
            "sample_dir": str(sample_dir.resolve()),
            "full_video_path": str(full_video_path.resolve()),
            "context_video_path": str(context_video_path.resolve()),
            "future_gt_video_path": str(future_gt_video_path.resolve()),
            "first_frame_path": str(first_frame_path.resolve()),
        },
        "source_paths": {
            "original_parquet_path": str(parquet_path.resolve()),
            "original_dataset_root": str(parquet_path.parent.resolve()),
        },
    }


def build_task(
    *,
    dataset_id: str,
    split: str,
    output_root: Path,
    context_frames: int,
    overwrite: bool,
    global_index: int,
    parquet_path: Path,
    local_row_index: int,
    info: dict[str, Any],
    raw_video: bytes,
) -> dict[str, Any]:
    source_video_name = str(info.get("video") or "").strip()
    sample_id = build_sample_id(parquet_path, local_row_index, source_video_name)
    return {
        "dataset_id": dataset_id,
        "split": split,
        "output_root": str(output_root),
        "context_frames": context_frames,
        "overwrite": overwrite,
        "global_index": global_index,
        "parquet_path": str(parquet_path),
        "local_row_index": local_row_index,
        "info": info,
        "raw_video": raw_video,
        "source_video_name": source_video_name,
        "sample_id": sample_id,
    }


def prepare_one_sample(task: dict[str, Any]) -> dict[str, Any]:
    parquet_path = Path(task["parquet_path"])
    output_root = Path(task["output_root"])
    sample_id = str(task["sample_id"])
    source_video_name = str(task["source_video_name"] or "")
    info = dict(task["info"])
    raw_video = bytes(task["raw_video"])
    context_frames = int(task["context_frames"])
    overwrite = bool(task["overwrite"])

    sample_dir = output_root / sample_id
    prepare_sample_dir(sample_dir, overwrite=overwrite)

    full_video_path = sample_dir / "full_video.mp4"
    context_video_path = sample_dir / "context_video.mp4"
    future_gt_video_path = sample_dir / "future_gt_video.mp4"
    first_frame_path = sample_dir / "first_frame.png"
    meta_path = sample_dir / "meta.json"

    full_video_path.write_bytes(raw_video)
    decoded_fps, decoded_frame_count = probe_video(full_video_path)
    if decoded_frame_count <= context_frames:
        raise ValueError(
            f"Video {sample_id} has only {decoded_frame_count} decoded frames, "
            f"which is not enough for context_frames={context_frames}."
        )

    export_split_media(
        full_video_path=full_video_path,
        context_video_path=context_video_path,
        future_gt_video_path=future_gt_video_path,
        first_frame_path=first_frame_path,
        context_frames=context_frames,
        fps=decoded_fps,
    )

    sample_json = build_sample_json(
        sample_id=sample_id,
        sample_dir=sample_dir,
        full_video_path=full_video_path,
        context_video_path=context_video_path,
        future_gt_video_path=future_gt_video_path,
        first_frame_path=first_frame_path,
        dataset_id=str(task["dataset_id"]),
        split=str(task["split"]),
        global_index=int(task["global_index"]),
        parquet_path=parquet_path,
        local_row_index=int(task["local_row_index"]),
        source_video_name=source_video_name or None,
        info=info,
        decoded_fps=decoded_fps,
        decoded_frame_count=decoded_frame_count,
        context_frames=context_frames,
    )
    write_json(meta_path, sample_json, overwrite=True)
    return sample_json


def main() -> None:
    args = parse_args()
    if args.context_frames < 1:
        raise ValueError(f"context_frames must be >= 1, got {args.context_frames}")
    if args.num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {args.num_workers}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.jsonl"
    summary_path = args.output_root / "summary.json"

    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing manifest: {manifest_path}. Pass --overwrite to replace it."
        )
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing summary: {summary_path}. Pass --overwrite to replace it."
        )

    sample_records: list[dict[str, Any]] = []
    processed = 0

    def task_iter() -> Iterable[dict[str, Any]]:
        produced = 0
        for global_index, parquet_path, local_row_index, info, raw_video in iter_samples(
            args.dataset_root
        ):
            if args.limit is not None and produced >= args.limit:
                break
            yield build_task(
                dataset_id=args.dataset_id,
                split=args.split,
                output_root=args.output_root,
                context_frames=args.context_frames,
                overwrite=args.overwrite,
                global_index=global_index,
                parquet_path=parquet_path,
                local_row_index=local_row_index,
                info=info,
                raw_video=raw_video,
            )
            produced += 1

    if args.num_workers == 1:
        results_iter = map(prepare_one_sample, task_iter())
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers)
        results_iter = executor.map(prepare_one_sample, task_iter(), chunksize=1)

    try:
        for sample_json in results_iter:
            sample_records.append(sample_json)
            processed += 1
            print(f"[{processed}] {sample_json['sample_id']}")
    finally:
        if args.num_workers != 1:
            executor.shutdown(wait=True, cancel_futures=False)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in sample_records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary_payload = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "dataset_root": str(args.dataset_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "sample_count": processed,
        "context_frames": args.context_frames,
        "manifest_jsonl": str(manifest_path.resolve()),
    }
    write_json(summary_path, summary_payload, overwrite=True)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
