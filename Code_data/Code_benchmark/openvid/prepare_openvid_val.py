#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
import torch


DEFAULT_DATASET_ROOT = Path(
    "/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/val"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/Code_benchmark/openvid/processed_val"
)
DEFAULT_DATASET_ID = "mvp-lab/OpenVidHD-0.4M-720p-48fps"
DEFAULT_SPLIT = "val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read the OpenVid val parquet shards and convert them into a "
            "benchmark-friendly manifest.jsonl."
        )
    )
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--export-videos",
        action="store_true",
        help="Write raw mp4 bytes from each parquet row into output_root/videos.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples for a quick smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite manifest/summary/schema files if they already exist.",
    )
    return parser.parse_args()


def ensure_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {path}. "
            "Pass --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


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


def parquet_schema_text(dataset_root: Path) -> str:
    first_path = gather_parquet_paths(dataset_root)[0]
    return str(pq.ParquetFile(first_path).schema)


def iter_samples(dataset_root: Path) -> Iterable[tuple[int, Path, int, dict[str, Any], bytes]]:
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


def build_sample_id(parquet_path: Path, local_row_index: int) -> str:
    return f"{parquet_path.stem}__{local_row_index:05d}"


def build_manifest_record(
    *,
    dataset_id: str,
    split: str,
    dataset_root: Path,
    output_root: Path,
    parquet_path: Path,
    global_index: int,
    local_row_index: int,
    info: dict[str, Any],
    raw_video: bytes,
    export_videos: bool,
) -> dict[str, Any]:
    sample_id = build_sample_id(parquet_path, local_row_index)
    raw_video_path = None
    if export_videos:
        raw_video_path = output_root / "videos" / f"{sample_id}.mp4"
        raw_video_path.parent.mkdir(parents=True, exist_ok=True)
        raw_video_path.write_bytes(raw_video)

    return {
        "sample_id": sample_id,
        "dataset_id": dataset_id,
        "split": split,
        "global_index": global_index,
        "parquet_file": parquet_path.name,
        "parquet_path": str(parquet_path.resolve()),
        "parquet_row_index": local_row_index,
        "source_video_name": info.get("video"),
        "caption": str(info.get("caption", "")).strip(),
        "fps": safe_float(info.get("fps")),
        "num_frames": safe_int(info.get("frame")),
        "duration_seconds": safe_float(info.get("seconds")),
        "camera_motion": info.get("camera motion"),
        "aesthetic_score": safe_float(info.get("aesthetic score")),
        "motion_score": safe_float(info.get("motion score")),
        "temporal_consistency_score": safe_float(
            info.get("temporal consistency score")
        ),
        "raw_video_num_bytes": len(raw_video),
        "raw_video_path": str(raw_video_path.resolve()) if raw_video_path else None,
        "dataset_root": str(dataset_root.resolve()),
    }


def write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    ensure_output_path(path, overwrite=overwrite)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_root / "manifest.jsonl"
    summary_path = args.output_root / "summary.json"
    schema_path = args.output_root / "dataset_format.json"

    for target in [manifest_path, summary_path, schema_path]:
        ensure_output_path(target, overwrite=args.overwrite)

    processed = 0
    total_raw_video_bytes = 0
    captions_with_text = 0
    example_record: dict[str, Any] | None = None

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for global_index, parquet_path, local_row_index, info, raw_video in iter_samples(
            args.dataset_root
        ):
            if args.limit is not None and processed >= args.limit:
                break

            record = build_manifest_record(
                dataset_id=args.dataset_id,
                split=args.split,
                dataset_root=args.dataset_root,
                output_root=args.output_root,
                parquet_path=parquet_path,
                global_index=global_index,
                local_row_index=local_row_index,
                info=info,
                raw_video=raw_video,
                export_videos=args.export_videos,
            )
            manifest_file.write(
                json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n"
            )
            processed += 1
            total_raw_video_bytes += record["raw_video_num_bytes"]
            if record["caption"]:
                captions_with_text += 1
            if example_record is None:
                example_record = record

            if processed % 100 == 0:
                print(f"[processed] {processed} samples")

    format_payload = {
        "source_dataset": {
            "dataset_id": args.dataset_id,
            "split": args.split,
            "dataset_root": str(args.dataset_root.resolve()),
            "source_parquet_schema": parquet_schema_text(args.dataset_root),
            "source_parquet_columns": [
                {
                    "name": "info",
                    "type": "binary",
                    "description": (
                        "torch-serialized Python dict with video metadata such as "
                        "caption/fps/frame count/scores."
                    ),
                },
                {
                    "name": "raw_video",
                    "type": "binary",
                    "description": "Raw mp4 bytes for the sample video.",
                },
            ],
            "decoded_info_keys": [
                "video",
                "caption",
                "aesthetic score",
                "motion score",
                "temporal consistency score",
                "camera motion",
                "frame",
                "fps",
                "seconds",
            ],
        },
        "normalized_manifest_jsonl": {
            "path": str(manifest_path.resolve()),
            "one_line_per_sample": True,
            "fields": [
                "sample_id",
                "dataset_id",
                "split",
                "global_index",
                "parquet_file",
                "parquet_path",
                "parquet_row_index",
                "source_video_name",
                "caption",
                "fps",
                "num_frames",
                "duration_seconds",
                "camera_motion",
                "aesthetic_score",
                "motion_score",
                "temporal_consistency_score",
                "raw_video_num_bytes",
                "raw_video_path",
                "dataset_root",
            ],
            "example": example_record,
        },
    }

    summary_payload = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "dataset_root": str(args.dataset_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "manifest_jsonl": str(manifest_path.resolve()),
        "dataset_format_json": str(schema_path.resolve()),
        "sample_count": processed,
        "captions_with_text": captions_with_text,
        "total_raw_video_bytes": total_raw_video_bytes,
        "export_videos": args.export_videos,
        "limit": args.limit,
    }

    write_json(summary_path, summary_payload, overwrite=True)
    write_json(schema_path, format_payload, overwrite=True)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
