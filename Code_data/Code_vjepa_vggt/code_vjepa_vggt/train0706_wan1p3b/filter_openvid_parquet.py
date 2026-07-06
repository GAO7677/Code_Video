#!/usr/bin/env python3
"""
Filter OpenVid parquet shards using the same acceptance rules as the old training loader.

Example:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/home/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/filter_openvid_parquet.py \
  --input-root /data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train \
  --report-root /data/gaoya/agent-data/outputs/openvid_filter_reports/current_scan \
  --num-frames 81

Example with filtered shard export:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/home/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/filter_openvid_parquet.py \
  --input-root /data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train \
  --output-root /data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train_filtered_ctx81 \
  --report-root /data/gaoya/agent-data/outputs/openvid_filter_reports/train_filtered_ctx81 \
  --num-frames 81
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import pyarrow as pa
import pyarrow.parquet as pq
import torch


def clean_text(text: Any) -> str:
    return " ".join(str(text).strip().split())


def decode_info(blob: bytes) -> dict[str, Any]:
    info = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    if not isinstance(info, dict):
        raise TypeError(f"Expected dict in info blob, got {type(info).__name__}.")
    return info


def probe_video_num_frames(raw_video: bytes) -> int:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            handle.write(raw_video)
            temp_path = handle.name
        reader = imageio.get_reader(temp_path)
        try:
            return int(reader.count_frames())
        finally:
            reader.close()
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter OpenVid parquet shards with the same rules used by the training loader."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Directory containing OpenVid parquet shards.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional directory to write filtered parquet shards. If omitted, only reports are produced.",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=None,
        help="Directory for summary/manifests. Defaults to <input-root>/../filter_reports/<timestamp>.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=81,
        help="Minimum frame count required by the downstream training loader.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Debug option: only inspect the first N parquet files.")
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Debug option: only inspect the first N rows in each parquet shard.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue scanning remaining files after a fatal shard-level error.",
    )
    return parser.parse_args()


def resolve_report_root(args: argparse.Namespace) -> Path:
    if args.report_root is not None:
        return args.report_root.expanduser().resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return (args.input_root.expanduser().resolve().parent / "filter_reports" / f"scan_{timestamp}").resolve()


def build_row_table(schema: pa.Schema, info_blob: bytes, raw_video: bytes) -> pa.Table:
    return pa.table(
        {
            "info": pa.array([info_blob], type=schema.field("info").type),
            "raw_video": pa.array([raw_video], type=schema.field("raw_video").type),
        },
        schema=schema,
    )


def classify_row(info_blob: bytes, raw_video: bytes, min_frames: int) -> tuple[bool, dict[str, Any]]:
    try:
        info = decode_info(info_blob)
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        return False, {"reason": "bad_info_blob", "error": f"{type(exc).__name__}: {exc}"}

    prompt = clean_text(info.get("caption", ""))
    if not prompt:
        return False, {"reason": "missing_caption", "caption": ""}

    try:
        num_frames = probe_video_num_frames(raw_video)
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        return False, {"reason": "video_decode_error", "caption": prompt, "error": f"{type(exc).__name__}: {exc}"}

    if num_frames < min_frames:
        return False, {
            "reason": "too_few_frames",
            "caption": prompt,
            "num_frames": int(num_frames),
            "required_min_frames": int(min_frames),
        }

    return True, {
        "reason": "accepted",
        "caption": prompt,
        "num_frames": int(num_frames),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve() if args.output_root is not None else None
    report_root = resolve_report_root(args)

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input OpenVid root does not exist: {input_root}")

    parquet_paths = sorted(input_root.glob("*.parquet"))
    if args.max_files is not None:
        parquet_paths = parquet_paths[: max(0, int(args.max_files))]
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under: {input_root}")

    report_root.mkdir(parents=True, exist_ok=True)
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    accepted_jsonl = (report_root / "accepted_rows.jsonl").open("w", encoding="utf-8")
    skipped_jsonl = (report_root / "skipped_rows.jsonl").open("w", encoding="utf-8")

    totals = Counter()
    fatal_files: list[dict[str, Any]] = []
    file_summaries: list[dict[str, Any]] = []
    started_at = time.time()

    try:
        for file_idx, parquet_path in enumerate(parquet_paths):
            file_summary = {
                "file_index": int(file_idx),
                "input_path": str(parquet_path),
                "rows_total": 0,
                "rows_scanned": 0,
                "rows_accepted": 0,
                "rows_skipped": 0,
                "skip_reasons": {},
                "output_path": None,
            }
            skip_reasons = Counter()
            writer = None

            try:
                parquet_file = pq.ParquetFile(parquet_path)
                schema = parquet_file.schema_arrow
                if "info" not in schema.names or "raw_video" not in schema.names:
                    raise KeyError(f"Expected parquet columns ['info', 'raw_video'], got {schema.names}")

                total_rows = int(parquet_file.metadata.num_rows)
                rows_to_scan = total_rows if args.max_rows_per_file is None else min(total_rows, int(args.max_rows_per_file))
                file_summary["rows_total"] = total_rows

                output_path = None
                if output_root is not None:
                    output_path = output_root / parquet_path.name
                    file_summary["output_path"] = str(output_path)

                for row_index in range(rows_to_scan):
                    row = parquet_file.read_row_group(row_index, columns=["info", "raw_video"])
                    info_blob = row.column("info")[0].as_py()
                    raw_video = row.column("raw_video")[0].as_py()

                    accepted, meta = classify_row(info_blob, raw_video, min_frames=args.num_frames)
                    base_row = {
                        "file_index": int(file_idx),
                        "file_name": parquet_path.name,
                        "input_path": str(parquet_path),
                        "row_index": int(row_index),
                        "raw_video_bytes": len(raw_video),
                    }
                    file_summary["rows_scanned"] += 1
                    totals["rows_scanned"] += 1

                    if accepted:
                        totals["rows_accepted"] += 1
                        file_summary["rows_accepted"] += 1
                        append_jsonl(accepted_jsonl, {**base_row, **meta})
                        if output_path is not None:
                            if writer is None:
                                writer = pq.ParquetWriter(output_path, schema=schema)
                            # Keep one row per row group so the existing training loader continues to work.
                            writer.write_table(build_row_table(schema, info_blob, raw_video))
                    else:
                        reason = str(meta.get("reason", "unknown"))
                        totals["rows_skipped"] += 1
                        totals[f"skip_{reason}"] += 1
                        file_summary["rows_skipped"] += 1
                        skip_reasons[reason] += 1
                        append_jsonl(skipped_jsonl, {**base_row, **meta})

                if writer is not None:
                    writer.close()
                    writer = None
                elif output_root is not None and output_path is not None and output_path.exists():
                    output_path.unlink()
            except Exception as exc:
                if writer is not None:
                    writer.close()
                    writer = None
                fatal_row = {
                    "file_index": int(file_idx),
                    "input_path": str(parquet_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                fatal_files.append(fatal_row)
                totals["fatal_files"] += 1
                if not args.keep_going:
                    raise
            finally:
                file_summary["skip_reasons"] = dict(skip_reasons)
                file_summaries.append(file_summary)

            print(
                f"[scan] {file_idx + 1}/{len(parquet_paths)} {parquet_path.name} "
                f"accepted={file_summary['rows_accepted']} skipped={file_summary['rows_skipped']}"
            )
    finally:
        accepted_jsonl.close()
        skipped_jsonl.close()

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root) if output_root is not None else None,
        "report_root": str(report_root),
        "num_input_files": len(parquet_paths),
        "num_frames_required": int(args.num_frames),
        "max_files": args.max_files,
        "max_rows_per_file": args.max_rows_per_file,
        "rows_scanned": int(totals["rows_scanned"]),
        "rows_accepted": int(totals["rows_accepted"]),
        "rows_skipped": int(totals["rows_skipped"]),
        "skip_reasons": {
            key.replace("skip_", "", 1): int(value)
            for key, value in sorted(totals.items())
            if key.startswith("skip_")
        },
        "fatal_files": fatal_files,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "accepted_manifest": str((report_root / "accepted_rows.jsonl").resolve()),
        "skipped_manifest": str((report_root / "skipped_rows.jsonl").resolve()),
        "file_summary_path": str((report_root / "file_summaries.json").resolve()),
    }
    write_json(report_root / "summary.json", summary)
    write_json(report_root / "file_summaries.json", file_summaries)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
