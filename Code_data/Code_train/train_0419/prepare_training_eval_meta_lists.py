#!/usr/bin/env python3
"""Prepare fixed-generation and validation meta lists for training-time evaluation."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import tempfile
from bisect import bisect_right
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image


TRAIN0419_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419")
OPENVID_ROOT = Path("/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train")
OPENVID_MYTEST_ROOT = Path("/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/mytest_train_eval")
MOVID_MYTEST_ROOT = Path("/data/gaoya/dataset/kubric_tfds_movi-d/mytest")
GENESIS_MYTEST_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/mytest"
)

FIXED_META_LIST_PATH = TRAIN0419_ROOT / "benchmark_meta_json_paths_fixed24.txt"
VALIDATION_META_LIST_PATH = TRAIN0419_ROOT / "benchmark_meta_json_paths_validation100.txt"
SUMMARY_PATH = TRAIN0419_ROOT / "benchmark_meta_json_paths_training_eval_summary.json"

OPENVID_FIXED_COUNT = 12
OPENVID_VALIDATION_COUNT = 50
MOVID_FIXED_COUNT = 6
MOVID_VALIDATION_COUNT = 25
GENESIS_FIXED_COUNT = 6
GENESIS_VALIDATION_COUNT = 25

CLIP_FRAMES = 24
CONTEXT_FRAMES = 8
OPENVID_SELECTION_SEED = 20260424
MOVID_SELECTION_SEED = 20260425
GENESIS_SELECTION_SEED = 20260426
COMBINED_SHUFFLE_SEED = 20260427
SAMPLE_DATASET_ORDER = ["MOVI-D", "Genesis", "OpenVid"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare training-time benchmark meta lists and sampled benchmark subsets."
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sample-from-meta-list", type=Path, default=None)
    parser.add_argument(
        "--sample-output-path",
        type=Path,
        default=TRAIN0419_ROOT / "benchmark_meta_json_paths_full_sample300.txt",
    )
    parser.add_argument("--sample-total", type=int, default=300)
    parser.add_argument("--sample-seed", type=int, default=42)
    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_txt(path: Path, values: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(value) for value in values) + "\n", encoding="utf-8")


def clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def read_meta_paths(path: Path) -> list[Path]:
    rows: list[Path] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line).expanduser()
        if not candidate.is_absolute():
            candidate = (path.parent / candidate).resolve()
        key = str(candidate)
        if key in seen:
            continue
        if not candidate.is_file():
            raise FileNotFoundError(f"meta.json not found: {candidate}")
        seen.add(key)
        rows.append(candidate)
    if not rows:
        raise ValueError(f"no meta paths found in: {path}")
    return rows


def infer_dataset_name_from_meta(meta_path: Path, meta_payload: dict) -> str:
    dataset_field = meta_payload.get("dataset")
    dataset_text = str(dataset_field).strip().lower() if dataset_field is not None else ""
    meta_path_text = str(meta_path)
    if dataset_text == "movi-d" or "/kubric_tfds_movi-d/" in meta_path_text or "/vLAR-PhysInOne/" in meta_path_text:
        return "MOVI-D"
    if "genesis" in dataset_text or "/version_1_genesis_rigid_data_all_cases/" in meta_path_text:
        return "Genesis"
    if (
        "openvid" in dataset_text
        or "/mvp-lab-OpenVidHD-0.4M-720p-48fps/" in meta_path_text
        or "/physics-iq-benchmark/" in meta_path_text
    ):
        return "OpenVid"
    raise ValueError(f"cannot infer dataset for meta path: {meta_path}")


def proportional_quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    total_available = sum(counts.values())
    if total_available < total:
        raise ValueError(f"requested {total} samples but only {total_available} are available")
    quotas = {name: int(total * count / total_available) for name, count in counts.items()}
    remainder = total - sum(quotas.values())
    if remainder > 0:
        ranked = sorted(
            counts,
            key=lambda name: ((total * counts[name] / total_available) - quotas[name], counts[name], name),
            reverse=True,
        )
        for name in ranked[:remainder]:
            quotas[name] += 1
    return quotas


def sample_existing_meta_list(
    input_path: Path,
    output_path: Path,
    *,
    total: int,
    seed: int,
) -> dict[str, object]:
    meta_paths = read_meta_paths(input_path)
    buckets: dict[str, list[Path]] = {name: [] for name in SAMPLE_DATASET_ORDER}
    for meta_path in meta_paths:
        meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        dataset_name = infer_dataset_name_from_meta(meta_path, meta_payload)
        buckets[dataset_name].append(meta_path)

    counts = {name: len(paths) for name, paths in buckets.items()}
    quotas = proportional_quotas(counts, total)
    rng = random.Random(seed)

    sampled: list[Path] = []
    sampled_by_dataset: dict[str, list[str]] = {}
    for dataset_name in SAMPLE_DATASET_ORDER:
        candidates = sorted(buckets[dataset_name], key=lambda item: str(item))
        rng.shuffle(candidates)
        selected = sorted(candidates[: quotas[dataset_name]], key=lambda item: str(item))
        sampled.extend(selected)
        sampled_by_dataset[dataset_name] = [str(path) for path in selected]

    sampled = sorted(sampled, key=lambda item: str(item))
    write_txt(output_path, sampled)
    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "seed": seed,
        "total_requested": total,
        "total_written": len(sampled),
        "dataset_counts": counts,
        "dataset_quotas": quotas,
        "sampled_by_dataset": sampled_by_dataset,
    }
    write_json(summary_path, summary)
    return summary


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=6,
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def save_frame_png(frame: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


class OpenVidSource:
    def __init__(self, root: Path):
        self.root = root
        self.files = self._gather_files(root)
        if not self.files:
            raise FileNotFoundError(f"No OpenVid parquet files found under {root}")
        self.total_rows = self.files[-1]["cumulative_rows"]
        self._parquet_handles: dict[str, pq.ParquetFile] = {}

    @staticmethod
    def _decode_info(blob):
        info = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
        if not isinstance(info, dict):
            raise TypeError(f"Expected dict info blob, got {type(info).__name__}.")
        return info

    @staticmethod
    def _gather_files(root: Path):
        parquet_paths = sorted(root.glob("*.parquet"))
        files = []
        cumulative_rows = 0
        for path in parquet_paths:
            parquet_file = pq.ParquetFile(path)
            rows = parquet_file.metadata.num_rows
            cumulative_rows += rows
            files.append(
                {
                    "path": str(path),
                    "rows": rows,
                    "cumulative_rows": cumulative_rows,
                }
            )
        return files

    def _get_parquet_file(self, path: str):
        parquet_file = self._parquet_handles.get(path)
        if parquet_file is None:
            parquet_file = pq.ParquetFile(path)
            self._parquet_handles[path] = parquet_file
        return parquet_file

    def _map_global_row(self, row_index: int):
        cumulative = [item["cumulative_rows"] for item in self.files]
        file_index = bisect_right(cumulative, row_index)
        previous_rows = 0 if file_index == 0 else cumulative[file_index - 1]
        local_row_index = row_index - previous_rows
        return self.files[file_index], local_row_index

    def read_row(self, row_index: int):
        file_info, local_row_index = self._map_global_row(row_index)
        parquet_file = self._get_parquet_file(file_info["path"])
        row = parquet_file.read_row_group(local_row_index, columns=["info", "raw_video"])
        info = self._decode_info(row.column("info")[0].as_py())
        raw_video = row.column("raw_video")[0].as_py()
        return info, raw_video, file_info["path"], local_row_index


def decode_openvid_clip(raw_video: bytes, sample_key: str, clip_frames: int) -> tuple[list[np.ndarray], int, int, int]:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(raw_video)
            temp_path = temp_file.name
        reader = imageio.get_reader(temp_path)
        try:
            total_frames = reader.count_frames()
            if total_frames < clip_frames:
                raise ValueError(
                    f"OpenVid sample has only {total_frames} frames, need at least {clip_frames}."
                )
            meta = reader.get_meta_data()
            fps = int(round(float(meta.get("fps", 24)))) if meta.get("fps") else 24
            max_start = total_frames - clip_frames
            start_frame = stable_hash(sample_key) % (max_start + 1)
            frames = [
                np.asarray(reader.get_data(frame_id), dtype=np.uint8)
                for frame_id in range(start_frame, start_frame + clip_frames)
            ]
            return frames, max(1, fps), start_frame, total_frames
        finally:
            reader.close()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def materialize_openvid_eval_samples(required_count: int, overwrite: bool) -> list[Path]:
    source = OpenVidSource(OPENVID_ROOT)
    candidate_indices = list(range(source.total_rows))
    random.Random(OPENVID_SELECTION_SEED).shuffle(candidate_indices)

    meta_paths: list[Path] = []
    for row_index in candidate_indices:
        if len(meta_paths) >= required_count:
            break
        try:
            info, raw_video, parquet_path, parquet_row_group = source.read_row(row_index)
            caption = clean_text(info.get("caption", ""))
            if not caption:
                continue

            sample_slot = len(meta_paths)
            sample_id = f"openvid_eval_{sample_slot:04d}__row_{row_index}"
            sample_dir = OPENVID_MYTEST_ROOT / sample_id
            meta_path = sample_dir / "meta.json"
            frames, fps, start_frame, raw_total_frames = decode_openvid_clip(
                raw_video,
                sample_key=sample_id,
                clip_frames=CLIP_FRAMES,
            )

            context = frames[:CONTEXT_FRAMES]
            future = frames[CONTEXT_FRAMES:]
            if not future:
                future = frames[-1:]

            if overwrite or not meta_path.exists():
                sample_dir.mkdir(parents=True, exist_ok=True)
                context_video_path = sample_dir / "context_video.mp4"
                future_gt_video_path = sample_dir / "future_gt_video.mp4"
                full_video_path = sample_dir / "full_video.mp4"
                first_frame_path = sample_dir / "first_frame.png"

                write_video(context_video_path, context, fps=fps)
                write_video(future_gt_video_path, future, fps=fps)
                write_video(full_video_path, frames, fps=fps)
                save_frame_png(context[0], first_frame_path)

                payload = {
                    "sample_id": sample_id,
                    "caption": caption,
                    "description": caption,
                    "dataset": "OpenVid",
                    "split": "train_eval",
                    "fps": fps,
                    "context_frames": len(context),
                    "future_frames": len(future),
                    "raw_frames": len(frames),
                    "raw_total_frames": raw_total_frames,
                    "clip_start_frame": start_frame,
                    "paths": {
                        "sample_dir": str(sample_dir),
                        "context_video_path": str(context_video_path),
                        "future_gt_video_path": str(future_gt_video_path),
                        "full_video_path": str(full_video_path),
                        "first_frame_path": str(first_frame_path),
                    },
                    "source_paths": {
                        "meta_json_path": str(meta_path),
                        "parquet_path": str(parquet_path),
                        "parquet_row_group": parquet_row_group,
                        "global_row_index": row_index,
                    },
                }
                write_json(meta_path, payload)
            meta_paths.append(meta_path)
        except Exception:
            continue

    if len(meta_paths) < required_count:
        raise RuntimeError(
            f"Only prepared {len(meta_paths)} OpenVid eval samples, expected {required_count}."
        )
    return meta_paths


def choose_disjoint_paths(all_paths: list[Path], fixed_count: int, validation_count: int, seed: int):
    shuffled = list(all_paths)
    random.Random(seed).shuffle(shuffled)
    fixed = shuffled[:fixed_count]
    validation = shuffled[fixed_count : fixed_count + validation_count]
    if len(fixed) != fixed_count or len(validation) != validation_count:
        raise RuntimeError(
            f"Insufficient paths: need fixed={fixed_count}, validation={validation_count}, "
            f"got fixed={len(fixed)}, validation={len(validation)}."
        )
    return fixed, validation


def combine_and_shuffle(groups: list[list[Path]], seed: int) -> list[Path]:
    combined = [path for group in groups for path in group]
    random.Random(seed).shuffle(combined)
    return combined


def main() -> None:
    args = parse_args()
    if args.sample_from_meta_list is not None:
        summary = sample_existing_meta_list(
            args.sample_from_meta_list,
            args.sample_output_path,
            total=args.sample_total,
            seed=args.sample_seed,
        )
        print(
            json.dumps(
                {
                    "sampled_total": summary["total_written"],
                    "sample_output_path": str(args.sample_output_path),
                    "sample_summary_path": str(args.sample_output_path.with_suffix(".summary.json")),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    openvid_meta_paths = materialize_openvid_eval_samples(
        required_count=OPENVID_FIXED_COUNT + OPENVID_VALIDATION_COUNT,
        overwrite=args.overwrite,
    )
    openvid_fixed = openvid_meta_paths[:OPENVID_FIXED_COUNT]
    openvid_validation = openvid_meta_paths[OPENVID_FIXED_COUNT:]

    movid_all = sorted(MOVID_MYTEST_ROOT.glob("*/meta.json"))
    genesis_all = sorted(GENESIS_MYTEST_ROOT.glob("*/meta.json"))
    movid_fixed, movid_validation = choose_disjoint_paths(
        movid_all,
        fixed_count=MOVID_FIXED_COUNT,
        validation_count=MOVID_VALIDATION_COUNT,
        seed=MOVID_SELECTION_SEED,
    )
    genesis_fixed, genesis_validation = choose_disjoint_paths(
        genesis_all,
        fixed_count=GENESIS_FIXED_COUNT,
        validation_count=GENESIS_VALIDATION_COUNT,
        seed=GENESIS_SELECTION_SEED,
    )

    fixed_paths = combine_and_shuffle(
        [openvid_fixed, movid_fixed, genesis_fixed],
        seed=COMBINED_SHUFFLE_SEED,
    )
    validation_paths = combine_and_shuffle(
        [openvid_validation, movid_validation, genesis_validation],
        seed=COMBINED_SHUFFLE_SEED + 1,
    )

    write_txt(FIXED_META_LIST_PATH, fixed_paths)
    write_txt(VALIDATION_META_LIST_PATH, validation_paths)

    write_json(
        SUMMARY_PATH,
        {
            "fixed24": {
                "path": str(FIXED_META_LIST_PATH),
                "num_samples": len(fixed_paths),
                "counts": {
                    "OpenVid": len(openvid_fixed),
                    "MOVI-D": len(movid_fixed),
                    "GenesisRigid": len(genesis_fixed),
                },
            },
            "validation100": {
                "path": str(VALIDATION_META_LIST_PATH),
                "num_samples": len(validation_paths),
                "counts": {
                    "OpenVid": len(openvid_validation),
                    "MOVI-D": len(movid_validation),
                    "GenesisRigid": len(genesis_validation),
                },
            },
            "openvid_eval_root": str(OPENVID_MYTEST_ROOT),
        },
    )

    print(
        json.dumps(
            {
                "fixed24": len(fixed_paths),
                "validation100": len(validation_paths),
                "fixed24_path": str(FIXED_META_LIST_PATH),
                "validation100_path": str(VALIDATION_META_LIST_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
