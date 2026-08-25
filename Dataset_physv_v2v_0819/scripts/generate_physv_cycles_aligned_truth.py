#!/usr/bin/env python3
"""Generate CYCLES-pixel-aligned dynamic-object truth for the PhysV dataset.

The RGB videos already present in each sample are the source of truth for the
render configuration. For every sample this wrapper reads
``videos/rgb_cycles.json`` and invokes Blender with the same resolution,
Cycles sample count, exposure, camera construction, scene construction, and
trajectory. The Blender companion script writes only a new Object Index pass;
the original ``raw`` truth and ``rgb_cycles.mp4`` are never modified.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
DEFAULT_OUTPUT_ROOT = DEFAULT_DATASET_ROOT / "physv_v2v_0819_cycles_aligned_truth_v1"
DEFAULT_BLENDER = Path("/data/gaoya/agent-data/tools/blender-3.6.23-linux-x64/blender")
BLENDER_SCRIPT = Path(__file__).with_name("render_physv_cycles_aligned_truth.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument(
        "--gpus",
        default="0,1,2",
        help="Physical GPU ids. One Blender process is assigned to each GPU.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        dest="sample_ids",
        help="Limit generation to this sample id; may be supplied more than once.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all selected samples.")
    parser.add_argument(
        "--rerun-complete",
        action="store_true",
        help="Regenerate cases even when a complete aligned-truth case is present.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cycle_config(sample_dir: Path) -> dict:
    cycle_path = sample_dir / "videos" / "rgb_cycles.json"
    data = load_json(cycle_path)
    resolution = data.get("resolution")
    if not resolution and data.get("video"):
        resolution = [data["video"]["width"], data["video"]["height"]]
    if not resolution or len(resolution) != 2:
        raise ValueError(f"missing CYCLES resolution in {cycle_path}")
    video = data.get("video") or {}
    frame_count = int(data.get("frame_count") or video.get("nb_frames") or 0)
    if frame_count <= 0:
        raise ValueError(f"missing CYCLES frame count in {cycle_path}")
    metadata = load_json(sample_dir / "metadata.json")
    return {
        "path": str(cycle_path),
        "engine": str(data.get("engine", "CYCLES")),
        "width": int(resolution[0]),
        "height": int(resolution[1]),
        "samples": int(data.get("samples", 32)),
        "exposure": float(data.get("exposure", 0.0)),
        "frame_count": frame_count,
        "fps": int(metadata["simulation"]["fps"]),
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def complete_case(case_dir: Path, config: dict) -> bool:
    metadata_path = case_dir / "truth_metadata.json"
    masks_path = case_dir / "dynamic_masks.npz"
    trajectory_path = case_dir / "trajectory_pixels.npz"
    if not (metadata_path.is_file() and masks_path.is_file() and trajectory_path.is_file()):
        return False
    try:
        metadata = load_json(metadata_path)
        with np.load(masks_path, allow_pickle=False) as arrays:
            masks = arrays["masks_thw"]
            union = arrays["union_thw"]
        with np.load(trajectory_path, allow_pickle=False) as arrays:
            centers = arrays["centers_tnc"]
        return (
            metadata.get("schema_version") == "physv_cycles_aligned_truth_v1"
            and metadata.get("frame_count") == config["frame_count"]
            and metadata.get("resolution") == [config["width"], config["height"]]
            and masks.ndim == 4
            and masks.shape[1:] == (config["frame_count"], config["height"], config["width"])
            and union.shape == (config["frame_count"], config["height"], config["width"])
            and centers.shape[0] == config["frame_count"]
            and centers.shape[1] == masks.shape[0]
        )
    except Exception:
        return False


def run_case(
    sample_dir: Path,
    output_root: Path,
    logs_dir: Path,
    blender: Path,
    gpu: str,
    rerun_complete: bool,
) -> dict:
    sample_id = sample_dir.name
    config = cycle_config(sample_dir)
    case_dir = output_root / "cases" / sample_id
    log_path = logs_dir / f"{sample_id}.log"
    case_dir.mkdir(parents=True, exist_ok=True)
    if not rerun_complete and complete_case(case_dir, config):
        return {
            "sample_id": sample_id,
            "status": "skipped_complete",
            "gpu": gpu,
            "resolution": [config["width"], config["height"]],
            "frame_count": config["frame_count"],
            "output_dir": str(case_dir),
            "log": str(log_path),
        }

    command = [
        str(blender), "-b", "--python", str(BLENDER_SCRIPT), "--",
        "--sample-dir", str(sample_dir),
        "--output-dir", str(case_dir),
        "--width", str(config["width"]),
        "--height", str(config["height"]),
        "--samples", str(config["samples"]),
        "--exposure", str(config["exposure"]),
        "--device", "CUDA",
        "--frame-limit", "0",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.write(f"CUDA_VISIBLE_DEVICES={gpu}\n")
        log.flush()
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        return {
            "sample_id": sample_id,
            "status": "failed",
            "gpu": gpu,
            "returncode": result.returncode,
            "elapsed_seconds": elapsed,
            "output_dir": str(case_dir),
            "log": str(log_path),
        }
    if not complete_case(case_dir, config):
        return {
            "sample_id": sample_id,
            "status": "incomplete",
            "gpu": gpu,
            "elapsed_seconds": elapsed,
            "output_dir": str(case_dir),
            "log": str(log_path),
        }
    report = load_json(case_dir / "truth_metadata.json")
    source_video = sample_dir / "videos" / "rgb_cycles.mp4"
    return {
        "sample_id": sample_id,
        "status": "complete",
        "gpu": gpu,
        "resolution": [config["width"], config["height"]],
        "frame_count": config["frame_count"],
        "dynamic_objects": report.get("dynamic_objects", []),
        "elapsed_seconds": elapsed,
        "source_rgb_cycles_sha256": sha256_file(source_video) if source_video.is_file() else None,
        "output_dir": str(case_dir),
        "log": str(log_path),
    }


def sample_dirs(dataset_root: Path, sample_ids: list[str] | None, start: int, limit: int) -> list[Path]:
    all_dirs = sorted(
        path for path in (dataset_root / "samples").iterdir()
        if path.is_dir() and (path / "metadata.json").is_file()
    )
    if sample_ids:
        wanted = set(sample_ids)
        all_dirs = [path for path in all_dirs if path.name in wanted]
        missing = wanted - {path.name for path in all_dirs}
        if missing:
            raise FileNotFoundError(f"unknown sample ids: {sorted(missing)}")
    selected = all_dirs[start:]
    return selected if limit <= 0 else selected[:limit]


def write_manifest(output_root: Path, payload: dict) -> None:
    (output_root / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    blender = args.blender.expanduser().resolve()
    if not blender.is_file():
        raise FileNotFoundError(blender)
    if not BLENDER_SCRIPT.is_file():
        raise FileNotFoundError(BLENDER_SCRIPT)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    if "4" in gpus:
        raise ValueError("GPU 4 is reserved and cannot be used")
    selected = sample_dirs(dataset_root, args.sample_ids, args.start_index, args.limit)
    if not selected:
        raise RuntimeError("no samples selected")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "cases").mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []
    manifest = {
        "schema_version": "physv_cycles_aligned_truth_manifest_v1",
        "status": "running",
        "started_at": started_at,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "blender": str(blender),
        "blender_script": str(BLENDER_SCRIPT),
        "gpus": gpus,
        "selected_sample_count": len(selected),
        "records": records,
        "mapping": {
            "rgb_cycles": "samples/<sample_id>/videos/rgb_cycles.mp4",
            "cycles_config": "samples/<sample_id>/videos/rgb_cycles.json",
            "aligned_dynamic_masks": "cases/<sample_id>/dynamic_masks.npz",
            "aligned_trajectory_pixels": "cases/<sample_id>/trajectory_pixels.npz",
            "case_metadata": "cases/<sample_id>/truth_metadata.json",
            "simulator_trajectory": "samples/<sample_id>/raw/trajectories.npz",
            "simulator_mask": "samples/<sample_id>/raw/masks.npz (not used as the CYCLES pixel mask)",
            "collision_truth": "samples/<sample_id>/raw/contacts.json and physics_supervision.npz",
        },
    }
    write_manifest(output_root, manifest)
    print(
        f"[cycles-truth] selected={len(selected)} gpus={','.join(gpus)} output={output_root}",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(
                run_case,
                sample_dir,
                output_root,
                logs_dir,
                blender,
                gpus[index % len(gpus)],
                args.rerun_complete,
            ): sample_dir.name
            for index, sample_dir in enumerate(selected)
        }
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            manifest["records"] = sorted(records, key=lambda item: item["sample_id"])
            write_manifest(output_root, manifest)
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    manifest["records"] = sorted(records, key=lambda item: item["sample_id"])
    manifest["status"] = (
        "complete"
        if counts.get("failed", 0) == 0 and counts.get("incomplete", 0) == 0
        else "completed_with_errors"
    )
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["status_counts"] = counts
    write_manifest(output_root, manifest)
    print(json.dumps({"status": manifest["status"], "status_counts": counts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
