#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
GENERATE_SCRIPT = SCRIPT_DIR / "generate_sim_dataset.py"
PREPARE_SCRIPT = SCRIPT_DIR / "prepare_sim_episodes.py"

DEFAULT_RAW_OUTPUT_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500"
)
DEFAULT_EPISODE_OUTPUT_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500"
)
DEFAULT_FAMILY_RATIOS = "F1=0.25,F2=0.30,F3=0.20,F4=0.15,F5=0.10"


@dataclass
class ShardSpec:
    shard_id: int
    gpu_id: int
    train_count: int
    val_count: int
    test_count: int
    start_index: int
    seed: int

    @property
    def total_count(self) -> int:
        return self.train_count + self.val_count + self.test_count

    @property
    def slug(self) -> str:
        return f"shard_{self.shard_id:02d}_gpu{self.gpu_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel raw simulation dataset generation with explicit GPU sharding.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RAW_OUTPUT_ROOT)
    parser.add_argument("--episodes-output-root", type=Path, default=DEFAULT_EPISODE_OUTPUT_ROOT)
    parser.add_argument("--theme", default="industrial")
    parser.add_argument("--train-count", type=int, default=1200)
    parser.add_argument("--val-count", type=int, default=150)
    parser.add_argument("--test-count", type=int, default=150)
    parser.add_argument("--family-ratios", default=DEFAULT_FAMILY_RATIOS)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--gpus", default="6,7")
    parser.add_argument("--procs-per-gpu", type=int, default=2)
    parser.add_argument("--python-bin", default="/data/gaoya/miniconda3/envs/wan/bin/python")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-episodes", action="store_true")
    parser.add_argument("--episodes-height", type=int, default=144)
    parser.add_argument("--episodes-width", type=int, default=256)
    parser.add_argument("--context-steps", type=int, default=8)
    parser.add_argument("--future-steps", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--max-objects", type=int, default=6)
    parser.add_argument("--appearance-dim", type=int, default=16)
    return parser.parse_args()


def split_counts(total: int, num_parts: int) -> list[int]:
    base = total // num_parts
    remainder = total % num_parts
    return [base + (1 if idx < remainder else 0) for idx in range(num_parts)]


def build_shards(args: argparse.Namespace) -> list[ShardSpec]:
    gpu_ids = [int(token.strip()) for token in args.gpus.split(",") if token.strip()]
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    num_shards = len(gpu_ids) * args.procs_per_gpu
    train_counts = split_counts(args.train_count, num_shards)
    val_counts = split_counts(args.val_count, num_shards)
    test_counts = split_counts(args.test_count, num_shards)

    shards: list[ShardSpec] = []
    next_index = args.start_index
    for shard_id in range(num_shards):
        gpu_id = gpu_ids[shard_id % len(gpu_ids)]
        shard = ShardSpec(
            shard_id=shard_id,
            gpu_id=gpu_id,
            train_count=train_counts[shard_id],
            val_count=val_counts[shard_id],
            test_count=test_counts[shard_id],
            start_index=next_index,
            seed=args.seed + shard_id * 100_003,
        )
        next_index += shard.total_count
        shards.append(shard)
    return shards


def shard_output_root(base_root: Path, shard: ShardSpec) -> Path:
    return base_root.parent / f"{base_root.name}_shards" / shard.slug


def run_shards(args: argparse.Namespace, shards: list[ShardSpec]) -> None:
    shard_parent = args.output_root.parent / f"{args.output_root.name}_shards"
    if args.overwrite and shard_parent.exists():
        shutil.rmtree(shard_parent)
    shard_parent.mkdir(parents=True, exist_ok=True)

    running: list[tuple[subprocess.Popen[str], ShardSpec, Path]] = []
    pending = list(shards)

    while pending or running:
        launched_any = True
        while pending and launched_any:
            launched_any = False
            gpu_loads = {gpu_id: 0 for gpu_id in {shard.gpu_id for shard in shards}}
            for _, active_shard, _ in running:
                gpu_loads[active_shard.gpu_id] += 1

            still_pending: list[ShardSpec] = []
            for shard in pending:
                if gpu_loads.get(shard.gpu_id, 0) >= args.procs_per_gpu:
                    still_pending.append(shard)
                    continue

                shard_root = shard_output_root(args.output_root, shard)
                shard_root.parent.mkdir(parents=True, exist_ok=True)
                log_path = shard_root.parent / f"{shard.slug}.log"
                env = os.environ.copy()
                env["PYOPENGL_PLATFORM"] = "egl"
                env["EGL_DEVICE_ID"] = str(shard.gpu_id)
                cmd = [
                    args.python_bin,
                    str(GENERATE_SCRIPT),
                    "--output-root",
                    str(shard_root),
                    "--theme",
                    args.theme,
                    "--train-count",
                    str(shard.train_count),
                    "--val-count",
                    str(shard.val_count),
                    "--test-count",
                    str(shard.test_count),
                    "--family-ratios",
                    args.family_ratios,
                    "--seed",
                    str(shard.seed),
                    "--start-index",
                    str(shard.start_index),
                ]
                if args.overwrite:
                    cmd.append("--overwrite")
                log_fp = log_path.open("w", encoding="utf-8")
                try:
                    proc = subprocess.Popen(
                        cmd,
                        env=env,
                        stdout=log_fp,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                finally:
                    log_fp.close()
                running.append((proc, shard, log_path))
                gpu_loads[shard.gpu_id] += 1
                launched_any = True
                print(
                    f"[launch] {shard.slug} gpu={shard.gpu_id} "
                    f"counts=({shard.train_count},{shard.val_count},{shard.test_count}) "
                    f"start_index={shard.start_index} pid={proc.pid}",
                    flush=True,
                )
            pending = still_pending

        time.sleep(5.0)
        next_running: list[tuple[subprocess.Popen[str], ShardSpec, Path]] = []
        for proc, shard, log_path in running:
            ret = proc.poll()
            if ret is None:
                next_running.append((proc, shard, log_path))
                continue
            if ret != 0:
                raise RuntimeError(f"shard {shard.slug} failed with code {ret}; inspect {log_path}")
            print(f"[done] {shard.slug} gpu={shard.gpu_id} log={log_path}", flush=True)
        running = next_running


def sample_key(record: dict) -> tuple[int, str]:
    sample_id = str(record.get("sample_id", "sample_000000"))
    try:
        numeric = int(sample_id.split("_")[-1])
    except Exception:
        numeric = sys.maxsize
    return numeric, sample_id


def merge_shards(args: argparse.Namespace, shards: list[ShardSpec]) -> None:
    final_root = args.output_root
    if args.overwrite and final_root.exists():
        shutil.rmtree(final_root)
    final_root.mkdir(parents=True, exist_ok=True)
    (final_root / "manifests").mkdir(parents=True, exist_ok=True)

    split_records: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for shard in shards:
        shard_root = shard_output_root(args.output_root, shard)
        for split in ("train", "val", "test"):
            src_split = shard_root / split
            if src_split.exists():
                for family_dir in sorted(path for path in src_split.iterdir() if path.is_dir()):
                    dst_family = final_root / split / family_dir.name
                    dst_family.mkdir(parents=True, exist_ok=True)
                    for sample_dir in sorted(path for path in family_dir.iterdir() if path.is_dir()):
                        dst_sample = dst_family / sample_dir.name
                        if dst_sample.exists():
                            raise FileExistsError(f"duplicate sample while merging: {dst_sample}")
                        shutil.move(str(sample_dir), str(dst_sample))
            split_manifest_path = shard_root / "manifests" / f"{split}.json"
            if split_manifest_path.exists():
                payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
                split_records[split].extend(payload.get("records", []))

    for split, records in split_records.items():
        records.sort(key=sample_key)
        split_counts[split] = len(records)
        payload = {
            "split": split,
            "count": len(records),
            "theme": args.theme,
            "records": records,
        }
        (final_root / "manifests" / f"{split}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "theme": args.theme,
        "output_root": str(final_root),
        "splits": split_counts,
        "family_ratios": args.family_ratios,
        "total_records": sum(split_counts.values()),
        "start_index": args.start_index,
        "next_index": args.start_index + sum(split_counts.values()),
        "gpus": [int(token.strip()) for token in args.gpus.split(",") if token.strip()],
        "procs_per_gpu": args.procs_per_gpu,
        "num_shards": len(shards),
        "shards": [
            {
                "shard_id": shard.shard_id,
                "gpu_id": shard.gpu_id,
                "slug": shard.slug,
                "train_count": shard.train_count,
                "val_count": shard.val_count,
                "test_count": shard.test_count,
                "start_index": shard.start_index,
                "seed": shard.seed,
            }
            for shard in shards
        ],
    }
    (final_root / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def prepare_episodes(args: argparse.Namespace) -> None:
    if args.overwrite and args.episodes_output_root.exists():
        shutil.rmtree(args.episodes_output_root)
    cmd = [
        args.python_bin,
        str(PREPARE_SCRIPT),
        "--input-root",
        str(args.output_root),
        "--output-root",
        str(args.episodes_output_root),
        "--height",
        str(args.episodes_height),
        "--width",
        str(args.episodes_width),
        "--context-steps",
        str(args.context_steps),
        "--future-steps",
        str(args.future_steps),
        "--frame-stride",
        str(args.frame_stride),
        "--window-stride",
        str(args.window_stride),
        "--max-objects",
        str(args.max_objects),
        "--appearance-dim",
        str(args.appearance_dim),
        "--overwrite",
    ]
    print("[episodes] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    shards = build_shards(args)
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "episodes_output_root": str(args.episodes_output_root),
                "gpus": args.gpus,
                "procs_per_gpu": args.procs_per_gpu,
                "total_requested": args.train_count + args.val_count + args.test_count,
                "num_shards": len(shards),
                "shards": [shard.__dict__ for shard in shards],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    run_shards(args, shards)
    merge_shards(args, shards)
    if args.prepare_episodes:
        prepare_episodes(args)


if __name__ == "__main__":
    main()
