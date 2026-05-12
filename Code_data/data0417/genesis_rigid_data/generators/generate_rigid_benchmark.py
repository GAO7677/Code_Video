#!/usr/bin/env python3
# 用途：统一生成与修复 rigid benchmark 和 held-out 集。
"""Unified rigid benchmark entry point.

该脚本用于统一生成 Genesis 刚体 benchmark 数据；输入为 PhysXNet 资产、已有训练集根目录或 heldout 配置，输出为 output_root 下的 benchmark 样本、manifest、caption 和预处理子集。

This script consolidates the previous benchmark drivers into one CLI with
subcommands:

- ``physxnet_pool``: build a flat PhysXNet rigid benchmark under
  ``train/rigid/<scene>/<count_bucket>/<sample>``.
- ``stage1_heldout``: build a held-out single-object motion benchmark and,
  optionally, Stage-1 window subsets.
- ``benchmark_v1``: build the benchmark_v1 dev/test subsets derived from
  ``try1_physxnet_benchmark.py``.

The underlying large Genesis/PhysXNet generators are intentionally reused
instead of duplicated here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from generators import try1_physxnet_benchmark as try1
from repair import filter_single_object_motion_cases as motion_qa

DEFAULT_ARTICULATION_SCRIPT = SCRIPT_DIR / "try1_physxnet_articulation_mpm0417.py"
DEFAULT_STAGE1_BUILD_SCRIPT = SCRIPT_DIR.parent.parent.parent / "Code_train" / "train_0419" / "state_adapter" / "build_stage1_subsets.py"
DEFAULT_CAPTION_SCRIPT = SCRIPT_DIR / "generate_video_captions.py"
DEFAULT_PHYSX_ROOT = Path("/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet")
DEFAULT_BENCHMARK_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data_benchmark")
DEFAULT_STAGE1_TRAIN_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases")


BENCHMARK_V1_NAME = "benchmark_v1"
BENCHMARK_V1_TRACK = "rigid"
BENCHMARK_V1_TOTAL_FRAMES = 49
BENCHMARK_V1_CONTEXT_FRAMES = 8
BENCHMARK_V1_PRED_FRAMES = 41
BENCHMARK_V1_FUTURE_START = BENCHMARK_V1_CONTEXT_FRAMES
BENCHMARK_V1_DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data/benchmark_v1")
BENCHMARK_V1_DEFAULT_TRAIN_MANIFEST = Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data/try3_rigid0417/dataset_manifest.json")
BENCHMARK_V1_HELDOUT_POOL_NAME = "benchmark_v1_reserved_seedspace_try1"
BENCHMARK_V1_SEED_BASE = 1_900_000
BENCHMARK_V1_SUBSETS = [
    "smooth_single",
    "smooth_multi_no_contact",
    "event_pair",
    "event_multi",
    "ood_physical",
    "ood_composition",
]
INVALID_SAMPLE_MARKERS = {"invalid_case900_901", "invalid_by_qa", "_qa_invalid"}
SAMPLE_META_FILENAMES = ("meta.json", "metadata.json")


def find_sample_meta_path(sample_dir: Path) -> Optional[Path]:
    for filename in SAMPLE_META_FILENAMES:
        candidate = sample_dir / filename
        if candidate.exists():
            return candidate
    return None


def iter_sample_meta_paths(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    seen: set[Path] = set()
    for filename in SAMPLE_META_FILENAMES:
        for path in root.rglob(filename):
            sample_dir = path.parent
            if sample_dir in seen:
                continue
            seen.add(sample_dir)
            yield path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Genesis rigid benchmark generator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pool = subparsers.add_parser(
        "physxnet_pool",
        help="Build a PhysXNet rigid benchmark pool under train/rigid/<scene>/<count_bucket>/<sample>.",
    )
    add_physxnet_pool_args(pool)

    stage1 = subparsers.add_parser(
        "stage1_heldout",
        help="Build a held-out single-object motion benchmark and optional Stage-1 subsets.",
    )
    add_stage1_heldout_args(stage1)

    v1 = subparsers.add_parser(
        "benchmark_v1",
        help="Build benchmark_v1 dev/test subsets from try1_physxnet_benchmark.py.",
    )
    add_benchmark_v1_args(v1)

    qa_existing = subparsers.add_parser(
        "qa_existing",
        help="Scan existing benchmark roots and apply the same rigid motion QA to all discovered samples.",
    )
    add_qa_existing_args(qa_existing)

    return parser


def add_physxnet_pool_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--script", type=Path, default=DEFAULT_ARTICULATION_SCRIPT)
    parser.add_argument("--physx_root", type=Path, default=DEFAULT_PHYSX_ROOT)
    parser.add_argument("--version", type=str, default="version_1")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument(
        "--work_root",
        type=Path,
        default=None,
        help="Temporary/cache root. Default: <output_root>_work_cache.",
    )
    parser.add_argument("--num_objects", type=int, default=50)
    parser.add_argument(
        "--object_ids",
        type=str,
        nargs="*",
        default=None,
        help="Explicit PhysXNet object ids to run before random sampling.",
    )
    parser.add_argument("--random_seed", type=int, default=20260420)
    parser.add_argument("--cases_per_object", type=int, default=8)
    parser.add_argument(
        "--case_pool",
        type=int,
        nargs="*",
        default=list(range(9)),
        help="Candidate case indices to sample from. Default samples from 0..8.",
    )
    parser.add_argument("--striker_speed_min", type=float, default=2.0)
    parser.add_argument("--striker_speed_max", type=float, default=4.8)
    parser.add_argument("--entry_speed_min", type=float, default=0.6)
    parser.add_argument("--entry_speed_max", type=float, default=2.2)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.003)
    parser.add_argument("--substeps", type=int, default=40)
    parser.add_argument("--ball_posx", type=float, default=0.03)
    parser.add_argument(
        "--rigid_target_object_count",
        type=int,
        default=3,
        help="Total rigid objects in each scene: main PhysXNet object + striker + aux objects.",
    )
    parser.add_argument(
        "--physxnet_volume_threshold_m3",
        type=float,
        default=999999.0,
        help="Large default keeps entry-motion templates available for benchmark coverage.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep_work", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    add_visibility_filter_args(parser)
    add_motion_qa_args(parser)
    add_caption_args(parser)


def add_stage1_heldout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--script", type=Path, default=DEFAULT_ARTICULATION_SCRIPT)
    parser.add_argument("--build_script", type=Path, default=DEFAULT_STAGE1_BUILD_SCRIPT)
    parser.add_argument("--physx_root", type=Path, default=DEFAULT_PHYSX_ROOT)
    parser.add_argument("--version", type=str, default="version_1")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_BENCHMARK_ROOT / "stage1_count01_benchmark")
    parser.add_argument("--stage1_train_root", "--stage1-train-root", dest="stage1_train_root", type=Path, default=DEFAULT_STAGE1_TRAIN_ROOT)
    parser.add_argument("--heldout_seed", "--heldout-seed", dest="heldout_seed", type=int, default=20260421)
    parser.add_argument("--heldout_count", "--heldout-count", dest="heldout_count", type=int, default=8)
    parser.add_argument("--heldout_ids", "--heldout-ids", dest="heldout_ids", type=str, nargs="*", default=None)
    parser.add_argument("--num_random_cases", "--num-random-cases", dest="num_random_cases", type=int, default=12)
    parser.add_argument("--case_index_filter", type=int, nargs="*", default=[900, 901])
    parser.add_argument("--dt", type=float, default=0.003)
    parser.add_argument("--substeps", type=int, default=40)
    parser.add_argument("--steps", type=int, default=49)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--rigid_target_object_count", type=int, default=1)
    parser.add_argument("--physxnet_volume_threshold_m3", type=float, default=999999.0)
    parser.add_argument("--motion_case_max_retries", type=int, default=8)
    parser.add_argument("--skip_subset_build", "--skip-subset-build", dest="skip_subset_build", action="store_true")
    parser.add_argument("--max_source_samples", "--max-source-samples", dest="max_source_samples", type=int, default=0)
    parser.add_argument("--max_windows_per_subset", "--max-windows-per-subset", dest="max_windows_per_subset", type=int, default=100)
    parser.add_argument(
        "--future_main_visibility_threshold",
        "--future-main-visibility-threshold",
        dest="future_main_visibility_threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument("--subset_count_buckets", "--subset-count-buckets", dest="subset_count_buckets", type=str, default="count_01")
    parser.add_argument("--dry_run", action="store_true")
    add_visibility_filter_args(parser)
    add_motion_qa_args(parser)
    add_caption_args(parser)


def add_benchmark_v1_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_root", "--dataset-root", dest="dataset_root", type=Path, default=BENCHMARK_V1_DEFAULT_DATASET_ROOT)
    parser.add_argument("--train_manifest", "--train-manifest", dest="train_manifest", type=Path, default=BENCHMARK_V1_DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--physx_root", "--physx-root", dest="physx_root", type=str, default=str(DEFAULT_PHYSX_ROOT))
    parser.add_argument("--version", type=str, default="version_1")
    parser.add_argument("--dev_per_subset", "--dev-per-subset", dest="dev_per_subset", type=int, default=1)
    parser.add_argument("--test_per_subset", "--test-per-subset", dest="test_per_subset", type=int, default=1)
    parser.add_argument("--subsets", nargs="*", default=BENCHMARK_V1_SUBSETS)
    parser.add_argument("--steps", type=int, default=49)
    parser.add_argument("--dt", type=float, default=0.003)
    parser.add_argument("--substeps", type=int, default=40)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--striker_speed", "--striker-speed", dest="striker_speed", type=float, default=2.8)
    parser.add_argument("--striker_radius", "--striker-radius", dest="striker_radius", type=float, default=0.08)
    parser.add_argument("--camera_distance_mult", "--camera-distance-mult", dest="camera_distance_mult", type=float, default=1.0)
    parser.add_argument(
        "--max_attempts_per_sample",
        "--max-attempts-per-sample",
        dest="max_attempts_per_sample",
        type=int,
        default=20,
    )
    parser.add_argument("--object_pool_size", "--object-pool-size", dest="object_pool_size", type=int, default=64)
    parser.add_argument("--object_ids", "--object-ids", dest="object_ids", nargs="*", default=None)
    parser.add_argument("--resolution", type=int, nargs=2, default=[960, 720])
    add_visibility_filter_args(parser)
    add_motion_qa_args(parser)
    add_caption_args(parser)


def add_qa_existing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=[DEFAULT_BENCHMARK_ROOT],
        help="Existing benchmark root(s) to scan, e.g. /data/.../0417data_benchmark or a specific benchmark folder.",
    )
    parser.add_argument("--dry_run", action="store_true")
    add_visibility_filter_args(parser)
    add_motion_qa_args(parser)


def add_motion_qa_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--disable_motion_qa",
        "--disable-motion-qa",
        dest="disable_motion_qa",
        action="store_true",
        help="Skip the post-generation / explicit rigid motion QA scan.",
    )
    parser.add_argument(
        "--qa_margin_px",
        "--qa-margin-px",
        dest="qa_margin_px",
        type=float,
        default=24.0,
        help="Pixel margin used by rigid motion QA when checking projected center safety.",
    )
    parser.add_argument(
        "--qa_include_invalid",
        "--qa-include-invalid",
        dest="qa_include_invalid",
        action="store_true",
        help="Also rescan samples already under invalid_case900_901, invalid_by_qa, or _qa_invalid.",
    )
    parser.add_argument(
        "--qa_quarantine_root",
        "--qa-quarantine-root",
        dest="qa_quarantine_root",
        type=Path,
        default=None,
        help="Optional quarantine root. Default: <scanned_root>/_qa_invalid.",
    )
    parser.add_argument(
        "--qa_report_path",
        "--qa-report-path",
        dest="qa_report_path",
        type=Path,
        default=None,
        help="Optional JSON report path. For a single scanned root, defaults to <root>/qa_report.json.",
    )


def add_visibility_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max_no_object_ratio",
        "--max-no-object-ratio",
        dest="max_no_object_ratio",
        type=float,
        default=1.0 / 3.0,
        help="Reject samples where no object is visible for more than this fraction of frames.",
    )
    parser.add_argument(
        "--disable_visibility_filter",
        "--disable-visibility-filter",
        dest="disable_visibility_filter",
        action="store_true",
        help="Disable the no-object visibility quality filter.",
    )


def add_caption_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--generate_captions",
        "--generate-captions",
        dest="generate_captions",
        action="store_true",
        help="After dataset generation, run generate_video_captions.py on the generated root.",
    )
    parser.add_argument(
        "--caption_script",
        "--caption-script",
        dest="caption_script",
        type=Path,
        default=DEFAULT_CAPTION_SCRIPT,
        help="Caption generation script to call after benchmark generation.",
    )
    parser.add_argument(
        "--caption_roots",
        "--caption-roots",
        dest="caption_roots",
        type=Path,
        nargs="*",
        default=None,
        help="Optional roots passed to generate_video_captions.py. Defaults to the generated dataset root.",
    )
    parser.add_argument(
        "--caption_manifest",
        "--caption-manifest",
        dest="caption_manifest",
        type=Path,
        default=None,
        help="Optional jsonl manifest path for generated captions.",
    )
    parser.add_argument(
        "--caption_include_invalid",
        "--caption-include-invalid",
        dest="caption_include_invalid",
        action="store_true",
        help="Also caption samples under invalid_case900_901, invalid_by_qa, or _qa_invalid.",
    )
    parser.add_argument(
        "--caption_overwrite",
        "--caption-overwrite",
        dest="caption_overwrite",
        action="store_true",
        help="Overwrite existing caption files instead of only filling missing ones.",
    )


def make_json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): make_json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [make_json_safe(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def list_physxnet_object_ids(physx_root: Path, version: str) -> List[str]:
    finaljson_dir = physx_root / version / "finaljson"
    if not finaljson_dir.exists():
        raise FileNotFoundError(f"Cannot find finaljson dir: {finaljson_dir}")
    object_ids = sorted(path.stem for path in finaljson_dir.glob("*.json"))
    if not object_ids:
        raise RuntimeError(f"No PhysXNet json files under {finaljson_dir}")
    return object_ids


def sample_candidates(object_ids: List[str], seed: int) -> List[str]:
    rng = random.Random(int(seed))
    candidates = list(object_ids)
    rng.shuffle(candidates)
    return candidates


def sample_case_indices(case_pool: Sequence[int], count: int, rng: random.Random) -> List[int]:
    pool = [int(idx) for idx in case_pool]
    if not pool:
        raise ValueError("--case_pool cannot be empty")
    if count <= len(pool):
        return rng.sample(pool, int(count))
    selected = rng.sample(pool, len(pool))
    selected.extend(rng.choice(pool) for _ in range(int(count) - len(pool)))
    return selected


def sample_speed_config(args: argparse.Namespace, rng: random.Random) -> Dict[str, float]:
    striker_min = float(args.striker_speed_min)
    striker_max = max(striker_min, float(args.striker_speed_max))
    entry_min = float(args.entry_speed_min)
    entry_max = max(entry_min, float(args.entry_speed_max))
    return {
        "striker_speed": float(rng.uniform(striker_min, striker_max)),
        "entry_speed_min": float(rng.uniform(entry_min, 0.5 * (entry_min + entry_max))),
        "entry_speed_max": float(rng.uniform(0.5 * (entry_min + entry_max), entry_max)),
    }


def iter_sample_dirs(train_root: Path) -> Iterable[Path]:
    if not train_root.exists():
        return []
    return sorted(path.parent for path in iter_sample_meta_paths(train_root))


def is_invalid_sample_path(path: Path) -> bool:
    return any(part in INVALID_SAMPLE_MARKERS for part in path.parts)


def iter_qa_sample_dirs(root: Path, *, include_invalid: bool = False) -> Iterable[Path]:
    if not root.exists():
        return []
    seen: set[Path] = set()
    for metadata_path in iter_sample_meta_paths(root):
        sample_dir = metadata_path.parent
        if sample_dir in seen:
            continue
        if not include_invalid and is_invalid_sample_path(sample_dir):
            continue
        if not (sample_dir / "physics" / "rigid_kinematics.npz").exists():
            continue
        if not (sample_dir / "physics" / "anchor_targets.npz").exists():
            continue
        seen.add(sample_dir)
        yield sample_dir


def compute_sample_visibility_stats(sample_dir: Path) -> Dict[str, Any]:
    anchor_path = sample_dir / "physics" / "anchor_targets.npz"
    if not anchor_path.exists():
        return {
            "available": False,
            "reason": "missing_anchor_targets",
            "sample_dir": str(sample_dir),
        }
    try:
        data = np.load(anchor_path)
        if "visibility_mask" not in data:
            return {
                "available": False,
                "reason": "missing_visibility_mask",
                "sample_dir": str(sample_dir),
            }
        visibility_mask = np.asarray(data["visibility_mask"])
    except Exception as exc:
        return {
            "available": False,
            "reason": f"load_failed:{type(exc).__name__}",
            "sample_dir": str(sample_dir),
        }

    if visibility_mask.ndim == 1:
        any_visible = visibility_mask.astype(bool)
    elif visibility_mask.ndim >= 2:
        any_visible = np.any(visibility_mask.astype(bool), axis=1)
    else:
        return {
            "available": False,
            "reason": f"invalid_visibility_rank:{visibility_mask.ndim}",
            "sample_dir": str(sample_dir),
        }

    total_frames = int(any_visible.shape[0])
    if total_frames <= 0:
        return {
            "available": False,
            "reason": "empty_visibility_mask",
            "sample_dir": str(sample_dir),
        }

    no_object_frames = int(np.count_nonzero(~any_visible))
    no_object_ratio = float(no_object_frames / max(total_frames, 1))
    visible_frames = int(total_frames - no_object_frames)
    return {
        "available": True,
        "sample_dir": str(sample_dir),
        "total_frames": total_frames,
        "visible_frames": visible_frames,
        "no_object_frames": no_object_frames,
        "no_object_ratio": no_object_ratio,
    }


def check_sample_visibility_filter(sample_dir: Path, args: argparse.Namespace) -> Tuple[bool, Dict[str, Any]]:
    stats = compute_sample_visibility_stats(sample_dir)
    if bool(getattr(args, "disable_visibility_filter", False)):
        stats["filter_enabled"] = False
        return True, stats
    stats["filter_enabled"] = True
    stats["max_no_object_ratio"] = float(getattr(args, "max_no_object_ratio", 1.0 / 3.0))
    if not bool(stats.get("available", False)):
        return False, stats
    passed = float(stats["no_object_ratio"]) <= float(stats["max_no_object_ratio"])
    return passed, stats


def write_rgb_first8(sample_dir: Path, fps: int) -> bool:
    import imageio.v2 as imageio

    out_path = sample_dir / "videos" / "rgb_first8.mp4"
    rgb_path = sample_dir / "videos" / "rgb.mp4"
    if out_path.exists():
        return True
    if not rgb_path.exists():
        return False

    frames = []
    reader = imageio.get_reader(rgb_path)
    try:
        for idx, frame in enumerate(reader):
            if idx >= 8:
                break
            frames.append(frame)
    finally:
        reader.close()

    if not frames:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=int(fps), quality=8)
    return True


def merge_generated_samples(
    *,
    args: argparse.Namespace,
    tmp_output_root: Path,
    final_output_root: Path,
    object_id: str,
    fps: int,
    cases_per_object: int,
    overwrite: bool,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    copied: List[str] = []
    filtered: List[Dict[str, Any]] = []
    for src_sample in iter_sample_dirs(tmp_output_root / "train"):
        try:
            meta_path = find_sample_meta_path(src_sample)
            if meta_path is None:
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(meta.get("object_id", "")) != str(object_id):
            continue

        passed_visibility, visibility_stats = check_sample_visibility_filter(src_sample, args)
        if not passed_visibility:
            filtered.append(
                {
                    "object_id": str(object_id),
                    "sample_dir": str(src_sample),
                    "reason": "too_many_no_object_frames",
                    "visibility": visibility_stats,
                }
            )
            print(
                f"[filter] sample={src_sample} "
                f"no_object_ratio={visibility_stats.get('no_object_ratio')} "
                f"threshold={visibility_stats.get('max_no_object_ratio')}"
            )
            continue

        rel = src_sample.relative_to(tmp_output_root)
        dst_sample = final_output_root / rel
        if dst_sample.exists():
            if overwrite:
                shutil.rmtree(dst_sample)
            else:
                passed_existing, existing_visibility = check_sample_visibility_filter(dst_sample, args)
                if not passed_existing:
                    filtered.append(
                        {
                            "object_id": str(object_id),
                            "sample_dir": str(dst_sample),
                            "reason": "existing_sample_failed_visibility_filter",
                            "visibility": existing_visibility,
                        }
                    )
                    print(
                        f"[filter] existing_sample={dst_sample} "
                        f"no_object_ratio={existing_visibility.get('no_object_ratio')} "
                        f"threshold={existing_visibility.get('max_no_object_ratio')}"
                    )
                    shutil.rmtree(dst_sample, ignore_errors=True)
                else:
                    write_rgb_first8(dst_sample, fps=fps)
                    copied.append(str(dst_sample))
                    if len(copied) >= cases_per_object:
                        break
                    continue

        dst_sample.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_sample, dst_sample)
        write_rgb_first8(dst_sample, fps=fps)
        copied.append(str(dst_sample))
        if len(copied) >= cases_per_object:
            break
    return copied, filtered


def filter_existing_samples_for_object(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    object_id: str,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    kept: List[str] = []
    filtered: List[Dict[str, Any]] = []
    train_root = dataset_root / "train"
    for sample_dir in iter_sample_dirs(train_root):
        try:
            meta_path = find_sample_meta_path(sample_dir)
            if meta_path is None:
                continue
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(metadata.get("object_id", "")) != str(object_id):
            continue
        passed_visibility, visibility_stats = check_sample_visibility_filter(sample_dir, args)
        if passed_visibility:
            kept.append(str(sample_dir))
            continue
        filtered.append(
            {
                "object_id": str(object_id),
                "sample_dir": str(sample_dir),
                "reason": "too_many_no_object_frames",
                "visibility": visibility_stats,
            }
        )
        print(
            f"[filter] sample={sample_dir} "
            f"no_object_ratio={visibility_stats.get('no_object_ratio')} "
            f"threshold={visibility_stats.get('max_no_object_ratio')}"
        )
        shutil.rmtree(sample_dir, ignore_errors=True)
    return kept, filtered


def run_subprocess(cmd: Sequence[str], *, dry_run: bool = False) -> None:
    print(" ".join(str(part) for part in cmd))
    if not dry_run:
        subprocess.run(list(cmd), check=True)


def evaluate_benchmark_sample_qa(sample_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    motion_metrics = motion_qa.evaluate_sample(sample_dir, margin=float(getattr(args, "qa_margin_px", 24.0)))
    motion_valid = bool(motion_metrics.get("valid", False))
    visibility_valid, visibility_stats = check_sample_visibility_filter(sample_dir, args)
    combined_reasons = list(motion_metrics.get("reasons", []))
    if not visibility_valid:
        combined_reasons.append("too_many_no_object_frames")

    merged = dict(motion_metrics)
    merged["motion_qa_valid"] = motion_valid
    merged["motion_qa_reasons"] = list(motion_metrics.get("reasons", []))
    merged["visibility_filter_valid"] = bool(visibility_valid)
    merged["visibility_filter"] = make_json_safe(visibility_stats)
    merged["valid"] = bool(motion_valid and visibility_valid)
    merged["reasons"] = combined_reasons
    merged["qa_version"] = "benchmark_motion_visibility_v1"
    return merged


def write_sample_qa_metrics(sample_dir: Path, record: Dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        return
    (sample_dir / "qa_metrics.json").write_text(
        json.dumps(make_json_safe(record), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def default_qa_quarantine_root(root: Path, args: argparse.Namespace) -> Path:
    custom_root = getattr(args, "qa_quarantine_root", None)
    if custom_root is not None:
        return Path(custom_root)
    return root / "_qa_invalid"


def quarantine_sample_dir(sample_dir: Path, *, scan_root: Path, quarantine_root: Path, dry_run: bool) -> Path:
    try:
        rel = sample_dir.relative_to(scan_root)
    except ValueError:
        rel = Path(sample_dir.name)
    dst = quarantine_root / rel
    print(f"[qa] quarantine {sample_dir} -> {dst}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        shutil.move(str(sample_dir), str(dst))
    return dst


def run_motion_qa_scan(args: argparse.Namespace, root: Path) -> Dict[str, Any]:
    root = Path(root)
    report: Dict[str, Any] = {
        "root": str(root),
        "qa_enabled": not bool(getattr(args, "disable_motion_qa", False)),
        "visibility_filter_enabled": not bool(getattr(args, "disable_visibility_filter", False)),
        "max_no_object_ratio": float(getattr(args, "max_no_object_ratio", 1.0 / 3.0)),
        "qa_margin_px": float(getattr(args, "qa_margin_px", 24.0)),
        "include_invalid": bool(getattr(args, "qa_include_invalid", False)),
        "quarantine_root": str(default_qa_quarantine_root(root, args)),
        "samples_total": 0,
        "samples_valid": 0,
        "samples_invalid": 0,
        "invalid_samples": [],
    }
    if bool(getattr(args, "disable_motion_qa", False)):
        return report

    sample_dirs = list(iter_qa_sample_dirs(root, include_invalid=bool(getattr(args, "qa_include_invalid", False))))
    report["samples_total"] = int(len(sample_dirs))
    quarantine_root = default_qa_quarantine_root(root, args)
    dry_run = bool(getattr(args, "dry_run", False))

    for sample_dir in sample_dirs:
        qa_record = evaluate_benchmark_sample_qa(sample_dir, args)
        write_sample_qa_metrics(sample_dir, qa_record, dry_run=dry_run)
        if bool(qa_record.get("valid", False)):
            report["samples_valid"] = int(report["samples_valid"]) + 1
            continue
        report["samples_invalid"] = int(report["samples_invalid"]) + 1
        invalid_entry = {
            "sample_dir": str(sample_dir),
            "reasons": list(qa_record.get("reasons", [])),
            "motion_qa_valid": bool(qa_record.get("motion_qa_valid", False)),
            "visibility_filter_valid": bool(qa_record.get("visibility_filter_valid", False)),
        }
        if not is_invalid_sample_path(sample_dir):
            dst = quarantine_sample_dir(sample_dir, scan_root=root, quarantine_root=quarantine_root, dry_run=dry_run)
            invalid_entry["quarantined_to"] = str(dst)
        report["invalid_samples"].append(invalid_entry)

    return report


def persist_motion_qa_reports(args: argparse.Namespace, reports: Sequence[Dict[str, Any]]) -> None:
    report_path = getattr(args, "qa_report_path", None)
    payload = {
        "qa_version": "benchmark_motion_visibility_v1",
        "num_roots": int(len(reports)),
        "reports": [make_json_safe(report) for report in reports],
    }
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if len(reports) == 1:
        root = Path(str(reports[0]["root"]))
        (root / "qa_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    for report in reports:
        root = Path(str(report["root"]))
        single_payload = {
            "qa_version": "benchmark_motion_visibility_v1",
            "num_roots": 1,
            "reports": [make_json_safe(report)],
        }
        (root / "qa_report.json").write_text(json.dumps(single_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def maybe_run_motion_qa(args: argparse.Namespace, roots: Sequence[Path]) -> List[Dict[str, Any]]:
    if bool(getattr(args, "disable_motion_qa", False)):
        return []
    reports: List[Dict[str, Any]] = []
    for root in roots:
        print(f"[post] motion QA scan root={root}")
        report = run_motion_qa_scan(args, Path(root))
        print(
            f"[post] motion QA done root={root} "
            f"total={report['samples_total']} valid={report['samples_valid']} invalid={report['samples_invalid']}"
        )
        reports.append(report)
    persist_motion_qa_reports(args, reports)
    return reports


def maybe_generate_captions(args: argparse.Namespace, default_root: Path) -> None:
    if not bool(getattr(args, "generate_captions", False)):
        return

    roots = getattr(args, "caption_roots", None)
    caption_roots = [Path(root) for root in roots] if roots else [Path(default_root)]
    cmd = [
        sys.executable,
        str(args.caption_script),
        "--roots",
        *[str(root) for root in caption_roots],
    ]
    if bool(getattr(args, "caption_include_invalid", False)):
        cmd.append("--include_invalid")
    if bool(getattr(args, "caption_overwrite", False)):
        cmd.append("--overwrite")
    if getattr(args, "caption_manifest", None) is not None:
        cmd.extend(["--manifest", str(args.caption_manifest)])

    print("[post] generating captions")
    run_subprocess(cmd, dry_run=bool(getattr(args, "dry_run", False)))


def run_physxnet_pool_one_object(
    args: argparse.Namespace,
    object_id: str,
    work_root: Path,
    case_indices: Sequence[int],
    speed_cfg: Dict[str, float],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    tmp_output_root = work_root / str(object_id)
    if tmp_output_root.exists() and args.overwrite:
        shutil.rmtree(tmp_output_root)
    tmp_output_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(args.script),
        "--physx_root",
        str(args.physx_root),
        "--version",
        str(args.version),
        "--object_id",
        str(object_id),
        "--output_root",
        str(tmp_output_root),
        "--run_genesis",
        "--num_random_cases",
        str(args.cases_per_object),
        "--case_scene_mode",
        "diverse",
        "--case_index_filter",
        *[str(idx) for idx in case_indices],
        "--prefer_existing_runtime_meshes",
        "--dt",
        str(args.dt),
        "--substeps",
        str(args.substeps),
        "--ball_posx",
        str(args.ball_posx),
        "--steps",
        str(args.steps),
        "--fps",
        str(args.fps),
        "--striker_speed",
        str(speed_cfg["striker_speed"]),
        "--physxnet_entry_speed_min",
        str(speed_cfg["entry_speed_min"]),
        "--physxnet_entry_speed_max",
        str(speed_cfg["entry_speed_max"]),
        "--simulator_mode",
        "rigid",
        "--rigid_target_object_count",
        str(args.rigid_target_object_count),
        "--physxnet_volume_threshold_m3",
        str(args.physxnet_volume_threshold_m3),
    ]
    if int(args.rigid_target_object_count) <= 1:
        cmd.append("--disable_striker")

    print(f"[run] object_id={object_id}")
    run_subprocess(cmd, dry_run=bool(args.dry_run))

    copied, filtered = ([], []) if args.dry_run else merge_generated_samples(
        args=args,
        tmp_output_root=tmp_output_root,
        final_output_root=args.output_root,
        object_id=str(object_id),
        fps=int(args.fps),
        cases_per_object=int(args.cases_per_object),
        overwrite=bool(args.overwrite),
    )
    if not args.keep_work and not args.dry_run:
        shutil.rmtree(tmp_output_root, ignore_errors=True)
    return copied, filtered


def cmd_physxnet_pool(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    work_root = args.work_root
    if work_root is None:
        work_root = args.output_root.parent / f"{args.output_root.name}_work_cache"
    work_root.mkdir(parents=True, exist_ok=True)

    object_ids = list_physxnet_object_ids(args.physx_root, args.version)
    if args.object_ids:
        valid = set(object_ids)
        explicit = [str(obj_id) for obj_id in args.object_ids if str(obj_id) in valid]
        missing = [str(obj_id) for obj_id in args.object_ids if str(obj_id) not in valid]
        if missing:
            print(f"[warn] ignoring missing object ids: {missing}")
        random_tail = [obj_id for obj_id in sample_candidates(object_ids, args.random_seed) if obj_id not in set(explicit)]
        candidates = explicit + random_tail
    else:
        candidates = sample_candidates(object_ids, args.random_seed)

    manifest: Dict[str, Any] = {
        "generator": "generate_rigid_benchmark.py::physxnet_pool",
        "physx_root": str(args.physx_root),
        "version": str(args.version),
        "output_root": str(args.output_root),
        "work_root": str(work_root),
        "random_seed": int(args.random_seed),
        "target_num_objects": int(args.num_objects),
        "cases_per_object": int(args.cases_per_object),
        "case_pool": [int(idx) for idx in args.case_pool],
        "speed_ranges": {
            "striker_speed": [float(args.striker_speed_min), float(args.striker_speed_max)],
            "entry_speed": [float(args.entry_speed_min), float(args.entry_speed_max)],
        },
        "visibility_filter": {
            "enabled": not bool(args.disable_visibility_filter),
            "max_no_object_ratio": float(args.max_no_object_ratio),
        },
        "objects": [],
        "skipped": [],
    }

    completed = 0
    for candidate_idx, object_id in enumerate(candidates):
        if completed >= int(args.num_objects):
            break
        if args.dry_run and len(manifest["skipped"]) >= int(args.num_objects):
            break

        digit_value = int(object_id) if str(object_id).isdigit() else 0
        rng = random.Random(int(args.random_seed) + 1009 * int(candidate_idx) + 17 * digit_value)
        case_indices = sample_case_indices(args.case_pool, int(args.cases_per_object), rng)
        speed_cfg = sample_speed_config(args, rng)
        try:
            copied, filtered = run_physxnet_pool_one_object(args, object_id, work_root, case_indices, speed_cfg)
        except subprocess.CalledProcessError as exc:
            print(f"[skip] object_id={object_id} subprocess failed: {exc}")
            manifest["skipped"].append({"object_id": str(object_id), "reason": "subprocess_failed"})
            continue
        except Exception as exc:
            print(f"[skip] object_id={object_id} failed: {type(exc).__name__}: {exc}")
            manifest["skipped"].append({"object_id": str(object_id), "reason": f"{type(exc).__name__}: {exc}"})
            continue

        if not copied:
            print(f"[skip] object_id={object_id} no copied samples")
            skip_reason = "no_samples_after_visibility_filter" if filtered else "no_samples"
            manifest["skipped"].append(
                {
                    "object_id": str(object_id),
                    "reason": skip_reason,
                    "filtered_samples": filtered,
                }
            )
            continue

        completed += 1
        print(f"[ok] object_id={object_id} copied={len(copied)} completed_objects={completed}/{args.num_objects}")
        manifest["objects"].append(
            {
                "object_id": str(object_id),
                "case_indices": [int(idx) for idx in case_indices],
                "speed_config": speed_cfg,
                "samples": copied,
                "filtered_samples": filtered,
            }
        )
        (args.output_root / "benchmark_manifest.json").write_text(
            json.dumps(make_json_safe(manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    manifest["completed_num_objects"] = completed
    manifest["total_samples"] = sum(len(item["samples"]) for item in manifest["objects"])
    manifest["scene_composition_counts"] = summarize_scene_counts(args.output_root)
    qa_reports = maybe_run_motion_qa(args, [args.output_root])
    if qa_reports:
        manifest["post_generation_motion_qa"] = qa_reports[0]
    (args.output_root / "benchmark_manifest.json").write_text(
        json.dumps(make_json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] completed_objects={completed} manifest={args.output_root / 'benchmark_manifest.json'}")
    maybe_generate_captions(args, args.output_root)


def summarize_scene_counts(output_root: Path) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for sample_dir in iter_sample_dirs(output_root / "train"):
        try:
            meta_path = find_sample_meta_path(sample_dir)
            if meta_path is None:
                continue
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = f"{metadata.get('scene_composition', 'unknown')}__n{len(metadata.get('objects', []))}"
        stats[key] = stats.get(key, 0) + 1
    return stats


def extract_object_id_from_sample_dir(sample_dir: str) -> Optional[str]:
    match = re.search(r"/([^/]+)__case", str(sample_dir))
    return match.group(1) if match else None


def collect_stage1_used_object_ids(stage1_subset_root: Path) -> List[str]:
    used: set[str] = set()
    if not stage1_subset_root.exists():
        return []
    for manifest_path in stage1_subset_root.rglob("manifest.json"):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in data.get("accepted", []):
            sample_dir = item.get("sample_dir", "")
            object_id = extract_object_id_from_sample_dir(sample_dir)
            if object_id:
                used.add(object_id)
    return sorted(used)


def choose_stage1_heldout_ids(args: argparse.Namespace) -> List[str]:
    if args.heldout_ids:
        return [str(x) for x in args.heldout_ids]
    stage1_subset_root = args.stage1_train_root / "preprocess_v1" / "stage1_subsets_v1"
    used = set(collect_stage1_used_object_ids(stage1_subset_root))
    all_ids = list_physxnet_object_ids(args.physx_root, args.version)
    held = [obj_id for obj_id in all_ids if obj_id not in used]
    rng = random.Random(int(args.heldout_seed))
    rng.shuffle(held)
    return held[: int(args.heldout_count)]


def write_lines(path: Path, items: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(item) for item in items)
    if items:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def run_stage1_object(args: argparse.Namespace, object_id: str, log_path: Path) -> bool:
    cmd = [
        sys.executable,
        str(args.script),
        "--physx_root",
        str(args.physx_root),
        "--version",
        str(args.version),
        "--object_id",
        str(object_id),
        "--output_root",
        str(args.output_root),
        "--run_genesis",
        "--num_random_cases",
        str(args.num_random_cases),
        "--case_scene_mode",
        "diverse",
        "--case_index_filter",
        *[str(idx) for idx in args.case_index_filter],
        "--prefer_existing_runtime_meshes",
        "--dt",
        str(args.dt),
        "--substeps",
        str(args.substeps),
        "--steps",
        str(args.steps),
        "--fps",
        str(args.fps),
        "--simulator_mode",
        "rigid",
        "--rigid_target_object_count",
        str(args.rigid_target_object_count),
        "--physxnet_volume_threshold_m3",
        str(args.physxnet_volume_threshold_m3),
        "--case_seed",
        str(args.heldout_seed),
        "--motion_case_max_retries",
        str(args.motion_case_max_retries),
        "--disable_striker",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] stage1 object_id={object_id}")
    print(" ".join(str(part) for part in cmd))
    if args.dry_run:
        return True
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    return result.returncode == 0


def build_stage1_subsets(args: argparse.Namespace, log_path: Path) -> None:
    cmd = [
        sys.executable,
        str(args.build_script),
        "--dataset_root",
        str(args.output_root),
        "--out_root",
        str(args.output_root / "preprocess_v1" / "stage1_subsets_v1"),
        "--max_source_samples",
        str(args.max_source_samples),
        "--max_windows_per_subset",
        str(args.max_windows_per_subset),
        "--future_main_visibility_threshold",
        str(args.future_main_visibility_threshold),
        "--count_buckets",
        str(args.subset_count_buckets),
    ]
    print(" ".join(str(part) for part in cmd))
    if args.dry_run:
        return
    shutil.rmtree(args.output_root / "preprocess_v1" / "stage1_subsets_v1", ignore_errors=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True)


def cmd_stage1_heldout(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_root / "logs"
    heldout_ids_path = args.output_root / "heldout_ids.txt"
    excluded_ids_path = args.output_root / "excluded_stage1_train_ids.txt"
    stage1_subset_root = args.stage1_train_root / "preprocess_v1" / "stage1_subsets_v1"

    used_ids = collect_stage1_used_object_ids(stage1_subset_root)
    heldout_ids = choose_stage1_heldout_ids(args)
    write_lines(excluded_ids_path, used_ids)
    write_lines(heldout_ids_path, heldout_ids)

    manifest: Dict[str, Any] = {
        "generator": "generate_rigid_benchmark.py::stage1_heldout",
        "output_root": str(args.output_root),
        "physx_root": str(args.physx_root),
        "version": str(args.version),
        "heldout_seed": int(args.heldout_seed),
        "heldout_ids": list(heldout_ids),
        "excluded_ids": list(used_ids),
        "case_index_filter": [int(idx) for idx in args.case_index_filter],
        "visibility_filter": {
            "enabled": not bool(args.disable_visibility_filter),
            "max_no_object_ratio": float(args.max_no_object_ratio),
        },
        "generated": [],
        "failed": [],
        "filtered_samples": [],
    }

    total = len(heldout_ids)
    for idx, object_id in enumerate(heldout_ids, start=1):
        success = run_stage1_object(args, str(object_id), log_dir / f"{object_id}.log")
        if success:
            kept_samples, filtered_samples = ([], []) if args.dry_run else filter_existing_samples_for_object(
                args=args,
                dataset_root=args.output_root,
                object_id=str(object_id),
            )
            manifest["filtered_samples"].extend(filtered_samples)
            if args.dry_run or kept_samples:
                print(f"[ok] [{idx}/{total}] object_id={object_id} kept_samples={len(kept_samples)}")
                manifest["generated"].append(
                    {
                        "object_id": str(object_id),
                        "samples": kept_samples,
                    }
                )
            else:
                print(f"[fail] [{idx}/{total}] object_id={object_id} all samples filtered")
                manifest["failed"].append(
                    {
                        "object_id": str(object_id),
                        "reason": "all_samples_filtered_by_visibility",
                    }
                )
        else:
            print(f"[fail] [{idx}/{total}] object_id={object_id}")
            manifest["failed"].append({"object_id": str(object_id), "reason": "subprocess_failed"})

    qa_reports = maybe_run_motion_qa(args, [args.output_root])
    if qa_reports:
        manifest["post_generation_motion_qa"] = qa_reports[0]
    if not args.skip_subset_build:
        build_stage1_subsets(args, log_dir / "build_stage1_subsets.log")
        summary_path = args.output_root / "preprocess_v1" / "stage1_subsets_v1" / "summary.json"
        if summary_path.exists():
            manifest["subset_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    (args.output_root / "benchmark_manifest.json").write_text(
        json.dumps(make_json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    maybe_generate_captions(args, args.output_root)


def load_train_seen(train_manifest: Path) -> Tuple[set[str], set[int]]:
    scene_ids: set[str] = set()
    seeds: set[int] = set()
    if not train_manifest.exists():
        return scene_ids, seeds
    payload = json.loads(train_manifest.read_text(encoding="utf-8"))
    for item in payload.get("scenes", []):
        scene_id = item.get("scene_id")
        seed = item.get("seed")
        if scene_id is not None:
            scene_ids.add(str(scene_id))
        if seed is not None:
            seeds.add(int(seed))
    return scene_ids, seeds


def sample_benchmark_v1_object_pool(args: argparse.Namespace) -> List[str]:
    if args.object_ids:
        return [str(x) for x in args.object_ids]
    finaljson_dir = Path(args.physx_root) / str(args.version) / "finaljson"
    object_ids = sorted(path.stem for path in finaljson_dir.glob("*.json"))
    heldout_ids = [obj_id for obj_id in object_ids if stable_hash(obj_id + BENCHMARK_V1_HELDOUT_POOL_NAME) % 5 == 0]
    if not heldout_ids:
        heldout_ids = object_ids
    random.Random(BENCHMARK_V1_SEED_BASE).shuffle(heldout_ids)
    return heldout_ids[: max(1, min(int(args.object_pool_size), len(heldout_ids)))]


def subset_case_filter(subset: str, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for case in cases:
        label = str(case.get("scene_label", ""))
        use_entry_motion = bool(case.get("use_entry_motion", False))
        z_offset = float(np.asarray(case.get("placed_pos_offset", [0.0, 0.0, 0.0]), dtype=np.float64)[2])
        if subset == "smooth_single":
            if use_entry_motion and label.startswith("entry_"):
                out.append(case)
            elif (not use_entry_motion) and z_offset <= 1e-6 and label in {"static_center", "static_left", "static_right"}:
                out.append(case)
        elif subset == "smooth_multi_no_contact":
            if label in {"static_center", "static_left", "static_right"}:
                out.append(case)
        elif subset == "event_pair":
            if label in {"static_center", "static_left", "static_right", "static_highdrop", "entry_left", "entry_right", "entry_fast_center"}:
                out.append(case)
        elif subset == "event_multi":
            if label in {"static_center", "static_left", "static_right", "entry_left", "entry_right", "entry_fast_center"}:
                out.append(case)
        elif subset == "ood_physical":
            out.append(case)
        elif subset == "ood_composition":
            if label in {"static_center", "entry_left", "entry_right", "entry_fast_center", "static_highdrop"}:
                out.append(case)
    return out


def adjust_case_for_subset(subset: str, case_cfg: Dict[str, Any], sample_seed: int, sample_id: str) -> Dict[str, Any]:
    case = copy.deepcopy(case_cfg)
    case["seed"] = int(sample_seed)
    case["case_name"] = sample_id
    case["scene_label"] = str(case.get("scene_label", sample_id))
    role_tag = subset

    if subset == "smooth_single":
        case["custom_objects"] = []
        case["placed_pos_offset"] = [0.0, 0.0, 0.90]
        case["use_entry_motion"] = True
        case["entry_linear_velocity"] = [-0.85, 0.0, 0.0]
        case["entry_angular_velocity"] = [0.0, 0.0, 0.0]
        case["gravity_z_override"] = -9.81
        case["skip_ground_alignment"] = True
    elif subset == "smooth_multi_no_contact":
        custom_objects = []
        case["placed_pos_offset"] = [0.0, 0.0, 0.95]
        lanes = [-1.10, 0.0, 1.10]
        for idx, lane in enumerate(lanes, start=1):
            custom_objects.append(
                {
                    "custom_object_id": f"{role_tag}_aux_{idx:02d}",
                    "spawn_offset": [0.55 * idx, lane, 0.96 + 0.10 * idx],
                    "linear_velocity": [0.0, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.0],
                    "color_rgba": [0.2 + 0.2 * idx, 0.7 - 0.1 * idx, 0.8 - 0.1 * idx, 1.0],
                    "radius": 0.055,
                    "runtime_solver": "rigid_approx",
                    "friction": 0.02,
                    "density": 900.0,
                }
            )
        case["custom_objects"] = custom_objects
        case["use_entry_motion"] = False
        case["entry_linear_velocity"] = [0.0, 0.0, 0.0]
        case["entry_angular_velocity"] = [0.0, 0.0, 0.0]
        case["gravity_z_override"] = -9.81
        case["skip_ground_alignment"] = True
    elif subset == "event_pair":
        case["custom_objects"] = [
            {
                "custom_object_id": f"{role_tag}_striker_01",
                "spawn_offset": [0.95, 0.0, 0.0],
                "linear_velocity": [-2.4, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
                "color_rgba": [0.95, 0.72, 0.18, 1.0],
                "radius": 0.06,
                "runtime_solver": "rigid_approx",
                "friction": 0.01,
                "density": 1200.0,
            }
        ]
        case["use_entry_motion"] = False
        case["entry_linear_velocity"] = [0.0, 0.0, 0.0]
        case["entry_angular_velocity"] = [0.0, 0.0, 0.0]
        case["placed_pos_offset"] = [0.0, 0.0, 0.32]
        case["gravity_z_override"] = -9.81
        case["skip_ground_alignment"] = False
    elif subset == "event_multi":
        case["placed_pos_offset"] = [0.0, 0.0, 0.32]
        case["custom_objects"] = [
            {
                "custom_object_id": f"{role_tag}_target_01",
                "spawn_offset": [0.36, 0.0, 0.0],
                "linear_velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
                "color_rgba": [0.82, 0.38, 0.28, 1.0],
                "radius": 0.055,
                "runtime_solver": "rigid_approx",
                "friction": 0.02,
                "density": 1100.0,
            },
            {
                "custom_object_id": f"{role_tag}_target_02",
                "spawn_offset": [0.72, 0.0, 0.0],
                "linear_velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
                "color_rgba": [0.68, 0.52, 0.34, 1.0],
                "radius": 0.055,
                "runtime_solver": "rigid_approx",
                "friction": 0.02,
                "density": 1100.0,
            },
            {
                "custom_object_id": f"{role_tag}_striker_01",
                "spawn_offset": [1.10, 0.0, 0.0],
                "linear_velocity": [-3.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
                "color_rgba": [0.92, 0.72, 0.18, 1.0],
                "radius": 0.06,
                "runtime_solver": "rigid_approx",
                "friction": 0.01,
                "density": 1200.0,
            },
        ]
        case["use_entry_motion"] = False
        case["entry_linear_velocity"] = [0.0, 0.0, 0.0]
        case["entry_angular_velocity"] = [0.0, 0.0, 0.0]
        case["gravity_z_override"] = -9.81
        case["skip_ground_alignment"] = False
    elif subset == "ood_physical":
        case["custom_objects"] = []
        case["placed_pos_offset"] = [0.0, 0.0, 0.0]
        case["use_entry_motion"] = True
        case["entry_linear_velocity"] = [-2.2, 0.0, 0.0]
        case["entry_angular_velocity"] = [0.0, 0.0, 0.0]
        case["gravity_z_override"] = -9.81
        case["skip_ground_alignment"] = False
    elif subset == "ood_composition":
        custom_objects = []
        launch_specs = [
            ("left", [0.15, -0.55, 0.0], [0.0, 0.0, 0.0]),
            ("mid", [0.32, 0.12, 0.0], [0.0, 0.0, 0.0]),
            ("right", [0.48, 0.78, 0.0], [0.0, 0.0, 0.0]),
            ("drop", [0.10, 0.42, 0.36], [0.0, 0.0, -0.2]),
        ]
        for idx, (tag, offset, linvel) in enumerate(launch_specs, start=1):
            custom_objects.append(
                {
                    "custom_object_id": f"{role_tag}_{tag}_{idx:02d}",
                    "spawn_offset": offset,
                    "linear_velocity": linvel,
                    "angular_velocity": [0.0, 0.0, 0.0],
                    "color_rgba": [0.25 + 0.1 * idx, 0.55, 0.9 - 0.1 * idx, 1.0],
                    "radius": 0.05,
                    "runtime_solver": "rigid_approx",
                    "friction": 0.15 if idx % 2 else 1.8,
                    "density": 5000.0 if idx % 2 else 120.0,
                }
            )
        case["custom_objects"] = custom_objects
        case["use_entry_motion"] = True
        case["entry_linear_velocity"] = [-2.1, 0.0, 0.0]
        case["entry_angular_velocity"] = [0.0, 0.0, 0.0]
        case["gravity_z_override"] = -9.81
        case["skip_ground_alignment"] = False

    case["export_split"] = None
    case["export_case_dir"] = None
    return case


def compute_scene_signature(metadata: Dict[str, Any]) -> str:
    signature = {
        "object_id": metadata.get("object_id"),
        "seed": metadata.get("seed"),
        "objects": [
            {
                "object_id": obj.get("object_id"),
                "seg_id": obj.get("seg_id"),
                "role": obj.get("role"),
                "motion_type": obj.get("motion_type"),
                "motion_group": obj.get("motion_group"),
                "source_tag": obj.get("source_tag"),
            }
            for obj in metadata.get("objects", [])
        ],
    }
    text = json.dumps(make_json_safe(signature), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def overlaps_future(frame_start: int, frame_end: int) -> bool:
    return int(frame_end) >= BENCHMARK_V1_FUTURE_START


def compute_benchmark_v1_stats(sample_dir: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
    physics_dir = sample_dir / "physics"
    contact_graph = np.load(physics_dir / "contact_graph.npy")
    env_contact = np.load(physics_dir / "env_contact.npy")
    event_windows = json.loads((physics_dir / "event_windows.json").read_text(encoding="utf-8"))
    num_event_windows_future = sum(
        1
        for item in event_windows
        if overlaps_future(int(item.get("start_frame", -1)), int(item.get("end_frame", -1)))
    )
    return {
        "num_objects": int(metadata.get("num_objects", len(metadata.get("objects", [])))),
        "has_future_event": bool(num_event_windows_future > 0),
        "num_event_windows_future": int(num_event_windows_future),
        "num_collision_events_future": int(num_event_windows_future),
        "future_contact_sum": int(np.asarray(contact_graph[BENCHMARK_V1_FUTURE_START:], dtype=np.int64).sum()),
        "future_env_contact_sum": int(np.asarray(env_contact[BENCHMARK_V1_FUTURE_START:], dtype=np.int64).sum()),
    }


def validate_benchmark_v1_subset(subset: str, stats: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
    num_objects = int(stats["num_objects"])
    if subset == "smooth_single":
        return num_objects == 1 and stats["future_contact_sum"] == 0 and stats["future_env_contact_sum"] == 0
    if subset == "smooth_multi_no_contact":
        return 2 <= num_objects <= 4 and stats["future_contact_sum"] == 0 and stats["future_env_contact_sum"] == 0
    if subset == "event_pair":
        return num_objects == 2 and stats["has_future_event"]
    if subset == "event_multi":
        return 3 <= num_objects <= 6 and (stats["num_event_windows_future"] >= 2 or stats["num_collision_events_future"] >= 2)
    if subset == "ood_physical":
        return metadata.get("benchmark", {}).get("subset") == "ood_physical"
    if subset == "ood_composition":
        return metadata.get("benchmark", {}).get("subset") == "ood_composition"
    return False


def finalize_benchmark_v1_metadata(sample_dir: Path, split: str, subset: str) -> Dict[str, Any]:
    metadata_path = find_sample_meta_path(sample_dir)
    if metadata_path is None:
        raise FileNotFoundError(f"Missing meta.json/metadata.json under {sample_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stats = compute_benchmark_v1_stats(sample_dir, metadata)
    try:
        metadata["output_relpath"] = str(sample_dir.relative_to(sample_dir.parents[3]))
    except Exception:
        pass
    metadata["benchmark"] = {
        "benchmark_name": BENCHMARK_V1_NAME,
        "benchmark_track": BENCHMARK_V1_TRACK,
        "split": str(split),
        "subset": str(subset),
        "total_frames": BENCHMARK_V1_TOTAL_FRAMES,
        "context_frames": BENCHMARK_V1_CONTEXT_FRAMES,
        "pred_frames": BENCHMARK_V1_PRED_FRAMES,
        "is_benchmark_sample": True,
    }
    metadata["benchmark_stats"] = {
        "num_objects": int(stats["num_objects"]),
        "has_future_event": bool(stats["has_future_event"]),
        "num_event_windows_future": int(stats["num_event_windows_future"]),
        "num_collision_events_future": int(stats["num_collision_events_future"]),
    }
    metadata["split"] = str(split)
    metadata["heldout_pool"] = BENCHMARK_V1_HELDOUT_POOL_NAME
    metadata["scene_signature"] = compute_scene_signature(metadata)
    metadata_path.write_text(json.dumps(make_json_safe(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def build_benchmark_v1_manifest_record(metadata: Dict[str, Any]) -> Dict[str, Any]:
    stats = metadata["benchmark_stats"]
    benchmark = metadata["benchmark"]
    return {
        "sample_id": str(metadata["scene_id"]),
        "track": BENCHMARK_V1_TRACK,
        "split": str(benchmark["split"]),
        "subset": str(benchmark["subset"]),
        "relative_path": str(metadata.get("output_relpath", "")),
        "num_objects": int(stats["num_objects"]),
        "context_frames": int(benchmark["context_frames"]),
        "pred_frames": int(benchmark["pred_frames"]),
        "has_future_event": bool(stats["has_future_event"]),
        "num_event_windows_future": int(stats["num_event_windows_future"]),
        "num_collision_events_future": int(stats["num_collision_events_future"]),
    }


def write_benchmark_v1_format(dataset_root: Path) -> None:
    text = f"""# {BENCHMARK_V1_NAME}

Rigid-only benchmark set derived from `try1_physxnet_benchmark.py`.

## Protocol

- track: `{BENCHMARK_V1_TRACK}`
- total frames: `{BENCHMARK_V1_TOTAL_FRAMES}`
- context frames: `{BENCHMARK_V1_CONTEXT_FRAMES}`
- prediction frames: `{BENCHMARK_V1_PRED_FRAMES}`
- future interval: frames `8..48`
- resolution: `960x720` by default

## Layout

```text
benchmark_v1/
  BENCHMARK_FORMAT.md
  benchmark_manifest.json
  rigid/
    dev/
      smooth_single/
      smooth_multi_no_contact/
      event_pair/
      event_multi/
      ood_physical/
      ood_composition/
    test/
      smooth_single/
      smooth_multi_no_contact/
      event_pair/
      event_multi/
      ood_physical/
      ood_composition/
```
"""
    (dataset_root / "BENCHMARK_FORMAT.md").write_text(text, encoding="utf-8")


def make_try1_args(args: argparse.Namespace, object_id: str) -> argparse.Namespace:
    return try1.build_argparser().parse_args(
        [
            "--physx_root",
            str(args.physx_root),
            "--version",
            str(args.version),
            "--object_id",
            str(object_id),
            "--output_root",
            str(args.dataset_root / "_try1_cache"),
            "--simulator_mode",
            "rigid",
            "--steps",
            str(int(args.steps)),
            "--dt",
            str(float(args.dt)),
            "--substeps",
            str(int(args.substeps)),
            "--fps",
            str(int(args.fps)),
            "--striker_speed",
            str(float(args.striker_speed)),
            "--striker_radius",
            str(float(args.striker_radius)),
            "--camera_distance_mult",
            str(float(args.camera_distance_mult)),
            "--disable_striker",
        ]
    )


def generate_benchmark_v1_sample(
    *,
    args: argparse.Namespace,
    object_id: str,
    split: str,
    subset: str,
    sample_index: int,
    sample_seed: int,
) -> Path:
    runner_args = make_try1_args(args, object_id)
    prepared = try1.prepare_physxnet_object(
        physx_root=Path(runner_args.physx_root),
        version=runner_args.version,
        object_id=str(object_id),
        output_root=Path(runner_args.output_root),
        voxel_pitch=float(runner_args.voxel_pitch),
        json_override=Path(runner_args.json_override) if runner_args.json_override else None,
        object_scale_mult=float(runner_args.object_scale_mult),
        solver_family_override=runner_args.solver_family_override,
        all_parts_youngs_threshold_gpa=runner_args.all_parts_youngs_threshold_gpa,
        rigid_visual_double_sided_shell=True,
        simulator_mode="rigid",
    )
    base_cases = try1.build_preview_case_configs(
        prepared=prepared,
        output_root=Path(runner_args.output_root),
        object_fixed=bool(runner_args.object_fixed),
        args=runner_args,
    )
    candidate_cases = subset_case_filter(subset, base_cases)
    if not candidate_cases:
        raise RuntimeError(f"No candidate try1 cases for subset={subset} object_id={object_id}")
    chosen = copy.deepcopy(candidate_cases[sample_seed % len(candidate_cases)])
    sample_id = f"sample_{sample_index:06d}"
    chosen = adjust_case_for_subset(subset, chosen, sample_seed, sample_id)
    sample_dir = args.dataset_root / BENCHMARK_V1_TRACK / split / subset / sample_id
    chosen["export_case_dir"] = str(sample_dir)
    chosen["export_split"] = str(split)
    chosen["scene_id_override"] = sample_id
    chosen["scene_label"] = f"{subset}__{chosen.get('scene_label', 'case')}"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    try1.simulate_in_genesis(
        prepared=prepared,
        output_root=Path(runner_args.output_root),
        steps=int(args.steps),
        dt=float(args.dt),
        substeps=int(args.substeps),
        fps=int(args.fps),
        default_friction=float(runner_args.default_friction),
        object_fixed=bool(runner_args.object_fixed),
        striker_radius=float(args.striker_radius),
        striker_speed=float(args.striker_speed),
        args=runner_args,
        case_cfg=chosen,
    )
    return sample_dir


def cmd_benchmark_v1(args: argparse.Namespace) -> None:
    for subset in args.subsets:
        if subset not in BENCHMARK_V1_SUBSETS:
            raise ValueError(f"Unknown subset: {subset}")

    args.dataset_root = Path(args.dataset_root).resolve()
    try1.ensure_dir(args.dataset_root)
    try1.ensure_dir(args.dataset_root / BENCHMARK_V1_TRACK / "dev")
    try1.ensure_dir(args.dataset_root / BENCHMARK_V1_TRACK / "test")
    try1.ensure_dir(args.dataset_root / "_try1_cache")
    try1.EXPORT_CAMERA_RESOLUTION = (int(args.resolution[0]), int(args.resolution[1]))
    write_benchmark_v1_format(args.dataset_root)

    train_scene_ids, train_seeds = load_train_seen(args.train_manifest)
    object_pool = sample_benchmark_v1_object_pool(args)

    manifest: Dict[str, Any] = {
        "generator": "generate_rigid_benchmark.py::benchmark_v1",
        "benchmark_name": BENCHMARK_V1_NAME,
        "track": BENCHMARK_V1_TRACK,
        "splits": ["dev", "test"],
        "subsets": list(args.subsets),
        "total_frames": BENCHMARK_V1_TOTAL_FRAMES,
        "context_frames": BENCHMARK_V1_CONTEXT_FRAMES,
        "pred_frames": BENCHMARK_V1_PRED_FRAMES,
        "image_size": [int(args.resolution[0]), int(args.resolution[1])],
        "heldout_pool": BENCHMARK_V1_HELDOUT_POOL_NAME,
        "train_manifest": str(args.train_manifest),
        "visibility_filter": {
            "enabled": not bool(args.disable_visibility_filter),
            "max_no_object_ratio": float(args.max_no_object_ratio),
        },
        "records": [],
        "failed": [],
    }

    requested = {"dev": int(args.dev_per_subset), "test": int(args.test_per_subset)}
    global_sample_index = 0
    seed_cursor = 0

    try:
        for split in ["dev", "test"]:
            for subset in args.subsets:
                accepted = 0
                attempts = 0
                max_attempts = requested[split] * int(args.max_attempts_per_sample)
                while accepted < requested[split] and attempts < max_attempts:
                    sample_seed = BENCHMARK_V1_SEED_BASE + seed_cursor
                    object_id = object_pool[seed_cursor % len(object_pool)]
                    seed_cursor += 1
                    attempts += 1
                    sample_id = f"sample_{global_sample_index:06d}"
                    if sample_id in train_scene_ids or sample_seed in train_seeds:
                        continue
                    sample_dir = args.dataset_root / BENCHMARK_V1_TRACK / split / subset / sample_id
                    try:
                        generated_dir = generate_benchmark_v1_sample(
                            args=args,
                            object_id=object_id,
                            split=split,
                            subset=subset,
                            sample_index=global_sample_index,
                            sample_seed=sample_seed,
                        )
                        if generated_dir != sample_dir:
                            raise RuntimeError(f"Unexpected sample dir: {generated_dir} != {sample_dir}")
                        passed_visibility, visibility_stats = check_sample_visibility_filter(sample_dir, args)
                        if not passed_visibility:
                            manifest["failed"].append(
                                {
                                    "split": split,
                                    "subset": subset,
                                    "sample_id": sample_id,
                                    "seed": int(sample_seed),
                                    "object_id": str(object_id),
                                    "error": "filtered_by_visibility",
                                    "visibility": visibility_stats,
                                }
                            )
                            shutil.rmtree(sample_dir, ignore_errors=True)
                            continue
                        qa_record = evaluate_benchmark_sample_qa(sample_dir, args)
                        if not bool(qa_record.get("valid", False)):
                            write_sample_qa_metrics(sample_dir, qa_record, dry_run=False)
                            manifest["failed"].append(
                                {
                                    "split": split,
                                    "subset": subset,
                                    "sample_id": sample_id,
                                    "seed": int(sample_seed),
                                    "object_id": str(object_id),
                                    "error": "filtered_by_motion_qa",
                                    "qa": qa_record,
                                }
                            )
                            shutil.rmtree(sample_dir, ignore_errors=True)
                            continue
                        metadata = finalize_benchmark_v1_metadata(sample_dir, split, subset)
                        stats = compute_benchmark_v1_stats(sample_dir, metadata)
                        visibility_bundle = qa_record.get("visibility_filter", visibility_stats)
                        stats["visible_frames"] = int(visibility_bundle.get("visible_frames", 0))
                        stats["no_object_frames"] = int(visibility_bundle.get("no_object_frames", 0))
                        stats["no_object_ratio"] = float(visibility_bundle.get("no_object_ratio", 0.0))
                        if not validate_benchmark_v1_subset(subset, stats, metadata):
                            shutil.rmtree(sample_dir, ignore_errors=True)
                            continue
                        metadata["benchmark_stats"] = {
                            "num_objects": int(stats["num_objects"]),
                            "has_future_event": bool(stats["has_future_event"]),
                            "num_event_windows_future": int(stats["num_event_windows_future"]),
                            "num_collision_events_future": int(stats["num_collision_events_future"]),
                            "visible_frames": int(stats["visible_frames"]),
                            "no_object_frames": int(stats["no_object_frames"]),
                            "no_object_ratio": float(stats["no_object_ratio"]),
                        }
                        (sample_dir / "meta.json").write_text(
                            json.dumps(make_json_safe(metadata), ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        manifest["records"].append(build_benchmark_v1_manifest_record(metadata))
                        accepted += 1
                        global_sample_index += 1
                        print(f"[ok] {split}/{subset}/{sample_id} object={object_id}")
                    except Exception as exc:
                        shutil.rmtree(sample_dir, ignore_errors=True)
                        manifest["failed"].append(
                            {
                                "split": split,
                                "subset": subset,
                                "sample_id": sample_id,
                                "seed": int(sample_seed),
                                "object_id": str(object_id),
                                "error": str(exc),
                            }
                        )
                        print(f"[fail] {split}/{subset}/{sample_id} object={object_id} err={exc}")
                if accepted < requested[split]:
                    raise RuntimeError(f"Subset {split}/{subset} only generated {accepted}/{requested[split]} valid samples")
    finally:
        qa_reports = maybe_run_motion_qa(args, [args.dataset_root])
        if qa_reports:
            manifest["post_generation_motion_qa"] = qa_reports[0]
        (args.dataset_root / "benchmark_manifest.json").write_text(
            json.dumps(make_json_safe(manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            import genesis as gs

            gs.destroy()
        except Exception:
            pass
    maybe_generate_captions(args, args.dataset_root)


def cmd_qa_existing(args: argparse.Namespace) -> None:
    reports = maybe_run_motion_qa(args, [Path(root) for root in args.roots])
    total = sum(int(report.get("samples_total", 0)) for report in reports)
    valid = sum(int(report.get("samples_valid", 0)) for report in reports)
    invalid = sum(int(report.get("samples_invalid", 0)) for report in reports)
    print(f"[done] qa_existing roots={len(reports)} total={total} valid={valid} invalid={invalid}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "physxnet_pool":
        cmd_physxnet_pool(args)
        return
    if args.command == "stage1_heldout":
        cmd_stage1_heldout(args)
        return
    if args.command == "benchmark_v1":
        cmd_benchmark_v1(args)
        return
    if args.command == "qa_existing":
        cmd_qa_existing(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()


'''

  physxnet_pool 是“原始 benchmark 样本池生成器”。它的作用是从 PhysXNet 里随机挑一批物体，再按你给的
  case_pool、物体数量、速度范围这些参数，生成一大批 rigid benchmark 场景，输出到 train/rigid/
  <scene_composition>/<count_bucket>/<sample>。这个模式适合先把一个基础 benchmark 池铺出来，后续再从里面
  挑子集或做分析。



#### 先不用
  # 1) 平铺 PhysXNet benchmark pool
  /data/gaoya/miniconda3/envs/wan/bin/python \
    /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/generate_rigid_benchmark.py \
    physxnet_pool \
    --output_root /data/gaoya/AAA_test_video/Dataset_physV/0417data_benchmark \
    --num_objects 50 \
    --random_seed 20260423 \
    --cases_per_object 3 \
    --case_pool 5 6 7 \
    --rigid_target_object_count 1 \
    --overwrite

    

'''



# 2) 单物体 drop/抛物线运动
#  Stage1 held-out benchmark

#   stage1_heldout 是“。它会先避开训练阶段已经用过的 object id（stage1_train_root目录中的），再从剩下
#   的 PhysXNet 物体里抽一批 held-out 物体，只生成指定的 motion case，默认就是 900/901，也就是
#   random_parabola 和 high_drop。然后它还能顺手调用 build_stage1_subsets.py，把这些视频进一步切成 Stage-1
#   训练/评测窗口，所以它更偏向“给 state predictor / window-based benchmark 准备数据”。

#   放在 /data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1_count01_benchmark  
'''
  1. 先在 train/rigid/single_object_preview/... 生成完整样本
  2. build_stage1_subsets.py 读取这些完整样本里的：
      - metadata.json
      - physics/anchor_targets.npz
      - physics/contact_graph.npy
      - physics/event_windows.json
      - 其他时序物理信息
  3. 再按 Stage1 规则切成很多窗口
  4. 输出到 preprocess_v1/stage1_subsets_v1/...


python /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/generate_rigid_benchmark.py \
      stage1_heldout \
      --output_root /data/gaoya/AAA_test_video/Dataset_physV/0417data_benchmark/stage1_count01_benchmark \
      --stage1_train_root /data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases \
      --heldout_seed 0 \
      --heldout_count 80 \
      --case_index_filter 900 901 \
      --generate_captions \
      --caption_include_invalid


      

fuser -k 8765/tcp

python3 /home/gaoya/video_caption_viewer/build_manifest.py

python3 -m http.server 8765 --directory /

然后浏览器打开：

http://127.0.0.1:8765/home/gaoya/video_caption_viewer/index.html
'''









# 3) benchmark_v1
#   benchmark_v1 是“按固定 benchmark protocol 生成 dev/test 子集”的模式。它不是随便铺一个大池子，而是严格
#   按 smooth_single、event_pair、ood_physical 这类 subset 定义去造样本，并且会检查生成结果是否满足该
#   subset 的规则，比如物体数、是否有未来碰撞、是否属于 OOD 物理设定等。这个模式适合真正做标准化评测，因为
#   输出结构和筛选规则都是固定的。


'''
    

  /data/gaoya/miniconda3/envs/wan/bin/python \
    /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/generate_rigid_benchmark.py \
    benchmark_v1 \
    --dataset-root /data/gaoya/AAA_test_video/Dataset_physV/0417data_benchmark/benchmark_v1_all_cases \
    --dev-per-subset 1 \
    --test-per-subset 0 \
    --subsets smooth_single smooth_multi_no_contact event_pair event_multi ood_physical ood_composition \
    --object-ids 19925



'''
