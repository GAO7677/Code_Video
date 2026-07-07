#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, hf_hub_download


ACTIVITIES = ("SinglePhysics", "DoublePhysics", "TriplePhysics")


@dataclass(slots=True)
class CaseRecord:
    case_id: str
    split: str
    activity_type: str
    phenomena: list[str]
    phenomena_key: str
    part_id: str
    repo_id: str
    hf_zip_path: str

    @property
    def cache_key(self) -> tuple[str, str]:
        return (self.repo_id, self.hf_zip_path)


def _chunked(items: list[CaseRecord], size: int) -> Iterable[list[CaseRecord]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _bytes_to_gib_str(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def _parse_assignment_line(line: str, repo_map: dict[str, str]) -> CaseRecord | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    ue_path, part_id = line.split()
    path_parts = ue_path.split("/")
    split = path_parts[4]
    activity_type = path_parts[5]
    object_ref = path_parts[6]
    case_id = object_ref.split(".")[-1]
    phenomena_key = case_id.split("__", 1)[0]
    phenomena = [token for token in phenomena_key.split("_") if token]
    repo_id = repo_map[part_id]
    hf_zip_path = f"{split}/{activity_type}/{case_id}_trajectory.zip"
    return CaseRecord(
        case_id=case_id,
        split=split,
        activity_type=activity_type,
        phenomena=phenomena,
        phenomena_key=phenomena_key,
        part_id=part_id,
        repo_id=repo_id,
        hf_zip_path=hf_zip_path,
    )


def load_cases(assets_root: Path, split: str) -> list[CaseRecord]:
    metadata_root = assets_root / "metadata"
    repo_map = json.loads((metadata_root / "repo_map.json").read_text(encoding="utf-8"))
    assignment_path = metadata_root / "repo_assignment.txt"
    wanted_split = split.strip().capitalize()
    cases: list[CaseRecord] = []
    with assignment_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parsed = _parse_assignment_line(raw_line, repo_map=repo_map)
            if parsed is None:
                continue
            if parsed.split != wanted_split:
                continue
            cases.append(parsed)
    return cases


def build_round_robin_order(cases: list[CaseRecord], seed: int) -> list[CaseRecord]:
    rng = random.Random(seed)
    by_phenomena: dict[str, list[CaseRecord]] = defaultdict(list)
    for case in cases:
        by_phenomena[case.phenomena_key].append(case)

    keys = sorted(by_phenomena.keys())
    rng.shuffle(keys)

    queues: dict[str, deque[CaseRecord]] = {}
    for key, bucket in by_phenomena.items():
        bucket = list(bucket)
        rng.shuffle(bucket)
        queues[key] = deque(bucket)

    active_keys = deque(keys)
    ordered: list[CaseRecord] = []
    while active_keys:
        key = active_keys.popleft()
        bucket = queues[key]
        if not bucket:
            continue
        ordered.append(bucket.popleft())
        if bucket:
            active_keys.append(key)
    return ordered


def prefetch_sizes(
    *,
    api: HfApi,
    cases: list[CaseRecord],
    size_cache: dict[tuple[str, str], int],
    batch_size: int,
) -> None:
    missing = [case for case in cases if case.cache_key not in size_cache]
    if not missing:
        return

    by_repo: dict[str, list[CaseRecord]] = defaultdict(list)
    for case in missing:
        by_repo[case.repo_id].append(case)

    for repo_id, repo_cases in by_repo.items():
        for chunk in _chunked(repo_cases, batch_size):
            infos = api.get_paths_info(
                repo_id=repo_id,
                paths=[case.hf_zip_path for case in chunk],
                repo_type="dataset",
            )
            info_by_path = {info.path: info for info in infos}
            for case in chunk:
                info = info_by_path.get(case.hf_zip_path)
                if info is None or getattr(info, "size", None) is None:
                    raise RuntimeError(
                        f"failed to fetch file size for {case.repo_id}:{case.hf_zip_path}"
                    )
                size_cache[case.cache_key] = int(info.size)


def select_balanced_cases(
    *,
    cases: list[CaseRecord],
    target_bytes: int,
    seed: int,
    api: HfApi,
    size_query_batch: int,
    prefetch_window: int,
) -> tuple[list[CaseRecord], dict[tuple[str, str], int]]:
    cases_by_activity: dict[str, list[CaseRecord]] = defaultdict(list)
    for case in cases:
        cases_by_activity[case.activity_type].append(case)

    ordered_by_activity = {
        activity: build_round_robin_order(cases_by_activity.get(activity, []), seed + idx)
        for idx, activity in enumerate(ACTIVITIES)
    }
    indices = {activity: 0 for activity in ACTIVITIES}
    size_cache: dict[tuple[str, str], int] = {}
    selected: list[CaseRecord] = []
    bytes_by_activity = {activity: 0 for activity in ACTIVITIES}
    total_bytes = 0
    target_per_activity = target_bytes // len(ACTIVITIES)

    def take_next(activity: str) -> bool:
        nonlocal total_bytes
        ordered = ordered_by_activity[activity]
        index = indices[activity]
        if index >= len(ordered):
            return False
        prefetch_sizes(
            api=api,
            cases=ordered[index : index + prefetch_window],
            size_cache=size_cache,
            batch_size=size_query_batch,
        )
        case = ordered[index]
        indices[activity] += 1
        case_size = size_cache[case.cache_key]
        selected.append(case)
        bytes_by_activity[activity] += case_size
        total_bytes += case_size
        return True

    while total_bytes < target_bytes:
        progressed = False
        activities = sorted(ACTIVITIES, key=lambda name: bytes_by_activity[name] / max(target_per_activity, 1))
        for activity in activities:
            if bytes_by_activity[activity] >= target_per_activity:
                continue
            if take_next(activity):
                progressed = True
                if total_bytes >= target_bytes:
                    break
        if not progressed:
            break

    while total_bytes < target_bytes:
        progressed = False
        for activity in sorted(ACTIVITIES, key=lambda name: bytes_by_activity[name]):
            if take_next(activity):
                progressed = True
                break
        if not progressed:
            break

    return selected, size_cache


def summarize_selection(
    *,
    selected: list[CaseRecord],
    size_cache: dict[tuple[str, str], int],
) -> dict:
    counts = Counter(case.activity_type for case in selected)
    bytes_by_activity = Counter()
    unique_phenomena_by_activity: dict[str, set[str]] = defaultdict(set)
    part_counts = Counter(case.part_id for case in selected)
    total_bytes = 0
    for case in selected:
        size = size_cache[case.cache_key]
        total_bytes += size
        bytes_by_activity[case.activity_type] += size
        unique_phenomena_by_activity[case.activity_type].add(case.phenomena_key)

    summary = {
        "num_cases": len(selected),
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1024 ** 3), 3),
        "activity_counts": dict(counts),
        "activity_gib": {
            key: round(bytes_by_activity[key] / (1024 ** 3), 3) for key in ACTIVITIES
        },
        "activity_unique_phenomena": {
            key: len(unique_phenomena_by_activity[key]) for key in ACTIVITIES
        },
        "top_parts": dict(part_counts.most_common(10)),
    }
    return summary


def download_selected_cases(
    *,
    selected: list[CaseRecord],
    output_root: Path,
    cache_dir: Path,
    endpoint: str,
    token: str | None,
    force_download: bool,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for case in selected:
        local_path = output_root / case.hf_zip_path
        if local_path.is_file() and not force_download:
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=case.repo_id,
            filename=case.hf_zip_path,
            repo_type="dataset",
            token=token,
            endpoint=endpoint,
            cache_dir=cache_dir,
            local_dir=output_root,
            local_dir_use_symlinks=False,
            force_download=force_download,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select and download an approximately size-bounded PhysInOne subset, "
            "balanced across Single/Double/Triple physics categories."
        )
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path("/data/gaoya/dataset/vLAR-PhysInOne/vLAR-PhysInOne-assets/assets"),
        help="Root of the downloaded PhysInOne assets directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/gaoya/dataset/vLAR-PhysInOne/TrainBalanced100G"),
        help="Destination root. Files will be placed under Split/ActivityType/*.zip.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download: train / val / test.",
    )
    parser.add_argument(
        "--target-size-gib",
        type=float,
        default=100.0,
        help="Target total download size in GiB.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for within-category shuffling.",
    )
    parser.add_argument(
        "--hf-endpoint",
        type=str,
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help="Hugging Face endpoint.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/data/gaoya/agent-data/cache/huggingface/hub"),
        help="Hugging Face cache directory.",
    )
    parser.add_argument(
        "--size-query-batch",
        type=int,
        default=64,
        help="Batch size for Hub size metadata queries.",
    )
    parser.add_argument(
        "--prefetch-window",
        type=int,
        default=64,
        help="How many upcoming cases to prefetch size metadata for each activity stream.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only build the selection manifest, do not download files.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    args = parser.parse_args()

    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("all_proxy", None)
    os.environ.pop("ALL_PROXY", None)
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = str(args.cache_dir.parent)
    os.environ["HF_HUB_CACHE"] = str(args.cache_dir)

    api = HfApi(endpoint=args.hf_endpoint, token=args.hf_token)
    cases = load_cases(args.assets_root, split=args.split)
    target_bytes = int(args.target_size_gib * (1024 ** 3))
    selected, size_cache = select_balanced_cases(
        cases=cases,
        target_bytes=target_bytes,
        seed=args.seed,
        api=api,
        size_query_batch=max(1, int(args.size_query_batch)),
        prefetch_window=max(1, int(args.prefetch_window)),
    )
    summary = summarize_selection(selected=selected, size_cache=size_cache)

    manifest = {
        "config": {
            "assets_root": str(args.assets_root),
            "output_root": str(args.output_root),
            "split": args.split,
            "target_size_gib": args.target_size_gib,
            "seed": args.seed,
            "hf_endpoint": args.hf_endpoint,
            "cache_dir": str(args.cache_dir),
        },
        "summary": summary,
        "cases": [asdict(case) for case in selected],
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Selection summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Manifest written to: {manifest_path}")

    if args.plan_only:
        return

    download_selected_cases(
        selected=selected,
        output_root=args.output_root,
        cache_dir=args.cache_dir,
        endpoint=args.hf_endpoint,
        token=args.hf_token,
        force_download=args.force_download,
    )

    final_size = 0
    for path in args.output_root.rglob("*.zip"):
        final_size += path.stat().st_size
    print(f"Downloaded size under {args.output_root}: {_bytes_to_gib_str(final_size)}")


if __name__ == "__main__":
    main()
