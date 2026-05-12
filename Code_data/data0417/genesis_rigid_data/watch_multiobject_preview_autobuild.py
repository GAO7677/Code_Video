# 用途：监听多物体预览目录并自动重建页面。
"""Wait for multi-object regeneration outputs, then auto-build a small validation preview page."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


CASE_INDEX_TO_NAME = {
    0: "case000_static_center",
    1: "case001_static_left",
    2: "case002_static_right",
    3: "case003_static_highdrop",
    5: "case005_entry_left",
    6: "case006_entry_right",
    7: "case007_entry_fast_center",
    100: "case000_static_center_v2",
    101: "case001_static_left_v2",
    102: "case002_static_right_v2",
    900: "case900_random_parabola",
    901: "case901_high_drop",
}
CASE_PRIORITY = [3, 5, 7, 0, 1, 2, 6, 100, 101, 102, 900, 901]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-build a preview page after multi-object regeneration starts producing outputs.")
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--done_jobs", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--validator_script", type=Path, required=True)
    parser.add_argument("--wan_python", type=Path, default=Path("/data/gaoya/miniconda3/envs/wan/bin/python"))
    parser.add_argument("--min_jobs", type=int, default=6)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--poll_seconds", type=int, default=30)
    parser.add_argument("--port", type=int, default=8047)
    return parser.parse_args()


def load_done_jobs(path: Path) -> list[tuple[str, str, int, list[int]]]:
    if not path.exists():
        return []
    jobs: list[tuple[str, str, int, list[int]]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split("\t")
        if len(parts) != 4:
            continue
        object_id, bucket, target_count_raw, case_csv = parts
        try:
            target_count = int(target_count_raw)
        except ValueError:
            continue
        case_indices = []
        for token in case_csv.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                case_indices.append(int(token))
            except ValueError:
                continue
        jobs.append((object_id, bucket, target_count, case_indices))
    return jobs


def choose_case(case_indices: list[int]) -> int | None:
    for preferred in CASE_PRIORITY:
        if preferred in case_indices:
            return preferred
    return case_indices[0] if case_indices else None


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_preview_wrappers(dataset_root: Path, jobs: list[tuple[str, str, int, list[int]]], out_root: Path, max_samples: int) -> list[str]:
    preview_root = out_root
    ensure_clean_dir(preview_root)
    selected_ids: list[str] = []
    multi_root = dataset_root / "train" / "rigid" / "interaction_pair_plus_dynamic"

    for object_id, bucket, _target_count, case_indices in jobs:
        if len(selected_ids) >= max_samples:
            break
        case_idx = choose_case(case_indices)
        if case_idx is None:
            continue
        case_name = CASE_INDEX_TO_NAME.get(case_idx)
        if not case_name:
            continue
        sample_name = f"{object_id}__{case_name}"
        sample_dir = multi_root / bucket / sample_name
        if not (sample_dir / "metadata.json").exists():
            continue
        meta = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
        wrapper_dir = preview_root / sample_name
        wrapper_dir.mkdir(parents=True, exist_ok=True)
        fps = int(meta.get("simulation", {}).get("fps", meta.get("fps", 12) or 12))
        payload = {
            "sample_id": sample_name,
            "scene_id": sample_name,
            "object_id": str(meta.get("object_id", "")),
            "fps": fps,
            "scene_composition": str(meta.get("scene_composition", "")),
            "interaction_pattern": str(meta.get("interaction_pattern", "")),
            "source_paths": {"source_sample_dir": str(sample_dir.resolve())},
        }
        (wrapper_dir / "meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.copy2(sample_dir / "videos" / "rgb.mp4", wrapper_dir / "full_video.mp4")
        shutil.copy2(sample_dir / "videos" / "rgb.mp4", wrapper_dir / "future_gt_video.mp4")
        selected_ids.append(sample_name)

    return selected_ids


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    done_jobs = args.done_jobs.resolve()
    out_root = args.out_root.resolve()
    validator_script = args.validator_script.resolve()
    wan_python = args.wan_python.resolve()

    print(f"[watcher] waiting for >= {args.min_jobs} done multi-object jobs: {done_jobs}", flush=True)
    jobs: list[tuple[str, str, int, list[int]]] = []
    while True:
        jobs = load_done_jobs(done_jobs)
        if len(jobs) >= int(args.min_jobs):
            break
        time.sleep(max(5, int(args.poll_seconds)))

    selected = build_preview_wrappers(dataset_root, jobs, out_root, max_samples=int(args.max_samples))
    if not selected:
        raise RuntimeError("No preview samples could be selected from done jobs.")

    validator_assets = out_root / "validator_assets"
    cmd = [
        str(wan_python),
        str(validator_script),
        "--dataset_root",
        str(out_root),
        "--output_dir",
        str(validator_assets),
        "--serve",
        "--port",
        str(int(args.port)),
    ]
    log_path = out_root / "autobuild.log"
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write("[watcher] selected samples:\n")
        for item in selected:
            log_f.write(f"{item}\n")
        log_f.write(f"[watcher] launching validator on port {args.port}\n")
        log_f.flush()
        subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    print(f"[watcher] built preview with {len(selected)} samples at {out_root}", flush=True)
    print(f"[watcher] validator launching on http://127.0.0.1:{args.port}/state_validation_browser.html", flush=True)


if __name__ == "__main__":
    main()
