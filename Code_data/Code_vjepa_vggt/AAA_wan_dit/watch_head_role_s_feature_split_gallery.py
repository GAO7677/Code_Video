#!/usr/bin/env python3
"""Refresh the shared case gallery while S-feature split jobs complete."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
GALLERY_BUILDER = SCRIPT_DIR / "build_head_role_dose_control_case_gallery.py"
PRIMARY_CONFIG = SCRIPT_DIR / "head_role_dose_control_pilot.json"
PRIMARY_S_FEATURE_CONFIG = SCRIPT_DIR / "head_role_s_feature_split_pilot.json"
S_FEATURE_UNION_CONFIG = SCRIPT_DIR / "head_role_s_feature_union_pilot.json"
S_FEATURE_PHASED_CONFIG = SCRIPT_DIR / "head_role_s_feature_phased_pilot.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["storage"]["output_root"]).expanduser().resolve()
    manifest = json.loads(
        Path(config["matched_subset_manifest"])
        .expanduser()
        .resolve()
        .read_text(encoding="utf-8")
    )
    expected = (
        len(config["models"])
        * len(config["seeds"])
        * len(config["step_ranges"])
        * len(manifest["subsets"])
    )
    last_signature: tuple[tuple[str, int], ...] | None = None
    while True:
        states = []
        for path in sorted((root / "state").glob("*.json")):
            try:
                states.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                states.append({"status": "invalid"})
        counts = Counter(str(state.get("status", "invalid")) for state in states)
        ready_videos = sum(
            1
            for video in (root / "generation").rglob("*.mp4")
            if video.stat().st_size > 1024
            and video.with_suffix(".json").is_file()
            and not video.with_suffix(".json.lock").exists()
        )
        signature = (*tuple(sorted(counts.items())), ("ready_videos", ready_videos))
        atomic_json(
            root / "progress.json",
            {
                "expected_tasks": expected,
                "state_counts": dict(counts),
                "ready_videos": ready_videos,
                "expected_videos": expected * int(config["expected_cases"]),
                "updated_utc": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                ),
            },
        )
        if signature != last_signature:
            subprocess.run(
                [
                    str(PYTHON),
                    str(GALLERY_BUILDER),
                    "--config",
                    str(PRIMARY_CONFIG),
                    "--s-feature-config",
                    str(PRIMARY_S_FEATURE_CONFIG),
                    "--s-feature-union-config",
                    str(S_FEATURE_UNION_CONFIG),
                    "--s-feature-phased-config",
                    str(S_FEATURE_PHASED_CONFIG),
                ],
                check=True,
            )
            print(
                f"[s-feature-gallery] {dict(counts)} / expected={expected}; "
                f"videos={ready_videos}/{expected * int(config['expected_cases'])}",
                flush=True,
            )
            last_signature = signature
        if counts.get("complete", 0) == expected:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
