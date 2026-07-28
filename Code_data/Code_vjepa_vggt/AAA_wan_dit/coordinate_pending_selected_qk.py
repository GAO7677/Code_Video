#!/usr/bin/env python3
"""Coordinate final role selection, pending Q@K workers, and gallery refreshes."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--gallery-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    return parser.parse_args()


def _status(path: Path) -> str:
    if not path.is_file():
        return "missing"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status", "missing"))
    except json.JSONDecodeError:
        return "invalid"


def _run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[coordinator] command: {' '.join(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def _refresh_gallery(
    *,
    config_path: Path,
    snapshot_path: Path,
    gallery_dir: Path,
    log: Path,
) -> None:
    _run(
        [
            str(PYTHON),
            str(SCRIPT_DIR / "build_multiseed_qk_gallery.py"),
            "--config",
            str(config_path),
            "--snapshot",
            str(snapshot_path),
            "--output-dir",
            str(gallery_dir),
        ],
        log,
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    snapshot_path = args.snapshot.expanduser().resolve()
    gallery_dir = args.gallery_dir.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    root = Path(config["storage"]["output_root"]).expanduser().resolve()
    pending_root = root / "pending_selected_qk"
    logs = pending_root / "logs"
    ready = pending_root / "READY.json"
    compact_tasks = [
        (str(model), int(seed))
        for model in config["models"]
        for seed in config["seeds"]
    ]
    qk_tasks = [
        (str(model), int(item["seed"]))
        for model, rows in snapshot["pending"].items()
        for item in rows
    ]
    _refresh_gallery(
        config_path=config_path,
        snapshot_path=snapshot_path,
        gallery_dir=gallery_dir,
        log=logs / "gallery.log",
    )
    while not ready.is_file():
        compact_counts = {}
        for model, seed in compact_tasks:
            status = _status(root / "state" / model / f"seed-{seed:06d}.json")
            compact_counts[status] = compact_counts.get(status, 0) + 1
        print(
            f"[coordinator] compact={compact_counts}/{len(compact_tasks)}",
            flush=True,
        )
        if compact_counts.get("complete", 0) != len(compact_tasks):
            time.sleep(int(args.poll_seconds))
            _refresh_gallery(
                config_path=config_path,
                snapshot_path=snapshot_path,
                gallery_dir=gallery_dir,
                log=logs / "gallery.log",
            )
            continue

        analysis = root / "analysis"
        _run(
            [
                str(PYTHON),
                str(SCRIPT_DIR / "analyze_fulltoken_head_roles_batch.py"),
                "--config",
                str(config_path),
            ],
            logs / "final_analysis.log",
        )
        selection = pending_root / "fixed_role_selection.json"
        _run(
            [
                str(PYTHON),
                str(SCRIPT_DIR / "build_fixed_role_qk_selection.py"),
                "--head-role-report",
                str(analysis / "head_role_report.json"),
                "--input-list",
                str(root / "input_lists" / "test5_unique20.txt"),
                "--output",
                str(selection),
            ],
            logs / "selection.log",
        )
        ready.write_text(
            json.dumps(
                {
                    "created_at_unix": time.time(),
                    "selection": str(selection),
                    "pending_jobs": len(qk_tasks),
                    "output_bins": 512,
                    "softmax_only_rendering": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[coordinator] QK queue ready: {ready}", flush=True)

    while True:
        qk_counts = {}
        for model, seed in qk_tasks:
            status = _status(
                pending_root / "state" / model / f"seed-{seed:06d}.json"
            )
            qk_counts[status] = qk_counts.get(status, 0) + 1
        print(f"[coordinator] pending-QK={qk_counts}/{len(qk_tasks)}", flush=True)
        _refresh_gallery(
            config_path=config_path,
            snapshot_path=snapshot_path,
            gallery_dir=gallery_dir,
            log=logs / "gallery.log",
        )
        if qk_counts.get("complete", 0) == len(qk_tasks):
            done = pending_root / "DONE.json"
            done.write_text(
                json.dumps(
                    {
                        "completed_at_unix": time.time(),
                        "jobs": len(qk_tasks),
                        "gallery": str(gallery_dir),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"[coordinator] complete: {done}", flush=True)
            return
        time.sleep(int(args.poll_seconds))


if __name__ == "__main__":
    main()
