#!/usr/bin/env python3
"""Pull complete SSH118 tasks, validate them locally, and rewrite state paths."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

from run_head_role_dose_control_pilot_worker import (
    _atomic_json,
    _input_cases,
    _sha256,
    _validate_job,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="118")
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--local-config", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def rsync(source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--partial",
            "--protect-args",
            source,
            str(destination),
        ],
        check=True,
    )


def job_root(
    root: Path,
    model: str,
    seed: int,
    subset_id: str,
    start: int,
    end: int,
) -> Path:
    variant = f"{subset_id}_steps{start:02d}_{end:02d}"
    return root / "generation" / model / f"seed-{seed:06d}" / variant


def main() -> None:
    args = parse_args()
    config = json.loads(
        args.local_config.expanduser().resolve().read_text(encoding="utf-8")
    )
    local_root = Path(config["storage"]["output_root"]).expanduser().resolve()
    remote_root = args.remote_root
    manifest = Path(config["matched_subset_manifest"]).expanduser().resolve()
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_sha256 = _sha256(manifest)
    cases = _input_cases(Path(config["input_list"]).expanduser().resolve())
    expected = (
        len(config["models"])
        * len(config["seeds"])
        * len(config["step_ranges"])
        * len(manifest_payload["subsets"])
    )
    mirror = local_root / ".ssh118_state_mirror"

    while True:
        try:
            rsync(f"{args.host}:{remote_root}/state/", mirror / "state")
            rsync(f"{args.host}:{remote_root}/logs/", mirror / "logs")
        except subprocess.CalledProcessError as error:
            print(f"[ssh118-pull] state sync failed: {error}; retrying", flush=True)
            time.sleep(args.poll_seconds)
            continue

        remote_states = []
        for path in sorted((mirror / "state").glob("*.json")):
            try:
                remote_states.append(
                    (path, json.loads(path.read_text(encoding="utf-8")))
                )
            except json.JSONDecodeError:
                continue
        counts = Counter(str(state.get("status", "invalid")) for _, state in remote_states)
        copied = 0
        for state_path, state in remote_states:
            if state.get("status") != "complete":
                continue
            local_state = local_root / "state" / state_path.name
            if local_state.is_file():
                try:
                    if json.loads(local_state.read_text(encoding="utf-8")).get(
                        "status"
                    ) == "complete":
                        continue
                except json.JSONDecodeError:
                    pass

            model = str(state["model"])
            seed = int(state["seed"])
            subset_id = str(state["subset_id"])
            start, end = (int(value) for value in state["step_range"])
            remote_job = job_root(
                remote_root, model, seed, subset_id, start, end
            )
            local_job = job_root(
                local_root, model, seed, subset_id, start, end
            )
            rsync(f"{args.host}:{remote_job}/", local_job)
            k = int(manifest_payload["subsets"][subset_id]["k"])
            videos = _validate_job(
                local_job,
                cases=cases,
                subset_id=subset_id,
                manifest_sha256=manifest_sha256,
                k=k,
                start=start,
                end=end,
            )
            rewritten = {
                **state,
                "source_host": args.host,
                "source_output_root": str(remote_root),
                "local_output_root": str(local_root),
                "videos": videos,
            }
            _atomic_json(local_state, rewritten)
            copied += 1
            print(
                f"[ssh118-pull] validated and installed {state['task_id']}",
                flush=True,
            )

        complete_local = 0
        for path in (local_root / "state").glob("*.json"):
            try:
                complete_local += (
                    json.loads(path.read_text(encoding="utf-8")).get("status")
                    == "complete"
                )
            except json.JSONDecodeError:
                pass
        print(
            f"[ssh118-pull] remote={dict(counts)} "
            f"local_complete={complete_local}/{expected} newly_copied={copied}",
            flush=True,
        )
        if complete_local == expected:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
