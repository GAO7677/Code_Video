#!/usr/bin/env python3
"""Watch the newest complete 30-case validation checkpoints.

The watcher intentionally tracks only the newest checkpoint above the largest
step already registered for each target. This prevents a restarted watcher
from backfilling historical checkpoints while still keeping the page current.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterator


HERE = Path(__file__).resolve().parent
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
PROJECT_ROOT = HERE.parents[2]
DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
STEP_PATTERN = re.compile(r"^step-(\d+)$")
CHECKPOINT_FILES = ("checkpoint.safetensors", "training_state.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch newest latent-mask and CoTracker checkpoints for 30-case validation."
    )
    parser.add_argument("--watch-config", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Watcher:
    def __init__(self, settings: dict[str, Any], *, dry_run: bool) -> None:
        self.settings = settings
        self.dry_run = dry_run
        self.validation_config = Path(settings["validation_config"]).resolve()
        self.validation_root = Path(
            read_json(self.validation_config)["output_root"]
        ).resolve()
        self.status_path = self.validation_root / "latest_checkpoint_watch_status.json"
        self.log_path = self.validation_root / "watch_logs" / "latest_checkpoint_watch.log"
        self.lock_path = self.validation_root / "watch_locks" / "latest_checkpoint_watch.lock"
        self.minimum_age_seconds = int(
            settings.get("minimum_checkpoint_age_seconds", 120)
        )
        self.retry_cooldown_seconds = int(settings.get("retry_cooldown_seconds", 300))
        self.task_timeout_seconds = int(settings.get("task_timeout_seconds", 14_400))
        self.video_gpu = int(settings.get("video_gpu", 2))
        self.loss_gpu = int(settings.get("loss_gpu", 3))
        self.targets = list(settings["targets"])
        if self.video_gpu == 4 or self.loss_gpu == 4:
            raise ValueError("GPU4 is prohibited by workspace rules")
        if self.minimum_age_seconds < 0 or self.retry_cooldown_seconds < 0:
            raise ValueError("watch timing values must be non-negative")

    def log(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def checkpoint_ready(self, checkpoint: Path) -> bool:
        newest_mtime = 0.0
        for filename in CHECKPOINT_FILES:
            path = checkpoint / filename
            if not path.is_file() or path.stat().st_size <= 0:
                return False
            newest_mtime = max(newest_mtime, path.stat().st_mtime)
        return time.time() - newest_mtime >= self.minimum_age_seconds

    def latest_checkpoint(self, target: dict[str, Any]) -> tuple[int, Path] | None:
        root = Path(target["checkpoint_root"]).resolve()
        if not root.is_dir():
            return None
        candidates: list[tuple[int, Path]] = []
        for child in root.iterdir():
            match = STEP_PATTERN.match(child.name)
            if not match or not child.is_dir() or not self.checkpoint_ready(child):
                continue
            candidates.append((int(match.group(1)), child.resolve()))
        return max(candidates, default=None, key=lambda item: item[0])

    @staticmethod
    def entry_id(target: dict[str, Any], step: int) -> str:
        return f"{target['entry_id_prefix']}_step{int(step):04d}"

    def build_entry(
        self,
        target: dict[str, Any],
        step: int,
        checkpoint: Path,
    ) -> dict[str, Any]:
        entry = {
            "entry_id": self.entry_id(target, step),
            "method_key": target["method_key"],
            "method_label": target["method_label"],
            "version": target["version"],
            "step": int(step),
            "checkpoint": str(checkpoint),
            "config": str(Path(target["config"]).resolve()),
            "color": target["color"],
        }
        source_config = target.get("source_config")
        if source_config:
            entry["source_config"] = str(Path(source_config).resolve())
        return entry

    def ensure_entry(
        self,
        target: dict[str, Any],
        step: int,
        checkpoint: Path,
    ) -> tuple[dict[str, Any], bool]:
        config_lock = self.validation_config.with_suffix(
            self.validation_config.suffix + ".watch.lock"
        )
        with exclusive_lock(config_lock) as acquired:
            if not acquired:
                raise RuntimeError(f"validation config is locked: {self.validation_config}")
            config = read_json(self.validation_config)
            entry_id = self.entry_id(target, step)
            existing = next(
                (entry for entry in config["entries"] if entry["entry_id"] == entry_id),
                None,
            )
            if existing is not None:
                return existing, False
            entry = self.build_entry(target, step, checkpoint)
            config["entries"].append(entry)
            if not self.dry_run:
                atomic_json(self.validation_config, config)
            return entry, True

    @staticmethod
    def completed_loss(root: Path, entry_id: str, total_cases: int) -> bool:
        path = root / "losses" / f"{entry_id}.json"
        if not path.is_file():
            return False
        payload = read_json(path)
        return (
            payload.get("state") == "complete"
            and len(payload.get("cases", [])) == total_cases
        )

    @staticmethod
    def completed_videos(
        root: Path,
        entry_id: str,
        cases: list[dict[str, Any]],
    ) -> bool:
        video_root = root / "videos" / entry_id
        if not video_root.is_dir():
            return False
        for case in cases:
            candidates = list(video_root.glob(f"{case['case_id']}*.mp4"))
            candidates.extend(video_root.glob(f"*/{case['case_id']}*.mp4"))
            if not any(path.is_file() and path.stat().st_size > 0 for path in candidates):
                return False
        return True

    def entry_complete(
        self,
        entry_id: str,
        cases: list[dict[str, Any]],
    ) -> bool:
        return self.completed_videos(
            self.validation_root, entry_id, cases
        ) and self.completed_loss(self.validation_root, entry_id, len(cases))

    def target_entries(
        self,
        config: dict[str, Any],
        target: dict[str, Any],
    ) -> list[dict[str, Any]]:
        prefix = f"{target['entry_id_prefix']}_step"
        return [
            entry
            for entry in config["entries"]
            if entry.get("method_key") == target["method_key"]
            and str(entry.get("entry_id", "")).startswith(prefix)
        ]

    def candidate_for_target(
        self,
        target: dict[str, Any],
        config: dict[str, Any],
        cases: list[dict[str, Any]],
        previous_target_state: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        latest = self.latest_checkpoint(target)
        target_state: dict[str, Any] = {
            "checkpoint_root": str(Path(target["checkpoint_root"]).resolve()),
            "state": "waiting_for_checkpoint",
        }
        if latest is None:
            return None, target_state
        step, checkpoint = latest
        target_state.update(
            latest_step=step,
            latest_checkpoint=str(checkpoint),
        )
        entries = self.target_entries(config, target)
        existing_by_step = {int(entry["step"]): entry for entry in entries}
        max_registered_step = max(existing_by_step, default=None)
        target_state["max_registered_step"] = max_registered_step
        existing = existing_by_step.get(step)
        if existing is not None and self.entry_complete(existing["entry_id"], cases):
            target_state["state"] = "up_to_date"
            return None, target_state
        if existing is None and max_registered_step is not None and step < max_registered_step:
            target_state["state"] = "latest_step_already_superseded"
            return None, target_state
        last_failed_step = previous_target_state.get("last_failed_step")
        last_failed_at = float(previous_target_state.get("last_failed_at", 0.0))
        if (
            last_failed_step == step
            and time.time() - last_failed_at < self.retry_cooldown_seconds
        ):
            target_state["state"] = "retry_cooldown"
            target_state["retry_after_seconds"] = int(
                self.retry_cooldown_seconds - (time.time() - last_failed_at)
            )
            return None, target_state
        target_state["state"] = "pending_validation"
        return {
            "target": target,
            "step": step,
            "checkpoint": checkpoint,
            "existing_entry": existing,
        }, target_state

    def build_page(self) -> None:
        command = [
            str(PYTHON),
            str(HERE / "build_validation_30cases_hub.py"),
            "--config",
            str(self.validation_config),
        ]
        process = subprocess.run(
            command,
            cwd=HERE,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            raise RuntimeError(
                "page build failed: " + (process.stderr or process.stdout).strip()
            )

    def run_validation(self, entry: dict[str, Any]) -> tuple[int, int]:
        entry_id = entry["entry_id"]
        log_dir = self.validation_root / "watch_logs" / entry_id
        log_dir.mkdir(parents=True, exist_ok=True)
        common_env = {
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": f"{PROJECT_ROOT}:{DIFFSYNTH_ROOT}",
        }
        commands = {
            "video": (
                [
                    str(PYTHON),
                    "-u",
                    str(HERE / "run_validation_30cases.py"),
                    "--config",
                    str(self.validation_config),
                    "--gpu",
                    str(self.video_gpu),
                    "--entry-id",
                    entry_id,
                    "--case-count",
                    "30",
                ],
                self.video_gpu,
            ),
            "loss": (
                [
                    str(PYTHON),
                    "-u",
                    str(HERE / "run_validation_30_loss.py"),
                    "--config",
                    str(self.validation_config),
                    "--gpu",
                    str(self.loss_gpu),
                    "--entry-id",
                    entry_id,
                ],
                self.loss_gpu,
            ),
        }
        processes: dict[str, subprocess.Popen[str]] = {}
        handles = []
        try:
            for name, (command, gpu) in commands.items():
                handle = (log_dir / f"{name}.log").open("a", encoding="utf-8")
                handles.append(handle)
                processes[name] = subprocess.Popen(
                    command,
                    cwd=HERE,
                    env={**common_env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            deadline = time.monotonic() + self.task_timeout_seconds
            return_codes: dict[str, int] = {}
            for name, process in processes.items():
                timeout = max(1.0, deadline - time.monotonic())
                try:
                    return_codes[name] = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        return_codes[name] = process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        return_codes[name] = process.wait()
                    self.log(f"timed out entry={entry_id} stage={name}")
            return return_codes["video"], return_codes["loss"]
        finally:
            for handle in handles:
                handle.close()

    def write_status(self, status: dict[str, Any]) -> None:
        status["updated_utc"] = utc_now()
        if not self.dry_run:
            atomic_json(self.status_path, status)

    def run_scan(self, status: dict[str, Any]) -> None:
        config = read_json(self.validation_config)
        manifest = read_json(Path(config["cases_manifest"]))
        cases = list(manifest.get("cases", []))
        if len(cases) != 30:
            raise ValueError(f"Expected 30 fixed validation cases, got {len(cases)}")
        previous_targets = dict(status.get("targets", {}))
        target_states: dict[str, Any] = {}
        pending: list[dict[str, Any]] = []
        for target in self.targets:
            key = str(target["method_key"])
            task, target_state = self.candidate_for_target(
                target, config, cases, previous_targets.get(key, {})
            )
            target_states[key] = target_state
            if task is not None:
                pending.append(task)
        status["targets"] = target_states
        status["current_task"] = None
        self.write_status(status)
        if not pending:
            return
        task = pending[0]
        target = task["target"]
        step = int(task["step"])
        if self.dry_run:
            self.log(
                f"dry-run pending method={target['method_key']} step={step} "
                f"checkpoint={task['checkpoint']}"
            )
            return
        entry, added = self.ensure_entry(target, step, task["checkpoint"])
        if added:
            self.log(f"registered entry={entry['entry_id']}")
        else:
            self.log(f"resuming entry={entry['entry_id']}")
        status["current_task"] = {
            "entry_id": entry["entry_id"],
            "method_key": entry["method_key"],
            "step": step,
            "state": "running",
            "started_utc": utc_now(),
        }
        target_states[str(target["method_key"])]["state"] = "running_validation"
        self.write_status(status)
        self.build_page()
        self.log(
            f"start validation entry={entry['entry_id']} "
            f"video_gpu={self.video_gpu} loss_gpu={self.loss_gpu}"
        )
        video_code, loss_code = self.run_validation(entry)
        complete = self.entry_complete(entry["entry_id"], cases)
        target_state = target_states[str(target["method_key"])]
        target_state["last_attempt_step"] = step
        target_state["last_attempt_utc"] = utc_now()
        target_state["video_return_code"] = video_code
        target_state["loss_return_code"] = loss_code
        target_state["state"] = "complete" if complete else "failed"
        if not complete:
            target_state["last_failed_step"] = step
            target_state["last_failed_at"] = time.time()
        status["current_task"] = None
        self.write_status(status)
        self.build_page()
        self.log(
            f"finish validation entry={entry['entry_id']} complete={complete} "
            f"video_rc={video_code} loss_rc={loss_code}"
        )

    def run(self, *, poll_seconds: int, once: bool) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll-seconds must be positive")
        with exclusive_lock(self.lock_path) as acquired:
            if not acquired:
                raise RuntimeError(f"watcher already active: {self.lock_path}")
            previous = read_json(self.status_path) if self.status_path.is_file() else {}
            status: dict[str, Any] = {
                **previous,
                "schema_version": 1,
                "state": "dry_run" if self.dry_run else "running",
                "watch_config": str(Path(self.settings["_path"]).resolve()),
                "validation_config": str(self.validation_config),
                "poll_seconds": poll_seconds,
                "video_gpu": self.video_gpu,
                "loss_gpu": self.loss_gpu,
                "started_utc": previous.get("started_utc", utc_now()),
            }
            self.write_status(status)
            self.log(f"watcher started once={once} dry_run={self.dry_run}")
            try:
                while True:
                    self.run_scan(status)
                    if once:
                        break
                    time.sleep(poll_seconds)
            except KeyboardInterrupt:
                self.log("watcher interrupted")
            finally:
                if not self.dry_run:
                    status["state"] = "stopped"
                    status["current_task"] = None
                    self.write_status(status)


def main() -> None:
    args = parse_args()
    settings = read_json(args.watch_config.resolve())
    settings["_path"] = str(args.watch_config.resolve())
    watcher = Watcher(settings, dry_run=args.dry_run)
    poll_seconds = (
        args.poll_seconds
        if args.poll_seconds is not None
        else int(settings.get("poll_seconds", 60))
    )
    watcher.run(poll_seconds=poll_seconds, once=args.once)


if __name__ == "__main__":
    main()
