#!/usr/bin/env python3
"""Watch a Scheme-D run and audit every newly completed checkpoint."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from audit_checkpoint import atomic_write_json, audit_checkpoint


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("step-"))
    except ValueError:
        return -1


def discover_checkpoints(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.glob("step-*")
            if checkpoint_step(path) >= 0
            and (path / "checkpoint.safetensors").is_file()
        ),
        key=checkpoint_step,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.poll_seconds <= 0.0:
        raise ValueError("poll-seconds must be positive")
    args.checkpoint_root = args.checkpoint_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audited_steps: set[int] = set()
    for output in args.output_dir.glob("step-*.json"):
        step = checkpoint_step(Path(output.stem))
        if step >= 0:
            audited_steps.add(step)

    while True:
        checkpoints = discover_checkpoints(args.checkpoint_root)
        for index, checkpoint in enumerate(checkpoints):
            step = checkpoint_step(checkpoint)
            if step in audited_steps:
                continue
            previous = checkpoints[index - 1] if index > 0 else None
            try:
                payload = audit_checkpoint(checkpoint, compare=previous)
            except Exception as exc:
                payload = {
                    "status": "failed",
                    "checkpoint": str(checkpoint),
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            payload["step"] = step
            payload["audited_at_unix"] = time.time()
            output_path = args.output_dir / f"step-{step:06d}.json"
            atomic_write_json(output_path, payload)
            audited_steps.add(step)
            print(
                f"step={step} status={payload['status']} output={output_path}",
                flush=True,
            )
        if (
            args.max_step is not None
            and any(step >= args.max_step for step in audited_steps)
        ):
            break
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
