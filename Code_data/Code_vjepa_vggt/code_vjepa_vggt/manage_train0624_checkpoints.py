from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List / move / delete train0624 checkpoints with an explicit dry-run mode.",
    )
    parser.add_argument(
        "--source-dir",
        default="/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67",
        help="directory containing step_*.pt checkpoints",
    )
    parser.add_argument(
        "--archive-dir",
        default="/home/gaoya/AAA_train0624_checkpoint_archive",
        help="destination for move mode",
    )
    parser.add_argument(
        "--mode",
        choices=["list", "move", "delete"],
        default="list",
        help="operation to perform",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help="checkpoint steps to operate on, for example: 20 40 60 80",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned actions without modifying files",
    )
    return parser.parse_args()


def _checkpoint_path(source_dir: Path, step: int) -> Path:
    return source_dir / f"step_{step:07d}.pt"


def _list_checkpoints(source_dir: Path) -> list[Path]:
    return sorted(source_dir.glob("step_*.pt"))


def _describe(path: Path) -> str:
    if not path.exists():
        return f"MISSING {path}"
    size_gb = path.stat().st_size / (1024 ** 3)
    return f"{path}  {size_gb:.2f} GiB"


def main() -> None:
    args = _parse_args()
    source_dir = Path(args.source_dir)
    archive_dir = Path(args.archive_dir)

    if args.mode == "list":
        for path in _list_checkpoints(source_dir):
            print(_describe(path))
        return

    if args.steps is None or len(args.steps) == 0:
        raise SystemExit("--steps is required for move/delete modes")

    targets = [_checkpoint_path(source_dir, step) for step in args.steps]
    total_bytes = 0
    for path in targets:
        print(_describe(path))
        if path.exists():
            total_bytes += path.stat().st_size
    print(f"total_selected_gib={total_bytes / (1024 ** 3):.2f}")

    if args.dry_run:
        print("dry_run=true")
        return

    if args.mode == "move":
        archive_dir.mkdir(parents=True, exist_ok=True)
        for path in targets:
            if not path.exists():
                continue
            destination = archive_dir / path.name
            print(f"MOVE {path} -> {destination}")
            shutil.move(str(path), str(destination))
        return

    if args.mode == "delete":
        for path in targets:
            if not path.exists():
                continue
            print(f"DELETE {path}")
            path.unlink()
        return

    raise SystemExit(f"unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
