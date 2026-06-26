from __future__ import annotations

import argparse
from pathlib import Path

from .paths import A_OUTPUT, DATA_ROOT
from .records import load_payload, save_payload, set_wmreward
from .single_case.wmreward import score_case
from .wmreward_official import WMRewardRunner


DEFAULT_VIDEO_DIRS = [
    DATA_ROOT / "videos" / "ball_block",
    DATA_ROOT / "videos" / "jepa_sensitivity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch WMReward evaluation using the shared single-case entrypoint.")
    parser.add_argument(
        "--mode",
        choices=["pdibench", "sim", "custom"],
        default="sim",
        help="Which dataset layout to scan.",
    )
    parser.add_argument(
        "--video-dir",
        action="append",
        type=Path,
        default=None,
        help="Custom video directory to scan for *.mp4 and sibling *.json files. Can be repeated.",
    )
    parser.add_argument("--cuda-visible-devices", default="2")
    return parser.parse_args()


def iter_video_json_pairs(mode: str, custom_dirs: list[Path] | None) -> list[tuple[Path, Path]]:
    if mode == "pdibench":
        pairs: list[tuple[Path, Path]] = []
        for video_path in sorted(A_OUTPUT.rglob("*.mp4")):
            json_path = video_path.with_suffix(".json")
            if json_path.exists():
                pairs.append((video_path, json_path))
        return pairs

    if mode == "custom":
        if not custom_dirs:
            raise ValueError("--video-dir is required when --mode custom")
        roots = custom_dirs
    else:
        roots = DEFAULT_VIDEO_DIRS

    pairs = []
    for video_dir in roots:
        if not video_dir.exists():
            continue
        for video_path in sorted(video_dir.glob("*.mp4")):
            json_path = video_path.with_suffix(".json")
            if json_path.exists():
                pairs.append((video_path, json_path))
    return pairs


def main() -> None:
    args = parse_args()
    runner = WMRewardRunner(cuda_visible_devices=args.cuda_visible_devices)
    pairs = iter_video_json_pairs(args.mode, args.video_dir)
    print(f"Found {len(pairs)} video/json pairs", flush=True)

    for index, (video_path, json_path) in enumerate(pairs, start=1):
        print(f"[{index}/{len(pairs)}] {video_path}", end=" ", flush=True)
        payload = load_payload(json_path)
        result = score_case(video_path, runner=runner)
        set_wmreward(payload, result)
        save_payload(json_path, payload)
        print(
            f"surprise={result['surprise']:.6f} similarity={result['similarity']:.6f}",
            flush=True,
        )

    print("Done", flush=True)


if __name__ == "__main__":
    main()
