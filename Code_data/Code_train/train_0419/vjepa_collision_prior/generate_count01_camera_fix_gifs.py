#!/usr/bin/env python3
from __future__ import annotations

import json
import concurrent.futures
import subprocess
from pathlib import Path


ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/Code_train/train_0419/vjepa_collision_prior/count01_camera_fix_parallel"
)
FFMPEG = Path(
    "/data/gaoya/miniconda3/envs/wan/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)


def build_gif(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    palette = dst.with_suffix(".palette.png")
    vf = "fps=6,scale=320:-1:flags=lanczos"
    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-i",
            str(src),
            "-vf",
            f"{vf},palettegen=stats_mode=diff",
            str(palette),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-i",
            str(src),
            "-i",
            str(palette),
            "-lavfi",
            f"{vf} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3",
            "-loop",
            "0",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    palette.unlink(missing_ok=True)


def build_gif_task(task: tuple[str, str]) -> int:
    src, dst = task
    build_gif(Path(src), Path(dst))
    return 1


def main() -> None:
    reports = sorted(ROOT.glob("workers/device_*/**/repair_report.json"))
    tasks: list[tuple[str, str]] = []
    for report in reports:
        payload = json.loads(report.read_text(encoding="utf-8"))
        repairs = payload.get("repairs", [])
        if not repairs:
            continue
        item = repairs[0]
        sample = item["sample_name"]
        asset_dir = report.parent / "assets" / sample
        pairs = [
            (asset_dir / "before.mp4", ROOT / "gif_assets" / sample / "before.gif"),
            (asset_dir / "after.mp4", ROOT / "gif_assets" / sample / "after.gif"),
        ]
        for src, dst in pairs:
            if src.exists():
                tasks.append((str(src), str(dst)))
    total = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for count in executor.map(build_gif_task, tasks):
            total += int(count)
    print(f"generated_gifs={total}")


if __name__ == "__main__":
    main()
