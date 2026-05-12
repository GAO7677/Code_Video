#!/usr/bin/env python3
# 用途：批量生成 counterfactual 画廊所需 GIF。
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import subprocess
import sys
from pathlib import Path


FFMPEG = Path(
    "/data/gaoya/miniconda3/envs/wan/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
GALLERY_SCRIPT = Path(
    "/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/build_counterfactual_rgb_gallery.py"
)


def load_gallery_module():
    spec = importlib.util.spec_from_file_location("cf_gallery", GALLERY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_gif(src: Path, dst: Path, width: int, fps: int) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    palette = dst.with_suffix(".palette.png")
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(
        [str(FFMPEG), "-y", "-i", str(src), "-vf", f"{vf},palettegen=stats_mode=diff", str(palette)],
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


def build_gif_task(task: tuple[str, str, int, int]) -> tuple[int, str]:
    src_s, dst_s, width, fps = task
    src = Path(src_s)
    dst = Path(dst_s)
    try:
        build_gif(src, dst, width=width, fps=fps)
        return 1, ""
    except Exception as exc:
        return 0, f"{src} :: {type(exc).__name__}: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GIF previews for the counterfactual gallery.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid"),
    )
    parser.add_argument(
        "--portal-root",
        type=Path,
        default=Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419/counterfactual_gallery_portal"),
    )
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = load_gallery_module()
    groups = module.build_groups(args.dataset_root)
    tasks: list[tuple[str, str, int, int]] = []
    for group in groups:
        for item in group["items"]:
            src = args.dataset_root / item["rgb_video"]
            dst = args.portal_root / item["gif_preview"]
            if src.exists():
                tasks.append((str(src), str(dst), int(args.width), int(args.fps)))
                if args.limit > 0 and len(tasks) >= args.limit:
                    break
        if args.limit > 0 and len(tasks) >= args.limit:
            break
    total = 0
    failures: list[str] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
        for count, error in executor.map(build_gif_task, tasks):
            total += int(count)
            if error:
                failures.append(error)
    print(f"generated_gifs={total}")
    print(f"failed_gifs={len(failures)}")
    if failures:
        fail_path = args.portal_root / "gif_generation_failures.txt"
        fail_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"failure_log={fail_path}")


if __name__ == "__main__":
    main()
