from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

from code_vjepa_vggt.utils.video_io import read_video_uniform


DESCRIPTION_TO_PROMPT: dict[str, str] = {
    "球体落到平台后继续滚动并离开支撑面，测试支撑切换。": (
        "A sphere rolls after landing on the platform and leaves the support surface, testing support switching."
    ),
    "圆柱立在窄底座上并带轻微倾斜，开始后在重力作用下自然失稳倒伏。": (
        "A cylinder stands on a narrow base with a slight tilt, then topples naturally under gravity after the start."
    ),
    "扁平圆盘斜向滑入并撞击方块，观察速度传递和偏转。": (
        "A flat disc slides in diagonally and hits a block, observing velocity transfer and deflection."
    ),
    "卧放圆柱滚入后撞击立柱，观察碰撞后的平移与转动耦合。": (
        "A horizontally lying cylinder rolls in and collides with an upright pillar, observing translational and rotational coupling after impact."
    ),
    "两个球体从左右两侧入镜，在遮挡柱后交叉并重现，检查 identity 保持。": (
        "Two spheres enter from left and right, cross behind an occluding pillar, and reappear, checking identity preservation."
    ),
    "使用两个静态柱体做遮挡，动态球体从画外连续入镜并完成遮挡-重现。": (
        "Two static pillars create occlusion, while a moving sphere enters continuously from off-screen and undergoes occlusion and reappearance."
    ),
    "球体先撞第一个方块，再带动第二个方块，测试简单几何下的因果传播。": (
        "A sphere first hits the first block and then drives the second block, testing causal propagation in a simple geometry."
    ),
    "胶囊体推动方块，再由方块碰到圆柱，形成三体链式传播。": (
        "A capsule pushes a block, which then hits a cylinder, forming a three-body chain of propagation."
    ),
    "胶囊体以初速度入镜，在地面摩擦作用下滑行并伴随姿态变化。": (
        "A capsule enters with initial velocity, sliding under ground friction while changing orientation."
    ),
    "胶囊体带反向角速度与较强横向速度分量横扫画面，观察反向滚滑耦合。": (
        "A capsule sweeps across the frame with reverse angular velocity and a strong lateral velocity component, observing reverse roll-slide coupling."
    ),
    "胶囊体以较小角速度和中等线速度入镜，主要观察长距离滑行与缓慢转姿。": (
        "A capsule enters with small angular velocity and moderate linear velocity, mainly observing long-distance sliding and slow reorientation."
    ),
    "胶囊体以更大的初始倾角入镜，在滑行中发生明显翻滚和轴向姿态切换。": (
        "A capsule enters with a larger initial tilt, showing pronounced tumbling and axial orientation changes while sliding."
    ),
    "胶囊体以更高线速度但较低角速度快速掠过画面，观察平移主导的运动形态。": (
        "A capsule moves quickly across the frame with higher linear speed but lower angular speed, observing a translation-dominated motion pattern."
    ),
    "球体从画外连续入镜，在真实重力下落地后发生弹跳和滚动。": (
        "A sphere enters continuously from off-screen and bounces and rolls after landing under real gravity."
    ),
    "胶囊体带更大的横向分量和更高角速度斜向滑入，观察快速姿态耦合变化。": (
        "A capsule slides in diagonally with a larger lateral component and higher angular velocity, observing rapid pose-coupled changes."
    ),
}

NUM_SOURCE_FRAMES = 24
NUM_CONTEXT_FRAMES = 12


def _write_mp4(path: Path, frames_thwc_uint8, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg") or "/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg"
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for h264 encoding but was not found in PATH")

    height, width = int(frames_thwc_uint8.shape[1]), int(frames_thwc_uint8.shape[2])
    tmp_path = path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer for {tmp_path}")
    try:
        for frame in frames_thwc_uint8:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    import subprocess

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(tmp_path),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tmp_path.unlink(missing_ok=True)


def _update_meta(meta_path: Path, *, dry_run: bool) -> None:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    description = str(meta.get("description", ""))
    if description not in DESCRIPTION_TO_PROMPT:
        raise KeyError(f"missing English translation for description: {description}")

    sample_dir = meta_path.parent
    source_video = sample_dir / "video.mp4"
    if not source_video.exists():
        source_video = Path(str(meta["video"]))
    input_video_path = sample_dir / "context_video.mp4"

    if not dry_run:
        frames_thwc, _ = read_video_uniform(source_video, NUM_SOURCE_FRAMES)
        legacy_source_video_path = sample_dir / "source_video.mp4"
        legacy_source_video_path.unlink(missing_ok=True)
        _write_mp4(input_video_path, frames_thwc[:NUM_CONTEXT_FRAMES], fps=int(meta.get("fps", 30)))

    meta["source_video"] = str(source_video.resolve())
    meta["input_video"] = str(input_video_path.resolve())
    meta["source_num_frames"] = NUM_SOURCE_FRAMES
    meta["input_num_frames"] = NUM_CONTEXT_FRAMES
    meta["source_frame_indices"] = list(range(NUM_SOURCE_FRAMES))
    meta["input_frame_indices"] = list(range(NUM_CONTEXT_FRAMES))
    meta["input_prompt"] = DESCRIPTION_TO_PROMPT[description]

    if not dry_run:
        tmp_path = meta_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.write("\n")
        tmp_path.replace(meta_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    meta_paths = sorted(root.rglob("meta.json"))
    if not meta_paths:
        raise RuntimeError(f"no meta.json found under {root}")

    for meta_path in meta_paths:
        _update_meta(meta_path, dry_run=bool(args.dry_run))

    print(
        json.dumps(
            {
                "root": str(root),
                "num_samples": len(meta_paths),
                "source_video_mode": f"uniform_{NUM_SOURCE_FRAMES}_frames",
                "input_video_mode": f"prefix_{NUM_CONTEXT_FRAMES}_frames",
                "dry_run": bool(args.dry_run),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
