from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2


DEFAULT_SUMMARY = "/data/gaoya/agent-data/outputs/depth_anything_test_5_sources/summary.json"
DEFAULT_OUTPUT_ROOT = "/data/gaoya/agent-data/outputs/depth_gt_compare_test_5"
DEFAULT_COMPARE_SCRIPT = "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/compare_depth_gt_branches.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--compare-script", default=DEFAULT_COMPARE_SCRIPT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sample_npz_from_json_name(json_name: str) -> Path | None:
    if not json_name.startswith("0613pybullet_sample_") or not json_name.endswith(".json"):
        return None
    sample_stub = json_name[len("0613pybullet_") : -len(".json")]
    candidate = Path(
        "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/"
        "industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500/val"
    ) / f"{sample_stub}.npz"
    return candidate if candidate.is_file() else None


def _read_video_rgb(path: Path) -> tuple[list, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    return frames, fps


def _write_h264_mp4(path: Path, frames_rgb: list, fps: float) -> None:
    ffmpeg = (
        subprocess.check_output(
            [
                "/home/gaoya/miniconda3/envs/wan-cu128/bin/python",
                "-c",
                "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())",
            ],
            text=True,
        )
        .strip()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames_rgb[0].shape[0]), int(frames_rgb[0].shape[1])
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{float(fps):.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None
    try:
        for frame in frames_rgb:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed for {path} with code {ret}")
    finally:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()


def _num_context_frames_from_npz(sample_npz: Path) -> int:
    import numpy as np

    with np.load(sample_npz) as data:
        return int(data["context_frames"].shape[0])


def main() -> None:
    args = _parse_args()
    summary_path = Path(args.summary_json).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    compare_script = Path(args.compare_script).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records = json.loads(summary_path.read_text(encoding="utf-8"))
    results = []
    for record in records:
        json_path = Path(record["json_path"])
        depth_video = Path(record["output_video"]).expanduser().resolve()
        raw_context_video = Path(record["source_video"]).expanduser().resolve()
        sample_npz = _sample_npz_from_json_name(json_path.name)
        out_dir = output_root / json_path.stem
        if sample_npz is None:
            results.append(
                {
                    "json_path": str(json_path),
                    "status": "skipped_no_training_npz",
                    "reason": "no episodes_v1 npz with context_states/context_boxes was found for this json naming scheme",
                    "depth_video": str(depth_video),
                }
            )
            continue
        panel_video = out_dir / f"{sample_npz.stem}__state_vs_depth_anything_panel.mp4"
        results.append(
            {
                "json_path": str(json_path),
                "status": "ran" if (not panel_video.is_file() or args.overwrite) else "skipped_existing",
                "sample_npz": str(sample_npz),
                "raw_context_video": str(raw_context_video),
                "depth_video": str(depth_video),
                "output_dir": str(out_dir),
            }
        )
        if panel_video.is_file() and not args.overwrite:
            continue
        num_context_frames = _num_context_frames_from_npz(sample_npz)
        depth_frames, depth_fps = _read_video_rgb(depth_video)
        if len(depth_frames) < num_context_frames:
            raise RuntimeError(
                f"depth video {depth_video} only has {len(depth_frames)} frames, "
                f"smaller than required context_frames={num_context_frames}"
            )
        depth_prefix_video = out_dir / f"{sample_npz.stem}__depth_prefix_{num_context_frames:02d}f.mp4"
        if not depth_prefix_video.is_file() or args.overwrite:
            _write_h264_mp4(depth_prefix_video, depth_frames[:num_context_frames], fps=depth_fps)
        cmd = [
            "/home/gaoya/miniconda3/envs/wan-cu128/bin/python",
            str(compare_script),
            "--sample-npz",
            str(sample_npz),
            "--raw-context-video",
            str(raw_context_video),
            "--depth-video",
            str(depth_prefix_video),
            "--output-dir",
            str(out_dir),
        ]
        subprocess.run(cmd, check=True)

    final_summary = output_root / "batch_summary.json"
    with open(final_summary, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[done] summary={final_summary}")


if __name__ == "__main__":
    main()
