from __future__ import annotations

import argparse
import http.server
import json
import shutil
import socketserver
from pathlib import Path

import cv2
import numpy as np
import torch

from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset


def _video_cthw_to_uint8_thwc(video_cthw: torch.Tensor) -> np.ndarray:
    video = video_cthw.detach().cpu().clamp(-1.0, 1.0)
    video = ((video + 1.0) * 127.5).to(torch.uint8)
    return video.permute(1, 2, 3, 0).contiguous().numpy()


def _write_mp4(path: Path, frames_thwc_uint8: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg") or "/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg"
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for h264 encoding but was not found")

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


def render_case_block(sample: dict, case_id: int) -> str:
    payload = {
        "caption": sample["caption"],
        "video_shape": list(sample["video"].shape),
        "context_video_shape": list(sample["context_video"].shape),
        "video_path": sample["video_path"],
        "frame_indices": sample["frame_indices"].tolist(),
        "context_frame_indices": sample["context_frame_indices"].tolist(),
        "num_context_frames": sample["num_context_frames"],
        "metadata": sample["metadata"],
    }
    return f"""
  <section class="case">
    <h2>Case {case_id}</h2>
    <p><b>Caption:</b> {sample["caption"]}</p>
    <p><b>Video path:</b> {sample["video_path"]}</p>
    <p><b>Full video shape:</b> {list(sample["video"].shape)}</p>
    <p><b>Context video shape:</b> {list(sample["context_video"].shape)}</p>
    <p><b>Context frame indices:</b> {sample["context_frame_indices"].tolist()}</p>
    <div class="grid">
      <figure>
        <video controls playsinline preload="metadata" src="./{sample["full_video_rel"]}"></video>
        <figcaption>Full video</figcaption>
      </figure>
      <figure>
        <video controls playsinline preload="metadata" src="./{sample["context_video_rel"]}"></video>
        <figcaption>Context video</figcaption>
      </figure>
    </div>
    <pre>{json.dumps(payload, indent=2, ensure_ascii=False)}</pre>
  </section>
"""


def build_report(samples: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    blocks = []
    for case_id, sample in enumerate(samples):
        full_rel = f"case_{case_id:03d}_full.mp4"
        context_rel = f"case_{case_id:03d}_context.mp4"
        fps = int(sample["metadata"].get("fps", 30))
        _write_mp4(output_dir / full_rel, _video_cthw_to_uint8_thwc(sample["video"]), fps=fps)
        _write_mp4(output_dir / context_rel, _video_cthw_to_uint8_thwc(sample["context_video"]), fps=fps)

        sample = dict(sample)
        sample["full_video_rel"] = full_rel
        sample["context_video_rel"] = context_rel

        summary.append(
            {
                "case_id": case_id,
                "video_path": sample["video_path"],
                "caption": sample["caption"],
                "context_frame_indices": sample["context_frame_indices"].tolist(),
                "full_video_rel": full_rel,
                "context_video_rel": context_rel,
            }
        )
        blocks.append(render_case_block(sample, case_id))

    with open(output_dir / "shape_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Training Input Viewer</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #f6f4ee; color: #222; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    .case {{ margin-bottom: 44px; padding-bottom: 24px; border-bottom: 1px solid #ddd; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 16px; }}
    video {{ width: 100%; border: 1px solid #ccc; background: #000; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 12px; color: #444; margin-top: 4px; }}
    pre {{ background: #fff; border: 1px solid #ddd; padding: 16px; white-space: pre-wrap; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>PhysState Training Input Viewer</h1>
  <p>这里展示的是训练脚本实际送入 WAN 的输入：`video` 是整段 episode 窗口，`context_video` 是从第 0 帧开始的前缀 context。当前页面使用 mp4 播放。</p>
  {"".join(blocks)}
</body>
</html>
"""
    html_path = output_dir / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--num-context-frames", type=int, default=12)
    parser.add_argument("--context-fraction", type=float, default=0.5)
    parser.add_argument("--random-context-frames", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="/data/gaoya/AAA_test_video/0529/vjepa_vggt/tmp/train_input_viewer_mp4",
    )
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--resolution", type=int, nargs=2, default=[704, 1280])
    args = parser.parse_args()

    dataset = PhysStateEpisodeDataset(
        root=args.root,
        split=args.split,
        resolution=tuple(args.resolution),
        num_context_frames=args.num_context_frames,
        context_fraction=args.context_fraction,
        random_context_frames=bool(args.random_context_frames),
    )
    samples = [dataset[i] for i in range(args.start_index, min(len(dataset), args.start_index + args.num_cases))]
    output_dir = Path(args.output_dir)
    html_path = build_report(samples, output_dir)
    print(f"training input report: {html_path}")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(output_dir), **handler_kwargs)

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("0.0.0.0", args.port), Handler) as httpd:
        print(f"serving report at http://0.0.0.0:{args.port}/index.html")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
