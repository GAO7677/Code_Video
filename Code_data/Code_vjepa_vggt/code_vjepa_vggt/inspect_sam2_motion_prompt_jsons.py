from __future__ import annotations

import argparse
import http.server
import json
import shutil
import socketserver
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from code_vjepa_vggt.adapters.sam2_motion import SAM2MotionTracker, build_motion_prompt_box


PROMPT_COLOR = (255, 140, 0)
FINAL_PROMPT_COLOR = (255, 0, 255)
SAM2_BOX_COLOR = (32, 160, 96)
MASK_COLOR = np.array([32, 160, 96], dtype=np.uint8)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 0, 0)


def read_video_cv2(video_path: Path, *, max_frames: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if max_frames > 0 and len(frames) >= int(max_frames):
                break
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"decoded zero frames from video: {video_path}")
    return np.stack(frames, axis=0)


def write_mp4(path: Path, frames_thwc_uint8: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames_thwc_uint8.shape[1]), int(frames_thwc_uint8.shape[2])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    try:
        for frame in frames_thwc_uint8:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def find_ffmpeg() -> str:
    candidates = [
        shutil.which("ffmpeg"),
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg",
        "/data/gaoya/miniconda3/envs/wan/bin/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError("ffmpeg not found")


def ensure_browser_video(source_path: Path) -> Path:
    out_path = source_path.with_name(f"{source_path.stem}.browser.mp4")
    if out_path.exists() and out_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns and out_path.stat().st_size > 0:
        return out_path
    ffmpeg = find_ffmpeg()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source_path),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_path


def draw_box_rgb(image: np.ndarray, box_xyxy: np.ndarray, color_rgb: tuple[int, int, int], label: str) -> None:
    x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy.tolist()]
    if x1 <= x0 or y1 <= y0:
        return
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.rectangle(image, (x0, y0), (x1, y1), color_bgr, 2)
    cv2.putText(image, label, (x0 + 2, max(y0 + 14, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)


def draw_text_panel(image: np.ndarray, lines: list[str]) -> None:
    x = 8
    y = 22
    line_h = 20
    panel_w = max(220, min(image.shape[1] - 16, max((len(line) for line in lines), default=0) * 8 + 16))
    panel_h = line_h * len(lines) + 8
    cv2.rectangle(image, (4, 4), (4 + panel_w, 4 + panel_h), TEXT_BG, thickness=-1)
    for line in lines:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
        y += line_h


def render_overlay_video(
    *,
    frames_thwc: np.ndarray,
    case_key: str,
    input_prompt: str,
    source_video: str,
    prompt_frame_idx: int,
    motion_prompt_box_xyxy: np.ndarray,
    final_prompt_box_xyxy: np.ndarray,
    sam_masks_thw: np.ndarray,
    sam_boxes_t4: np.ndarray,
) -> np.ndarray:
    rendered = []
    for frame_idx in range(frames_thwc.shape[0]):
        frame = frames_thwc[frame_idx].copy()
        mask = sam_masks_thw[frame_idx] > 0
        if mask.any():
            frame[mask] = ((0.60 * frame[mask]) + (0.40 * MASK_COLOR[None, :])).astype(np.uint8)

        draw_box_rgb(frame, sam_boxes_t4[frame_idx].astype(np.float32), SAM2_BOX_COLOR, "sam2_box")
        if frame_idx == int(prompt_frame_idx):
            draw_box_rgb(frame, motion_prompt_box_xyxy.astype(np.float32), PROMPT_COLOR, "motion_prompt")
            draw_box_rgb(frame, final_prompt_box_xyxy.astype(np.float32), FINAL_PROMPT_COLOR, "final_prompt")

        draw_text_panel(
            frame,
            [
                f"case: {case_key}",
                f"frame: {frame_idx}/{frames_thwc.shape[0] - 1}",
                f"prompt_frame: {prompt_frame_idx}",
                f"source: {Path(source_video).name}",
                f"prompt: {input_prompt[:70]}",
            ],
        )
        rendered.append(frame)
    return np.stack(rendered, axis=0)


def save_prompt_preview(
    *,
    output_path: Path,
    frame_hwc: np.ndarray,
    case_key: str,
    input_prompt: str,
    prompt_frame_idx: int,
    motion_prompt_box_xyxy: np.ndarray,
    final_prompt_box_xyxy: np.ndarray,
    mask_hw: np.ndarray,
) -> None:
    frame = frame_hwc.copy()
    mask = mask_hw > 0
    if mask.any():
        frame[mask] = ((0.60 * frame[mask]) + (0.40 * MASK_COLOR[None, :])).astype(np.uint8)
    draw_box_rgb(frame, motion_prompt_box_xyxy.astype(np.float32), PROMPT_COLOR, "motion_prompt")
    draw_box_rgb(frame, final_prompt_box_xyxy.astype(np.float32), FINAL_PROMPT_COLOR, "final_prompt")
    draw_text_panel(
        frame,
        [
            f"case: {case_key}",
            f"prompt_frame: {prompt_frame_idx}",
            f"prompt: {input_prompt[:70]}",
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(output_path)


def load_cases(json_dir: Path, *, start_index: int, num_cases: int) -> list[dict]:
    json_paths = sorted(json_dir.glob("*.json"))
    selected = json_paths[start_index : start_index + num_cases]
    cases: list[dict] = []
    for path in selected:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        source_video = data.get("source_video")
        if not source_video:
            continue
        cases.append(
            {
                "json_path": str(path),
                "case_key": str(data.get("case_key", path.stem)),
                "input_prompt": str(data.get("input_prompt", "")),
                "source_video": str(source_video),
            }
        )
    return cases


def evaluate_case(
    case: dict,
    *,
    tracker: SAM2MotionTracker,
    output_dir: Path,
    case_id: int,
    fps: int,
    max_frames: int,
) -> dict:
    video_path = Path(case["source_video"])
    frames_thwc = read_video_cv2(video_path, max_frames=max_frames)
    frames_tchw_01 = np.transpose(frames_thwc.astype(np.float32) / 255.0, (0, 3, 1, 2))
    prompt_frame_idx = max(int(frames_tchw_01.shape[0]) - 1, 0)
    motion_prompt_box_xyxy = build_motion_prompt_box(frames_tchw_01, prompt_frame_idx=prompt_frame_idx)
    sam_out = tracker.track(
        frames_tchw_01,
        prompt_frame_idx=prompt_frame_idx,
        prompt_box_xyxy=motion_prompt_box_xyxy,
        caption="",
    )

    overlay = render_overlay_video(
        frames_thwc=frames_thwc,
        case_key=case["case_key"],
        input_prompt=case["input_prompt"],
        source_video=case["source_video"],
        prompt_frame_idx=prompt_frame_idx,
        motion_prompt_box_xyxy=motion_prompt_box_xyxy,
        final_prompt_box_xyxy=sam_out.prompt_box_xyxy.astype(np.float32),
        sam_masks_thw=sam_out.masks_thw.astype(np.uint8),
        sam_boxes_t4=sam_out.boxes_t4.astype(np.float32),
    )

    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    stem = f"case_{case_id:03d}__{case['case_key']}"
    raw_video_path = assets_dir / f"{stem}__sam2_motion_prompt_overlay.mp4"
    write_mp4(raw_video_path, overlay, fps=fps)
    browser_video_path = ensure_browser_video(raw_video_path)

    preview_path = assets_dir / f"{stem}__prompt_frame_preview.png"
    save_prompt_preview(
        output_path=preview_path,
        frame_hwc=frames_thwc[prompt_frame_idx],
        case_key=case["case_key"],
        input_prompt=case["input_prompt"],
        prompt_frame_idx=prompt_frame_idx,
        motion_prompt_box_xyxy=motion_prompt_box_xyxy,
        final_prompt_box_xyxy=sam_out.prompt_box_xyxy.astype(np.float32),
        mask_hw=sam_out.masks_thw[prompt_frame_idx].astype(np.uint8),
    )

    return {
        "case_key": case["case_key"],
        "json_path": case["json_path"],
        "source_video": case["source_video"],
        "input_prompt": case["input_prompt"],
        "prompt_frame_idx": int(prompt_frame_idx),
        "prompt_mode": sam_out.prompt_mode,
        "prompt_text": sam_out.prompt_text,
        "motion_prompt_box_xyxy": motion_prompt_box_xyxy.astype(np.float32).tolist(),
        "final_prompt_box_xyxy": sam_out.prompt_box_xyxy.astype(np.float32).tolist(),
        "shapes": {
            "frames_thwc": list(frames_thwc.shape),
            "sam_masks_thw": list(sam_out.masks_thw.shape),
            "sam_boxes_t4": list(sam_out.boxes_t4.shape),
        },
        "overlay_video": str(browser_video_path.relative_to(output_dir)),
        "prompt_preview": str(preview_path.relative_to(output_dir)),
        "num_mask_pixels_prompt_frame": int(sam_out.masks_thw[prompt_frame_idx].sum()),
        "max_mask_pixels_any_frame": int(sam_out.masks_thw.reshape(sam_out.masks_thw.shape[0], -1).sum(axis=1).max()),
    }


def build_report(results: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = []
    for idx, result in enumerate(results):
        blocks.append(
            f"""
  <section class="case">
    <h2>Case {idx}: {result['case_key']}</h2>
    <p><b>JSON:</b> {result['json_path']}</p>
    <p><b>Source Video:</b> {result['source_video']}</p>
    <p><b>Input Prompt:</b> {result['input_prompt']}</p>
    <p><b>Prompt Frame:</b> {result['prompt_frame_idx']} | <b>Prompt Mode:</b> {result['prompt_mode']} | <b>Prompt Text:</b> {result['prompt_text']}</p>
    <p><b>Motion Prompt Box:</b> {result['motion_prompt_box_xyxy']}</p>
    <p><b>Final Prompt Box:</b> {result['final_prompt_box_xyxy']}</p>
    <p><b>Mask Pixels:</b> prompt_frame={result['num_mask_pixels_prompt_frame']} | max_any_frame={result['max_mask_pixels_any_frame']}</p>
    <div class="media-grid">
      <figure>
        <img src="{result['prompt_preview']}" alt="prompt preview">
        <figcaption>Prompt frame preview: motion prompt / final prompt / mask</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{result['overlay_video']}"></video>
        <figcaption>Full overlay video: prompt + mask + SAM2 box</figcaption>
      </figure>
    </div>
    <pre>{json.dumps({'shapes': result['shapes'], 'motion_prompt_box_xyxy': result['motion_prompt_box_xyxy'], 'final_prompt_box_xyxy': result['final_prompt_box_xyxy'], 'prompt_mode': result['prompt_mode'], 'prompt_text': result['prompt_text'], 'num_mask_pixels_prompt_frame': result['num_mask_pixels_prompt_frame'], 'max_mask_pixels_any_frame': result['max_mask_pixels_any_frame']}, indent=2, ensure_ascii=False)}</pre>
  </section>
"""
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SAM2 Motion Prompt On GT JSON Source Videos</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #f6f4ee; color: #222; }}
    .case {{ margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid #ddd; }}
    .media-grid {{ display: grid; grid-template-columns: 360px minmax(480px, 1fr); gap: 16px; align-items: start; }}
    img, video {{ width: 100%; border: 1px solid #ccc; background: #000; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 12px; color: #444; margin-top: 4px; }}
    pre {{ background: #fff; border: 1px solid #ddd; padding: 16px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>SAM2 Motion Prompt -> SAM2 Mask on GT JSON Source Videos</h1>
  <p>这页只看一条链路：从 GT json 里取 <code>source_video</code>，对视频生成 <code>motion prompt box</code>，把它喂给 SAM2，然后把 <code>prompt / mask / sam2 track box</code> 叠回原视频。这里禁用了文本检测，避免混入 GroundingDINO，专门排查 motion prompt 本身有没有问题。</p>
  {''.join(blocks)}
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
        "--json-dir",
        default="/data/gaoya/AAA_test_video/Output_try0526/ABD_test/B/GT",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/outputs/sam2_motion_prompt_json_viewer",
    )
    parser.add_argument("--port", type=int, default=8797)
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    cases = load_cases(json_dir, start_index=int(args.start_index), num_cases=int(args.num_cases))
    if not cases:
        raise RuntimeError(f"no usable json cases found in {json_dir}")

    tracker = SAM2MotionTracker(device="cuda" if torch.cuda.is_available() else "cpu", enable_text_prompt=False)
    output_dir = Path(args.output_dir)
    results = []
    for case_id, case in enumerate(cases):
        results.append(
            evaluate_case(
                case,
                tracker=tracker,
                output_dir=output_dir,
                case_id=case_id,
                fps=int(args.fps),
                max_frames=int(args.max_frames),
            )
        )

    html_path = build_report(results, output_dir)
    print(f"eval report: {html_path}")

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
