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

from code_vjepa_vggt.adapters.sam2_motion import SAM2MotionTracker, build_motion_prompt_box
from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset
from code_vjepa_vggt.utils.config import load_yaml_config


PALETTE = [
    (214, 40, 40),
    (247, 127, 0),
    (252, 191, 73),
    (42, 157, 143),
    (39, 125, 161),
    (106, 76, 147),
]
PROMPT_COLOR = (255, 140, 0)
SAM2_COLOR = (32, 160, 96)
OTHER_GT_COLOR = (160, 160, 160)


def tensor_frame_to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    x = frame_chw.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).contiguous()
    return x.numpy()


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
    cmd = [
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
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def box_valid(box_xyxy: torch.Tensor, eps: float = 1e-6) -> bool:
    return bool((box_xyxy[2] - box_xyxy[0] > eps).item() and (box_xyxy[3] - box_xyxy[1] > eps).item())


def box_center_xy(box_xyxy: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    width = image_hw[1]
    height = image_hw[0]
    x0, y0, x1, y1 = box_xyxy
    cx = 0.5 * (x0 + x1) * width
    cy = 0.5 * (y0 + y1) * height
    return torch.stack([cx, cy], dim=0)


def track_inside_box(point_xy: torch.Tensor, box_xyxy: torch.Tensor, image_hw: tuple[int, int]) -> bool:
    width = image_hw[1]
    height = image_hw[0]
    x0 = float(box_xyxy[0].item()) * width
    y0 = float(box_xyxy[1].item()) * height
    x1 = float(box_xyxy[2].item()) * width
    y1 = float(box_xyxy[3].item()) * height
    x = float(point_xy[0].item())
    y = float(point_xy[1].item())
    return x0 <= x <= x1 and y0 <= y <= y1


def draw_box_rgb(image: np.ndarray, box_xyxy_px: np.ndarray, color_rgb: tuple[int, int, int], label: str) -> None:
    x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy_px.tolist()]
    if x1 <= x0 or y1 <= y0:
        return
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.rectangle(image, (x0, y0), (x1, y1), color_bgr, 2)
    cv2.putText(image, label, (x0 + 2, max(y0 + 14, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)


def norm_box_to_px(box_xyxy: torch.Tensor, image_hw: tuple[int, int]) -> np.ndarray:
    width = image_hw[1]
    height = image_hw[0]
    scale = torch.tensor([width, height, width, height], dtype=box_xyxy.dtype)
    return (box_xyxy.cpu() * scale).numpy().astype(np.float32)


def render_overlay_video(
    context_video: torch.Tensor,
    context_boxes: torch.Tensor,
    *,
    prompt_box_xyxy: np.ndarray,
    sam_boxes_t4: np.ndarray,
    highlight_gt_idx: int | None,
) -> np.ndarray:
    frames = []
    image_hw = (context_video.shape[-2], context_video.shape[-1])
    for t in range(context_video.shape[1]):
        frame = tensor_frame_to_uint8_hwc(context_video[:, t]).copy()
        if highlight_gt_idx is None:
            for obj_idx in range(context_boxes.shape[1]):
                gt_box = context_boxes[t, obj_idx]
                if not box_valid(gt_box):
                    continue
                draw_box_rgb(frame, norm_box_to_px(gt_box, image_hw), PALETTE[obj_idx % len(PALETTE)], f"gt{obj_idx}")
        else:
            for obj_idx in range(context_boxes.shape[1]):
                gt_box = context_boxes[t, obj_idx]
                if not box_valid(gt_box):
                    continue
                color = PALETTE[obj_idx % len(PALETTE)] if obj_idx == highlight_gt_idx else OTHER_GT_COLOR
                label = f"gt{obj_idx}"
                draw_box_rgb(frame, norm_box_to_px(gt_box, image_hw), color, label)
        draw_box_rgb(frame, prompt_box_xyxy.astype(np.float32), PROMPT_COLOR, "prompt")
        draw_box_rgb(frame, sam_boxes_t4[t].astype(np.float32), SAM2_COLOR, "sam2")
        frames.append(frame)
    return np.stack(frames, axis=0)


def evaluate_sample(
    sample: dict,
    tracker: SAM2MotionTracker,
    *,
    output_dir: Path,
    case_id: int,
    fps: int,
    prompt_frame_mode: str = "last",
) -> dict:
    context_video = sample["context_video"]
    context_boxes = sample["context_boxes"]
    caption = sample["caption"]
    frames_tchw_01 = ((context_video.permute(1, 0, 2, 3).float() + 1.0) / 2.0).cpu().numpy()
    prompt_frame_idx = max(context_video.shape[1] - 1, 0) if prompt_frame_mode == "last" else 0
    motion_prompt_box_xyxy = build_motion_prompt_box(frames_tchw_01, prompt_frame_idx=prompt_frame_idx)
    sam_out = tracker.track(
        frames_tchw_01,
        prompt_frame_idx=prompt_frame_idx,
        prompt_box_xyxy=motion_prompt_box_xyxy,
        caption=caption,
    )

    image_hw = (context_video.shape[-2], context_video.shape[-1])
    valid_gt_indices = []
    per_object_metrics = []
    for obj_idx in range(context_boxes.shape[1]):
        l1_values = []
        inside_hits = []
        for t in range(context_video.shape[1]):
            gt_box = context_boxes[t, obj_idx]
            sam_box = torch.from_numpy(sam_out.boxes_norm_t4[t])
            if not (box_valid(gt_box) and box_valid(sam_box)):
                continue
            gt_center = box_center_xy(gt_box, image_hw)
            sam_center = box_center_xy(sam_box, image_hw)
            l1_values.append(float(torch.abs(gt_center - sam_center).sum().item()))
            inside_hits.append(float(track_inside_box(sam_center, gt_box, image_hw)))
        if not l1_values:
            continue
        valid_gt_indices.append(obj_idx)
        per_object_metrics.append(
            {
                "gt_idx": int(obj_idx),
                "mean_center_l1_px": float(sum(l1_values) / len(l1_values)),
                "inside_box_rate": float(sum(inside_hits) / len(inside_hits)),
                "valid_track_points": int(len(l1_values)),
            }
        )

    if per_object_metrics:
        best_metric = min(per_object_metrics, key=lambda item: item["mean_center_l1_px"])
        best_gt_idx = int(best_metric["gt_idx"])
    else:
        best_metric = {"mean_center_l1_px": 0.0, "inside_box_rate": 0.0, "valid_track_points": 0}
        best_gt_idx = -1

    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    all_gt_video_rel = f"assets/case_{case_id:03d}__all_gt_overlay.mp4"
    all_gt_video = render_overlay_video(
        context_video,
        context_boxes,
        prompt_box_xyxy=sam_out.prompt_box_xyxy,
        sam_boxes_t4=sam_out.boxes_t4,
        highlight_gt_idx=None,
    )
    raw_all_gt_path = output_dir / all_gt_video_rel
    write_mp4(raw_all_gt_path, all_gt_video, fps)
    browser_all_gt_path = ensure_browser_video(raw_all_gt_path)

    object_videos = []
    for metric in per_object_metrics:
        obj_idx = int(metric["gt_idx"])
        video_rel = f"assets/case_{case_id:03d}__gt{obj_idx}_overlay.mp4"
        overlay = render_overlay_video(
            context_video,
            context_boxes,
            prompt_box_xyxy=sam_out.prompt_box_xyxy,
            sam_boxes_t4=sam_out.boxes_t4,
            highlight_gt_idx=obj_idx,
        )
        raw_obj_path = output_dir / video_rel
        write_mp4(raw_obj_path, overlay, fps)
        browser_obj_path = ensure_browser_video(raw_obj_path)
        object_videos.append(
            {
                "gt_idx": obj_idx,
                "color_rgb": PALETTE[obj_idx % len(PALETTE)],
                "video": str(browser_obj_path.relative_to(output_dir)),
                "metrics": metric,
            }
        )

    return {
        "caption": caption,
        "video_path": sample["video_path"],
        "context_frame_indices": sample["context_frame_indices"].tolist(),
        "prompt_frame_idx": int(prompt_frame_idx),
        "prompt_mode": sam_out.prompt_mode,
        "prompt_text": sam_out.prompt_text,
        "best_gt_idx": int(best_gt_idx),
        "best_metrics": best_metric,
        "per_object_metrics": per_object_metrics,
        "shapes": {
            "context_video": list(context_video.unsqueeze(0).shape),
            "context_boxes": list(context_boxes.unsqueeze(0).shape),
            "sam_masks_thw": list(sam_out.masks_thw.shape),
            "sam_boxes_t4": list(sam_out.boxes_t4.shape),
            "sam_boxes_norm_t4": list(sam_out.boxes_norm_t4.shape),
        },
        "motion_prompt_box_xyxy": motion_prompt_box_xyxy.tolist(),
        "prompt_box_xyxy": sam_out.prompt_box_xyxy.tolist(),
        "all_gt_video": str(browser_all_gt_path.relative_to(output_dir)),
        "object_videos": object_videos,
    }


def build_report(results: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "num_cases": len(results),
        "avg_best_mean_center_l1_px": sum(r["best_metrics"]["mean_center_l1_px"] for r in results) / max(len(results), 1),
        "avg_best_inside_box_rate": sum(r["best_metrics"]["inside_box_rate"] for r in results) / max(len(results), 1),
        "cases": [
            {
                "case_id": idx,
                "video_path": r["video_path"],
                "caption": r["caption"],
                "prompt_mode": r["prompt_mode"],
                "prompt_text": r["prompt_text"],
                "best_gt_idx": r["best_gt_idx"],
                "best_metrics": r["best_metrics"],
                "per_object_metrics": r["per_object_metrics"],
            }
            for idx, r in enumerate(results)
        ],
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    blocks = []
    for idx, result in enumerate(results):
        video_cards = []
        video_cards.append(
            f"""
      <figure>
        <video controls preload="none" playsinline src="{result['all_gt_video']}"></video>
        <figcaption>all gt overlay</figcaption>
      </figure>
"""
        )
        for item in result["object_videos"]:
            color = item["color_rgb"]
            video_cards.append(
                f"""
      <figure>
        <video controls preload="none" playsinline src="{item['video']}"></video>
        <figcaption>gt{item['gt_idx']} color=rgb{tuple(color)} | L1={item['metrics']['mean_center_l1_px']:.2f} | inside={item['metrics']['inside_box_rate']:.3f}</figcaption>
      </figure>
"""
            )

        blocks.append(
            f"""
  <section class="case">
    <h2>Case {idx}</h2>
    <p><b>Caption:</b> {result['caption']}</p>
    <p><b>Context frames:</b> {result['context_frame_indices']}</p>
    <p><b>Prompt frame:</b> {result['prompt_frame_idx']} | <b>Best gt idx:</b> {result['best_gt_idx']}</p>
    <p><b>Prompt mode:</b> {result['prompt_mode']} | <b>Prompt text:</b> {result['prompt_text']}</p>
    <p><b>Best Metrics:</b> mean_center_l1_px={result['best_metrics']['mean_center_l1_px']:.2f}, inside_box_rate={result['best_metrics']['inside_box_rate']:.3f}, valid_track_points={result['best_metrics']['valid_track_points']}</p>
    <p><b>Motion prompt box xyxy:</b> {result['motion_prompt_box_xyxy']}</p>
    <p><b>Final prompt box xyxy:</b> {result['prompt_box_xyxy']}</p>
    <div class="video-grid">
      {''.join(video_cards)}
    </div>
    <pre>{json.dumps({'best_metrics': result['best_metrics'], 'per_object_metrics': result['per_object_metrics'], 'shapes': result['shapes'], 'prompt_mode': result['prompt_mode'], 'prompt_text': result['prompt_text'], 'motion_prompt_box_xyxy': result['motion_prompt_box_xyxy'], 'prompt_box_xyxy': result['prompt_box_xyxy']}, indent=2, ensure_ascii=False)}</pre>
  </section>
"""
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SAM2 Script-Style Prompt vs Box Supervision</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #f6f4ee; color: #222; }}
    .case {{ margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid #ddd; }}
    .video-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }}
    .video-grid video {{ width: 100%; border: 1px solid #ccc; background: #000; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 12px; color: #444; margin-top: 4px; }}
    pre {{ background: #fff; border: 1px solid #ddd; padding: 16px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>SAM2 Script-Style Prompt vs Box Supervision</h1>
  <p>当前页面按官方脚本思路运行：先从 caption 提取对象词，再用 Grounding DINO 在 prompt frame 上检测，接着用 SAM image predictor 把框细化成 mask，最后把 mask 注册到 SAM2 video predictor 并分段传播。若文本检测失败，则退回 motion proxy box。橙框是最终送进 SAM2 的 prompt，绿框是 SAM2 跟踪结果。</p>
  <p><b>Overall:</b> avg_best_mean_center_l1_px={summary['avg_best_mean_center_l1_px']:.2f}, avg_best_inside_box_rate={summary['avg_best_inside_box_rate']:.3f}, num_cases={summary['num_cases']}</p>
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
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/inspect_phys_state_vjepa_vggt.yaml",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/outputs/sam2_motion_eval_viewer",
    )
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    data_cfg = cfg["data"]
    dataset = PhysStateEpisodeDataset(
        root=data_cfg["root"],
        split=args.split,
        resolution=tuple(data_cfg["resolution"]),
        num_context_frames=int(data_cfg["num_context_frames"]),
        context_fraction=float(data_cfg.get("context_fraction", 0.5)),
        random_context_frames=bool(data_cfg.get("random_context_frames", True)),
        seed=int(cfg.get("experiment", {}).get("seed", 42)),
    )

    output_dir = Path(args.output_dir)
    tracker = SAM2MotionTracker(device="cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for case_id, idx in enumerate(range(args.start_index, min(len(dataset), args.start_index + args.num_cases))):
        results.append(
            evaluate_sample(
                dataset[idx],
                tracker,
                output_dir=output_dir,
                case_id=case_id,
                fps=args.fps,
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
