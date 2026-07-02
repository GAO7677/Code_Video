"""
Visualize CoTracker predicted tracks vs GT boxes, overlaid on video frames.

Usage:
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
  python3 /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/scripts/viz_cotracker_vs_gt.py \
    [--npz PATH] [--out-dir DIR] [--port PORT] [--scale SCALE]
"""
from __future__ import annotations
import argparse
import http.server
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path("/home/gaoya/Code_Video/co-tracker-main")))

from code_vjepa_vggt.utils.npz_io import load_npz_tensor_dict
from code_vjepa_vggt.adapters.cotracker_adapter import CoTrackerAdapter


def _sample_points_from_box(box_xyxy: torch.Tensor, points_per_object: int) -> torch.Tensor:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy.tolist()]
    if x1 <= x0 or y1 <= y0:
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        return torch.tensor([[cx, cy]] * points_per_object, dtype=torch.float32)
    cols = max(1, int(math.ceil(math.sqrt(float(points_per_object)))))
    rows = max(1, int(math.ceil(float(points_per_object) / float(cols))))
    xs = torch.linspace(x0 + 0.2 * (x1 - x0), x0 + 0.8 * (x1 - x0), cols)
    ys = torch.linspace(y0 + 0.2 * (y1 - y0), y0 + 0.8 * (y1 - y0), rows)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
    return pts[:points_per_object].contiguous()

COTRACKER_CKPT = "/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth"

COLORS = [
    (255,  60,  60),   # red
    ( 60, 200,  60),   # green
    ( 60, 100, 255),   # blue
    (255, 200,   0),   # yellow
    (255,  60, 220),   # magenta
    (  0, 220, 220),   # cyan
]

GT_ALPHA   = 0.45   # filled rect transparency
TRAIL_LEN  = 8      # how many past frames to draw trail


def draw_frame(
    frame_bgr: np.ndarray,
    gt_boxes_n4: np.ndarray,       # [N, 4] normalized xyxy
    pred_tracks_n2: np.ndarray,    # [N, 2] pixel xy (this frame)
    pred_tracks_all: np.ndarray,   # [T_so_far, N, 2] pixel xy
    frame_idx: int,
    valid_mask: np.ndarray,        # [N] bool
    H: int,
    W: int,
) -> np.ndarray:
    out = frame_bgr.copy()

    for n in range(len(gt_boxes_n4)):
        if not valid_mask[n]:
            continue
        color = COLORS[n % len(COLORS)]
        bgr = (color[2], color[1], color[0])

        # GT box: semi-transparent filled + solid border
        x0, y0, x1, y1 = gt_boxes_n4[n]
        x0p, y0p, x1p, y1p = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
        overlay = out.copy()
        cv2.rectangle(overlay, (x0p, y0p), (x1p, y1p), bgr, -1)
        out = cv2.addWeighted(overlay, GT_ALPHA, out, 1 - GT_ALPHA, 0)
        cv2.rectangle(out, (x0p, y0p), (x1p, y1p), bgr, 2)
        cv2.putText(out, f"GT{n}", (x0p + 2, y0p - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1, cv2.LINE_AA)

        # GT center crosshair
        cx, cy = int((x0 + x1) / 2 * W), int((y0 + y1) / 2 * H)
        cv2.drawMarker(out, (cx, cy), bgr, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)

        # CoTracker predicted track dot
        px, py = int(pred_tracks_n2[n, 0]), int(pred_tracks_n2[n, 1])
        cv2.circle(out, (px, py), 5, bgr, -1, cv2.LINE_AA)
        cv2.circle(out, (px, py), 7, (255, 255, 255), 1, cv2.LINE_AA)

        # Trail
        start = max(0, frame_idx - TRAIL_LEN + 1)
        pts = pred_tracks_all[start : frame_idx + 1, n].astype(int)
        for i in range(1, len(pts)):
            alpha = (i + 1) / len(pts)
            t_color = tuple(int(c * alpha) for c in bgr)
            cv2.line(out, tuple(pts[i - 1]), tuple(pts[i]), t_color, 2, cv2.LINE_AA)

        # Error arrow GT center → pred
        cv2.arrowedLine(out, (cx, cy), (px, py), (255, 255, 255), 1,
                        cv2.LINE_AA, tipLength=0.3)

    # Legend
    cv2.putText(out, f"frame {frame_idx:02d}", (8, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(out, "filled=GT  dot=CoTracker  arrow=error", (8, H - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def run(
    npz_path: Path,
    out_dir: Path,
    scale: float,
    device: str,
    port: int,
    max_objects: int = 4,
    points_per_object: int = 2,
    cotracker_input_hw: tuple[int, int] = (384, 512),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = load_npz_tensor_dict(npz_path)
    ctx_frames = tensors["context_frames"].float()   # [T, 3, H, W], [0,1]
    ctx_boxes  = tensors["context_boxes"].float()    # [T, N, 4]
    T, C, H, W = ctx_frames.shape
    N_gt = ctx_boxes.shape[1]

    disp_h = max(1, int(H * scale))
    disp_w = max(1, int(W * scale))
    print(f"Sample: {npz_path.stem}  T={T} H={H} W={W} N_gt={N_gt}")

    # Build query points from GT boxes (same logic as ContextOnlyInjectionTrainer)
    grouped_queries = torch.zeros(1, max_objects, points_per_object, 2)
    valid_mask_t   = torch.zeros(1, max_objects)
    for n in range(min(max_objects, N_gt)):
        for t in range(T):
            box = ctx_boxes[t, n]
            if float(box[2] - box[0]) > 1e-6 and float(box[3] - box[1]) > 1e-6:
                pts = _sample_points_from_box(box.detach().float().cpu(), points_per_object)
                pts[:, 0] *= W
                pts[:, 1] *= H
                grouped_queries[0, n] = pts
                valid_mask_t[0, n] = 1.0
                break

    valid_objs = int(valid_mask_t[0].sum().item())
    print(f"Valid objects: {valid_objs}/{max_objects}")

    query_points_prior = grouped_queries.view(1, max_objects * points_per_object, 2)

    # CoTracker
    adapter = CoTrackerAdapter(
        checkpoint_path=COTRACKER_CKPT,
        num_queries=max_objects * points_per_object,
        device=device,
        input_hw=cotracker_input_hw,
        window_len=60,
    )
    frames_bthwc = ctx_frames.permute(1, 2, 3, 0).unsqueeze(0)   # [1, T, H, W, 3] wrong
    # correct: ctx_frames [T,3,H,W] → [1,T,H,W,3]
    frames_bthwc = ctx_frames.permute(0, 2, 3, 1).unsqueeze(0).to(device)  # [1,T,H,W,3]

    with torch.no_grad():
        cot_out = adapter(
            frames_bthwc,
            query_points_prior=query_points_prior.to(device),
            query_image_hw=(H, W),
        )

    # cot_out.tracks: [1, T, N_q, 2] at native (H,W)
    tracks_np = cot_out.tracks[0].cpu().numpy()   # [T, N_q, 2]
    # Group back: N_q = max_objects * points_per_object → take mean per object
    tracks_np_grouped = tracks_np.reshape(T, max_objects, points_per_object, 2).mean(axis=2)  # [T, max_objects, 2]

    # Build per-object valid mask (from valid_mask_t)
    valid_np = valid_mask_t[0].numpy().astype(bool)  # [max_objects]

    # Render frames
    annotated = []
    for t in range(T):
        frame_hwc = (ctx_frames[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_hwc, cv2.COLOR_RGB2BGR)

        # boxes for this frame: use all N_gt but only show up to max_objects
        gt_boxes_n4 = ctx_boxes[t, :max_objects].numpy()

        drawn = draw_frame(
            frame_bgr,
            gt_boxes_n4=gt_boxes_n4,
            pred_tracks_n2=tracks_np_grouped[t],
            pred_tracks_all=tracks_np_grouped,
            frame_idx=t,
            valid_mask=valid_np,
            H=H,
            W=W,
        )
        if scale != 1.0:
            drawn = cv2.resize(drawn, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        annotated.append(drawn)

    # Print per-frame error table
    print(f"\n{'obj':>4} {'frame':>5} {'pred_x':>7} {'pred_y':>7} {'gt_x':>7} {'gt_y':>7} {'err_x':>7} {'err_y':>7}")
    for n in range(max_objects):
        if not valid_np[n]:
            continue
        for t in range(T):
            gt_cx = float((ctx_boxes[t, n, 0] + ctx_boxes[t, n, 2]) / 2 * W)
            gt_cy = float((ctx_boxes[t, n, 1] + ctx_boxes[t, n, 3]) / 2 * H)
            px, py = tracks_np_grouped[t, n]
            print(f"{n:>4} {t:>6} {px:>7.1f} {py:>7.1f} {gt_cx:>7.1f} {gt_cy:>7.1f} {abs(px-gt_cx):>7.1f} {abs(py-gt_cy):>7.1f}")

    # Save MP4 (loop 3×, slower fps for easier review)
    out_mp4 = out_dir / f"{npz_path.stem}_cotracker_vs_gt.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 4
    vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (disp_w, disp_h))
    for rep in range(3):
        for frame in annotated:
            vw.write(frame)
    vw.release()
    print(f"\nSaved: {out_mp4}")

    # Also save individual PNGs for easy browsing
    for t, frame in enumerate(annotated):
        cv2.imwrite(str(out_dir / f"frame_{t:02d}.png"), frame)

    # Simple HTML viewer
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>CoTracker vs GT — {npz_path.stem}</title>
<style>body{{background:#111;color:#eee;font-family:monospace;padding:20px}}
video{{max-width:100%;border:2px solid #555}}
.frames{{display:flex;flex-wrap:wrap;gap:6px;margin-top:16px}}
img{{border:1px solid #444;width:{disp_w}px;height:{disp_h}px}}
</style></head><body>
<h2>CoTracker vs GT boxes — {npz_path.stem}</h2>
<p>Filled rect = GT box &nbsp;|&nbsp; Colored dot = CoTracker pred &nbsp;|&nbsp; Arrow = error</p>
<video controls autoplay loop muted>
  <source src="{out_mp4.name}" type="video/mp4">
</video>
<h3>Individual frames</h3>
<div class="frames">
{''.join(f'<div><p>t={t}</p><img src="frame_{t:02d}.png"></div>' for t in range(T))}
</div>
</body></html>"""
    html_path = out_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"HTML:  {html_path}")
    print(f"\nServing on http://0.0.0.0:{port}  →  open http://localhost:{port}/")
    print("Press Ctrl+C to stop.\n")

    handler = http.server.SimpleHTTPRequestHandler
    # serve from out_dir
    import os
    os.chdir(out_dir)
    with http.server.HTTPServer(("0.0.0.0", port), handler) as httpd:
        httpd.serve_forever()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path,
                   default=Path("/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/"
                                "industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500/val/"
                                "sample_000301_w000.npz"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/data/gaoya/agent-data/outputs/cotracker_vs_gt_viz"))
    p.add_argument("--port", type=int, default=8899)
    p.add_argument("--scale", type=float, default=3.0)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        npz_path=args.npz,
        out_dir=args.out_dir,
        scale=args.scale,
        device=args.device,
        port=args.port,
    )
