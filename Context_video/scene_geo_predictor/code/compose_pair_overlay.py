#!/usr/bin/env python3
"""Composite GT and predictor paths over the canonical pair rerenders."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "validation_20260920/paired_pair_overlay_v1"
FPS = 30.0
SEED = 20262001
PANEL_W, PANEL_H = 896, 512
HEADER_H = 88


def camera_from_render(render_metadata: dict):
    width, height = map(int, render_metadata["resolution"])
    camera = render_metadata["camera"]
    eye = np.asarray(camera["location"], dtype=np.float64)
    target = np.asarray(camera["target"], dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.stack((right, -up, forward))
    rt = np.column_stack((rotation, -rotation @ eye))
    f = height / (2.0 * np.tan(np.deg2rad(float(camera["effective_yfov_deg"])) / 2.0))
    k = np.asarray([[f, 0.0, (width - 1) / 2.0],
                    [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]])
    return k, rt, width, height


def project(points: np.ndarray, k: np.ndarray, rt: np.ndarray) -> np.ndarray:
    camera = np.asarray(points) @ rt[:, :3].T + rt[:, 3]
    pixels = camera @ k.T
    uv = pixels[..., :2] / np.maximum(pixels[..., 2:,], 1e-8)
    uv[camera[..., 2] <= 0] = np.nan
    return uv


def line(img, pts, color, width=3, dash=False):
    pts = np.asarray(pts, dtype=np.float64)
    valid = np.isfinite(pts).all(axis=1)
    if valid.sum() < 2:
        return
    points = np.round(pts[valid]).astype(np.int32).reshape(-1, 1, 2)
    if not dash:
        cv2.polylines(img, [points], False, color, width, cv2.LINE_AA)
        return
    for i in range(len(points) - 1):
        if i % 2 == 0:
            cv2.line(img, tuple(points[i, 0]), tuple(points[i + 1, 0]), color, width, cv2.LINE_AA)


def text(img, value, xy, scale=0.55, color=(240, 240, 240), thick=1):
    cv2.putText(img, str(value), tuple(map(int, xy)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def fmt_mm(value):
    return f"{float(value) * 1000.0:.2f} mm"


def compose():
    payload = json.loads((OUT / "data.json").read_text(encoding="utf-8"))
    keys = list(payload["arms"])
    assert len(keys) == 2
    exact = payload["selection"]["metrics"]["geometry_dense_resample"]
    exact_motion = payload["selection"]["metrics"]["motion_only"]
    render_dirs = [OUT / "rendered" / key for key in keys]
    camera_info = [json.loads((d / "render_metadata.json").read_text()) for d in render_dirs]
    cameras = [camera_from_render(x) for x in camera_info]

    # Check the projection independently against Blender's saved normalized
    # diagnostic before drawing any trajectories.
    projection_check = []
    for key, info, (k, rt, width, height) in zip(keys, camera_info, cameras):
        observed = np.asarray(payload["arms"][key]["observed_positions"], dtype=np.float64)
        uv = project(observed[0:1], k, rt)[0]
        diag = info["camera"]["object_projections_xy_depth"][payload["arms"][key]["dynamic_object"]]
        expected = np.asarray([diag[0] * width - 0.5, (1.0 - diag[1]) * height - 0.5])
        projection_check.append({"key": key, "our_pixel": uv.tolist(), "blender_pixel": expected.tolist(),
                                 "max_abs_error_px": float(np.max(np.abs(uv - expected)))})
    (OUT / "projection_check.json").write_text(json.dumps(projection_check, indent=2) + "\n", encoding="utf-8")

    video_path = OUT / "overlay_pair_g01_w46_w70_seed20262001.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (PANEL_W * 2, PANEL_H + HEADER_H))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {video_path}")

    colors = {"GT": (40, 220, 220), "geometry": (55, 65, 245), "motion": (50, 215, 95),
              "observed": (210, 210, 210)}
    for frame_index in range(49):
        panels = []
        for panel_index, (key, (k, rt, _, _), render_dir) in enumerate(zip(keys, cameras, render_dirs)):
            frame_path = render_dir / f"frame_{frame_index + 1:04d}.png"
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(frame_path)
            observed = np.asarray(payload["arms"][key]["observed_positions"], dtype=np.float64)
            gt = np.asarray(payload["arms"][key]["target_positions"], dtype=np.float64)
            geom = np.asarray(payload["arms"][key]["geometry_dense_resample"], dtype=np.float64)
            motion = np.asarray(payload["arms"][key]["motion_only"], dtype=np.float64)
            observed_uv = project(observed, k, rt)
            gt_uv = project(np.concatenate((observed[-1:], gt)), k, rt)
            geom_uv = project(np.concatenate((observed[-1:], geom)), k, rt)
            motion_uv = project(np.concatenate((observed[-1:], motion)), k, rt)
            # The first eight positions are observed. Future paths become
            # visible at RGB8, so the overlay does not reveal future truth.
            observed_end = min(frame_index + 1, 8)
            line(image, observed_uv[:observed_end], colors["observed"], 2, dash=True)
            future_count = max(0, frame_index - 7)
            for path, color, label in ((gt_uv, colors["GT"], "GT"),
                                       (geom_uv, colors["geometry"], "geometry"),
                                       (motion_uv, colors["motion"], "motion-only")):
                end = min(1 + future_count, len(path))
                line(image, path[:end], color, 3)
                if end > 0 and np.isfinite(path[end - 1]).all():
                    point = tuple(np.round(path[end - 1]).astype(int))
                    cv2.circle(image, point, 5, color, -1, cv2.LINE_AA)
            text(image, f"RGB{frame_index}  |  {key}", (16, 28), 0.55, (255, 255, 255), 1)
            if frame_index >= 8:
                text(image, "GT", (16, 54), 0.48, colors["GT"], 1)
                text(image, "geometry", (70, 54), 0.48, colors["geometry"], 1)
                text(image, "motion-only", (165, 54), 0.48, colors["motion"], 1)
            text(image, "background: canonical CPU rerender", (16, PANEL_H - 12), 0.42, (230, 230, 230), 1)
            panels.append(image)
        header = np.zeros((HEADER_H, PANEL_W * 2, 3), dtype=np.uint8)
        header[:] = (26, 29, 36)
        text(header, "Future Query Predictor paired overlay | same history, different door width", (16, 30), 0.66, (245, 245, 245), 1)
        text(header, f"aggregate: Dgt={fmt_mm(payload['aggregate']['D_gt_m'])}  Dpred={fmt_mm(payload['aggregate']['D_pred_m'])}  Rdelta={payload['aggregate']['R_delta']:.5f}", (16, 60), 0.52, (220, 220, 220), 1)
        text(header, f"selected pair: geometry Dgt={fmt_mm(exact['D_gt_m'])}  Dpred={fmt_mm(exact['D_pred_m'])}  Rdelta={float(exact['R_delta']):.5f}", (900, 30), 0.52, colors["geometry"], 1)
        text(header, f"motion-only Dpred={fmt_mm(exact_motion['D_pred_m'])}  |  seed {payload['selection']['sampling_seed']}", (900, 60), 0.52, colors["motion"], 1)
        writer.write(np.vstack((header, np.hstack(panels))))
        if frame_index == 24:
            cv2.imwrite(str(OUT / "overlay_frame_RGB24.png"), np.vstack((header, np.hstack(panels))))
    writer.release()

    page = f"""<!doctype html>
<html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Paired Future Query Overlay</title>
<style>
body{{font:15px system-ui,-apple-system,sans-serif;max-width:1420px;margin:24px auto;padding:0 18px;color:#20242b;background:#f5f6f8}}
h1{{font-size:25px;margin:0 0 8px}} h2{{font-size:17px;margin:26px 0 8px}}
.note{{background:#fff7e6;border-left:4px solid #d98b20;padding:12px 14px;line-height:1.55}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(145px,1fr));gap:10px;margin:16px 0}}
.card{{background:white;border:1px solid #e3e6eb;border-radius:7px;padding:12px}} .value{{font-size:20px;font-weight:650}}
video,img{{width:100%;height:auto;background:#1a1d24;border-radius:7px;display:block}} small{{color:#66707d}}
table{{background:white;border-collapse:collapse;width:100%;margin-top:10px}}th,td{{padding:9px;border-bottom:1px solid #e5e7eb;text-align:left}} code{{font-size:12px}}
@media(max-width:760px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body>
<h1>同历史、不同门宽：轨迹 overlay</h1>
<p><b>选定 pair：</b><code>history_door_g01</code>，门宽 0.46 m ↔ 0.70 m，sampling seed {SEED}。</p>
<div class=\"note\"><b>重要：</b>这批 independent-history 样本只有 physics states，没有原始 RGB/MP4。下面的背景是用同一门场景渲染器、同一相机和状态轨迹做的 <b>canonical CPU rerender</b>，不是原始输入视频；轨迹投影使用渲染器保存的相机诊断。</div>
<div class=\"grid\">
<div class=\"card\"><small>用户给出的汇总 Dgt</small><div class=\"value\">{payload['aggregate']['D_gt_m']*1000:.2f} mm</div><small>144 train nondegenerate pair/seed</small></div>
<div class=\"card\"><small>用户给出的汇总 Dpred</small><div class=\"value\">{payload['aggregate']['D_pred_m']*1000:.2f} mm</div><small>geometry_dense_resample</small></div>
<div class=\"card\"><small>用户给出的汇总 Rdelta</small><div class=\"value\">{payload['aggregate']['R_delta']:.5f}</div><small>不是单个 case</small></div>
<div class=\"card\"><small>本页 selected pair</small><div class=\"value\">{float(exact['D_gt_m'])*1000:.2f} / {float(exact['D_pred_m'])*1000:.2f} mm</div><small>Dgt / Dpred</small></div>
</div>
<video controls autoplay muted loop playsinline poster=\"overlay_frame_RGB24.png\"><source src=\"overlay_pair_g01_w46_w70_seed20262001.mp4\" type=\"video/mp4\"></video>
<h2>RGB24 静帧</h2><img src=\"overlay_frame_RGB24.png\" alt=\"RGB24 paired trajectory overlay\">
<h2>精确指标</h2>
<table><thead><tr><th>arm</th><th>Dgt</th><th>Dpred</th><th>Rdelta</th><th>解释</th></tr></thead><tbody>
<tr><td>motion-only</td><td>{float(exact_motion['D_gt_m'])*1000:.3f} mm</td><td>{float(exact_motion['D_pred_m'])*1000:.3f} mm</td><td>{float(exact_motion['R_delta']):.6f}</td><td>不读取场景，pair response 为 0</td></tr>
<tr><td>geometry_dense_resample</td><td>{float(exact['D_gt_m'])*1000:.3f} mm</td><td>{float(exact['D_pred_m'])*1000:.3f} mm</td><td>{float(exact['R_delta']):.6f}</td><td>叠加红色轨迹</td></tr>
</tbody></table>
<p><small>颜色：青色 GT，红色 geometry_dense_resample，绿色 motion-only，灰色为 RGB0–RGB7 观察段。数据与 provenance：<a href=\"data.json\">data.json</a>；投影核对：<a href=\"projection_check.json\">projection_check.json</a>。</small></p>
</body></html>"""
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"video": str(video_path), "poster": str(OUT / "overlay_frame_RGB24.png"),
                      "projection_check": projection_check}, indent=2))


if __name__ == "__main__":
    compose()
