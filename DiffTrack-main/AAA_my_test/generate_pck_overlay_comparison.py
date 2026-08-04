#!/usr/bin/env python3
"""Render the exact PCK@32 comparisons used by the saved S039 Q@K tracks."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


DATA_ROOT = Path(
    "/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case"
)
RANKING_CSV = Path(
    "/data/gaoya/agent-data/outputs/three_model_all720_uniform_diagonal_5case/"
    "all720_uniform_diagonal_summary.csv"
)
OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_high_low_overlay_case001_s039"
)
CASE_KEY = "case_001_ball_roll"
STEP = 39
THRESHOLD_PX = 32.0
QUERY_LATENT_INDEX = 1
CLEAN_PREFIX_LATENTS = 2
MODEL_SPECS = (
    ("gt", "GT teacher-forced", (71, 196, 255)),
    ("lora", "LoRA", (247, 195, 79)),
    ("baseline", "Wan2.2 Baseline", (120, 111, 239)),
)
REGION_COLORS = ((255, 211, 76), (72, 168, 255))  # object, background (BGR)
FFMPEG = "/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg"


def read_ranked_combinations() -> list[dict]:
    with RANKING_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["pck32"]), reverse=True)
    selected = []
    for group, values in (("top", rows[:3]), ("bottom", rows[-3:])):
        ordered = values if group == "top" else list(reversed(values))
        for rank, row in enumerate(ordered, 1):
            selected.append(
                {
                    "group": group,
                    "rank": rank,
                    "block": int(row["block"]),
                    "head": int(row["head"]),
                    "ranking_pck32": float(row["pck32"]),
                }
            )
    return selected


def case_dir(model_key: str) -> Path:
    return DATA_ROOT / model_key / "cases" / CASE_KEY


def load_original_frames() -> tuple[list[np.ndarray], float, str]:
    manifest = json.loads((case_dir("gt") / "manifest.json").read_text())
    video_path = manifest["gt_video"]
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open original video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 6.0)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from: {video_path}")
    return frames, fps, video_path


def load_tracks(model_key: str, block: int, head: int) -> dict:
    directory = case_dir(model_key)
    with np.load(directory / "cotracker_pseudo_gt.npz") as gt_file:
        gt_tracks = gt_file["tracks"].astype(np.float32)
        visibility = gt_file["visibility"].astype(bool)
        anchors = gt_file["latent_anchor_frames"].astype(int)
    key = f"qk_head{head:02d}_layer{block:02d}_step{STEP:03d}_predictions"
    with np.load(directory / "predicted_tracks.npz") as prediction_file:
        predictions = prediction_file[key].astype(np.float32)
    gt_at_anchors = gt_tracks[anchors]
    visible_at_anchors = visibility[anchors]
    valid = visible_at_anchors & visible_at_anchors[QUERY_LATENT_INDEX : QUERY_LATENT_INDEX + 1]
    valid[:CLEAN_PREFIX_LATENTS] = False
    error = np.linalg.norm(predictions - gt_at_anchors, axis=-1)
    return {
        "gt": gt_at_anchors,
        "pred": predictions,
        "valid": valid,
        "error": error,
        "anchors": anchors,
    }


def pck(values: np.ndarray) -> float | None:
    return float(np.mean(values <= THRESHOLD_PX) * 100.0) if values.size else None


def metrics(track: dict) -> dict:
    valid = track["valid"]
    error = track["error"]
    return {
        "all": pck(error[valid]),
        "object": pck(error[:, :8][valid[:, :8]]),
        "background": pck(error[:, 8:16][valid[:, 8:16]]),
        "comparisons": int(valid.sum()),
    }


def put_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (8, 12, 16), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_polyline(image: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    if len(points) > 1:
        cv2.polylines(image, [np.asarray(points, np.int32)], False, color, 1, cv2.LINE_AA)


def render_panel(
    frame: np.ndarray,
    frame_index: int,
    model_label: str,
    model_color: tuple[int, int, int],
    track: dict,
    summary: dict,
) -> np.ndarray:
    height, width = frame.shape[:2]
    top, bottom = 54, 64
    panel = np.full((height + top + bottom, width, 3), (17, 21, 24), np.uint8)
    panel[top : top + height] = frame
    cv2.rectangle(panel, (0, 0), (width - 1, top - 1), (25, 31, 35), -1)
    cv2.rectangle(panel, (0, top + height), (width - 1, height + top + bottom - 1), (25, 31, 35), -1)
    put_text(panel, model_label, (16, 24), 0.62, model_color, 2)
    put_text(
        panel,
        f"case PCK@32  all {summary['all']:.1f}%  object {summary['object']:.1f}%  bg {summary['background']:.1f}%",
        (16, 46),
        0.48,
    )

    anchors = track["anchors"]
    anchor_lookup = {int(pixel_frame): latent for latent, pixel_frame in enumerate(anchors)}
    latent = anchor_lookup.get(frame_index)
    y0 = top
    if latent is None:
        cv2.rectangle(panel, (12, y0 + 12), (285, y0 + 43), (20, 20, 20), -1)
        put_text(panel, "non-anchor frame: excluded from PCK", (22, y0 + 34), 0.48, (170, 180, 188))
    elif latent < CLEAN_PREFIX_LATENTS:
        label = "context anchor" if latent == 0 else "query anchor: region points"
        cv2.rectangle(panel, (12, y0 + 12), (315, y0 + 43), (20, 20, 20), -1)
        put_text(panel, f"T{latent} / F{frame_index}: {label}", (22, y0 + 34), 0.48)
        for point_id in range(16):
            x, y = np.rint(track["gt"][latent, point_id]).astype(int)
            color = REGION_COLORS[0 if point_id < 8 else 1]
            cv2.circle(panel, (x, y + y0), 6, color, -1, cv2.LINE_AA)
            cv2.circle(panel, (x, y + y0), 8, (250, 250, 250), 1, cv2.LINE_AA)
    else:
        valid_now = track["valid"][latent]
        error_now = track["error"][latent]
        current_values = error_now[valid_now]
        current_pck = pck(current_values)
        hits = int(np.sum(current_values <= THRESHOLD_PX))
        cv2.rectangle(panel, (12, y0 + 12), (390, y0 + 43), (20, 20, 20), -1)
        put_text(
            panel,
            f"T{latent} / F{frame_index}: evaluated  PCK@32 {hits}/{len(current_values)} = {current_pck:.1f}%",
            (22, y0 + 34),
            0.48,
        )
        for point_id in range(16):
            if not valid_now[point_id]:
                continue
            region_color = REGION_COLORS[0 if point_id < 8 else 1]
            gt_xy = np.rint(track["gt"][latent, point_id]).astype(int)
            pred_xy = np.rint(track["pred"][latent, point_id]).astype(int)
            gt_point = (int(gt_xy[0]), int(gt_xy[1] + y0))
            pred_point = (int(pred_xy[0]), int(pred_xy[1] + y0))
            is_hit = error_now[point_id] <= THRESHOLD_PX
            hit_color = (76, 221, 123) if is_hit else (74, 84, 239)
            gt_history, pred_history = [], []
            for prior in range(CLEAN_PREFIX_LATENTS, latent + 1):
                if track["valid"][prior, point_id]:
                    gx, gy = np.rint(track["gt"][prior, point_id]).astype(int)
                    px, py = np.rint(track["pred"][prior, point_id]).astype(int)
                    gt_history.append((int(gx), int(gy + y0)))
                    pred_history.append((int(px), int(py + y0)))
            draw_polyline(panel, gt_history, (245, 245, 245))
            draw_polyline(panel, pred_history, region_color)
            cv2.circle(panel, gt_point, int(THRESHOLD_PX), hit_color, 1, cv2.LINE_AA)
            cv2.line(panel, gt_point, pred_point, hit_color, 1, cv2.LINE_AA)
            cv2.circle(panel, gt_point, 7, (250, 250, 250), 2, cv2.LINE_AA)
            cv2.circle(panel, pred_point, 5, region_color, -1, cv2.LINE_AA)

    legend_y = top + height + 23
    cv2.circle(panel, (20, legend_y - 4), 6, (250, 250, 250), 2, cv2.LINE_AA)
    put_text(panel, "GT + 32px circle", (34, legend_y), 0.43)
    cv2.circle(panel, (190, legend_y - 4), 5, REGION_COLORS[0], -1, cv2.LINE_AA)
    put_text(panel, "Q@K object", (204, legend_y), 0.43)
    cv2.circle(panel, (330, legend_y - 4), 5, REGION_COLORS[1], -1, cv2.LINE_AA)
    put_text(panel, "Q@K background", (344, legend_y), 0.43)
    put_text(panel, "green connector = hit; red = miss", (16, legend_y + 26), 0.43, (190, 202, 208))
    return panel


def render_video(combination: dict, frames: list[np.ndarray], source_fps: float) -> dict:
    block, head = combination["block"], combination["head"]
    model_tracks = []
    model_metrics = {}
    for model_key, model_label, model_color in MODEL_SPECS:
        track = load_tracks(model_key, block, head)
        result = metrics(track)
        model_tracks.append((model_label, model_color, track, result))
        model_metrics[model_key] = result

    output_name = f"{combination['group']}_L{block:02d}_H{head:02d}_S{STEP:03d}.mp4"
    output_path = OUTPUT_ROOT / output_name
    first = render_panel(frames[0], 0, *model_tracks[0])
    out_height, panel_width = first.shape[:2]
    out_width = panel_width * len(model_tracks)
    fps = min(max(source_fps, 4.0), 8.0)
    command = [
        FFMPEG, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{out_width}x{out_height}", "-r", f"{fps:.6f}", "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame_index, frame in enumerate(frames):
            panels = [
                render_panel(frame, frame_index, model_label, color, track, result)
                for model_label, color, track, result in model_tracks
            ]
            process.stdin.write(np.concatenate(panels, axis=1).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
    except BrokenPipeError:
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        process.wait()
        raise RuntimeError(stderr)
    if return_code:
        raise RuntimeError(f"ffmpeg failed ({return_code}): {stderr}")
    return {**combination, "video": output_name, "models": model_metrics}


def fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.1f}%"


def card_html(item: dict) -> str:
    rows = "".join(
        f"<tr><td>{label}</td><td>{fmt(item['models'][key]['all'])}</td>"
        f"<td>{fmt(item['models'][key]['object'])}</td><td>{fmt(item['models'][key]['background'])}</td>"
        f"<td>{item['models'][key]['comparisons']}</td></tr>"
        for key, label, _ in MODEL_SPECS
    )
    return f"""
    <article class="card">
      <div class="card-head"><div><span class="rank">{item['group'].upper()} {item['rank']}</span>
      <h3>L{item['block']:02d} / H{item['head']:02d}</h3></div>
      <div class="score"><b>{item['ranking_pck32']:.2f}%</b><small>综合排名 PCK@32</small></div></div>
      <video controls muted loop playsinline preload="metadata" src="/pck-overlay-comparison/assets/{item['video']}"></video>
      <table><thead><tr><th>模型</th><th>全部</th><th>目标</th><th>背景</th><th>有效比较</th></tr></thead><tbody>{rows}</tbody></table>
    </article>"""


def write_page(items: list[dict], video_path: str) -> None:
    top_cards = "".join(card_html(item) for item in items if item["group"] == "top")
    bottom_cards = "".join(card_html(item) for item in items if item["group"] == "bottom")
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PCK@32 轨迹核验</title>
<style>
:root{{--ink:#17201d;--paper:#f4f0e7;--card:#fffdf7;--line:#c8c1b3;--green:#176b52;--red:#a33d31;}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:radial-gradient(circle at 10% 5%,#dce9d7 0,transparent 28%),linear-gradient(135deg,#f7f1e4,#e9e2d6);font-family:"Noto Sans CJK SC","Source Han Sans SC",sans-serif}}
header{{padding:42px clamp(20px,5vw,72px) 30px;border-bottom:1px solid var(--line)}}h1{{font-family:"Noto Serif CJK SC","Source Han Serif SC",serif;font-size:clamp(32px,5vw,68px);line-height:1;margin:0 0 16px}}header p{{max-width:1000px;line-height:1.8;margin:0;color:#48534f}}main{{padding:28px clamp(16px,4vw,64px) 64px}}.method{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);margin-bottom:38px}}.method div{{background:var(--card);padding:18px}}.method b{{display:block;color:var(--green);margin-bottom:8px}}.method code{{font-size:15px}}h2{{font-family:"Noto Serif CJK SC","Source Han Serif SC",serif;font-size:32px;margin:36px 0 16px}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}}.card{{background:var(--card);border:1px solid var(--line);box-shadow:0 12px 30px #3d443b18}}.card-head{{display:flex;align-items:center;justify-content:space-between;padding:16px 18px}}.card h3{{font-size:24px;margin:5px 0 0}}.rank{{font:700 12px monospace;color:var(--green);letter-spacing:.12em}}.score{{text-align:right}}.score b{{display:block;font:700 22px monospace}}.score small{{color:#69716d}}video{{display:block;width:100%;background:#111;aspect-ratio:4.31/1;object-fit:contain}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:9px 10px;border-top:1px solid #ddd6c9;text-align:right}}th:first-child,td:first-child{{text-align:left}}.note{{padding:16px 18px;color:#59625e;font-size:13px;line-height:1.7}}a{{color:var(--green)}}
@media(max-width:1100px){{.grid{{grid-template-columns:1fr}}.method{{grid-template-columns:1fr 1fr}}}}@media(max-width:620px){{.method{{grid-template-columns:1fr}}header{{padding-top:28px}}}}
</style></head><body><header><a href="/">返回实验总览</a><h1>PCK@32 轨迹核验</h1>
<p>同一个原视频上并排展示 GT teacher-forced、LoRA 和 Wan2.2 Baseline。案例为 <b>{CASE_KEY}</b>，扩散步为 <b>S039</b>。排名值来自三个模型、5 个 case 的综合表；卡片表格是当前 case 的实际重算值。</p></header><main>
<section class="method"><div><b>1. 比较对象</b>Q@K argmax 预测坐标与 CoTracker pseudo-GT 坐标。</div><div><b>2. 有效条件</b>点在 query 帧和目标帧均可见。</div><div><b>3. 时间范围</b>只统计 T2-T6，对应原视频 F8/F12/F16/F20/F24。</div><div><b>4. 判定公式</b><code>PCK@32 = mean(||p_pred-p_gt||2 <= 32px)</code></div></section>
<p class="note">视频中白圈中心是 pseudo-GT，半径 32 px 的圆就是命中区域；蓝/橙实心点分别是 object/background 的 Q@K 预测。连接线为绿色表示命中、红色表示未命中。F0/F4 属于 context/query，其他非 anchor 帧都不进入分母。</p>
<h2>高 PCK 组合</h2><section class="grid">{top_cards}</section>
<h2>低 PCK 组合</h2><section class="grid">{bottom_cards}</section>
<p class="note">原视频：{video_path}<br>完整机器可读记录：<a href="/pck-overlay-comparison/assets/catalog.json">catalog.json</a></p>
</main></body></html>"""
    (OUTPUT_ROOT / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frames, fps, video_path = load_original_frames()
    items = [render_video(combo, frames, fps) for combo in read_ranked_combinations()]
    catalog = {
        "case": CASE_KEY,
        "step": STEP,
        "threshold_px": THRESHOLD_PX,
        "query_latent_index": QUERY_LATENT_INDEX,
        "clean_prefix_latents": CLEAN_PREFIX_LATENTS,
        "source_video": video_path,
        "items": items,
    }
    (OUTPUT_ROOT / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    write_page(items, video_path)
    print(json.dumps({"output": str(OUTPUT_ROOT), "videos": len(items)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
