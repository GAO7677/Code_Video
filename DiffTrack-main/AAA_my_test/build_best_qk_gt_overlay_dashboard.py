#!/usr/bin/env python3
"""Render GT/CoTracker and best Q@K tracks on a synchronized four-model video grid."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


OUTPUTS = Path("/data/gaoya/agent-data/outputs")
DEFAULT_OUTPUT = OUTPUTS / "three_model_best_qk_gt_overlay"
MODELS = {
    "gt": {
        "label": "GT teacher-forced",
        "root": OUTPUTS / "wan22_ti2v_5b_gt_real_sam2_regions_steps40",
        "video": "gt.mp4",
        "layer": 5,
        "step": 29,
    },
    "stage1b": {
        "label": "Stage1b step-004000",
        "root": OUTPUTS / "stage1b_kubric_step004000_sam2_regions_steps40",
        "video": "generated.mp4",
        "layer": 23,
        "step": 39,
    },
    "lora": {
        "label": "LoRA step-000500",
        "root": OUTPUTS / "wan_openvid_0613pybullet_lora_step000500_sam2_regions_steps40",
        "video": "generated.mp4",
        "layer": 5,
        "step": 39,
    },
    "baseline": {
        "label": "Wan2.2 baseline",
        "root": OUTPUTS / "wan22_ti2v_5b_baseline_sam2_regions_steps40",
        "video": "generated.mp4",
        "layer": 23,
        "step": 39,
    },
}
MODEL_ORDER = ("gt", "stage1b", "lora", "baseline")
PANEL_SIZE = (640, 360)
QUERY_FRAME = 4
TRAIL_LENGTH = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def point_colors(count: int) -> list[tuple[int, int, int]]:
    colors = []
    for index in range(count):
        hue = int(round(179 * index / max(count, 1)))
        value = cv2.cvtColor(
            np.uint8([[[hue, 210, 245]]]), cv2.COLOR_HSV2BGR
        )[0, 0]
        colors.append(tuple(int(channel) for channel in value))
    return colors


def draw_dashed_line(
    canvas: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    dash: float = 8.0,
) -> None:
    delta = end.astype(np.float32) - start.astype(np.float32)
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return
    direction = delta / length
    position = 0.0
    while position < length:
        segment_end = min(position + dash, length)
        p0 = tuple(np.rint(start + direction * position).astype(int))
        p1 = tuple(np.rint(start + direction * segment_end).astype(int))
        cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)
        position += dash * 1.75


def object_slices(manifest: dict) -> list[slice]:
    return [
        slice(int(region["point_start"]), int(region["point_end"]))
        for region in manifest["query_regions"]
        if region["region_type"] == "object"
    ]


def object_indices(manifest: dict) -> np.ndarray:
    slices = object_slices(manifest)
    if not slices:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(
        [np.arange(item.start, item.stop, dtype=np.int64) for item in slices]
    )


def aggregate_object_metrics(case_dir: Path, layer: int, step: int) -> dict:
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    object_names = {
        region["region_name"]
        for region in manifest["query_regions"]
        if region["region_type"] == "object"
    }
    rows = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
    selected = [
        row
        for row in rows
        if row["method"] == "qk"
        and int(row["layer"]) == layer
        and int(row["step_index"]) == step
        and row.get("region_name") in object_names
        and int(row.get("comparisons", 0)) > 0
    ]
    comparisons = sum(int(row["comparisons"]) for row in selected)
    if not comparisons:
        return {"pck32": None, "mean_error_px": None, "comparisons": 0}
    return {
        "pck32": sum(float(row["pck32"]) * int(row["comparisons"]) for row in selected)
        / comparisons,
        "mean_error_px": sum(
            float(row["mean_error_px"]) * int(row["comparisons"]) for row in selected
        )
        / comparisons,
        "comparisons": comparisons,
    }


def load_track_bundle(case_dir: Path, model: dict) -> dict:
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    indices = object_indices(manifest)
    gt = np.load(case_dir / "cotracker_pseudo_gt.npz")
    predictions = np.load(case_dir / "predicted_tracks.npz")
    prediction_key = (
        f"qk_layer{int(model['layer']):02d}_step{int(model['step']):03d}_predictions"
    )
    return {
        "indices": indices,
        "tracks": gt["tracks"][:, indices].astype(np.float32),
        "visibility": gt["visibility"][:, indices].astype(bool),
        "anchors": gt["latent_anchor_frames"].astype(np.int64),
        "predictions": predictions[prediction_key][:, indices].astype(np.float32),
        "metrics": aggregate_object_metrics(
            case_dir, int(model["layer"]), int(model["step"])
        ),
    }


def resize_points(points: np.ndarray, source_wh: tuple[int, int]) -> np.ndarray:
    source_w, source_h = source_wh
    target_w, target_h = PANEL_SIZE
    scale = np.asarray([target_w / source_w, target_h / source_h], dtype=np.float32)
    return points * scale


def draw_panel(
    frame: np.ndarray,
    frame_index: int,
    model: dict,
    bundle: dict,
    colors: list[tuple[int, int, int]],
) -> np.ndarray:
    source_h, source_w = frame.shape[:2]
    canvas = cv2.resize(frame, PANEL_SIZE, interpolation=cv2.INTER_AREA)
    canvas = cv2.addWeighted(canvas, 0.84, np.zeros_like(canvas), 0.16, 0)
    tracks = resize_points(bundle["tracks"], (source_w, source_h))
    predictions = resize_points(bundle["predictions"], (source_w, source_h))
    visibility = bundle["visibility"]
    anchors = bundle["anchors"]
    start_frame = max(QUERY_FRAME, frame_index - TRAIL_LENGTH)

    if frame_index >= QUERY_FRAME:
        for point_index, color in enumerate(colors):
            for current in range(start_frame + 1, min(frame_index + 1, len(tracks))):
                if visibility[current - 1, point_index] and visibility[current, point_index]:
                    p0 = tuple(np.rint(tracks[current - 1, point_index]).astype(int))
                    p1 = tuple(np.rint(tracks[current, point_index]).astype(int))
                    cv2.line(canvas, p0, p1, (8, 12, 10), 5, cv2.LINE_AA)
                    cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)

            visible_anchors = np.flatnonzero((anchors >= QUERY_FRAME) & (anchors <= frame_index))
            for offset in range(1, len(visible_anchors)):
                first = visible_anchors[offset - 1]
                second = visible_anchors[offset]
                draw_dashed_line(
                    canvas,
                    predictions[first, point_index],
                    predictions[second, point_index],
                    (6, 10, 8),
                    5,
                )
                draw_dashed_line(
                    canvas,
                    predictions[first, point_index],
                    predictions[second, point_index],
                    color,
                    2,
                )

            if frame_index < len(tracks) and visibility[frame_index, point_index]:
                point = tuple(np.rint(tracks[frame_index, point_index]).astype(int))
                cv2.circle(canvas, point, 6, (8, 12, 10), -1, cv2.LINE_AA)
                cv2.circle(canvas, point, 4, color, -1, cv2.LINE_AA)
            if len(visible_anchors):
                point = tuple(
                    np.rint(predictions[visible_anchors[-1], point_index]).astype(int)
                )
                cv2.rectangle(
                    canvas,
                    (point[0] - 6, point[1] - 6),
                    (point[0] + 6, point[1] + 6),
                    (8, 12, 10),
                    4,
                )
                cv2.rectangle(
                    canvas,
                    (point[0] - 5, point[1] - 5),
                    (point[0] + 5, point[1] + 5),
                    color,
                    2,
                )

    metrics = bundle["metrics"]
    pck = "NA" if metrics["pck32"] is None else f"{metrics['pck32']:.1f}%"
    error = "NA" if metrics["mean_error_px"] is None else f"{metrics['mean_error_px']:.1f}px"
    cv2.rectangle(canvas, (0, 0), (PANEL_SIZE[0], 62), (14, 20, 18), -1)
    cv2.putText(
        canvas,
        model["label"],
        (14, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (244, 239, 222),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Q@K L{model['layer']:02d}/S{model['step']:03d}  PCK32 {pck}  error {error}",
        (14, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (174, 213, 194),
        1,
        cv2.LINE_AA,
    )
    anchor_index = int(np.searchsorted(anchors, frame_index, side="right") - 1)
    anchor_text = "before query" if anchor_index < 1 else f"Q@K anchor frame {anchors[anchor_index]}"
    cv2.putText(
        canvas,
        f"frame {frame_index:02d} | {anchor_text}",
        (14, PANEL_SIZE[1] - 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (244, 239, 222),
        1,
        cv2.LINE_AA,
    )
    return canvas


def open_video(path: Path) -> tuple[cv2.VideoCapture, float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return capture, fps, frames


def render_case(case_key: str, output_path: Path) -> dict:
    captures = {}
    bundles = {}
    fps_values = []
    frame_counts = []
    metrics = {}
    max_points = 0
    for name in MODEL_ORDER:
        model = MODELS[name]
        case_dir = model["root"] / "cases" / case_key
        bundle = load_track_bundle(case_dir, model)
        capture, fps, frame_count = open_video(case_dir / model["video"])
        captures[name] = capture
        bundles[name] = bundle
        fps_values.append(fps)
        frame_counts.append(frame_count)
        metrics[name] = bundle["metrics"]
        max_points = max(max_points, len(bundle["indices"]))
    colors = point_colors(max_points)
    last_anchor = min(int(bundle["anchors"][-1]) for bundle in bundles.values())
    frame_count = min(min(frame_counts), last_anchor + 1)
    fps = min(fps_values)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "1280x720",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame_index in range(frame_count):
            panels = []
            for name in MODEL_ORDER:
                ok, frame = captures[name].read()
                if not ok or frame is None:
                    raise RuntimeError(f"{case_key}/{name}: failed at frame {frame_index}")
                panels.append(
                    draw_panel(
                        frame,
                        frame_index,
                        MODELS[name],
                        bundles[name],
                        colors[: len(bundles[name]["indices"])],
                    )
                )
            montage = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
            assert encoder.stdin is not None
            encoder.stdin.write(montage.tobytes())
    finally:
        for capture in captures.values():
            capture.release()
        if encoder.stdin is not None:
            encoder.stdin.close()
    return_code = encoder.wait()
    if return_code:
        raise RuntimeError(f"H.264 encoding failed for {case_key}: {return_code}")
    return {"case_key": case_key, "video": f"videos/{output_path.name}", "metrics": metrics}


STYLE = """
:root{--paper:#e9e2d2;--ink:#16201c;--card:#fffdf6;--line:#b9ad96;--rust:#b9442c;--teal:#14695b;--muted:#68716b}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 8% 0,#df765344,transparent 32rem),radial-gradient(circle at 92% 12%,#39947d3d,transparent 30rem),repeating-linear-gradient(90deg,#0000 0 47px,#786f6010 48px),var(--paper);font-family:"Avenir Next","Trebuchet MS",sans-serif}main{width:min(1800px,calc(100% - 28px));margin:auto;padding:32px 0 70px}h1,h2{font-family:Georgia,serif;margin:0}h1{font-size:clamp(46px,7vw,92px);line-height:.88;letter-spacing:-.055em}.eyebrow{color:var(--rust);font-size:12px;font-weight:900;letter-spacing:.17em;text-transform:uppercase}.lead{max-width:1050px;color:var(--muted);line-height:1.65}.toolbar,.case-card,.protocol{background:#fffdf8e8;border:1px solid var(--line);box-shadow:0 16px 44px #1b282119}.toolbar{position:sticky;top:0;z-index:5;padding:12px;margin:22px 0 14px;backdrop-filter:blur(12px)}.toolbar input{width:100%;border:1px solid var(--ink);background:var(--card);padding:12px 14px;font-weight:800}.case-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.case-card{padding:11px;min-width:0}.case-card h2{font-size:20px;margin:2px 2px 9px}.case-card video{display:block;width:100%;background:#080d0b;aspect-ratio:16/9}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;margin-top:8px}.metric{border-left:3px solid var(--teal);padding:5px 7px;background:#edf0e9}.metric span{display:block;color:var(--muted);font-size:9px}.metric b{display:block;font:700 18px/1.15 Georgia;margin:2px 0}.protocol{padding:15px;margin:0 0 14px;color:var(--muted);line-height:1.6}.protocol strong{color:var(--rust)}@media(max-width:1050px){.case-grid{grid-template-columns:1fr}}@media(max-width:600px){.metrics{grid-template-columns:1fr 1fr}}
"""


def build_dashboard(cases: list[dict], output: Path) -> None:
    labels = {name: model["label"] for name, model in MODELS.items()}
    cards = []
    for case in cases:
        metric_cards = []
        for name in MODEL_ORDER:
            metric = case["metrics"][name]
            pck = "NA" if metric["pck32"] is None else f"{metric['pck32']:.1f}%"
            error = (
                "NA"
                if metric["mean_error_px"] is None
                else f"{metric['mean_error_px']:.1f}px"
            )
            metric_cards.append(
                f'<div class="metric"><span>{html.escape(labels[name])}</span>'
                f'<b>{pck}</b><span>error {error}</span></div>'
            )
        cards.append(
            f'<article class="case-card" data-case="{html.escape(case["case_key"].lower())}">'
            f'<h2>{html.escape(case["case_key"])}</h2>'
            f'<video controls loop playsinline preload="metadata" src="{html.escape(case["video"])}"></video>'
            f'<div class="metrics">{"".join(metric_cards)}</div></article>'
        )
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Best Q@K vs GT overlays</title><style>{STYLE}</style></head><body><main><header><div class="eyebrow">50 cases / fixed global-best configurations</div><h1>GT meets<br>Q@K</h1><p class="lead">All cases on one page. Each synchronized 2x2 video overlays its own CoTracker pseudo-GT with Q@K tracks from the global-best configuration recorded in RESULTS.md.</p></header><section class="protocol"><strong>Legend.</strong> Solid trails and circles are CoTracker pseudo-GT. Dashed trails and squares are Q@K argmax tracks. Only object queries are shown. Q@K markers update at latent anchors [0,4,8,12,16,20,24]; no interpolated position is presented as a model prediction.</section><section class="toolbar"><input id="filter" type="search" placeholder="Filter case name..." autocomplete="off"></section><section class="case-grid">{"".join(cards)}</section></main><script>
const input=document.getElementById('filter'),cards=[...document.querySelectorAll('.case-card')];input.addEventListener('input',()=>{{const q=input.value.trim().toLowerCase();cards.forEach(card=>card.hidden=q&&!card.dataset.case.includes(q))}});
</script></body></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "dashboard_data.json").write_text(
        json.dumps({"cases": cases}, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    common_cases = None
    for model in MODELS.values():
        names = {path.name for path in (model["root"] / "cases").glob("case_*")}
        common_cases = names if common_cases is None else common_cases & names
    case_keys = sorted(common_cases or [])
    if not case_keys:
        raise RuntimeError("no common cases found")
    cases = []
    for index, case_key in enumerate(case_keys, start=1):
        video_path = output / "videos" / f"{case_key}.mp4"
        if video_path.exists() and not args.force:
            metrics = {
                name: aggregate_object_metrics(
                    model["root"] / "cases" / case_key,
                    int(model["layer"]),
                    int(model["step"]),
                )
                for name, model in MODELS.items()
            }
            result = {"case_key": case_key, "video": f"videos/{video_path.name}", "metrics": metrics}
        else:
            result = render_case(case_key, video_path)
        cases.append(result)
        print(f"[{index:02d}/{len(case_keys):02d}] {case_key}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    build_dashboard(cases, output)
    print(f"dashboard: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
