#!/usr/bin/env python3
"""Render selected xSSC object cross-attention layers from saved layer maps."""
from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


FFMPEG = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg")


def _read_video_bgr(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 8.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video has no frames: {path}")
    return frames, fps


def _temporal_resize_lowres(attn: np.ndarray, output_frames: int) -> np.ndarray:
    tensor = torch.from_numpy(attn.astype(np.float32))[None, None]
    resized = F.interpolate(
        tensor,
        size=(int(output_frames), int(attn.shape[1]), int(attn.shape[2])),
        mode="trilinear",
        align_corners=True,
    )
    return resized[0, 0].numpy()


def _heat_overlay(frame: np.ndarray, heat_lowres: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = np.clip((heat_lowres - low) / max(high - low, 1.0e-12), 0.0, 1.0)
    heat = cv2.resize(
        (normalized * 255.0).astype(np.uint8),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    return cv2.addWeighted(frame, 0.56, color, 0.44, 0.0)


def _label(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    label_height = 26 * len(lines) + 8
    out = cv2.copyMakeBorder(
        frame,
        label_height,
        0,
        0,
        0,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    for index, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (8, 22 + 26 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _write_h264(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError(f"cannot write empty video: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".temporary.mp4")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {temporary}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(temporary),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
    )
    temporary.unlink()


def _insert_page_section(attention_dir: Path, layer_sections: list[str]) -> None:
    index_path = attention_dir / "index.html"
    page = index_path.read_text(encoding="utf-8")
    start = "<!-- manual-layer-visualization:start -->"
    end = "<!-- manual-layer-visualization:end -->"
    section = f"{start}\n{''.join(layer_sections)}\n{end}\n"
    if start in page and end in page:
        page = page.split(start, 1)[0] + section + page.split(end, 1)[1]
    elif "<section><h2>Layer-Mean Reference</h2>" in page:
        before, after = page.split("<section><h2>Layer-Mean Reference</h2>", 1)
        page = before + section + "<section><h2>Layer-Mean Reference</h2>" + after
    else:
        page = page.replace("</main>", section + "</main>")
    index_path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-dir", type=Path, required=True)
    parser.add_argument("--layers", required=True, help="Comma-separated layer ids, e.g. 11,29")
    parser.add_argument("--stage", default="all")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--update-html", action="store_true")
    args = parser.parse_args()

    attention_dir = args.attention_dir.expanduser().resolve()
    summary = json.loads((attention_dir / "summary.json").read_text(encoding="utf-8"))
    generated_video = Path(str(summary["generated_video"]))
    layer_npz = attention_dir / str(summary["layer_maps_npz"])
    frames, measured_fps = _read_video_bgr(generated_video)
    fps = int(args.fps) if int(args.fps) > 0 else int(round(measured_fps))
    stage = str(args.stage)
    layers = [int(part.strip()) for part in str(args.layers).split(",") if part.strip()]

    outputs: dict[str, list[dict[str, int | str]]] = {}
    layer_sections: list[str] = []
    with np.load(layer_npz) as payload:
        for layer_id in layers:
            layer_dir = attention_dir / f"{stage}_layer{layer_id:02d}"
            videos: list[dict[str, int | str]] = []
            figures: list[str] = []
            for slot_id in range(int(summary["slot_count"])):
                key = f"{stage}_layer{layer_id:02d}_slot{slot_id:02d}"
                if key not in payload:
                    raise KeyError(f"missing map {key} in {layer_npz}")
                lowres = payload[key].astype(np.float32)
                aligned = _temporal_resize_lowres(lowres, len(frames))
                low, high = np.percentile(aligned, [2.0, 99.0]).tolist()
                rendered = []
                for frame_id, frame in enumerate(frames):
                    overlay = _heat_overlay(frame, aligned[frame_id], float(low), float(high))
                    rendered.append(
                        _label(
                            overlay,
                            [
                                f"xSSC object cross-attn | {stage} | layer {layer_id:02d} | slot {slot_id:02d}",
                                f"generated frame={frame_id:02d}",
                            ],
                        )
                    )
                video_name = f"slot{slot_id:02d}_{stage}_layer{layer_id:02d}_object_cross_attention.mp4"
                video_path = layer_dir / video_name
                _write_h264(video_path, rendered, fps=fps)
                rel = f"{layer_dir.name}/{video_name}"
                videos.append({"slot": slot_id, "video": rel})
                figures.append(
                    "<figure>"
                    f"<video controls muted loop src='{html.escape(rel)}'></video>"
                    f"<figcaption>slot {slot_id:02d}</figcaption>"
                    "</figure>"
                )
            outputs[f"{stage}_layer{layer_id:02d}"] = videos
            layer_sections.append(
                "<section>"
                f"<h2>{html.escape(stage)} layer {layer_id:02d}</h2>"
                "<p>Manual layer visualization from saved per-layer object cross-attention maps.</p>"
                f"<div class='grid'>{''.join(figures)}</div>"
                "</section>"
            )

    manifest = {
        "attention_dir": str(attention_dir),
        "stage": stage,
        "layers": layers,
        "generated_video": str(generated_video),
        "outputs": outputs,
    }
    manifest_path = attention_dir / f"{stage}_manual_layers_{'_'.join(str(v) for v in layers)}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.update_html:
        _insert_page_section(attention_dir, layer_sections)
    print(json.dumps({"manifest": str(manifest_path), "updated_html": bool(args.update_html)}, indent=2))


if __name__ == "__main__":
    main()
