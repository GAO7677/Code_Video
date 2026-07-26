#!/usr/bin/env python3
"""Build a static gallery for completed moving-query attention cases."""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from motion_query_map import _center_crop_resize, _read_video
from moving_query_attention import FEATURE_NAMES, moving_query_coords


MODELS = ("wan_lora", "xssc", "physrvg")
MODEL_LABELS = {
    "wan_lora": "Wan + LoRA",
    "xssc": "Wan + xSSC",
    "physrvg": "PhysRVG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--query-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _query_rect(
    coords: tuple[tuple[int, int, int], ...], latent_time: int
) -> tuple[int, int, int, int]:
    current = [coord for coord in coords if coord[0] == latent_time]
    rows = [coord[1] for coord in current]
    columns = [coord[2] for coord in current]
    return (
        min(columns) * 896 // 28,
        min(rows) * 512 // 16,
        (max(columns) + 1) * 896 // 28,
        (max(rows) + 1) * 512 // 16,
    )


def _render_query_assets(item: dict, output_dir: Path, case: str) -> tuple[Path, Path]:
    video_path = output_dir / f"{case}_moving_query.mp4"
    sheet_path = output_dir / f"{case}_moving_query_contact.jpg"
    if video_path.is_file() and sheet_path.is_file():
        return video_path, sheet_path
    frames = [
        _center_crop_resize(frame)
        for frame in _read_video(Path(item["source_video"]))
    ]
    coords = moving_query_coords(
        item["trajectory"],
        frame_shape=tuple(int(value) for value in item["frame_shape"]),
    )
    rendered = []
    for frame_index in range(49):
        source = frames[min(frame_index, len(frames) - 1)].copy()
        latent_time = min(12, int(round(frame_index / 4)))
        x0, y0, x1, y1 = _query_rect(coords, latent_time)
        cv2.rectangle(source, (x0, y0), (x1, y1), (40, 245, 90), 4)
        cv2.putText(
            source,
            f"video f={frame_index:02d}  latent t={latent_time:02d}",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rendered.append(source)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (896, 512),
    )
    for frame in rendered:
        writer.write(frame)
    writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        encoded_path = video_path.with_suffix(".h264.mp4")
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(encoded_path),
            ],
            check=True,
        )
        encoded_path.replace(video_path)
    thumbnails = [rendered[4 * time] for time in range(13)]
    thumb_w, thumb_h = 336, 192
    sheet = np.full((3 * thumb_h, 5 * thumb_w, 3), 24, dtype=np.uint8)
    for index, frame in enumerate(thumbnails):
        resized = cv2.resize(frame, (thumb_w, thumb_h))
        row, column = divmod(index, 5)
        sheet[row * thumb_h : (row + 1) * thumb_h, column * thumb_w : (column + 1) * thumb_w] = resized
    cv2.imwrite(str(sheet_path), sheet)
    return video_path, sheet_path


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values, axis=-1), axis=-1)
    return order.astype(np.float32) / max(1, values.shape[-1] - 1)


def _render_feature_heatmap(summary_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    panels = []
    steps = []
    for entry in summary["steps"]:
        with np.load(
            summary_path.parent / entry["directory"] / entry["features_npz"]
        ) as arrays:
            matrix = np.stack([arrays[name] for name in FEATURE_NAMES])
        panels.append(_rank01(matrix))
        steps.append(int(entry["step_number_one_based"]))
    figure, axes = plt.subplots(1, len(panels), figsize=(22, 5.4), sharey=True)
    for axis, panel, step in zip(np.atleast_1d(axes), panels, steps):
        image = axis.imshow(panel, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_title(f"Denoise step {step}")
        axis.set_xlabel("Head")
        axis.set_xticks(range(24))
        axis.tick_params(axis="x", labelsize=7)
    axes = np.atleast_1d(axes)
    axes[0].set_yticks(range(len(FEATURE_NAMES)))
    axes[0].set_yticklabels(FEATURE_NAMES, fontsize=8)
    figure.colorbar(image, ax=axes.tolist(), label="Within-block head rank", shrink=0.82)
    figure.suptitle(
        f"{summary['model']} · {summary['case']} · Block {summary['block_id']:02d}",
        fontsize=13,
    )
    figure.subplots_adjust(left=0.10, right=0.94, bottom=0.12, top=0.86, wspace=0.12)
    figure.savefig(output_path, dpi=145)
    plt.close(figure)


def _generated_video(root: Path, model: str, case: str) -> Path | None:
    matches = sorted((root / "generated" / model).glob(f"**/{case}.mp4"))
    return matches[0] if matches else None


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    query_map = json.loads(
        args.query_map.expanduser().resolve().read_text(encoding="utf-8")
    )["cases"]
    completed: dict[str, list[tuple[str, Path]]] = {}
    for model in MODELS:
        model_root = root / "block17" / "matrices" / model
        for summary_path in sorted(model_root.glob("*/summary.json")):
            completed.setdefault(summary_path.parent.name, []).append(
                (model, summary_path)
            )

    sections = []
    for case in query_map:
        entries = completed.get(case, [])
        if not entries:
            continue
        query_video, query_sheet = _render_query_assets(
            query_map[case], assets, case
        )
        method_cards = []
        for model, summary_path in entries:
            heatmap = assets / f"{case}_{model}_block17_features.png"
            _render_feature_heatmap(summary_path, heatmap)
            generated = _generated_video(root, model, case)
            video_html = (
                f"<video controls preload='metadata' src='../{html.escape(str(generated.relative_to(root)))}'></video>"
                if generated is not None
                else "<p class='missing'>Generated video is still being written.</p>"
            )
            method_cards.append(
                f"""<article>
<h3>{html.escape(MODEL_LABELS[model])}</h3>
{video_html}
<a href='assets/{heatmap.name}'><img loading='lazy' src='assets/{heatmap.name}'></a>
</article>"""
            )
        sections.append(
            f"""<section>
<h2>{html.escape(case)}</h2>
<div class="query">
<div><h3>Per-frame moving query</h3><video controls loop muted preload="metadata" src="assets/{query_video.name}"></video></div>
<a href="assets/{query_sheet.name}"><img loading="lazy" src="assets/{query_sheet.name}"></a>
</div>
<div class="methods">{''.join(method_cards)}</div>
</section>"""
        )
    counts = {
        model: sum(model == entry[0] for entries in completed.values() for entry in entries)
        for model in MODELS
    }
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Moving-query attention progress</title>
<style>
body{{margin:0;background:#f2f3f0;color:#202421;font:14px Arial,sans-serif}}
header,main{{max-width:1800px;margin:auto;padding:18px 24px}}header{{background:#202421;color:white;max-width:none}}
h1,h2,h3{{letter-spacing:0}}h1{{margin:0 0 7px}}header p{{margin:0;color:#cfd6d0}}
section{{border-top:2px solid #202421;margin:26px 0 42px;padding-top:14px}}
.query{{display:grid;grid-template-columns:minmax(360px,0.8fr) minmax(600px,1.7fr);gap:14px;align-items:start}}
.methods{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:16px}}
article{{background:white;border:1px solid #c9cec9;padding:10px;border-radius:4px}}
video,img{{display:block;width:100%;height:auto;background:#111}}h3{{margin:0 0 8px;font-size:15px}}
.missing{{height:200px;display:grid;place-items:center;background:#e4e6e2;color:#666}}
@media(max-width:1000px){{.query,.methods{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>Per-frame moving-object query attention</h1>
<p>Completed Block17 results: Wan+LoRA {counts['wan_lora']}/20 · Wan+xSSC {counts['xssc']}/20 · PhysRVG {counts['physrvg']}/20. Green boxes follow the object at each latent time.</p></header>
<main>{''.join(sections)}</main></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    print(output / "index.html")


if __name__ == "__main__":
    main()
