#!/usr/bin/env python3
"""Overlay moving-query attention from classified heads on source-video frames."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from motion_query_map import _center_crop_resize, _read_video


MODELS = ("wan_lora", "xssc", "physrvg")
MODEL_LABELS = {
    "wan_lora": "Wan + LoRA",
    "xssc": "Wan + xSSC",
    "physrvg": "PhysRVG",
}
ROLE_ORDER = ("S", "T", "P", "C", "G")
ROLE_ENGLISH = {
    "S": "within-frame spatial",
    "T": "trajectory propagation",
    "P": "fixed-position temporal",
    "C": "history/context",
    "G": "global aggregation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-root", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--query-map", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--block", type=int, default=17)
    parser.add_argument("--step", type=int, default=35)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _query_rect(
    coords: np.ndarray, latent_time: int, frame_h: int, frame_w: int
) -> tuple[int, int, int, int]:
    current = coords[coords[:, 0] == latent_time]
    if not len(current):
        raise ValueError(f"no moving-query token at latent time {latent_time}")
    rows, columns = current[:, 1], current[:, 2]
    return (
        int(columns.min()) * frame_w // 28,
        int(rows.min()) * frame_h // 16,
        (int(columns.max()) + 1) * frame_w // 28,
        (int(rows.max()) + 1) * frame_h // 16,
    )


def _normalize_maps(maps: np.ndarray) -> np.ndarray:
    values = np.log10(np.maximum(maps.astype(np.float32), 1.0e-12))
    low, high = np.percentile(values, (5.0, 99.5))
    if high <= low:
        high = low + 1.0
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _attention_overlay(frame: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heatmap = cv2.resize(
        heatmap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC
    )
    color = cv2.applyColorMap(
        np.asarray(np.clip(heatmap * 255.0, 0, 255), dtype=np.uint8),
        cv2.COLORMAP_TURBO,
    )
    alpha = (0.15 + 0.55 * heatmap)[..., None]
    return np.asarray(frame * (1.0 - alpha) + color * alpha, dtype=np.uint8)


def _label_panel(
    frame: np.ndarray,
    title: str,
    subtitle: str,
    query_rect: tuple[int, int, int, int] | None,
) -> np.ndarray:
    output = frame.copy()
    if query_rect is not None:
        x0, y0, x1, y1 = query_rect
        cv2.rectangle(output, (x0, y0), (x1, y1), (40, 245, 90), 3)
    cv2.rectangle(output, (0, 0), (output.shape[1], 64), (18, 18, 18), -1)
    cv2.putText(
        output,
        title,
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        subtitle,
        (14, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (210, 218, 212),
        1,
        cv2.LINE_AA,
    )
    return output


def _encode_h264(raw_path: Path, final_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        candidate = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg")
        ffmpeg = str(candidate) if candidate.is_file() else None
    if ffmpeg is None:
        raw_path.replace(final_path)
        return
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(raw_path),
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
            str(final_path),
        ],
        check=True,
    )
    raw_path.unlink()


def _render_head(
    *,
    source_frames: list[np.ndarray],
    attention: np.ndarray,
    query_coords: np.ndarray,
    model: str,
    role: str,
    head: int,
    block: int,
    step: int,
    output_stem: Path,
) -> tuple[Path, Path]:
    video_path = output_stem.with_suffix(".mp4")
    sheet_path = output_stem.with_suffix(".jpg")
    if video_path.stat().st_size > 1024 if video_path.is_file() else False:
        if sheet_path.stat().st_size > 1024 if sheet_path.is_file() else False:
            return video_path, sheet_path
    output_stem.with_suffix(".raw.mp4").unlink(missing_ok=True)
    sync = np.stack([attention[t, t] for t in range(13)])
    cross = np.stack(
        [
            np.delete(attention[:, key_time], key_time, axis=0).mean(axis=0)
            for key_time in range(13)
        ]
    )
    sync = _normalize_maps(sync)
    cross = _normalize_maps(cross)

    frame_h, frame_w = source_frames[0].shape[:2]
    panel_w, panel_h = frame_w // 2, frame_h // 2
    output_size = (panel_w * 3, panel_h)
    raw_video = output_stem.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(
        str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, output_size
    )
    rendered_latent_frames = []
    for frame_index in range(49):
        latent_time = min(12, int(round(frame_index / 4)))
        source = source_frames[min(frame_index, len(source_frames) - 1)]
        rect = _query_rect(query_coords, latent_time, frame_h, frame_w)
        subtitle = (
            f"Block {block} H{head} {role} | denoise {step} | "
            f"video {frame_index:02d} / latent {latent_time:02d}"
        )
        original = _label_panel(source, "Original + moving query", subtitle, rect)
        sync_frame = _label_panel(
            _attention_overlay(source, sync[latent_time]),
            "Same-time attention A(q_t, k_t)",
            subtitle,
            rect,
        )
        cross_frame = _label_panel(
            _attention_overlay(source, cross[latent_time]),
            "Cross-time mean A(q_s, k_t), s != t",
            subtitle,
            rect,
        )
        combined = np.concatenate(
            [
                cv2.resize(panel, (panel_w, panel_h))
                for panel in (original, sync_frame, cross_frame)
            ],
            axis=1,
        )
        writer.write(combined)
        if frame_index % 4 == 0 or frame_index == 48:
            rendered_latent_frames.append(combined)
    writer.release()
    _encode_h264(raw_video, video_path)

    thumb_w, thumb_h = output_size[0] // 2, output_size[1] // 2
    sheet = np.full((3 * thumb_h, 5 * thumb_w, 3), 24, dtype=np.uint8)
    for index, frame in enumerate(rendered_latent_frames[:13]):
        row, column = divmod(index, 5)
        sheet[
            row * thumb_h : (row + 1) * thumb_h,
            column * thumb_w : (column + 1) * thumb_w,
        ] = cv2.resize(frame, (thumb_w, thumb_h))
    cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 91])
    return video_path, sheet_path


def main() -> None:
    args = parse_args()
    maps_root = args.maps_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    classification = json.loads(
        args.classification.expanduser().resolve().read_text(encoding="utf-8")
    )
    query_map = json.loads(
        args.query_map.expanduser().resolve().read_text(encoding="utf-8")
    )["cases"]
    item = query_map[args.case]
    source_frames = [
        _center_crop_resize(frame)
        for frame in _read_video(Path(item["source_video"]))
    ]
    if not source_frames:
        raise RuntimeError(f"source video has no frames: {item['source_video']}")

    sections = []
    manifest = {
        "case": args.case,
        "block": args.block,
        "denoise_step": args.step,
        "source_video": item["source_video"],
        "models": {},
    }
    for model in MODELS:
        summary_path = (
            maps_root
            / f"block{args.block:02d}"
            / "matrices"
            / model
            / args.case
            / "summary.json"
        )
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        step_entry = next(
            entry
            for entry in summary["steps"]
            if int(entry["step_number_one_based"]) == args.step
        )
        npz_path = summary_path.parent / step_entry["directory"] / step_entry["maps_npz"]
        with np.load(npz_path) as arrays:
            maps = arrays["attention"].astype(np.float32)
            selected_heads = arrays["selected_heads"].astype(int)
            query_coords = arrays["query_coords"].astype(int)
        head_to_index = {
            int(head): index for index, head in enumerate(selected_heads.tolist())
        }
        cards = []
        manifest["models"][model] = {}
        for role in ROLE_ORDER:
            head = int(
                classification["representatives"][model]["unique"][role]
            )
            stem = assets / f"{model}_{role}_head{head:02d}"
            video_path, sheet_path = _render_head(
                source_frames=source_frames,
                attention=maps[head_to_index[head]],
                query_coords=query_coords,
                model=model,
                role=role,
                head=head,
                block=args.block,
                step=args.step,
                output_stem=stem,
            )
            manifest["models"][model][role] = {
                "head": head,
                "role": ROLE_ENGLISH[role],
                "video": str(video_path.relative_to(output)),
                "contact_sheet": str(sheet_path.relative_to(output)),
            }
            cards.append(
                f"""<article>
<h3>{role} / Head {head}: {html.escape(ROLE_ENGLISH[role])}</h3>
<video controls loop muted preload="metadata" src="assets/{video_path.name}"></video>
<a href="assets/{sheet_path.name}"><img loading="lazy" src="assets/{sheet_path.name}"></a>
</article>"""
            )
        sections.append(
            f"""<section>
<h2>{html.escape(MODEL_LABELS[model])}</h2>
<div class="heads">{''.join(cards)}</div>
</section>"""
        )

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Moving-query head overlays</title>
<style>
body{{margin:0;background:#f1f3f0;color:#202421;font:14px Arial,sans-serif}}
header,main{{max-width:1800px;margin:auto;padding:18px 24px}}
header{{max-width:none;background:#202421;color:#fff}}
h1,h2,h3{{letter-spacing:0}}h1{{margin:0 0 8px}}header p{{margin:5px 0;color:#d1d7d2}}
section{{margin:28px 0 44px;border-top:2px solid #202421;padding-top:12px}}
.heads{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}
article{{background:#fff;border:1px solid #c9cec9;border-radius:4px;padding:10px}}
h3{{font-size:15px;margin:0 0 9px}}video,img{{width:100%;height:auto;display:block;background:#111}}
img{{margin-top:8px}}code{{color:#bde5c6}}
@media(max-width:1000px){{.heads{{grid-template-columns:1fr}}header,main{{padding:14px}}}}
</style></head><body>
<header><h1>Moving-query classified-head overlays</h1>
<p>{html.escape(args.case)} · Block {args.block} · denoise step {args.step}</p>
<p>Green box: current-frame moving-object query tokens. Left: source frame. Middle:
same-time attention A(q_t,k_t). Right: attention received by frame t from all other
moving queries, mean A(q_s,k_t), s != t. Color is log-attention normalized within
each head and view.</p></header>
<main>{''.join(sections)}</main></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    print(output / "index.html")


if __name__ == "__main__":
    main()
