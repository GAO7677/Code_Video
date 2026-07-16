#!/usr/bin/env python3
"""Visualize fixed train and hand-off clips from both Stage 1 sources."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


TEXTOCVP_ROOT = Path("/home/gaoya/Code_Video/TextOCVP-master")
sys.path.insert(0, str(TEXTOCVP_ROOT / "src"))
os.chdir(TEXTOCVP_ROOT)

from data.Stage1Indexed import Stage1Indexed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-source-split", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def balanced_indices(records, count):
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(record["group"], []).append(index)
    selected = []
    offset = 0
    while len(selected) < count:
        added = False
        for group in sorted(groups):
            values = groups[group]
            if offset < len(values):
                selected.append(values[offset])
                added = True
                if len(selected) == count:
                    return selected
        if not added:
            break
        offset += 1
    return selected


def tensor_video_to_uint8(video):
    return (
        video.clamp(0, 1)
        .permute(0, 2, 3, 1)
        .mul(255)
        .round()
        .byte()
        .numpy()
    )


def write_h264(path, frames, fps):
    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=None,
        ffmpeg_log_level="error",
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def annotate(frames, metadata):
    annotated = []
    for index, frame in enumerate(frames):
        header = np.full((64, frame.shape[1], 3), 247, dtype=np.uint8)
        cv2.putText(
            header,
            f"{metadata['source']} | {metadata['group']} | {metadata['sample_id']}",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            header,
            f"raw frame {metadata['frame_ids'][index]} | start {metadata['start_frame']} | input 216x384",
            (12, 51),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
        annotated.append(np.concatenate([header, frame], axis=0))
    return annotated


def build_html(output_dir, reports):
    sections = []
    for source in ("pybullet", "kubric"):
        for split in ("train", "handoff"):
            cards = []
            for report in reports:
                if report["source"] != source or report["split"] != split:
                    continue
                cards.append(
                    f"""
                    <article>
                      <h3>{html.escape(report['group'])} / {html.escape(report['sample_id'])}</h3>
                      <p><code>{html.escape(report['video_path'])}</code></p>
                      <p>start={report['start_frame']}; frames=<code>{report['frame_ids']}</code></p>
                      <video controls loop muted src="{report['preview_file']}"></video>
                      <a href="{report['exact_file']}">Exact 216x384 training tensor</a>
                    </article>
                    """
                )
            sections.append(
                f"<section><h2>{source.title()} / {split}</h2><div class='grid'>{''.join(cards)}</div></section>"
            )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAVi fixed dataset preview</title>
<style>
body{{margin:0;background:#f3f5f6;color:#17202a;font-family:"IBM Plex Sans","Noto Sans",sans-serif}}header,section{{padding:20px 28px;border-bottom:1px solid #ccd2d7}}h1,h2,h3{{letter-spacing:0}}h1{{margin:0 0 8px}}h2{{margin:0 0 14px}}h3{{font-size:15px;margin:0 0 7px}}p{{font-size:12px;overflow-wrap:anywhere}}code{{font-family:"IBM Plex Mono",monospace}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}article{{border-top:3px solid #a33a2b;padding-top:10px}}video{{display:block;width:100%;background:#111;margin:10px 0 7px}}a{{font-size:13px;color:#174f7a}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}header,section{{padding:16px}}}}
</style></head><body><header><h1>Fixed SAVi Stage 1 train and hand-off clips</h1><p>All clips use raw frames 0-49, 10 contiguous frames, stride 1, and exact model input HxW=216x384.</p></header>{''.join(sections)}</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    reports = []
    for source in ("pybullet", "kubric"):
        for split_name, dataset_split in (("train", "train"), ("handoff", "valid")):
            dataset = Stage1Indexed(
                index_root=args.index_root,
                dataset_mode=source,
                split=dataset_split,
                num_frames=10,
                img_size=(216, 384),
                frame_stride=1,
                random_start=True,
            )
            indices = balanced_indices(dataset.records, args.samples_per_source_split)
            for local_index, dataset_index in enumerate(indices):
                video, metadata = dataset[dataset_index]
                frames = tensor_video_to_uint8(video)
                stem = f"{source}_{split_name}_{local_index:02d}_{metadata['group']}_{metadata['sample_id']}"
                exact_file = f"{stem}_exact.mp4"
                preview_file = f"{stem}_preview.mp4"
                write_h264(args.output_dir / exact_file, frames, fps=30)
                write_h264(args.output_dir / preview_file, annotate(frames, metadata), fps=10)
                reports.append(
                    {
                        "source": source,
                        "split": split_name,
                        "group": metadata["group"],
                        "sample_id": metadata["sample_id"],
                        "video_path": metadata["video_path"],
                        "start_frame": metadata["start_frame"],
                        "frame_ids": metadata["frame_ids"],
                        "exact_file": exact_file,
                        "preview_file": preview_file,
                    }
                )
    (args.output_dir / "preview_manifest.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    build_html(args.output_dir, reports)
    print(json.dumps(reports, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
