#!/usr/bin/env python3
"""Compare fixed-t2 and per-time moving-object query head roles."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from analyze_multiblock_ball_query_heads import (
    ROLE_LABELS,
    _feature_rows,
    _role_scores,
)
from motion_query_map import _read_video
from moving_query_attention import FEATURE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moving-map-root", type=Path, required=True)
    parser.add_argument("--generated-video", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--model", default="wan_lora")
    parser.add_argument("--block", type=int, default=17)
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _classify_features(features_by_step: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    features = {
        name: np.stack([item[name] for item in features_by_step]).mean(axis=0)
        for name in FEATURE_NAMES
    }
    scores = _role_scores(features)
    roles = list(ROLE_LABELS)
    matrix = np.stack([scores[role] for role in roles], axis=1)
    order = np.argsort(matrix, axis=1)
    return {
        "features": features,
        "scores": scores,
        "score_matrix": matrix,
        "primary": np.asarray([roles[int(index)] for index in order[:, -1]]),
        "secondary": np.asarray([roles[int(index)] for index in order[:, -2]]),
        "margin": (
            np.take_along_axis(matrix, order[:, -1:], axis=1)[:, 0]
            - np.take_along_axis(matrix, order[:, -2:-1], axis=1)[:, 0]
        ),
    }


def _read_paired_aggregate(
    root: Path,
    model: str,
    case: str,
    block: int,
    *,
    fixed_time: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = (
        root / f"block{block:02d}" / "matrices" / model / case / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fixed_by_step = []
    moving_by_step = []
    for entry in summary["steps"]:
        npz_path = (
            summary_path.parent / entry["directory"] / entry["maps_npz"]
        )
        with np.load(npz_path) as arrays:
            attention = arrays["attention"].astype(np.float64)
            selected_heads = arrays["selected_heads"].astype(np.int64)
            query_coords = arrays["query_coords"].astype(np.int64)
        if attention.shape[:3] != (24, 13, 13):
            raise ValueError(f"unexpected attention shape {attention.shape}: {npz_path}")
        if not np.array_equal(selected_heads, np.arange(24)):
            order = np.argsort(selected_heads)
            attention = attention[order]
            if not np.array_equal(selected_heads[order], np.arange(24)):
                raise ValueError(f"maps do not contain all 24 heads: {npz_path}")
        trajectory_tokens = []
        for time in range(13):
            current = query_coords[query_coords[:, 0] == time]
            trajectory_tokens.append(
                current[:, 0] * 16 * 28 + current[:, 1] * 28 + current[:, 2]
            )
        fixed_coords = query_coords[query_coords[:, 0] == fixed_time]
        fixed_features, _ = _feature_rows(
            attention[:, fixed_time],
            query_coords=fixed_coords,
            trajectory_tokens=trajectory_tokens,
        )
        moving_time_features = []
        for query_time in range(13):
            current_coords = query_coords[query_coords[:, 0] == query_time]
            current_features, _ = _feature_rows(
                attention[:, query_time],
                query_coords=current_coords,
                trajectory_tokens=trajectory_tokens,
            )
            moving_time_features.append(current_features)
        moving_features = {
            name: np.stack(
                [item[name] for item in moving_time_features]
            ).mean(axis=0)
            for name in FEATURE_NAMES
        }
        fixed_by_step.append(fixed_features)
        moving_by_step.append(moving_features)
    fixed = _classify_features(fixed_by_step)
    moving = _classify_features(moving_by_step)
    fixed["summary"] = summary
    moving["summary"] = summary
    return fixed, moving


def _normalize_joint(fixed: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fixed_log = np.log10(np.maximum(fixed.astype(np.float32), 1.0e-12))
    moving_log = np.log10(np.maximum(moving.astype(np.float32), 1.0e-12))
    combined = np.concatenate([fixed_log.reshape(-1), moving_log.reshape(-1)])
    low, high = np.percentile(combined, (5.0, 99.5))
    if high <= low:
        high = low + 1.0
    return (
        np.clip((fixed_log - low) / (high - low), 0.0, 1.0),
        np.clip((moving_log - low) / (high - low), 0.0, 1.0),
    )


def _rect(
    coords: np.ndarray, time: int, frame_h: int, frame_w: int
) -> tuple[int, int, int, int]:
    current = coords[coords[:, 0] == time]
    if not len(current):
        raise ValueError(f"no query coordinates at latent time {time}")
    rows, columns = current[:, 1], current[:, 2]
    return (
        int(columns.min()) * frame_w // 28,
        int(rows.min()) * frame_h // 16,
        (int(columns.max()) + 1) * frame_w // 28,
        (int(rows.max()) + 1) * frame_h // 16,
    )


def _overlay(frame: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        heatmap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC
    )
    color = cv2.applyColorMap(
        np.asarray(np.clip(resized * 255.0, 0, 255), dtype=np.uint8),
        cv2.COLORMAP_TURBO,
    )
    alpha = (0.12 + 0.58 * resized)[..., None]
    return np.asarray(frame * (1.0 - alpha) + color * alpha, dtype=np.uint8)


def _annotate(
    frame: np.ndarray,
    title: str,
    subtitle: str,
    *,
    moving_rect: tuple[int, int, int, int] | None = None,
    fixed_rect: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    output = frame.copy()
    if moving_rect is not None:
        x0, y0, x1, y1 = moving_rect
        cv2.rectangle(output, (x0, y0), (x1, y1), (50, 235, 90), 3)
    if fixed_rect is not None:
        x0, y0, x1, y1 = fixed_rect
        cv2.rectangle(output, (x0, y0), (x1, y1), (255, 200, 50), 3)
    cv2.rectangle(output, (0, 0), (output.shape[1], 68), (17, 17, 17), -1)
    cv2.putText(
        output,
        title,
        (13, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        subtitle,
        (13, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (218, 224, 220),
        1,
        cv2.LINE_AA,
    )
    return output


def _render_video(
    *,
    frames: list[np.ndarray],
    fixed_attention: np.ndarray,
    moving_attention: np.ndarray,
    fixed_coords: np.ndarray,
    moving_coords: np.ndarray,
    block: int,
    head: int,
    step: int,
    fixed_role: str,
    moving_role: str,
    output_path: Path,
) -> None:
    fixed_maps, moving_maps = _normalize_joint(
        fixed_attention,
        np.stack([moving_attention[time, time] for time in range(13)]),
    )
    frame_h, frame_w = frames[0].shape[:2]
    panel_w, panel_h = 448, 256
    raw_path = output_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (panel_w * 3, panel_h),
    )
    fixed_time = int(fixed_coords[0, 0])
    fixed_box = _rect(fixed_coords, fixed_time, frame_h, frame_w)
    for frame_index in range(49):
        latent_time = min(12, int(round(frame_index / 4)))
        frame = frames[min(frame_index, len(frames) - 1)]
        moving_box = _rect(moving_coords, latent_time, frame_h, frame_w)
        subtitle = (
            f"B{block:02d} H{head:02d} step{step:02d} | "
            f"fixed={fixed_role}, moving={moving_role} | f{frame_index:02d}/t{latent_time:02d}"
        )
        panels = [
            _annotate(
                frame,
                "Generated video + source-derived probes",
                subtitle,
                moving_rect=moving_box,
                fixed_rect=fixed_box,
            ),
            _annotate(
                _overlay(frame, fixed_maps[latent_time]),
                f"Fixed Q(t={fixed_time}) -> K(t={latent_time})",
                subtitle,
                fixed_rect=fixed_box,
            ),
            _annotate(
                _overlay(frame, moving_maps[latent_time]),
                f"Moving Q(t={latent_time}) -> K(t={latent_time})",
                subtitle,
                moving_rect=moving_box,
            ),
        ]
        writer.write(
            np.hstack([cv2.resize(panel, (panel_w, panel_h)) for panel in panels])
        )
    writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        candidate = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg")
        ffmpeg = str(candidate) if candidate.is_file() else None
    if ffmpeg is None:
        raw_path.replace(output_path)
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
            "-an",
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
            str(output_path),
        ],
        check=True,
    )
    raw_path.unlink()


def _map_npz(
    root: Path, model: str, case: str, block: int, step: int
) -> Path:
    case_dir = root / f"block{block:02d}" / "matrices" / model / case
    summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
    entry = next(
        item
        for item in summary["steps"]
        if int(item["step_number_one_based"]) == step
    )
    return case_dir / entry["directory"] / entry["maps_npz"]


def main() -> None:
    args = parse_args()
    moving_root = args.moving_map_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    records = []
    block_summaries = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    block_payloads = {}
    for block in range(30):
        fixed, moving = _read_paired_aggregate(
            moving_root, args.model, args.case, block
        )
        block_payloads[block] = (fixed, moving)
        matches = fixed["primary"] == moving["primary"]
        for head in range(24):
            fixed_role = str(fixed["primary"][head])
            moving_role = str(moving["primary"][head])
            confusion[fixed_role][moving_role] += 1
            records.append(
                {
                    "model": args.model,
                    "case": args.case,
                    "block": block,
                    "head": head,
                    "fixed_role": fixed_role,
                    "moving_role": moving_role,
                    "same_role": bool(matches[head]),
                    "fixed_secondary": str(fixed["secondary"][head]),
                    "moving_secondary": str(moving["secondary"][head]),
                    "fixed_margin": float(fixed["margin"][head]),
                    "moving_margin": float(moving["margin"][head]),
                }
            )
        block_summaries.append(
            {
                "block": block,
                "same_role_heads": int(matches.sum()),
                "same_role_fraction": float(matches.mean()),
                "clear_both_heads": int(
                    ((fixed["margin"] >= 0.1) & (moving["margin"] >= 0.1)).sum()
                ),
            }
        )

    with (output / "head_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output / "block_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(block_summaries[0]))
        writer.writeheader()
        writer.writerows(block_summaries)

    roles = list(ROLE_LABELS)
    same_count = sum(int(row["same_role"]) for row in records)
    clear_records = [
        row
        for row in records
        if float(row["fixed_margin"]) >= 0.1 and float(row["moving_margin"]) >= 0.1
    ]
    summary_payload = {
        "model": args.model,
        "case": args.case,
        "fixed_query": "four source-object tokens at latent t=2/video frame 8",
        "moving_query": "four source-object tokens at each of 13 latent times",
        "classification_roles": ROLE_LABELS,
        "total_heads": len(records),
        "same_role_heads": same_count,
        "same_role_fraction": same_count / len(records),
        "clear_both_total": len(clear_records),
        "clear_both_same_role_fraction": (
            sum(int(row["same_role"]) for row in clear_records) / len(clear_records)
            if clear_records
            else None
        ),
        "confusion_fixed_rows_moving_columns": {
            fixed_role: {
                moving_role: int(confusion[fixed_role][moving_role])
                for moving_role in roles
            }
            for fixed_role in roles
        },
        "blocks": block_summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    moving_npz = _map_npz(
        moving_root,
        args.model,
        args.case,
        args.block,
        args.step,
    )
    with np.load(moving_npz) as arrays:
        moving_attention = arrays["attention"].astype(np.float32)
        selected_heads = arrays["selected_heads"].astype(np.int64)
        moving_coords = arrays["query_coords"].astype(np.int64)
    fixed_time = 2
    fixed_coords = moving_coords[moving_coords[:, 0] == fixed_time]
    head_index = {
        int(head): index for index, head in enumerate(selected_heads.tolist())
    }
    frames = _read_video(args.generated_video.expanduser().resolve())
    if len(frames) < 49:
        raise RuntimeError(f"generated video contains only {len(frames)} frames")
    fixed_block, moving_block = block_payloads[args.block]
    cards = []
    for head in range(24):
        fixed_role = str(fixed_block["primary"][head])
        moving_role = str(moving_block["primary"][head])
        state = "same" if fixed_role == moving_role else "changed"
        video_path = assets / f"block{args.block:02d}_head{head:02d}_{state}.mp4"
        _render_video(
            frames=frames,
            fixed_attention=moving_attention[head_index[head], fixed_time],
            moving_attention=moving_attention[head_index[head]],
            fixed_coords=fixed_coords,
            moving_coords=moving_coords,
            block=args.block,
            head=head,
            step=args.step,
            fixed_role=fixed_role,
            moving_role=moving_role,
            output_path=video_path,
        )
        cards.append(
            f"""<article class="{state}">
<h3>Head {head:02d}: fixed {fixed_role} &rarr; moving {moving_role}</h3>
<video controls loop muted preload="metadata" src="assets/{video_path.name}"></video>
<p>fixed margin {fixed_block['margin'][head]:.3f}; moving margin {moving_block['margin'][head]:.3f}</p>
</article>"""
        )
    confusion_rows = "".join(
        "<tr><th>{}</th>{}</tr>".format(
            fixed_role,
            "".join(
                f"<td>{confusion[fixed_role][moving_role]}</td>"
                for moving_role in roles
            ),
        )
        for fixed_role in roles
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fixed vs moving object query heads</title>
<style>
body{{margin:0;background:#111;color:#eee;font:15px/1.5 Arial,sans-serif}}
header,main{{max-width:1500px;margin:auto;padding:20px}}
h1{{font-size:24px}} .grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}
article{{background:#1d1d1d;border:1px solid #383838;padding:10px;border-radius:6px}}
article.changed{{border-color:#a85a45}} video{{display:block;width:100%;background:#000}}
table{{border-collapse:collapse}} th,td{{border:1px solid #555;padding:5px 10px;text-align:center}}
.note{{color:#c8d0cc}} @media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><header>
<h1>Wan+LoRA fixed-Q vs moving-Q head roles</h1>
<p>{html.escape(args.case)} | all-block match {same_count}/{len(records)}
({same_count / len(records):.1%}) | videos show Block {args.block}, denoise step {args.step}</p>
<p class="note">Left: generated frame. Middle: fixed source-object Q at latent t=2 to current K frame.
Right: source-derived moving-object Q at the current latent time to the same-time K frame.
Green box is the moving source probe; cyan box is the fixed source probe. Heatmaps share one
log normalization per head. Boxes are probes, not generated-object detections.</p>
</header><main>
<h2>Role confusion: fixed rows, moving columns</h2>
<table><thead><tr><th>fixed \\ moving</th>{''.join(f'<th>{role}</th>' for role in roles)}</tr></thead>
<tbody>{confusion_rows}</tbody></table>
<h2>Block {args.block}: all 24 heads</h2><div class="grid">{''.join(cards)}</div>
</main></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    print(f"wrote {output / 'index.html'}")


if __name__ == "__main__":
    main()
