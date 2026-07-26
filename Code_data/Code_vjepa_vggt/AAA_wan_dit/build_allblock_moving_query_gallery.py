#!/usr/bin/env python3
"""Render all-block role summaries and representative moving-query overlays."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from build_moving_query_head_overlay_gallery import (
    _attention_overlay,
    _label_panel,
    _normalize_maps,
    _query_rect,
)
from motion_query_map import _center_crop_resize, _read_video


MODELS = ("wan_lora", "xssc", "physrvg")
MODEL_LABELS = {
    "wan_lora": "Wan + LoRA",
    "xssc": "Wan + xSSC",
    "physrvg": "PhysRVG",
}
ROLES = ("S", "T", "P", "C", "G")
ROLE_LABELS = {
    "S": "within-frame spatial",
    "T": "moving-object trajectory",
    "P": "fixed-position temporal",
    "C": "history/context",
    "G": "global aggregation",
}
ROLE_COLORS = {
    "S": "#248a52",
    "T": "#2376b7",
    "P": "#d28b16",
    "C": "#c64c79",
    "G": "#7359a5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-root", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--query-map", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--step", type=int, default=35)
    parser.add_argument("--overlay-blocks", default="0,5,11,17,23,29")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _plot_role_grid(rows: list[dict], model: str, path: Path) -> None:
    indexed = {(int(row["block"]), int(row["head"])): row for row in rows}
    figure, axis = plt.subplots(figsize=(15, 11), dpi=150)
    for block in range(30):
        for head in range(24):
            row = indexed[(block, head)]
            stability = float(row["step_stability"])
            confidence = str(row["confidence"])
            axis.add_patch(
                Rectangle(
                    (head - 0.5, block - 0.5),
                    1,
                    1,
                    facecolor=ROLE_COLORS[str(row["role"])],
                    alpha=0.28 + 0.72 * stability,
                    edgecolor="#111" if confidence == "clear" else "#fff",
                    linewidth=0.9 if confidence == "clear" else 0.3,
                )
            )
            if confidence == "unstable":
                axis.text(
                    head,
                    block,
                    "x",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="#111",
                )
    axis.set_xlim(-0.5, 23.5)
    axis.set_ylim(29.5, -0.5)
    axis.set_xticks(range(24))
    axis.set_yticks(range(30))
    axis.set_xlabel("Attention head")
    axis.set_ylabel("DiT block")
    axis.set_title(
        f"{MODEL_LABELS[model]}: moving-object query head roles "
        "(steps 5/15/25/35)"
    )
    legend = [
        Patch(facecolor=ROLE_COLORS[role], label=f"{role}: {ROLE_LABELS[role]}")
        for role in ROLES
    ]
    axis.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(path)
    plt.close(figure)


def _load_maps(
    root: Path, model: str, case: str, block: int, step: int
) -> tuple[np.ndarray, np.ndarray]:
    summary_path = (
        root / f"block{block:02d}" / "matrices" / model / case / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in summary["steps"]
        if int(item["step_number_one_based"]) == step
    )
    with np.load(summary_path.parent / entry["directory"] / entry["maps_npz"]) as data:
        return (
            data["attention"].astype(np.float32),
            data["query_coords"].astype(int),
        )


def _write_overlay_video(
    *,
    source_frames: list[np.ndarray],
    attention: np.ndarray,
    query_coords: np.ndarray,
    selections: dict[str, int],
    actual_roles: dict[str, str],
    model: str,
    block: int,
    step: int,
    output_path: Path,
) -> Path:
    poster = output_path.with_suffix(".jpg")
    if (
        output_path.is_file()
        and output_path.stat().st_size > 1024
        and poster.is_file()
        and poster.stat().st_size > 1024
    ):
        return poster
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        ffmpeg = "/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg"
    panel_w, panel_h = 336, 192
    output_w, output_h = panel_w * 4, panel_h * 3
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{output_w}x{output_h}",
        "-r",
        "30",
        "-i",
        "-",
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
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None

    maps = {}
    for role, head in selections.items():
        head_maps = attention[head]
        maps[(role, "same")] = _normalize_maps(
            np.stack([head_maps[time, time] for time in range(13)])
        )
        maps[(role, "cross")] = _normalize_maps(
            np.stack(
                [
                    np.delete(head_maps[:, key_time], key_time, axis=0).mean(0)
                    for key_time in range(13)
                ]
            )
        )

    poster_frames = []
    try:
        for frame_index in range(49):
            latent_time = min(12, int(round(frame_index / 4)))
            source = source_frames[min(frame_index, len(source_frames) - 1)]
            rect = _query_rect(
                query_coords, latent_time, source.shape[0], source.shape[1]
            )
            subtitle = (
                f"B{block:02d} step {step:02d} | frame {frame_index:02d} "
                f"latent {latent_time:02d}"
            )
            original = _label_panel(
                source,
                f"{MODEL_LABELS[model]} | moving object Q",
                subtitle,
                rect,
            )
            panels = [cv2.resize(original, (panel_w, panel_h))]
            for mode in ("same", "cross"):
                for role in ROLES:
                    head = selections[role]
                    actual = actual_roles[role]
                    role_text = role if actual == role else f"{role}->{actual}"
                    title = (
                        f"{role_text} H{head:02d} | "
                        f"{'A(q_t,k_t)' if mode == 'same' else 'mean A(q_s,k_t), s!=t'}"
                    )
                    panel = _label_panel(
                        _attention_overlay(
                            source, maps[(role, mode)][latent_time]
                        ),
                        title,
                        subtitle,
                        rect,
                    )
                    panels.append(cv2.resize(panel, (panel_w, panel_h)))
            note = np.full((panel_h, panel_w, 3), 25, dtype=np.uint8)
            lines = [
                "S spatial | T trajectory",
                "P fixed position | C history",
                "G global",
                "green box: moving query",
            ]
            for index, text in enumerate(lines):
                cv2.putText(
                    note,
                    text,
                    (10, 35 + index * 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (230, 235, 230),
                    1,
                    cv2.LINE_AA,
                )
            panels.append(note)
            canvas = np.vstack(
                [
                    np.hstack(panels[0:4]),
                    np.hstack(panels[4:8]),
                    np.hstack(panels[8:12]),
                ]
            )
            process.stdin.write(canvas.tobytes())
            if frame_index in (0, 8, 20, 32, 44, 48):
                poster_frames.append(cv2.resize(canvas, (672, 288)))
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {output_path}")
    cv2.imwrite(
        str(poster),
        np.vstack(poster_frames),
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    return poster


def _head_table(rows: list[dict], model: str) -> str:
    cells = []
    for row in sorted(
        (item for item in rows if item["model"] == model),
        key=lambda item: (int(item["block"]), int(item["head"])),
    ):
        cells.append(
            "<tr>"
            f"<td>{int(row['block'])}</td><td>H{int(row['head']):02d}</td>"
            f"<td>{html.escape(str(row['role']))}</td>"
            f"<td>{html.escape(str(row['confidence']))}</td>"
            f"<td>{float(row['step_stability']):.0%}</td>"
            f"<td>{float(row['role_margin']):.3f}</td></tr>"
        )
    return (
        "<details><summary>All 720 block/head assignments for this model</summary>"
        "<div class='table'><table><thead><tr><th>Block</th><th>Head</th>"
        "<th>Role</th><th>Confidence</th><th>Step stability</th>"
        f"<th>Margin</th></tr></thead><tbody>{''.join(cells)}</tbody></table>"
        "</div></details>"
    )


def main() -> None:
    args = parse_args()
    root = args.maps_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    payload = json.loads(
        args.classification.expanduser().resolve().read_text(encoding="utf-8")
    )
    rows = payload["heads"]
    query_cases = json.loads(
        args.query_map.expanduser().resolve().read_text(encoding="utf-8")
    )["cases"]
    item = query_cases[args.case]
    source_frames = [
        _center_crop_resize(frame)
        for frame in _read_video(Path(item["source_video"]))
    ]
    blocks = tuple(int(value) for value in args.overlay_blocks.split(","))
    sections = []
    manifest = {"case": args.case, "step": args.step, "models": {}}

    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        grid_path = assets / f"{model}_allblock_role_grid.png"
        _plot_role_grid(model_rows, model, grid_path)
        cards = []
        manifest["models"][model] = {}
        for block in blocks:
            selections = {
                role: int(head)
                for role, head in payload["representatives"][model][
                    str(block)
                ].items()
            }
            row_index = {
                (int(row["block"]), int(row["head"])): str(row["role"])
                for row in model_rows
            }
            actual_roles = {
                role: row_index[(block, head)]
                for role, head in selections.items()
            }
            attention, coords = _load_maps(
                root, model, args.case, block, args.step
            )
            video = assets / f"{model}_block{block:02d}_representatives.mp4"
            poster = _write_overlay_video(
                source_frames=source_frames,
                attention=attention,
                query_coords=coords,
                selections=selections,
                actual_roles=actual_roles,
                model=model,
                block=block,
                step=args.step,
                output_path=video,
            )
            manifest["models"][model][str(block)] = selections
            labels = " | ".join(
                (
                    f"{role}=H{selections[role]:02d}"
                    if actual_roles[role] == role
                    else (
                        f"{role} candidate=H{selections[role]:02d} "
                        f"(actual {actual_roles[role]})"
                    )
                )
                for role in ROLES
            )
            cards.append(
                f"<article><h3>Block {block:02d}</h3><p>{labels}</p>"
                f"<video controls loop muted preload='metadata' "
                f"poster='assets/{poster.name}' src='assets/{video.name}'></video>"
                "</article>"
            )
        sections.append(
            f"<section><h2>{MODEL_LABELS[model]}</h2>"
            f"<a href='assets/{grid_path.name}'><img class='grid' "
            f"src='assets/{grid_path.name}'></a>"
            f"{_head_table(rows, model)}"
            f"<div class='cards'>{''.join(cards)}</div></section>"
        )

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>All-block moving-query head roles</title><style>
body{{margin:0;background:#eef1ee;color:#202622;font:14px/1.45 Arial,sans-serif}}
header,main{{max-width:1760px;margin:auto;padding:18px 24px}}header{{max-width:none;background:#202622;color:#fff}}
h1,h2,h3{{letter-spacing:0}}h1{{margin:0 0 8px}}section{{margin:32px 0 52px;border-top:2px solid #222}}
.grid{{width:100%;max-width:1500px;background:#fff}}.cards{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}}
article{{background:#fff;border:1px solid #c8ceca;border-radius:4px;padding:10px}}article h3,article p{{margin:0 0 7px}}
video{{display:block;width:100%;background:#111}}details{{background:#fff;border:1px solid #ccd2ce;padding:10px;margin:12px 0}}
.table{{max-height:420px;overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:5px 8px;border-bottom:1px solid #ddd;text-align:right}}
th{{position:sticky;top:0;background:#26342c;color:#fff}}@media(max-width:1000px){{.cards{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>All-block moving-object query head roles</h1>
<p>{html.escape(args.case)} | denoise steps 5/15/25/35 | overlays use step {args.step}</p>
<p>S/T/P/C/G are relative within-block specializations. A black cell border means stability >=75% and score margin >=0.10; x means unstable.</p>
<p>Each overlay video shows original, five same-time maps A(q_t,k_t), and five cross-time maps mean A(q_s,k_t), s!=t.</p>
</header><main>{''.join(sections)}</main></body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
    print(output / "index.html")


if __name__ == "__main__":
    main()
