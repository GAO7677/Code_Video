#!/usr/bin/env python3
"""Build focused visual evidence for representative attention-head roles."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_moving_query_head_overlay_gallery import (
    _attention_overlay,
    _label_panel,
    _normalize_maps,
    _query_rect,
)
from classify_allblock_moving_query_maps import (
    ROLE_LABELS,
    ROLES,
    _features_from_maps,
    _role_scores,
)
from motion_query_map import _center_crop_resize, _read_video


MODEL_LABELS = {
    "wan_lora": "Wan + LoRA",
    "xssc": "Wan + xSSC",
    "physrvg": "PhysRVG",
}
ROLE_COLORS = {
    "S": "#248a52",
    "T": "#2376b7",
    "P": "#d28b16",
    "C": "#c64c79",
    "G": "#7359a5",
}
ROLE_EVIDENCE = {
    "S": "same-frame mass is high; response should collapse near k=q",
    "T": "cross-time response should follow the green moving-object boxes",
    "P": "cross-time response should prefer the cyan fixed screen position",
    "C": "middle/late queries should place excess mass on earlier key times",
    "G": "attention should remain broad, high-entropy, and weakly object-specific",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-root", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--query-map", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--examples",
        default=(
            "S:physrvg:2:10,T:wan_lora:12:16,P:xssc:17:11,"
            "C:xssc:0:5,G:xssc:10:7"
        ),
    )
    parser.add_argument("--step", type=int, default=35)
    parser.add_argument("--query-time", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _parse_examples(text: str) -> list[tuple[str, str, int, int]]:
    examples = []
    for raw in text.split(","):
        role, model, block, head = raw.split(":")
        if role not in ROLES or model not in MODEL_LABELS:
            raise ValueError(f"invalid example: {raw}")
        examples.append((role, model, int(block), int(head)))
    return examples


def _generated_video(root: Path, model: str, case: str) -> Path:
    matches = sorted((root / "generated" / model).glob(f"**/{case}.mp4"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one generated video for {model}/{case}, found {matches}"
        )
    return matches[0]


def _load_step(
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


def _rect_for_time(
    coords: np.ndarray, time: int, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    return _query_rect(coords, time, shape[0], shape[1])


def _draw_rectangles(
    frame: np.ndarray,
    *,
    moving_rect: tuple[int, int, int, int],
    fixed_rect: tuple[int, int, int, int],
) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(
        output,
        moving_rect[:2],
        moving_rect[2:],
        (45, 235, 85),
        3,
    )
    cv2.rectangle(
        output,
        fixed_rect[:2],
        fixed_rect[2:],
        (255, 210, 40),
        2,
    )
    return output


def _annotate(
    frame: np.ndarray,
    title: str,
    subtitle: str,
    *,
    moving_rect: tuple[int, int, int, int],
    fixed_rect: tuple[int, int, int, int],
) -> np.ndarray:
    return _label_panel(
        _draw_rectangles(
            frame, moving_rect=moving_rect, fixed_rect=fixed_rect
        ),
        title,
        subtitle,
        None,
    )


def _render_evidence_figure(
    *,
    role: str,
    model: str,
    block: int,
    head: int,
    step: int,
    query_time: int,
    attention: np.ndarray,
    coords: np.ndarray,
    scores: dict[str, float],
    features: dict[str, float],
    step_features: list[dict[str, np.ndarray]],
    step_numbers: tuple[int, ...],
    source_frames: list[np.ndarray],
    output_path: Path,
) -> None:
    head_attention = attention[head]
    temporal = head_attention.sum(axis=(2, 3))
    temporal /= np.maximum(temporal.sum(axis=1, keepdims=True), 1.0e-30)
    selected_times = (0, 3, 6, 9, 12)
    fixed_maps = _normalize_maps(head_attention[query_time])
    frame_h, frame_w = source_frames[0].shape[:2]
    fixed_rect = _rect_for_time(coords, query_time, (frame_h, frame_w))
    overlays = []
    for key_time in selected_times:
        frame_index = 0 if key_time == 0 else key_time * 4
        source = source_frames[min(frame_index, len(source_frames) - 1)]
        moving_rect = _rect_for_time(coords, key_time, (frame_h, frame_w))
        panel = _annotate(
            _attention_overlay(source, fixed_maps[key_time]),
            f"q{query_time} -> k{key_time}",
            f"mass {temporal[query_time, key_time]:.3f}",
            moving_rect=moving_rect,
            fixed_rect=fixed_rect,
        )
        overlays.append(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))

    figure = plt.figure(figsize=(18, 10), dpi=150)
    grid = figure.add_gridspec(2, 5, height_ratios=(1.0, 1.05))
    score_axis = figure.add_subplot(grid[0, 0])
    score_values = [scores[candidate] for candidate in ROLES]
    score_axis.bar(
        ROLES,
        score_values,
        color=[ROLE_COLORS[candidate] for candidate in ROLES],
    )
    score_axis.set_ylim(0, 1.05)
    score_axis.set_title("Aggregate relative role scores")
    score_axis.set_ylabel("rank-based score")
    for index, value in enumerate(score_values):
        score_axis.text(index, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)

    matrix_axis = figure.add_subplot(grid[0, 1:3])
    image = matrix_axis.imshow(
        temporal,
        cmap="magma",
        vmin=0,
        vmax=float(np.percentile(temporal, 99.0)),
        interpolation="nearest",
    )
    matrix_axis.axhline(query_time - 0.5, color="#35c9ff", linewidth=1.0)
    matrix_axis.axhline(query_time + 0.5, color="#35c9ff", linewidth=1.0)
    matrix_axis.plot(range(13), range(13), color="#45ed79", linewidth=1.0)
    matrix_axis.set_title("Temporal attention mass (rows Q time, columns K time)")
    matrix_axis.set_xlabel("Key latent time")
    matrix_axis.set_ylabel("Query latent time")
    matrix_axis.set_xticks(range(13))
    matrix_axis.set_yticks(range(13))
    figure.colorbar(image, ax=matrix_axis, fraction=0.046)

    diagnostic_axis = figure.add_subplot(grid[0, 3])
    if role == "S":
        diagnostic = [
            sample["same_frame_mass"][head] for sample in step_features
        ]
        diagnostic_axis.axhline(
            1.0 / 13.0, color="#555", linestyle="--", linewidth=1
        )
        diagnostic_label = "same-frame mass"
    elif role in ("T", "P"):
        diagnostic = [
            sample["cross_ball_enrichment"][head]
            / max(sample["aligned_enrichment"][head], 1.0e-12)
            for sample in step_features
        ]
        diagnostic_axis.axhline(
            1.0, color="#555", linestyle="--", linewidth=1
        )
        diagnostic_label = "trajectory / fixed enrichment"
    elif role == "C":
        diagnostic = [sample["history_bias"][head] for sample in step_features]
        diagnostic_axis.axhline(
            0.0, color="#555", linestyle="--", linewidth=1
        )
        diagnostic_label = "past - future mass"
    else:
        diagnostic = [sample["entropy"][head] for sample in step_features]
        diagnostic_axis.axhline(
            0.8, color="#555", linestyle="--", linewidth=1
        )
        diagnostic_label = "normalized entropy"
    diagnostic_axis.plot(
        step_numbers,
        diagnostic,
        marker="o",
        linewidth=2,
        color=ROLE_COLORS[role],
    )
    diagnostic_axis.set_xticks(step_numbers)
    diagnostic_axis.set_title("Role diagnostic across denoise steps")
    diagnostic_axis.set_xlabel("Denoise step")
    diagnostic_axis.set_ylabel(diagnostic_label)
    diagnostic_axis.grid(alpha=0.2)

    text_axis = figure.add_subplot(grid[0, 4])
    text_axis.axis("off")
    uniform = 1.0 / 13.0
    lines = [
        textwrap.fill(
            f"Expected evidence: {ROLE_EVIDENCE[role]}",
            width=42,
        ),
        "",
        f"same-frame mass: {features['same_frame_mass']:.3f} "
        f"(uniform frame baseline {uniform:.3f})",
        f"cross-trajectory enrichment: {features['cross_ball_enrichment']:.2f}x",
        f"fixed-position enrichment: {features['aligned_enrichment']:.2f}x",
        f"first-frame mass: {features['first_frame_mass']:.3f}",
        f"history bias (past - future): {features['history_bias']:+.3f}",
        f"normalized entropy: {features['entropy']:.3f}",
        f"step diagnostic: {[round(float(value), 3) for value in diagnostic]}",
        "",
        "Green box: source-derived moving probe",
        f"Cyan box: fixed q{query_time} probe coordinate",
    ]
    text_axis.text(
        0,
        1,
        "\n".join(lines),
        va="top",
        fontsize=9,
        family="monospace",
        linespacing=1.45,
    )
    for index, (key_time, overlay) in enumerate(zip(selected_times, overlays)):
        axis = figure.add_subplot(grid[1, index])
        axis.imshow(overlay)
        axis.set_title(f"q{query_time} -> k{key_time}")
        axis.axis("off")
    figure.suptitle(
        f"{role}: {ROLE_LABELS[role]} | {MODEL_LABELS[model]} "
        f"Block {block:02d} Head {head:02d} | denoise step {step}",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path)
    plt.close(figure)


def _render_video(
    *,
    role: str,
    model: str,
    block: int,
    head: int,
    step: int,
    query_time: int,
    attention: np.ndarray,
    coords: np.ndarray,
    source_frames: list[np.ndarray],
    output_path: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg") or (
        "/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg"
    )
    panel_w, panel_h = 448, 256
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
        f"{panel_w * 3}x{panel_h}",
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
    head_attention = attention[head]
    fixed = _normalize_maps(head_attention[query_time])
    moving_same = _normalize_maps(
        np.stack([head_attention[time, time] for time in range(13)])
    )
    frame_h, frame_w = source_frames[0].shape[:2]
    fixed_rect = _rect_for_time(coords, query_time, (frame_h, frame_w))
    try:
        for frame_index in range(49):
            key_time = min(12, int(round(frame_index / 4)))
            source = source_frames[min(frame_index, len(source_frames) - 1)]
            moving_rect = _rect_for_time(coords, key_time, (frame_h, frame_w))
            subtitle = (
                f"{MODEL_LABELS[model]} B{block:02d} H{head:02d} "
                f"| frame {frame_index:02d} latent {key_time:02d}"
            )
            original = _annotate(
                source,
                "Generated frame: moving probe / fixed q6 probe",
                subtitle,
                moving_rect=moving_rect,
                fixed_rect=fixed_rect,
            )
            fixed_panel = _annotate(
                _attention_overlay(source, fixed[key_time]),
                f"Fixed q{query_time}: A(q{query_time}, k{key_time})",
                subtitle,
                moving_rect=moving_rect,
                fixed_rect=fixed_rect,
            )
            moving_panel = _annotate(
                _attention_overlay(source, moving_same[key_time]),
                f"Moving query: A(q{key_time}, k{key_time})",
                subtitle,
                moving_rect=moving_rect,
                fixed_rect=fixed_rect,
            )
            canvas = np.hstack(
                [
                    cv2.resize(panel, (panel_w, panel_h))
                    for panel in (original, fixed_panel, moving_panel)
                ]
            )
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {output_path}")


def main() -> None:
    args = parse_args()
    root = args.maps_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    classification = json.loads(
        args.classification.expanduser().resolve().read_text(encoding="utf-8")
    )
    classified = {
        (row["model"], int(row["block"]), int(row["head"])): row
        for row in classification["heads"]
    }
    query_item = json.loads(
        args.query_map.expanduser().resolve().read_text(encoding="utf-8")
    )["cases"][args.case]
    generated_root = args.generated_root.expanduser().resolve()
    generated_cache: dict[str, tuple[Path, list[np.ndarray]]] = {}
    cards = []
    manifest = []
    for role, model, block, head in _parse_examples(args.examples):
        row = classified[(model, block, head)]
        if row["role"] != role or row["confidence"] != "clear":
            raise ValueError(
                f"{model} B{block} H{head} is {row['role']}/{row['confidence']}"
            )
        step_maps = []
        coords = None
        target_attention = None
        for step in (5, 15, 25, 35):
            attention, current_coords = _load_step(
                root, model, args.case, block, step
            )
            step_maps.append(attention)
            coords = current_coords
            if step == args.step:
                target_attention = attention
        assert coords is not None and target_attention is not None
        if model not in generated_cache:
            generated_path = _generated_video(generated_root, model, args.case)
            generated_cache[model] = (
                generated_path,
                [
                    _center_crop_resize(frame)
                    for frame in _read_video(generated_path)
                ],
            )
        generated_path, source_frames = generated_cache[model]
        step_features = [
            _features_from_maps(attention, coords) for attention in step_maps
        ]
        step_scores = [_role_scores(features) for features in step_features]
        features = {
            name: float(
                np.mean([sample[name][head] for sample in step_features])
            )
            for name in step_features[0]
        }
        scores = {
            candidate: float(
                np.mean([sample[candidate][head] for sample in step_scores])
            )
            for candidate in ROLES
        }
        stem = f"{role}_{model}_block{block:02d}_head{head:02d}_generated"
        figure_path = assets / f"{stem}_evidence.png"
        video_path = assets / f"{stem}_overlay.mp4"
        _render_evidence_figure(
            role=role,
            model=model,
            block=block,
            head=head,
            step=args.step,
            query_time=args.query_time,
            attention=target_attention,
            coords=coords,
            scores=scores,
            features=features,
            step_features=step_features,
            step_numbers=(5, 15, 25, 35),
            source_frames=source_frames,
            output_path=figure_path,
        )
        _render_video(
            role=role,
            model=model,
            block=block,
            head=head,
            step=args.step,
            query_time=args.query_time,
            attention=target_attention,
            coords=coords,
            source_frames=source_frames,
            output_path=video_path,
        )
        manifest.append(
            {
                "role": role,
                "model": model,
                "block": block,
                "head": head,
                "confidence": row["confidence"],
                "step_stability": row["step_stability"],
                "role_margin": row["role_margin"],
                "scores": scores,
                "features": features,
                "generated_video": str(generated_path),
                "probe_coordinate_source": str(query_item["source_video"]),
            }
        )
        cards.append(
            f"<article class='{role}'><h2>{role}: "
            f"{html.escape(ROLE_LABELS[role])}</h2>"
            f"<p>{MODEL_LABELS[model]} | Block {block:02d} Head {head:02d} | "
            f"stability {float(row['step_stability']):.0%} | "
            f"margin {float(row['role_margin']):.3f}</p>"
            f"<p>{html.escape(ROLE_EVIDENCE[role])}</p>"
            f"<a href='assets/{figure_path.name}'><img "
            f"src='assets/{figure_path.name}'></a>"
            f"<video controls loop muted preload='metadata' "
            f"src='assets/{video_path.name}'></video></article>"
        )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Head-role evidence</title><style>
body{{margin:0;background:#eef1ee;color:#202622;font:14px/1.45 Arial,sans-serif}}
header,main{{max-width:1720px;margin:auto;padding:18px 24px}}header{{max-width:none;background:#202622;color:#fff}}
h1,h2{{letter-spacing:0}}h1{{margin:0 0 8px}}article{{background:#fff;border:1px solid #c7cdc9;border-left:8px solid #555;margin:22px 0;padding:12px}}
article.S{{border-left-color:{ROLE_COLORS['S']}}}article.T{{border-left-color:{ROLE_COLORS['T']}}}
article.P{{border-left-color:{ROLE_COLORS['P']}}}article.C{{border-left-color:{ROLE_COLORS['C']}}}
article.G{{border-left-color:{ROLE_COLORS['G']}}}article h2,article p{{margin:0 0 8px}}
img,video{{display:block;width:100%;height:auto;background:#111;margin-top:10px}}
</style></head><body><header><h1>Five falsifiable head-role examples</h1>
<p>{html.escape(args.case)} | q{args.query_time} fixed-query test | denoise step {args.step}</p>
<p>Heatmaps are overlaid on each model's generated 49-frame output. Green = moving probe coordinates derived from the source video; cyan = fixed q{args.query_time} probe coordinates.</p>
<p>The boxes are attention sampling coordinates, not detected boxes of generated objects. Thus T means source-trajectory probe alignment, not proven semantic tracking of a generated object.</p>
</header><main>{''.join(cards)}</main></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    print(output / "index.html")


if __name__ == "__main__":
    main()
