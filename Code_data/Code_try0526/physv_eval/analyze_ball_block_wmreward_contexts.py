#!/usr/bin/env python3
"""Score ball-block videos with full-video WMReward under several context lengths."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .wmreward_official import WMRewardRunner


DATASET_DIR = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/ball_block")
OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/wmreward_ball_block_context_full150_corrected_v2"
)
CONTEXT_LENGTHS = (2, 4, 8, 10)
BASELINE = "e07_mu05_m1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--context-frames", type=int, nargs="+", default=list(CONTEXT_LENGTHS))
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--model", default="vitg384")
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def case_group(stem: str) -> str:
    if stem == BASELINE:
        return "baseline"
    if stem.startswith("extreme_restitution") or stem.startswith(("e03_", "e05_", "e09_")):
        return "restitution"
    if stem.startswith("extreme_friction") or "_mu01_" in stem or "_mu10_" in stem:
        return "friction"
    if stem.startswith("extreme_mass") or stem in {"e07_mu05_m01", "e07_mu05_m5"}:
        return "mass"
    if stem.startswith("motion_speed") or stem.startswith("extreme_speed"):
        return "speed"
    if stem.startswith("motion_direction") or stem.startswith("extreme_direction"):
        return "direction"
    if stem.startswith("motion_distance") or stem.startswith("extreme_distance"):
        return "distance"
    return "other"


def load_metadata(video_path: Path) -> dict[str, Any]:
    metadata_path = video_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def read_first_frames(video_path: Path, count: int) -> tuple[list[np.ndarray], float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) < count:
        raise RuntimeError(f"{video_path} has only {len(frames)} readable frames; need {count}")
    return frames, fps, total_frames


def write_contact_sheet(frames: list[np.ndarray], output_path: Path) -> None:
    count = len(frames)
    columns = 1 if count == 1 else (4 if count == 8 else 5)
    rows = math.ceil(count / columns)
    tile_width, tile_height = 256, 144
    label_height = 24
    canvas = np.full((rows * (tile_height + label_height), columns * tile_width, 3), 242, np.uint8)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        tile = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        top = row * (tile_height + label_height)
        left = column * tile_width
        canvas[top : top + tile_height, left : left + tile_width] = tile
        cv2.putText(
            canvas,
            f"frame {index}",
            (left + 7, top + tile_height + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (38, 53, 48),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Failed to write {output_path}")


def ensure_video_link(output_dir: Path, dataset_dir: Path) -> None:
    link = output_dir / "videos"
    if link.is_symlink() and link.resolve() == dataset_dir.resolve():
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Unexpected output video path: {link}")
    link.symlink_to(dataset_dir, target_is_directory=True)


def save_json(records: list[dict[str, Any]], path: Path, args: argparse.Namespace) -> None:
    payload = {
        "metric": "WMReward V-JEPA causal future-prediction loss",
        "surprise_direction": "lower is more predictable",
        "similarity": "1 - surprise",
        "full_video_frames": args.max_frames,
        "window_size": args.window_size,
        "stride": args.stride,
        "context_lengths": args.context_frames,
        "context_policy": "tubelet-aligned; context_frames divisible by tubelet_size",
        "cosine_dim": -1,
        "records": records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def score_all(args: argparse.Namespace, videos: list[Path]) -> list[dict[str, Any]]:
    scores_path = args.output_dir / "wmreward_scores.json"
    records: list[dict[str, Any]] = []
    if scores_path.exists():
        records = json.loads(scores_path.read_text(encoding="utf-8")).get("records", [])
    completed = {(item["video_stem"], int(item["context_frames"])) for item in records}
    pending = [
        (video, context)
        for video in videos
        for context in args.context_frames
        if (video.stem, context) not in completed
    ]
    if not pending:
        print("All WMReward scores already exist; model loading is skipped.")
        return records

    runner = WMRewardRunner(
        cuda_visible_devices=args.cuda_visible_devices,
        model_name=args.model,
        window_size=args.window_size,
        context_frames=args.context_frames[0],
        stride=args.stride,
        seed=args.seed,
        max_frames=args.max_frames,
        cosine_dim=-1,
        require_tubelet_aligned_context=True,
    )
    print(f"Pending scores: {len(pending)}", flush=True)
    for video_index, video_path in enumerate(videos, 1):
        missing_contexts = [
            context
            for context in args.context_frames
            if (video_path.stem, context) not in completed
        ]
        if not missing_contexts:
            continue
        print(f"[{video_index}/{len(videos)}] decode {video_path.stem}", flush=True)
        video_tensor = runner.load_video(video_path, max_frames=args.max_frames)
        loaded_frames = int(video_tensor.shape[2])
        if loaded_frames != args.max_frames:
            raise RuntimeError(
                f"Expected {args.max_frames} loaded frames for {video_path}, got {loaded_frames}"
            )
        metadata = load_metadata(video_path)
        for context in missing_contexts:
            print(f"  context={context}", flush=True)
            result = runner.score_tensor(video_tensor, context_frames=context)
            record = {
                "video_stem": video_path.stem,
                "video": str(video_path),
                "group": case_group(video_path.stem),
                "caption": metadata.get("caption", ""),
                **result,
            }
            records.append(record)
            completed.add((video_path.stem, context))
            save_json(records, scores_path, args)
        del video_tensor
        if runner._torch.cuda.is_available():
            runner._torch.cuda.empty_cache()
    return records


def write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    fields = (
        "video_stem",
        "group",
        "context_frames",
        "effective_context_frames",
        "context_tubelets",
        "tubelet_size",
        "cosine_dim",
        "surprise",
        "similarity",
        "video_frames_loaded",
        "window_size",
        "stride",
        "model",
        "video",
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda item: (item["video_stem"], item["context_frames"])))


def plot_distributions(records: list[dict[str, Any]], output_path: Path, contexts: list[int]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    rng = np.random.default_rng(42)
    for axis, metric, title, direction in (
        (axes[0], "surprise", "WMReward surprise", "lower = more predictable"),
        (axes[1], "similarity", "WMReward similarity", "higher = more predictable"),
    ):
        values = [
            [float(item[metric]) for item in records if int(item["context_frames"]) == context]
            for context in contexts
        ]
        boxes = axis.boxplot(values, positions=range(len(contexts)), widths=0.54, patch_artist=True)
        for box in boxes["boxes"]:
            box.set(facecolor="#dfd5bc", edgecolor="#234b43", linewidth=1.3)
        for index, group_values in enumerate(values):
            jitter = rng.normal(index, 0.045, size=len(group_values))
            axis.scatter(jitter, group_values, s=22, alpha=0.65, color="#c4512d", edgecolor="none")
        axis.set_xticks(range(len(contexts)), [str(value) for value in contexts])
        axis.set_xlabel("Context frames")
        axis.set_ylabel(metric)
        axis.set_title(f"{title}\n{direction}", loc="left", fontweight="bold")
        axis.grid(axis="y", color="#d8d0bf", alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Full 150-frame ball-block videos", fontsize=19, fontweight="bold", color="#19352f")
    figure.savefig(output_path, dpi=180, facecolor="#f1ecdf")
    plt.close(figure)


def plot_curves(records: list[dict[str, Any]], output_path: Path, contexts: list[int]) -> None:
    by_video: dict[str, dict[int, float]] = defaultdict(dict)
    groups: dict[str, str] = {}
    for item in records:
        by_video[item["video_stem"]][int(item["context_frames"])] = float(item["surprise"])
        groups[item["video_stem"]] = item["group"]
    colors = {
        "baseline": "#111111",
        "restitution": "#d95f3b",
        "friction": "#d8a225",
        "mass": "#527b70",
        "speed": "#286c9b",
        "direction": "#8b5c91",
        "distance": "#668a34",
        "other": "#777777",
    }
    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    for stem, values in by_video.items():
        if any(context not in values for context in contexts):
            continue
        group = groups[stem]
        axis.plot(
            contexts,
            [values[context] for context in contexts],
            color=colors[group],
            alpha=0.42 if group != "baseline" else 1.0,
            linewidth=3.0 if group == "baseline" else 1.4,
        )
    for group, color in colors.items():
        members = [stem for stem in by_video if groups[stem] == group]
        if not members:
            continue
        means = [np.mean([by_video[stem][context] for stem in members]) for context in contexts]
        axis.plot(contexts, means, color=color, linewidth=3.2, marker="o", label=f"{group} mean")
    axis.set_xticks(contexts)
    axis.set_xlabel("Context frames")
    axis.set_ylabel("WMReward surprise (lower is more predictable)")
    axis.set_title("Per-video context sensitivity", loc="left", fontsize=19, fontweight="bold")
    axis.grid(color="#d8d0bf", alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncols=4, fontsize=9)
    figure.savefig(output_path, dpi=180, facecolor="#f1ecdf")
    plt.close(figure)


def plot_heatmap(records: list[dict[str, Any]], output_path: Path, contexts: list[int]) -> None:
    by_video: dict[str, dict[int, float]] = defaultdict(dict)
    for item in records:
        by_video[item["video_stem"]][int(item["context_frames"])] = float(item["surprise"])
    stems = sorted(by_video, key=lambda stem: np.mean(list(by_video[stem].values())), reverse=True)
    matrix = np.asarray([[by_video[stem][context] for context in contexts] for stem in stems])
    figure, axis = plt.subplots(figsize=(9, 13), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd")
    axis.set_xticks(range(len(contexts)), [f"ctx {value}" for value in contexts])
    axis.set_yticks(range(len(stems)), stems, fontsize=8)
    axis.set_title("WMReward surprise by video", loc="left", fontsize=18, fontweight="bold")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, shrink=0.7, label="surprise")
    figure.savefig(output_path, dpi=180, facecolor="#f1ecdf")
    plt.close(figure)


def write_dashboard(
    records: list[dict[str, Any]],
    videos: list[Path],
    args: argparse.Namespace,
) -> None:
    score_map = {
        (item["video_stem"], int(item["context_frames"])): item for item in records
    }
    cards = []
    for video in videos:
        metadata = load_metadata(video)
        rows = []
        images = []
        for context in args.context_frames:
            result = score_map[(video.stem, context)]
            rows.append(
                f"<tr><td>{context}</td><td>{result['surprise']:.6f}</td>"
                f"<td>{result['similarity']:.6f}</td></tr>"
            )
            images.append(
                f'<figure><img src="context_frames/{html.escape(video.stem)}/ctx_{context:02d}.jpg" '
                f'alt="first {context} frames"><figcaption>Context: first {context} frame(s)</figcaption></figure>'
            )
        cards.append(
            f"""<article class="card" data-group="{case_group(video.stem)}">
<div class="card-head"><div><span>{case_group(video.stem)}</span><h2>{html.escape(video.stem)}</h2></div><button onclick="this.closest('.card').querySelector('video').play()">Play</button></div>
<video controls muted loop preload="metadata" src="videos/{html.escape(video.name)}"></video>
<p>{html.escape(str(metadata.get('caption', 'Ball colliding with a wooden block')))}</p>
<table><thead><tr><th>Context</th><th>Surprise ↓</th><th>Similarity ↑</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class="contexts">{''.join(images)}</div></article>"""
        )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WMReward context audit</title><style>
:root{{--ink:#19352f;--paper:#f1ecdf;--card:#fffdf7;--accent:#c4512d;--line:#d8d0bf}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 88% 3%,#cedbc9 0,transparent 28%),var(--paper);color:var(--ink);font-family:Georgia,'Times New Roman',serif}}main{{width:min(1540px,95vw);margin:auto;padding:48px 0 80px}}.eyebrow{{font:700 12px sans-serif;letter-spacing:.18em;text-transform:uppercase;color:var(--accent)}}h1{{font-size:clamp(42px,6vw,82px);line-height:.94;margin:12px 0 20px;max-width:1100px}}.intro{{max-width:980px;font-size:18px;line-height:1.6}}.notice{{margin:25px 0;padding:17px 20px;background:#fff6e6;border-left:5px solid var(--accent);font:14px/1.55 sans-serif}}.actions{{display:flex;gap:10px;margin:22px 0}}button{{border:1px solid var(--ink);background:transparent;color:var(--ink);padding:9px 13px;font-weight:700;cursor:pointer}}button:hover{{background:var(--ink);color:white}}.plot{{width:100%;display:block;margin:25px 0;border:1px solid var(--line);background:white}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px;margin-top:35px}}.card{{min-width:0;background:var(--card);border:1px solid var(--line);padding:18px;box-shadow:0 10px 30px #19352f12}}.card-head{{display:flex;justify-content:space-between;align-items:center;gap:10px}}.card-head span{{font:700 10px sans-serif;letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}}h2{{margin:4px 0 12px;font-size:22px;overflow-wrap:anywhere}}video{{width:100%;aspect-ratio:16/9;background:#111}}.card p,table,figcaption{{font:13px/1.45 sans-serif}}table{{width:100%;border-collapse:collapse;margin:12px 0}}th,td{{padding:7px;border-bottom:1px solid #e5dfd2;text-align:left}}.contexts{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}figure{{margin:0;min-width:0}}figure img{{width:100%;display:block;border:1px solid var(--line)}}figcaption{{padding:5px 0;color:#63726c}}@media(max-width:900px){{.grid,.contexts{{grid-template-columns:1fr}}main{{padding-top:30px}}}}
</style></head><body><main><div class="eyebrow">WMReward / corrected context-length audit</div><h1>How much context makes motion predictable?</h1><p class="intro">Thirty complete 150-frame ball-block videos are scored with tubelet-aligned causal context lengths 2, 4, 8, and 10. Every setting uses the same ViT-G/384 model, 16-frame window, stride 8, seed 42, and per-token feature cosine on dim=-1.</p><div class="notice"><strong>Metric:</strong> surprise is the V-JEPA future-prediction loss and lower is more predictable; similarity is 1-surprise. Context is an internal causal mask within every sliding window. Increasing context also shortens the predicted future, so this is a joint context/horizon comparison rather than an isolated causal effect of context length.</div><div class="actions"><button onclick="location.reload()">Manual refresh</button><button onclick="document.querySelectorAll('video').forEach(v=>{{v.currentTime=0;v.play()}})">Replay all</button><a href="wmreward_scores.csv"><button>Download CSV</button></a><a href="wmreward_scores.json"><button>Download JSON</button></a></div><img class="plot" src="plots/distributions.png" alt="score distributions"><img class="plot" src="plots/context_curves.png" alt="context curves"><img class="plot" src="plots/score_heatmap.png" alt="score heatmap"><div class="grid">{''.join(cards)}</div></main></body></html>"""
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")


def print_summary(records: list[dict[str, Any]], contexts: list[int]) -> None:
    print("\nWMReward distribution summary")
    for context in contexts:
        values = np.asarray(
            [item["surprise"] for item in records if int(item["context_frames"]) == context],
            dtype=np.float64,
        )
        print(
            f"ctx={context:2d}: mean={values.mean():.6f} std={values.std():.6f} "
            f"min={values.min():.6f} median={np.median(values):.6f} max={values.max():.6f}"
        )


def write_statistical_summary(
    records: list[dict[str, Any]], contexts: list[int], output_path: Path
) -> None:
    summaries = []
    for context in contexts:
        values = np.asarray(
            [item["surprise"] for item in records if int(item["context_frames"]) == context],
            dtype=np.float64,
        )
        sample_sd = float(values.std(ddof=1))
        half_width = 1.959963984540054 * sample_sd / math.sqrt(len(values))
        summaries.append(
            {
                "context_frames": context,
                "n_videos": int(len(values)),
                "mean_surprise": float(values.mean()),
                "sample_sd": sample_sd,
                "normal_approx_ci95": [
                    float(values.mean() - half_width),
                    float(values.mean() + half_width),
                ],
                "predicted_future_frames": int(16 - context),
            }
        )
    payload = {
        "unit": "video-level aggregate over 17 overlapping windows",
        "ci_note": "95% normal-approximation interval across the 30 deterministic videos",
        "causal_limit": "context length and predicted future horizon change together",
        "summaries": summaries,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.context_frames = sorted(set(args.context_frames))
    if any(context <= 0 or context >= args.window_size for context in args.context_frames):
        raise ValueError("Every context length must be between 1 and window_size-1")
    if any(context % 2 for context in args.context_frames):
        raise ValueError("Every context length must align to the ViT-G tubelet_size=2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plots").mkdir(exist_ok=True)
    videos = sorted(args.dataset_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No MP4 videos in {args.dataset_dir}")
    ensure_video_link(args.output_dir, args.dataset_dir)

    max_context = max(args.context_frames)
    for video in videos:
        frames, fps, total_frames = read_first_frames(video, max_context)
        if total_frames < args.max_frames:
            raise RuntimeError(f"{video} has {total_frames} frames; expected at least {args.max_frames}")
        for context in args.context_frames:
            write_contact_sheet(
                frames[:context],
                args.output_dir / "context_frames" / video.stem / f"ctx_{context:02d}.jpg",
            )
        print(f"context sheets: {video.stem} ({total_frames} frames, {fps:g} FPS)")

    records = score_all(args, videos)
    expected = len(videos) * len(args.context_frames)
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} score records, got {len(records)}")
    write_csv(records, args.output_dir / "wmreward_scores.csv")
    write_statistical_summary(records, args.context_frames, args.output_dir / "summary.json")
    plot_distributions(records, args.output_dir / "plots" / "distributions.png", args.context_frames)
    plot_curves(records, args.output_dir / "plots" / "context_curves.png", args.context_frames)
    plot_heatmap(records, args.output_dir / "plots" / "score_heatmap.png", args.context_frames)
    write_dashboard(records, videos, args)
    print_summary(records, args.context_frames)
    print(f"\nDashboard: {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
