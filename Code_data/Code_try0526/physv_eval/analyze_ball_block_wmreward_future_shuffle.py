#!/usr/bin/env python3
"""Measure WMReward sensitivity to window-local future-frame shuffling."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
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
RESULT_ROOT = Path("/data/gaoya/agent-data/outputs/wmreward_ball_block_context_full150")
CONTEXT_LENGTHS = (1, 5, 8, 10)
SHUFFLE_SEED = 20260808


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--context-frames", type=int, nargs="+", default=list(CONTEXT_LENGTHS))
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--model", default="vitg384")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-seed", type=int, default=SHUFFLE_SEED)
    parser.add_argument("--cuda-visible-devices", default=None)
    return parser.parse_args()


def load_original_scores(path: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Original WMReward results are required: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    lookup = {(item["video_stem"], int(item["context_frames"])): item for item in records}
    return records, lookup


def save_results(records: list[dict[str, Any]], path: Path, args: argparse.Namespace) -> None:
    payload = {
        "intervention": "independent future-frame permutation inside every 16-frame WMReward window",
        "context_policy": "first N frames of each window remain fixed and ordered",
        "shuffle_seed": args.shuffle_seed,
        "max_frames": args.max_frames,
        "window_size": args.window_size,
        "stride": args.stride,
        "records": records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def future_order(context: int, window_size: int, seed: int, window_index: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + window_index)
    shuffled = np.arange(context, window_size)
    rng.shuffle(shuffled)
    if np.array_equal(shuffled, np.arange(context, window_size)):
        shuffled = np.roll(shuffled, 1)
    return np.concatenate([np.arange(context), shuffled])


def read_window(video_path: Path, window_size: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < window_size:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != window_size:
        raise RuntimeError(f"Need {window_size} frames from {video_path}, got {len(frames)}")
    return frames


def write_window_sheet(
    frames: list[np.ndarray],
    context: int,
    output_path: Path,
    seed: int,
) -> None:
    order = future_order(context, len(frames), seed)
    tile_width, tile_height, label_height = 160, 90, 22
    columns = 8
    sequence_rows = math.ceil(len(frames) / columns)
    band_height = sequence_rows * (tile_height + label_height) + 28
    canvas = np.full((2 * band_height, columns * tile_width, 3), 244, np.uint8)

    def draw_sequence(frame_order: np.ndarray, offset: int, title: str) -> None:
        cv2.putText(canvas, title, (8, offset + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 55, 48), 1, cv2.LINE_AA)
        for position, frame_index in enumerate(frame_order):
            row, column = divmod(position, columns)
            top = offset + 28 + row * (tile_height + label_height)
            left = column * tile_width
            tile = cv2.resize(frames[int(frame_index)], (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            canvas[top : top + tile_height, left : left + tile_width] = tile
            is_context = position < context
            color = (44, 112, 92) if is_context else (50, 80, 190)
            cv2.rectangle(canvas, (left, top), (left + tile_width - 1, top + tile_height - 1), color, 2)
            label = f"pos {position}: f{int(frame_index)} {'CTX' if is_context else 'FUT'}"
            cv2.putText(canvas, label, (left + 4, top + tile_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.37, color, 1, cv2.LINE_AA)

    draw_sequence(np.arange(len(frames)), 0, "Original window")
    draw_sequence(order, band_height, "Future shuffled; context unchanged")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 91]):
        raise RuntimeError(f"Failed to write {output_path}")


def create_sheets(videos: list[Path], args: argparse.Namespace) -> None:
    for video in videos:
        frames = read_window(video, args.window_size)
        for context in args.context_frames:
            write_window_sheet(
                frames,
                context,
                args.result_root / "future_shuffle" / "windows" / video.stem / f"ctx_{context:02d}.jpg",
                args.shuffle_seed,
            )
        print(f"shuffle sheets: {video.stem}")


def score_shuffled(
    videos: list[Path],
    originals: dict[tuple[str, int], dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    output_path = args.result_root / "future_shuffle" / "future_shuffle_scores.json"
    records: list[dict[str, Any]] = []
    if output_path.exists():
        records = json.loads(output_path.read_text(encoding="utf-8")).get("records", [])
    completed = {(item["video_stem"], int(item["context_frames"])) for item in records}
    if len(completed) == len(videos) * len(args.context_frames):
        print("All shuffled scores already exist; model loading is skipped.")
        return records

    runner = WMRewardRunner(
        cuda_visible_devices=args.cuda_visible_devices,
        model_name=args.model,
        window_size=args.window_size,
        context_frames=args.context_frames[0],
        stride=args.stride,
        seed=args.seed,
        max_frames=args.max_frames,
    )
    for video_index, video in enumerate(videos, 1):
        missing = [context for context in args.context_frames if (video.stem, context) not in completed]
        if not missing:
            continue
        print(f"[{video_index}/{len(videos)}] decode {video.stem}", flush=True)
        tensor = runner.load_video(video, max_frames=args.max_frames)
        if int(tensor.shape[2]) != args.max_frames:
            raise RuntimeError(f"Expected {args.max_frames} frames for {video}")
        for context in missing:
            print(f"  shuffled context={context}", flush=True)
            shuffled = runner.score_tensor(
                tensor,
                context_frames=context,
                shuffle_future=True,
                shuffle_seed=args.shuffle_seed,
            )
            original = originals[(video.stem, context)]
            original_surprise = float(original["surprise"])
            shuffled_surprise = float(shuffled["surprise"])
            record = {
                "video_stem": video.stem,
                "group": original["group"],
                "video": str(video),
                "context_frames": context,
                "original_surprise": original_surprise,
                "shuffled_surprise": shuffled_surprise,
                "surprise_delta": shuffled_surprise - original_surprise,
                "surprise_ratio": shuffled_surprise / original_surprise,
                "original_similarity": float(original["similarity"]),
                "shuffled_similarity": float(shuffled["similarity"]),
                "video_frames_loaded": int(shuffled["video_frames_loaded"]),
                "windows": 1 + (args.max_frames - args.window_size) // args.stride,
                "shuffle_seed": args.shuffle_seed,
            }
            records.append(record)
            completed.add((video.stem, context))
            save_results(records, output_path, args)
        del tensor
        if runner._torch.cuda.is_available():
            runner._torch.cuda.empty_cache()
    return records


def write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    fields = (
        "video_stem", "group", "context_frames", "original_surprise", "shuffled_surprise",
        "surprise_delta", "surprise_ratio", "original_similarity", "shuffled_similarity",
        "video_frames_loaded", "windows", "shuffle_seed", "video",
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda item: (item["video_stem"], item["context_frames"])))


def plot_delta(records: list[dict[str, Any]], output_path: Path, contexts: list[int]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    rng = np.random.default_rng(7)
    deltas = [[item["surprise_delta"] for item in records if item["context_frames"] == context] for context in contexts]
    ratios = [[item["surprise_ratio"] for item in records if item["context_frames"] == context] for context in contexts]
    for axis, values, ylabel, reference in (
        (axes[0], deltas, "Shuffled - original surprise", 0.0),
        (axes[1], ratios, "Shuffled / original surprise", 1.0),
    ):
        boxes = axis.boxplot(values, positions=range(len(contexts)), widths=.55, patch_artist=True)
        for box in boxes["boxes"]:
            box.set(facecolor="#d9dfca", edgecolor="#224d43")
        for index, group in enumerate(values):
            axis.scatter(rng.normal(index, .045, len(group)), group, s=24, color="#c4512d", alpha=.7)
        axis.axhline(reference, color="#202a27", linestyle="--", linewidth=1)
        axis.set_xticks(range(len(contexts)), contexts)
        axis.set_xlabel("Context frames")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#d8d0bf", alpha=.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Paired WMReward response to future-frame shuffle", fontsize=19, fontweight="bold")
    figure.savefig(output_path, dpi=180, facecolor="#f1ecdf")
    plt.close(figure)


def plot_scatter(records: list[dict[str, Any]], output_path: Path, contexts: list[int]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
    for axis, context in zip(axes.flat, contexts):
        subset = [item for item in records if item["context_frames"] == context]
        original = np.asarray([item["original_surprise"] for item in subset])
        shuffled = np.asarray([item["shuffled_surprise"] for item in subset])
        low, high = min(original.min(), shuffled.min()), max(original.max(), shuffled.max())
        axis.scatter(original, shuffled, color="#c4512d", alpha=.75)
        axis.plot([low, high], [low, high], color="#19352f", linestyle="--")
        axis.set_xlabel("Original surprise")
        axis.set_ylabel("Shuffled surprise")
        axis.set_title(f"Context {context}: {np.mean(shuffled > original):.0%} increased", loc="left", fontweight="bold")
        axis.grid(color="#d8d0bf", alpha=.65)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output_path, dpi=180, facecolor="#f1ecdf")
    plt.close(figure)


def plot_heatmap(records: list[dict[str, Any]], output_path: Path, contexts: list[int]) -> None:
    by_video: dict[str, dict[int, float]] = defaultdict(dict)
    for item in records:
        by_video[item["video_stem"]][item["context_frames"]] = item["surprise_delta"]
    stems = sorted(by_video, key=lambda stem: np.mean(list(by_video[stem].values())), reverse=True)
    matrix = np.asarray([[by_video[stem][context] for context in contexts] for stem in stems])
    bound = float(np.max(np.abs(matrix)))
    figure, axis = plt.subplots(figsize=(9, 13), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    axis.set_xticks(range(len(contexts)), [f"ctx {context}" for context in contexts])
    axis.set_yticks(range(len(stems)), stems, fontsize=8)
    axis.set_title("Surprise increase after future shuffle", loc="left", fontsize=18, fontweight="bold")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:+.3f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, shrink=.72, label="shuffled - original")
    figure.savefig(output_path, dpi=180, facecolor="#f1ecdf")
    plt.close(figure)


def write_dashboard(records: list[dict[str, Any]], videos: list[Path], args: argparse.Namespace) -> None:
    lookup = {(item["video_stem"], item["context_frames"]): item for item in records}
    cards = []
    for video in videos:
        rows, sheets = [], []
        for context in args.context_frames:
            item = lookup[(video.stem, context)]
            rows.append(
                f"<tr><td>{context}</td><td>{item['original_surprise']:.6f}</td>"
                f"<td>{item['shuffled_surprise']:.6f}</td><td>{item['surprise_delta']:+.6f}</td>"
                f"<td>{item['surprise_ratio']:.3f}x</td></tr>"
            )
            sheets.append(
                f'<figure><img src="future_shuffle/windows/{html.escape(video.stem)}/ctx_{context:02d}.jpg" '
                f'alt="context {context} shuffle"><figcaption>Context {context}; representative window 0</figcaption></figure>'
            )
        cards.append(
            f"""<article class="card"><div class="head"><h2>{html.escape(video.stem)}</h2><button onclick="this.closest('.card').querySelector('video').play()">Play</button></div><video controls muted loop preload="metadata" src="videos/{html.escape(video.name)}"></video><table><thead><tr><th>Ctx</th><th>Original</th><th>Shuffled</th><th>Delta</th><th>Ratio</th></tr></thead><tbody>{''.join(rows)}</tbody></table><div class="sheets">{''.join(sheets)}</div></article>"""
        )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WMReward future shuffle</title><style>
:root{{--ink:#19352f;--paper:#f1ecdf;--card:#fffdf7;--accent:#c4512d;--line:#d8d0bf}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 88% 3%,#cedbc9 0,transparent 28%),var(--paper);color:var(--ink);font-family:Georgia,'Times New Roman',serif}}main{{width:min(1540px,95vw);margin:auto;padding:48px 0 80px}}.eyebrow{{font:700 12px sans-serif;letter-spacing:.18em;text-transform:uppercase;color:var(--accent)}}h1{{font-size:clamp(42px,6vw,80px);line-height:.95;margin:12px 0 20px;max-width:1100px}}.intro{{max-width:1000px;font-size:18px;line-height:1.6}}.notice{{margin:24px 0;padding:17px 20px;background:#fff6e6;border-left:5px solid var(--accent);font:14px/1.55 sans-serif}}.actions{{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0}}button,.button{{border:1px solid var(--ink);background:transparent;color:var(--ink);padding:9px 13px;font:700 13px sans-serif;text-decoration:none;cursor:pointer}}.plot{{display:block;width:100%;margin:24px 0;background:white;border:1px solid var(--line)}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px;margin-top:34px}}.card{{min-width:0;background:var(--card);border:1px solid var(--line);padding:18px}}.head{{display:flex;justify-content:space-between;gap:10px;align-items:center}}h2{{font-size:21px;overflow-wrap:anywhere}}video{{width:100%;aspect-ratio:16/9;background:#111}}table{{width:100%;border-collapse:collapse;margin:12px 0;font:12px sans-serif}}th,td{{padding:7px;border-bottom:1px solid #e4ddcf;text-align:left}}.sheets{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}figure{{margin:0;min-width:0}}figure img{{display:block;width:100%;border:1px solid var(--line)}}figcaption{{padding:5px 0;font:12px sans-serif;color:#66736d}}@media(max-width:900px){{.grid,.sheets{{grid-template-columns:1fr}}}}
</style></head><body><main><div class="eyebrow">WMReward / temporal-order intervention</div><h1>Does future order matter?</h1><p class="intro">For every 16-frame window, the first N context frames remain fixed while only the future frames are independently permuted. Scores aggregate all 17 windows from each complete 150-frame video.</p><div class="notice"><strong>Paired comparison:</strong> positive delta means shuffling made prediction more surprising. Overlapping windows are shuffled independently, so the intervention is a window-level diagnostic rather than one globally coherent shuffled MP4.</div><div class="actions"><button onclick="location.reload()">Manual refresh</button><button onclick="document.querySelectorAll('video').forEach(v=>{{v.currentTime=0;v.play()}})">Replay all</button><a class="button" href="index.html">Original context audit</a><a class="button" href="future_shuffle/future_shuffle_scores.csv">CSV</a><a class="button" href="future_shuffle/future_shuffle_scores.json">JSON</a></div><img class="plot" src="future_shuffle/plots/paired_delta.png" alt="paired deltas"><img class="plot" src="future_shuffle/plots/original_vs_shuffled.png" alt="paired scatter"><img class="plot" src="future_shuffle/plots/delta_heatmap.png" alt="delta heatmap"><div class="grid">{''.join(cards)}</div></main></body></html>"""
    (args.result_root / "future_shuffle.html").write_text(page, encoding="utf-8")


def print_summary(records: list[dict[str, Any]], contexts: list[int]) -> None:
    print("\nFuture-shuffle paired summary")
    for context in contexts:
        subset = [item for item in records if item["context_frames"] == context]
        original = np.asarray([item["original_surprise"] for item in subset])
        shuffled = np.asarray([item["shuffled_surprise"] for item in subset])
        delta = shuffled - original
        print(
            f"ctx={context:2d}: original={original.mean():.6f} shuffled={shuffled.mean():.6f} "
            f"delta={delta.mean():+.6f} ratio={(shuffled / original).mean():.4f}x "
            f"increased={np.mean(delta > 0):.1%}"
        )


def main() -> None:
    args = parse_args()
    args.context_frames = sorted(set(args.context_frames))
    output_dir = args.result_root / "future_shuffle"
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    videos = sorted(args.dataset_dir.glob("*.mp4"))
    _, originals = load_original_scores(args.result_root / "wmreward_scores.json")
    expected_keys = {(video.stem, context) for video in videos for context in args.context_frames}
    if expected_keys - originals.keys():
        raise RuntimeError("Original score matrix is incomplete")
    create_sheets(videos, args)
    records = score_shuffled(videos, originals, args)
    if len(records) != len(expected_keys):
        raise RuntimeError(f"Expected {len(expected_keys)} shuffled scores, got {len(records)}")
    write_csv(records, output_dir / "future_shuffle_scores.csv")
    plot_delta(records, output_dir / "plots" / "paired_delta.png", args.context_frames)
    plot_scatter(records, output_dir / "plots" / "original_vs_shuffled.png", args.context_frames)
    plot_heatmap(records, output_dir / "plots" / "delta_heatmap.png", args.context_frames)
    write_dashboard(records, videos, args)
    print_summary(records, args.context_frames)
    print(f"\nDashboard: {args.result_root / 'future_shuffle.html'}")


if __name__ == "__main__":
    main()
