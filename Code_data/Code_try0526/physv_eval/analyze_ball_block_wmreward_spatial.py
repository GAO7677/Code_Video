#!/usr/bin/env python3
"""Visualize spatial WMReward token surprise for original and shuffled futures."""

from __future__ import annotations

import argparse
import html
import json
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
CONTEXTS = (1, 5, 8, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--context-frames", type=int, nargs="+", default=list(CONTEXTS))
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--model", default="vitg384")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-seed", type=int, default=20260808)
    parser.add_argument("--cuda-visible-devices", default=None)
    return parser.parse_args()


def cache_path(root: Path, stem: str, context: int) -> Path:
    return root / "spatial" / "maps" / stem / f"ctx_{context:02d}.npz"


def compute_maps(videos: list[Path], args: argparse.Namespace) -> None:
    pending = [
        (video, context)
        for video in videos
        for context in args.context_frames
        if not cache_path(args.result_root, video.stem, context).exists()
    ]
    if not pending:
        print("All spatial maps already exist; model loading is skipped.")
        return
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
        missing = [
            context for context in args.context_frames
            if not cache_path(args.result_root, video.stem, context).exists()
        ]
        if not missing:
            continue
        print(f"[{video_index}/{len(videos)}] decode {video.stem}", flush=True)
        tensor = runner.load_video(video, max_frames=args.max_frames)
        for context in missing:
            print(f"  spatial context={context}: original", flush=True)
            original = runner.spatial_score_tensor(
                tensor, context_frames=context, shuffle_future=False
            )
            print(f"  spatial context={context}: shuffled", flush=True)
            shuffled = runner.spatial_score_tensor(
                tensor,
                context_frames=context,
                shuffle_future=True,
                shuffle_seed=args.shuffle_seed,
            )
            output = cache_path(args.result_root, video.stem, context)
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output,
                original=original.astype(np.float32),
                shuffled=shuffled.astype(np.float32),
                delta=(shuffled - original).astype(np.float32),
            )
        del tensor
        if runner._torch.cuda.is_available():
            runner._torch.cuda.empty_cache()


def first_frame(video: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read first frame: {video}")
    return frame


def colorize(values: np.ndarray, low: float, high: float, colormap: int) -> np.ndarray:
    normalized = np.clip((values - low) / max(high - low, 1e-8), 0.0, 1.0)
    return cv2.applyColorMap((normalized * 255).astype(np.uint8), colormap)


def overlay(frame: np.ndarray, heatmap: np.ndarray, low: float, high: float) -> np.ndarray:
    resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    colored = colorize(resized, low, high, cv2.COLORMAP_TURBO)
    return cv2.addWeighted(frame, 0.55, colored, 0.45, 0.0)


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 42), (20, 34, 30), -1)
    cv2.putText(result, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .72, (245, 242, 232), 2, cv2.LINE_AA)
    return result


def collect_maps(videos: list[Path], args: argparse.Namespace) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    maps = {}
    for video in videos:
        for context in args.context_frames:
            with np.load(cache_path(args.result_root, video.stem, context)) as payload:
                maps[(video.stem, context)] = {
                    key: payload[key].copy() for key in ("original", "shuffled", "delta")
                }
    return maps


def render_assets(
    videos: list[Path],
    maps: dict[tuple[str, int], dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, float]:
    absolute_maps = np.concatenate([
        item[key].reshape(-1)
        for item in maps.values()
        for key in ("original", "shuffled")
    ])
    delta_maps = np.concatenate([item["delta"].reshape(-1) for item in maps.values()])
    low, high = (float(value) for value in np.percentile(absolute_maps, [1, 99]))
    delta_bound = float(np.percentile(np.abs(delta_maps), 99))
    image_root = args.result_root / "spatial" / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    for video in videos:
        frame = first_frame(video)
        for context in args.context_frames:
            item = maps[(video.stem, context)]
            original_panel = label(overlay(frame, item["original"], low, high), f"Original spatial surprise | ctx={context}")
            shuffled_panel = label(overlay(frame, item["shuffled"], low, high), "Future shuffled")
            delta_resized = cv2.resize(item["delta"], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
            delta_color = colorize(delta_resized, -delta_bound, delta_bound, cv2.COLORMAP_TURBO)
            delta_panel = label(cv2.addWeighted(frame, .45, delta_color, .55, 0), "Delta: shuffled - original")
            triptych = np.concatenate([label(frame, "Raw first frame"), original_panel, shuffled_panel, delta_panel], axis=1)
            output = image_root / video.stem / f"ctx_{context:02d}.jpg"
            output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output), triptych, [cv2.IMWRITE_JPEG_QUALITY, 91])

    figure, axes = plt.subplots(len(args.context_frames), 3, figsize=(13, 3.5 * len(args.context_frames)), constrained_layout=True)
    for row, context in enumerate(args.context_frames):
        originals = np.stack([maps[(video.stem, context)]["original"] for video in videos]).mean(0)
        shuffled = np.stack([maps[(video.stem, context)]["shuffled"] for video in videos]).mean(0)
        delta = shuffled - originals
        for column, (values, title, cmap, vmin, vmax) in enumerate((
            (originals, "Original", "turbo", low, high),
            (shuffled, "Shuffled", "turbo", low, high),
            (delta, "Shuffled - original", "RdBu_r", -delta_bound, delta_bound),
        )):
            image = axes[row, column].imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, column].set_title(f"ctx={context} | {title}", loc="left", fontweight="bold")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            figure.colorbar(image, ax=axes[row, column], shrink=.75)
    figure.suptitle("Dataset-average spatial token surprise", fontsize=20, fontweight="bold")
    figure.savefig(args.result_root / "spatial" / "dataset_average.png", dpi=180, facecolor="#f1ecdf")
    plt.close(figure)
    return {"absolute_low": low, "absolute_high": high, "delta_bound": delta_bound}


def write_summary(
    videos: list[Path],
    maps: dict[tuple[str, int], dict[str, np.ndarray]],
    scales: dict[str, float],
    args: argparse.Namespace,
) -> None:
    records = []
    for video in videos:
        for context in args.context_frames:
            item = maps[(video.stem, context)]
            peak = np.unravel_index(np.argmax(item["original"]), item["original"].shape)
            records.append({
                "video_stem": video.stem,
                "context_frames": context,
                "original_mean": float(item["original"].mean()),
                "original_max": float(item["original"].max()),
                "shuffled_mean": float(item["shuffled"].mean()),
                "delta_mean": float(item["delta"].mean()),
                "delta_abs_mean": float(np.abs(item["delta"]).mean()),
                "original_peak_patch_yx": [int(peak[0]), int(peak[1])],
            })
    payload = {
        "metric": "per predicted patch token: 1-cosine over feature dimension",
        "aggregation": "mean over temporal target tokens and all sliding windows",
        "warning": "not an exact decomposition of the upstream scalar loss, which uses cosine dim=1",
        "scales": scales,
        "records": records,
    }
    (args.result_root / "spatial" / "spatial_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def write_dashboard(videos: list[Path], args: argparse.Namespace) -> None:
    cards = []
    for video in videos:
        figures = "".join(
            f'<figure><img src="spatial/images/{html.escape(video.stem)}/ctx_{context:02d}.jpg" '
            f'alt="spatial WMReward context {context}"><figcaption>Context {context}</figcaption></figure>'
            for context in args.context_frames
        )
        cards.append(
            f'<article><div class="head"><h2>{html.escape(video.stem)}</h2><button onclick="this.closest(\'article\').querySelector(\'video\').play()">Play</button></div>'
            f'<video controls muted loop preload="metadata" src="videos/{html.escape(video.name)}"></video><div class="maps">{figures}</div></article>'
        )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Spatial WMReward</title><style>
:root{{--ink:#19352f;--paper:#f1ecdf;--card:#fffdf7;--accent:#c4512d;--line:#d8d0bf}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 88% 3%,#cedbc9 0,transparent 28%),var(--paper);color:var(--ink);font-family:Georgia,'Times New Roman',serif}}main{{width:min(1560px,96vw);margin:auto;padding:48px 0 80px}}.eyebrow{{font:700 12px sans-serif;letter-spacing:.18em;text-transform:uppercase;color:var(--accent)}}h1{{font-size:clamp(42px,6vw,80px);line-height:.95;margin:12px 0 20px}}.intro{{font-size:18px;line-height:1.6;max-width:1050px}}.notice{{margin:24px 0;padding:17px 20px;background:#fff6e6;border-left:5px solid var(--accent);font:14px/1.55 sans-serif}}.actions{{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0}}button,.button{{border:1px solid var(--ink);background:transparent;color:var(--ink);padding:9px 13px;font:700 13px sans-serif;text-decoration:none;cursor:pointer}}.average{{display:block;width:100%;background:white;border:1px solid var(--line)}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px;margin-top:30px}}article{{min-width:0;background:var(--card);padding:18px;border:1px solid var(--line)}}.head{{display:flex;justify-content:space-between;align-items:center;gap:10px}}h2{{font-size:21px;overflow-wrap:anywhere}}video{{width:100%;aspect-ratio:16/9;background:#111}}.maps{{display:grid;grid-template-columns:1fr;gap:12px;margin-top:13px}}figure{{margin:0;min-width:0}}figure img{{display:block;width:100%;border:1px solid var(--line)}}figcaption{{padding:5px 0;font:12px sans-serif;color:#65716c}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><div class="eyebrow">WMReward / spatial token audit</div><h1>Where is prediction surprising?</h1><p class="intro">Patch-level future prediction surprise is aggregated over all temporal targets and 17 sliding windows. Each row compares the raw frame, original future, shuffled future, and their spatial difference for contexts 1, 5, 8, and 10.</p><div class="notice"><strong>Important:</strong> this map uses per-token cosine over the feature dimension. It is a spatial diagnostic aligned with WMReward models and masks, but not an exact decomposition of the upstream scalar implementation, which uses cosine <code>dim=1</code>.</div><div class="actions"><button onclick="location.reload()">Manual refresh</button><button onclick="document.querySelectorAll('video').forEach(v=>{{v.currentTime=0;v.play()}})">Replay all</button><a class="button" href="index.html">Context audit</a><a class="button" href="future_shuffle.html">Shuffle audit</a><a class="button" href="spatial/spatial_summary.json">JSON</a></div><img class="average" src="spatial/dataset_average.png" alt="dataset average spatial maps"><div class="grid">{''.join(cards)}</div></main></body></html>"""
    (args.result_root / "spatial_wmreward.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.context_frames = sorted(set(args.context_frames))
    videos = sorted(args.dataset_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(args.dataset_dir)
    (args.result_root / "spatial").mkdir(parents=True, exist_ok=True)
    compute_maps(videos, args)
    maps = collect_maps(videos, args)
    scales = render_assets(videos, maps, args)
    write_summary(videos, maps, scales, args)
    write_dashboard(videos, args)
    print(f"Dashboard: {args.result_root / 'spatial_wmreward.html'}")


if __name__ == "__main__":
    main()
