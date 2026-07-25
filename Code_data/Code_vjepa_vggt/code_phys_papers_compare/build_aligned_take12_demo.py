from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


TRY0526_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
OFFICIAL_REPO_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/"
    "code_phys_papers_compare/google-deepmind-physics-iq-benchmark"
)
for path in (TRY0526_ROOT, OFFICIAL_REPO_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from physiq.binary_mask_generator import generate_mask  # noqa: E402
from physiq.calculate_and_write_metrics_to_csv import (  # noqa: E402
    ViewPaths,
    compute_view_metrics,
    load_view,
)
from physv_eval.single_case.physics_iq import _read_video, _write_video  # noqa: E402
from physv_eval.single_case.physics_iq_verified_proxy import (  # noqa: E402
    _component_scores,
    _export_scored_inputs,
)


DEFAULT_RESULT_JSON = Path(
    "/data/gaoya/AAA_test_video/0623/test/v2v_wan/PhyRVG/baseline/physicIQ/"
    "physRVG_steps40_512x896_08_49f/"
    "physicIQ_Solid_Mechanics_0107_perspective-center_trimmed-marble-run-y.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physics_iq_aligned_take12_demo_fixed/"
    "physrvg_baseline_0107_marble_run_y"
)
DEFAULT_DATASET_ROOT = Path(
    "/data/gaoya/dataset/Anates-Labs-Research-Physics-IQ-Verified"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an event-relative aligned Take-1/Take-2 Physics-IQ proxy demo."
    )
    parser.add_argument("--result-json", type=Path, default=DEFAULT_RESULT_JSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--take1-id", type=int, default=107)
    parser.add_argument("--take2-id", type=int, default=305)
    parser.add_argument("--view", default="center")
    parser.add_argument("--scene", default="marble-run-y")
    parser.add_argument("--context-frames", type=int, default=8)
    return parser.parse_args()


def probe(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    metadata = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return metadata


def load_thumbnails(path: Path, size: tuple[int, int] = (160, 90)) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        thumb = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32))
    capture.release()
    if not frames:
        raise RuntimeError(f"No readable frames: {path}")
    return np.stack(frames)


def load_alignment_features(path: Path) -> np.ndarray:
    frames = load_thumbnails(path, size=(64, 36)).reshape(-1, 64 * 36)
    means = frames.mean(axis=1, keepdims=True)
    standard_deviations = frames.std(axis=1, keepdims=True)
    return (frames - means) / (standard_deviations + 1e-4)


def find_sequence_offset(query: np.ndarray, target: np.ndarray) -> tuple[int, float, float]:
    if len(query) > len(target):
        raise ValueError("Query sequence is longer than target")
    sample_count = min(32, len(query))
    sample_indices = np.unique(
        np.linspace(0, len(query) - 1, num=sample_count).round().astype(np.int64)
    )
    scores: list[float] = []
    for start in range(len(target) - len(query) + 1):
        difference = query[sample_indices] - target[start + sample_indices]
        scores.append(float(np.mean(difference**2)))
    order = np.argsort(scores)
    best = int(order[0])
    second = float(scores[int(order[1])]) if len(order) > 1 else float(scores[best])
    return best, float(scores[best]), second


def subsequence_dtw(
    query: np.ndarray,
    target: np.ndarray,
) -> tuple[dict[int, int], float, int, int]:
    query_norm = np.sum(query * query, axis=1)[:, None]
    target_norm = np.sum(target * target, axis=1)[None, :]
    pairwise_cost = np.maximum(
        query_norm + target_norm - 2.0 * query @ target.T,
        0.0,
    ) / query.shape[1]
    query_count, target_count = pairwise_cost.shape
    accumulated = np.full((query_count + 1, target_count + 1), np.inf)
    accumulated[0, :] = 0.0
    predecessors = np.zeros((query_count + 1, target_count + 1), dtype=np.uint8)
    for query_index in range(1, query_count + 1):
        for target_index in range(1, target_count + 1):
            options = (
                accumulated[query_index - 1, target_index - 1],
                accumulated[query_index - 1, target_index],
                accumulated[query_index, target_index - 1],
            )
            predecessor = int(np.argmin(options))
            predecessors[query_index, target_index] = predecessor
            accumulated[query_index, target_index] = (
                pairwise_cost[query_index - 1, target_index - 1]
                + options[predecessor]
            )

    query_index = query_count
    target_index = int(np.argmin(accumulated[query_count, 1:])) + 1
    path: list[tuple[int, int]] = []
    while query_index > 0:
        path.append((query_index - 1, target_index - 1))
        predecessor = predecessors[query_index, target_index]
        if predecessor == 0:
            query_index -= 1
            target_index -= 1
        elif predecessor == 1:
            query_index -= 1
        else:
            target_index -= 1
    path.reverse()

    targets_by_query: dict[int, list[int]] = {}
    for query_frame, target_frame in path:
        targets_by_query.setdefault(query_frame, []).append(target_frame)
    mapping = {
        query_frame: int(round(float(np.median(target_frames))))
        for query_frame, target_frames in targets_by_query.items()
    }
    normalized_cost = float(
        accumulated[query_count, path[-1][1] + 1] / max(len(path), 1)
    )
    return mapping, normalized_cost, path[0][1], path[-1][1]


def read_video_range(source: Path, start: int, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    frames: list[np.ndarray] = []
    frame_index = 0
    while frame_index < start + count:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index >= start:
            frames.append(frame)
        frame_index += 1
    capture.release()
    if len(frames) != count:
        raise ValueError(
            f"Requested {count} frames from {source} at {start}, read {len(frames)}"
        )
    return frames


def clip_split_timeline(
    conditioning: Path,
    testing: Path,
    boundary: int,
    start: int,
    count: int,
    destination: Path,
    fps: float,
) -> None:
    frames: list[np.ndarray] = []
    remaining = count
    cursor = start
    if cursor < boundary:
        conditioning_count = min(remaining, boundary - cursor)
        frames.extend(read_video_range(conditioning, cursor, conditioning_count))
        remaining -= conditioning_count
        cursor += conditioning_count
    if remaining:
        testing_start = cursor - boundary
        if testing_start < 0:
            raise ValueError(f"Invalid testing start derived from cursor={cursor}")
        frames.extend(read_video_range(testing, testing_start, remaining))
    if len(frames) != count:
        raise ValueError(f"Expected {count} combined frames, got {len(frames)}")
    _write_video(frames, destination, fps)


def generated_mask_path(mask_dir: Path, take1_id: int, fps: int, view: str, scene: str) -> Path:
    return mask_dir / (
        f"{take1_id:04d}_video-masks_{fps}FPS_perspective-{view}_"
        f"take-1_trimmed-{scene}.mp4"
    )


def real_mask_path(
    mask_dir: Path,
    case_id: int,
    fps: int,
    view: str,
    take: int,
    scene: str,
) -> Path:
    return mask_dir / (
        f"{case_id:04d}_video-masks_{fps}FPS_perspective-{view}_"
        f"take-{take}_trimmed-{scene}.mp4"
    )


def mask_activity(path: Path) -> dict[str, int]:
    capture = cv2.VideoCapture(str(path))
    counts: list[int] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        counts.append(int(np.count_nonzero(frame[:, :, 0] > 127)))
    capture.release()
    return {
        "frames": len(counts),
        "empty_frames": sum(count == 0 for count in counts),
        "nonempty_frames": sum(count > 0 for count in counts),
        "active_pixels_total": sum(counts),
        "active_pixels_max": max(counts, default=0),
    }


def build_gallery(output_root: Path, manifest: dict[str, object]) -> None:
    cards = [
        ("Generated", "scored_inputs/generated_video.mp4", "scored_inputs/generated_mask.mp4"),
        ("Take-1", "scored_inputs/take1_video.mp4", "scored_inputs/take1_mask.mp4"),
        ("Take-2", "scored_inputs/take2_video.mp4", "scored_inputs/take2_mask.mp4"),
    ]
    card_html = "\n".join(
        f"""
        <article>
          <h2>{html.escape(title)}</h2>
          <h3>Video</h3>
          <video controls loop muted preload="metadata" src="{video}"></video>
          <h3>Mask</h3>
          <video controls loop muted preload="metadata" src="{mask}"></video>
        </article>
        """
        for title, video, mask in cards
    )
    alignment = manifest["alignment"]
    score = manifest["score"]
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Aligned Physics-IQ Take-1/Take-2 demo</title>
  <style>
    :root {{ color-scheme: dark; font-family: Arial, sans-serif; background: #111; color: #eee; }}
    body {{ margin: 0; padding: 24px; }}
    header {{ max-width: 1180px; margin: 0 auto 22px; }}
    h1 {{ margin: 0 0 10px; font-size: 26px; letter-spacing: 0; }}
    p {{ color: #bbb; line-height: 1.55; }}
    .facts {{ display: flex; flex-wrap: wrap; gap: 10px 22px; color: #ddd; }}
    main {{ max-width: 1500px; margin: 0 auto; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    article {{ border: 1px solid #333; border-radius: 6px; padding: 12px; background: #181818; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; }}
    h3 {{ margin: 12px 0 7px; color: #aaa; font-size: 13px; text-transform: uppercase; }}
    video {{ display: block; width: 100%; aspect-ratio: 16 / 9; background: #000; }}
    a {{ color: #7ec8ff; }}
    @media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Aligned Take-1 / Take-2 / Generated</h1>
    <p>These six 41-frame, 30 FPS files are the resized arrays passed to the official
       <code>compute_view_metrics()</code>. Masks are shown without overlays.</p>
    <div class="facts">
      <span>Take-1 start: frame {alignment["take1_future_start"]}</span>
      <span>Take-2 start: frame {alignment["take2_future_start"]}</span>
      <span>Official boundary offset: {alignment["relative_to_official_boundary_frames"]} frames</span>
      <span>Score: {score["final_score_100"]}</span>
      <span><a href="alignment_manifest.json">alignment_manifest.json</a></span>
    </div>
  </header>
  <main>{card_html}</main>
</body>
</html>
"""
    (output_root / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_json = args.result_json.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    result_payload = json.loads(result_json.read_text(encoding="utf-8"))
    input_json = Path(result_payload["input_json"])
    input_payload = json.loads(input_json.read_text(encoding="utf-8"))
    source_video = Path(input_payload["source_video"])
    context_video = Path(input_payload["input_video"])
    candidate_video = Path(result_payload["output_video"])

    fps = int(round(probe(candidate_video)["fps"]))
    take1_full = next(
        (dataset_root / "full-videos" / "take-1" / f"{fps}FPS").glob(
            f"{args.take1_id:04d}_full-videos_{fps}FPS_perspective-{args.view}_"
            f"take-1_trimmed-{args.scene}.mp4"
        )
    )
    take2_full = next(
        (dataset_root / "full-videos" / "take-2" / f"{fps}FPS").glob(
            f"{args.take2_id:04d}_full-videos_{fps}FPS_perspective-{args.view}_"
            f"take-2_trimmed-{args.scene}.mp4"
        )
    )
    take1_conditioning = (
        dataset_root
        / "split-videos"
        / "conditioning"
        / f"{fps}FPS"
        / (
            f"{args.take1_id:04d}_conditioning-videos_{fps}FPS_"
            f"perspective-{args.view}_take-1_trimmed-{args.scene}.mp4"
        )
    )
    take2_conditioning = (
        dataset_root
        / "split-videos"
        / "conditioning"
        / f"{fps}FPS"
        / (
            f"{args.take2_id:04d}_conditioning-videos_{fps}FPS_"
            f"perspective-{args.view}_take-2_trimmed-{args.scene}.mp4"
        )
    )
    take1_testing = (
        dataset_root
        / "split-videos"
        / "testing"
        / f"{fps}FPS"
        / (
            f"{args.take1_id:04d}_testing-videos_{fps}FPS_"
            f"perspective-{args.view}_take-1_trimmed-{args.scene}.mp4"
        )
    )
    take2_testing = (
        dataset_root
        / "split-videos"
        / "testing"
        / f"{fps}FPS"
        / (
            f"{args.take2_id:04d}_testing-videos_{fps}FPS_"
            f"perspective-{args.view}_take-2_trimmed-{args.scene}.mp4"
        )
    )

    source_features = load_alignment_features(source_video)
    take1_features = load_alignment_features(take1_full)
    source_to_take1, source_match_cost, source_start, source_end = subsequence_dtw(
        source_features,
        take1_features,
    )
    source_thumbnails = load_thumbnails(source_video)
    context_thumbnails = load_thumbnails(context_video)
    context_start, context_match_mse, context_second_mse = find_sequence_offset(
        context_thumbnails,
        source_thumbnails,
    )

    candidate_frames, candidate_fps = _read_video(candidate_video)
    generated_frames = candidate_frames[int(args.context_frames) :]
    generated_count = len(generated_frames)
    take1_boundary = int(probe(take1_conditioning)["frames"])
    take2_boundary = int(probe(take2_conditioning)["frames"])
    future_source_frame = context_start + int(args.context_frames)
    if future_source_frame not in source_to_take1:
        raise ValueError(
            f"DTW mapping does not contain source frame {future_source_frame}"
        )
    take1_future_start = source_to_take1[future_source_frame]
    relative_offset = take1_future_start - take1_boundary
    take2_future_start = take2_boundary + relative_offset

    raw_dir = output_root / "aligned_raw"
    mask_dir = output_root / "aligned_raw_masks"
    generated_mask_dir = mask_dir / "generated"
    take1_mask_dir = mask_dir / "take1"
    take2_mask_dir = mask_dir / "take2"
    raw_dir.mkdir(parents=True, exist_ok=True)
    generated_mask_dir.mkdir(parents=True, exist_ok=True)
    take1_mask_dir.mkdir(parents=True, exist_ok=True)
    take2_mask_dir.mkdir(parents=True, exist_ok=True)
    generated_name = (
        f"{args.take1_id:04d}_perspective-{args.view}_trimmed-{args.scene}.mp4"
    )
    take1_name = (
        f"{args.take1_id:04d}_testing-videos_{fps}FPS_perspective-{args.view}_"
        f"take-1_trimmed-{args.scene}.mp4"
    )
    take2_name = (
        f"{args.take2_id:04d}_testing-videos_{fps}FPS_perspective-{args.view}_"
        f"take-2_trimmed-{args.scene}.mp4"
    )
    generated_path = raw_dir / generated_name
    take1_path = raw_dir / take1_name
    take2_path = raw_dir / take2_name
    _write_video(generated_frames, generated_path, candidate_fps)
    clip_split_timeline(
        take1_conditioning,
        take1_testing,
        take1_boundary,
        take1_future_start,
        generated_count,
        take1_path,
        fps,
    )
    clip_split_timeline(
        take2_conditioning,
        take2_testing,
        take2_boundary,
        take2_future_start,
        generated_count,
        take2_path,
        fps,
    )

    generate_mask(
        str(generated_path),
        str(generated_mask_dir / generated_name),
        False,
        10,
    )
    generate_mask(str(take1_path), str(take1_mask_dir / take1_name), True, 10)
    generate_mask(str(take2_path), str(take2_mask_dir / take2_name), True, 10)
    generated_mask = generated_mask_path(
        generated_mask_dir, args.take1_id, fps, args.view, args.scene
    )
    take1_mask = real_mask_path(
        take1_mask_dir, args.take1_id, fps, args.view, 1, args.scene
    )
    take2_mask = real_mask_path(
        take2_mask_dir, args.take2_id, fps, args.view, 2, args.scene
    )

    view_paths = ViewPaths(
        real_v1=str(take1_path),
        real_v2=str(take2_path),
        generated=str(generated_path),
        mask_v1=str(take1_mask),
        mask_v2=str(take2_mask),
        mask_generated=str(generated_mask),
    )
    frames = load_view(view_paths, 0, generated_count, generated_count)
    if frames is None:
        raise RuntimeError("Official load_view could not load the aligned Take-1 clip")
    metrics = compute_view_metrics(frames)
    components = _component_scores(metrics)
    scored_paths = _export_scored_inputs(frames, output_root, fps)
    component_keys = (
        "score_spatiotemporal_iou",
        "score_spatial_iou",
        "score_weighted_spatial_iou",
        "score_mse",
    )
    final_score_01 = float(np.mean([components[key] for key in component_keys]))

    shutil.copy2(result_json, output_root / "source_result.json")
    shutil.copy2(input_json, output_root / "source_input_case.json")
    manifest: dict[str, object] = {
        "method": "event_relative_aligned_take12_proxy_demo",
        "official": False,
        "case": {
            "result_json": str(result_json),
            "input_json": str(input_json),
            "candidate_video": str(candidate_video),
            "source_video": str(source_video),
            "context_video": str(context_video),
            "take1_full": str(take1_full),
            "take2_full": str(take2_full),
        },
        "alignment": {
            "source_start_in_take1_full": source_start,
            "source_end_in_take1_full": source_end,
            "source_match_method": "subsequence_dtw_per_frame_normalized_grayscale",
            "source_match_normalized_cost": source_match_cost,
            "context_start_in_source": context_start,
            "context_match_mse_thumbnail": context_match_mse,
            "context_second_match_mse_thumbnail": context_second_mse,
            "context_frames": int(args.context_frames),
            "take1_official_boundary": take1_boundary,
            "take2_official_boundary": take2_boundary,
            "take1_future_start": take1_future_start,
            "take2_future_start": take2_future_start,
            "relative_to_official_boundary_frames": relative_offset,
            "generated_frames": generated_count,
            "fps": fps,
        },
        "raw_inputs": {
            "generated": str(generated_path),
            "take1": str(take1_path),
            "take2": str(take2_path),
            "generated_mask": str(generated_mask),
            "take1_mask": str(take1_mask),
            "take2_mask": str(take2_mask),
        },
        "scored_inputs": scored_paths,
        "mask_activity": {
            "generated": mask_activity(Path(scored_paths["scored_generated_mask"])),
            "take1": mask_activity(Path(scored_paths["scored_take1_mask"])),
            "take2": mask_activity(Path(scored_paths["scored_take2_mask"])),
        },
        "score": {
            "final_score_01": final_score_01,
            "final_score_100": round(final_score_01 * 100.0, 2),
            **components,
        },
    }
    (output_root / "alignment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    build_gallery(output_root, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
