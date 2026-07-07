from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from decord import VideoReader, cpu


_RECOMMENDED_LABELS = [
    "rigid body motion",
    "collision",
    "elastic motion",
    "deformation",
    "liquid motion",
    "melting",
    "solidification",
    "vaporization",
    "explosion",
]

_LOW_PRIORITY_LABELS = [
    "interference and diffraction",
    "unnatural light source",
    "liquefaction",
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _video_index(videos_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(videos_root.rglob("*.mp4")):
        mapping.setdefault(path.name, path)
    return mapping


def _sample_split_name(text: str, train_ratio: float, val_ratio: float) -> str:
    import hashlib

    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    u = int(digest[:12], 16) / float(16**12 - 1)
    if u < train_ratio:
        return "train"
    if u < train_ratio + val_ratio:
        return "val"
    return "test"


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _score_band(value: float | None, low: float, high: float) -> float:
    if value is None:
        return 0.0
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (float(value) - low) / max(high - low, 1.0e-6)


def _motion_proxy(frames_thwc: np.ndarray) -> float:
    if frames_thwc.shape[0] <= 1:
        return 0.0
    gray = np.asarray(
        [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames_thwc],
        dtype=np.float32,
    )
    diffs = np.abs(gray[1:] - gray[:-1])
    return float(diffs.mean() / 255.0)


def _blur_proxy(frames_thwc: np.ndarray) -> float:
    if frames_thwc.shape[0] <= 0:
        return 0.0
    values = []
    for frame in frames_thwc:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        values.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
    return float(np.mean(values)) if values else 0.0


def _probe_video(video_path: Path, prefix_frames: int) -> dict[str, Any]:
    vr = VideoReader(str(video_path), ctx=cpu(0))
    frame_count = int(len(vr))
    if frame_count <= 0:
        raise RuntimeError("decoded zero frames")
    sample_count = min(int(prefix_frames), frame_count)
    frame_idx = np.arange(sample_count, dtype=np.int64)
    frames = vr.get_batch(frame_idx).asnumpy()
    return {
        "frame_count": frame_count,
        "prefix_frames_decoded": int(sample_count),
        "motion_proxy": _motion_proxy(frames),
        "blur_proxy": _blur_proxy(frames),
    }


def _caption_from_entry(entry: dict[str, Any]) -> str:
    value = entry.get("captions")
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return " ".join(parts[:4])
    return ""


def _recommended_label_score(label: str) -> float:
    if label in _RECOMMENDED_LABELS:
        return 1.0
    if label in _LOW_PRIORITY_LABELS:
        return 0.0
    return 0.4


def _score_record(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    label = str(record.get("label", "") or "")
    duration = _safe_float(record.get("duration"))
    fps = _safe_float(record.get("fps"))
    quality = _safe_float(record.get("visual_quality_score"))
    text_ratio = _safe_float(record.get("text_bbox_ratio"))
    text_bbox_num = _safe_float(record.get("text_bbox_num"))
    width = _safe_int(record.get("width"))
    height = _safe_int(record.get("height"))
    probe = record.get("probe", {})
    motion_proxy = _safe_float(probe.get("motion_proxy"))
    blur_proxy = _safe_float(probe.get("blur_proxy"))

    quality_s = _score_band(quality, args.min_quality, args.min_quality + 1.0)
    fps_s = _score_band(fps, args.min_fps, max(args.min_fps + 10.0, args.min_fps + 1.0))
    duration_s = 0.0
    if duration is not None:
        if duration < args.min_duration or duration > args.max_duration:
            duration_s = 0.0
        else:
            center = 0.5 * (args.min_duration + args.max_duration)
            radius = max(0.5 * (args.max_duration - args.min_duration), 1.0e-6)
            duration_s = _clip01(1.0 - abs(duration - center) / radius)
    resolution_s = 0.0
    if width is not None and height is not None:
        resolution_s = 0.5 * _score_band(width, args.min_width, args.min_width * 2.0) + 0.5 * _score_band(
            height, args.min_height, args.min_height * 2.0
        )
    text_s = 1.0
    if text_ratio is not None:
        text_s = min(text_s, _clip01(1.0 - max(text_ratio, 0.0) / max(args.max_text_ratio, 1.0e-6)))
    if text_bbox_num is not None:
        text_s = min(text_s, _clip01(1.0 - max(text_bbox_num, 0.0) / max(args.max_text_bbox_num, 1.0)))
    motion_s = _score_band(motion_proxy, args.min_motion_proxy, args.min_motion_proxy + 0.06)
    blur_s = _score_band(blur_proxy, args.min_blur_proxy, args.min_blur_proxy + 80.0)
    label_s = _recommended_label_score(label)

    weighted = {
        "label": label_s,
        "quality": quality_s,
        "fps": fps_s,
        "duration": duration_s,
        "resolution": resolution_s,
        "text_cleanliness": text_s,
        "motion_proxy": motion_s,
        "blur_proxy": blur_s,
    }
    total = (
        0.22 * weighted["label"]
        + 0.16 * weighted["quality"]
        + 0.10 * weighted["fps"]
        + 0.10 * weighted["duration"]
        + 0.10 * weighted["resolution"]
        + 0.12 * weighted["text_cleanliness"]
        + 0.10 * weighted["motion_proxy"]
        + 0.10 * weighted["blur_proxy"]
    )
    tier = "C"
    if total >= args.tier_a_score:
        tier = "A"
    elif total >= args.tier_b_score:
        tier = "B"
    return {
        "score_total": round(float(total), 6),
        "score_breakdown": {k: round(float(v), 6) for k, v in weighted.items()},
        "tier": tier,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter WISA videos into a train0705-friendly no-GT-box subset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/gaoya/dataset/qihoo360-WISA-80K"),
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=Path("/data/gaoya/dataset/qihoo360-WISA-80K/videos"),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("/data/gaoya/dataset/qihoo360-WISA-80K/data/wisa-80k.json"),
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--split-train-ratio", type=float, default=0.9)
    parser.add_argument("--split-val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix-frames", type=int, default=24)
    parser.add_argument("--min-quality", type=float, default=4.5)
    parser.add_argument("--min-fps", type=float, default=20.0)
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--min-width", type=int, default=960)
    parser.add_argument("--min-height", type=int, default=540)
    parser.add_argument("--max-text-ratio", type=float, default=0.01)
    parser.add_argument("--max-text-bbox-num", type=float, default=2.0)
    parser.add_argument("--min-motion-proxy", type=float, default=0.01)
    parser.add_argument("--min-blur-proxy", type=float, default=15.0)
    parser.add_argument("--min-frames", type=int, default=24)
    parser.add_argument("--max-per-label", type=int, default=0)
    parser.add_argument("--probe-all-local-videos", action="store_true", default=False)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--only-recommended-labels", action="store_true", default=False)
    parser.add_argument("--exclude-low-priority-labels", action="store_true", default=True)
    parser.add_argument("--include-low-priority-labels", dest="exclude_low_priority_labels", action="store_false")
    parser.add_argument("--tier-a-score", type=float, default=0.72)
    parser.add_argument("--tier-b-score", type=float, default=0.58)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/gaoya/agent-data/outputs/wisa_clean_train0705"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    videos_root = args.videos_root.expanduser().resolve()
    metadata_path = args.metadata_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not videos_root.is_dir():
        raise FileNotFoundError(f"videos root not found: {videos_root}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata json not found: {metadata_path}")

    video_map = _video_index(videos_root)
    metadata = _read_json(metadata_path)
    if not isinstance(metadata, list):
        raise RuntimeError(f"expected list metadata, got {type(metadata).__name__}")

    records_all: list[dict[str, Any]] = []
    keep_records: list[dict[str, Any]] = []
    reject_records: list[dict[str, Any]] = []
    reject_reason_counts = Counter()
    available_by_label = Counter()

    local_candidate_index = 0
    for row_index, entry in enumerate(metadata):
        if not isinstance(entry, dict):
            continue
        video_name = entry.get("video_name")
        label = entry.get("label")
        if not isinstance(video_name, str) or not video_name.strip():
            continue
        if not isinstance(label, str) or not label.strip():
            continue
        local_video = video_map.get(video_name)
        if local_video is None:
            continue

        local_candidate_index += 1
        available_by_label[label] += 1
        item = dict(entry)
        item["row_index"] = row_index
        item["local_video_path"] = str(local_video)
        item["caption"] = _caption_from_entry(entry)
        item["split"] = _sample_split_name(video_name, args.split_train_ratio, args.split_val_ratio)
        reasons: list[str] = []

        if args.split != "all" and item["split"] != args.split:
            reasons.append("split_mismatch")
        if args.only_recommended_labels and label not in _RECOMMENDED_LABELS:
            reasons.append("label_not_in_recommended_set")
        if args.exclude_low_priority_labels and label in _LOW_PRIORITY_LABELS:
            reasons.append("label_low_priority")

        width = _safe_int(entry.get("width"))
        height = _safe_int(entry.get("height"))
        fps = _safe_float(entry.get("fps"))
        duration = _safe_float(entry.get("duration"))
        quality = _safe_float(entry.get("visual_quality_score"))
        text_ratio = _safe_float(entry.get("text_bbox_ratio"))
        text_bbox_num = _safe_float(entry.get("text_bbox_num"))

        if width is None or width < args.min_width:
            reasons.append("width_below_threshold")
        if height is None or height < args.min_height:
            reasons.append("height_below_threshold")
        if fps is None or fps < args.min_fps:
            reasons.append("fps_below_threshold")
        if duration is None or duration < args.min_duration:
            reasons.append("duration_too_short")
        elif duration > args.max_duration:
            reasons.append("duration_too_long")
        if quality is None or quality < args.min_quality:
            reasons.append("quality_below_threshold")
        if text_ratio is not None and text_ratio > args.max_text_ratio:
            reasons.append("text_ratio_too_high")
        if text_bbox_num is not None and text_bbox_num > args.max_text_bbox_num:
            reasons.append("text_bbox_num_too_high")

        should_probe = args.probe_all_local_videos or not reasons
        probe = {}
        if should_probe:
            try:
                probe = _probe_video(local_video, args.prefix_frames)
            except Exception as exc:  # noqa: BLE001
                item["probe_error"] = f"{type(exc).__name__}: {exc}"
                reasons.append("video_decode_failed")
            item["probe"] = probe
            if probe:
                if int(probe.get("frame_count", 0)) < args.min_frames:
                    reasons.append("frame_count_too_small")
                if float(probe.get("motion_proxy", 0.0)) < args.min_motion_proxy:
                    reasons.append("motion_proxy_too_low")
                if float(probe.get("blur_proxy", 0.0)) < args.min_blur_proxy:
                    reasons.append("blur_proxy_too_low")
        else:
            item["probe"] = {"skipped": True}

        scoring = _score_record(item, args)
        item.update(scoring)
        item["reject_reasons"] = sorted(set(reasons))
        item["decision"] = "keep" if not reasons else "reject"
        records_all.append(item)

        if item["decision"] == "keep":
            keep_records.append(item)
        else:
            reject_records.append(item)
            for reason in item["reject_reasons"]:
                reject_reason_counts[reason] += 1

        if args.progress_every > 0 and local_candidate_index % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "progress_local_candidates": local_candidate_index,
                        "keep_so_far": len(keep_records),
                        "reject_so_far": len(reject_records),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    keep_records_sorted = sorted(
        keep_records,
        key=lambda x: (-float(x["score_total"]), str(x.get("label", "")), str(x.get("video_name", ""))),
    )

    balanced_keep_records = keep_records_sorted
    if args.max_per_label > 0:
        grouped = defaultdict(list)
        for item in keep_records_sorted:
            grouped[str(item.get("label", ""))].append(item)
        balanced_keep_records = []
        rng = random.Random(args.seed)
        for label in sorted(grouped):
            items = list(grouped[label])
            rng.shuffle(items)
            items = sorted(items, key=lambda x: -float(x["score_total"]))
            balanced_keep_records.extend(items[: args.max_per_label])
        balanced_keep_records = sorted(
            balanced_keep_records,
            key=lambda x: (-float(x["score_total"]), str(x.get("label", "")), str(x.get("video_name", ""))),
        )

    keep_by_label = Counter(str(x.get("label", "")) for x in keep_records)
    balanced_by_label = Counter(str(x.get("label", "")) for x in balanced_keep_records)
    tier_counts = Counter(str(x.get("tier", "")) for x in keep_records)

    summary = {
        "dataset_root": str(dataset_root),
        "videos_root": str(videos_root),
        "metadata_path": str(metadata_path),
        "split": args.split,
        "available_local_videos": int(sum(available_by_label.values())),
        "available_by_label": dict(sorted(available_by_label.items())),
        "keep_count": len(keep_records),
        "reject_count": len(reject_records),
        "balanced_keep_count": len(balanced_keep_records),
        "keep_by_label": dict(sorted(keep_by_label.items())),
        "balanced_keep_by_label": dict(sorted(balanced_by_label.items())),
        "tier_counts_keep": dict(sorted(tier_counts.items())),
        "reject_reason_counts": dict(sorted(reject_reason_counts.items())),
        "thresholds": {
            "min_quality": args.min_quality,
            "min_fps": args.min_fps,
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
            "min_width": args.min_width,
            "min_height": args.min_height,
            "max_text_ratio": args.max_text_ratio,
            "max_text_bbox_num": args.max_text_bbox_num,
            "min_motion_proxy": args.min_motion_proxy,
            "min_blur_proxy": args.min_blur_proxy,
            "min_frames": args.min_frames,
            "only_recommended_labels": args.only_recommended_labels,
            "exclude_low_priority_labels": args.exclude_low_priority_labels,
            "max_per_label": args.max_per_label,
        },
    }

    files = {
        "summary": output_dir / "summary.json",
        "keep": output_dir / "keep_manifest.json",
        "keep_balanced": output_dir / "keep_manifest_balanced.json",
        "reject": output_dir / "reject_manifest.json",
        "all": output_dir / "all_scored_records.json",
    }
    files["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files["keep"].write_text(json.dumps(keep_records_sorted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files["keep_balanced"].write_text(json.dumps(balanced_keep_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files["reject"].write_text(json.dumps(reject_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files["all"].write_text(json.dumps(records_all, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary": str(files["summary"]),
                "keep_manifest": str(files["keep"]),
                "keep_manifest_balanced": str(files["keep_balanced"]),
                "reject_manifest": str(files["reject"]),
                "keep_count": len(keep_records),
                "balanced_keep_count": len(balanced_keep_records),
                "reject_count": len(reject_records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
