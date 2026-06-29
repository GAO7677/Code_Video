from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .single_case.physics_iq import score_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch single-view approximate Physics-IQ scoring against a fixed GT/source video."
    )
    parser.add_argument(
        "--gt-video",
        type=Path,
        required=True,
        help="GT/reference video used for all candidate comparisons.",
    )
    parser.add_argument(
        "--video-list-file",
        type=Path,
        required=True,
        help="Text file containing one candidate video path per line.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory used to save per-case aligned videos, JSON records, and summary outputs.",
    )
    parser.add_argument(
        "--threshold-value",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=4,
    )
    return parser.parse_args()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return slug or "case"


def _case_name(video_path: Path) -> str:
    rel = str(video_path).lstrip("/")
    return _slugify(rel.replace("/", "__"))


def _load_video_list(path: Path) -> list[Path]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    videos = [Path(line).resolve() for line in lines if line and not line.startswith("#")]
    return videos


def main() -> None:
    args = parse_args()
    gt_video = args.gt_video.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    videos = _load_video_list(args.video_list_file)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, video_path in enumerate(videos, start=1):
        if not video_path.is_file():
            failures.append({"video": str(video_path), "error": "candidate video not found"})
            continue
        case_name = _case_name(video_path)
        case_dir = output_root / case_name
        print(f"[{index}/{len(videos)}] scoring {video_path}")
        try:
            result = score_case(
                str(video_path),
                source_video_path=str(gt_video),
                threshold_value=args.threshold_value,
                downsample_factor=args.downsample_factor,
                aligned_video_dir=case_dir,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"video": str(video_path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"  failed: {exc}")
            continue

        record = {
            "case_name": case_name,
            "rank_hint_score": result["score"],
            "candidate_video": str(video_path),
            "gt_video": str(gt_video),
            **result,
        }
        (case_dir / "result.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        results.append(record)

    results.sort(key=lambda item: item.get("score", -1), reverse=True)

    summary = {
        "gt_video": str(gt_video),
        "video_list_file": str(args.video_list_file.resolve()),
        "output_root": str(output_root),
        "num_candidates": len(videos),
        "num_scored": len(results),
        "num_failed": len(failures),
        "threshold_value": args.threshold_value,
        "downsample_factor": args.downsample_factor,
        "results": results,
        "failures": failures,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "summary_table.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote summary: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
