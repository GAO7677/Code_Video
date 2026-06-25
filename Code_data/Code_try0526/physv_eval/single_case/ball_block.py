from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ..case_inputs import coerce_eval_case
from ..official_pdi import run_single_case as run_pdi_single_case
from .common import emit_result, load_eval_case, result_record

DATA_DIR = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp")
TMP_DIR = DATA_DIR / "tmp_eval"


def run_pdi(video_path: Path, *, caption: str = "ball") -> dict[str, Any] | None:
    try:
        return run_pdi_single_case(video_path, text_query=caption)
    except Exception:
        return None


def run_jepa(video_path: Path) -> dict[str, Any] | None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rerank_video.scorers import JEPAPredictiveScorer
    from rerank_video.schemas import JEPAScoreConfig
    from rerank_video.video_utils import ensure_dir, load_video_frames, uniform_subsample_frames
    import cv2

    frames = load_video_frames(video_path)
    total = len(frames)
    if total < 30:
        return None
    split = min(60, total // 2)
    ctx = uniform_subsample_frames(frames[:split], 8)
    fut = uniform_subsample_frames(frames[split:], 16)

    tmp = ensure_dir(TMP_DIR / "jepa" / video_path.stem)
    ctx_p = tmp / "context.mp4"
    fut_p = tmp / "future.mp4"

    def write_video(path: Path, frs: list[Any]) -> None:
        height, width = frs[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 16, (width, height))
        try:
            for frame in frs:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    write_video(ctx_p, ctx)
    write_video(fut_p, fut)

    scorer = JEPAPredictiveScorer(
        JEPAScoreConfig(
            backend="vjepa2",
            device="cuda",
            max_frames=32,
            context_frames=8,
            future_frames=16,
            context_repeat_frames=8,
            crop_size=384,
            vjepa_checkpoint=Path("/data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt"),
            vjepa_repo_root=Path("/home/gaoya/Code_Video/vjepa2-main"),
            vjepa_model_name="vjepa2_1_vit_large_384",
        )
    )
    score, _details = scorer.score(context_video_path=ctx_p, candidate_video_path=fut_p)
    return {"jepa_score": float(score)}


def score_case(
    case: Path | str | dict[str, Any],
    *,
    caption: str = "ball",
    skip_pdi: bool = False,
    skip_jepa: bool = False,
) -> dict[str, Any]:
    normalized = coerce_eval_case(case, caption=caption)
    result: dict[str, Any] = {"caption": normalized.caption or caption}
    if not skip_pdi:
        result["pdi"] = run_pdi(normalized.video_path, caption=normalized.caption or caption)
    if not skip_jepa:
        result["jepa"] = run_jepa(normalized.video_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-case ball-block evaluation.")
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--caption", default="ball")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--skip-pdi", action="store_true")
    parser.add_argument("--skip-jepa", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalized = load_eval_case(input_json=args.input_json, video=args.video, caption=args.caption)
    result = score_case(
        normalized,
        caption=args.caption,
        skip_pdi=args.skip_pdi,
        skip_jepa=args.skip_jepa,
    )
    emit_result(result_record(normalized, result), output_json=args.output_json)


if __name__ == "__main__":
    main()
