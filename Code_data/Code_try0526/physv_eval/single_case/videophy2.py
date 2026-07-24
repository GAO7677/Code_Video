from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from ..case_inputs import EvalCase, coerce_eval_case
from ..videophy2_auto import VideoPhy2Runner
from ..paths import VIDEOPHY2_CKPT, VIDEOPHY_ROOT
from .common import emit_result, load_eval_case, result_record
from .physics_iq import (
    _clip_frames,
    _ensure_video,
    _read_video,
    _resolve_context_frames,
    _write_video,
)


GENERATED_ONLY_TASK = "generated_only_sa_pc_joint"
TASK_CHOICES = ("sa", "pc", "rule", GENERATED_ONLY_TASK)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-case VideoPhy-2 AutoEval.")
    parser.add_argument("--task", default=GENERATED_ONLY_TASK, choices=TASK_CHOICES)
    parser.add_argument("--input-json", type=Path, default=None, help="Case JSON containing video metadata.")
    parser.add_argument("--video", type=Path, default=None, help="Video path for the single case.")
    parser.add_argument("--caption", default=None, help="Caption for SA task.")
    parser.add_argument("--rule", default=None, help="Rule text for rule task.")
    parser.add_argument("--context-video", type=Path, default=None, help="Optional context video path.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--context-frames", type=int, default=None)
    return parser.parse_args()


def _existing_pc_raw_result(case: EvalCase) -> dict[str, Any] | None:
    metadata = case.metadata or {}
    existing = metadata.get("videophy2")
    if not isinstance(existing, dict):
        return None
    nested = existing.get("pc_raw")
    if isinstance(nested, dict) and isinstance(nested.get("score"), (int, float)):
        return dict(nested)
    if existing.get("task") == "pc" and isinstance(existing.get("score"), (int, float)):
        return dict(existing)
    return None


def score_generated_only_case(
    case: EvalCase | Path | str | dict[str, Any],
    *,
    caption: str | None = None,
    context_frames: int | None = None,
    runner: VideoPhy2Runner | None = None,
) -> dict[str, Any]:
    active_runner = runner or VideoPhy2Runner()
    normalized = coerce_eval_case(case, caption=caption)
    resolved_caption = normalized.caption
    if not resolved_caption:
        raise ValueError("Generated-only VideoPhy2 SA requires a caption")

    resolved_context_frames = _resolve_context_frames(
        normalized,
        context_frames,
        "without_context",
    )
    _ensure_video(normalized.video_path)
    all_frames, fps = _read_video(normalized.video_path)
    generated_frames = _clip_frames(
        all_frames,
        resolved_context_frames,
        label="generated_video",
    )

    pc_raw = _existing_pc_raw_result(normalized)
    if pc_raw is None:
        pc_raw = active_runner.score_video(normalized.video_path, task="pc")

    temp_root = Path("/tmp/gaoya/videophy2_generated_only")
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="case_", dir=temp_root) as temp_dir:
        generated_only_path = Path(temp_dir) / "generated_only.mp4"
        _write_video(generated_frames, generated_only_path, fps)
        sa_result = active_runner.score_video(
            generated_only_path,
            task="sa",
            caption=resolved_caption,
        )
        pc_result = active_runner.score_video(generated_only_path, task="pc")

    sa_score = int(sa_result["score"])
    pc_score = int(pc_result["score"])
    joint_pass = int(sa_score >= 4 and pc_score >= 4)
    return {
        "task": GENERATED_ONLY_TASK,
        "score": joint_pass,
        "sa_score": sa_score,
        "pc_score": pc_score,
        "joint_pass": joint_pass,
        "joint_rate": float(joint_pass),
        "pc_raw_score": int(pc_raw["score"]),
        "sa_pass": int(sa_score >= 4),
        "pc_pass": int(pc_score >= 4),
        "context_frames_removed": int(resolved_context_frames),
        "input_frames": int(len(all_frames)),
        "generated_only_frames": int(len(generated_frames)),
        "input_fps": float(fps),
        "caption": resolved_caption,
        "generated_only": {
            "sa": sa_result,
            "pc": pc_result,
        },
        "pc_raw": pc_raw,
        "joint_definition": "SA>=4 and PC>=4",
        "score_semantics": "per_case_joint_pass; dataset mean is joint rate",
    }


def score_case(
    case: EvalCase | Path | str | dict[str, Any],
    *,
    task: str = GENERATED_ONLY_TASK,
    caption: str | None = None,
    rule: str | None = None,
    context_frames: int | None = None,
    runner: VideoPhy2Runner | None = None,
) -> dict[str, Any]:
    active_runner = runner or VideoPhy2Runner()
    if task == GENERATED_ONLY_TASK:
        return score_generated_only_case(
            case,
            caption=caption,
            context_frames=context_frames,
            runner=active_runner,
        )
    return active_runner.score_case(case, task=task, caption=caption, rule=rule)


def main() -> None:
    args = parse_args()
    case = load_eval_case(
        input_json=args.input_json,
        video=args.video,
        caption=args.caption,
        rule=args.rule,
        context_video=args.context_video,
    )
    runner = VideoPhy2Runner(
        checkpoint=args.checkpoint or VIDEOPHY2_CKPT,
        repo_root=args.repo_root or VIDEOPHY_ROOT,
        device=args.device,
        dtype=args.dtype,
        num_frames=args.num_frames,
    )
    result = score_case(
        case,
        task=args.task,
        caption=args.caption,
        rule=args.rule,
        context_frames=args.context_frames,
        runner=runner,
    )
    emit_result(result_record(case, result), output_json=args.output_json)


if __name__ == "__main__":
    main()
