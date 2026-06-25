from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..videophy2_auto import VideoPhy2Runner
from ..paths import VIDEOPHY2_CKPT, VIDEOPHY_ROOT
from .common import emit_result, load_eval_case, result_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-case VideoPhy-2 AutoEval.")
    parser.add_argument("--task", default="pc", choices=["sa", "pc", "rule"])
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
    return parser.parse_args()


def score_case(
    case: Path | str | dict[str, Any],
    *,
    task: str = "pc",
    caption: str | None = None,
    rule: str | None = None,
    runner: VideoPhy2Runner | None = None,
) -> dict[str, Any]:
    active_runner = runner or VideoPhy2Runner()
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
    result = score_case(case, task=args.task, caption=args.caption, rule=args.rule, runner=runner)
    emit_result(result_record(case, result), output_json=args.output_json)


if __name__ == "__main__":
    main()
