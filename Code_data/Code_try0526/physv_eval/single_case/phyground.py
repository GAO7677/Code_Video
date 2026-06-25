from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..phyground_official import OfficialPhyGroundRunner
from ..paths import PHYJUDGE_ADAPTER, PHYJUDGE_INFER
from .common import emit_result, load_eval_case, result_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-case PhyGround evaluation.")
    parser.add_argument("--input-json", type=Path, default=None, help="Case JSON containing video metadata.")
    parser.add_argument("--video", type=Path, default=None, help="Video path for the single case.")
    parser.add_argument("--caption", default=None, help="Caption for the video.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--general-only", action="store_true")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-pixels", type=int, default=360 * 640)
    return parser.parse_args()


def score_case(
    case: Path | str | dict[str, Any],
    *,
    caption: str | None = None,
    metrics: list[str] | None = None,
    laws: list[str] | None = None,
    criteria_overrides: dict[str, str] | None = None,
    runner: OfficialPhyGroundRunner | None = None,
) -> dict[str, Any]:
    active_runner = runner or OfficialPhyGroundRunner()
    return active_runner.score_case(
        case,
        caption=caption,
        metrics=metrics,
        laws=laws,
        criteria_overrides=criteria_overrides,
    )


def main() -> None:
    args = parse_args()
    case = load_eval_case(input_json=args.input_json, video=args.video, caption=args.caption)
    runner = OfficialPhyGroundRunner(
        adapter_dir=PHYJUDGE_ADAPTER,
        infer_script=PHYJUDGE_INFER,
        cuda_visible_devices=args.cuda_visible_devices,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        fps=args.fps,
        max_pixels=args.max_pixels,
    )
    metrics = None
    laws = [] if args.general_only else None
    result = score_case(case, caption=args.caption, metrics=metrics, laws=laws, runner=runner)
    emit_result(result_record(case, result), output_json=args.output_json)


if __name__ == "__main__":
    main()
