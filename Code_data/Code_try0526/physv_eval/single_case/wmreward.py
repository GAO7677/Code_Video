from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..wmreward_official import WMRewardRunner
from .common import emit_result, load_eval_case, result_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-case WMReward evaluation.")
    parser.add_argument("--input-json", type=Path, default=None, help="Case JSON containing video metadata.")
    parser.add_argument("--video", type=Path, default=None, help="Video path for the single case.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--cuda-visible-devices", default=None)
    return parser.parse_args()


def score_case(
    case: Path | str | dict[str, Any],
    *,
    runner: WMRewardRunner | None = None,
) -> dict[str, Any]:
    active_runner = runner or WMRewardRunner()
    return active_runner.score_case(case)


def main() -> None:
    args = parse_args()
    case = load_eval_case(input_json=args.input_json, video=args.video)
    runner = WMRewardRunner(cuda_visible_devices=args.cuda_visible_devices)
    result = score_case(case, runner=runner)
    emit_result(result_record(case, result), output_json=args.output_json)


if __name__ == "__main__":
    main()
