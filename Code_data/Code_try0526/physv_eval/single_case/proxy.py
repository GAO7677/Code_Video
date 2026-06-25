from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..proxy_runner import ProxyRunner
from .common import emit_result, load_eval_case, result_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-case proxy evaluation.")
    parser.add_argument("--input-json", type=Path, default=None, help="Case JSON containing video metadata.")
    parser.add_argument("--video", type=Path, default=None, help="Candidate video path.")
    parser.add_argument("--context-video", type=Path, default=None, help="Optional context video path.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def score_case(
    case: Path | str | dict[str, Any],
    *,
    context_video_path: Path | str | None = None,
    runner: ProxyRunner | None = None,
) -> dict[str, Any] | None:
    active_runner = runner or ProxyRunner()
    return active_runner.score_case(case, context_video_path=context_video_path)


def main() -> None:
    args = parse_args()
    case = load_eval_case(
        input_json=args.input_json,
        video=args.video,
        context_video=args.context_video,
    )
    runner = ProxyRunner(device=args.device)
    result = score_case(case, context_video_path=args.context_video, runner=runner)
    if result is None:
        raise RuntimeError("Proxy scoring failed for this case")
    emit_result(result_record(case, result), output_json=args.output_json)


if __name__ == "__main__":
    main()
