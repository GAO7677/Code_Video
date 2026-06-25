from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..official_pdi import OfficialPDIRunner
from .common import emit_result, load_eval_case, result_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-case official PDI evaluation.")
    parser.add_argument("--input-json", type=Path, default=None, help="Case JSON containing video metadata.")
    parser.add_argument("--video", type=Path, default=None, help="Video path for the single case.")
    parser.add_argument("--caption", default=None, help="Caption or text query used by PDI.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    return parser.parse_args()


def score_case(
    case: Path | str | dict[str, Any],
    *,
    text_query: str | None = None,
    refresh: bool = False,
    runner: OfficialPDIRunner | None = None,
) -> dict[str, Any]:
    active_runner = runner or OfficialPDIRunner()
    return active_runner.run_case(case, text_query=text_query, refresh=refresh)


def main() -> None:
    args = parse_args()
    case = load_eval_case(input_json=args.input_json, video=args.video, caption=args.caption)
    runner = OfficialPDIRunner(
        python_bin=args.python_bin,
        cuda_visible_devices=args.cuda_visible_devices,
    )
    result = score_case(case, text_query=args.caption, refresh=args.refresh, runner=runner)
    emit_result(result_record(case, result), output_json=args.output_json)


if __name__ == "__main__":
    main()
