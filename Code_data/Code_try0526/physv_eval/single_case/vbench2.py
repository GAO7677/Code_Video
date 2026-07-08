from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..vbench2_official import OfficialVBench2Runner
from .common import emit_result, load_eval_case, result_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-case VBench-2.0 evaluation.")
    parser.add_argument("--dimension", required=True, help="Single VBench-2.0 dimension to evaluate.")
    parser.add_argument("--input-json", type=Path, default=None, help="Case JSON containing video metadata.")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Video path for the single case. For Diversity, pass the folder containing prompt-index.mp4 files.",
    )
    parser.add_argument("--caption", default=None, help="Optional prompt override for custom_input mode.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--output-path", type=Path, default=None, help="Optional directory for official VBench-2.0 outputs.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--full-json-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-ckpt-from-local", action="store_true")
    parser.add_argument("--read-frame", action="store_true")
    return parser.parse_args()


def score_case(
    case: Path | str | dict[str, Any],
    *,
    dimension: str,
    caption: str | None = None,
    output_path: Path | None = None,
    runner: OfficialVBench2Runner | None = None,
) -> dict[str, Any]:
    active_runner = runner or OfficialVBench2Runner()
    return active_runner.score_case(
        case,
        dimension=dimension,
        caption=caption,
        output_path=output_path,
    )


def main() -> None:
    args = parse_args()
    case = load_eval_case(input_json=args.input_json, video=args.video, caption=args.caption)
    runner = OfficialVBench2Runner(
        repo_root=args.repo_root,
        full_json_dir=args.full_json_dir,
        device=args.device,
        load_ckpt_from_local=args.load_ckpt_from_local,
        read_frame=args.read_frame,
    )
    result = score_case(
        case,
        dimension=args.dimension,
        caption=args.caption,
        output_path=args.output_path,
        runner=runner,
    )
    emit_result(result_record(case, result), output_json=args.output_json)


if __name__ == "__main__":
    main()
