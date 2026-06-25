from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..cosmos_reason1_official import OfficialCosmosReason1Runner
from ..paths import COSMOS_REASON1_MODEL, COSMOS_REASON1_ROOT
from .common import emit_result, load_eval_case, result_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-case Cosmos-Reason1 evaluation.")
    parser.add_argument("--input-json", type=Path, default=None, help="Case JSON containing video metadata.")
    parser.add_argument("--video", type=Path, default=None, help="Video path for the single case.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--prompt-path", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--total-pixels", type=int, default=8192 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def score_case(
    case: Path | str | dict[str, Any],
    *,
    runner: OfficialCosmosReason1Runner | None = None,
) -> dict[str, Any]:
    active_runner = runner or OfficialCosmosReason1Runner()
    return active_runner.score_case(case)


def main() -> None:
    args = parse_args()
    case = load_eval_case(input_json=args.input_json, video=args.video)
    runner = OfficialCosmosReason1Runner(
        model_path=args.model_path or COSMOS_REASON1_MODEL,
        prompt_path=args.prompt_path
        or (
            COSMOS_REASON1_ROOT.parent
            / "cosmos-cookbook"
            / "docs"
            / "recipes"
            / "post_training"
            / "reason1"
            / "physical-plausibility-check"
            / "assets"
            / "video_reward.yaml"
        ),
        fps=args.fps,
        total_pixels=args.total_pixels,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    result = score_case(case, runner=runner)
    emit_result(result_record(case, result), output_json=args.output_json)


if __name__ == "__main__":
    main()
