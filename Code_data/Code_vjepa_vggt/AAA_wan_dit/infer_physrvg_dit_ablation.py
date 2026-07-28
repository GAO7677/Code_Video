"""xSSC-matched PhysRVG batch inference with one runtime DiT ablation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from accelerate.utils import set_seed

from common22_public_head_targets import (
    ROLE_CHOICES as PUBLIC_HEAD_ROLES,
    targets_for_role as public_targets_for_role,
)
from score_extreme_head_targets import GROUPS, targets_for_score_group
from matched_head_subset_targets import load_matched_subset
from physrvg_ablation import (
    ABLATION_MODES,
    PhysRVGAblationSpec,
    get_ablation_call_count,
    install_grouped_physrvg_head_ablation,
    install_physrvg_ablation,
)
from grouped_head_targets import CATEGORY_TARGETS, targets_for_category


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PHYSRVG_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/"
    "code_phys_papers_compare/PhysRVG-main"
)
MATCHED_XSSC_NEGATIVE_PROMPT = (
    "模糊，低质量，变形，伪影，文字，水印，过曝，欠曝，颜色异常，几何扭曲，"
    "物体融化，物理不合理"
)
MATCHED_XSSC_CONFIG = {
    "height": 512,
    "width": 896,
    "num_frames": 49,
    "context_frames": 8,
    "num_inference_steps": 40,
    "guidance_scale": 5.0,
    "do_cfg": False,
    "fps": 30,
    "seed": 42,
    "negative_prompt": MATCHED_XSSC_NEGATIVE_PROMPT,
}


def _extract_ablation_args(
    argv: list[str],
) -> tuple[
    PhysRVGAblationSpec,
    str | None,
    list[tuple[int, int]],
    dict[str, object] | None,
    tuple[int, int] | None,
    int,
    bool,
    Path,
    list[str],
]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--physrvg-ablation-mode",
        choices=ABLATION_MODES,
        default="baseline",
    )
    parser.add_argument("--physrvg-ablation-block", type=int, default=None)
    parser.add_argument("--physrvg-ablation-head", type=int, default=None)
    parser.add_argument(
        "--physrvg-grouped-head-category",
        choices=tuple(CATEGORY_TARGETS),
        default=None,
    )
    parser.add_argument("--physrvg-public-head-report", type=Path)
    parser.add_argument(
        "--physrvg-public-head-role",
        choices=PUBLIC_HEAD_ROLES,
    )
    parser.add_argument("--physrvg-score-extreme-selection", type=Path)
    parser.add_argument("--physrvg-score-extreme-group", choices=GROUPS)
    parser.add_argument("--physrvg-matched-subset-manifest", type=Path)
    parser.add_argument("--physrvg-matched-subset-id")
    parser.add_argument("--physrvg-ablation-step-start", type=int)
    parser.add_argument("--physrvg-ablation-step-end", type=int)
    parser.add_argument("--expected-context-frames", type=int, default=8)
    parser.add_argument(
        "--physrvg-root",
        type=Path,
        default=Path(os.environ.get("PHYSRVG_ROOT", DEFAULT_PHYSRVG_ROOT)),
    )
    args, remaining = parser.parse_known_args(argv)
    spec = PhysRVGAblationSpec(
        mode=str(args.physrvg_ablation_mode),
        block_id=args.physrvg_ablation_block,
        head_id=args.physrvg_ablation_head,
    )
    spec.validate(30)
    grouped_category = args.physrvg_grouped_head_category
    using_public_report = args.physrvg_public_head_report is not None
    using_public_role = args.physrvg_public_head_role is not None
    using_score_selection = args.physrvg_score_extreme_selection is not None
    using_score_group = args.physrvg_score_extreme_group is not None
    using_matched_manifest = args.physrvg_matched_subset_manifest is not None
    using_matched_id = args.physrvg_matched_subset_id is not None
    if using_public_report != using_public_role:
        raise ValueError(
            "--physrvg-public-head-report and --physrvg-public-head-role "
            "must be specified together"
        )
    if using_score_selection != using_score_group:
        raise ValueError(
            "--physrvg-score-extreme-selection and "
            "--physrvg-score-extreme-group must be specified together"
        )
    if using_matched_manifest != using_matched_id:
        raise ValueError(
            "--physrvg-matched-subset-manifest and "
            "--physrvg-matched-subset-id must be specified together"
        )
    if sum(
        (
            grouped_category is not None,
            using_public_report,
            using_score_selection,
            using_matched_manifest,
        )
    ) > 1:
        raise ValueError(
            "Legacy grouped category, common22 role, score extreme, and "
            "matched subset are mutually exclusive"
        )
    if grouped_category is not None and spec.mode != "baseline":
        raise ValueError(
            "Grouped Head category cannot be combined with a standard ablation"
        )
    if (
        using_public_report
        or using_score_selection
        or using_matched_manifest
    ) and spec.mode != "baseline":
        raise ValueError(
            "Grouped target selection cannot be combined with a standard ablation"
        )
    target_source: dict[str, object] | None = None
    if using_public_report:
        grouped_category = str(args.physrvg_public_head_role)
        grouped_targets, target_source = public_targets_for_role(
            args.physrvg_public_head_report,
            grouped_category,
        )
        target_source = {
            "kind": "common22_cross_model_public_stable_role",
            **target_source,
        }
    elif using_score_selection:
        grouped_category, grouped_targets, target_source = targets_for_score_group(
            args.physrvg_score_extreme_selection,
            args.physrvg_score_extreme_group,
        )
    elif using_matched_manifest:
        grouped_category, grouped_targets, target_source = load_matched_subset(
            args.physrvg_matched_subset_manifest,
            args.physrvg_matched_subset_id,
        )
    else:
        grouped_targets = (
            []
            if grouped_category is None
            else targets_for_category(str(grouped_category))
        )
    if (
        args.physrvg_ablation_step_start is None
    ) != (args.physrvg_ablation_step_end is None):
        raise ValueError(
            "--physrvg-ablation-step-start and "
            "--physrvg-ablation-step-end must be paired"
        )
    step_range = (
        None
        if args.physrvg_ablation_step_start is None
        else (
            int(args.physrvg_ablation_step_start),
            int(args.physrvg_ablation_step_end),
        )
    )
    if step_range is not None and grouped_category is None:
        raise ValueError("Denoise-step gating requires a grouped Head ablation")
    if args.expected_context_frames != MATCHED_XSSC_CONFIG["context_frames"]:
        raise ValueError(
            "--expected-context-frames must remain "
            f"{MATCHED_XSSC_CONFIG['context_frames']}"
        )
    return (
        spec,
        grouped_category,
        grouped_targets,
        target_source,
        step_range,
        int(args.expected_context_frames),
        using_public_report or using_score_selection or using_matched_manifest,
        args.physrvg_root,
        remaining,
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _count_video_frames(base, video_path: Path) -> int:
    with base._open_video_reader(video_path) as reader:
        return int(base._safe_video_length(reader))


def _validate_matched_config(
    args: argparse.Namespace,
    expected_context_frames: int,
    *,
    allow_arbitrary_seed: bool,
) -> dict[str, object]:
    actual = {
        "height": int(args.height),
        "width": int(args.width),
        "num_frames": int(args.num_frames),
        "context_frames": int(expected_context_frames),
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "do_cfg": False,
        "fps": int(args.fps),
        "seed": int(args.seed),
        "negative_prompt": MATCHED_XSSC_NEGATIVE_PROMPT,
    }
    expected = dict(MATCHED_XSSC_CONFIG)
    if allow_arbitrary_seed:
        expected["seed"] = int(args.seed)
    if actual != expected:
        raise ValueError(
            "PhysRVG xSSC-matched inference configuration was changed: "
            f"expected {expected}, got {actual}"
        )
    return actual


def _annotate_top_level_jsons(
    output_root: Path,
    metadata: dict[str, object],
    inference_config: dict[str, object],
) -> None:
    for path in sorted(output_root.rglob("*.json")):
        if path.name not in {"batch_manifest.json", "summary.json", "result.json"}:
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            continue
        payload["physrvg_ablation"] = metadata
        payload["inference_config"] = inference_config
        payload["config_policy"] = "matched_to_previous_xssc_except_cfg"
        _atomic_write_json(path, payload)


def _completed_failure_count(output_root: Path) -> int:
    failures = 0
    for path in output_root.rglob("summary.json"):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        failures += int(payload.get("num_failed", 0))
    return failures


def main() -> None:
    (
        spec,
        grouped_category,
        grouped_targets,
        target_source,
        step_range,
        expected_context_frames,
        allow_arbitrary_seed,
        physrvg_root,
        remaining,
    ) = _extract_ablation_args(sys.argv[1:])
    physrvg_root = physrvg_root.expanduser().resolve()
    if not physrvg_root.is_dir():
        raise FileNotFoundError(f"PhysRVG root not found: {physrvg_root}")
    sys.path.insert(0, str(physrvg_root))

    import batch_infer_from_input_json_lists as base

    original_parse_args = base.parse_args
    original_load_pipe = base._load_pipe
    original_run_single_case = base._run_single_case
    state: dict[str, object] = {}

    def parse_args_matched() -> argparse.Namespace:
        args = original_parse_args()
        state["inference_config"] = _validate_matched_config(
            args,
            expected_context_frames,
            allow_arbitrary_seed=allow_arbitrary_seed,
        )
        state["output_root"] = args.output_root.expanduser().resolve()
        return args

    def load_pipe_with_ablation(args: argparse.Namespace):
        # Official order: base Wan -> full PhysRVG DiT -> PhysRVG LoRA -> device.
        pipe = original_load_pipe(args)
        if grouped_category is None:
            metadata = install_physrvg_ablation(pipe.transformer, spec)
        else:
            if step_range is not None and not (
                0 <= step_range[0] < step_range[1] <= int(args.num_inference_steps)
            ):
                raise ValueError(
                    f"Invalid ablation step range {step_range} for "
                    f"{args.num_inference_steps} steps"
                )
            category = (
                str(grouped_category)
                if step_range is None
                else (
                    f"{grouped_category}_STEPS"
                    f"{step_range[0]:02d}_{step_range[1]:02d}"
                )
            )
            metadata = install_grouped_physrvg_head_ablation(
                pipe.transformer,
                category=category,
                targets=grouped_targets,
                active_step_range=step_range,
                total_steps=int(args.num_inference_steps),
                calls_per_step=1,
            )
            if target_source is not None:
                metadata["target_selection"] = target_source
        state["metadata"] = metadata
        print(
            f"[physrvg_ablation] {json.dumps(metadata, sort_keys=True)}",
            flush=True,
        )
        return pipe

    def run_single_case_with_ablation(**kwargs):
        args = kwargs["args"]
        payload = kwargs["payload"]
        input_json_path = kwargs["input_json_path"]
        dataset_name = kwargs["dataset_name"]
        summary_entries = kwargs["summary_entries"]
        pipe = kwargs["pipe"]

        input_video = Path(
            base._ensure_str_field(payload, "input_video", input_json_path)
        ).expanduser().resolve()
        context_frames = _count_video_frames(base, input_video)
        if context_frames != expected_context_frames:
            raise ValueError(
                f"Expected exactly {expected_context_frames} context frames, "
                f"found {context_frames}: {input_video}"
            )

        # The official pipeline currently ignores its generator in randn().
        # Resetting global RNG here makes paired ablations use identical noise.
        set_seed(int(args.seed))
        before = get_ablation_call_count(pipe.transformer)
        did_run, message = original_run_single_case(**kwargs)
        after = get_ablation_call_count(pipe.transformer)

        method_name = base._method_name(
            int(args.num_inference_steps),
            int(args.height),
            int(args.width),
            context_frames,
            int(args.num_frames),
        )
        output_json = (
            args.output_root
            / dataset_name
            / method_name
            / f"{input_json_path.stem}.json"
        )

        if not did_run and output_json.is_file():
            with output_json.open("r", encoding="utf-8") as handle:
                existing_result = json.load(handle)
            summary_entries[method_name].append(existing_result)

        if did_run:
            observed_calls = (
                None if before is None or after is None else int(after - before)
            )
            expected_calls = (
                (
                    len(grouped_targets)
                    * (
                        int(args.num_inference_steps)
                        if step_range is None
                        else step_range[1] - step_range[0]
                    )
                    if grouped_category is not None
                    else None
                    if spec.mode == "baseline"
                    else int(args.num_inference_steps)
                )
            )
            call_count_ok = (
                None
                if expected_calls is None
                else observed_calls == expected_calls
            )
            case_metadata = dict(state["metadata"])
            case_metadata.update(
                {
                    "observed_target_forward_calls": observed_calls,
                    "expected_target_forward_calls": expected_calls,
                    "target_forward_call_count_ok": call_count_ok,
                }
            )
            with output_json.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
            result.update(
                {
                    "physrvg_ablation": case_metadata,
                    "inference_config": state["inference_config"],
                    "config_policy": "matched_to_previous_xssc_except_cfg",
                    "context_policy": "all_8_json_input_video_frames",
                    "prompt_policy": "input_caption_from_physiciq_json",
                    "do_cfg": False,
                    "negative_prompt": MATCHED_XSSC_NEGATIVE_PROMPT,
                    "negative_prompt_effective_in_denoising": False,
                }
            )
            _atomic_write_json(output_json, result)
            if summary_entries[method_name]:
                summary_entries[method_name][-1].update(result)
            if call_count_ok is False:
                raise SystemExit(
                    f"Target module call count mismatch: expected "
                    f"{expected_calls}, observed {observed_calls}"
                )
        return did_run, message

    base.DEFAULT_NEGATIVE_PROMPT = MATCHED_XSSC_NEGATIVE_PROMPT
    base.parse_args = parse_args_matched
    base._load_pipe = load_pipe_with_ablation
    base._run_single_case = run_single_case_with_ablation
    sys.argv = [sys.argv[0], *remaining]
    base.main()

    output_root = state.get("output_root")
    metadata = state.get("metadata")
    inference_config = state.get("inference_config")
    if (
        isinstance(output_root, Path)
        and isinstance(metadata, dict)
        and isinstance(inference_config, dict)
    ):
        _annotate_top_level_jsons(output_root, metadata, inference_config)
        failure_count = _completed_failure_count(output_root)
        if failure_count:
            raise SystemExit(
                f"PhysRVG batch completed with {failure_count} failed case(s)"
            )


if __name__ == "__main__":
    main()
