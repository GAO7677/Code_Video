"""xSSC-matched PhysRVG batch inference with one runtime DiT ablation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from accelerate.utils import set_seed

from physrvg_ablation import (
    ABLATION_MODES,
    PhysRVGAblationSpec,
    get_ablation_call_count,
    install_physrvg_ablation,
)


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
) -> tuple[PhysRVGAblationSpec, int, Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--physrvg-ablation-mode",
        choices=ABLATION_MODES,
        default="baseline",
    )
    parser.add_argument("--physrvg-ablation-block", type=int, default=None)
    parser.add_argument("--physrvg-ablation-head", type=int, default=None)
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
    if args.expected_context_frames != MATCHED_XSSC_CONFIG["context_frames"]:
        raise ValueError(
            "--expected-context-frames must remain "
            f"{MATCHED_XSSC_CONFIG['context_frames']}"
        )
    return spec, int(args.expected_context_frames), args.physrvg_root, remaining


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
) -> None:
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
    if actual != MATCHED_XSSC_CONFIG:
        raise ValueError(
            "PhysRVG xSSC-matched inference configuration was changed: "
            f"expected {MATCHED_XSSC_CONFIG}, got {actual}"
        )


def _annotate_top_level_jsons(
    output_root: Path,
    metadata: dict[str, object],
) -> None:
    for path in sorted(output_root.rglob("*.json")):
        if path.name not in {"batch_manifest.json", "summary.json", "result.json"}:
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            continue
        payload["physrvg_ablation"] = metadata
        payload["inference_config"] = MATCHED_XSSC_CONFIG
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
    spec, expected_context_frames, physrvg_root, remaining = _extract_ablation_args(
        sys.argv[1:]
    )
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
        _validate_matched_config(args, expected_context_frames)
        state["output_root"] = args.output_root.expanduser().resolve()
        return args

    def load_pipe_with_ablation(args: argparse.Namespace):
        # Official order: base Wan -> full PhysRVG DiT -> PhysRVG LoRA -> device.
        pipe = original_load_pipe(args)
        metadata = install_physrvg_ablation(pipe.transformer, spec)
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
                None if spec.mode == "baseline" else int(args.num_inference_steps)
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
                    "inference_config": MATCHED_XSSC_CONFIG,
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
    if isinstance(output_root, Path) and isinstance(metadata, dict):
        _annotate_top_level_jsons(output_root, metadata)
        failure_count = _completed_failure_count(output_root)
        if failure_count:
            raise SystemExit(
                f"PhysRVG batch completed with {failure_count} failed case(s)"
            )


if __name__ == "__main__":
    main()
