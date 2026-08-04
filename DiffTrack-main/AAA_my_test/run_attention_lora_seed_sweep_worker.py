#!/usr/bin/env python3
"""Seed-aware 49-frame Wan+LoRA attention intervention worker."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RANKING_CSV = Path(
    "/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case/"
    "three_model_combined_summary.csv"
)
NUM_STEPS = 40


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import worker: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--profile",
        choices=(
            "alpha090",
            "alpha150",
            "zero",
            "uniform",
            "temporal_causal",
            "head_output_zero",
        ),
        required=True,
    )
    parser.add_argument("--stage", choices=("all_steps", "steps00_09"), required=True)
    parser.add_argument("--input-json-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def seeded_generate(base, seed: int):
    torch = base.torch
    top5 = base.top5

    def generate(pipe, zeroer, context, prompt, group, steps):
        top5.source.probe.seed_everything(seed)
        zeroer.set_variant(group, steps)
        with torch.inference_mode():
            return pipe(
                prompt=prompt,
                negative_prompt="",
                input_image=context[0],
                context_video=context,
                height=512,
                width=896,
                num_frames=top5.source.target.core.align_generation_num_frames(48),
                seed=seed,
                cfg_scale=5.0,
                num_inference_steps=40,
                tiled=True,
            )

    return generate


def base_argv(args: argparse.Namespace) -> list[str]:
    return [
        str(Path(__file__)),
        "--model",
        "lora",
        "--input-json-list",
        str(args.input_json_list),
        "--output-root",
        str(args.output_root),
        "--shard-index",
        "0",
        "--num-shards",
        "1",
        "--ranking-pool",
        "all720",
        "--extreme-count",
        "100",
    ]


def run_attention(args: argparse.Namespace) -> None:
    is_alpha = args.profile.startswith("alpha")
    if is_alpha:
        filename = (
            "run_pck_step_adaptive_qk_probability_noise_49f_worker.py"
            if args.stage == "all_steps"
            else "run_pck_step_adaptive_qk_probability_noise_00_09_49f_worker.py"
        )
    else:
        filename = (
            "run_pck_step_adaptive_attention_replacement_49f_worker.py"
            if args.stage == "all_steps"
            else "run_pck_step_adaptive_attention_replacement_00_09_49f_worker.py"
        )
    stage = import_path("seed_sweep_attention_stage", HERE / filename)
    worker = stage.worker
    worker.original_generate = seeded_generate(worker.base, args.seed)
    worker.base.LEGACY_ROOTS = ()
    sys.argv = base_argv(args)
    if hasattr(worker, "load_capture_prompt_cases"):
        worker.load_capture_prompt_cases()
    worker.write_experiment_metadata()
    worker.base.main()


def adaptive_heads(_ranking_pool: str, extreme_count: int) -> dict[str, list[dict]]:
    rows_by_step: dict[int, list[dict]] = {step: [] for step in range(NUM_STEPS)}
    with RANKING_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("scope") != "objects":
                continue
            step = int(row["step"])
            if step in rows_by_step:
                rows_by_step[step].append(
                    {
                        "step": step,
                        "block": int(row["block"]),
                        "head": int(row["head"]),
                        "macro_pck32": float(row["macro_pck32"]),
                    }
                )
    groups: dict[str, list[dict]] = {}
    for step, rows in rows_by_step.items():
        ordered = sorted(
            rows,
            key=lambda row: (-row["macro_pck32"], row["block"], row["head"]),
        )
        if len({(row["block"], row["head"]) for row in ordered}) != 720:
            raise RuntimeError(f"Step {step} does not contain 720 unique heads")
        groups[f"top{extreme_count}_step_{step:02d}"] = ordered[:extreme_count]
        groups[f"bottom{extreme_count}_step_{step:02d}"] = list(
            reversed(ordered[-extreme_count:])
        )
    return groups


def run_head_output_zero(args: argparse.Namespace) -> None:
    adaptive = import_path(
        "seed_sweep_head_zero",
        HERE / "run_pck_step_adaptive_head_zero_ablation_worker.py",
    )
    base = adaptive.base
    parent_zeroer = adaptive.StepAdaptiveExtremeHeadZeroer.__mro__[1]

    def set_variant(self, group, steps):
        self.adaptive_prefix = group
        parent_zeroer.set_variant(self, group, steps)

    adaptive.StepAdaptiveExtremeHeadZeroer.set_variant = set_variant
    base.select_heads = adaptive_heads
    base.ExtremeHeadZeroer = adaptive.StepAdaptiveExtremeHeadZeroer
    if args.stage == "all_steps":
        base.STAGE_RANGES = (("steps_00_40", tuple(range(NUM_STEPS))),)
    else:
        base.STAGE_RANGES = (("steps_00_10", tuple(range(10))),)
    base.LEGACY_ROOTS = ()
    base.generate = seeded_generate(base, args.seed)
    sys.argv = base_argv(args)
    base.main()


def main() -> None:
    args = parse_args()
    if not 0 <= args.seed <= 100000:
        raise ValueError("seed must be in [0, 100000]")
    if args.profile == "head_output_zero":
        run_head_output_zero(args)
    else:
        run_attention(args)


if __name__ == "__main__":
    main()
