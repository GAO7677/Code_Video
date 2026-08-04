#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


BASE_WORKER = Path(__file__).with_name("run_pck_extreme_head_zero_ablation_worker.py")
RANKING_CSV = Path(
    "/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case/"
    "three_model_combined_summary.csv"
)
NUM_DENOISING_STEPS = 40

spec = importlib.util.spec_from_file_location("pck_extreme_base_worker", BASE_WORKER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base worker: {BASE_WORKER}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def step_adaptive_heads(ranking_pool: str, extreme_count: int) -> dict[str, list[dict]]:
    if ranking_pool != "all720":
        raise ValueError("Step-adaptive ablation requires --ranking-pool all720")
    if extreme_count != 30:
        raise ValueError("This experiment requires --extreme-count 30")

    rows_by_step: dict[int, list[dict]] = {step: [] for step in range(NUM_DENOISING_STEPS)}
    with RANKING_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("scope") != "objects":
                continue
            step = int(row["step"])
            if step not in rows_by_step:
                continue
            rows_by_step[step].append(
                {
                    "step": step,
                    "block": int(row["block"]),
                    "head": int(row["head"]),
                    "macro_pck32": float(row["macro_pck32"]),
                    "timestep": float(row["timestep"]),
                    "sigma": float(row["sigma"]),
                    "models": int(row["models"]),
                    "valid_cases": int(row["valid_cases"]),
                }
            )

    groups: dict[str, list[dict]] = {"top30": [], "bottom30": []}
    for step in range(NUM_DENOISING_STEPS):
        rows = rows_by_step[step]
        unique = {(row["block"], row["head"]) for row in rows}
        if len(rows) != 720 or len(unique) != 720:
            raise RuntimeError(
                f"Expected 720 unique object heads at step {step}, "
                f"got rows={len(rows)} unique={len(unique)}"
            )
        descending = sorted(
            rows,
            key=lambda row: (-row["macro_pck32"], row["block"], row["head"]),
        )
        top = [dict(row, rank_within_step=index + 1) for index, row in enumerate(descending[:30])]
        bottom_rows = sorted(
            descending[-30:],
            key=lambda row: (row["macro_pck32"], row["block"], row["head"]),
        )
        bottom = [
            dict(row, rank_within_step=720 - index)
            for index, row in enumerate(bottom_rows)
        ]
        groups[f"top30_step_{step:02d}"] = top
        groups[f"bottom30_step_{step:02d}"] = bottom
    return groups


class StepAdaptiveExtremeHeadZeroer(base.ExtremeHeadZeroer):
    def __init__(self, pipe, groups: dict[str, list[dict]]) -> None:
        self.adaptive_prefix: str | None = None
        super().__init__(pipe, groups)

    def set_variant(self, group: str | None, steps: tuple[int, ...]) -> None:
        self.adaptive_prefix = group
        super().set_variant(group, tuple(range(NUM_DENOISING_STEPS)))

    def _wrapped_model_fn(self, *args, **kwargs):
        timestep = kwargs.get("timestep")
        self.current_step = self._scheduler_step(timestep) if timestep is not None else -1
        if self.adaptive_prefix is None or self.current_step < 0:
            self.group = None
        else:
            self.group = f"{self.adaptive_prefix}_step_{self.current_step:02d}"
        return self.original_model_fn(*args, **kwargs)


base.select_heads = step_adaptive_heads
base.ExtremeHeadZeroer = StepAdaptiveExtremeHeadZeroer
base.STAGE_RANGES = (("steps_00_40", tuple(range(NUM_DENOISING_STEPS))),)


if __name__ == "__main__":
    base.main()
