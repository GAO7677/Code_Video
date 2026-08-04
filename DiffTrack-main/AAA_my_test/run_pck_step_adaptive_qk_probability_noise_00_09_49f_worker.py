#!/usr/bin/env python3
"""Run 49-frame adaptive Q@K intervention only at denoising steps 00-09."""

from __future__ import annotations

import importlib.util
from pathlib import Path


BASE = Path(__file__).with_name(
    "run_pck_step_adaptive_qk_probability_noise_49f_worker.py"
)
spec = importlib.util.spec_from_file_location("qk_probability_noise_49f", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import worker: {BASE}")
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)
worker = stage.worker


def set_variant_00_09(self, group: str | None, steps: tuple[int, ...]) -> None:
    self.adaptive_prefix = group
    self.group = group
    self.active_steps = set(steps)


worker.AdaptiveQKLogitNoise.set_variant = set_variant_00_09
worker.base.STAGE_RANGES = (("steps_00_10", tuple(range(0, 10))),)


if __name__ == "__main__":
    worker.load_capture_prompt_cases()
    worker.write_experiment_metadata()
    worker.base.main()
