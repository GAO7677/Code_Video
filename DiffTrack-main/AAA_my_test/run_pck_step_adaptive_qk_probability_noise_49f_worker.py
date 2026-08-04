#!/usr/bin/env python3
"""Run the probability-noise worker with generation fixed to 49 frames."""

from __future__ import annotations

import importlib.util
from pathlib import Path


WORKER = Path(__file__).with_name("run_pck_step_adaptive_qk_logit_noise_worker.py")
spec = importlib.util.spec_from_file_location("probability_noise_worker_49f_base", WORKER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import worker: {WORKER}")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

core = worker.base.top5.source.target.core
original_align = core.align_generation_num_frames


def align_to_49_frames(_: int) -> int:
    aligned = int(original_align(48))
    if aligned != 49:
        raise RuntimeError(f"Expected frame alignment 48 -> 49, got {aligned}")
    return aligned


core.align_generation_num_frames = align_to_49_frames


if __name__ == "__main__":
    worker.write_experiment_metadata()
    worker.base.main()
