#!/usr/bin/env python3
"""Capture exact all-token temporal attention for every PhysRVG block."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate.utils import set_seed

from allblock_ball_query_utils import parse_block_ids
from self_attention_matrix import (
    MatrixCaptureConfig,
    PhysRVGAttentionProcessorRecorder,
    pool_full_attention_matrix_with_temporal,
)


DEFAULT_PHYSRVG_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/"
    "code_phys_papers_compare/PhysRVG-main"
)


def _extract_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attention-output-root", type=Path, required=True)
    parser.add_argument("--attention-blocks", required=True)
    parser.add_argument("--attention-step", type=int, default=25)
    parser.add_argument(
        "--physrvg-root",
        type=Path,
        default=Path(os.environ.get("PHYSRVG_ROOT", DEFAULT_PHYSRVG_ROOT)),
    )
    return parser.parse_known_args(argv)


class AllTokenTemporalRecorder:
    def __init__(
        self,
        *,
        config: MatrixCaptureConfig,
        output_root: Path,
    ) -> None:
        config.validate()
        self.config = config
        self.output_root = output_root.expanduser().resolve()
        self.active = False
        self.current_step: int | None = None
        self.grid: tuple[int, int, int] | None = None
        self.case_key: str | None = None
        self.case_metadata: dict[str, Any] = {}
        self.captures: dict[int, dict[str, Any]] = {}

    def begin_case(
        self, case_key: str, *, metadata: dict[str, Any] | None = None
    ) -> None:
        self.case_key = str(case_key)
        self.case_metadata = dict(metadata or {})
        self.captures = {}

    def set_grid(self, grid: tuple[int, int, int]) -> None:
        self.grid = tuple(int(value) for value in grid)

    @torch.no_grad()
    def capture(self, *, q: torch.Tensor, k: torch.Tensor, num_heads: int) -> None:
        step = self.current_step
        if not self.active or step is None or step not in self.config.step_numbers:
            return
        if self.grid is None:
            raise RuntimeError("latent grid is not configured")
        print(
            f"[physrvg-all-token] block={self.config.block_id} step={step}",
            flush=True,
        )
        block_mean, key_mass, metadata, temporal = (
            pool_full_attention_matrix_with_temporal(
                q,
                k,
                num_heads=int(num_heads),
                output_bins=int(self.config.output_bins),
                query_chunk=int(self.config.query_chunk),
                temporal_grid=self.grid,
            )
        )
        self.captures[int(step)] = {
            "block_mean": block_mean,
            "key_mass": key_mass,
            "metadata": metadata,
            "temporal": temporal,
        }

    def finalize_case(self) -> Path:
        if self.case_key is None or self.grid is None:
            raise RuntimeError("case/grid is missing")
        missing = sorted(set(self.config.step_numbers) - set(self.captures))
        if missing:
            raise RuntimeError(
                f"Block {self.config.block_id} missing captures: {missing}"
            )
        case_dir = (
            self.output_root
            / f"block{self.config.block_id:02d}"
            / "matrices"
            / "physrvg"
            / self.case_key
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for step in self.config.step_numbers:
            capture = self.captures[int(step)]
            step_dir = case_dir / f"step_{step:02d}"
            step_dir.mkdir(exist_ok=True)
            name = f"block{self.config.block_id:02d}_all_token_matrix.npz"
            np.savez_compressed(
                step_dir / name,
                block_mean=capture["block_mean"].astype(np.float32),
                key_mass=capture["key_mass"].astype(np.float32),
                **{
                    key: value.astype(np.float32)
                    for key, value in capture["temporal"].items()
                },
            )
            entries.append(
                {
                    "step_number_one_based": int(step),
                    "directory": step_dir.name,
                    "full_matrix_npz": name,
                    "full_matrix_metadata": capture["metadata"],
                }
            )
        summary = {
            "model": "physrvg",
            "case": self.case_key,
            "block_id": int(self.config.block_id),
            "latent_grid": list(self.grid),
            "cfg_branch": "official PhysRVG single model call",
            "query_sampling": "none",
            "exact_self_policy": (
                "removed per query; remaining key attention renormalized"
            ),
            "case_metadata": self.case_metadata,
            "steps": entries,
        }
        path = case_dir / "summary.json"
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[physrvg-all-token] wrote {path}", flush=True)
        return path


def main() -> None:
    custom, remaining = _extract_args(sys.argv[1:])
    physrvg_root = custom.physrvg_root.expanduser().resolve()
    if not physrvg_root.is_dir():
        raise FileNotFoundError(physrvg_root)
    sys.path.insert(0, str(physrvg_root))

    import batch_infer_from_input_json_lists as base
    from fastvideo.models.wan_v2v import model_wan_v2v as model_module

    blocks = parse_block_ids(custom.attention_blocks)
    output_root = custom.attention_output_root.expanduser().resolve()
    original_run_case = base._run_single_case
    active_processors: list[PhysRVGAttentionProcessorRecorder] = []

    def run_case_with_recorders(**kwargs):
        input_json = Path(kwargs["input_json_path"]).expanduser().resolve()
        pipe = kwargs["pipe"]
        args = kwargs["args"]
        recorders = [
            AllTokenTemporalRecorder(
                config=MatrixCaptureConfig(
                    block_id=block,
                    step_numbers=(int(custom.attention_step),),
                ),
                output_root=output_root,
            )
            for block in blocks
        ]
        processors = [
            PhysRVGAttentionProcessorRecorder(
                recorder=recorder,
                model_module=model_module,
            )
            for recorder in recorders
        ]
        active_processors[:] = processors
        for processor in processors:
            processor.install(pipe.transformer)

        transformer = getattr(
            getattr(pipe.transformer, "base_model", None),
            "model",
            pipe.transformer,
        )
        patch_t, patch_h, patch_w = (
            int(value) for value in transformer.config.patch_size
        )
        latent_frames = (
            (int(args.num_frames) - 1) // int(pipe.vae_scale_factor_temporal)
        ) + 1
        grid = (
            latent_frames // patch_t,
            (int(args.height) // int(pipe.vae_scale_factor_spatial)) // patch_h,
            (int(args.width) // int(pipe.vae_scale_factor_spatial)) // patch_w,
        )
        for recorder in recorders:
            recorder.set_grid(grid)
            recorder.begin_case(
                input_json.stem,
                metadata={"input_json": str(input_json)},
            )
        for processor in processors:
            processor.begin_case()
        set_seed(int(args.seed))
        try:
            result = original_run_case(**kwargs)
            if result[0]:
                for recorder in recorders:
                    recorder.finalize_case()
            return result
        finally:
            for processor in reversed(processors):
                processor.restore()
            active_processors.clear()

    base._run_single_case = run_case_with_recorders
    sys.argv = [sys.argv[0], *remaining]
    try:
        base.main()
    finally:
        for processor in reversed(active_processors):
            processor.restore()


if __name__ == "__main__":
    main()
