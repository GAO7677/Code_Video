#!/usr/bin/env python3
"""Small CPU test for synchronized compact all-block ball-query capture."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from allblock_ball_query_utils import build_recorder_group, install_diffsynth_group
from self_attention_matrix import DiffSynthAttentionScope


class FakeInner:
    def forward(self, q, k, v):
        return v


class FakeDiT:
    patch_size = (1, 1, 1)

    def __init__(self) -> None:
        self.blocks = [
            SimpleNamespace(
                self_attn=SimpleNamespace(num_heads=2, attn=FakeInner())
            )
            for _ in range(3)
        ]


def test_allblock_capture() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        preview = output / "preview.png"
        preview.write_bytes(b"preview")
        group = build_recorder_group(
            blocks_text="0,1,2",
            steps_text="1,2",
            model_label="test",
            output_root=output,
            query_coords_text="0:0:0",
            query_video_frame=0,
            query_preview=preview,
        )
        group.begin_case("case")
        dit = FakeDiT()
        originals = [block.self_attn.attn.forward for block in dit.blocks]

        def model_fn(*args, **kwargs):
            q = kwargs["q"]
            for block in dit.blocks:
                block.self_attn.attn.forward(q, q, q)
            return q

        pipe = SimpleNamespace(model_fn=model_fn)
        restore = install_diffsynth_group(dit, group)
        scope = DiffSynthAttentionScope(pipe=pipe, recorder=group, cfg_scale=1.0)
        scope.install()
        try:
            q = torch.randn(1, 4, 8)
            latents = torch.randn(1, 1, 1, 2, 2)
            for _ in range(2):
                pipe.model_fn(q=q, latents=latents, dit=dit)
        finally:
            scope.restore()
            restore()
        summaries = group.finalize_case()
        assert set(summaries) == {0, 1, 2}
        for block, summary_path in summaries.items():
            assert summary_path.is_file()
            matrix = (
                summary_path.parent
                / "step_01"
                / f"block{block:02d}_ball_query_attention.npz"
            )
            with np.load(matrix) as arrays:
                attention = arrays["attention"]
            assert attention.shape == (2, 1, 2, 2)
            np.testing.assert_allclose(
                attention.sum(axis=(1, 2, 3)), 1.0, atol=2.0e-7
            )
            assert not (summary_path.parent / "index.html").exists()
        for original, block in zip(originals, dit.blocks):
            assert block.self_attn.attn.forward == original


if __name__ == "__main__":
    test_allblock_capture()
    print("all-block compact capture test passed")
