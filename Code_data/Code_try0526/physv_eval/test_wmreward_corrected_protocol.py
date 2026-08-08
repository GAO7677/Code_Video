#!/usr/bin/env python3
"""CPU checks for the corrected ball-block WMReward protocol."""

from __future__ import annotations

import unittest

import torch

from .wmreward_official import WMRewardRunner


class _Encoder:
    tubelet_size = 2


class CorrectedWMRewardProtocolTest(unittest.TestCase):
    def runner(self) -> tuple[WMRewardRunner, dict]:
        runner = WMRewardRunner(
            window_size=16,
            context_frames=2,
            cosine_dim=-1,
            require_tubelet_aligned_context=True,
        )
        runner._torch = torch
        runner._models = (_Encoder(), object(), object(), 384)
        captured = {}

        def compute(**kwargs):
            captured.update(kwargs)
            return torch.tensor(0.25)

        runner._compute_loss = compute
        return runner, captured

    def test_corrected_score_uses_feature_cosine_and_aligned_context(self) -> None:
        runner, captured = self.runner()
        result = runner.score_tensor(torch.zeros(1, 3, 16, 2, 2), context_frames=4)
        self.assertEqual(captured["cosine_dim"], -1)
        self.assertEqual(captured["context_frames"], 4)
        self.assertEqual(result["effective_context_frames"], 4)
        self.assertEqual(result["context_tubelets"], 2)
        self.assertEqual(result["tubelet_size"], 2)

    def test_odd_context_is_rejected(self) -> None:
        runner, _ = self.runner()
        with self.assertRaisesRegex(ValueError, "divisible by tubelet_size"):
            runner.score_tensor(torch.zeros(1, 3, 16, 2, 2), context_frames=5)

    def test_context_shorter_than_one_tubelet_is_rejected(self) -> None:
        runner, _ = self.runner()
        with self.assertRaisesRegex(ValueError, "divisible by tubelet_size"):
            runner.score_tensor(torch.zeros(1, 3, 16, 2, 2), context_frames=1)


if __name__ == "__main__":
    unittest.main()
