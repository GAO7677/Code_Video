from __future__ import annotations

import unittest

import torch

from AAA_my_test.object_query_ablation_metrics.run_top100_m1_perturbed_attention_guidance import (
    adjusted_conditional_prediction,
)


class GuidanceEquationTest(unittest.TestCase):
    def test_adjusted_conditional_reproduces_difftrack_pag_cfg_equation(self) -> None:
        uncond = torch.tensor([1.0, -2.0])
        clean = torch.tensor([3.0, 4.0])
        perturbed = torch.tensor([-1.0, 2.0])
        cfg_scale = 5.0
        pag_scale = 0.75

        returned_cond = adjusted_conditional_prediction(
            clean,
            perturbed,
            cfg_scale=cfg_scale,
            pag_scale=pag_scale,
        )
        actual = uncond + cfg_scale * (returned_cond - uncond)
        expected = (
            uncond
            + cfg_scale * (clean - uncond)
            + pag_scale * (clean - perturbed)
        )

        torch.testing.assert_close(actual, expected)

    def test_zero_pag_scale_leaves_conditional_unchanged(self) -> None:
        clean = torch.tensor([3.0, 4.0])
        perturbed = torch.tensor([-1.0, 2.0])

        actual = adjusted_conditional_prediction(
            clean,
            perturbed,
            cfg_scale=5.0,
            pag_scale=0.0,
        )

        torch.testing.assert_close(actual, clean)


if __name__ == "__main__":
    unittest.main()
