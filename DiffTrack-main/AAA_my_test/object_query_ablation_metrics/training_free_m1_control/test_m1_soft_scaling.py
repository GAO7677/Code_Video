from __future__ import annotations

import unittest

import torch

from AAA_my_test.object_query_ablation_metrics.training_free_m1_control.run_m1_soft_scaling import (
    fp32_attention_decomposition_audit,
    soft_scaled_output,
)


class M1SoftScalingAlgebraTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
        self.m1 = torch.tensor([[0.5, 1.0], [-1.0, 2.0]])

    def test_alpha_zero_is_exact_noop(self) -> None:
        actual = soft_scaled_output(self.original, self.m1, 0.0)
        self.assertTrue(torch.equal(actual, self.original))

    def test_alpha_minus_one_is_exact_knockout_formula(self) -> None:
        actual = soft_scaled_output(self.original, self.m1, -1.0)
        torch.testing.assert_close(actual, self.original - self.m1)

    def test_symmetric_scaling_has_expected_signed_delta(self) -> None:
        weakened = soft_scaled_output(self.original, self.m1, -0.5)
        enhanced = soft_scaled_output(self.original, self.m1, 0.5)
        torch.testing.assert_close(weakened - self.original, -0.5 * self.m1)
        torch.testing.assert_close(enhanced - self.original, 0.5 * self.m1)

    def test_fp32_attention_decomposition_is_strict(self) -> None:
        audit = fp32_attention_decomposition_audit()
        self.assertTrue(audit["passed"])
        self.assertLessEqual(audit["max_abs_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()
