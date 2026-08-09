from __future__ import annotations

import math
import unittest

import torch

from AAA_my_test.run_legacy_ti2v_temporal_object_tube_ablations import (
    TEMPORAL_DIRECTIONAL_MODES,
    apply_temporal_directional_ablation,
    temporal_directional_groups,
)


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch, query_count, width = q.shape
    num_heads, head_dim = 2, width // 2
    qh = q.reshape(batch, query_count, num_heads, head_dim).transpose(1, 2)
    kh = k.reshape(batch, k.shape[1], num_heads, head_dim).transpose(1, 2)
    vh = v.reshape(batch, v.shape[1], num_heads, head_dim).transpose(1, 2)
    weights = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(head_dim), dim=-1)
    return (weights @ vh).transpose(1, 2).reshape(batch, query_count, width)


class TemporalDirectionalAblationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokens_by_time = [[0], [2], [4]]
        self.frame_token_count = 2

    def test_strict_future_and_past_groups(self) -> None:
        future = temporal_directional_groups(
            self.tokens_by_time, self.frame_token_count, "self_future", torch.device("cpu")
        )
        past = temporal_directional_groups(
            self.tokens_by_time, self.frame_token_count, "self_past", torch.device("cpu")
        )
        self.assertEqual(
            [(target.tolist(), source.tolist()) for target, source in future],
            [([2], [0]), ([4], [0, 2])],
        )
        self.assertEqual(
            [(target.tolist(), source.tolist()) for target, source in past],
            [([0], [2, 4]), ([2], [4])],
        )

    def test_same_time_groups_keep_only_equal_time_indices(self) -> None:
        expected = {
            "self_same": [([0], [0]), ([2], [2]), ([4], [4])],
            "incoming_same": [([0], [1]), ([2], [3]), ([4], [5])],
            "outgoing_same": [([1], [0]), ([3], [2]), ([5], [4])],
        }
        for mask_mode, groups_expected in expected.items():
            with self.subTest(mask_mode=mask_mode):
                groups = temporal_directional_groups(
                    self.tokens_by_time,
                    self.frame_token_count,
                    mask_mode,
                    torch.device("cpu"),
                )
                self.assertEqual(
                    [(target.tolist(), source.tolist()) for target, source in groups],
                    groups_expected,
                )

    def test_every_mode_matches_explicit_post_softmax_subtraction(self) -> None:
        torch.manual_seed(47326)
        q = torch.randn(1, 6, 6, dtype=torch.float64)
        k = torch.randn(1, 6, 6, dtype=torch.float64)
        v = torch.randn(1, 6, 6, dtype=torch.float64)
        num_heads, selected_head = 2, 1
        head_dim = q.shape[-1] // num_heads
        qh = q.reshape(1, 6, num_heads, head_dim).transpose(1, 2)
        kh = k.reshape(1, 6, num_heads, head_dim).transpose(1, 2)
        vh = v.reshape(1, 6, num_heads, head_dim).transpose(1, 2)
        probabilities = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(head_dim), dim=-1)

        for mask_mode in TEMPORAL_DIRECTIONAL_MODES:
            with self.subTest(mask_mode=mask_mode):
                groups = temporal_directional_groups(
                    self.tokens_by_time,
                    self.frame_token_count,
                    mask_mode,
                    torch.device("cpu"),
                )
                baseline = reference_attention(q, k, v)
                actual = baseline.clone()
                apply_temporal_directional_ablation(
                    actual,
                    q,
                    k,
                    v,
                    reference_attention,
                    [selected_head],
                    num_heads,
                    groups,
                )

                expected_heads = baseline.reshape(1, 6, num_heads, head_dim).clone()
                for target_rows, source_rows in groups:
                    weights = probabilities[:, selected_head].index_select(1, target_rows)
                    weights = weights.index_select(2, source_rows)
                    values = vh[:, selected_head].index_select(1, source_rows)
                    contribution = torch.einsum("bqs,bsd->bqd", weights, values)
                    expected_heads[:, target_rows, selected_head, :] -= contribution
                expected = expected_heads.reshape_as(baseline)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
