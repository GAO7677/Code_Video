#!/usr/bin/env python3

from __future__ import annotations

import unittest

import torch

from AAA_my_test.object_query_ablation_metrics.training_free_m1_control.run_multi_object_guidance_search import (
    apply_grouped_m1_ablation,
    block_diagonal_groups,
    validate_window,
)


class BlockDiagonalGroupsTest(unittest.TestCase):
    def test_disjoint_objects_preserve_cross_pairs(self) -> None:
        groups, audit = block_diagonal_groups(
            {
                "object_A": torch.tensor([0, 1]),
                "object_B": torch.tensor([2, 3]),
            },
            torch.device("cpu"),
        )
        pairs = {
            (int(q), int(k))
            for queries, keys in groups
            for q in queries.tolist()
            for k in keys.tolist()
        }
        self.assertEqual(
            pairs,
            {(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 3), (3, 2), (3, 3)},
        )
        self.assertNotIn((0, 2), pairs)
        self.assertNotIn((3, 1), pairs)
        self.assertEqual(audit["deleted_pair_count_per_head"], 8)

    def test_overlap_is_set_union_not_double_subtraction(self) -> None:
        groups, audit = block_diagonal_groups(
            {
                "object_A": torch.tensor([0, 1]),
                "object_B": torch.tensor([1, 2]),
            },
            torch.device("cpu"),
        )
        pairs = [
            (int(q), int(k))
            for queries, keys in groups
            for q in queries.tolist()
            for k in keys.tolist()
        ]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(audit["deleted_pair_count_per_head"], 7)
        self.assertEqual(audit["duplicate_pair_subtractions_prevented"], 1)
        self.assertEqual(audit["overlap_token_count"], 1)


class ExactContributionTest(unittest.TestCase):
    def test_grouped_result_matches_explicit_attention_pair_zeroing(self) -> None:
        torch.manual_seed(7)
        batch, tokens, heads, width = 1, 5, 2, 3
        q = torch.randn(batch, tokens, heads * width)
        k = torch.randn(batch, tokens, heads * width)
        v = torch.randn(batch, tokens, heads * width)

        def attention(qx, kx, vx):
            qh = qx.reshape(qx.shape[0], qx.shape[1], heads, width).transpose(1, 2)
            kh = kx.reshape(kx.shape[0], kx.shape[1], heads, width).transpose(1, 2)
            vh = vx.reshape(vx.shape[0], vx.shape[1], heads, width).transpose(1, 2)
            weights = torch.softmax(qh @ kh.transpose(-1, -2) / width**0.5, dim=-1)
            return (weights @ vh).transpose(1, 2).reshape(qx.shape[0], qx.shape[1], -1)

        output = attention(q, k, v)
        expected = output.clone().reshape(batch, tokens, heads, width)
        qh = q.reshape(batch, tokens, heads, width).transpose(1, 2)
        kh = k.reshape(batch, tokens, heads, width).transpose(1, 2)
        vh = v.reshape(batch, tokens, heads, width).transpose(1, 2)
        weights = torch.softmax(qh @ kh.transpose(-1, -2) / width**0.5, dim=-1)
        object_sets = ([0, 1], [3, 4])
        for rows in object_sets:
            for query in rows:
                expected[0, query, 1] -= sum(
                    weights[0, 1, query, key] * vh[0, 1, key] for key in rows
                )

        groups, _ = block_diagonal_groups(
            {
                "object_A": torch.tensor(object_sets[0]),
                "object_B": torch.tensor(object_sets[1]),
            },
            torch.device("cpu"),
        )
        actual = output.clone()
        calls, rows, pairs = apply_grouped_m1_ablation(
            output=actual,
            q=q,
            k=k,
            v=v,
            original=attention,
            heads=[1],
            num_heads=heads,
            groups=groups,
            group_batch_size=2,
        )
        self.assertEqual(calls, 1)
        self.assertEqual(rows, 4)
        self.assertEqual(pairs, 8)
        torch.testing.assert_close(actual, expected.reshape_as(actual), rtol=1e-5, atol=1e-6)


class WindowValidationTest(unittest.TestCase):
    def test_valid_windows(self) -> None:
        for end in (4, 9, 19, 39):
            validate_window(0, end)

    def test_invalid_windows(self) -> None:
        for start, end in ((-1, 4), (5, 4), (0, 40)):
            with self.assertRaises(ValueError):
                validate_window(start, end)


if __name__ == "__main__":
    unittest.main()
