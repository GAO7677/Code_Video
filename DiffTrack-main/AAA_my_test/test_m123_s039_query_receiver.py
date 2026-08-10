#!/usr/bin/env python3
"""CPU tests for exact M1/M2/M3 query-side receiver maps."""

from __future__ import annotations

import unittest

import torch

from AAA_my_test.run_legacy_m123_s039_query_receiver import (
    attention_receiver_values,
    receiver_groups,
)


class QueryReceiverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.r_by_time = [[0], [5], [10]]

    def group_sets(self, mode: str):
        return [
            (set(q.tolist()), set(k.tolist()))
            for q, k in receiver_groups(
                self.r_by_time, 4, mode, torch.device("cpu")
            )
        ]

    def test_m3_all_time_receiver_and_source_partitions(self) -> None:
        groups = self.group_sets("outgoing_only")
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[0][0], {1, 2, 3})
        self.assertEqual(groups[1][0], {4, 6, 7})
        self.assertEqual(groups[2][0], {8, 9, 11})
        self.assertTrue(all(source == {0, 5, 10} for _, source in groups))

    def test_m1_m2_all_time_partitions(self) -> None:
        m1 = self.group_sets("self_only")
        m2 = self.group_sets("incoming_only")
        self.assertEqual([receiver for receiver, _ in m1], [{0}, {5}, {10}])
        self.assertTrue(all(source == {0, 5, 10} for _, source in m1))
        self.assertEqual([receiver for receiver, _ in m2], [{0}, {5}, {10}])
        expected_c = {1, 2, 3, 4, 6, 7, 8, 9, 11}
        self.assertTrue(all(source == expected_c for _, source in m2))

    def test_m3_time_partitions(self) -> None:
        same = self.group_sets("outgoing_same")
        future = self.group_sets("outgoing_future")
        past = self.group_sets("outgoing_past")
        self.assertEqual([source for _, source in same], [{0}, {5}, {10}])
        self.assertEqual([source for _, source in future], [{0}, {0, 5}])
        self.assertEqual([source for _, source in past], [{5, 10}, {10}])

    def test_coefficient_and_value_outputs(self) -> None:
        q = torch.zeros((1, 3, 4), dtype=torch.float32)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        v[0, 0, :2] = torch.tensor([3.0, 4.0])

        def attention(query, key, value):
            heads = 2
            query_h = query.reshape(1, query.shape[1], heads, 2).transpose(1, 2)
            key_h = key.reshape(1, key.shape[1], heads, 2).transpose(1, 2)
            value_h = value.reshape(1, value.shape[1], heads, 2).transpose(1, 2)
            weights = torch.softmax(query_h @ key_h.transpose(-1, -2), dim=-1)
            output = weights @ value_h
            return output.transpose(1, 2).reshape(1, query.shape[1], 4)

        coefficient, value_norm, instances = attention_receiver_values(
            q,
            k,
            v,
            attention,
            [0],
            2,
            torch.tensor([0, 1, 2]),
            torch.tensor([0]),
        )
        torch.testing.assert_close(coefficient, torch.full((3,), 1 / 3))
        torch.testing.assert_close(value_norm, torch.full((3,), 5 / 3))
        self.assertEqual(instances, 1)


if __name__ == "__main__":
    unittest.main()
