#!/usr/bin/env python3
"""Small CPU tests for the exact temporal coefficient masks."""

from __future__ import annotations

import unittest

import numpy as np

from AAA_my_test.run_legacy_m123_head_scope_s039_top100_mean import build_delete_mask


class DeleteMaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.r_by_time = [[0], [5], [10]]
        self.q = np.asarray([5, 4], dtype=np.int64)  # R_t1, C_t1

    def selected(self, mode: str, query_offset: int) -> set[int]:
        mask = build_delete_mask(self.r_by_time, 4, self.q, mode)
        return set(np.flatnonzero(mask[query_offset]).tolist())

    def test_m1_temporal_partition(self) -> None:
        self.assertEqual(self.selected("self_only", 0), {0, 5, 10})
        self.assertEqual(self.selected("self_same", 0), {5})
        self.assertEqual(self.selected("self_future", 0), {0})
        self.assertEqual(self.selected("self_past", 0), {10})
        self.assertEqual(self.selected("self_only", 1), set())

    def test_m2_temporal_partition(self) -> None:
        self.assertEqual(
            self.selected("incoming_only", 0), {1, 2, 3, 4, 6, 7, 8, 9, 11}
        )
        self.assertEqual(self.selected("incoming_same", 0), {4, 6, 7})
        self.assertEqual(self.selected("incoming_future", 0), {1, 2, 3})
        self.assertEqual(self.selected("incoming_past", 0), {8, 9, 11})
        self.assertEqual(self.selected("incoming_only", 1), set())

    def test_m3_temporal_partition(self) -> None:
        self.assertEqual(self.selected("outgoing_only", 1), {0, 5, 10})
        self.assertEqual(self.selected("outgoing_same", 1), {5})
        self.assertEqual(self.selected("outgoing_future", 1), {0})
        self.assertEqual(self.selected("outgoing_past", 1), {10})
        self.assertEqual(self.selected("outgoing_only", 0), set())

    def test_same_future_past_are_disjoint_union(self) -> None:
        for base in ("self", "incoming", "outgoing"):
            target_offset = 1 if base == "outgoing" else 0
            all_entries = self.selected(f"{base}_only", target_offset)
            parts = [
                self.selected(f"{base}_{suffix}", target_offset)
                for suffix in ("same", "future", "past")
            ]
            self.assertEqual(set.union(*parts), all_entries)
            self.assertTrue(
                all(
                    parts[i].isdisjoint(parts[j])
                    for i in range(3)
                    for j in range(i)
                )
            )


if __name__ == "__main__":
    unittest.main()
