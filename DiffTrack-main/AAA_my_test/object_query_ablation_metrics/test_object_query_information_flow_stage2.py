#!/usr/bin/env python3
"""CPU gates for Stage 2 object-query information-flow experiments."""

from __future__ import annotations

import json
import math
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from AAA_my_test.run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations import (
    AttentionMatrixAblator,
)
from AAA_my_test.run_legacy_ti2v_temporal_object_tube_ablations import (
    apply_temporal_directional_ablation,
    head_scope_counts,
    selected_head_entries,
    temporal_directional_groups,
)


HEAD_SCOPES = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/head_scopes_latest3350_with_random100.json"
)


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch, query_tokens, width = q.shape
    key_tokens = k.shape[1]
    heads, head_dim = width // 128, 128
    qh = q.reshape(batch, query_tokens, heads, head_dim).transpose(1, 2)
    kh = k.reshape(batch, key_tokens, heads, head_dim).transpose(1, 2)
    vh = v.reshape(batch, key_tokens, heads, head_dim).transpose(1, 2)
    weights = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(head_dim), dim=-1)
    return (weights @ vh).transpose(1, 2).reshape(batch, query_tokens, width)


class Stage2InformationFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(47326)
        self.time_count = 3
        self.frame_tokens = 4
        self.tokens_by_time = [[0], [5], [10]]
        token_count = self.time_count * self.frame_tokens
        self.q = torch.randn(1, token_count, 256, generator=generator, dtype=torch.float64)
        self.k = torch.randn(1, token_count, 256, generator=generator, dtype=torch.float64)
        self.v = torch.randn(1, token_count, 256, generator=generator, dtype=torch.float64)
        self.rows = torch.tensor([0, 5, 10], dtype=torch.long)
        self.heads = [1]

    def base_ablator(self, mode: str, record_dose: bool = False) -> AttentionMatrixAblator:
        instance = AttentionMatrixAblator(
            pipe=object(),
            entries=[{"block": 0, "head": 1}],
            query_points=[[0.0, 0.0]],
            region_slices={},
            pixel_hw=(1, 1),
            target_scope="all_objects",
            mask_mode=mode,
            region=None,
            record_dose=record_dose,
        )
        instance.active = True
        instance.current_step = 0
        instance.current_cfg_call = 0
        instance._rows = lambda device: self.rows.to(device)  # type: ignore[method-assign]
        return instance

    def test_temporal_partitions_are_disjoint_and_complete(self) -> None:
        for prefix in ("self", "incoming", "outgoing"):
            edge_sets = []
            for direction in ("same", "future", "past"):
                groups = temporal_directional_groups(
                    self.tokens_by_time,
                    self.frame_tokens,
                    f"{prefix}_{direction}",
                    torch.device("cpu"),
                )
                edges = {
                    (int(query), int(key))
                    for queries, keys in groups
                    for query in queries.tolist()
                    for key in keys.tolist()
                }
                edge_sets.append(edges)
            self.assertFalse(edge_sets[0] & edge_sets[1])
            self.assertFalse(edge_sets[0] & edge_sets[2])
            self.assertFalse(edge_sets[1] & edge_sets[2])

            rows = set(self.rows.tolist())
            complement = set(range(self.time_count * self.frame_tokens)) - rows
            if prefix == "self":
                expected = {(q, k) for q in rows for k in rows}
            elif prefix == "incoming":
                expected = {(q, k) for q in rows for k in complement}
            else:
                expected = {(q, k) for q in complement for k in rows}
            self.assertEqual(set.union(*edge_sets), expected)

    def test_all_time_equals_same_plus_future_plus_past(self) -> None:
        mapping = {
            "self": "self_only",
            "incoming": "incoming_only",
            "outgoing": "outgoing_only",
        }
        for prefix, all_time_mode in mapping.items():
            with self.subTest(flow=prefix):
                expected = self.base_ablator(all_time_mode)._attention(
                    self.q, self.k, self.v, dense_attention, block=0
                )
                actual = dense_attention(self.q, self.k, self.v)
                for direction in ("same", "future", "past"):
                    groups = temporal_directional_groups(
                        self.tokens_by_time,
                        self.frame_tokens,
                        f"{prefix}_{direction}",
                        torch.device("cpu"),
                    )
                    apply_temporal_directional_ablation(
                        actual,
                        self.q,
                        self.k,
                        self.v,
                        dense_attention,
                        self.heads,
                        2,
                        groups,
                    )
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_noop_path_is_exact(self) -> None:
        instance = self.base_ablator("self_only")
        instance.active = False
        actual = instance._attention(self.q, self.k, self.v, dense_attention, block=0)
        expected = dense_attention(self.q, self.k, self.v)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dose_matches_dense_attention_definition(self) -> None:
        qh = self.q.reshape(1, 12, 2, 128).transpose(1, 2)
        kh = self.k.reshape(1, 12, 2, 128).transpose(1, 2)
        vh = self.v.reshape(1, 12, 2, 128).transpose(1, 2)
        weights = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(128), dim=-1)
        original = weights @ vh
        rows = self.rows
        all_rows = torch.arange(12)
        complement = all_rows[~torch.isin(all_rows, rows)]
        definitions = {
            "self_only": (rows, rows),
            "incoming_only": (rows, complement),
            "outgoing_only": (complement, rows),
        }
        for mode, (targets, sources) in definitions.items():
            with self.subTest(mode=mode):
                instance = self.base_ablator(mode, record_dose=True)
                instance._attention(self.q, self.k, self.v, dense_attention, block=0)
                selected_weights = weights[:, 1][:, targets][:, :, sources]
                selected_values = vh[:, 1][:, sources]
                contribution = selected_weights @ selected_values
                expected_mass = float(selected_weights.sum(dim=-1).mean())
                expected_removed = float(torch.linalg.vector_norm(contribution, dim=-1).mean())
                expected_original = float(
                    torch.linalg.vector_norm(original[:, 1][:, targets], dim=-1).mean()
                )
                self.assertAlmostEqual(
                    float(instance.dose_attention_mass[0, 0, 0, 1]), expected_mass, places=5
                )
                self.assertAlmostEqual(
                    float(instance.dose_removed_value_norm[0, 0, 0, 1]),
                    expected_removed,
                    places=5,
                )
                self.assertAlmostEqual(
                    float(instance.dose_original_output_norm[0, 0, 0, 1]),
                    expected_original,
                    places=5,
                )

    def test_random_scopes_match_layers_and_exclude_fixed_extremes(self) -> None:
        payload = json.loads(HEAD_SCOPES.read_text(encoding="utf-8"))
        entries = payload["entries"]
        definitions = payload["head_scopes"]
        counts = head_scope_counts(payload)
        self.assertEqual(counts["random100_layer_matched_draw0"], 100)
        top = {(int(row["block"]), int(row["head"])) for row in entries[:100]}
        bottom = {(int(row["block"]), int(row["head"])) for row in entries[-100:]}
        top_hist = Counter(block for block, _ in top)
        for draw in range(3):
            name = f"random100_layer_matched_draw{draw}"
            selected = selected_head_entries(entries, name, definitions)
            pairs = {(int(row["block"]), int(row["head"])) for row in selected}
            self.assertEqual(len(pairs), 100)
            self.assertFalse(pairs & top)
            self.assertFalse(pairs & bottom)
            self.assertEqual(Counter(block for block, _ in pairs), top_hist)


if __name__ == "__main__":
    unittest.main()
