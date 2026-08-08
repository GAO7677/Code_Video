#!/usr/bin/env python3
"""CPU checks for the PhysicIQ67 attention-matrix intervention algebra."""

from __future__ import annotations

import math
import unittest

import torch

from AAA_my_test.run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations import (
    AttentionMatrixAblator,
    MATRIX_MASKS,
    build_tasks,
)


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch, tokens, width = q.shape
    heads, head_dim = width // 128, 128
    qh = q.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
    kh = k.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
    vh = v.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
    weights = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(head_dim), dim=-1)
    output = weights @ vh
    return output.transpose(1, 2).reshape(batch, tokens, width)


class AttentionMatrixAblationTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(47326)
        self.q = torch.randn(1, 5, 256, generator=generator, dtype=torch.float64)
        self.k = torch.randn(1, 5, 256, generator=generator, dtype=torch.float64)
        self.v = torch.randn(1, 5, 256, generator=generator, dtype=torch.float64)
        self.rows = torch.tensor([1, 3], dtype=torch.long)
        self.head = 1

    def ablator(self, mask_mode: str) -> AttentionMatrixAblator:
        target_scope = (
            "all_tokens"
            if mask_mode in {"qk_logits_zero", "full_head_output"}
            else "all_objects"
        )
        instance = AttentionMatrixAblator(
            pipe=object(),
            entries=[{"block": 0, "head": self.head}],
            query_points=[[0.0, 0.0]],
            region_slices={},
            pixel_hw=(1, 1),
            target_scope=target_scope,
            mask_mode=mask_mode,
            region=None,
        )
        instance.active = True
        instance._rows = lambda device: (  # type: ignore[method-assign]
            None
            if target_scope == "all_tokens"
            else self.rows.to(device)
        )
        return instance

    def expected_matrix_mask(self, mask_mode: str) -> torch.Tensor:
        batch, tokens, width = self.q.shape
        heads, head_dim = width // 128, 128
        qh = self.q.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
        kh = self.k.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
        vh = self.v.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
        weights = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(head_dim), dim=-1)
        selected = torch.zeros(tokens, dtype=torch.bool)
        selected[self.rows] = True
        complement = ~selected
        complement_rows = torch.arange(tokens)[complement]
        target = weights.clone()
        head_weights = target[:, self.head]
        if mask_mode in {"self_only", "query_row", "key_value_column", "row_and_column"}:
            head_weights[:, self.rows[:, None], self.rows] = 0
        if mask_mode in {"incoming_only", "query_row", "cross_boundary", "row_and_column"}:
            head_weights[:, self.rows[:, None], complement] = 0
        if mask_mode in {"outgoing_only", "key_value_column", "cross_boundary", "row_and_column"}:
            head_weights[:, complement_rows[:, None], self.rows] = 0
        output = target @ vh
        return output.transpose(1, 2).reshape(batch, tokens, width)

    def test_all_seven_matrix_masks_match_dense_definition(self) -> None:
        for mask_mode in MATRIX_MASKS:
            with self.subTest(mask_mode=mask_mode):
                actual = self.ablator(mask_mode)._attention(
                    self.q, self.k, self.v, dense_attention, block=0
                )
                expected = self.expected_matrix_mask(mask_mode)
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_literal_kv_zero_is_not_post_softmax_column_zero(self) -> None:
        actual = self.ablator("literal_kv_zero")._attention(
            self.q, self.k, self.v, dense_attention, block=0
        )
        kh = self.k.reshape(1, 5, 2, 128).clone()
        vh = self.v.reshape(1, 5, 2, 128).clone()
        kh[:, self.rows, self.head] = 0
        vh[:, self.rows, self.head] = 0
        expected = dense_attention(self.q, kh.reshape_as(self.k), vh.reshape_as(self.v))
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        column_zero = self.expected_matrix_mask("key_value_column")
        self.assertGreater(float((actual - column_zero).abs().max()), 1e-6)

    def test_full_head_output_zero_is_not_qk_logits_zero(self) -> None:
        full_zero = self.ablator("full_head_output")._attention(
            self.q, self.k, self.v, dense_attention, block=0
        )
        qk_zero = self.ablator("qk_logits_zero")._attention(
            self.q, self.k, self.v, dense_attention, block=0
        )
        full_heads = full_zero.reshape(1, 5, 2, 128)
        qk_heads = qk_zero.reshape(1, 5, 2, 128)
        self.assertEqual(float(full_heads[:, :, self.head].abs().max()), 0.0)
        expected_mean_v = self.v.reshape(1, 5, 2, 128)[:, :, self.head].mean(dim=1)
        torch.testing.assert_close(
            qk_heads[:, :, self.head],
            expected_mean_v[:, None, :].expand(-1, 5, -1),
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertGreater(float(qk_heads[:, :, self.head].abs().max()), 1e-6)
        torch.testing.assert_close(
            full_heads[:, :, 0],
            dense_attention(self.q, self.k, self.v).reshape(1, 5, 2, 128)[:, :, 0],
        )
        torch.testing.assert_close(
            qk_heads[:, :, 0],
            dense_attention(self.q, self.k, self.v).reshape(1, 5, 2, 128)[:, :, 0],
        )

    def test_task_matrix_includes_object_union_and_controls(self) -> None:
        manifest = {
            "samples": [
                {
                    "case": "case",
                    "seed": 47326,
                    "regions": [
                        {"region_type": "object", "region_name": "object_A"},
                        {"region_type": "object", "region_name": "object_B"},
                    ],
                }
            ]
        }
        tasks = build_tasks(manifest)
        self.assertEqual(len(tasks), 78)
        self.assertEqual(sum(task["mask_mode"] == "full_head_output" for task in tasks), 3)
        self.assertEqual(sum(task["mask_mode"] == "qk_logits_zero" for task in tasks), 3)
        self.assertEqual(sum(task["mask_mode"] == "literal_kv_zero" for task in tasks), 9)


if __name__ == "__main__":
    unittest.main()
