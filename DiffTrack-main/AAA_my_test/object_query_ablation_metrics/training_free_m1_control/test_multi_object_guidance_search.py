#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from AAA_my_test.object_query_ablation_metrics.training_free_m1_control.run_multi_object_guidance_search import (
    WindowedDirectAttentionAblation,
    WindowedMultiObjectM1Guidance,
    apply_grouped_m1_ablation,
    block_diagonal_groups,
    direct_variant_directory,
    multi_object_flow_groups,
    variant_directory,
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


class MultiObjectFlowGroupsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.objects = {
            "object_A": torch.tensor([0, 1]),
            "object_B": torch.tensor([2, 3]),
        }

    @staticmethod
    def pairs(groups):
        return {
            (int(q), int(k))
            for queries, keys in groups
            for q in queries.tolist()
            for k in keys.tolist()
        }

    def test_m2_deletes_each_objects_incoming_context_including_other_object(self) -> None:
        groups, audit = multi_object_flow_groups(
            self.objects, 5, "M2", torch.device("cpu")
        )
        self.assertEqual(
            self.pairs(groups),
            {
                *{(q, k) for q in (0, 1) for k in (2, 3, 4)},
                *{(q, k) for q in (2, 3) for k in (0, 1, 4)},
            },
        )
        self.assertEqual(audit["deleted_pair_count_per_head"], 12)
        self.assertTrue(audit["object_specific_complements"])

    def test_m3_deletes_each_objects_outgoing_values_including_cross_object(self) -> None:
        groups, audit = multi_object_flow_groups(
            self.objects, 5, "M3", torch.device("cpu")
        )
        self.assertEqual(
            self.pairs(groups),
            {
                *{(q, k) for q in (2, 3, 4) for k in (0, 1)},
                *{(q, k) for q in (0, 1, 4) for k in (2, 3)},
            },
        )
        self.assertEqual(audit["deleted_pair_count_per_head"], 12)

    def test_overlap_pairs_are_subtracted_once_for_m2_and_m3(self) -> None:
        overlapping = {
            "object_A": torch.tensor([0, 1]),
            "object_B": torch.tensor([1, 2]),
        }
        for flow_id in ("M2", "M3"):
            with self.subTest(flow_id=flow_id):
                groups, audit = multi_object_flow_groups(
                    overlapping, 4, flow_id, torch.device("cpu")
                )
                pairs = self.pairs(groups)
                self.assertEqual(len(pairs), 7)
                self.assertEqual(audit["deleted_pair_count_per_head"], 7)
                self.assertEqual(audit["duplicate_pair_subtractions_prevented"], 1)

    def test_variant_directories_keep_m2_and_m3_separate(self) -> None:
        root = Path("/tmp/flow-test")
        m2 = variant_directory(
            root, "case", 47326, -0.5, 0, 9, "m2_multi_object_independent"
        )
        m3 = variant_directory(
            root, "case", 47326, -0.5, 0, 9, "m3_multi_object_independent"
        )
        self.assertIn("multi_object_independent__m2_all_time__top100", str(m2))
        self.assertIn("multi_object_independent__m3_all_time__top100", str(m3))
        self.assertNotEqual(m2, m3)


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


class DirectAblationTest(unittest.TestCase):
    def test_variant_directory_has_no_guidance_scale(self) -> None:
        root = Path("/tmp/direct-flow-test")
        result = direct_variant_directory(
            root, "case", 13248, 0, 9, "m2_multi_object_independent"
        )

        self.assertIn("direct_multi_object_independent__m2_all_time__top100", str(result))
        self.assertIn("denoise_00_09", str(result))
        self.assertNotIn("pag", str(result))

    def test_both_cfg_branches_are_directly_ablated_inside_window(self) -> None:
        class FakePipe:
            def __init__(self) -> None:
                self.model_fn = lambda *args, **kwargs: torch.tensor(1.0)

        class FakeAblator:
            entries = [{"block": 0, "head": 0}, {"block": 1, "head": 1}]
            dose_attention_mass = np.full((40, 2, 30, 24), np.nan, dtype=np.float32)
            record_dose = False
            mask_mode = "full_head_output"
            target_scope = "all_tokens"

            def __init__(self, pipe) -> None:
                self.pipe = pipe
                self.model_call_counts = {}
                self.modified_head_events = 0
                self._original_model_fn = None

            @staticmethod
            def _step(timestep) -> int:
                return int(timestep.item())

            def install(self) -> None:
                self._original_model_fn = self.pipe.model_fn

                def perturbed(*args, **kwargs):
                    step = self._step(kwargs["timestep"])
                    self.model_call_counts[step] = self.model_call_counts.get(step, 0) + 1
                    self.modified_head_events += len(self.entries)
                    return torch.tensor(2.0)

                self.pipe.model_fn = perturbed

            def remove(self) -> None:
                self.pipe.model_fn = self._original_model_fn

        pipe = FakePipe()
        ablator = FakeAblator(pipe)
        direct = WindowedDirectAttentionAblation(
            pipe, ablator, denoise_start=0, denoise_end=0, expected_steps=2
        )
        direct.install()
        try:
            active = [
                pipe.model_fn(timestep=torch.tensor(step), latents=torch.zeros(1)).item()
                for step in (0, 0)
            ]
            inactive = [
                pipe.model_fn(timestep=torch.tensor(step), latents=torch.zeros(1)).item()
                for step in (1, 1)
            ]
        finally:
            direct.remove()

        self.assertEqual(active, [2.0, 2.0])
        self.assertEqual(inactive, [1.0, 1.0])
        report = direct.audit()
        self.assertEqual(report["direct_calls_by_step"], {0: 2})
        self.assertEqual(report["modified_cfg_branches"], ["conditional", "unconditional"])
        self.assertEqual(report["modified_head_events"], 4)
        self.assertNotIn("perturbation_delta_l2_by_step", report)


class FullHeadAuditTest(unittest.TestCase):
    def test_all_token_ablator_does_not_require_block_diagonal_audit(self) -> None:
        class FakeAllTokenAblator:
            entries = [{"block": 0, "head": 0}, {"block": 1, "head": 1}]
            model_call_counts = {0: 1}
            modified_head_events = 2
            dose_attention_mass = np.full((40, 2, 30, 24), np.nan, dtype=np.float32)
            record_dose = False
            mask_mode = "full_head_output"
            target_scope = "all_tokens"

        guidance = WindowedMultiObjectM1Guidance(
            None,
            FakeAllTokenAblator(),
            cfg_scale=5.0,
            pag_scale=1.0,
            denoise_start=0,
            denoise_end=0,
        )
        guidance.pipeline_calls_by_step = {step: 2 for step in range(40)}
        guidance.guided_calls_by_step = {0: 1}
        report = guidance.audit()
        self.assertNotIn("block_diagonal", report)
        self.assertEqual(
            report["all_token_head_output_zero"],
            {
                "all_query_tokens": True,
                "selected_head_count": 2,
                "removed_flows": ["R->R", "C->R", "R->C", "C->C"],
            },
        )


if __name__ == "__main__":
    unittest.main()
