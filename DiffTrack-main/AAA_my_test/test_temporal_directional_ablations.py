from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from AAA_my_test.run_legacy_ti2v_temporal_object_tube_ablations import (
    TEMPORAL_DIRECTIONAL_MODES,
    TemporalObjectTubeAblator,
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

    def test_temporal_mode_records_exact_removed_dose(self) -> None:
        torch.manual_seed(47326)
        q = torch.randn(1, 6, 256, dtype=torch.float64)
        k = torch.randn(1, 6, 256, dtype=torch.float64)
        v = torch.randn(1, 6, 256, dtype=torch.float64)
        qh = q.reshape(1, 6, 2, 128).transpose(1, 2)
        kh = k.reshape(1, 6, 2, 128).transpose(1, 2)
        vh = v.reshape(1, 6, 2, 128).transpose(1, 2)
        weights = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(128), dim=-1)
        original = reference_attention(q, k, v).reshape(1, 6, 2, 128)

        for mask_mode in TEMPORAL_DIRECTIONAL_MODES:
            with self.subTest(mask_mode=mask_mode):
                instance = TemporalObjectTubeAblator(
                    pipe=object(),
                    entries=[{"block": 0, "head": 1}],
                    query_points=np.asarray([[0.0, 0.0]], dtype=np.float32),
                    region_slices={},
                    pixel_hw=(1, 2),
                    target_scope="all_objects",
                    mask_mode=mask_mode,
                    region=None,
                    tracks=np.zeros((3, 1, 2), dtype=np.float32),
                    anchor_frames=np.arange(3, dtype=np.int64),
                    record_dose=True,
                )
                instance.active = True
                instance.current_grid = (3, 1, 2)
                instance.current_step = 0
                instance.current_cfg_call = 0
                instance._attention(q, k, v, reference_attention, block=0)

                groups = temporal_directional_groups(
                    [[0], [2], [4]], 2, mask_mode, torch.device("cpu")
                )
                target_parts = []
                contributions = []
                masses = []
                for targets, sources in groups:
                    selected_weights = weights[:, 1].index_select(1, targets)
                    selected_weights = selected_weights.index_select(2, sources)
                    selected_values = vh[:, 1].index_select(1, sources)
                    target_parts.append(targets)
                    contributions.append(selected_weights @ selected_values)
                    masses.append(selected_weights.sum(dim=-1))
                target_rows = torch.cat(target_parts)
                contribution = torch.cat(contributions, dim=1)
                expected_mass = float(torch.cat(masses, dim=1).mean())
                expected_removed = float(
                    torch.linalg.vector_norm(contribution, dim=-1).mean()
                )
                expected_original = float(
                    torch.linalg.vector_norm(
                        original[:, target_rows, 1], dim=-1
                    ).mean()
                )

                self.assertAlmostEqual(
                    float(instance.dose_attention_mass[0, 0, 0, 1]),
                    expected_mass,
                    places=5,
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
                self.assertEqual(
                    int(instance.dose_target_query_count[0, 0, 0, 1]),
                    int(target_rows.numel()),
                )

    def test_temporal_audit_rejects_incomplete_dose_coverage(self) -> None:
        instance = TemporalObjectTubeAblator(
            pipe=object(),
            entries=[{"block": 0, "head": 1}],
            query_points=np.asarray([[0.0, 0.0]], dtype=np.float32),
            region_slices={},
            pixel_hw=(1, 2),
            target_scope="all_objects",
            mask_mode="self_future",
            region=None,
            tracks=np.zeros((3, 1, 2), dtype=np.float32),
            anchor_frames=np.arange(3, dtype=np.int64),
            record_dose=True,
        )
        instance.query_token_indices = [0, 2, 4]
        instance.model_call_counts = {step: 2 for step in range(40)}
        instance.modified_head_events = 80
        with self.assertRaisesRegex(RuntimeError, "dose coverage mismatch"):
            instance.audit()


if __name__ == "__main__":
    unittest.main()
