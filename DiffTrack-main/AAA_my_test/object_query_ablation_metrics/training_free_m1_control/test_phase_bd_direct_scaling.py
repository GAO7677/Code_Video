from __future__ import annotations

import argparse
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from AAA_my_test.object_query_ablation_metrics.training_free_m1_control.run_m1_direct_scaling_phase_bd import (
    M1DirectScalingAblator,
    SAM2FullMaskM1DirectScalingAblator,
    TIME_SCOPE_TO_MASK,
    alpha_at_step,
    output_directory,
    validate_denoising_window,
    validate_phase_configuration,
)
from AAA_my_test.object_query_ablation_metrics.full_mask_signature_regions import (
    SignaturePartition,
)
from AAA_my_test.object_query_ablation_metrics.training_free_m1_control.select_test5_phase_b_alpha import (
    center_ade_d0,
    rank_two,
    video_mse,
)
def configuration(**overrides) -> argparse.Namespace:
    values = {
        "sampling_steps": 40,
        "cfg_scale": 5.0,
        "alpha": 0.1,
        "denoise_start": 0,
        "denoise_end": 39,
        "record_dose": True,
        "phase_label": "phase_b",
        "time_scope": "all_time",
        "case": "case_x",
        "seed": 47326,
        "region": "object_A",
        "output_root": Path("/tmp/phase_bd_test"),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch, query_count, width = q.shape
    num_heads, head_dim = 2, width // 2
    qh = q.reshape(batch, query_count, num_heads, head_dim).transpose(1, 2)
    kh = k.reshape(batch, k.shape[1], num_heads, head_dim).transpose(1, 2)
    vh = v.reshape(batch, v.shape[1], num_heads, head_dim).transpose(1, 2)
    weights = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(head_dim), dim=-1)
    return (weights @ vh).transpose(1, 2).reshape(batch, query_count, width)


class PhaseBDDirectScalingTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(47326)
        self.q = torch.randn(1, 6, 256, dtype=torch.float64)
        self.k = torch.randn(1, 6, 256, dtype=torch.float64)
        self.v = torch.randn(1, 6, 256, dtype=torch.float64)
    def ablator(
        self,
        time_scope: str,
        denoise_start: int = 0,
        denoise_end: int = 39,
    ) -> M1DirectScalingAblator:
        instance = M1DirectScalingAblator(
            pipe=object(),
            entries=[{"block": 0, "head": 1}],
            query_points=np.asarray([[0.0, 0.0]], dtype=np.float32),
            region_slices={"object_A": slice(0, 1)},
            pixel_hw=(1, 2),
            target_scope="single_object",
            mask_mode=TIME_SCOPE_TO_MASK[time_scope],
            region="object_A",
            tracks=np.zeros((3, 1, 2), dtype=np.float32),
            anchor_frames=np.arange(3, dtype=np.int64),
            record_dose=False,
            alpha=0.25,
            time_scope=time_scope,
            denoise_start=denoise_start,
            denoise_end=denoise_end,
        )
        instance.active = True
        instance.current_grid = (3, 1, 2)
        instance.current_step = 0
        instance.current_cfg_call = 0
        return instance

    def test_denoising_window_is_inclusive(self) -> None:
        values = [alpha_at_step(0.25, step, 2, 4) for step in range(7)]
        self.assertEqual(values, [0.0, 0.0, 0.25, 0.25, 0.25, 0.0, 0.0])

    def test_invalid_denoising_windows_are_rejected(self) -> None:
        for start, end in ((-1, 9), (10, 9), (0, 40)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                validate_denoising_window(start, end)

    def test_phase_b_controls_are_frozen(self) -> None:
        validate_phase_configuration(configuration(alpha=0.1))
        validate_phase_configuration(configuration(alpha=0.25))
        for invalid in (
            {"alpha": 0.5},
            {"time_scope": "future"},
            {"denoise_end": 9},
            {"record_dose": False},
            {"cfg_scale": 4.0},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_phase_configuration(configuration(**invalid))

    def test_phase_d_accepts_only_all_time_windows(self) -> None:
        validate_phase_configuration(
            configuration(
                phase_label="phase_d",
                alpha=0.25,
                time_scope="all_time",
                denoise_start=0,
                denoise_end=9,
            )
        )
        with self.assertRaises(ValueError):
            validate_phase_configuration(
                configuration(
                    phase_label="phase_d",
                    alpha=0.25,
                    time_scope="future",
                    denoise_start=0,
                    denoise_end=39,
                )
            )

    def test_time_scopes_map_to_existing_exact_masks(self) -> None:
        self.assertEqual(
            TIME_SCOPE_TO_MASK,
            {"all_time": "self_only"},
        )

    def test_all_time_adds_only_exact_post_softmax_contribution(self) -> None:
        baseline = reference_attention(self.q, self.k, self.v)
        qh = self.q.reshape(1, 6, 2, 128).transpose(1, 2)
        kh = self.k.reshape(1, 6, 2, 128).transpose(1, 2)
        vh = self.v.reshape(1, 6, 2, 128).transpose(1, 2)
        weights = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(128), dim=-1)
        rows = torch.tensor([0, 2, 4])

        selected_weights = weights[:, 1].index_select(1, rows)
        selected_weights = selected_weights.index_select(2, rows)
        selected_values = vh[:, 1].index_select(1, rows)
        contribution = selected_weights @ selected_values
        expected_heads = baseline.reshape(1, 6, 2, 128).clone()
        expected_heads[:, rows, 1, :] += 0.25 * contribution

        actual = self.ablator("all_time")._attention(
            self.q, self.k, self.v, reference_attention, block=0
        )
        torch.testing.assert_close(
            actual,
            expected_heads.reshape_as(baseline),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_window_outside_step_is_exact_output_noop(self) -> None:
        instance = self.ablator("all_time", denoise_start=10, denoise_end=19)
        actual = instance._attention(
            self.q, self.k, self.v, reference_attention, block=0
        )
        self.assertTrue(torch.equal(actual, reference_attention(self.q, self.k, self.v)))
        self.assertEqual(instance.applied_head_events, 0)

    def test_output_identity_contains_scope_gain_and_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = configuration(
                phase_label="phase_d",
                output_root=Path(directory),
                alpha=0.25,
                time_scope="all_time",
                denoise_start=0,
                denoise_end=19,
            )
            output = output_directory(args)
            self.assertIn("phase_d/case_x/seed_47326", output.as_posix())
            self.assertIn("m1_all_time__top100", output.name)
            self.assertIn("alpha_0p25", output.name)
            self.assertIn("denoise_00_19", output.name)

    def test_full_mask_rows_use_every_frozen_mask_cell(self) -> None:
        rows = (0, 1, 880, 11439)
        partition = SignaturePartition(
            object_names=("object_A",),
            anchor_frames=tuple(range(0, 49, 4)),
            grid=(13, 22, 40),
            signature_rows={1: rows},
            signature_rows_by_time={
                1: tuple(
                    tuple(value for value in rows if value // 880 == time)
                    for time in range(13)
                )
            },
            occupancy_by_time=np.zeros((13, 1, 22, 40), dtype=np.float32),
        )
        instance = SAM2FullMaskM1DirectScalingAblator(
            pipe=object(),
            entries=[{"block": 0, "head": 1}],
            query_points=np.asarray([[0.0, 0.0]], dtype=np.float32),
            region_slices={"object_A": slice(0, 1)},
            pixel_hw=(704, 1280),
            target_scope="single_object",
            mask_mode="self_only",
            region="object_A",
            tracks=np.zeros((49, 1, 2), dtype=np.float32),
            anchor_frames=np.arange(13, dtype=np.int64) * 4,
            record_dose=False,
            alpha=0.1,
            time_scope="all_time",
            denoise_start=0,
            denoise_end=39,
            partition=partition,
        )
        instance.current_grid = (13, 22, 40)
        actual = instance._rows(torch.device("cpu"))
        self.assertEqual(actual.tolist(), list(rows))
        self.assertEqual(
            instance.query_token_indices_by_latent_frame,
            [[0, 1], [880], *([[]] * 10), [11439]],
        )

    def test_mse_is_zero_only_for_identical_frames(self) -> None:
        reference = np.zeros((2, 3, 4, 3), dtype=np.uint8)
        self.assertEqual(video_mse(reference, reference.copy()), 0.0)
        candidate = reference.copy()
        candidate[0, 0, 0, 0] = 255
        self.assertAlmostEqual(video_mse(candidate, reference), 1.0 / candidate.size)

    def test_unit_free_rank_prefers_lower_metric(self) -> None:
        self.assertEqual(rank_two(0.1, 0.2), (1.0, 2.0))
        self.assertEqual(rank_two(0.2, 0.1), (2.0, 1.0))
        self.assertEqual(rank_two(0.1, 0.1), (1.5, 1.5))

    def test_trajectory_gate_retains_and_normalizes_center_ade(self) -> None:
        gt = np.zeros((49, 2, 2), dtype=np.float32)
        candidate = gt.copy()
        candidate[..., 0] = 2.0
        visible = np.ones((49, 2), dtype=bool)
        ade, valid, rate = center_ade_d0(
            candidate, visible, gt, visible, d0=10.0
        )
        self.assertAlmostEqual(ade, 0.2)
        self.assertEqual(valid, 49)
        self.assertEqual(rate, 1.0)

    def test_trajectory_gate_rejects_too_few_common_frames(self) -> None:
        tracks = np.zeros((49, 2, 2), dtype=np.float32)
        visible = np.zeros((49, 2), dtype=bool)
        visible[:24] = True
        ade, valid, _ = center_ade_d0(
            tracks, visible, tracks, visible, d0=10.0
        )
        self.assertIsNone(ade)
        self.assertEqual(valid, 24)


if __name__ == "__main__":
    unittest.main()
