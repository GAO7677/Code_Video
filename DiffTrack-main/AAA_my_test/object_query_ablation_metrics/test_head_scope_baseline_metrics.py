from __future__ import annotations

import unittest

import numpy as np

from AAA_my_test.object_query_ablation_metrics.compute_head_scope_baseline_metrics import (
    assign_ranks,
    compare_frames,
)
from AAA_my_test.object_query_ablation_metrics.compute_head_scope_trajectory_metrics import (
    object_centers,
    object_trajectory_metrics,
)
from AAA_my_test.object_query_ablation_metrics.compute_head_scope_object_survival_metrics import (
    object_survival_metrics,
    pack_masks,
    rank_records as rank_survival_records,
    unpack_masks,
)


class HeadScopeBaselineMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = np.zeros((49, 16, 24, 3), dtype=np.uint8)
        self.target = np.zeros((49, 16, 24), dtype=bool)
        self.target[:, 3:9, 5:13] = True
        self.all_objects = self.target.copy()

    def test_identical_video_has_zero_effect(self) -> None:
        metrics = compare_frames(
            self.baseline, self.baseline.copy(), self.target, self.all_objects
        )
        self.assertEqual(metrics["impact_score_0_100"], 0.0)
        self.assertEqual(metrics["global"]["ssim_mean"], 1.0)
        self.assertEqual(metrics["global"]["mae_0_1"], 0.0)
        self.assertIsNone(metrics["global"]["psnr_db"])

    def test_target_change_is_local_and_positive(self) -> None:
        candidate = self.baseline.copy()
        candidate[self.target] = 255
        metrics = compare_frames(
            self.baseline, candidate, self.target, self.all_objects
        )
        self.assertGreater(metrics["impact_score_0_100"], 0.0)
        self.assertEqual(metrics["target_roi"]["mae_0_1"], 1.0)
        self.assertEqual(metrics["outside_objects"]["mae_0_1"], 0.0)

    def test_largest_score_receives_rank_one(self) -> None:
        records = []
        for value in (2.0, 7.0, 4.0):
            records.append(
                {
                    "metrics": {
                        "impact_score_0_100": value,
                        "global": {
                            "ssim_mean": 1.0 - value / 100.0,
                            "mae_0_1": value / 200.0,
                            "temporal_delta_mae_0_1": value / 300.0,
                        },
                        "target_roi": {
                            "mae_0_1": value / 100.0,
                            "temporal_delta_mae_0_1": value / 200.0,
                        },
                        "outside_objects": {
                            "mae_0_1": value / 400.0,
                            "temporal_delta_mae_0_1": value / 500.0,
                        },
                    }
                }
            )
        assign_ranks(records)
        self.assertEqual(records[1]["impact_rank_within_case_seed"], 1.0)
        self.assertEqual(records[0]["impact_rank_within_case_seed"], 3.0)
        self.assertEqual(records[1]["impact_percentile_within_case_seed"], 100.0)
        for category in (
            "global_appearance",
            "target_local",
            "temporal_appearance",
            "outside_spillover",
        ):
            self.assertEqual(
                records[1]["category_ranks_within_case_seed"][category], 1.0
            )
            self.assertGreater(
                records[1]["metrics"]["category_scores_0_100"][category],
                records[0]["metrics"]["category_scores_0_100"][category],
            )


class HeadScopeTrajectoryMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.slices = {"object_A": slice(0, 8), "object_B": slice(8, 16)}
        self.baseline = np.zeros((49, 16, 2), dtype=np.float32)
        self.baseline[..., 0] = np.arange(16, dtype=np.float32)[None]
        self.visibility = np.ones((49, 16), dtype=bool)
        self.baseline_centers, self.baseline_valid = object_centers(
            self.baseline, self.visibility, self.slices
        )

    def metrics(self, candidate: np.ndarray, visibility: np.ndarray) -> dict:
        centers, valid = object_centers(candidate, visibility, self.slices)
        return object_trajectory_metrics(
            candidate,
            visibility,
            centers["object_A"],
            valid["object_A"],
            self.baseline,
            self.visibility,
            self.baseline_centers["object_A"],
            self.baseline_valid["object_A"],
            self.slices["object_A"],
            100.0,
            4,
            0.8,
        )

    def test_center_ade_uses_tracks_not_pixels(self) -> None:
        candidate = self.baseline.copy()
        candidate[:, :8, 0] += 10.0
        metrics = self.metrics(candidate, self.visibility)
        self.assertTrue(metrics["quality_pass"])
        self.assertEqual(metrics["center_ade_px"], 10.0)
        self.assertEqual(metrics["center_ade_norm"], 0.1)
        self.assertEqual(metrics["common_center_coverage"], 1.0)
        self.assertEqual(metrics["track_retention_score_0_100"], 100.0)
        self.assertEqual(metrics["track_loss_score_0_100"], 0.0)

    def test_low_visibility_is_unrankable(self) -> None:
        candidate_visibility = self.visibility.copy()
        candidate_visibility[:, :8] = False
        candidate_visibility[:3, :8] = True
        metrics = self.metrics(self.baseline.copy(), candidate_visibility)
        self.assertFalse(metrics["quality_pass"])
        self.assertEqual(metrics["common_center_valid_frames"], 3)
        self.assertLess(metrics["common_center_coverage"], 0.8)
        self.assertAlmostEqual(metrics["track_loss_score_0_100"], 93.87755102)


class HeadScopeObjectSurvivalMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.masks = np.zeros((49, 32, 32), dtype=bool)
        self.masks[:, 8:24, 8:24] = True
        self.features = np.zeros((49, 4), dtype=np.float32)
        self.features[:, 0] = 1.0

    def test_mask_pack_round_trip(self) -> None:
        packed = pack_masks(self.masks)
        restored = unpack_masks(packed, self.masks.shape)
        np.testing.assert_array_equal(restored, self.masks)

    def test_identical_object_has_full_survival(self) -> None:
        metrics = object_survival_metrics(
            self.masks,
            self.masks,
            self.features,
            self.features,
            self.masks[0],
            0.8,
        )
        self.assertTrue(metrics["quality_pass"])
        self.assertEqual(metrics["survival_rate"], 1.0)
        self.assertEqual(metrics["disappearance_score_0_100"], 0.0)
        self.assertIsNone(metrics["first_sustained_loss_frame"])

    def test_identity_loss_is_measured_without_tracker(self) -> None:
        candidate = self.features.copy()
        candidate[10:, 0] = 0.0
        candidate[10:, 1] = 1.0
        metrics = object_survival_metrics(
            self.masks,
            self.masks,
            candidate,
            self.features,
            self.masks[0],
            0.8,
        )
        self.assertTrue(metrics["quality_pass"])
        self.assertEqual(metrics["alive_frame_count"], 10)
        self.assertEqual(metrics["first_sustained_loss_frame"], 10)
        self.assertAlmostEqual(metrics["disappearance_score_0_100"], 100 * 39 / 49)

    def test_retention_and_mask_absence_receive_independent_ranks(self) -> None:
        records = [
            {
                "variant_id": "identity_only",
                "metrics": {
                    "target_worst_disappearance_score_0_100": 90.0,
                    "target_worst_mask_absence_score_0_100": 0.0,
                },
            },
            {
                "variant_id": "mask_absent",
                "metrics": {
                    "target_worst_disappearance_score_0_100": 70.0,
                    "target_worst_mask_absence_score_0_100": 70.0,
                },
            },
        ]
        rank_survival_records(records)
        self.assertEqual(records[0]["disappearance_rank_within_case_seed"], 1)
        self.assertEqual(records[1]["mask_absence_rank_within_case_seed"], 1)


if __name__ == "__main__":
    unittest.main()
