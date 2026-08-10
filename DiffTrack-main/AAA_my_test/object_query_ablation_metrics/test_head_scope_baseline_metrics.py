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

    def test_low_visibility_is_unrankable(self) -> None:
        candidate_visibility = self.visibility.copy()
        candidate_visibility[:, :8] = False
        candidate_visibility[:3, :8] = True
        metrics = self.metrics(self.baseline.copy(), candidate_visibility)
        self.assertFalse(metrics["quality_pass"])
        self.assertEqual(metrics["common_center_valid_frames"], 3)
        self.assertLess(metrics["common_center_coverage"], 0.8)


if __name__ == "__main__":
    unittest.main()
