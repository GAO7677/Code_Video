from __future__ import annotations

import unittest

import numpy as np

from AAA_my_test.object_query_ablation_metrics.compute_head_scope_baseline_metrics import (
    assign_ranks,
    compare_frames,
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
        records = [
            {"metrics": {"impact_score_0_100": value}}
            for value in (2.0, 7.0, 4.0)
        ]
        assign_ranks(records)
        self.assertEqual(records[1]["impact_rank_within_case_seed"], 1.0)
        self.assertEqual(records[0]["impact_rank_within_case_seed"], 3.0)
        self.assertEqual(records[1]["impact_percentile_within_case_seed"], 100.0)


if __name__ == "__main__":
    unittest.main()
