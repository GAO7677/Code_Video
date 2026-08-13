from __future__ import annotations

import unittest

from AAA_my_test.object_query_ablation_metrics.training_free_m1_control import dashboard


class TrainingFreeM1DashboardTest(unittest.TestCase):
    def test_catalog_has_frozen_six_case_seed_rows(self) -> None:
        payload = dashboard.catalog()
        self.assertEqual(len(payload["cases"]), 3)
        self.assertEqual(payload["seeds"], [47326, 42])
        self.assertEqual(len(payload["rows"]), 6)
        self.assertEqual(payload["progress"]["interventions_total"], 48)
        self.assertEqual(payload["progress"]["baselines_total"], 6)

    def test_asset_rejects_coordinates_outside_frozen_matrix(self) -> None:
        self.assertIsNone(dashboard.asset("soft", "../invalid", 47326, -1.0))
        self.assertIsNone(
            dashboard.asset("contrast", dashboard._matrix()["cases"][0], 999, 1.0)
        )
        self.assertIsNone(
            dashboard.asset("contrast", dashboard._matrix()["cases"][0], 47326, 9.0)
        )

    def test_zero_value_reuses_same_seed_baseline(self) -> None:
        case = dashboard._matrix()["cases"][0]
        soft = dashboard.asset("soft", case, 47326, 0.0)
        contrast = dashboard.asset("contrast", case, 47326, 0.0)
        self.assertIsNotNone(soft)
        self.assertEqual(soft, contrast)


if __name__ == "__main__":
    unittest.main()
