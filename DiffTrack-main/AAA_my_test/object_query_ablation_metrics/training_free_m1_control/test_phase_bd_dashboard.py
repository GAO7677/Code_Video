from __future__ import annotations

import unittest

from AAA_my_test.object_query_ablation_metrics.training_free_m1_control import (
    phase_bd_dashboard,
)


class PhaseBDDashboardTest(unittest.TestCase):
    def test_catalog_matches_frozen_test5_matrix(self) -> None:
        payload = phase_bd_dashboard.catalog()
        self.assertEqual(payload["progress"]["case_count"], 20)
        self.assertEqual(payload["progress"]["sample_count"], 100)
        self.assertEqual(payload["progress"]["phase_b_total"], 200)
        self.assertEqual(payload["progress"]["phase_d_total"], 200)
        self.assertTrue(all(len(case["seeds"]) == 5 for case in payload["cases"]))

    def test_asset_rejects_coordinates_outside_manifest(self) -> None:
        self.assertIsNone(phase_bd_dashboard.asset("../invalid", 90094, "baseline"))
        case = phase_bd_dashboard.catalog()["case_names"][0]
        self.assertIsNone(phase_bd_dashboard.asset(case, 999, "baseline"))
        self.assertIsNone(phase_bd_dashboard.asset(case, 90094, "../../generated"))

    def test_ready_assets_are_resolvable(self) -> None:
        payload = phase_bd_dashboard.catalog()
        ready = next(
            (row, item)
            for case in payload["cases"]
            for row in case["seeds"]
            for item in row["phase_b"]
            if item["ready"]
        )
        row, item = ready
        result = phase_bd_dashboard.asset(row["case"], row["seed"], item["asset_id"])
        self.assertIsNotNone(result)
        self.assertTrue(result.is_file())


if __name__ == "__main__":
    unittest.main()
