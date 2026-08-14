from __future__ import annotations

import unittest

from AAA_my_test.object_query_ablation_metrics.training_free_m1_control import (
    multi_object_search_dashboard as dashboard,
)


class MultiObjectSearchDashboardTest(unittest.TestCase):
    def test_head_zero_matrix_is_limited_to_the_five_controlled_cases(self) -> None:
        payload = dashboard.catalog()

        self.assertEqual(payload["progress"]["head_zero_expected"], 400)
        self.assertEqual(len(payload["head_zero_cases"]), 5)
        for row in payload["rows"]:
            expected = 16 if row["case"] in payload["head_zero_cases"] else 0
            self.assertEqual(len(row["head_zero_records"]), expected)
            self.assertEqual(row["progress"]["head_zero_expected"], expected)

    def test_two_guidance_families_have_distinct_paths(self) -> None:
        m1 = dashboard._variant(-0.5, 0, 9, "m1_multi")
        head_zero = dashboard._variant(-0.5, 0, 9, "head_zero")

        self.assertEqual(
            m1,
            "multi_object_blockdiag__m1_all_time__top100__pagm0p5__denoise_00_09",
        )
        self.assertEqual(
            head_zero,
            "all_token__full_head_output_zero__top100__pagm0p5__denoise_00_09",
        )

    def test_page_explains_and_renders_the_controlled_comparison(self) -> None:
        page = dashboard.page()

        self.assertIn("FULL-HEAD ZERO · ALL FLOWS", page)
        self.assertIn("M1 MULTI · Rᵢ→Rᵢ ONLY", page)
        self.assertIn("card('head_zero',row,z)", page)
        self.assertIn("Baseline 只显示一次", page)

    def test_asset_rejects_head_zero_for_an_unselected_case(self) -> None:
        payload = dashboard.catalog()
        row = next(
            row
            for row in payload["rows"]
            if row["case"] not in dashboard.HEAD_ZERO_CASES
        )

        self.assertIsNone(
            dashboard.asset("head_zero", row["case"], row["seed"], -1.0, 0, 4)
        )


if __name__ == "__main__":
    unittest.main()
