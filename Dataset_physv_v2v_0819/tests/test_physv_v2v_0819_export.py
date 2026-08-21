"""Small structural tests for the PhysV V2V RigidBench-style exporter."""

from __future__ import annotations

import unittest
from collections import Counter

from Dataset_physv_v2v_0819.scripts.export_physv_v2v_0819_dataset import build_export_cases


class PhysvV2v0819ExportTests(unittest.TestCase):
    def test_control_groups_have_the_confirmed_composition(self) -> None:
        cases = build_export_cases()

        self.assertEqual(len(cases), 50)
        self.assertEqual(len({case.case_id for case in cases}), 50)
        self.assertEqual(
            Counter(case.source_group for case in cases),
            {
                "v2v_control": 30,
                "v2v_obstacle_ball_size": 5,
                "f11_table_height": 5,
                "f12_incline": 5,
                "f12_ramp_length": 5,
            },
        )

    def test_f11_direction_variants_are_excluded(self) -> None:
        f11_case_ids = [
            case.case_id
            for case in build_export_cases()
            if case.source_group == "f11_table_height"
        ]

        self.assertTrue(all(case_id.endswith("_sr048") for case_id in f11_case_ids))
        self.assertNotIn("difficulty_l2_f11_h085_sr030", f11_case_ids)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
