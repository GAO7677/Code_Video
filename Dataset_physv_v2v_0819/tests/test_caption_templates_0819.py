"""Tests for explicit and variable-blind PhysV captions."""

from __future__ import annotations

import unittest

from Dataset_physv_v2v_0819.scripts.caption_templates_0819 import build_caption_bundle


class CaptionTemplateTests(unittest.TestCase):
    def test_ramp_has_explicit_and_abstract_variants(self) -> None:
        metadata = {
            "family_key": "F12",
            "task_type": "incline_release",
            "control": {"value_label": "24 deg", "value": 24.0, "units": "deg"},
        }

        bundle = build_caption_bundle(metadata)

        self.assertIn("24 degrees", bundle["specific"])
        self.assertNotIn("24", bundle["abstract"])
        self.assertIn("inclined surface", bundle["abstract"])

    def test_all_control_families_produce_nonempty_captions(self) -> None:
        cases = (
            ("V2V_GAP", "gap_rolloff"),
            ("V2V_OBSTACLE", "obstacle_collision"),
            ("V2V_BOWL", "bowl_descent"),
            ("V2V_PENDULUM", "pendulum_swing"),
            ("V2V_PENDULUM_CABINET", "pendulum_cabinet_collision"),
            ("V2V_SEESAW", "seesaw_rotation"),
            ("V2V_DOMINO", "domino_chain"),
            ("F11", "table_rolloff"),
        )
        for family_key, task_type in cases:
            with self.subTest(family_key=family_key):
                bundle = build_caption_bundle(
                    {
                        "family_key": family_key,
                        "task_type": task_type,
                        "control": {"value_label": "0.50 m", "value": 0.5, "units": "m"},
                    }
                )
                self.assertTrue(bundle["specific"])
                self.assertTrue(bundle["abstract"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
