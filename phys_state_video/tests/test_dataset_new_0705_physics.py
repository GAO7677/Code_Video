from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset_new_0705.scene_generators_0705 import (
    EARTH_GRAVITY,
    _collision_vertical_extent,
    generate_scenario_blueprint,
    validate_blueprint_physics,
)
from scripts.dataset_new_0705.render_sim_0705 import build_object_phrase_bundle


class DatasetNew0705PhysicsTests(unittest.TestCase):
    def test_generated_static_objects_are_grounded_and_dynamic_objects_use_gravity(self) -> None:
        for family_index in range(1, 12):
            family_key = f"F{family_index}"
            for sample_index in range(30):
                blueprint = generate_scenario_blueprint(
                    family_key=family_key,
                    sample_key=f"{family_key.lower()}_physics_{sample_index:03d}",
                    seed=20260717 + family_index * 1009 + sample_index,
                )
                self.assertEqual(blueprint.gravity, EARTH_GRAVITY)
                for obj in blueprint.objects:
                    if obj.dynamic:
                        self.assertGreater(obj.mass, 0.0)
                    else:
                        self.assertEqual(obj.mass, 0.0)
                        if obj.role.startswith("anchored_"):
                            continue
                        self.assertAlmostEqual(obj.position[2], _collision_vertical_extent(obj), places=7)

    def test_validation_rejects_zero_mass_dynamic_object(self) -> None:
        blueprint = generate_scenario_blueprint("F1", "invalid_mass", seed=20260717)
        invalid_object = replace(blueprint.objects[0], mass=0.0)
        with self.assertRaisesRegex(ValueError, "dynamic object mass must be positive"):
            validate_blueprint_physics(replace(blueprint, objects=(invalid_object,)))

    def test_visible_supports_are_dynamic_and_grounded(self) -> None:
        for family_key in ("F5", "F6", "F9", "F10"):
            blueprint = generate_scenario_blueprint(
                family_key, f"{family_key}_support", 20260717
            )
            supports = [obj for obj in blueprint.objects if "support" in obj.role]
            self.assertTrue(supports, family_key)
            for obj in supports:
                self.assertTrue(obj.dynamic, (family_key, obj.name))
                self.assertGreater(obj.mass, 0.0, (family_key, obj.name))
                self.assertAlmostEqual(
                    obj.position[2], _collision_vertical_extent(obj), places=7
                )

    def test_table_rolloff_cases_share_speed_across_heights(self) -> None:
        heights = (0.46, 0.68, 0.92)
        speeds = []
        for index, table_height in enumerate(heights):
            blueprint = generate_scenario_blueprint(
                "F11",
                f"F11_table_{index}",
                20260717,
                "left_to_right",
                table_height_m=table_height,
                initial_speed_mps=1.25,
            )
            self.assertAlmostEqual(blueprint.metadata["table_height_m"], table_height, places=5)
            mover = next(obj for obj in blueprint.objects if obj.name == "roller_0")
            self.assertTrue(mover.dynamic)
            self.assertAlmostEqual(mover.linear_velocity[0], 1.25, places=5)
            speeds.append(mover.linear_velocity[0])
        self.assertTrue(all(abs(speed - speeds[0]) < 1e-6 for speed in speeds))

    def test_validation_rejects_nonstandard_gravity(self) -> None:
        blueprint = generate_scenario_blueprint("F1", "invalid_gravity", seed=20260717)
        with self.assertRaisesRegex(ValueError, "require gravity"):
            validate_blueprint_physics(replace(blueprint, gravity=3.71))

    def test_left_and_right_cases_are_complete_mirrors(self) -> None:
        for family_index in range(1, 12):
            family_key = f"F{family_index}"
            left = generate_scenario_blueprint(family_key, "left", 20260718, "left_to_right")
            right = generate_scenario_blueprint(family_key, "right", 20260718, "right_to_left")
            self.assertEqual(left.metadata["direction_mode"], "left_to_right")
            self.assertEqual(right.metadata["direction_mode"], "right_to_left")
            for left_obj, right_obj in zip(left.objects, right.objects):
                self.assertEqual(left_obj.family_key, right_obj.family_key)
                self.assertEqual(left_obj.material_key, right_obj.material_key)
                self.assertEqual(left_obj.mass, right_obj.mass)
                self.assertAlmostEqual(left_obj.position[0], -right_obj.position[0])
                self.assertAlmostEqual(left_obj.position[1], right_obj.position[1])
                self.assertAlmostEqual(left_obj.position[2], right_obj.position[2])
                self.assertAlmostEqual(left_obj.linear_velocity[0], -right_obj.linear_velocity[0])
                self.assertAlmostEqual(left_obj.linear_velocity[1], right_obj.linear_velocity[1])
                self.assertAlmostEqual(left_obj.linear_velocity[2], right_obj.linear_velocity[2])

    def test_auto_direction_alternates_with_seed_parity(self) -> None:
        even = generate_scenario_blueprint("F1", "even", 20260718)
        odd = generate_scenario_blueprint("F1", "odd", 20260719)
        self.assertEqual(even.metadata["direction_mode"], "left_to_right")
        self.assertEqual(odd.metadata["direction_mode"], "right_to_left")

    def test_vertical_mode_removes_horizontal_velocity(self) -> None:
        for family_key in ("F5", "F8"):
            blueprint = generate_scenario_blueprint(family_key, "vertical", 20260718, "vertical")
            for obj in blueprint.objects:
                if obj.dynamic:
                    self.assertEqual(obj.linear_velocity[:2], (0.0, 0.0))

    def test_object_phrase_bundle_covers_dynamic_and_static_objects(self) -> None:
        blueprint = generate_scenario_blueprint("F4", "object_phrases", 20260718)
        bundle = build_object_phrase_bundle(blueprint)
        self.assertEqual(len(bundle["object_phrases"]), len(blueprint.objects))
        self.assertEqual(
            len(bundle["dynamic_object_phrases"]) + len(bundle["static_object_phrases"]),
            len(blueprint.objects),
        )
        self.assertGreater(len(bundle["static_object_phrases"]), 0)
        self.assertEqual(len(bundle["object_phrase_details"]), len(blueprint.objects))


if __name__ == "__main__":
    unittest.main()
