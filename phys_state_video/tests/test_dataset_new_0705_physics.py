from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset_new_0705.scene_generators_0705 import (
    EARTH_GRAVITY,
    F11_SCREEN_RIGHT_TRAVEL_ANGLE_DEG,
    _collision_vertical_extent,
    generate_scenario_blueprint,
    validate_blueprint_physics,
)
from scripts.dataset_new_0705.render_sim_0705 import build_object_phrase_bundle
from scripts.dataset_new_0705.audit_physv_initialization import audit_suite
from scripts.dataset_new_0705.generate_difficulty_pilot import (
    RAMP_INCLINE_CASES,
    RAMP_LENGTH_CONTROL_CASES,
    TABLE_ROLLOFF_CASES,
)


class DatasetNew0705PhysicsTests(unittest.TestCase):
    def test_generated_static_objects_are_grounded_and_dynamic_objects_use_gravity(self) -> None:
        for family_index in range(1, 13):
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
        for family_key in ("F5", "F6", "F9", "F10", "F12"):
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
        heights = (0.30, 0.58, 0.85, 1.12, 1.40)
        self.assertAlmostEqual(F11_SCREEN_RIGHT_TRAVEL_ANGLE_DEG, -48.0, places=5)
        speeds = []
        for index, table_height in enumerate(heights):
            blueprint = generate_scenario_blueprint(
                "F11",
                f"F11_table_{index}",
                20260717,
                "left_to_right",
                table_height_m=table_height,
                initial_speed_mps=1.25,
                travel_angle_deg=F11_SCREEN_RIGHT_TRAVEL_ANGLE_DEG,
            )
            self.assertAlmostEqual(blueprint.metadata["table_height_m"], table_height, places=5)
            self.assertEqual(blueprint.camera_key, "cam_09")
            self.assertGreater(blueprint.camera.eye[0], 0.0)
            self.assertLess(blueprint.camera.eye[1], -3.0)
            self.assertGreater(blueprint.camera.target[0], 0.0)
            self.assertAlmostEqual(blueprint.camera.target[1], 0.0, places=5)
            self.assertAlmostEqual(blueprint.camera.yfov_deg, 54.0, places=5)
            camera_horizontal_distance = math.hypot(
                blueprint.camera.eye[0] - blueprint.camera.target[0],
                blueprint.camera.eye[1] - blueprint.camera.target[1],
            )
            camera_downward_angle_deg = math.degrees(
                math.atan2(
                    abs(blueprint.camera.eye[2] - blueprint.camera.target[2]),
                    camera_horizontal_distance,
                )
            )
            self.assertLess(camera_downward_angle_deg, 12.0)
            self.assertAlmostEqual(blueprint.metadata["floor_restitution"], 0.62, places=5)
            mover = next(obj for obj in blueprint.objects if obj.name == "roller_0")
            self.assertTrue(mover.dynamic)
            self.assertAlmostEqual(
                blueprint.metadata["travel_angle_deg"],
                F11_SCREEN_RIGHT_TRAVEL_ANGLE_DEG,
                places=5,
            )
            self.assertGreater(mover.linear_velocity[0], 0.0)
            self.assertLess(mover.linear_velocity[1], 0.0)
            self.assertAlmostEqual(mover.mass, 2.50, places=5)
            self.assertAlmostEqual(mover.friction, 0.40, places=5)
            self.assertAlmostEqual(mover.restitution, 1.0, places=5)
            self.assertAlmostEqual(mover.linear_damping, 0.0, places=5)
            self.assertAlmostEqual(mover.angular_damping, 0.0, places=5)
            speeds.append(math.hypot(mover.linear_velocity[0], mover.linear_velocity[1]))
        self.assertTrue(all(abs(speed - speeds[0]) < 1e-6 for speed in speeds))

    def test_table_rolloff_pilot_cases_use_rightward_screen_trajectories(self) -> None:
        self.assertEqual(len(TABLE_ROLLOFF_CASES), 10)
        self.assertTrue(all(float(case["travel_angle_deg"]) < 0.0 for case in TABLE_ROLLOFF_CASES))
        primary_cases = [case for case in TABLE_ROLLOFF_CASES if case["angle_label"] == "sr048"]
        self.assertEqual([case["table_height_m"] for case in primary_cases], [0.30, 0.58, 0.85, 1.12, 1.40])
        self.assertTrue(
            all(
                float(case["travel_angle_deg"]) == F11_SCREEN_RIGHT_TRAVEL_ANGLE_DEG
                for case in primary_cases
            )
        )

    def test_table_rolloff_supports_multiple_velocity_directions(self) -> None:
        speed = 1.25
        radius = 0.14
        for angle_deg in (-48.0, -36.0, -30.0, -24.0, -18.0, -12.0):
            blueprint = generate_scenario_blueprint(
                "F11",
                f"F11_table_angle_{angle_deg}",
                20260717,
                "left_to_right",
                table_height_m=0.85,
                initial_speed_mps=speed,
                travel_angle_deg=angle_deg,
            )
            mover = next(obj for obj in blueprint.objects if obj.name == "roller_0")
            heading = math.radians(angle_deg)
            expected_vx = speed * math.cos(heading)
            expected_vy = speed * math.sin(heading)
            self.assertAlmostEqual(blueprint.metadata["travel_angle_deg"], angle_deg, places=5)
            self.assertAlmostEqual(mover.linear_velocity[0], expected_vx, places=5)
            self.assertAlmostEqual(mover.linear_velocity[1], expected_vy, places=5)
            self.assertAlmostEqual(
                math.hypot(mover.linear_velocity[0], mover.linear_velocity[1]),
                speed,
                places=5,
            )
            self.assertAlmostEqual(mover.angular_velocity[0], expected_vy / radius, places=5)
            self.assertAlmostEqual(mover.angular_velocity[1], -expected_vx / radius, places=5)

    def test_incline_control_uses_dynamic_floor_supported_bodies(self) -> None:
        expected_angles = [8.0, 16.0, 24.0, 33.0, 42.0]
        self.assertEqual(
            [float(case["ramp_angle_deg"]) for case in RAMP_INCLINE_CASES],
            expected_angles,
        )
        board_heights = []
        block_specs = []
        for angle_deg in expected_angles:
            blueprint = generate_scenario_blueprint(
                "F12",
                f"f12_angle_{angle_deg:03.0f}",
                20260817,
                "left_to_right",
                ramp_angle_deg=angle_deg,
            )
            self.assertEqual(blueprint.gravity, EARTH_GRAVITY)
            self.assertEqual(blueprint.camera_key, "cam_10")
            self.assertGreater(blueprint.camera.eye[0], 1.5)
            self.assertLess(blueprint.camera.eye[1], -5.0)
            self.assertGreater(blueprint.camera.target[0], 1.5)
            self.assertAlmostEqual(blueprint.camera.yfov_deg, 50.0, places=5)
            camera_horizontal_distance = math.hypot(
                blueprint.camera.eye[0] - blueprint.camera.target[0],
                blueprint.camera.eye[1] - blueprint.camera.target[1],
            )
            camera_downward_angle_deg = math.degrees(
                math.atan2(
                    abs(blueprint.camera.eye[2] - blueprint.camera.target[2]),
                    camera_horizontal_distance,
                )
            )
            self.assertGreater(camera_downward_angle_deg, 4.0)
            self.assertLess(camera_downward_angle_deg, 6.0)
            self.assertEqual(blueprint.surface_key, "painted_concrete_floor")
            self.assertEqual(blueprint.lighting_key, "hall_bright")
            self.assertAlmostEqual(blueprint.metadata["ramp_angle_deg"], angle_deg, places=5)
            self.assertEqual(blueprint.metadata["support_mode"], "dynamic_floor_supported_risers")
            self.assertTrue(all(obj.dynamic for obj in blueprint.objects))

            block = next(obj for obj in blueprint.objects if obj.name == "block_0")
            board = next(obj for obj in blueprint.objects if obj.name == "incline_board_0")
            risers = [obj for obj in blueprint.objects if obj.name.startswith("incline_riser_")]
            self.assertEqual(block.family_key, "wood_block")
            self.assertEqual(block.shape, "box")
            self.assertEqual(block.material_key, "wood_red")
            self.assertAlmostEqual(block.size["hx"], 0.20, places=5)
            self.assertAlmostEqual(block.size["hy"], 0.16, places=5)
            self.assertAlmostEqual(block.size["hz"], 0.14, places=5)
            self.assertAlmostEqual(block.mass, 2.50, places=5)
            self.assertAlmostEqual(block.friction, 0.12, places=5)
            self.assertAlmostEqual(block.restitution, 0.08, places=5)
            self.assertEqual(block.linear_velocity, (0.0, 0.0, 0.0))
            self.assertEqual(block.angular_velocity, (0.0, 0.0, 0.0))
            self.assertEqual(block.orientation_euler_deg, (0.0, angle_deg, 0.0))
            self.assertAlmostEqual(board.orientation_euler_deg[1], angle_deg, places=5)
            self.assertEqual(board.role, "dynamic_ramp")
            self.assertEqual(len(risers), 2)
            for riser in risers:
                self.assertEqual(riser.role, "dynamic_support")
                self.assertAlmostEqual(
                    riser.position[2], _collision_vertical_extent(riser), places=7
                )
            board_heights.append(board.position[2])
            block_specs.append((block.size, block.mass, block.friction, block.restitution))

        self.assertEqual(block_specs, [block_specs[0]] * len(block_specs))
        self.assertEqual(board_heights, sorted(board_heights))

    def test_validation_rejects_nonstandard_gravity(self) -> None:
        blueprint = generate_scenario_blueprint("F1", "invalid_gravity", seed=20260717)
        with self.assertRaisesRegex(ValueError, "require gravity"):
            validate_blueprint_physics(replace(blueprint, gravity=3.71))

    def test_left_and_right_cases_are_complete_mirrors(self) -> None:
        for family_index in range(1, 13):
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

    def test_full_difficulty_pilot_passes_initialization_audit(self) -> None:
        report = audit_suite(
            difficulty_seed_base=20260817,
            per_level=4,
            include_difficulty=True,
            include_v2v=False,
        )
        self.assertEqual(report["total_cases"], 32)
        self.assertEqual(report["failed_cases"], 0)


if __name__ == "__main__":
    unittest.main()
