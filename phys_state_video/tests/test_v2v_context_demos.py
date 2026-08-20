import unittest

from dataset_new_0705.generate_v2v_context_demos import (
    BOWL_BALL_CENTER_HEIGHT_ABOVE_BOTTOM_M,
    CONTEXT_FRAMES,
    CONTEXT_FRAME_OPTIONS,
    FPS,
    audit_v2v_case_initialization,
    build_demo_cases,
)
from dataset_new_0705.scene_generators_0705 import EARTH_GRAVITY


class V2VContextDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = build_demo_cases()

    def test_control_families_have_five_control_values(self):
        families = {}
        for case in self.cases:
            families.setdefault(case.family_key, []).append(case)
        self.assertEqual(len(self.cases), 35)
        self.assertEqual(set(families), {
            "V2V_GAP",
            "V2V_OBSTACLE",
            "V2V_OBSTACLE_SIZE",
            "V2V_BOWL",
            "V2V_PENDULUM",
            "V2V_SEESAW",
            "V2V_DOMINO",
        })
        self.assertTrue(all(len(cases) == 5 for cases in families.values()))
        self.assertTrue(all(len({case.controlled_value for case in cases}) == 5 for cases in families.values()))

    def test_physics_and_context_contract(self):
        for case in self.cases:
            blueprint = case.blueprint
            self.assertAlmostEqual(blueprint.gravity, EARTH_GRAVITY)
            self.assertEqual(blueprint.metadata["fps"], FPS)
            self.assertEqual(blueprint.metadata["context_frames"], CONTEXT_FRAMES)
            self.assertEqual(
                tuple(blueprint.metadata["context_frame_options"]),
                CONTEXT_FRAME_OPTIONS,
            )
            self.assertAlmostEqual(
                blueprint.metadata["context_duration_s"], CONTEXT_FRAMES / FPS
            )
            self.assertTrue(all(obj.mass > 0 for obj in blueprint.objects if obj.dynamic))
            self.assertTrue(all(obj.mass == 0 for obj in blueprint.objects if not obj.dynamic))

    def test_domino_trigger_is_fixed_across_spacing_group(self):
        cases = [case for case in self.cases if case.family_key == "V2V_DOMINO"]
        trigger_specs = []
        for case in cases:
            trigger = next(obj for obj in case.blueprint.objects if obj.name == "domino_trigger_ball")
            trigger_specs.append((trigger.position, trigger.mass, trigger.linear_velocity, trigger.restitution))
        self.assertEqual(len(set(trigger_specs)), 1)
        self.assertEqual(
            {case.blueprint.metadata["domino_gap_m"] for case in cases},
            {0.0, 0.045, 0.09, 0.135, 0.18},
        )

    def test_bowl_ball_starts_at_one_fixed_physical_height(self):
        cases = [case for case in self.cases if case.family_key == "V2V_BOWL"]
        for case in cases:
            metadata = case.blueprint.metadata
            ball = next(obj for obj in case.blueprint.objects if obj.name == "bowl_ball")
            self.assertAlmostEqual(
                ball.position[2] - float(metadata["bowl_bottom_z_m"]),
                BOWL_BALL_CENTER_HEIGHT_ABOVE_BOTTOM_M,
                places=5,
            )
            self.assertEqual(ball.linear_velocity, (0.0, 0.0, 0.0))
            self.assertEqual(ball.angular_velocity, (0.0, 0.0, 0.0))

    def test_pendulum_longest_bob_remains_above_the_floor_at_release(self):
        cases = [case for case in self.cases if case.family_key == "V2V_PENDULUM"]
        lowest_bob_z = min(
            next(obj for obj in case.blueprint.objects if obj.name == "pendulum_bob").position[2]
            for case in cases
        )
        self.assertGreater(lowest_bob_z, 0.60)

    def test_ball_appearance_mapping_is_fixed_within_each_control_group(self):
        for family_key in (
            "V2V_GAP",
            "V2V_OBSTACLE",
            "V2V_BOWL",
            "V2V_PENDULUM",
            "V2V_DOMINO",
        ):
            groups = {
                next(
                    obj.metadata["appearance_group"]
                    for obj in case.blueprint.objects
                    if obj.metadata.get("appearance_group")
                )
                for case in self.cases
                if case.family_key == family_key
            }
            self.assertEqual(len(groups), 1, family_key)

    def test_all_v2v_cases_pass_initialization_geometry_audit(self):
        for index, case in enumerate(self.cases):
            report = audit_v2v_case_initialization(case, seed=20260819 + index * 1009)
            self.assertTrue(report["passed"], case.case_id)
            self.assertEqual(
                [stage["stage"] for stage in report["stages"]],
                [
                    "post_creation",
                    "post_creation_contract",
                    "post_pre_roll",
                    "post_pre_roll_contract",
                    "video_frame_0",
                    "video_frame_0_contract",
                ],
            )


if __name__ == "__main__":
    unittest.main()
