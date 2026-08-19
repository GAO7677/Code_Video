import unittest

from dataset_new_0705.generate_v2v_context_demos import (
    CONTEXT_FRAMES,
    FPS,
    build_demo_cases,
)
from dataset_new_0705.scene_generators_0705 import EARTH_GRAVITY


class V2VContextDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = build_demo_cases()

    def test_six_families_have_three_control_values(self):
        families = {}
        for case in self.cases:
            families.setdefault(case.family_key, []).append(case)
        self.assertEqual(len(self.cases), 18)
        self.assertEqual(set(families), {
            "V2V_GAP",
            "V2V_OBSTACLE",
            "V2V_BOWL",
            "V2V_PENDULUM",
            "V2V_SEESAW",
            "V2V_DOMINO",
        })
        self.assertTrue(all(len(cases) == 3 for cases in families.values()))
        self.assertTrue(all(len({case.controlled_value for case in cases}) == 3 for cases in families.values()))

    def test_physics_and_context_contract(self):
        for case in self.cases:
            blueprint = case.blueprint
            self.assertAlmostEqual(blueprint.gravity, EARTH_GRAVITY)
            self.assertEqual(blueprint.metadata["fps"], FPS)
            self.assertEqual(blueprint.metadata["context_frames"], CONTEXT_FRAMES)
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
            {0.02, 0.06, 0.12},
        )


if __name__ == "__main__":
    unittest.main()
