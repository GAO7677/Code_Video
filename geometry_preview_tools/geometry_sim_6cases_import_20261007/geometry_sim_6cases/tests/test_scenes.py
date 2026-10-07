"""Registration and state smoke tests; these do not certify physical accuracy."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import scenes
from physics import SphereWorld3


class SceneSmokeTests(unittest.TestCase):
    def test_only_six_requested_families(self):
        expected = [
            '01_cylinder_offset', '02_barrier_length', '03_rail_gap',
            '04_hump_height', '05_hole_offset', '06_table_edge',
        ]
        self.assertEqual(len(scenes.FAMILIES), 6)
        self.assertEqual([build(values[0]).key for build, values in scenes.FAMILIES], expected)
        self.assertFalse(hasattr(scenes, 'build_bridge'))
        for build, values in scenes.FAMILIES:
            self.assertEqual(len(values), 3)
            self.assertEqual(len(set(values)), 3)
            for value in values:
                scene = build(value)
                self.assertEqual(scene.value, value)
                self.assertEqual(scene.metadata()['control']['value'], value)
                if not isinstance(scene.world, SphereWorld3):
                    self.assertFalse(any(b.pinned for b in scene.world.bodies))
                    self.assertNotIn('bridge', [b.name for b in scene.world.bodies])

    def test_eight_frame_prefix_is_finite_and_shared(self):
        for build, values in scenes.FAMILIES:
            prefixes = []
            cameras = []
            for value in values:
                scene = build(value)
                cameras.append(scene.camera)
                positions = []
                for frame in range(8):
                    state = scene.world.states()
                    for body in state.values():
                        self.assertTrue(np.isfinite(body['position']).all())
                        self.assertTrue(np.isfinite(body['velocity']).all())
                    positions.append(np.asarray(state[scene.target]['position']).copy())
                    if frame < 7:
                        for _ in range(24):
                            scene.world.step(1 / 720)
                prefixes.append(np.asarray(positions))
            self.assertTrue(all(camera == cameras[0] for camera in cameras))
            for prefix in prefixes[1:]:
                np.testing.assert_allclose(prefix, prefixes[0], rtol=0, atol=1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
