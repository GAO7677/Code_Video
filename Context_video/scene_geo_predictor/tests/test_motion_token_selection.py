import unittest
import numpy as np
from motion_token_selection import select_tokens


class SelectionTest(unittest.TestCase):
    def test_budget_determinism_and_observation_response(self):
        xyz = np.random.default_rng(0).uniform(-2, 2, (4000, 3))
        times = np.arange(8)/30
        position = np.zeros((8, 1, 3))
        position[:, 0, 0] = times
        size = np.ones((1, 3))*.2
        a, report = select_tokens(xyz, position, size, times, 100)
        b, _ = select_tokens(xyz, position, size, times, 100)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(np.unique(a)), 100)
        self.assertGreater(report['local_tokens'], 0)
        self.assertGreaterEqual(report['global_tokens'], 50)
        position[:, 0, 1], position[:, 0, 0] = times, 0
        c, _ = select_tokens(xyz, position, size, times, 100)
        self.assertFalse(np.array_equal(a, c))

    def test_small_scene_and_stationary_object(self):
        xyz = np.zeros((4, 3))
        selected, _ = select_tokens(xyz, np.zeros((8, 1, 3)), np.ones((1, 3)), np.arange(8)/30)
        np.testing.assert_array_equal(selected, np.arange(4))


if __name__ == '__main__':
    unittest.main()
