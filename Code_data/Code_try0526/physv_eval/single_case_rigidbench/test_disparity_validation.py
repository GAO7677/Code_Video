"""Relative VDA disparity accepts zero; metric GT depth remains positive."""
import unittest

import numpy as np

from rigidbench.eval.score.depth import compute_si_mse
from . import common, si_mse, ate3d


class DisparityValidationTests(unittest.TestCase):
    def test_zero_disparity_uses_the_original_affine_score(self):
        disp = np.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]], np.float32)
        gt = 1 / (2 * disp + 1)
        actual = si_mse.score_from_disparity(gt, disp)['si_mse']
        self.assertEqual(actual, compute_si_mse(gt, disp))
        self.assertLess(abs(actual), 1e-12)

    def test_zero_disparity_reaches_ate3d_without_clipping(self):
        disp = np.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]], np.float32)
        gt = 1 / (2 * disp + 1)
        tracks = np.array([[[0, 0], [0, 0]]], np.float32)
        visible = np.ones((1, 2), bool)
        camera = dict(intrinsics=dict(fx=1, fy=1, cx=0, cy=0),
                      extrinsics=dict(location=[0, 0, 0], rotation=[1, 0, 0, 0]))
        trajectory = {'object_positions': np.array([[0, 0, -1], [0, 0, -1/3]])}
        result = ate3d.score_from_predictions(tracks, visible, gt, trajectory,
                    ['object'], camera, [0, 1], tracks, visible, disp)
        self.assertEqual(result['ate3d_n'], 2)
        self.assertLess(result['ate3d'], 1e-6)

    def test_positive_disparity_matches_the_existing_score_exactly(self):
        rng = np.random.default_rng(17)
        disp = rng.uniform(.1, 4, (3, 4, 5)).astype(np.float32)
        gt = rng.uniform(.2, 5, disp.shape).astype(np.float32)
        self.assertEqual(si_mse.score_from_disparity(gt, disp)['si_mse'],
                         compute_si_mse(gt, disp))

    def test_invalid_disparity_still_fails(self):
        for value in [np.full((2, 2, 2), -1.), np.full((2, 2, 2), np.nan),
                      np.full((2, 2, 2), np.inf), np.zeros((2, 2)),
                      np.zeros((2, 2, 2), bool)]:
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaises(ValueError):
                    common.as_disparity(value, 'pred_disparity')

    def test_zero_gt_depth_still_fails(self):
        with self.assertRaises(ValueError):
            common.as_depth(np.zeros((2, 2, 2)), 'gt_depth')


if __name__ == '__main__':
    unittest.main()
