"""CPU-only tests: interval anchoring must not pretend to identify exact scale."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from align_observed_depth import (calibrate, crop_transform, main, morph3, ray_aabb_depth,
                                  resize_crop, scale_intervals, static_points,
                                  validate_reviewed_masks, world_rays)


class CropTests(unittest.TestCase):
    def test_anisotropic_resize_pixel_centers(self):
        tr = crop_transform((720, 1280))
        self.assertEqual(tr["processed_hw"], [294, 518])
        self.assertEqual(tr["crop_top"], 0)
        source_center = np.array([639.5, 359.5, 1.])
        np.testing.assert_allclose(tr["affine"] @ source_center, [258.5, 146.5, 1])
        self.assertNotEqual(tr["affine"][0, 0], tr["affine"][1, 1])

    def test_portrait_mask_matches_pil_resize_crop(self):
        mask = np.zeros((720, 360), dtype=bool)
        mask[210:390, 90:225] = True
        tr = crop_transform(mask.shape)
        self.assertEqual(tr["resized_hw"], [1036, 518])
        self.assertEqual(tr["crop_top"], 259)
        actual = resize_crop(mask, tr, is_mask=True)
        expected = Image.fromarray(mask.astype(np.uint8)*255).resize((518, 1036), Image.Resampling.NEAREST)
        expected = np.asarray(expected.crop((0, 259, 518, 777))) > 0
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (518, 518))
        self.assertEqual(actual.dtype, np.bool_)

    def test_reject_wrong_grid_and_nonbinary_mask(self):
        tr = crop_transform((20, 20))
        with self.assertRaises(ValueError):
            resize_crop(np.zeros((21, 20), bool), tr, is_mask=True)
        with self.assertRaises(ValueError):
            resize_crop(np.zeros((20, 20), np.float32), tr, is_mask=True)

    def test_morphology_uses_background_outside_image(self):
        mask = np.ones((5, 5), bool)
        self.assertEqual(morph3(mask, dilate=False).sum(), 9)
        point = np.zeros((5, 5), bool)
        point[2, 2] = True
        self.assertEqual(morph3(point, dilate=True).sum(), 9)


class RayTests(unittest.TestCase):
    def test_ray_parameter_is_z_not_euclidean_distance(self):
        origin = np.zeros(3)
        rays = np.array([[0., 0., 1.], [.1, 0., 1.]])
        near, far, hit = ray_aabb_depth(origin, rays, [0, 0, 5], [2, 2, 2])
        np.testing.assert_array_equal(hit, [True, True])
        np.testing.assert_allclose(near, [4, 4])
        np.testing.assert_allclose(far, [6, 6])

    def test_parallel_inside_outside_and_backward(self):
        rays = np.array([[0., 0., 1.], [0., 0., -1.], [1., 0., 0.]])
        near, far, hit = ray_aabb_depth([0, 0, 0], rays, [0, 0, 5], [2, 2, 2])
        np.testing.assert_array_equal(hit, [True, False, False])
        self.assertTrue(np.isfinite(near).all() and np.isfinite(far).all())
        _, _, hit_out = ray_aabb_depth([3, 0, 0], np.array([[0., 0., 1.]]), [0, 0, 5], [2, 2, 2])
        self.assertFalse(hit_out[0])

    def test_camera_inside_is_not_scale_anchor(self):
        _, _, hit = ray_aabb_depth([0, 0, 5], np.array([[0., 0., 1.]]), [0, 0, 5], [2, 2, 2])
        self.assertFalse(hit[0])

    def test_true_camera_world_transform(self):
        intrinsic = np.array([[2., 0, 1], [0, 2., 1], [0, 0, 1.]])
        rotation = np.array([[0., 1, 0], [-1., 0, 0], [0, 0, 1.]])
        camera_center = np.array([2., 3., 4.])
        rt = np.column_stack([rotation, -rotation @ camera_center])
        origin, rays = world_rays(intrinsic, rt, (3, 3))
        np.testing.assert_allclose(origin, camera_center)
        np.testing.assert_allclose(rays[1, 1], [0, 0, 1])
        np.testing.assert_allclose(rays[1, 2] @ rotation.T, [.5, 0, 1])


class IntervalTests(unittest.TestCase):
    def test_nonunique_scale_retains_width(self):
        r = scale_intervals(np.full(100, 2.), np.full(100, 3.))
        self.assertTrue(r["strict_feasible"] and r["robust_feasible"])
        self.assertEqual(r["robust_interval"], [2., 3.])
        self.assertEqual(r["scale_midpoint"], 2.5)
        self.assertEqual(r["interval_width"], 1.)
        self.assertEqual(r["full_interval_coverage"], 1.)
        self.assertFalse(r["scale_shift_fitted"])

    def test_inconsistent_intervals_fail(self):
        r = scale_intervals(np.array([1., 3.]), np.array([2., 4.]))
        self.assertFalse(r["robust_feasible"])
        self.assertIsNone(r["scale_midpoint"])
        self.assertEqual(r["full_interval_coverage"], 0.)

    def test_sparse_outlier_keeps_strict_failure_visible(self):
        lo, hi = np.full(100, 2.), np.full(100, 3.)
        lo[0], hi[0] = 4., 5.
        r = scale_intervals(lo, hi)
        self.assertFalse(r["strict_feasible"])
        self.assertTrue(r["robust_feasible"])
        self.assertAlmostEqual(r["full_interval_coverage"], .99)

    def test_empty_reversed_nonpositive_rejected(self):
        for lo, hi in (([], []), ([2.], [1.]), ([0.], [1.]), ([np.nan], [1.])):
            with self.assertRaises(ValueError):
                scale_intervals(np.array(lo), np.array(hi))


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.depth = np.ones((8, 9, 9), np.float64)
        self.masks = np.zeros((8, 1, 9, 9), bool)
        self.masks[:, :, 2:7, 2:7] = True
        self.k = np.array([[10., 0, 4], [0, 10., 4], [0, 0, 1.]])
        self.rt = np.column_stack([np.eye(3), np.zeros(3)])
        self.geometry = dict(positions_world=np.tile([0., 0., 5.], (8, 1, 1)),
                             size_m=np.array([[2., 2., 2.]]), camera_world_to_view=self.rt)

    def test_valid_sequence_coarse_interval_only(self):
        r = calibrate(self.depth, self.masks, self.geometry, self.k)
        self.assertTrue(r["accepted"])
        self.assertEqual(r["robust_interval"], [4., 6.])
        self.assertEqual(r["anchor_pixels"], 8*9)
        self.assertEqual(r["observed_center_depth_span_per_object_m"], [0.])

    def test_empty_foreground_frame_fails(self):
        self.masks[3] = False
        r = calibrate(self.depth, self.masks, self.geometry, self.k)
        self.assertFalse(r["accepted"])
        self.assertTrue(any("RGB3" in x for x in r["failure_reasons"]))

    def test_foreground_misses_box_fails(self):
        self.geometry["positions_world"][:, :, 0] = 5.
        r = calibrate(self.depth, self.masks, self.geometry, self.k)
        self.assertFalse(r["accepted"])
        self.assertEqual(r["ray_hit_fraction"], 0.)

    def test_depth_inconsistency_fails(self):
        self.depth[4:] = 5.
        r = calibrate(self.depth, self.masks, self.geometry, self.k)
        self.assertFalse(r["accepted"])
        self.assertFalse(r["robust_feasible"])

    def test_unreviewed_or_unaccepted_masks_fail(self):
        accepted = np.ones((8, 1), bool)
        union = self.masks.any(axis=1)
        with self.assertRaises(ValueError):
            validate_reviewed_masks(self.masks, union, accepted, False)
        accepted[2, 0] = False
        with self.assertRaises(ValueError):
            validate_reviewed_masks(self.masks, union, accepted, True)

    def test_backprojection_keeps_per_frame_pixel_identity(self):
        union = np.zeros((8, 9, 9), bool)
        union[0, 4, 4] = True
        points, valid = static_points(self.depth, union, self.k, self.rt, 5.)
        self.assertFalse(valid[0, 4, 4])
        self.assertTrue(valid[1, 4, 4])
        np.testing.assert_array_equal(points[~valid], 0)
        np.testing.assert_allclose(points[1, 4, 4], [0, 0, 5])
        np.testing.assert_allclose(points[1, 4, 5], [.5, 0, 5])
        self.assertEqual(int((~valid[0]).sum()), 9)



class FailureArtifactTests(unittest.TestCase):
    def test_unreviewed_cli_writes_only_failure_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"new_output"
            argv = ["align_observed_depth.py", "--input", "not_loaded.json",
                    "--geometry", "not_loaded.npz", "--masks", "not_loaded.npz",
                    "--raw", "not_loaded.npz", "--output", str(output)]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main()
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual([p.name for p in output.iterdir()], ["report.json"])
            report = json.loads((output/"report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["training_ready"])
            self.assertFalse(report["static_points_written"])


if __name__ == "__main__":
    unittest.main()

