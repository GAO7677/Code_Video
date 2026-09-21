import unittest

import cv2
import numpy as np

from context_rgb_pybullet_common import (
    estimate_metric_centers,
    metric_centers_from_sphere_circles,
    rgb_motion_circle_prompt,
    regularize_sphere_masks,
    robust_terminal_velocity,
    static_point_cloud,
)
from fit_context_collision_primitives import (
    fit_aperture,
    fit_deflector,
    fit_ground,
    fit_support_edge,
)


class ContextRgbPybulletTests(unittest.TestCase):
    def test_metric_center_recovery_from_sphere_silhouette_and_relative_depth(self):
        frame_count, height, width = 8, 128, 128
        focal, radius, relative_scale = 200.0, 0.11, 2.5
        intrinsic = np.array([[focal, 0, 63.5], [0, focal, 63.5], [0, 0, 1.0]])
        world_to_camera = np.column_stack((np.eye(3), np.zeros(3)))
        masks = np.zeros((frame_count, height, width), dtype=bool)
        depth = np.empty((frame_count, height, width), dtype=np.float64)
        expected = []
        for index, z_approx in enumerate(np.linspace(2.5, 2.2, frame_count)):
            radius_px = focal * radius / z_approx
            cv2.circle(masks[index].view(np.uint8), (64, 64), int(round(radius_px)), 1, -1)
            equivalent_radius = np.sqrt(float(masks[index].sum()) / np.pi)
            center_z = focal * radius / equivalent_radius
            depth[index] = (center_z - radius) / relative_scale
            expected.append([0.5 * center_z / focal, 0.5 * center_z / focal, center_z])

        centers, report = estimate_metric_centers(
            depth, masks, intrinsic, world_to_camera, radius_m=radius
        )

        np.testing.assert_allclose(centers, expected, atol=1e-6)
        self.assertAlmostEqual(report["metric_scale"], relative_scale, places=5)
        self.assertEqual(sum(report["scale_inliers"]), frame_count)

    def test_velocity_fit_uses_multiframe_inliers(self):
        times = np.arange(8, dtype=np.float64) / 30.0
        centers = np.column_stack(
            (
                1.0 + 2.0 * times + 0.5 * 0.6 * times**2,
                -0.3 + 0.4 * times,
                np.full(8, 0.11),
            )
        )
        centers[3] += [0.15, -0.10, 0.08]
        expected = np.array([2.0 + 0.6 * times[-1], 0.4, 0.0])

        velocity, report = robust_terminal_velocity(centers, times, degree=2)

        np.testing.assert_allclose(velocity, expected, atol=1e-8)
        self.assertEqual(report["excluded_frames"], [3])
        self.assertEqual(len(report["inlier_frames"]), 7)

    def test_rgb_motion_circle_prompt_rejects_elongated_shadow(self):
        frames = np.full((8, 128, 128, 3), 100, dtype=np.uint8)
        for index in range(8):
            x = 25 + 6 * index
            cv2.rectangle(frames[index], (90, 30), (110, 100), (40, 80, 180), -1)
            cv2.ellipse(frames[index], (x, 84), (13, 4), 0, 0, 360, (55, 55, 55), -1)
            cv2.circle(frames[index], (x, 75), 10, (190, 80, 60), -1)

        box, report = rgb_motion_circle_prompt(frames)
        center = 0.5 * (box[:2] + box[2:])

        np.testing.assert_allclose(center, [67, 75], atol=3.0)
        self.assertGreaterEqual(report["candidate_count"], 1)
        self.assertFalse(report["gt_or_future_used"])

    def test_sphere_regularization_removes_tracked_shadow(self):
        frames = np.full((8, 128, 128, 3), 110, dtype=np.uint8)
        sam_masks = np.zeros((8, 128, 128), dtype=bool)
        for index in range(8):
            center = (30 + 5 * index, 70)
            cv2.circle(frames[index], center, 11, (190, 70, 55), -1)
            cv2.ellipse(frames[index], (center[0] + 7, 82), (17, 5), 0, 0, 360, (55, 55, 55), -1)
            cv2.circle(sam_masks[index].view(np.uint8), center, 12, 1, -1)
            cv2.ellipse(sam_masks[index].view(np.uint8), (center[0] + 7, 82), (17, 5), 0, 0, 360, 1, -1)

        circles, regularized, report = regularize_sphere_masks(frames, sam_masks)

        np.testing.assert_allclose(circles[:, 0], 30 + 5 * np.arange(8), atol=2.0)
        np.testing.assert_allclose(circles[:, 1], 70, atol=2.0)
        np.testing.assert_allclose(circles[:, 2], 11, atol=2.0)
        self.assertTrue(np.all(regularized.sum(axis=(1, 2)) < sam_masks.sum(axis=(1, 2))))
        self.assertFalse(report["future_or_gt_used"])

    def test_metric_circle_centers_use_horizontal_support_estimated_from_context(self):
        intrinsic = np.array([[200.0, 0, 64.0], [0, 200.0, 64.0], [0, 0, 1.0]])
        world_to_camera = np.column_stack((np.eye(3), np.zeros(3)))
        x = np.linspace(-0.2, 0.3, 8)
        z = 2.0
        circles = np.column_stack((64 + 200 * x / z, np.full(8, 74.0), np.full(8, 11.0)))

        centers, report = metric_centers_from_sphere_circles(
            circles, intrinsic, world_to_camera, radius_m=0.11
        )

        np.testing.assert_allclose(centers[:, 0], x, atol=1e-8)
        np.testing.assert_allclose(centers[:, 2], z, atol=1e-8)
        self.assertTrue(report["horizontal_support_regularization"])
        self.assertFalse(report["future_or_gt_used"])

    def test_deflector_primitive_from_colored_points(self):
        rng = np.random.default_rng(3)
        yaw = np.deg2rad(58.0)
        normal = np.array([np.cos(yaw), np.sin(yaw)])
        long_axis = np.array([-np.sin(yaw), np.cos(yaw)])
        local_short = rng.uniform(-0.045, 0.045, 5000)
        local_long = rng.uniform(-0.72, 0.72, 5000)
        xy = np.array([0.65, 0.0]) + local_short[:, None] * normal + local_long[:, None] * long_axis
        points = np.column_stack((xy, rng.uniform(0.0, 0.5, 5000)))
        colors = np.tile(np.array([35, 80, 190], dtype=np.uint8), (len(points), 1))

        primitives, report = fit_deflector(points, colors)

        primitive = primitives[0]
        self.assertAlmostEqual(primitive["position_m"][0], 0.65, delta=0.03)
        self.assertAlmostEqual(primitive["size"]["half_extents_m"][1], 0.72, delta=0.04)
        self.assertLess(abs(primitive["orientation_yaw_deg"] - 58.0), 3.0)
        self.assertGreater(report["selected_point_count"], 4000)

    def test_support_gap_primitive_from_elevated_plane(self):
        rng = np.random.default_rng(4)
        left = np.column_stack((
            rng.uniform(-1.5, -0.08, 7000),
            rng.uniform(-0.52, 0.52, 7000),
            rng.normal(0.48, 0.003, 7000),
        ))
        right = np.column_stack((
            rng.uniform(0.08, 1.48, 7000),
            rng.uniform(-0.52, 0.52, 7000),
            rng.normal(0.48, 0.003, 7000),
        ))
        floor = np.column_stack((
            rng.uniform(-3, 3, 8000), rng.uniform(-2, 2, 8000), rng.normal(0.0, 0.003, 8000)
        ))

        primitives, report = fit_support_edge(
            np.vstack((left, right, floor)), np.array([-1.2, 0.08, 0.59]), 0.0
        )

        self.assertEqual(len(primitives), 2)
        self.assertAlmostEqual(report["gap_width_m"], 0.16, delta=0.05)
        self.assertAlmostEqual(report["top_z_m"], 0.48, delta=0.02)
        self.assertAlmostEqual(report["raw_depth_top_z_m"], 0.48, delta=0.02)

    def test_static_cloud_keeps_low_confidence_finite_surface(self):
        frame_count, height, width = 8, 20, 30
        depth = np.full((frame_count, height, width), 2.0, dtype=np.float64)
        confidence = np.full_like(depth, 20.0)
        confidence[:, 6:14, 10:20] = 10.0
        confidence[:, :1, :] = 1.0
        rgb = np.full((frame_count, height, width, 3), 128, dtype=np.uint8)
        masks = np.zeros((frame_count, height, width), dtype=bool)
        intrinsic = np.array([[20.0, 0, 14.5], [0, 20.0, 9.5], [0, 0, 1.0]])
        world_to_camera = np.column_stack((np.eye(3), np.zeros(3)))

        points, _, point_confidence, report = static_point_cloud(
            depth, confidence, rgb, masks, intrinsic, world_to_camera, 1.0,
            max_points=1000,
        )

        self.assertEqual(report["confidence_quantile"], 0.01)
        self.assertGreater(len(points), 500)
        self.assertTrue(np.any(point_confidence == 10.0))

    def test_ground_uses_observed_sphere_contact_without_gt(self):
        rng = np.random.default_rng(9)
        points = np.column_stack((
            rng.uniform(-2, 2, 2000),
            rng.uniform(-2, 2, 2000),
            rng.normal(0.18, 0.005, 2000),
        ))
        estimated_p7 = np.array([-0.8, 0.1, 0.15])

        primitive, report = fit_ground(
            points, estimated_p7, observed_ball_support=True
        )

        self.assertAlmostEqual(report["ground_top_z_m"], 0.04, places=8)
        self.assertAlmostEqual(report["raw_depth_ground_top_z_m"], 0.18, delta=0.02)
        self.assertAlmostEqual(primitive["position_m"][2], -0.01, places=8)
        self.assertTrue(report["observed_ball_support_constraint"])

    def test_aperture_primitives_from_front_plane(self):
        rng = np.random.default_rng(5)
        count = 15000
        y = rng.uniform(-1.55, 1.55, count)
        z = rng.uniform(0.05, 1.8, count)
        keep = ~((np.abs(y) < 0.24) & (z < 1.0))
        plane = np.column_stack((rng.normal(0.58, 0.004, int(keep.sum())), y[keep], z[keep]))
        side = np.column_stack((
            rng.uniform(0.58, 0.86, 3000),
            rng.choice([-0.24, 0.24], 3000) + rng.normal(0, 0.004, 3000),
            rng.uniform(0.05, 1.0, 3000),
        ))

        primitives, report = fit_aperture(
            np.vstack((plane, side)), np.array([-0.95, 0.10, 0.11]), 0.0
        )

        self.assertEqual(len(primitives), 3)
        self.assertAlmostEqual(report["opening_width_m"], 0.48, delta=0.06)
        self.assertAlmostEqual(report["opening_height_m"], 1.0, delta=0.06)
        self.assertAlmostEqual(report["front_surface_x_m"], 0.58, delta=0.02)


if __name__ == "__main__":
    unittest.main()
