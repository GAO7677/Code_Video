import unittest

import numpy as np
import torch

from future_query_predictor import Predictor, motion_features, interval_velocity, objective
from scene_token_cache import observed_normals, point_inputs, representative_indices


class ObservedSceneTests(unittest.TestCase):
    def plane(self):
        y, x = np.mgrid[:9, :11]
        xyz = np.stack(((x-5)*.03, (y-4)*.03, np.full_like(x, 2., dtype=float)), -1)
        points = np.repeat(xyz[None], 8, axis=0).astype(np.float32)
        valid = np.ones(points.shape[:-1], dtype=bool)
        return points, valid

    def test_normals_facing_camera(self):
        points, valid = self.plane()
        normals, keep = observed_normals(points, valid, np.zeros(3))
        self.assertEqual(int(keep.sum()), 8*7*9)
        np.testing.assert_allclose(normals[keep], np.tile([0, 0, -1], (keep.sum(), 1)), atol=1e-6)

    def test_mask_neighborhood_is_excluded(self):
        points, valid = self.plane()
        valid[:, 4, 5] = False
        normals, keep = observed_normals(points, valid, np.zeros(3))
        self.assertFalse(keep[:, 4, 5].any())
        self.assertFalse(keep[:, 4, 4].any())
        self.assertTrue(np.isfinite(normals).all())

    def test_rigid_transform_normals(self):
        points, valid = self.plane()
        rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
        shift = np.array([4., 5., 6.])
        a, ma = observed_normals(points, valid, np.zeros(3))
        b, mb = observed_normals(points@rotation.T+shift, valid, shift)
        np.testing.assert_array_equal(ma, mb)
        np.testing.assert_allclose(b[mb], a[ma]@rotation.T, atol=1e-6)

    def test_future_frames_rejected(self):
        points, valid = self.plane()
        aligned = dict(world_midpoint=np.concatenate((points, points[:1])),
                       static_valid=np.concatenate((valid, valid[:1])))
        with self.assertRaises(ValueError):
            point_inputs(aligned, {})

    def test_pixels_normals_and_world_coordinates_stay_aligned(self):
        points, valid = self.plane()
        aligned = dict(world_midpoint=points, static_valid=valid,
                       RT=np.column_stack((np.eye(3), np.zeros(3))), frame_times=np.arange(8)/30)
        images = np.full((8, 3, 9, 11), .4, dtype=np.float32)
        raw = dict(images=images, depth_conf=np.ones(valid.shape))
        a, ids, _, report = point_inputs(aligned, raw, 100)
        np.testing.assert_array_equal(a['coord'], points.reshape(-1, 3)[ids])
        np.testing.assert_allclose(a['color'], 102., atol=1e-6)
        self.assertEqual(report['encoder_input_points'], 100)
        aligned['oracle'] = np.full((100, 19), 99.)
        b, ids2, _, _ = point_inputs(aligned, raw, 100)
        np.testing.assert_array_equal(ids, ids2)
        np.testing.assert_array_equal(a['coord'], b['coord'])

    def test_inverse_representatives(self):
        result = representative_indices(np.array([1, 0, 1, 2]), np.array([4, 3, 2, 1]),
                                        np.array([0., 2., 7., 3., 1.]), 3)
        np.testing.assert_array_equal(result, [3, 2, 1])
        with self.assertRaises(ValueError):
            representative_indices(np.array([0]), np.array([0]), np.ones(1), 2)


class FutureQueryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        times = torch.arange(8).float()[None].expand(2, -1)/30
        velocity = torch.tensor([[[.2, .1, 0.]], [[.4, 0., -.1]]])
        positions = times[:, :, None, None]*velocity[:, None]
        motion, last, speed = motion_features(positions, torch.ones(2, 1, 3)*.2, times)
        self.batch = dict(motion=motion, object_mask=torch.ones(2, 1, dtype=torch.bool),
            last_position=last, last_velocity=speed,
            future_dt=torch.arange(1, 42).float()[None].expand(2, -1)/30,
            scene_xyz=torch.randn(2, 12, 3), scene_features=torch.randn(2, 12, 1386),
            scene_mask=torch.ones(2, 12, dtype=torch.bool))

    def model(self, mode='visual', nonzero=False):
        model = Predictor(torch.zeros(65), torch.ones(65), scene_mode=mode)
        if nonzero:
            with torch.no_grad():
                torch.manual_seed(9)
                model.position[-1].weight.normal_(std=.02)
        return model

    def test_initial_output_is_exact_cv_at_requested_times(self):
        pred = self.model()(self.batch)
        expected = self.batch['last_position'][:, :, None]+self.batch['last_velocity'][:, :, None]*self.batch['future_dt'][:, None, :, None]
        self.assertEqual(tuple(pred.shape), (2, 1, 41, 3))
        torch.testing.assert_close(pred, expected)
        vel = interval_velocity(pred, self.batch['last_position'], self.batch['future_dt'])
        torch.testing.assert_close(vel, self.batch['last_velocity'][:, :, None].expand_as(vel), atol=2e-6, rtol=2e-5)

    def test_future_truth_is_not_a_forward_input(self):
        with self.assertRaises(ValueError):
            self.model()({**self.batch, 'target_position': torch.zeros(2, 1, 41, 3)})

    def test_identical_parameters_for_constant_and_visual(self):
        a, b = self.model(), self.model('constant')
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)

    def test_scene_content_changes_prediction(self):
        model = self.model(nonzero=True)
        modified = {**self.batch, 'scene_features': self.batch['scene_features'].flip(0)}
        self.assertGreater(float((model(self.batch)-model(modified)).abs().max()), 1e-6)

    def test_constant_control_does_not_read_scene_content(self):
        model = self.model('constant', nonzero=True)
        changed = {**self.batch, 'scene_xyz': self.batch['scene_xyz']*100,
                   'scene_features': self.batch['scene_features']+33,
                   'scene_mask': torch.zeros_like(self.batch['scene_mask'])}
        torch.testing.assert_close(model(self.batch), model(changed), rtol=0, atol=0)

    def test_scene_and_motion_have_gradients(self):
        model = self.model(nonzero=True)
        prediction = model(self.batch)
        objective(prediction, torch.zeros_like(prediction), self.batch).backward()
        for module in (model.scene_projection[-1], model.motion_encoder[0], model.time_encoder[0]):
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(float(module.weight.grad.abs().sum()), 0.)

    def test_empty_scene_and_invalid_object_are_finite(self):
        batch = {**self.batch, 'scene_mask': torch.zeros_like(self.batch['scene_mask']),
                 'object_mask': torch.tensor([[True], [False]])}
        pred = self.model(nonzero=True)(batch)
        self.assertTrue(torch.isfinite(pred).all())
        self.assertEqual(float(pred[1].abs().sum()), 0.)

    def test_invalid_future_times_rejected(self):
        batch = {**self.batch, 'future_dt': torch.zeros_like(self.batch['future_dt'])}
        with self.assertRaises(ValueError):
            self.model()(batch)


if __name__ == '__main__':
    unittest.main()
