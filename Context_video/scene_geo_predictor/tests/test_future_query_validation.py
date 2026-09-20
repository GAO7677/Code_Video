"""CPU-only contracts for the current Future Query Predictor.

These tests intentionally target future_query_predictor.py, not the legacy
SG-O model used by test_predictor_scene_integration.py.
"""
import unittest

import torch

from future_query_predictor import Predictor, interval_velocity, motion_features


class FutureQueryValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def batch(self, objects=1, points=5):
        b = 2
        times = torch.arange(8, dtype=torch.float32)[None].expand(b, -1) / 30
        velocity = torch.tensor([[[1.2, .1, 0.]], [[.8, -.2, 0.]]])
        positions = times[:, :, None, None] * velocity[:, None]
        if objects > 1:
            positions = positions.expand(-1, -1, objects, -1).clone()
        motion, last, last_velocity = motion_features(
            positions, torch.ones(b, objects, 3) * .2, times)
        return dict(
            motion=motion, object_mask=torch.ones(b, objects, dtype=torch.bool),
            last_position=last, last_velocity=last_velocity,
            future_dt=torch.arange(1, 42, dtype=torch.float32)[None].expand(b, -1) / 30,
            scene_xyz=torch.linspace(-.2, .2, points * 3).reshape(1, points, 3).expand(b, -1, -1).clone(),
            scene_features=torch.zeros(b, points, 1386),
            scene_mask=torch.ones(b, points, dtype=torch.bool),
        )

    def test_future_grid_is_protocol_strict_and_first_interval_is_included(self):
        batch = self.batch()
        bad = {**batch, 'future_dt': batch['future_dt'].clone()}
        bad['future_dt'][:, 0] = 1.5 / 30
        with self.assertRaisesRegex(ValueError, 'RGB8-RGB48'):
            Predictor(torch.zeros(65), torch.ones(65))(bad)

        position = batch['last_position'][:, :, None] + torch.arange(1, 42)[None, None, :, None] / 30
        velocity = interval_velocity(position, batch['last_position'], batch['future_dt'])
        torch.testing.assert_close(velocity[:, :, 0], torch.ones(2, 1, 3))

    def test_geometry_only_has_a_real_geometry_value_path(self):
        batch = self.batch(points=4)
        model = Predictor(torch.zeros(65), torch.ones(65), scene_mode='geometry_only')
        with torch.no_grad():
            model.position[-1].weight.fill_(.05)
        changed = {**batch, 'scene_xyz': batch['scene_xyz'].clone()}
        changed['scene_xyz'][:, 0, 0] += 1.0
        with torch.no_grad():
            first, second = model(batch), model(changed)
        self.assertGreater(float((first - second).abs().max()), 1e-7)

    def test_geometry_only_ignores_utonia_features(self):
        batch = self.batch()
        model = Predictor(torch.zeros(65), torch.ones(65), scene_mode='geometry_only')
        with torch.no_grad():
            model.position[-1].weight.fill_(.05)
            a = model(batch)
            b = model({**batch, 'scene_features': torch.randn_like(batch['scene_features'])})
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_motion_only_ignores_scene_and_token_count(self):
        batch = self.batch()
        model = Predictor(torch.zeros(65), torch.ones(65), scene_mode='motion_only')
        with torch.no_grad():
            model.position[-1].weight.fill_(.05)
            a = model(batch)
            empty = {**batch, 'scene_xyz': batch['scene_xyz'][:, :0],
                     'scene_features': batch['scene_features'][:, :0],
                     'scene_mask': batch['scene_mask'][:, :0]}
            b = model(empty)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_invalid_padded_scene_and_object_do_not_change_valid_output(self):
        batch = self.batch(points=6)
        batch['scene_mask'][:, -2:] = False
        model = Predictor(torch.zeros(65), torch.ones(65))
        with torch.no_grad():
            model.position[-1].weight.fill_(.05)
            clean = model(batch)
            changed = {**batch, 'scene_xyz': batch['scene_xyz'].clone(),
                       'scene_features': batch['scene_features'].clone()}
            changed['scene_xyz'][:, -2:] = 1e6
            changed['scene_features'][:, -2:] = 1e6
            dirty = model(changed)
        torch.testing.assert_close(clean, dirty, rtol=0, atol=0)

        object_padded = {**batch, 'object_mask': torch.tensor([[True], [False]])}
        prediction = model(object_padded)
        self.assertEqual(float(prediction[1].abs().sum()), 0.)

    def test_zero_length_scene_is_finite(self):
        batch = self.batch(points=0)
        prediction = Predictor(torch.zeros(65), torch.ones(65))(batch)
        self.assertTrue(bool(torch.isfinite(prediction).all()))

    def test_nonfinite_declared_inputs_are_rejected(self):
        batch = self.batch()
        bad_motion = {**batch, 'motion': batch['motion'].clone()}
        bad_motion['motion'][0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite motion'):
            Predictor(torch.zeros(65), torch.ones(65))(bad_motion)
        bad_scene = {**batch, 'scene_xyz': batch['scene_xyz'].clone()}
        bad_scene['scene_xyz'][0, 0, 0] = float('inf')
        with self.assertRaisesRegex(ValueError, 'Nonfinite scene input'):
            Predictor(torch.zeros(65), torch.ones(65))(bad_scene)

    def test_experimental_controls_remove_unused_parameters_from_optimizer(self):
        geometry = Predictor(torch.zeros(65), torch.ones(65), scene_mode='geometry_only')
        motion = Predictor(torch.zeros(65), torch.ones(65), scene_mode='motion_only')
        self.assertFalse(any('scene_projection' in n for n, p in geometry.named_parameters() if p.requires_grad))
        self.assertFalse(any('feature_key' in n or 'feature_value' in n
                             for n, p in geometry.named_parameters() if p.requires_grad))
        self.assertFalse(any(n.startswith('reader.') for n, p in motion.named_parameters() if p.requires_grad))


if __name__ == '__main__':
    unittest.main()
