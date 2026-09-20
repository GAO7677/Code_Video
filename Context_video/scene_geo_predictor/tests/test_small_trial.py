"""Small-trial split and observation export contracts."""
import unittest
from types import SimpleNamespace

from prepare_small_trial import actor_metadata, select


class SplitTests(unittest.TestCase):
    def test_renderer_actor_contract(self):
        actor = SimpleNamespace(family_key='ball', role='dynamic', dynamic=True,
            shape='sphere', size={'radius':.2}, mass=1., friction=.2, restitution=.5,
            position=(1.,2.,3.), linear_velocity=(1.,0.,0.), angular_velocity=(0.,0.,0.))
        exported = actor_metadata(actor)
        for name in ('shape', 'size_m', 'initial_position_m', 'dynamic', 'object_id'):
            self.assertIn(name, exported)
        self.assertEqual(exported['initial_position_m'], actor.position)

    def records(self):
        return [dict(key=f'{split}_{family}_{i}', split=split, family=family,
                     lineage_cell=f'{split}:{family}:{i}', context_hash=f'{split}:{family}:c{i}',
                     motion_hash=f'{split}:{family}:m{i}', u=[i/100]*5)
                for split in ('train', 'val', 'test') for family in ('barrier', 'door', 'gap')
                for i in range(40)]

    def test_balanced_deterministic_and_no_test(self):
        rows = self.records()
        got = select(rows)
        self.assertEqual(got, select(rows))
        self.assertEqual(len(got), 120)
        for family in ('barrier', 'door', 'gap'):
            self.assertEqual(sum(r['family'] == family and r['split'] == 'train' for r in got), 30)
            self.assertEqual(sum(r['family'] == family and r['split'] == 'val' for r in got), 10)
        self.assertNotIn('test', {r['split'] for r in got})

    def test_reject_duplicate_context_across_splits(self):
        rows = self.records()
        for r in rows:
            r['context_hash'] = 'same'
        with self.assertRaisesRegex(ValueError, 'context_hash'):
            select(rows)

    def test_reject_insufficient_unique_cells(self):
        rows = self.records()
        for r in rows:
            r['lineage_cell'] = r['split']+r['family']
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            select(rows)


class TrainingTests(unittest.TestCase):
    def test_epoch_stream_is_complete_and_resumable(self):
        from train_small_trial import sample_indices
        first = [i for step in range(3) for i in sample_indices(90, step, 30, 42)]
        self.assertEqual(sorted(first), list(range(90)))
        self.assertEqual(sample_indices(90, 37, 30, 42), sample_indices(90, 37, 30, 42))
        self.assertNotEqual(first[:30], sample_indices(90, 3, 30, 42))

    def test_minibatch_accumulation_matches_full_batch(self):
        import torch
        from future_query_predictor import Predictor, objective
        torch.set_num_threads(2)
        torch.manual_seed(12)
        batch = dict(motion=torch.randn(4,1,65), object_mask=torch.ones(4,1,dtype=torch.bool),
            last_position=torch.randn(4,1,3), last_velocity=torch.randn(4,1,3),
            future_dt=torch.arange(1,42)[None].expand(4,-1)/30,
            scene_xyz=torch.randn(4,8,3), scene_features=torch.randn(4,8,1386),
            scene_mask=torch.ones(4,8,dtype=torch.bool))
        target = torch.randn(4,1,41,3)
        model = Predictor(torch.zeros(65), torch.ones(65))
        objective(model(batch), target, batch).backward()
        expected = {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        for start in (0,2):
            part = {k:v[start:start+2] for k,v in batch.items()}
            (objective(model(part), target[start:start+2], part)/2).backward()
        for name, p in model.named_parameters():
            if name in expected:
                torch.testing.assert_close(p.grad, expected[name], rtol=1e-4, atol=1e-6)


class MetricTests(unittest.TestCase):
    def test_gap_is_not_filled_by_union_bounding_box(self):
        import numpy as np
        from trial_metrics import sphere_distances
        boxes = [(np.array([-1.,0.,1.]), np.array([.3,1.,1.]), np.array([0.,0.,0.,1.])),
                 (np.array([1.,0.,1.]), np.array([.3,1.,1.]), np.array([0.,0.,0.,1.]))]
        distance = sphere_distances(np.array([[0.,0.,1.], [-1.,0.,1.]]), .2, boxes)
        self.assertTrue((distance[0] > 0).all())
        self.assertLess(distance[1,1], 0)

    def test_support_is_not_repeated_contact_event(self):
        import numpy as np
        from trial_metrics import first_contact_onset
        distance = np.zeros((49,1))
        self.assertIsNone(first_contact_onset(distance))
        distance[8:12] = .2
        self.assertEqual(first_contact_onset(distance), 12)

    def test_contact_at_first_future_frame_and_no_zero_time_interval(self):
        import numpy as np
        from trial_metrics import first_contact_onset
        distance = np.ones((49,1))
        distance[8:] = 0
        self.assertEqual(first_contact_onset(distance), 8)


if __name__ == '__main__':
    unittest.main()
