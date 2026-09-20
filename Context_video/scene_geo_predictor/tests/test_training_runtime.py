"""Real-cache CPU integration tests, not independent experiment results."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from prepare_small_trial import ROOT, dump
import train_small_trial as trainer


class RuntimeTests(unittest.TestCase):
    def make_root(self, root):
        source = ROOT/'small_trial_120'
        records = json.loads((source/'manifest.json').read_text())['records']
        picked = [dict(next(r for r in records if r['family'] == f)) for f in ('barrier','door','gap')]
        for i,row in enumerate(picked):
            row['split'] = 'train' if i < 2 else 'val'
        root.mkdir()
        for name in ('observed_context','aligned_scene','scene_features','samples'):
            (root/name).symlink_to(source/name, target_is_directory=True)
        dump(root/'manifest.json', dict(records=picked, role='unit_test_fixture_only',
            training=dict(seed=42, lr=3e-4, weight_decay=.01, max_steps=2, effective_batch=2, validate_every=1)))

    def test_resume_matches_uninterrupted_training(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir='/data/gaoya/agent-data/cache/sg-overlay/tmp') as temp:
            full, interrupted = Path(temp)/'full', Path(temp)/'interrupted'
            self.make_root(full)
            self.make_root(interrupted)
            trainer.train(full, 'visual', 'cpu')
            original = trainer.atomic_checkpoint
            def interrupt(path, state):
                original(path, state)
                raise RuntimeError('injected after durable checkpoint')
            with patch.object(trainer, 'atomic_checkpoint', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    trainer.train(interrupted, 'visual', 'cpu')
            trainer.train(interrupted, 'visual', 'cpu', resume=True)
            a = torch.load(full/'runs/visual/last.pt', weights_only=False)
            b = torch.load(interrupted/'runs/visual/last.pt', weights_only=False)
            self.assertEqual(a['step'], b['step'])
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            for slot in a['optimizer']['state']:
                for name in a['optimizer']['state'][slot]:
                    torch.testing.assert_close(a['optimizer']['state'][slot][name],b['optimizer']['state'][slot][name],rtol=0,atol=0)
            entries = [json.loads(line) for line in (interrupted/'runs/visual/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([e['step'] for e in entries], [1,2])


if __name__ == '__main__':
    unittest.main()
