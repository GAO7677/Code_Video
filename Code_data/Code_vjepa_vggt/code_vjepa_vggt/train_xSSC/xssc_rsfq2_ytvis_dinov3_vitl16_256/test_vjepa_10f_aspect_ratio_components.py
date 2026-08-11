#!/usr/bin/env python3
"""CPU tests for V-JEPA 10-frame aspect-ratio training components."""

from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))

from object_centric_bench.datum import (  # noqa: E402
    DistributedAspectRatioBatchSampler,
    choose_aspect_ratio_bucket,
)
from object_centric_bench.learn import CbLinearCosineRestart  # noqa: E402
from object_centric_bench.model import LearntPositionalEmbedding  # noqa: E402


BUCKETS = [(336, 192), (256, 256), (224, 288), (192, 336), (144, 448)]


class FakeDataset:
    def __init__(self, shapes):
        self.shapes = shapes

    def __len__(self):
        return len(self.shapes)

    def get_spatial_shapes(self):
        return self.shapes


class AspectRatioComponentTest(unittest.TestCase):
    def test_bucket_selection(self):
        self.assertEqual(choose_aspect_ratio_bucket(720, 1280, BUCKETS), (192, 336))
        self.assertEqual(choose_aspect_ratio_bucket(720, 406, BUCKETS), (336, 192))
        self.assertEqual(choose_aspect_ratio_bucket(404, 1280, BUCKETS), (144, 448))

    def test_distributed_batches_are_shape_aligned(self):
        shapes = [(720, 1280)] * 9 + [(720, 960)] * 5 + [(720, 406)] * 3
        dataset = FakeDataset(shapes)
        rank_batches = []
        for rank in range(2):
            sampler = DistributedAspectRatioBatchSampler(
                dataset,
                BUCKETS,
                batch_size=2,
                num_replicas=2,
                rank=rank,
                seed=42,
            )
            rank_batches.append(list(iter(sampler)))
        self.assertEqual(len(rank_batches[0]), len(rank_batches[1]))
        for batch0, batch1 in zip(*rank_batches):
            buckets0 = {
                choose_aspect_ratio_bucket(*shapes[index], BUCKETS)
                for index in batch0
            }
            buckets1 = {
                choose_aspect_ratio_bucket(*shapes[index], BUCKETS)
                for index in batch1
            }
            self.assertEqual(len(buckets0), 1)
            self.assertEqual(buckets0, buckets1)

    def test_restart_schedule_uses_phase_step(self):
        schedule = CbLinearCosineRestart(
            assigns=["state['lr']=value"],
            start_step=10000,
            nlin=500,
            ntotal=10000,
            vstart=0.0,
            vbase=5e-5,
            vfinal=5e-8,
        )
        state = {"lr": None}
        schedule(step_count=10000, state=state)
        self.assertEqual(state["lr"], 0.0)
        schedule(step_count=10500, state=state)
        self.assertAlmostEqual(state["lr"], 5e-5)
        schedule(step_count=20000, state=state)
        self.assertAlmostEqual(state["lr"], 5e-8)

    def test_positional_grid_interpolation_is_trainable(self):
        embedding = LearntPositionalEmbedding(
            resolut=[256], embed_dim=8, spatial_shape=[16, 16]
        )
        output = embedding.interpolate_2d(12, 21)
        self.assertEqual(tuple(output.shape), (1, 252, 8))
        output.square().mean().backward()
        self.assertIsNotNone(embedding._pe.grad)
        self.assertTrue(torch.isfinite(embedding._pe.grad).all())


if __name__ == "__main__":
    unittest.main()
