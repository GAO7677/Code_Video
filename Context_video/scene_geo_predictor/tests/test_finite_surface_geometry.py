import sys
import unittest
from pathlib import Path

import numpy as np
import torch


CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

from finite_surface_geometry import (  # noqa: E402
    FiniteSurfacePredictor,
    canonicalize_surfaces,
    sample_surface_points,
)


class FiniteSurfaceGeometryTest(unittest.TestCase):
    def _surfaces(self):
        center = np.asarray([[0.7, 0.0, 0.4], [0.7, 0.8, 0.4], [0.7, 0.0, 1.2], [0.0, 0.0, 0.0]], dtype=np.float32)
        normal = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        tangent_u = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        half = np.asarray([[0.1, 0.4], [0.2, 0.4], [0.2, 0.3], [0.0, 0.0]], dtype=np.float32)
        mask = np.asarray([True, True, True, False])
        return canonicalize_surfaces(center, normal, tangent_u, half, mask)

    def _batch(self, batch_size=2):
        surfaces = self._surfaces()
        return {
            "motion": torch.randn(batch_size, 1, 65),
            "object_mask": torch.ones(batch_size, 1, dtype=torch.bool),
            "last_position": torch.zeros(batch_size, 1, 3),
            "last_velocity": torch.ones(batch_size, 1, 3),
            "future_dt": torch.arange(1, 42, dtype=torch.float32).repeat(batch_size, 1) / 30.0,
            "surface_center": torch.from_numpy(np.repeat(surfaces["surface_center"][None], batch_size, axis=0)),
            "surface_normal": torch.from_numpy(np.repeat(surfaces["surface_normal"][None], batch_size, axis=0)),
            "surface_tangent_u": torch.from_numpy(np.repeat(surfaces["surface_tangent_u"][None], batch_size, axis=0)),
            "surface_half_extents": torch.from_numpy(np.repeat(surfaces["surface_half_extents"][None], batch_size, axis=0)),
            "surface_mask": torch.from_numpy(np.repeat(surfaces["surface_mask"][None], batch_size, axis=0)),
        }

    def test_canonicalization_and_sampler_preserve_finite_boundaries(self):
        surfaces = self._surfaces()
        self.assertTrue(np.allclose(np.linalg.norm(surfaces["surface_normal"][:3], axis=-1), 1.0))
        self.assertTrue(np.allclose((surfaces["surface_normal"][:3] * surfaces["surface_tangent_u"][:3]).sum(-1), 0.0))
        sampled = sample_surface_points(surfaces, seed=9, count=32, resolution=4)
        self.assertEqual(sampled["point_xyz"].shape, (32, 3))
        self.assertEqual(set(sampled["point_source_surface"].tolist()), {0, 1, 2})
        self.assertLessEqual(np.max(np.abs(sampled["point_xyz"][:, 0])), 0.9)

    def test_predictor_shape_backward_and_surface_permutation(self):
        model = FiniteSurfacePredictor(torch.zeros(65), torch.ones(65), seed=42)
        model.eval()
        batch = self._batch()
        output = model(batch)
        self.assertEqual(output.shape, (2, 1, 41, 3))
        self.assertTrue(torch.isfinite(output).all())
        model.train()
        loss = output.square().mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters()))

        model.eval()
        original = model(batch)
        perm = torch.tensor([2, 0, 3, 1])
        permuted = {key: value.clone() for key, value in batch.items()}
        for key in ("surface_center", "surface_normal", "surface_tangent_u", "surface_half_extents", "surface_mask"):
            permuted[key] = permuted[key][:, perm]
        self.assertTrue(torch.allclose(original, model(permuted), atol=1e-6, rtol=1e-6))

    def test_future_and_undeclared_fields_are_rejected(self):
        model = FiniteSurfacePredictor(torch.zeros(65), torch.ones(65), seed=42)
        batch = self._batch(batch_size=1)
        bad = dict(batch)
        bad["future_target"] = torch.zeros(1)
        with self.assertRaises(ValueError):
            model(bad)
        bad = dict(batch)
        bad["surface_half_extents"] = bad["surface_half_extents"].clone()
        bad["surface_half_extents"][0, 0, 0] = 0.0
        with self.assertRaises(ValueError):
            model(bad)


if __name__ == "__main__":
    unittest.main()
