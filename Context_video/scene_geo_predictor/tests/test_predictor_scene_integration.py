"""CPU computation-graph checks, not a physical prediction experiment."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

from predictor_scene_adapter import load_visual_scene, make_model_inputs


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT.parent / "p4_v2_sg_o_revised_20260916/round1/scripts/model.py"
SPEC = importlib.util.spec_from_file_location("_visual_input_pilot_model", MODEL_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CASES = ("pilot_gap_0000", "pilot_barrier_0004", "pilot_door_ball_0006")


def synthetic_context(last):
    """Synthetic motion/surface fixture anchored near the actual observed object."""
    rng = np.random.default_rng(42)
    last_position = np.zeros((2, 3), np.float32)
    last_position[0] = last
    surface = np.zeros((2, 64, 3), np.float32)
    surface[0] = last + rng.normal(0, 0.05, (64, 3))
    surface_mask = np.zeros((2, 64), bool)
    surface_mask[0] = True
    times = np.arange(1, 42, dtype=np.float32) / 30
    cv = np.zeros((2, 41, 3), np.float32)
    cv[0, :, 0] = times
    return dict(node=rng.normal(0, 0.1, (2, 90)).astype(np.float32),
                edge=np.zeros((2, 2, 54), np.float32), mask=np.array([True, False]),
                surface=surface, surface_mask=surface_mask,
                history_dt=np.arange(-7, 1, dtype=np.float32) / 30,
                future_dt=times, last_position=last_position, cv=cv)


class PredictorSceneIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.stats = {}
        for name, shape in (("node", (90,)), ("edge", (54,)), ("target", (41, 3))):
            cls.stats[name + "_mean"] = torch.zeros(shape)
            cls.stats[name + "_std"] = torch.ones(shape)
        rows = []
        cls.sources = []
        for case in CASES:
            scene = load_visual_scene(ROOT / "aligned_scene" / case)
            with np.load(ROOT / "observed_context" / case / "context_geometry.npz") as data:
                last = data["positions_world"][7, 0].copy()
            rows.append(make_model_inputs(synthetic_context(last), scene))
            cls.sources.append(scene)
        cls.batch = {key: torch.from_numpy(np.stack([row[key] for row in rows]))
                     for key in rows[0]}

    def models(self):
        models = [MODULE.Predictor(variant, self.stats, seed=42)
                  for variant in ("surface", "geometry_bounded")]
        for model in models:
            model.core.contact.requires_grad_(False)
        return models

    def test_real_cache_shape_and_source_identity(self):
        self.assertEqual(self.batch["scene_xyz"].shape, (3, 1792, 3))
        self.assertTrue(bool(self.batch["scene_mask"].all()))
        for case, scene in zip(CASES, self.sources):
            with np.load(ROOT / "aligned_scene" / case / "coarse_static_points.npz") as data:
                indices = scene["source_tyx"][scene["scene_mask"]]
                expected = data["world_midpoint"][tuple(indices.T)]
                np.testing.assert_array_equal(scene["scene_xyz"][scene["scene_mask"]], expected)

    def test_shared_initialization_and_baseline_invariance(self):
        baseline, visual = self.models()
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, visual.state_dict()[key]), key)
        changed = dict(self.batch, scene_xyz=self.batch["scene_xyz"] + 100)
        with torch.no_grad():
            a = baseline(self.batch)["position"]
            self.assertTrue(torch.equal(a, baseline(changed)["position"]))
            self.assertTrue(torch.equal(a, visual(self.batch)["position"]))
            self.assertEqual(a.shape, (3, 2, 41, 3))

    def test_scene_is_readable_and_receives_gradients(self):
        _, model = self.models()
        relation, valid, features = model.scene_inputs(self.batch)
        self.assertEqual(relation.shape, (3, 2, 1792, 4))
        self.assertIsNone(features)
        self.assertTrue(bool(valid[:, 0].any(-1).all()))
        self.assertFalse(bool(valid[:, 1].any()))
        optimizer = torch.optim.AdamW(model.active_parameters(), lr=3e-4)
        target = self.batch["last_position"][:, :, None] + self.batch["cv"]
        for _ in range(3):
            optimizer.zero_grad()
            prediction = model(self.batch)["position"]
            loss = torch.nn.functional.smooth_l1_loss(
                prediction[self.batch["mask"]], target[self.batch["mask"]])
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all())
                                for p in model.parameters()))
            optimizer.step()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().max() > 0)
                            for name, p in model.named_parameters()
                            if "geometry_attention" in name))
        with torch.no_grad():
            masked = dict(self.batch, scene_mask=torch.zeros_like(self.batch["scene_mask"]))
            self.assertFalse(torch.equal(model(self.batch)["position"], model(masked)["position"]))


if __name__ == "__main__":
    unittest.main()
