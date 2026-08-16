from __future__ import annotations

import unittest

import torch

import prepare_validation_30_cases as prepare
import run_validation_30_loss as runner


class Validation30LossManifestTests(unittest.TestCase):
    def test_model_kind_selects_trajectory_implementation(self) -> None:
        config = {"trajectory_loss": {"enabled": True}}

        self.assertEqual(runner.ev.model_kind(config), "trajectory")

    def test_model_kind_selects_combined_slot_dedup_xssc_implementation(self) -> None:
        config = {
            "conditioning": {"slot_dedup": {"mode": "merge"}},
            "xssc_loss_enabled": True,
        }

        self.assertEqual(runner.ev.model_kind(config), "slot_dedup_xssc")

    def test_preparer_uses_the_validation_seed_contract(self) -> None:
        self.assertEqual(
            prepare.stable_case_seed(20260815, "pybullet", 59),
            239753544,
        )

    def test_normalize_case_seeds_migrates_missing_and_preserves_existing(self) -> None:
        manifest = {
            "seed": 20260815,
            "cases": [
                {"source": "pybullet", "source_index": 59},
                {"source": "pybullet", "source_index": 88, "case_seed": 123},
            ],
        }

        normalized, changed = runner.normalize_case_seeds(manifest)

        self.assertTrue(changed)
        self.assertEqual(normalized["cases"][0]["case_seed"], 239753544)
        self.assertEqual(normalized["cases"][1]["case_seed"], 123)

    def test_evaluate_prepared_attaches_and_clears_trajectory_cache(self) -> None:
        class FakePipe:
            device = torch.device("cpu")
            torch_dtype = torch.float32

        class FakeTrajectoryModel:
            pipe = FakePipe()
            _trajectory_batch = None

            @staticmethod
            def transfer_data_to_device(prepared, _device, _dtype):
                return prepared

            def _compute_object_losses(self, _pipe, _shared, _positive):
                self.seen_cache = self._trajectory_batch
                return torch.tensor(1.5), {
                    "train/loss_main": 1.0,
                    "train/loss_trajectory": 0.5,
                }

        model = FakeTrajectoryModel()
        trajectory_cache = {"object_count": 1}

        loss, metrics = runner.ev.evaluate_prepared(
            model,
            ({}, {}, {}),
            case_seed=42,
            trajectory_cache=trajectory_cache,
        )

        self.assertEqual(loss, 1.0)
        self.assertEqual(metrics["train/loss_trajectory"], 0.5)
        self.assertEqual(model.seen_cache, [trajectory_cache])
        self.assertIsNone(model._trajectory_batch)


if __name__ == "__main__":
    unittest.main()
