from __future__ import annotations

import unittest

import prepare_validation_30_cases as prepare
import run_validation_30_loss as runner


class Validation30LossManifestTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
