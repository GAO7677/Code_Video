from __future__ import annotations

import json
import unittest
from pathlib import Path

import launch_from_config as launcher
import train_xssc_object_self_attn_lora as core


PROJECT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_DIR.parent
CONFIG_PATH = (
    PROJECT_DIR
    / "configs/t_head_pck32_s039_latest3350_top100_no_object_xssc_loss_dinov3_movic_step50000.json"
)
HEAD_CONFIG_PATH = (
    EXPERIMENT_ROOT
    / "configs/physiciq67_pck32_s039_latest3350_top100_heads.json"
)


class THeadXSSCLossConfigTests(unittest.TestCase):
    def test_head_selection_config_matches_declared_identity(self) -> None:
        heads_by_block, metadata = core.load_head_selection_config(
            HEAD_CONFIG_PATH,
            expected_subset_id="T_physiciq67_pck32_s039_latest3350_top100",
            expected_role="T",
            expected_feature_subtype="physiciq67_pck32_s039_latest3350",
            expected_num_heads=100,
            num_blocks=30,
            num_heads=24,
        )
        self.assertEqual(sum(len(heads) for heads in heads_by_block.values()), 100)
        self.assertEqual(len(heads_by_block), 25)
        self.assertEqual(metadata["ranking_step"], 39)
        self.assertEqual(metadata["completed_runs_at_selection"], 3350)
        source = json.loads(Path(metadata["selection_source"]).read_text())
        expected = [
            (int(item["block"]), int(item["head"]))
            for item in source["entries"][:100]
        ]
        actual = [
            (int(item["block"]), int(item["head"]))
            for item in metadata["targets"]
        ]
        self.assertEqual(actual, expected)

    def test_launcher_builds_no_object_t_head_xssc_loss_command(self) -> None:
        raw, _ = launcher.load_config(CONFIG_PATH)
        config = launcher.validate_config(raw, CONFIG_PATH.parent)
        command = launcher.build_command(config, Path("/tmp/test-t-head-xssc-loss"))
        command_text = " ".join(command)

        self.assertEqual(config["adaptation"]["mode"], "t_head")
        self.assertFalse(config["adaptation"]["enable_object_branch"])
        self.assertTrue(config["xssc_loss"]["enabled"])
        self.assertEqual(config["experiment"]["expected_trainable_params"], 11468800)
        expected_params = 25 * 4 * 32 * 3072 + 4 * 32 * 128 * 100
        self.assertEqual(config["experiment"]["expected_trainable_params"], expected_params)
        self.assertIn(str(launcher.XSSC_LOSS_TRAIN_SCRIPT), command)
        self.assertIn("--disable_object_branch", command)
        self.assertIn("--self_attn_adaptation_mode t_head", command_text)
        self.assertIn("--head_selection_expected_num_heads 100", command_text)
        self.assertIn("--xssc_loss_backend dinov3_movic", command_text)


if __name__ == "__main__":
    unittest.main()
