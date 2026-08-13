import tempfile
import unittest
from pathlib import Path
from unittest import mock

import build_xssc_lora_checkpoint_dashboard as dashboard


class PendingPhysicIQDashboardTests(unittest.TestCase):
    def test_static_checkpoint_is_visible_in_test5_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as output_root:
            checkpoint = Path(output_root) / "checkpoints" / "step-000000"
            config = {
                "paths": {"watch_root": output_root},
                "runtime": {
                    "num_inference_steps": 8,
                    "height": 512,
                    "width": 896,
                    "context_frames": 5,
                    "num_frames": 49,
                },
                "methods": [
                    {
                        "key": "baseline",
                        "label": "Baseline",
                        "static_checkpoints": [
                            {"step": 0, "path": str(checkpoint)}
                        ],
                    }
                ],
            }
            records = dashboard.load_live_test_manifests(config, [])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["method_key"], "baseline")
        self.assertEqual(records[0]["step"], 0)
        self.assertEqual(records[0]["origin"], "watcher-live")

    def test_static_checkpoint_is_discovered_for_pending_rows(self) -> None:
        with tempfile.TemporaryDirectory() as output_root:
            checkpoint = Path(output_root) / "step-000000"
            config = {
                "methods": [
                    {
                        "key": "baseline",
                        "label": "Baseline",
                        "static_checkpoints": [
                            {"step": 0, "path": str(checkpoint)}
                        ],
                    }
                ]
            }
            records = dashboard.load_configured_checkpoints(config)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["method_key"], "baseline")
        self.assertEqual(records[0]["step"], 0)
        self.assertEqual(records[0]["source"], "configured-static")

    def test_configured_checkpoint_is_visible_before_first_case_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as output_root:
            config = {
                "physiciq": {
                    "enabled": True,
                    "method_keys": ["new_method"],
                    "method_name_template": "run_{method_key}_{step}",
                    "output_root": output_root,
                },
                "methods": [
                    {
                        "key": "new_method",
                        "label": "New method",
                        "color": "#123456",
                    }
                ],
            }
            checkpoint = Path(output_root) / "checkpoints" / "step-000500"
            with mock.patch.object(
                dashboard,
                "load_configured_checkpoints",
                return_value=[
                    {
                        "method_key": "new_method",
                        "step": 500,
                        "checkpoint_dir": str(checkpoint),
                    }
                ],
            ):
                records = dashboard.load_live_physiciq_manifests(config, [])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["method_key"], "new_method")
        self.assertEqual(records[0]["step"], 500)
        self.assertEqual(records[0]["origin"], "watcher-live")


if __name__ == "__main__":
    unittest.main()
