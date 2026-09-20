import json
from pathlib import Path
import tempfile
import unittest

from predictor_config_guard import validate


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT.parent / "configs/predictor_experiments_v1.json"


class PredictorConfigGuardTest(unittest.TestCase):
    def test_motion_only_diagnostic_does_not_require_scene_cache(self):
        result = validate(CONFIG, protocol="diagnostic", variant="motion_only")
        self.assertEqual(result["status"], "PASS")

    def test_formal_legacy_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "legacy.json"
            report.write_text(json.dumps({"schema": "observed_utonia_tokens_v1"}))
            with self.assertRaisesRegex(ValueError, "formal provenance"):
                validate(CONFIG, protocol="formal", variant="geometry_utonia", scene_reports=[report])

    def test_formal_v2_without_provenance_fields_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "v2.json"
            report.write_text(json.dumps({"cache_schema": "observed_utonia_tokens_v2"}))
            with self.assertRaisesRegex(ValueError, "provenance fields"):
                validate(CONFIG, protocol="formal", variant="geometry_utonia", scene_reports=[report])

    def test_unimplemented_switch_is_rejected(self):
        config = json.loads(CONFIG.read_text())
        config["experimental_switches_default"]["two_pass_future_query"] = True
        with self.assertRaisesRegex(ValueError, "not implemented"):
            validate(config, protocol="diagnostic", variant="motion_only")

    def test_partial_oracle_is_explicitly_allowed_only_diagnostic(self):
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "oracle.json"
            report.write_text(json.dumps({"schema": "oracle_partial_visible_geometry_v1",
                                          "cache_schema": "oracle_partial_visible_geometry_v1",
                                          "privileged_geometry": True}))
            result = validate(CONFIG, protocol="diagnostic", variant="oracle_partial_visible_geometry",
                              scene_reports=[report])
            self.assertEqual(result["status"], "PASS")
            with self.assertRaises(ValueError):
                validate(CONFIG, protocol="formal", variant="oracle_partial_visible_geometry",
                         scene_reports=[report])


if __name__ == "__main__":
    unittest.main()
