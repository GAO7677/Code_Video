"""CPU contract tests for observed-only inputs."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import prepare_context as p


class ContextContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="visual-context-test-")
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        for i in range(8):
            Image.fromarray(np.full((12, 16, 3), i * 20, dtype=np.uint8)).save(
                self.source / f"frame_{i+1:04d}.png")
        (self.source / "render_metadata.json").write_text(
            json.dumps({"fps": 30, "camera": {"framing_profile": "test"}}))
        self.case = dict(case_id="sample_0", family="gap", frames_dir=str(self.source))

    def tearDown(self):
        self.temp.cleanup()

    def export(self, name="export"):
        return p.prepare_case(self.case, self.root / name)

    def test_eight_true_frames_and_times(self):
        row = self.export()
        rgb, times = p.load_model_input(Path(row["input_json"]))
        self.assertEqual(rgb.shape, (8, 12, 16, 3))
        np.testing.assert_allclose(times, np.arange(8) / 30)
        for i in range(8):
            self.assertTrue(np.all(rgb[i] == i * 20))
        self.assertFalse(row["static_scene_gt_in_input"])

    def test_future_pixels_cannot_change_context(self):
        row = self.export("first")
        (self.source / "frame_0009.png").write_bytes(b"not even an image")
        other = self.export("second")
        self.assertEqual(row["pixels_sha256"], other["pixels_sha256"])
        self.assertEqual(len(list(Path(other["input_json"]).parent.glob("rgb_*.png"))), 8)

    def test_missing_frame_is_not_repeated_or_padded(self):
        (self.source / "frame_0008.png").unlink()
        with self.assertRaisesRegex(ValueError, "missing observed"):
            self.export()
        self.assertFalse((self.root / "export").exists())

    def test_reject_future_input_index(self):
        row = self.export()
        path = Path(row["input_json"])
        payload = json.loads(path.read_text())
        payload["observed_indices"][-1] = 8
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "Only RGB0-7"):
            p.load_model_input(path)

    def test_reject_static_gt_field(self):
        row = self.export()
        path = Path(row["input_json"])
        payload = json.loads(path.read_text())
        payload["oracle"] = [1, 2, 3]
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "Unexpected input"):
            p.load_model_input(path)

    def test_pixel_change_is_detected(self):
        row = self.export()
        path = Path(row["input_json"])
        Image.fromarray(np.full((12, 16, 3), 255, dtype=np.uint8)).save(path.parent / "rgb_00.png")
        with self.assertRaisesRegex(ValueError, "pixels or shape changed"):
            p.load_model_input(path)

    def test_refuse_overwrite(self):
        self.export()
        with self.assertRaises(FileExistsError):
            self.export()

    def test_reject_nonmonotonic_times(self):
        row = self.export()
        path = Path(row["input_json"])
        payload = json.loads(path.read_text())
        payload["time_s"][2] = 0.0
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "Invalid observation times"):
            p.load_model_input(path)


if __name__ == "__main__":
    unittest.main()

