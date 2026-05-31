from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.extraction import AnnotationPseudoStateExtractor, compute_scale_depth_consistency
from phys_state_video.proxy_state import extract_primary_track
from scripts.curate_object_motion_data import resize_frames_uint8


class ExtractionTests(unittest.TestCase):
    def test_scale_depth_consistency_array(self):
        values = compute_scale_depth_consistency(np.array([1.0, 2.0]), np.array([2.0, 4.0]))
        self.assertEqual(values.shape, (2,))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_annotation_extraction_shapes(self):
        extractor = AnnotationPseudoStateExtractor(image_height=10, image_width=10)
        annotations = [
            [{"track_id": 1, "bbox": [1, 1, 3, 3], "depth": 1.0}],
            [{"track_id": 1, "bbox": [2, 1, 4, 3], "depth": 1.1}],
        ]
        outputs = extractor.extract(annotations)
        self.assertEqual(outputs["states"].shape, (2, 1, 10))
        self.assertEqual(outputs["boxes"].shape, (2, 1, 4))
        self.assertGreater(outputs["states"][1, 0, 4], 0.0)

    def test_proxy_track_shapes(self):
        frames = np.zeros((4, 3, 32, 32), dtype=np.float32)
        for idx in range(4):
            frames[idx, 0, 8:16, 4 + idx:12 + idx] = 1.0
        outputs = extract_primary_track(frames)
        self.assertEqual(outputs.boxes.shape, (4, 1, 4))
        self.assertEqual(outputs.states.shape, (4, 1, 10))
        self.assertEqual(outputs.appearance.shape, (1, 64))
        self.assertTrue(np.isfinite(outputs.states).all())

    def test_resize_frames_stretch(self):
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        frame[:, :10, 0] = 255
        frames, resize_meta = resize_frames_uint8([frame], height=16, width=16, resize_mode="stretch")
        self.assertEqual(frames.shape, (1, 3, 16, 16))
        self.assertEqual(resize_meta.mode, "stretch")
        self.assertEqual((resize_meta.original_height, resize_meta.original_width), (10, 20))
        self.assertEqual((resize_meta.pad_top, resize_meta.pad_bottom, resize_meta.pad_left, resize_meta.pad_right), (0, 0, 0, 0))

    def test_resize_frames_letterbox(self):
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        frame[:, :, 1] = 200
        frames, resize_meta = resize_frames_uint8([frame], height=16, width=16, resize_mode="letterbox")
        self.assertEqual(frames.shape, (1, 3, 16, 16))
        self.assertEqual(resize_meta.mode, "letterbox")
        self.assertEqual((resize_meta.resized_height, resize_meta.resized_width), (8, 16))
        self.assertEqual((resize_meta.pad_top, resize_meta.pad_bottom, resize_meta.pad_left, resize_meta.pad_right), (4, 4, 0, 0))
        letterboxed = np.transpose(frames[0], (1, 2, 0))
        self.assertTrue(np.allclose(letterboxed[:4], 0.0))
        self.assertTrue(np.allclose(letterboxed[12:], 0.0))
        self.assertGreater(float(letterboxed[4:12, :, 1].mean()), 0.7)


if __name__ == "__main__":
    unittest.main()
