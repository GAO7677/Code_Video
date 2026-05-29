from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.extraction import AnnotationPseudoStateExtractor, compute_scale_depth_consistency


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


if __name__ == "__main__":
    unittest.main()
