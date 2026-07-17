from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset_new_0705.compute_motion_degree import compute_motion_from_gray_frames


class MotionDegreeTests(unittest.TestCase):
    @staticmethod
    def _moving_square(direction: int) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for index in range(5):
            frame = np.zeros((64, 96), dtype=np.uint8)
            x0 = 12 + direction * index * 3 if direction > 0 else 60 + direction * index * 3
            cv2.rectangle(frame, (x0, 22), (x0 + 15, 37), 255, thickness=-1)
            frames.append(frame)
        return frames

    def test_static_sequence_has_zero_motion(self) -> None:
        frames = [np.zeros((64, 96), dtype=np.uint8) for _ in range(4)]
        metrics = compute_motion_from_gray_frames(frames, fps=30.0)
        self.assertLess(float(metrics["motion_global_px_per_frame"]), 1e-8)
        self.assertLess(float(metrics["motion_degree_diag_pct_per_second"]), 1e-8)

    def test_translating_object_has_positive_motion(self) -> None:
        metrics = compute_motion_from_gray_frames(self._moving_square(1), fps=30.0)
        self.assertGreater(float(metrics["motion_global_px_per_frame"]), 0.0)
        self.assertGreater(
            float(metrics["motion_active_diag_pct_per_second"]),
            float(metrics["motion_degree_diag_pct_per_second"]),
        )

    def test_left_and_right_motion_have_similar_magnitude(self) -> None:
        left = compute_motion_from_gray_frames(self._moving_square(-1), fps=30.0)
        right = compute_motion_from_gray_frames(self._moving_square(1), fps=30.0)
        left_score = float(left["motion_degree_diag_pct_per_second"])
        right_score = float(right["motion_degree_diag_pct_per_second"])
        self.assertLess(
            abs(left_score - right_score) / (0.5 * (left_score + right_score)),
            0.01,
        )

    def test_fps_normalization_converts_per_frame_motion_to_speed(self) -> None:
        frames = self._moving_square(1)
        fps_30 = compute_motion_from_gray_frames(frames, fps=30.0)
        fps_60 = compute_motion_from_gray_frames(frames, fps=60.0)
        self.assertAlmostEqual(
            float(fps_60["motion_degree_diag_pct_per_second"]),
            2.0 * float(fps_30["motion_degree_diag_pct_per_second"]),
            places=6,
        )

    def test_residual_motion_removes_global_translation(self) -> None:
        rng = np.random.default_rng(7)
        first = rng.integers(0, 256, size=(64, 96), dtype=np.uint8)
        matrix = np.float32([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]])
        second = cv2.warpAffine(first, matrix, (96, 64), borderMode=cv2.BORDER_REFLECT)
        metrics = compute_motion_from_gray_frames([first, second], fps=30.0)
        self.assertLess(
            float(metrics["motion_residual_px_per_frame"]),
            0.35 * float(metrics["motion_global_px_per_frame"]),
        )


if __name__ == "__main__":
    unittest.main()
