from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

import run_validation_30_trajectory_overlays as overlays
import trajectory_validation_preview as preview


class Validation30TrajectoryOverlayTests(unittest.TestCase):
    def test_tracker_video_is_resized_to_the_query_coordinate_system(self) -> None:
        video = torch.rand(1, 2, 3, 8, 12)

        resized = preview.resize_tracker_video(video)

        self.assertEqual(tuple(resized.shape), (1, 2, 3, 256, 448))

    def test_find_generated_video_supports_nested_entry_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = (
                root
                / "videos"
                / "trajectory_step0500"
                / "trajectory_step0500"
                / "case_01_example.mp4"
            )
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")

            found = overlays.find_generated_video(
                root, "trajectory_step0500", "case_01_example"
            )

        self.assertEqual(found, video.resolve())

    def test_find_generated_video_rejects_ambiguous_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_root = root / "videos" / "trajectory_step0500"
            for directory in ("a", "b"):
                video = video_root / directory / "case_01_example.mp4"
                video.parent.mkdir(parents=True)
                video.write_bytes(b"video")

            with self.assertRaisesRegex(RuntimeError, "found 2"):
                overlays.find_generated_video(
                    root, "trajectory_step0500", "case_01_example"
                )


if __name__ == "__main__":
    unittest.main()
