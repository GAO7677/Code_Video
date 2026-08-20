from pathlib import Path
import unittest

import bench


class ContextFrameResolutionTest(unittest.TestCase):
    def test_uses_nested_inference_effective_context_frames(self) -> None:
        record = bench.CaseRecord(
            result_json_path=Path("result.json"),
            result_payload={"inference": {"effective_context_frames": 12}},
            input_json_path=Path("input.json"),
            gt_video_path=Path("source.mp4"),
            candidate_video_path=Path("generated.mp4"),
        )

        self.assertEqual(bench.resolve_context_frames_override(record), 12)


if __name__ == "__main__":
    unittest.main()
