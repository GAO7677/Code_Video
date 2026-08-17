import json
import unittest

from PIL import Image

from infer_physv_evidence_pipeline import event_time_keys, event_window_indices, parse_event_probe
from physv_caption_ablation import dense_start_indices
from physv_caption_evidence_probe import split_storyboard


class DenseStartIndicesTest(unittest.TestCase):
    def test_preserves_dense_early_evidence_and_full_duration(self):
        indices = dense_start_indices(total_frames=90, dense_frames=12, target_frames=64)

        self.assertEqual(indices[:12], list(range(12)))
        self.assertEqual(len(indices), 64)
        self.assertEqual(indices[-1], 89)
        self.assertEqual(indices, sorted(set(indices)))


class StoryboardSplitTest(unittest.TestCase):
    def test_splits_a_two_by_three_storyboard_in_temporal_order(self):
        board = Image.new("RGB", (6, 4))
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for index, color in enumerate(colors):
            x = (index % 3) * 2
            y = (index // 3) * 2
            board.paste(Image.new("RGB", (2, 2), color=color), (x, y))

        tiles = split_storyboard(board)

        self.assertEqual(len(tiles), 6)
        self.assertEqual(tiles[0].getpixel((0, 0)), colors[0])
        self.assertEqual(tiles[-1].getpixel((0, 0)), colors[-1])


class EventWindowIndicesTest(unittest.TestCase):
    def test_shifts_a_boundary_window_without_duplicates(self):
        self.assertEqual(event_window_indices(total_frames=90, center=2, window_size=6), [0, 1, 2, 3, 4, 5])
        self.assertEqual(event_window_indices(total_frames=90, center=89, window_size=6), [84, 85, 86, 87, 88, 89])


class EventProbeTest(unittest.TestCase):
    def test_accepts_the_expected_ordered_time_keys(self):
        indices = [0, 1, 2, 3, 4, 5]
        keys = event_time_keys(indices, source_fps=30.0)
        expected = {key: "可见状态" for key in keys}

        self.assertEqual(
            parse_event_probe(json.dumps(expected, ensure_ascii=False), indices, source_fps=30.0),
            expected,
        )

    def test_rejects_an_incomplete_event_probe(self):
        with self.assertRaises(ValueError):
            parse_event_probe('{"t=0.00s":"可见状态"}', [0, 1, 2, 3, 4, 5], source_fps=30.0)


if __name__ == "__main__":
    unittest.main()
