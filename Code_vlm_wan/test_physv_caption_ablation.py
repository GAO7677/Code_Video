import unittest

from physv_caption_ablation import dense_start_indices


class DenseStartIndicesTest(unittest.TestCase):
    def test_preserves_dense_early_evidence_and_full_duration(self):
        indices = dense_start_indices(total_frames=90, dense_frames=12, target_frames=64)

        self.assertEqual(indices[:12], list(range(12)))
        self.assertEqual(len(indices), 64)
        self.assertEqual(indices[-1], 89)
        self.assertEqual(indices, sorted(set(indices)))


if __name__ == "__main__":
    unittest.main()
