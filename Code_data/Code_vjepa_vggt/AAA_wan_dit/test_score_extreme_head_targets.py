import unittest
from pathlib import Path

from score_extreme_head_targets import targets_for_score_group


SELECTION = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan_dit_common_s_score_extremes_qk_seed851/selection.json"
)


class ScoreExtremeHeadTargetsTest(unittest.TestCase):
    def test_top_and_bottom_targets(self):
        top_name, top, top_source = targets_for_score_group(SELECTION, "top")
        bottom_name, bottom, bottom_source = targets_for_score_group(
            SELECTION, "bottom"
        )
        self.assertEqual(top_name, "S_TOP10")
        self.assertEqual(bottom_name, "S_BOTTOM10")
        self.assertEqual(len(top), 10)
        self.assertEqual(len(bottom), 10)
        self.assertEqual(len(set(top)), 10)
        self.assertEqual(len(set(bottom)), 10)
        self.assertTrue(set(top).isdisjoint(bottom))
        self.assertEqual(top_source["num_targets"], 10)
        self.assertEqual(bottom_source["num_targets"], 10)


if __name__ == "__main__":
    unittest.main()
