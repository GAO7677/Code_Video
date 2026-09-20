import unittest

import numpy as np

from observed_masks import make_prompt, select_mask, validate_geometry, validate_masks


class GeometryContractTests(unittest.TestCase):
    def geometry(self):
        positions = np.zeros((8, 1, 3), np.float32)
        positions[..., 2] = 2
        return dict(positions_world=positions, size_m=np.ones((1, 3), np.float32),
                    camera_K=np.array([[10, 0, 5], [0, 10, 5], [0, 0, 1]], np.float32),
                    camera_world_to_view=np.column_stack([np.eye(3), np.zeros(3)]),
                    mask_boxes_xyxy=np.tile([2., 2., 8., 8.], (8, 1, 1)),
                    center_pixels=np.tile([5., 5.], (8, 1, 1)),
                    frame_times=np.arange(8) / 30.)

    def test_accept_context_only_geometry(self):
        g = self.geometry()
        validate_geometry(g, np.arange(8) / 30.)

    def test_reject_extra_static_information(self):
        g = self.geometry()
        g["static_boxes"] = np.zeros((2, 3))
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_geometry(g, np.arange(8) / 30.)

    def test_reject_future_positions(self):
        g = self.geometry()
        g["positions_world"] = np.zeros((49, 1, 3))
        with self.assertRaises(ValueError):
            validate_geometry(g, np.arange(8) / 30.)

    def test_reject_changed_times(self):
        g = self.geometry()
        g["frame_times"] = np.arange(8) / 24.
        with self.assertRaisesRegex(ValueError, "times"):
            validate_geometry(g, np.arange(8) / 30.)

    def test_reject_bad_projection(self):
        g = self.geometry()
        g["center_pixels"][0, 0] = [6, 5]
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_geometry(g, np.arange(8) / 30.)

    def test_reject_mirrored_camera(self):
        g = self.geometry()
        g["camera_world_to_view"][0, 0] = -1
        with self.assertRaisesRegex(ValueError, "rotation"):
            validate_geometry(g, np.arange(8) / 30.)

    def test_reject_nonfinite(self):
        g = self.geometry()
        g["size_m"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            validate_geometry(g, np.arange(8) / 30.)


class SelectionTests(unittest.TestCase):
    def test_clip_prompt_not_mask(self):
        box, points, labels = make_prompt([-5, -5, 20, 20], [5, 5], (10, 10))
        np.testing.assert_array_equal(box, [0, 0, 9, 9])
        np.testing.assert_array_equal(points, [[5, 5]])
        np.testing.assert_array_equal(labels, [1])

    def test_offscreen_center_is_not_fabricated(self):
        self.assertIsNone(make_prompt([-5, 2, 5, 8], [-1, 5], (10, 10)))

    def test_inconsistent_prompt_rejected(self):
        self.assertIsNone(make_prompt([2, 2, 4, 4], [6, 6], (10, 10)))

    def test_choose_small_center_mask_over_leaky_high_score(self):
        masks = np.ones((2, 10, 10), bool)
        masks[0] = False
        masks[0, 4:7, 4:7] = True
        selected, q = select_mask(masks, [.7, .99], [3, 3, 7, 7], [5, 5])
        np.testing.assert_array_equal(selected, masks[0])
        self.assertTrue(q["accepted"])
        self.assertEqual(q["pixels"], 9)
        self.assertEqual(q["box_containment"], 1.)

    def test_accepted_candidates_prioritize_model_score(self):
        masks = np.zeros((2, 12, 12), bool)
        masks[0, 5:7, 5:7] = True
        masks[1, 3:8, 3:8] = True
        masks[1, 2, 5] = True
        selected, q = select_mask(masks, [.7, .95], [3, 3, 8, 8], [5, 5])
        np.testing.assert_array_equal(selected, masks[1])
        self.assertTrue(q["accepted"])
        self.assertGreaterEqual(q["box_containment"], .95)
        self.assertLess(q["box_containment"], 1.)
        self.assertEqual(q["candidate"], 1)

    def test_center_missing_candidate_not_accepted(self):
        masks = np.zeros((1, 10, 10), bool)
        masks[0, 3:5, 3:5] = True
        _, q = select_mask(masks, [.9], [2, 2, 8, 8], [6, 6])
        self.assertFalse(q["accepted"])

    def test_leaky_mask_preserved_and_flagged(self):
        masks = np.ones((1, 10, 10), bool)
        selected, q = select_mask(masks, [.9], [3, 3, 7, 7], [5, 5])
        np.testing.assert_array_equal(selected, masks[0])
        self.assertFalse(q["accepted"])
        self.assertEqual(q["box_containment"], .25)

    def test_empty_mask_is_unknown(self):
        masks = np.zeros((1, 10, 10), bool)
        _, q = select_mask(masks, [.9], [3, 3, 7, 7], [5, 5])
        self.assertFalse(q["accepted"])
        self.assertEqual(q["pixels"], 0)

    def test_logits_not_silently_thresholded(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            select_mask(np.full((1, 10, 10), .3), [.9], [3, 3, 7, 7], [5, 5])

    def test_nonfinite_score_rejected(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            select_mask(np.ones((1, 10, 10), bool), [np.nan], [3, 3, 7, 7], [5, 5])

    def test_union_contract(self):
        per_object = np.zeros((8, 2, 10, 10), bool)
        per_object[:, 0, 3:5, 3:5] = True
        accepted = np.zeros((8, 2), bool)
        accepted[:, 0] = True
        validate_masks(per_object, per_object.any(1), accepted)
        with self.assertRaisesRegex(ValueError, "Union"):
            validate_masks(per_object, np.zeros((8, 10, 10), bool), accepted)

    def test_empty_accepted_mask_rejected(self):
        per_object = np.zeros((8, 1, 10, 10), bool)
        with self.assertRaisesRegex(ValueError, "Empty"):
            validate_masks(per_object, per_object.any(1), np.ones((8, 1), bool))


if __name__ == "__main__":
    unittest.main()
