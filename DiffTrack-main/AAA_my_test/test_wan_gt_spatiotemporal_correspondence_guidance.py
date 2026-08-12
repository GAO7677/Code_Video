#!/usr/bin/env python3
"""CPU unit tests for GT spatiotemporal correspondence guidance primitives."""

from __future__ import annotations

import math
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint

from AAA_my_test.run_wan_gt_spatiotemporal_correspondence_guidance import (
    cached_object_prompt_spec,
    cached_object_phrases,
    cached_segmentation_prompt_spec,
    deduplicated_json_paths,
    masks_to_token_rows,
    normalize_guidance_gradient,
    point_correspondence_loss,
    region_correspondence_loss,
    source_anchors,
)


def qk_for_two_frame_match(correct: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [B,S,H,D] Q/K with a known cross-frame token match."""
    q = torch.zeros((1, 8, 1, 2), dtype=torch.float32, requires_grad=True)
    k = torch.zeros((1, 8, 1, 2), dtype=torch.float32, requires_grad=True)
    # Frame 0 token 0 queries frame 1. Correct target is token 3 in each 2x2 frame.
    with torch.no_grad():
        q[0, 0, 0] = torch.tensor([8.0, 0.0])
        q[0, 7, 0] = torch.tensor([0.0, 8.0])
        if correct:
            k[0, 7, 0] = torch.tensor([8.0, 0.0])
            k[0, 0, 0] = torch.tensor([0.0, 8.0])
        else:
            k[0, 4, 0] = torch.tensor([8.0, 0.0])
            k[0, 3, 0] = torch.tensor([0.0, 8.0])
    return q, k


class CorrespondenceGuidanceTests(unittest.TestCase):
    def test_cached_segmentation_prompt_spec_scales_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_dir = root / "case_a"
            case_dir.mkdir()
            (case_dir / "regions.json").write_text(
                json.dumps(
                    {
                        "height": 100,
                        "width": 200,
                        "query_context_frame": 7,
                        "selected_annotations": [
                            {
                                "region_name": "ball_0",
                                "bbox_xywh": [10, 20, 30, 40],
                                "predicted_iou": 0.9,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            phrases, boxes, scores, frame_index, source = (
                cached_segmentation_prompt_spec(
                    "case_a", (root,), target_hw=(200, 400)
                )
            )
            self.assertEqual(phrases, ["ball_0"])
            np.testing.assert_allclose(boxes, [[20, 40, 80, 120]])
            np.testing.assert_allclose(scores, [0.9])
            self.assertEqual(frame_index, 7)
            self.assertEqual(source, case_dir / "regions.json")

    def test_cached_object_prompt_spec_loads_validated_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_dir = root / "case_a"
            case_dir.mkdir()
            (case_dir / "regions.json").write_text(
                json.dumps(
                    {
                        "object_phrases": ["table", "ball"],
                        "grounding_debug": {
                            "object_prompt_boxes_xyxy": [
                                [0.0, 2.0, 8.0, 9.0],
                                [4.0, 0.0, 6.0, 2.0],
                            ],
                            "object_scores": [0.8, 0.6],
                        },
                    }
                ),
                encoding="utf-8",
            )
            phrases, boxes, scores, source = cached_object_prompt_spec(
                "case_a", (root,)
            )
            self.assertEqual(phrases, ["table", "ball"])
            np.testing.assert_allclose(boxes, [[0, 2, 8, 9], [4, 0, 6, 2]])
            np.testing.assert_allclose(scores, [0.8, 0.6])
            self.assertEqual(source, case_dir / "regions.json")

    def test_mask_tube_maps_to_flat_token_rows(self) -> None:
        masks = np.zeros((2, 4, 4), dtype=np.uint8)
        masks[0, :2, :2] = 1
        masks[1, 2:, 2:] = 1
        rows = masks_to_token_rows(masks, (2, 2))
        self.assertEqual([row.tolist() for row in rows], [[0], [3]])

    def test_cached_object_phrases_preserve_duplicate_instances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_dir = root / "case_a"
            case_dir.mkdir()
            (case_dir / "regions.json").write_text(
                json.dumps({"object_phrases": ["sphere", "cylinder", "cylinder"]}),
                encoding="utf-8",
            )
            phrases, source = cached_object_phrases("case_a", (root,))
            self.assertEqual(phrases, ["sphere", "cylinder", "cylinder"])
            self.assertEqual(source, case_dir / "regions.json")

    def test_source_anchor_policy(self) -> None:
        np.testing.assert_array_equal(source_anchors(49), np.arange(13) * 4)
        anchors = source_anchors(30)
        self.assertEqual(len(anchors), 13)
        self.assertEqual(int(anchors[0]), 0)
        self.assertEqual(int(anchors[-1]), 29)
        self.assertTrue(np.all(np.diff(anchors) >= 0))

    def test_input_list_is_stably_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "a.json", root / "b.json"
            first.write_text("{}", encoding="utf-8")
            second.write_text("{}", encoding="utf-8")
            listing = root / "list.txt"
            listing.write_text(f"{first}\n{second}\n{first}\n", encoding="utf-8")
            self.assertEqual(deduplicated_json_paths(listing), [first, second])

    def test_region_loss_rewards_gt_tube_alignment(self) -> None:
        rows = [torch.tensor([0]), torch.tensor([3])]
        q_good, k_good = qk_for_two_frame_match(True)
        q_bad, k_bad = qk_for_two_frame_match(False)
        good, good_terms = region_correspondence_loss(q_good, k_good, rows, (2, 2))
        bad, bad_terms = region_correspondence_loss(q_bad, k_bad, rows, (2, 2))
        self.assertEqual(good_terms, 2)
        self.assertEqual(bad_terms, 2)
        self.assertLess(float(good), float(bad))
        gradient = torch.autograd.grad(good, q_good)[0]
        self.assertTrue(torch.isfinite(gradient).all())

    def test_point_loss_rewards_gaussian_correspondence(self) -> None:
        q_good, k_good = qk_for_two_frame_match(True)
        q_bad, k_bad = qk_for_two_frame_match(False)
        rows = torch.tensor([[0], [3]], dtype=torch.long)
        visible = torch.ones((2, 1), dtype=torch.bool)
        good, _ = point_correspondence_loss(
            q_good, k_good, rows, visible, (2, 2), sigma_tokens=0.15
        )
        bad, _ = point_correspondence_loss(
            q_bad, k_bad, rows, visible, (2, 2), sigma_tokens=0.15
        )
        self.assertLess(float(good), float(bad))
        gradient = torch.autograd.grad(good, k_good)[0]
        self.assertTrue(torch.isfinite(gradient).all())

    def test_rms_normalization_matches_noise_scale(self) -> None:
        gradient = torch.tensor([1.0, -1.0, 1.0, -1.0]).reshape(1, 1, 1, 2, 2)
        reference = torch.full_like(gradient, 3.0)
        normalized, audit = normalize_guidance_gradient(
            gradient, reference, mode="rms", max_rms_ratio=1.0
        )
        self.assertTrue(math.isclose(float(normalized.square().mean().sqrt()), 3.0))
        self.assertTrue(math.isclose(audit["normalized_gradient_rms"], 3.0))

    def test_flowmatch_sign_descends_loss(self) -> None:
        # For L(x)=x^2/2, grad=x. FlowMatch delta_sigma is negative, so
        # velocity += positive grad must reduce |x|.
        x = torch.tensor([2.0])
        gradient = x.clone()
        delta_sigma = -0.1
        updated = x + delta_sigma * gradient
        self.assertLess(float(updated.square()), float(x.square()))

    def test_side_loss_survives_nonreentrant_checkpoint(self) -> None:
        side_losses: list[torch.Tensor] = []

        class SideLossBlock(torch.nn.Module):
            def forward(self, value: torch.Tensor) -> torch.Tensor:
                hidden = value.square()
                side_losses.append(hidden.square().mean())
                return hidden.sin()

        value = torch.tensor([2.0], requires_grad=True)
        torch.utils.checkpoint.checkpoint(
            SideLossBlock(), value, use_reentrant=False
        )
        gradient = torch.autograd.grad(side_losses[0], value)[0]
        self.assertTrue(torch.allclose(gradient, torch.tensor([32.0])))
        self.assertEqual(len(side_losses), 2)  # forward + checkpoint recomputation


if __name__ == "__main__":
    unittest.main()
