#!/usr/bin/env python3
"""CPU unit tests for GT spatiotemporal correspondence guidance primitives."""

from __future__ import annotations

import math
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.utils.checkpoint

from AAA_my_test.run_wan_gt_spatiotemporal_correspondence_guidance import (
    GuidanceTarget,
    cached_object_prompt_spec,
    cached_object_phrases,
    cached_segmentation_prompt_spec,
    correspondence_component_modes,
    deduplicated_json_paths,
    masks_to_token_rows,
    mean_component_gradients,
    load_target_map,
    normalize_guidance_gradient,
    point_correspondence_loss,
    region_correspondence_loss,
    serializable_arguments,
    source_anchors,
    target_names_for_case,
    trajectory_metrics,
)
from AAA_my_test.analyze_wan_gt_guidance_frozen_validation import analyze


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
    def test_combined_guidance_uses_sequential_equivalent_gradients(self) -> None:
        self.assertEqual(correspondence_component_modes("region"), ("region",))
        self.assertEqual(
            correspondence_component_modes("combined"), ("region", "point")
        )

        value = torch.tensor([1.25, -0.75], requires_grad=True)
        region_loss = value.square().sum()
        point_loss = (value - 2.0).square().mean()
        joint_gradient = torch.autograd.grad(
            0.5 * (region_loss + point_loss), value
        )[0]

        sequential_value = value.detach().clone().requires_grad_(True)
        region_gradient = torch.autograd.grad(
            sequential_value.square().sum(), sequential_value
        )[0]
        point_gradient = torch.autograd.grad(
            (sequential_value - 2.0).square().mean(), sequential_value
        )[0]
        sequential_gradient = mean_component_gradients(
            (region_gradient, point_gradient)
        )
        self.assertTrue(torch.allclose(sequential_gradient, joint_gradient))

    def test_run_config_arguments_serialize_optional_path(self) -> None:
        arguments = SimpleNamespace(
            input_list=Path("/tmp/input.txt"),
            head_ranking=Path("/tmp/ranking.json"),
            output_root=Path("/tmp/output"),
            target_map=Path("/tmp/eligibility.json"),
            ordinary_value=3,
        )
        payload = serializable_arguments(arguments)
        self.assertEqual(payload["target_map"], "/tmp/eligibility.json")
        json.dumps(payload)

    def test_frozen_analysis_keeps_gate_failures_out_of_ade_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            screen = root / "screening" / "seed_47326"
            screen.mkdir(parents=True)
            eligible_rows = [
                {"case": case, "target": "object_A", "eligible": True}
                for case in ("case_a", "case_b")
            ]
            (screen / "baseline_eligibility.json").write_text(
                json.dumps(
                    {
                        "eligible_case_count": 2,
                        "eligible_target_count": 2,
                        "targets": eligible_rows,
                    }
                ),
                encoding="utf-8",
            )

            def write_metric(case: str, variant: str, row: dict) -> None:
                output = root / "generations" / case / "seed_47326" / variant
                output.mkdir(parents=True)
                (output / "trajectory_metrics.json").write_text(
                    json.dumps({"metrics": [{"target": "object_A", **row}]}),
                    encoding="utf-8",
                )

            baseline = {
                "quality_pass": True,
                "ade_d0": 1.0,
                "fde_d0": 1.0,
                "pck_10pct_d0": 0.2,
                "future_track_loss_score_0_100": 0.0,
            }
            improved = {
                "quality_pass": True,
                "ade_d0": 0.8,
                "fde_d0": 0.9,
                "pck_10pct_d0": 0.3,
                "future_track_loss_score_0_100": 0.0,
            }
            failed = {
                "quality_pass": False,
                "ade_d0": None,
                "fde_d0": None,
                "pck_10pct_d0": None,
                "future_track_loss_score_0_100": 100.0,
            }
            for case in ("case_a", "case_b"):
                write_metric(case, "baseline", baseline)
                write_metric(case, "region__object_A__lambda0p1", improved)
                write_metric(case, "point__object_A__lambda0p1", failed)

            report = analyze(root, 47326, (0.1,))
            self.assertEqual(report["trigger_modes"], ["region"])
            point = next(
                row
                for row in report["aggregate"]
                if row["mode"] == "point" and row["lambda"] == 0.1
            )
            self.assertEqual(point["fully_evaluable_case_count"], 0)
            self.assertEqual(point["case_balanced_mean_delta_track_loss"], 100.0)

            for case in ("case_a", "case_b"):
                write_metric(case, "region__object_A__lambda0p05", improved)
                write_metric(case, "region__object_A__lambda0p2", improved)
            sensitivity_report = analyze(root, 47326, (0.05, 0.1, 0.2))
            self.assertEqual(
                {
                    (row["lambda"], row["mode"])
                    for row in sensitivity_report["aggregate"]
                },
                {
                    (0.05, "region"),
                    (0.1, "region"),
                    (0.1, "point"),
                    (0.1, "combined"),
                    (0.2, "region"),
                },
            )

    def test_screening_target_map_is_case_specific(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eligibility.json"
            path.write_text(
                json.dumps(
                    {
                        "eligible_jobs": [
                            {"case": "case_a", "targets": ["object_A"]},
                            {"case": "case_b", "targets": ["object_B", "moving_union"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            mapping = load_target_map(path)
            self.assertEqual(mapping["case_a"], ("object_A",))
            self.assertEqual(
                target_names_for_case("case_b", None, mapping),
                ("object_B", "moving_union"),
            )

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

    def test_region_loss_skips_missing_mask_anchors(self) -> None:
        masks = np.zeros((3, 4, 4), dtype=np.uint8)
        masks[0, :2, :2] = 1
        masks[2, 2:, 2:] = 1
        rows = masks_to_token_rows(masks, (2, 2))
        self.assertEqual([row.tolist() for row in rows], [[0], [], [3]])

        q = torch.randn((1, 12, 1, 2), requires_grad=True)
        k = torch.randn((1, 12, 1, 2), requires_grad=True)
        loss, terms = region_correspondence_loss(q, k, rows, (2, 2))
        self.assertEqual(terms, 2)  # only 0 -> 2 and 2 -> 0 remain valid
        gradient = torch.autograd.grad(loss, q)[0]
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

    def test_trajectory_metrics_exclude_condition_frame_and_gate_track_loss(self) -> None:
        point_count = 4
        reference = np.zeros((13, point_count, 2), dtype=np.float32)
        candidate = np.zeros((49, point_count, 2), dtype=np.float32)
        candidate[np.arange(13) * 4, :, 0] = np.r_[0.0, np.full(12, 2.0)][:, None]
        masks = np.zeros((1, 13, 10, 10), dtype=np.uint8)
        masks[:, :, 2:6, 2:6] = 1
        tube = SimpleNamespace(
            tracks_tn2=reference,
            visibility_tn=np.ones((13, point_count), dtype=bool),
            masks_othw=masks,
            point_starts=np.asarray([0]),
            point_ends=np.asarray([point_count]),
        )
        target = GuidanceTarget("object_A", (0,))

        lost_visibility = np.zeros((49, point_count), dtype=bool)
        lost_visibility[0] = True
        lost = trajectory_metrics(candidate, lost_visibility, tube, target)
        self.assertFalse(lost["quality_pass"])
        self.assertEqual(lost["future_common_anchor_count"], 0)
        self.assertEqual(lost["future_track_loss_score_0_100"], 100.0)
        self.assertIsNone(lost["ade_px"])
        self.assertIsNone(lost["pck_10pct_d0"])

        full_visibility = np.zeros((49, point_count), dtype=bool)
        full_visibility[np.arange(13) * 4] = True
        tracked = trajectory_metrics(candidate, full_visibility, tube, target)
        self.assertTrue(tracked["quality_pass"])
        self.assertEqual(tracked["future_common_anchor_count"], 12)
        self.assertAlmostEqual(tracked["ade_px"], 2.0)
        self.assertAlmostEqual(tracked["raw_ade_px"], 2.0)

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
