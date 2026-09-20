"""Synthetic CPU contracts for observation-only coarse visual scene inputs."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from predictor_scene_adapter import CONTEXT_KEYS, load_visual_scene, make_model_inputs


class VisualSceneAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        points = np.arange(8 * 3 * 4 * 3, dtype=np.float32).reshape(8, 3, 4, 3) / 10
        y, x = np.mgrid[:3, :4]
        self.data = dict(world_midpoint=points, static_valid=np.ones((8, 3, 4), bool),
                         scale_interval=np.array([1., 1.1]), K_processed=np.eye(3),
                         RT=np.column_stack((np.eye(3), np.zeros(3))),
                         source_uv=np.stack((x * 2., y * 2.), axis=-1),
                         frame_times=np.arange(8, dtype=np.float64) / 30)
        self.report = dict(schema="observed_aabb_depth_alignment_v1",
                           status="coarse_interval_alignment_complete", training_ready=False,
                           coarse_alignment_only=True, static_scene_gt_used=False,
                           future_frames_used=False, gt_masks_used=False,
                           permitted_context_geometry_used=True, reviewed_masks=True,
                           static_points_written=True, observed_indices=list(range(8)),
                           alignment={"accepted": True}, world_midpoint_shape=list(points.shape))
        self.write()

    def write(self):
        path = self.root / "coarse_static_points.npz"
        np.savez_compressed(path, **self.data)
        self.report["output_npz_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.root / "report.json").write_text(json.dumps(self.report))

    def test_uniform_samples_keep_world_coordinates_and_origins(self):
        scene = load_visual_scene(self.root, points_per_frame=3)
        self.assertEqual(scene["scene_xyz"].shape, (24, 3))
        self.assertTrue(scene["scene_mask"].all())
        for t in range(8):
            rows = slice(3 * t, 3 * t + 3)
            selected = np.array([0, 5, 11])
            np.testing.assert_array_equal(scene["scene_xyz"][rows],
                                          self.data["world_midpoint"][t].reshape(-1, 3)[selected])
            y, x = np.divmod(selected, 4)
            np.testing.assert_array_equal(scene["source_tyx"][rows],
                                          np.column_stack((np.full(3, t), y, x)))
            np.testing.assert_array_equal(scene["source_uv"][rows], self.data["source_uv"][y, x])
        np.testing.assert_array_equal(scene["per_frame_counts"], np.full(8, 3))
        np.testing.assert_array_equal(scene["per_frame_candidate_counts"], np.full(8, 12))
        self.assertTrue(scene["coarse_alignment_only"])
        self.assertFalse(scene["training_ready"])

    def test_invalid_pixels_never_selected_and_short_frames_pad(self):
        self.data["static_valid"][:] = False
        self.data["static_valid"][2, 1, 2] = True
        self.data["world_midpoint"][~self.data["static_valid"]] = np.nan
        self.write()
        scene = load_visual_scene(self.root, 4)
        self.assertEqual(int(scene["scene_mask"].sum()), 1)
        np.testing.assert_array_equal(scene["source_tyx"][8], [2, 1, 2])
        padded = ~scene["scene_mask"]
        self.assertTrue((scene["scene_xyz"][padded] == 0).all())
        self.assertTrue((scene["source_tyx"][padded] == -1).all())
        self.assertTrue((scene["source_uv"][padded] == -1).all())

    def test_default_budget_is_1792_without_duplicate_padding(self):
        scene = load_visual_scene(self.root)
        self.assertEqual(scene["scene_xyz"].shape, (1792, 3))
        self.assertEqual(int(scene["scene_mask"].sum()), 96)
        origins = scene["source_tyx"][scene["scene_mask"]]
        self.assertEqual(len(np.unique(origins, axis=0)), 96)

    def test_sampling_is_deterministic(self):
        first, second = load_visual_scene(self.root, 5), load_visual_scene(self.root, 5)
        for key in ("scene_xyz", "scene_mask", "source_tyx", "source_uv"):
            np.testing.assert_array_equal(first[key], second[key])

    def test_empty_scene_remains_empty(self):
        self.data["static_valid"][:] = False
        self.write()
        scene = load_visual_scene(self.root, 2)
        self.assertFalse(scene["scene_mask"].any())
        self.assertTrue((scene["per_frame_counts"] == 0).all())

    def test_invalid_budgets_rejected(self):
        for count in (0, -1, 2.5, True, "3"):
            with self.subTest(count=count), self.assertRaises(ValueError):
                load_visual_scene(self.root, count)

    def test_unsafe_or_failed_report_flags_rejected(self):
        changes = {"status": "failed", "static_scene_gt_used": True,
                   "future_frames_used": True, "gt_masks_used": True,
                   "reviewed_masks": False, "training_ready": True,
                   "coarse_alignment_only": False, "static_points_written": False,
                   "permitted_context_geometry_used": False, "schema": "other"}
        for key, value in changes.items():
            old = self.report[key]
            self.report[key] = value
            self.write()
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_visual_scene(self.root, 2)
            self.report[key] = old

    def test_missing_or_nonboolean_provenance_rejected(self):
        for value in (None, 0, "false"):
            self.report["future_frames_used"] = value
            self.write()
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_visual_scene(self.root, 2)

    def test_observed_indices_and_accepted_alignment_required(self):
        self.report["observed_indices"] = list(range(1, 9))
        self.write()
        with self.assertRaises(ValueError):
            load_visual_scene(self.root, 2)
        self.report["observed_indices"] = list(range(8))
        self.report["alignment"]["accepted"] = False
        self.write()
        with self.assertRaises(ValueError):
            load_visual_scene(self.root, 2)

    def test_changed_archive_fails_hash_check(self):
        self.data["world_midpoint"][0, 0, 0, 0] += 1
        np.savez_compressed(self.root / "coarse_static_points.npz", **self.data)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_visual_scene(self.root, 2)

    def test_extra_cache_field_rejected(self):
        self.data["oracle"] = np.zeros((16, 19))
        self.write()
        with self.assertRaises(ValueError):
            load_visual_scene(self.root, 2)

    def test_invalid_shapes_masks_and_valid_values_rejected(self):
        changes = {"world_midpoint": np.zeros((9, 3, 4, 3), np.float32),
                   "static_valid": np.ones((8, 3, 4), np.float32),
                   "source_uv": np.zeros((3, 4, 3)), "RT": np.zeros((4, 4)),
                   "scale_interval": np.array([2., 1.]),
                   "frame_times": np.arange(1, 9) / 30}
        for key, value in changes.items():
            old = self.data[key]
            self.data[key] = value
            self.write()
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_visual_scene(self.root, 2)
            self.data[key] = old
        self.data["world_midpoint"][0, 0, 0, 0] = np.nan
        self.write()
        with self.assertRaises(ValueError):
            load_visual_scene(self.root, 2)


class ModelInputTests(unittest.TestCase):
    def setUp(self):
        self.context = dict(node=np.zeros((2, 90), np.float32), edge=np.zeros((2, 2, 54), np.float32),
                            mask=np.array([True, False]), surface=np.zeros((2, 4, 3), np.float32),
                            surface_mask=np.ones((2, 4), bool), history_dt=np.arange(-7, 1) / 30,
                            future_dt=np.arange(1, 42) / 30, last_position=np.zeros((2, 3), np.float32),
                            cv=np.zeros((2, 41, 3), np.float32))
        self.scene = dict(scene_xyz=np.zeros((5, 3), np.float32), scene_mask=np.ones(5, bool),
                          coarse_alignment_only=True, training_ready=False)

    def test_only_declared_model_inputs_and_values_preserved(self):
        result = make_model_inputs(self.context, self.scene)
        self.assertEqual(set(result), set(CONTEXT_KEYS) | {"scene_xyz", "scene_mask"})
        for key in CONTEXT_KEYS:
            self.assertIs(result[key], self.context[key])

    def test_target_oracle_material_pollution_does_not_change_inputs(self):
        clean = make_model_inputs(self.context, self.scene)
        polluted = {**self.context, "target_position": np.full((2, 41, 3), np.nan),
                    "target_contact": "future", "oracle": "static GT",
                    "object_material": "not permitted", "family": "barrier"}
        result = make_model_inputs(polluted, self.scene)
        for key in clean:
            self.assertIs(result[key], clean[key])

    def test_existing_scene_in_context_rejected(self):
        for key in ("scene_xyz", "scene_mask"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                make_model_inputs({**self.context, key: None}, self.scene)

    def test_missing_context_rejected(self):
        for key in CONTEXT_KEYS:
            with self.subTest(key=key), self.assertRaises(ValueError):
                make_model_inputs({k: v for k, v in self.context.items() if k != key}, self.scene)

    def test_invalid_context_shapes_and_masks_rejected(self):
        for key, value in (("node", np.zeros((1, 2, 90))), ("mask", np.ones(2)),
                           ("surface", np.zeros((2, 4, 4))), ("surface_mask", np.ones((2, 4)))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                make_model_inputs({**self.context, key: value}, self.scene)

    def test_invalid_scene_rejected(self):
        for key, value in (("scene_xyz", np.full((5, 3), np.nan)),
                           ("scene_mask", np.ones(5)), ("training_ready", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                make_model_inputs(self.context, {**self.scene, key: value})


if __name__ == "__main__":
    unittest.main()
