from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.predictor_wan_state import (
    WanStateLatentPredictor,
    WanStateLatentPredictorConfig,
    wan_state_predictor_loss,
)
from phys_state_video.predictor_wan_state_v2 import (
    WanStateLatentPredictorV2,
    WanStateLatentPredictorV2Config,
    resample_temporal_states,
    wan_state_predictor_v2_loss,
)
from phys_state_video.utils import require_torch
from phys_state_video.wan_bridge import (
    _apply_clean_prefix_to_latent,
    _align_wan_frame_num,
    _build_prefix_condition_mask,
    _build_prefix_latent_noise_mask,
    _pad_future_state_tokens,
    _resample_state_tokens_to_steps,
    _resample_video_latents_to_frame_steps,
)
from phys_state_video.wan_state_v2_helpers import (
    MockLatentExtractor,
    compute_future_latent_steps,
    compute_latent_step_count,
    resample_camera_to_latent_steps,
)

torch = require_torch()


class WanStatePredictorTests(unittest.TestCase):
    def test_predictor_shapes(self):
        batch, context_steps, future_steps, num_objects = 2, 4, 5, 3
        latent_channels, latent_h, latent_w = 16, 8, 8
        camera_dim = 8

        model = WanStateLatentPredictor(
            WanStateLatentPredictorConfig(
                latent_channels=latent_channels,
                camera_dim=camera_dim,
                future_steps=future_steps,
                max_objects=num_objects,
                max_context_steps=context_steps,
                hidden_dim=64,
                state_latent_dim=32,
                num_heads=4,
                num_encoder_layers=2,
                num_decoder_layers=2,
            )
        )
        outputs = model(
            context_latents=torch.randn(batch, context_steps, latent_channels, latent_h, latent_w),
            camera=torch.randn(batch, context_steps, camera_dim),
            prompts=["object motion"] * batch,
            future_steps=future_steps,
            num_objects=num_objects,
        )
        self.assertEqual(tuple(outputs["context_state_latents"].shape), (batch, context_steps, 32))
        self.assertEqual(tuple(outputs["future_state_latents"].shape), (batch, future_steps, 32))
        self.assertEqual(tuple(outputs["context_state_predictions"].shape), (batch, context_steps, num_objects, 10))
        self.assertEqual(tuple(outputs["future_state_predictions"].shape), (batch, future_steps, num_objects, 10))

    def test_loss_is_finite(self):
        batch, context_steps, future_steps, num_objects = 1, 3, 4, 2
        model = WanStateLatentPredictor(
            WanStateLatentPredictorConfig(
                latent_channels=16,
                camera_dim=8,
                future_steps=future_steps,
                max_objects=num_objects,
                max_context_steps=context_steps,
                hidden_dim=64,
                state_latent_dim=32,
                num_heads=4,
                num_encoder_layers=2,
                num_decoder_layers=2,
            )
        )
        outputs = model(
            context_latents=torch.randn(batch, context_steps, 16, 8, 8),
            camera=torch.randn(batch, context_steps, 8),
            prompts=["test prompt"],
            future_steps=future_steps,
            num_objects=num_objects,
        )
        losses = wan_state_predictor_loss(
            outputs,
            context_target=torch.randn(batch, context_steps, num_objects, 10),
            future_target=torch.randn(batch, future_steps, num_objects, 10),
        )
        self.assertTrue(torch.isfinite(losses["loss"]).item())

    def test_prefix_mask_helpers(self):
        self.assertEqual(_align_wan_frame_num(9), 9)
        self.assertEqual(_align_wan_frame_num(10), 13)

        condition_mask = _build_prefix_condition_mask(
            total_frames=9,
            context_steps=5,
            lat_h=1,
            lat_w=1,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(condition_mask.shape), (4, 3, 1, 1))
        self.assertTrue(torch.equal(condition_mask[:, 0], torch.ones_like(condition_mask[:, 0])))
        self.assertTrue(torch.equal(condition_mask[:, 2], torch.zeros_like(condition_mask[:, 2])))

        noise = torch.randn(16, 3, 2, 2)
        latent_noise_mask = _build_prefix_latent_noise_mask(
            noise_latent=noise,
            context_steps=5,
            temporal_stride=4,
        )
        self.assertEqual(tuple(latent_noise_mask.shape), (16, 3, 2, 2))
        self.assertTrue(torch.equal(latent_noise_mask[:, 0], torch.zeros_like(latent_noise_mask[:, 0])))
        self.assertTrue(torch.equal(latent_noise_mask[:, 1], torch.zeros_like(latent_noise_mask[:, 1])))
        self.assertTrue(torch.equal(latent_noise_mask[:, 2], torch.ones_like(latent_noise_mask[:, 2])))

    def test_apply_clean_prefix_to_latent_overwrites_time_axis_only(self):
        latents = torch.full((2, 4, 3, 5), fill_value=-1.0)
        clean_prefix = torch.arange(2 * 2 * 3 * 5, dtype=torch.float32).view(2, 2, 3, 5)

        updated = _apply_clean_prefix_to_latent(latents, clean_prefix)

        self.assertTrue(torch.equal(updated[:, :2], clean_prefix))
        self.assertTrue(torch.equal(updated[:, 2:], latents[:, 2:]))
        self.assertTrue(torch.equal(updated[:, :, :, 2:], torch.cat([clean_prefix, latents[:, 2:]], dim=1)[:, :, :, 2:]))

    def test_state_token_padding(self):
        tokens = torch.randn(1, 3, 8)
        padded = _pad_future_state_tokens(tokens, target_steps=5)
        self.assertEqual(tuple(padded.shape), (1, 5, 8))
        self.assertTrue(torch.equal(padded[:, :3], tokens))
        self.assertTrue(torch.equal(padded[:, 3], tokens[:, 2]))
        self.assertTrue(torch.equal(padded[:, 4], tokens[:, 2]))

    def test_state_token_resampling(self):
        tokens = torch.tensor([[[0.0], [10.0]]])
        resized = _resample_state_tokens_to_steps(tokens, target_steps=4)
        self.assertEqual(tuple(resized.shape), (1, 4, 1))
        self.assertAlmostEqual(float(resized[0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(resized[0, -1, 0]), 10.0, places=5)
        self.assertGreater(float(resized[0, 1, 0]), 0.0)
        self.assertLess(float(resized[0, 1, 0]), 10.0)
        self.assertGreater(float(resized[0, 2, 0]), float(resized[0, 1, 0]))

    def test_video_latent_resampling(self):
        latent = torch.tensor([[[[0.0]], [[10.0]]]])
        resized = _resample_video_latents_to_frame_steps(latent, target_steps=4)
        self.assertEqual(tuple(resized.shape), (4, 1, 1, 1))
        self.assertAlmostEqual(float(resized[0, 0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(resized[-1, 0, 0, 0]), 10.0, places=5)
        self.assertGreater(float(resized[1, 0, 0, 0]), 0.0)
        self.assertGreater(float(resized[2, 0, 0, 0]), float(resized[1, 0, 0, 0]))

    def test_v2_helper_step_counts(self):
        self.assertEqual(compute_latent_step_count(4, 4), 1)
        self.assertEqual(compute_latent_step_count(5, 4), 2)
        self.assertEqual(compute_future_latent_steps(4, 6, 4), 2)

    def test_v2_mock_latent_extractor_shape(self):
        extractor = MockLatentExtractor(latent_channels=16, latent_height=8, latent_width=8, temporal_stride=4)
        frames = torch.randn(2, 4, 3, 32, 32)
        latents = extractor.encode_context_frames_raw(frames)
        self.assertEqual(tuple(latents.shape), (2, 1, 16, 8, 8))

    def test_v2_predictor_shapes(self):
        batch, context_steps, future_steps, num_objects = 2, 3, 2, 2
        model = WanStateLatentPredictorV2(
            WanStateLatentPredictorV2Config(
                latent_channels=16,
                camera_dim=8,
                max_context_latent_steps=context_steps,
                max_future_latent_steps=future_steps,
                max_objects=num_objects,
                hidden_dim=64,
                state_latent_dim=32,
                state_map_height=2,
                state_map_width=2,
                num_heads=4,
                num_encoder_layers=2,
                num_decoder_layers=2,
            )
        )
        outputs = model(
            context_latents=torch.randn(batch, context_steps, 16, 8, 8),
            camera=torch.randn(batch, context_steps, 8),
            prompts=["latent time"] * batch,
            future_latent_steps=future_steps,
            num_objects=num_objects,
        )
        self.assertEqual(tuple(outputs["context_state_latents"].shape), (batch, context_steps, 2, 2, 32))
        self.assertEqual(tuple(outputs["future_state_latents"].shape), (batch, future_steps, 2, 2, 32))
        self.assertEqual(tuple(outputs["context_object_slots"].shape), (batch, context_steps, num_objects, 32))
        self.assertEqual(tuple(outputs["future_object_slots"].shape), (batch, future_steps, num_objects, 32))
        self.assertEqual(tuple(outputs["state_tokens"].shape), (batch, future_steps * 2 * 2, 32))
        self.assertEqual(tuple(outputs["memory_tokens"].shape), (batch, num_objects, 32))
        self.assertEqual(tuple(outputs["condition_maps"].shape), (batch, future_steps, 32, 2, 2))
        self.assertEqual(tuple(outputs["future_adapter_tokens"].shape), (batch, future_steps * 2 * 2, 32))
        self.assertEqual(tuple(outputs["context_state_predictions"].shape), (batch, context_steps, num_objects, 10))
        self.assertEqual(tuple(outputs["future_state_predictions"].shape), (batch, future_steps, num_objects, 10))

    def test_v2_resampling_shapes(self):
        camera = torch.randn(1, 4, 8)
        camera_resized = resample_camera_to_latent_steps(camera, 2)
        self.assertEqual(tuple(camera_resized.shape), (1, 2, 8))

        states = torch.randn(1, 6, 2, 10)
        state_resized = resample_temporal_states(states, 2)
        self.assertEqual(tuple(state_resized.shape), (1, 2, 2, 10))

    def test_v2_loss_stages_are_finite(self):
        batch, context_steps, future_steps, num_objects = 1, 2, 3, 2
        model = WanStateLatentPredictorV2(
            WanStateLatentPredictorV2Config(
                latent_channels=16,
                camera_dim=8,
                max_context_latent_steps=context_steps,
                max_future_latent_steps=future_steps,
                max_objects=num_objects,
                hidden_dim=64,
                state_latent_dim=32,
                state_map_height=2,
                state_map_width=2,
                num_heads=4,
                num_encoder_layers=2,
                num_decoder_layers=2,
            )
        )
        outputs = model(
            context_latents=torch.randn(batch, context_steps, 16, 8, 8),
            camera=torch.randn(batch, context_steps, 8),
            prompts=["grouped head test"],
            future_latent_steps=future_steps,
            num_objects=num_objects,
        )
        context_target = torch.randn(batch, context_steps, num_objects, 10)
        future_target = torch.randn(batch, future_steps, num_objects, 10)
        for stage in ("context_only", "future_only", "joint_finetune"):
            losses = wan_state_predictor_v2_loss(
                outputs=outputs,
                context_target=context_target,
                future_target=future_target,
                train_stage=stage,
            )
            self.assertTrue(torch.isfinite(losses["loss"]).item())


if __name__ == "__main__":
    unittest.main()
