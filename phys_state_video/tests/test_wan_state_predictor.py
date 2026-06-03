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
from phys_state_video.utils import require_torch
from phys_state_video.wan_bridge import (
    _align_wan_frame_num,
    _build_prefix_condition_mask,
    _build_prefix_latent_noise_mask,
    _pad_future_state_tokens,
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

    def test_state_token_padding(self):
        tokens = torch.randn(1, 3, 8)
        padded = _pad_future_state_tokens(tokens, target_steps=5)
        self.assertEqual(tuple(padded.shape), (1, 5, 8))
        self.assertTrue(torch.equal(padded[:, :3], tokens))
        self.assertTrue(torch.equal(padded[:, 3], tokens[:, 2]))
        self.assertTrue(torch.equal(padded[:, 4], tokens[:, 2]))


if __name__ == "__main__":
    unittest.main()
