from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.config import AdapterConfig, ConditioningConfig, PredictorConfig
from phys_state_video.utils import require_torch

torch = require_torch()

from phys_state_video.adapter import TinyVideoBackbone
from phys_state_video.pipeline import StateConditionedGenerationPipeline
from phys_state_video.predictor import FutureStatePredictor
from phys_state_video.projection import ConfidenceAwareProjector


class TorchPipelineTests(unittest.TestCase):
    def test_pipeline_shapes(self):
        batch, context_steps, future_steps, num_objects, height, width = 1, 4, 5, 2, 32, 32
        context_frames = torch.rand(batch, context_steps, 3, height, width)
        context_states = torch.rand(batch, context_steps, num_objects, 10)
        context_boxes = torch.rand(batch, context_steps, num_objects, 4)
        context_boxes[..., 2:] = torch.maximum(context_boxes[..., 2:], context_boxes[..., :2] + 0.05)
        appearance = torch.rand(batch, num_objects, 64)
        camera = torch.rand(batch, context_steps, 8)

        predictor = FutureStatePredictor(PredictorConfig(appearance_dim=64, camera_dim=8, future_steps=future_steps))
        video_model = TinyVideoBackbone(AdapterConfig(future_steps=future_steps))
        pipeline = StateConditionedGenerationPipeline(
            predictor=predictor,
            projector=ConfidenceAwareProjector(),
            video_model=video_model,
            conditioning_config=ConditioningConfig(frame_height=height, frame_width=width),
        )
        outputs = pipeline.generate(
            context_frames=context_frames,
            context_states=context_states,
            context_boxes=context_boxes,
            appearance=appearance,
            camera=camera,
            prompts=["a red box moves"],
        )
        self.assertEqual(tuple(outputs["generated_frames"].shape), (batch, future_steps, 3, height, width))
        self.assertEqual(tuple(outputs["condition_maps"].shape), (batch, future_steps, 7, height, width))
        self.assertTrue(torch.isfinite(outputs["generated_frames"]).all().item())


if __name__ == "__main__":
    unittest.main()
