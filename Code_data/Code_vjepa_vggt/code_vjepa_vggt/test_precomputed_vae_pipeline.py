from __future__ import annotations

from types import MethodType, SimpleNamespace
import unittest

import torch
from PIL import Image

from code_vjepa_vggt.train_v_newtrain import WanTrainingModule


class PrecomputedVaePipelineTests(unittest.TestCase):
    def test_cached_latent_disables_input_video_encoding(self) -> None:
        module = WanTrainingModule.__new__(WanTrainingModule)
        torch.nn.Module.__init__(module)
        module.pipe = SimpleNamespace(device=torch.device("cpu"))
        module.use_gradient_checkpointing = False
        module.use_gradient_checkpointing_offload = False
        module.max_timestep_boundary = 1.0
        module.min_timestep_boundary = 0.0
        module.extra_inputs = []

        def fixed_context_spec(_self, video, raw_sample=None):
            return {
                "frame_indices": [],
                "ctx_max_length": None,
                "sampled_ctx_last_index": -1,
                "sampled_ctx_num_frames": 0,
                "mode": "text_only",
            }

        module.sample_context_spec = MethodType(fixed_context_spec, module)
        latent = torch.zeros((48, 13, 32, 56), dtype=torch.bfloat16)
        inputs_shared, _, _ = module.get_pipeline_inputs(
            {
                "prompt": "test",
                "video": [Image.new("RGB", (896, 512)) for _ in range(49)],
                "precomputed_input_latents": latent,
            }
        )
        self.assertIsNone(inputs_shared["input_video"])
        self.assertIs(inputs_shared["input_latents"]._base, latent)
        self.assertEqual(inputs_shared["input_latents"].shape, (1, 48, 13, 32, 56))


if __name__ == "__main__":
    unittest.main()
