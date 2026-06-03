from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .adapter import TinyVideoBackbone
from .conditioning import ConditionBundle, build_condition_bundle
from .config import ConditioningConfig
from .experiment import apply_condition_mode
from .projection import ConfidenceAwareProjector
from .schemas import StateIndex
from .utils import require_torch

torch = require_torch()


def rollout_boxes_from_states(last_boxes: torch.Tensor, future_states: torch.Tensor) -> torch.Tensor:
    batch, future_steps, num_objects, _ = future_states.shape
    widths = (last_boxes[..., 2] - last_boxes[..., 0]).clamp_min(1e-4)
    heights = (last_boxes[..., 3] - last_boxes[..., 1]).clamp_min(1e-4)
    aspect = widths / heights
    boxes = torch.zeros((batch, future_steps, num_objects, 4), device=future_states.device, dtype=future_states.dtype)

    for step in range(future_steps):
        centers = future_states[:, step, :, StateIndex.CENTER_X:StateIndex.CENTER_Y + 1]
        area = torch.exp(future_states[:, step, :, StateIndex.LOG_SCALE]).clamp_min(1e-4)
        width = torch.sqrt(area * aspect).clamp_min(1e-3)
        height = (area / width).clamp_min(1e-3)
        x0 = (centers[..., 0] - 0.5 * width).clamp(0.0, 1.0)
        y0 = (centers[..., 1] - 0.5 * height).clamp(0.0, 1.0)
        x1 = (centers[..., 0] + 0.5 * width).clamp(0.0, 1.0)
        y1 = (centers[..., 1] + 0.5 * height).clamp(0.0, 1.0)
        boxes[:, step] = torch.stack([x0, y0, x1, y1], dim=-1)
    return boxes


@dataclass(slots=True)
class StateConditionedGenerationPipeline:
    predictor: object
    projector: ConfidenceAwareProjector
    video_model: TinyVideoBackbone
    conditioning_config: ConditioningConfig
    condition_mode: str = "state"

    def predict_states(
        self,
        context_states: torch.Tensor,
        appearance: torch.Tensor,
        camera: torch.Tensor,
        prompts: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predictor_outputs = self.predictor(context_states, appearance, camera, prompts)
        predicted = self.projector.project(predictor_outputs["states"])
        future_latents = predictor_outputs.get("latents")
        return predicted, future_latents

    def build_conditions(
        self,
        predicted_states: torch.Tensor,
        context_boxes: torch.Tensor,
        appearance: torch.Tensor,
    ) -> tuple[ConditionBundle, torch.Tensor]:
        future_boxes = rollout_boxes_from_states(context_boxes[:, -1], predicted_states)
        bundle = build_condition_bundle(predicted_states, future_boxes, appearance, self.conditioning_config)
        return bundle, future_boxes

    def generate(
        self,
        context_frames: torch.Tensor,
        context_states: torch.Tensor,
        context_boxes: torch.Tensor,
        appearance: torch.Tensor,
        camera: torch.Tensor,
        prompts: Sequence[str],
    ):
        predicted_states, future_latents = self.predict_states(context_states, appearance, camera, prompts)
        bundle, future_boxes = self.build_conditions(predicted_states, context_boxes, appearance)
        adapter_bundle = apply_condition_mode(bundle, self.condition_mode)
        outputs = self.video_model(
            context_frames,
            adapter_bundle.maps,
            adapter_bundle.memory_tokens,
            future_latent_tokens=future_latents,
            context_states=context_states,
            prompts=prompts,
        )
        return {
            "predicted_states": predicted_states,
            "future_latents": future_latents,
            "future_boxes": future_boxes,
            "condition_maps": bundle.maps,
            "adapter_condition_maps": adapter_bundle.maps,
            "generated_frames": outputs["frames"],
            "state_logits": outputs["state_logits"],
        }
