"""Capacity-controlled Stage-1 state predictors and frozen GT probes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional

from . import DYNAMIC_DIM, SLOT_DIM, STATIC_DIM


REPRESENTATIONS = {"dyn", "dyn_static", "full"}
CONTEXT_MODES = {"individual", "set"}


@dataclass
class SlotNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def normalize(self, slots: torch.Tensor) -> torch.Tensor:
        return (slots - self.mean.to(slots)) / self.std.to(slots).clamp_min(1e-6)

    def denormalize(self, slots: torch.Tensor) -> torch.Tensor:
        return slots * self.std.to(slots).clamp_min(1e-6) + self.mean.to(slots)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    @classmethod
    def from_state_dict(cls, payload: dict[str, torch.Tensor]):
        return cls(mean=payload["mean"].float(), std=payload["std"].float())


def representation_target(slots: torch.Tensor, representation: str) -> torch.Tensor:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unsupported representation: {representation}")
    if slots.shape[-1] != SLOT_DIM:
        raise ValueError(f"Expected {SLOT_DIM}-d slots, got {slots.shape[-1]}")
    if representation in {"dyn", "dyn_static"}:
        return slots[..., STATIC_DIM:]
    return slots


def compose_full_state(
    previous_full: torch.Tensor,
    predicted_target: torch.Tensor,
    representation: str,
) -> torch.Tensor:
    if representation in {"dyn", "dyn_static"}:
        if predicted_target.shape[-1] != DYNAMIC_DIM:
            raise ValueError("Dynamic prediction has the wrong channel count")
        return torch.cat([previous_full[..., :STATIC_DIM], predicted_target], dim=-1)
    if predicted_target.shape[-1] != SLOT_DIM:
        raise ValueError("Full prediction has the wrong channel count")
    return predicted_target


class StatePredictor(nn.Module):
    """Shared temporal predictor with capacity-matched context grouping."""

    def __init__(
        self,
        representation: str,
        history: int,
        context_mode: str,
        model_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        temporal_layers: int = 2,
        context_layers: int = 2,
        dropout: float = 0.1,
        max_history: int = 4,
    ):
        super().__init__()
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Unsupported representation: {representation}")
        if context_mode not in CONTEXT_MODES:
            raise ValueError(f"Unsupported context mode: {context_mode}")
        if history not in {1, 2, 4} or history > max_history:
            raise ValueError(f"Unsupported history: {history}")
        self.representation = representation
        self.history = int(history)
        self.context_mode = context_mode
        self.max_history = int(max_history)
        self.output_dim = SLOT_DIM if representation == "full" else DYNAMIC_DIM

        if representation == "dyn":
            self.input_adapter = nn.Linear(DYNAMIC_DIM, model_dim)
            self.static_adapter = None
        elif representation == "dyn_static":
            self.input_adapter = nn.Linear(DYNAMIC_DIM, model_dim)
            self.static_adapter = nn.Linear(STATIC_DIM, model_dim, bias=False)
        else:
            self.input_adapter = nn.Linear(SLOT_DIM, model_dim)
            self.static_adapter = None

        self.relative_time = nn.Parameter(torch.zeros(max_history, model_dim))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            temporal_layer, num_layers=temporal_layers, enable_nested_tensor=False
        )
        context_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            context_layer, num_layers=context_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_head = nn.Linear(model_dim, self.output_dim)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        history_full: torch.Tensor,
        slot_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the next representation from `[B,H,K,512]` history."""
        if history_full.ndim != 4 or history_full.shape[-1] != SLOT_DIM:
            raise ValueError(
                f"history_full must be [B,H,K,{SLOT_DIM}], got {history_full.shape}"
            )
        batch, supplied_history, num_slots, _ = history_full.shape
        if supplied_history < self.history:
            raise ValueError(
                f"Model needs {self.history} states, got {supplied_history}"
            )
        history_full = history_full[:, -self.history :]
        dynamic = history_full[..., STATIC_DIM:]
        if self.representation == "full":
            tokens = self.input_adapter(history_full)
        else:
            tokens = self.input_adapter(dynamic)
            if self.static_adapter is not None:
                static_condition = history_full[:, -1:, :, :STATIC_DIM]
                tokens = tokens + self.static_adapter(static_condition)

        positions = self.relative_time[-self.history :]
        tokens = tokens + positions[None, :, None, :]
        tokens = tokens.permute(0, 2, 1, 3).reshape(
            batch * num_slots, self.history, -1
        )
        encoded = self.temporal(tokens)[:, -1].reshape(batch, num_slots, -1)

        if self.context_mode == "individual":
            context_input = encoded.reshape(batch * num_slots, 1, -1)
            context_output = self.context(context_input).reshape(batch, num_slots, -1)
        else:
            padding_mask = None if slot_valid is None else ~slot_valid.bool()
            if padding_mask is not None and bool(padding_mask.all(dim=1).any()):
                raise ValueError("A sample cannot mask every slot")
            context_output = self.context(
                encoded, src_key_padding_mask=padding_mask
            )

        residual = self.output_head(self.output_norm(context_output))
        current = representation_target(history_full[:, -1], self.representation)
        return current + residual


def normalized_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    representation: str,
    slot_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    error = (prediction - target).square()
    if representation == "full":
        static_loss = error[..., :STATIC_DIM].mean(dim=-1)
        dynamic_loss = error[..., STATIC_DIM:].mean(dim=-1)
        error = 0.5 * static_loss + 0.5 * dynamic_loss
    else:
        error = error.mean(dim=-1)
    if slot_valid is None:
        return error.mean()
    selected = error[slot_valid.bool()]
    if not selected.numel():
        raise ValueError("No valid slots in prediction loss")
    return selected.mean()


class FrozenGTProbes(nn.Module):
    """Small supervised readouts used only to interpret frozen slot states."""

    def __init__(self, representation: str):
        super().__init__()
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Unsupported representation: {representation}")
        self.representation = representation
        input_dim = DYNAMIC_DIM if representation == "dyn" else SLOT_DIM
        self.norm = nn.LayerNorm(input_dim)
        self.position = nn.Linear(input_dim, 3)
        self.velocity = nn.Linear(input_dim, 3)
        self.image_position = nn.Linear(input_dim, 2)
        self.bbox = nn.Linear(input_dim, 4)
        self.presence = nn.Linear(input_dim, 1)

    def select(self, full_slots: torch.Tensor) -> torch.Tensor:
        if self.representation == "dyn":
            return full_slots[..., STATIC_DIM:]
        return full_slots

    def forward(self, full_slots: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.norm(self.select(full_slots))
        return {
            "position": self.position(value),
            "velocity": self.velocity(value),
            "image_position": self.image_position(value),
            "bbox": self.bbox(value).sigmoid(),
            "presence_logit": self.presence(value).squeeze(-1),
        }


def bbox_iou(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    left_top = torch.maximum(prediction[..., :2], target[..., :2])
    right_bottom = torch.minimum(prediction[..., 2:], target[..., 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    pred_area = (prediction[..., 2:] - prediction[..., :2]).clamp_min(0).prod(-1)
    target_area = (target[..., 2:] - target[..., :2]).clamp_min(0).prod(-1)
    return intersection / (pred_area + target_area - intersection).clamp_min(1e-8)


def probe_loss(
    output: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    mapped_valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    visible = mapped_valid & target["presence"].bool()
    mapped = mapped_valid.bool()
    if not bool(mapped.any()):
        raise ValueError("Probe batch has no mapped objects")

    losses = {
        "position": functional.mse_loss(
            output["position"][mapped], target["position"][mapped]
        ),
        "velocity": functional.mse_loss(
            output["velocity"][mapped], target["velocity"][mapped]
        ),
        "image_position": functional.mse_loss(
            output["image_position"][mapped], target["image_position"][mapped]
        ),
        "presence": functional.binary_cross_entropy_with_logits(
            output["presence_logit"][mapped], target["presence"][mapped].float()
        ),
    }
    if bool(visible.any()):
        losses["bbox"] = functional.smooth_l1_loss(
            output["bbox"][visible], target["bbox"][visible]
        )
    else:
        losses["bbox"] = output["bbox"].sum() * 0
    return sum(losses.values()), losses
