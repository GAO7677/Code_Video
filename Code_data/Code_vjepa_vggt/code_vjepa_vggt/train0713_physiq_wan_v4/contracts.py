from __future__ import annotations

from dataclasses import dataclass

import torch


def _require_shape(name: str, tensor: torch.Tensor, ndim: int, suffix: tuple[int, ...] = ()) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {list(tensor.shape)}")
    if suffix and tuple(int(value) for value in tensor.shape[-len(suffix) :]) != suffix:
        raise ValueError(f"{name} must end with {suffix}, got {list(tensor.shape)}")


def _require_prefix(name: str, tensor: torch.Tensor, prefix: tuple[int, ...]) -> None:
    actual = tuple(int(value) for value in tensor.shape[: len(prefix)])
    if actual != prefix:
        raise ValueError(f"{name} must start with {prefix}, got {list(tensor.shape)}")


@dataclass(frozen=True)
class ObservedObjectGraph:
    """Typed prefix observations before future prediction."""

    node_features: torch.Tensor  # [B, To, O, Dn]
    relation_features: torch.Tensor  # [B, To, O, O, Dr]
    boxes_xyxy: torch.Tensor  # [B, To, O, 4]
    point_tracks_xy: torch.Tensor  # [B, To, O, P, 2]
    object_valid_mask: torch.Tensor  # [B, O]
    observation_confidence: torch.Tensor  # [B, To, O]

    def validate(self) -> None:
        _require_shape("node_features", self.node_features, 4)
        _require_shape("relation_features", self.relation_features, 5)
        _require_shape("boxes_xyxy", self.boxes_xyxy, 4, (4,))
        _require_shape("point_tracks_xy", self.point_tracks_xy, 5, (2,))
        _require_shape("object_valid_mask", self.object_valid_mask, 2)
        _require_shape("observation_confidence", self.observation_confidence, 3)
        batch, observed, objects = (int(value) for value in self.node_features.shape[:3])
        _require_prefix("relation_features", self.relation_features, (batch, observed, objects, objects))
        _require_prefix("boxes_xyxy", self.boxes_xyxy, (batch, observed, objects))
        _require_prefix("point_tracks_xy", self.point_tracks_xy, (batch, observed, objects))
        _require_prefix("object_valid_mask", self.object_valid_mask, (batch, objects))
        _require_prefix("observation_confidence", self.observation_confidence, (batch, observed, objects))


@dataclass(frozen=True)
class FutureObjectState:
    """Prefix-only prediction over the target latent-time horizon."""

    boxes_xyxy: torch.Tensor  # [B, Tf, O, 4]
    point_tracks_xy: torch.Tensor  # [B, Tf, O, P, 2]
    point_visibility: torch.Tensor  # [B, Tf, O, P]
    relative_depth: torch.Tensor  # [B, Tf, O, 1]
    relation_logits: torch.Tensor  # [B, Tf, O, O, R]
    event_phase_logits: torch.Tensor  # [B, Tf, E]
    uncertainty: torch.Tensor  # [B, Tf, O]

    def validate(self) -> None:
        _require_shape("boxes_xyxy", self.boxes_xyxy, 4, (4,))
        _require_shape("point_tracks_xy", self.point_tracks_xy, 5, (2,))
        _require_shape("point_visibility", self.point_visibility, 4)
        _require_shape("relative_depth", self.relative_depth, 4, (1,))
        _require_shape("relation_logits", self.relation_logits, 5)
        _require_shape("event_phase_logits", self.event_phase_logits, 3)
        _require_shape("uncertainty", self.uncertainty, 3)
        batch, future, objects = (int(value) for value in self.boxes_xyxy.shape[:3])
        points = int(self.point_tracks_xy.shape[3])
        _require_prefix("point_tracks_xy", self.point_tracks_xy, (batch, future, objects, points))
        _require_prefix("point_visibility", self.point_visibility, (batch, future, objects, points))
        _require_prefix("relative_depth", self.relative_depth, (batch, future, objects))
        _require_prefix("relation_logits", self.relation_logits, (batch, future, objects, objects))
        _require_prefix("event_phase_logits", self.event_phase_logits, (batch, future))
        _require_prefix("uncertainty", self.uncertainty, (batch, future, objects))


@dataclass(frozen=True)
class DensePhysicsControl:
    """Spatially and temporally aligned control for a VACE-style side branch."""

    features: torch.Tensor  # [B, C, Tf, H, W]
    support: torch.Tensor  # [B, 1, Tf, H, W]
    confidence: torch.Tensor  # [B, 1, Tf, H, W]

    def validate(self) -> None:
        _require_shape("features", self.features, 5)
        _require_shape("support", self.support, 5)
        _require_shape("confidence", self.confidence, 5)
        batch, _, future, height, width = (int(value) for value in self.features.shape)
        _require_prefix("support", self.support, (batch, 1, future, height, width))
        _require_prefix("confidence", self.confidence, (batch, 1, future, height, width))


@dataclass(frozen=True)
class SparsePhysicsCondition:
    """Semantic tokens and their patch-level regional supports."""

    tokens: torch.Tensor  # [B, L, D]
    token_type_ids: torch.Tensor  # [B, L]
    support_bias: torch.Tensor  # [B, L, Tf, H, W]
    token_confidence: torch.Tensor  # [B, L]

    def validate(self) -> None:
        _require_shape("tokens", self.tokens, 3)
        _require_shape("token_type_ids", self.token_type_ids, 2)
        _require_shape("support_bias", self.support_bias, 5)
        _require_shape("token_confidence", self.token_confidence, 2)
        batch, tokens = (int(value) for value in self.tokens.shape[:2])
        _require_prefix("token_type_ids", self.token_type_ids, (batch, tokens))
        _require_prefix("support_bias", self.support_bias, (batch, tokens))
        _require_prefix("token_confidence", self.token_confidence, (batch, tokens))


@dataclass(frozen=True)
class PhysicsConditionBundle:
    """Explicit v4 condition boundary consumed by Wan injection."""

    dense: DensePhysicsControl
    sparse: SparsePhysicsCondition
    global_event_tokens: torch.Tensor  # [B, G, D]
    condition_confidence: torch.Tensor  # [B]

    def validate(self) -> None:
        self.dense.validate()
        self.sparse.validate()
        _require_shape("global_event_tokens", self.global_event_tokens, 3)
        _require_shape("condition_confidence", self.condition_confidence, 1)
        batch = int(self.dense.features.shape[0])
        if int(self.sparse.tokens.shape[0]) != batch:
            raise ValueError("dense and sparse condition batch sizes differ")
        _require_prefix("global_event_tokens", self.global_event_tokens, (batch,))
        _require_prefix("condition_confidence", self.condition_confidence, (batch,))
        if int(self.global_event_tokens.shape[-1]) != int(self.sparse.tokens.shape[-1]):
            raise ValueError("global event and sparse token dimensions differ")
