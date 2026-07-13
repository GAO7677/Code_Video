from __future__ import annotations

import torch

from .contracts import (
    DensePhysicsControl,
    FutureObjectState,
    ObservedObjectGraph,
    PhysicsConditionBundle,
    SparsePhysicsCondition,
)


def test_v4_condition_contracts() -> None:
    batch, observed, future, objects, points = 2, 2, 11, 4, 8
    graph = ObservedObjectGraph(
        node_features=torch.zeros(batch, observed, objects, 32),
        relation_features=torch.zeros(batch, observed, objects, objects, 16),
        boxes_xyxy=torch.zeros(batch, observed, objects, 4),
        point_tracks_xy=torch.zeros(batch, observed, objects, points, 2),
        object_valid_mask=torch.ones(batch, objects),
        observation_confidence=torch.ones(batch, observed, objects),
    )
    prediction = FutureObjectState(
        boxes_xyxy=torch.zeros(batch, future, objects, 4),
        point_tracks_xy=torch.zeros(batch, future, objects, points, 2),
        point_visibility=torch.ones(batch, future, objects, points),
        relative_depth=torch.zeros(batch, future, objects, 1),
        relation_logits=torch.zeros(batch, future, objects, objects, 6),
        event_phase_logits=torch.zeros(batch, future, 4),
        uncertainty=torch.zeros(batch, future, objects),
    )
    bundle = PhysicsConditionBundle(
        dense=DensePhysicsControl(
            features=torch.zeros(batch, 16, future, 32, 56),
            support=torch.zeros(batch, 1, future, 32, 56),
            confidence=torch.ones(batch, 1, future, 32, 56),
        ),
        sparse=SparsePhysicsCondition(
            tokens=torch.zeros(batch, 12, 3072),
            token_type_ids=torch.zeros(batch, 12, dtype=torch.long),
            support_bias=torch.zeros(batch, 12, future, 32, 56),
            token_confidence=torch.ones(batch, 12),
        ),
        global_event_tokens=torch.zeros(batch, 4, 3072),
        condition_confidence=torch.ones(batch),
    )
    graph.validate()
    prediction.validate()
    bundle.validate()


def test_contract_rejects_mismatched_object_count() -> None:
    graph = ObservedObjectGraph(
        node_features=torch.zeros(1, 2, 4, 8),
        relation_features=torch.zeros(1, 2, 3, 3, 4),
        boxes_xyxy=torch.zeros(1, 2, 4, 4),
        point_tracks_xy=torch.zeros(1, 2, 4, 2, 2),
        object_valid_mask=torch.ones(1, 4),
        observation_confidence=torch.ones(1, 2, 4),
    )
    try:
        graph.validate()
    except ValueError as error:
        assert "relation_features" in str(error)
    else:
        raise AssertionError("mismatched relation object count was accepted")
