#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import torch

from AAA_my_test.object_query_ablation_metrics.full_mask_signature_regions import (
    build_signature_partition,
    torch_signature_groups,
)
from AAA_my_test.object_query_ablation_metrics.training_free_m1_control.run_multi_object_guidance_search import (
    apply_grouped_m1_ablation,
)


def test_membership_signatures_are_disjoint_and_complete() -> None:
    masks = np.zeros((1, 2, 2, 4), dtype=bool)
    masks[0, 0, :, 0:2] = True
    masks[0, 1, :, 1:3] = True
    partition = build_signature_partition(
        masks, ("object_A", "object_B"), anchor_frames=np.asarray([0]), grid_hw=(1, 4)
    )
    counts = partition.audit()["signature_token_counts"]
    assert counts == {"object_A": 1, "object_A+object_B": 1, "object_B": 1}
    assert partition.union_rows == (0, 1, 2)


def test_shared_signature_is_its_own_m1_block() -> None:
    masks = np.zeros((1, 2, 2, 4), dtype=bool)
    masks[0, 0, :, 0:2] = True
    masks[0, 1, :, 1:3] = True
    partition = build_signature_partition(
        masks, ("object_A", "object_B"), anchor_frames=np.asarray([0]), grid_hw=(1, 4)
    )
    groups, union = torch_signature_groups(partition, torch.device("cpu"))
    assert sorted(target.item() for target, _ in groups) == [0, 1, 2]
    assert union.tolist() == [0, 1, 2]

    q = torch.zeros((1, 4, 2), dtype=torch.float32)
    k = torch.zeros_like(q)
    v = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]]])

    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        logits = query @ key.transpose(-1, -2)
        return logits.softmax(dim=-1) @ value

    baseline = attention(q, k, v)
    output = baseline.clone()
    apply_grouped_m1_ablation(
        output=output,
        q=q,
        k=k,
        v=v,
        original=attention,
        heads=[0],
        num_heads=1,
        groups=groups,
        group_batch_size=1,
    )
    expected = baseline.clone()
    expected[0, 0] -= v[0, 0] / 4
    expected[0, 1] -= v[0, 1] / 4
    expected[0, 2] -= v[0, 2] / 4
    assert torch.allclose(output, expected)
    assert torch.allclose(output[0, 3], baseline[0, 3])
