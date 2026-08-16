import tempfile
from pathlib import Path

import numpy as np
import torch

from train_xssc_object_self_attn_lora_frozen_motion_probe_latent_mask import (
    build_latent_mask_supervision,
    compute_latent_mask_training_objective,
    load_sample_gt_role_masks,
)


def _raw_sample(sample_key: str, *, frame_indices=None):
    if frame_indices is None:
        frame_indices = list(range(49))
    return {
        "video": torch.zeros(3, 49, 64, 96),
        "metadata": {
            "sample_key": sample_key,
            "sampled_frame_indices": frame_indices,
            "source_frame_count": 49,
        },
    }


def test_load_sample_gt_role_masks_uses_sample_key_and_sampled_frames():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        case_root = root / "cases" / "F1" / "case001"
        case_root.mkdir(parents=True)
        masks = np.zeros((2, 49, 64, 96), dtype=np.uint8)
        masks[0, :, 8:20, 10:28] = 1
        masks[1, :, 30:42, 50:70] = 1
        np.savez_compressed(case_root / "object_masks.npz", masks_othw=masks)

        selected, audit = load_sample_gt_role_masks(
            _raw_sample("F1/case001"),
            cache_root=root,
            mask_key="object_tracking_masks",
            object_index=1,
            expected_frames=49,
        )

        assert selected.shape == (49, 64, 96)
        assert int(selected.sum()) == 49 * 12 * 20
        assert audit["sample_key"] == "F1/case001"
        assert audit["source"] == "cache"
        assert audit["object_index"] == 1


def test_load_sample_gt_role_masks_rejects_ambiguous_frame_alignment():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        case_root = root / "cases" / "F2" / "case002"
        case_root.mkdir(parents=True)
        masks = np.ones((1, 49, 64, 96), dtype=np.uint8)
        np.savez_compressed(case_root / "object_masks.npz", masks_othw=masks)
        sample = _raw_sample("F2/case002", frame_indices=list(range(1, 50)))

        try:
            load_sample_gt_role_masks(
                sample,
                cache_root=root,
                mask_key="object_tracking_masks",
                object_index=0,
                expected_frames=49,
            )
        except ValueError as exc:
            assert "sampled frame indices" in str(exc)
        else:
            raise AssertionError("ambiguous mask/frame alignment must fail")


def test_build_latent_mask_supervision_maps_f04_to_latent_one():
    masks = np.zeros((49, 64, 96), dtype=np.uint8)
    masks[:, 16:48, 24:72] = 1

    occupancy, query_rows, query_weights, valid = build_latent_mask_supervision(
        masks,
        grid=(13, 2, 3),
        source_frame=1,
        device=torch.device("cpu"),
    )

    assert occupancy.shape == (1, 13, 2, 3)
    assert valid.tolist() == [[False, False] + [True] * 11]
    assert query_rows.numel() > 0
    assert torch.all(query_rows // 6 == 1)
    assert torch.allclose(query_weights.sum(), torch.tensor(1.0))
    support = occupancy[0, 1].flatten() > 0
    assert torch.equal(torch.sort(query_rows % 6).values, support.nonzero().flatten())


def test_latent_mask_training_objective_reaches_student_frame_heads():
    torch.manual_seed(9)
    teacher = torch.rand(1, 3, 13, 6)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True)
    student_logits = torch.randn(1, 3, 13, 6, requires_grad=True)
    student = student_logits.softmax(dim=-1)
    occupancy = torch.zeros(1, 13, 2, 3)
    occupancy[:, :, 0, 1] = 0.25
    occupancy[:, :, 1, 1] = 0.75
    valid = torch.zeros(1, 13, dtype=torch.bool)
    valid[:, 2:] = True

    result = compute_latent_mask_training_objective(
        teacher,
        student,
        object_token_occupancy_bthw=occupancy,
        valid_frames=valid,
        head_weights=torch.tensor([0.2, 0.3, 0.5]),
        lambda_mask=0.01,
    )

    assert torch.allclose(result["student_attention"].sum(-1), torch.ones(1, 13))
    assert torch.allclose(result["target"].sum(-1), torch.ones(1, 13))
    assert torch.allclose(result["loss"], 0.01 * result["raw_soft_ce"])
    result["loss"].backward()
    assert student_logits.grad is not None
    assert float(student_logits.grad.norm()) > 0.0
