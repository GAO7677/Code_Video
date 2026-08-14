from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from frozen_motion_probe import (
    TopHeadQKCollector,
    aggregate_head_probabilities,
    blend_with_fixed_probe_noise,
    capture_wan_self_attention_qk,
    fixed_query_head_probabilities,
    load_pck_head_weights,
    pck_weighted_teacher_student_head_kl,
    query_rows_from_mask,
    query_rows_from_points,
    student_teacher_heatmap_kl,
    trajectory_huber_loss,
)


class _TinyAttention(nn.Module):
    def forward(self, q, k, v):
        return v


class _TinySelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _TinyAttention()


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _TinySelfAttention()


class _TinyProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(3, 4, bias=False)
        self.k = nn.Linear(3, 4, bias=False)
        self.blocks = nn.ModuleList([_TinyBlock()])

    def forward(self, x, collector):
        q = self.q(x)
        k = self.k(x)
        v = q
        with capture_wan_self_attention_qk(self, collector):
            self.blocks[0].self_attn.attn(q, k, v)
        return collector.finalize()


def _collector() -> TopHeadQKCollector:
    return TopHeadQKCollector(
        selected_heads_by_block={0: (0,)},
        query_rows=torch.tensor([0, 1]),
        grid=(2, 2, 2),
        expected_num_heads=2,
    )


def test_frozen_probe_keeps_input_gradient_but_not_parameter_gradient():
    torch.manual_seed(7)
    probe = _TinyProbe().requires_grad_(False).eval()
    teacher_x = torch.randn(1, 8, 3)
    student_x = torch.randn(1, 8, 3, requires_grad=True)

    with torch.no_grad():
        teacher_heatmap = probe(teacher_x, _collector()).detach()
    student_heatmap = probe(student_x, _collector())
    heatmap_loss = student_teacher_heatmap_kl(student_heatmap, teacher_heatmap)
    trajectory_loss, student_traj, teacher_traj = trajectory_huber_loss(
        student_heatmap,
        teacher_heatmap,
        delta=0.05,
    )
    (heatmap_loss + trajectory_loss).backward()

    assert student_heatmap.requires_grad
    assert not teacher_heatmap.requires_grad
    assert student_traj.shape == teacher_traj.shape == (1, 2, 2)
    assert student_x.grad is not None
    assert float(student_x.grad.abs().sum()) > 0.0
    assert all(not parameter.requires_grad for parameter in probe.parameters())
    assert all(parameter.grad is None for parameter in probe.parameters())


def test_same_probe_noise_is_exactly_shared():
    x0_a = torch.zeros(1, 2, 2)
    x0_b = torch.ones(1, 2, 2)
    epsilon = torch.full_like(x0_a, 2.0)
    a = blend_with_fixed_probe_noise(x0_a, epsilon, noise_level=0.25)
    b = blend_with_fixed_probe_noise(x0_b, epsilon, noise_level=0.25)
    assert torch.allclose(a, torch.full_like(a, 0.5))
    assert torch.allclose(b, torch.full_like(b, 1.25))


def test_qk_map_as_explicit_checkpoint_output_backpropagates():
    torch.manual_seed(11)
    probe = _TinyProbe().requires_grad_(False).eval()
    student_x = torch.randn(1, 8, 3, requires_grad=True)

    def checkpointed_block(x):
        collector = _collector()
        q = probe.q(x)
        k = probe.k(x)
        with capture_wan_self_attention_qk(probe, collector):
            output = probe.blocks[0].self_attn.attn(q, k, q)
        return output, collector.finalize_head_probabilities()

    with torch.autograd.graph.save_on_cpu():
        _, head_maps = checkpoint(
            checkpointed_block,
            student_x,
            use_reentrant=False,
        )
    head_maps[..., 0].sum().backward()
    assert student_x.grad is not None
    assert float(student_x.grad.abs().sum()) > 0.0


def test_gt_mask_and_points_map_to_fixed_query_frame():
    mask = torch.zeros(49, 8, 8)
    mask[0, 2:6, 2:6] = 1
    mask_rows = query_rows_from_mask(
        mask,
        grid=(13, 4, 4),
        query_latent_frame=0,
    )
    assert mask_rows.numel() == 4
    assert bool((mask_rows < 16).all())

    thin_mask = torch.zeros(8, 8)
    thin_mask[1, 1] = 1
    thin_rows = query_rows_from_mask(
        thin_mask,
        grid=(13, 4, 4),
        query_latent_frame=1,
    )
    assert thin_rows.tolist() == [16]

    points = torch.tensor([[0.25, 0.25], [0.75, 0.75]])
    point_rows = query_rows_from_points(
        points,
        grid=(13, 4, 4),
        query_latent_frame=2,
        image_size=None,
    )
    assert point_rows.tolist() == [37, 47]
    assert bool((torch.div(point_rows, 16, rounding_mode="floor") == 2).all())


def test_student_map_reuses_exact_teacher_query_representation():
    torch.manual_seed(19)
    teacher_q = torch.randn(1, 8, 4)
    student_q_a = torch.randn(1, 8, 4)
    student_q_b = torch.randn(1, 8, 4) * 7.0
    student_k = torch.randn(1, 8, 4, requires_grad=True)
    rows = torch.tensor([0, 1])
    fixed_teacher_q = teacher_q.reshape(1, 8, 2, 2).permute(0, 2, 1, 3)[
        :, [0], :, :
    ][:, :, rows, :]

    map_a = fixed_query_head_probabilities(
        student_q_a,
        student_k,
        head_indices=(0,),
        query_rows=rows,
        num_heads=2,
        fixed_query=fixed_teacher_q,
    )
    map_b = fixed_query_head_probabilities(
        student_q_b,
        student_k,
        head_indices=(0,),
        query_rows=rows,
        num_heads=2,
        fixed_query=fixed_teacher_q,
    )
    assert torch.equal(map_a, map_b)
    map_a[..., 0].sum().backward()
    assert student_k.grad is not None
    assert float(student_k.grad.abs().sum()) > 0.0


def test_fixed_teacher_query_loss_reaches_first_pass_v_prediction():
    torch.manual_seed(23)
    probe = _TinyProbe().requires_grad_(False).eval()
    target_x0 = torch.randn(1, 8, 3)
    latent_xt = torch.randn(1, 8, 3)
    v_pred = torch.randn(1, 8, 3, requires_grad=True)
    pred_x0 = latent_xt - 0.6 * v_pred

    teacher_collector = _collector()
    with torch.no_grad():
        teacher_heatmap = probe(target_x0, teacher_collector).detach()
        fixed_teacher_q = {
            0: teacher_collector.finalize_query_representations().detach()
        }
    student_collector = TopHeadQKCollector(
        selected_heads_by_block={0: (0,)},
        query_rows=torch.tensor([0, 1]),
        grid=(2, 2, 2),
        expected_num_heads=2,
        fixed_query_by_block=fixed_teacher_q,
    )
    student_heatmap = probe(pred_x0, student_collector)
    loss = student_teacher_heatmap_kl(student_heatmap, teacher_heatmap)
    loss.backward()

    assert v_pred.grad is not None
    assert float(v_pred.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in probe.parameters())


def test_pck_weighted_teacher_student_head_kl_matches_manual_formula():
    teacher = torch.tensor(
        [[[0.75, 0.25], [0.20, 0.80]]], dtype=torch.float32
    )
    student = torch.tensor(
        [[[0.50, 0.50], [0.40, 0.60]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    weights = torch.tensor([0.75, 0.25])
    loss, per_head = pck_weighted_teacher_student_head_kl(
        student,
        teacher,
        weights,
    )
    manual = (
        teacher * (teacher.log() - student.log())
    ).sum(dim=-1)
    assert torch.allclose(per_head, manual)
    assert torch.allclose(loss, (manual * weights).sum(dim=1).mean())
    loss.backward()
    assert student.grad is not None
    assert float(student.grad.abs().sum()) > 0.0


def test_pck_weighted_aggregate_uses_requested_head_weights():
    heads = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32
    )
    weighted = aggregate_head_probabilities(
        heads,
        grid=(1, 1, 2),
        head_weights=torch.tensor([0.8, 0.2]),
    )
    equal = aggregate_head_probabilities(heads, grid=(1, 1, 2))
    assert torch.allclose(weighted.flatten(1), torch.tensor([[0.8, 0.2]]))
    assert torch.allclose(equal.flatten(1), torch.tensor([[0.5, 0.5]]))


def test_latest3350_pck_weights_align_with_selected_top100():
    config_path = (
        Path(__file__).resolve().parent.parent
        / "configs/physiciq67_pck32_s039_latest3350_top100_heads.json"
    )
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    metadata["config_path"] = str(config_path)
    selected: dict[int, list[int]] = {}
    for target in metadata["targets"]:
        selected.setdefault(int(target["block"]), []).append(int(target["head"]))
    weights, audit = load_pck_head_weights(metadata, selected)
    assert weights.shape == (100,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0))
    assert audit["score_key"] == "pck32"
    assert audit["ranking_step"] == 39
    assert audit["completed_runs_at_selection"] == 3350
    assert audit["score_min"] > 0.0
    assert audit["weight_max"] > audit["weight_min"]
