from __future__ import annotations

import torch
import torch.nn as nn

from code_vjepa_vggt.models.object_entity_id_binder import (
    EntityIDBindingObjectConditionAdapter,
)
from code_vjepa_vggt.train0715_scheme_d_object_tube_resampler.models import (
    BottleneckObjectCrossAttention,
    ObjectTubeResampler,
    TrajectoryTokenEncoder,
    fourier_encode,
    install_bottleneck_object_cross_attention,
    parse_block_ids,
    prune_object_cross_attention_blocks,
)


def _inputs():
    torch.manual_seed(7)
    batch, track_frames, objects, points = 1, 8, 3, 4
    height, width = 32, 48
    tracks = torch.rand(batch, track_frames, objects, points, 2)
    tracks[..., 0] *= width - 1
    tracks[..., 1] *= height - 1
    visibility = torch.ones(batch, track_frames, objects, points)
    confidence = torch.ones_like(visibility)
    return {
        "jepa_patch_tokens": torch.randn(1, 4, 6, 8, 12),
        "context_latents": torch.randn(1, 6, 2, 4, 6),
        "tracks": tracks,
        "visibility": visibility,
        "confidence": confidence,
        "track_image_hw": (height, width),
        "object_valid_mask": torch.tensor([[1.0, 1.0, 0.0]]),
        "box_prior_xyxy": torch.tensor(
            [[[0.1, 0.1, 0.3, 0.3], [0.5, 0.2, 0.7, 0.4], [0.0, 0.0, 0.0, 0.0]]]
        ),
    }


def _model(num_output_tokens: int = 4) -> ObjectTubeResampler:
    return ObjectTubeResampler(
        jepa_dim=12,
        latent_dim=6,
        output_dim=32,
        hidden_dim=32,
        num_output_tokens=num_output_tokens,
        num_heads=4,
        num_layers=2,
        max_objects=3,
        max_points=4,
        modality_dropout_prob=0.0,
    )


def test_output_shape_and_invalid_slot_mask() -> None:
    model = _model().eval()
    output = model(**_inputs())
    assert output.object_latent_tokens.shape == (1, 4, 3, 32)
    assert output.active_track_summary.shape == (1, 4, 3, 6)
    assert output.active_box_xyxy.shape == (1, 4, 3, 4)
    assert torch.count_nonzero(output.object_latent_tokens[:, :, 2]) == 0
    assert torch.isfinite(output.object_latent_tokens).all()
    diagnostics = model.pop_diagnostics()
    assert diagnostics is not None
    assert diagnostics.source_tokens_per_object == (2 + 4) * 4 + 4
    assert diagnostics.output_tokens_per_object == 4
    assert diagnostics.motion_tokens_per_object == 4
    assert diagnostics.valid_objects == 2


def test_resampling_is_independent_across_slots() -> None:
    model = _model().eval()
    inputs = _inputs()
    first = model(**inputs).object_latent_tokens.detach()
    changed = dict(inputs)
    changed["tracks"] = inputs["tracks"].clone()
    changed["tracks"][:, :, 1] = torch.flip(changed["tracks"][:, :, 1], dims=[1])
    second = model(**changed).object_latent_tokens.detach()
    torch.testing.assert_close(first[:, :, 0], second[:, :, 0], rtol=0.0, atol=0.0)
    assert not torch.equal(first[:, :, 1], second[:, :, 1])


def test_resampling_is_invariant_to_track_point_order() -> None:
    model = _model().eval()
    inputs = _inputs()
    first = model(**inputs).object_latent_tokens.detach()
    permutation = torch.tensor([2, 0, 3, 1])
    changed = dict(inputs)
    changed["tracks"] = inputs["tracks"][:, :, :, permutation, :]
    changed["visibility"] = inputs["visibility"][:, :, :, permutation]
    changed["confidence"] = inputs["confidence"][:, :, :, permutation]
    second = model(**changed).object_latent_tokens.detach()
    torch.testing.assert_close(first, second, rtol=1.0e-5, atol=1.0e-5)


def test_visual_inputs_receive_gradients() -> None:
    model = _model().train()
    inputs = _inputs()
    inputs["jepa_patch_tokens"].requires_grad_(True)
    inputs["context_latents"].requires_grad_(True)
    output = model(**inputs).object_latent_tokens
    output[:, :, :2].square().mean().backward()
    assert inputs["jepa_patch_tokens"].grad is not None
    assert inputs["context_latents"].grad is not None
    assert torch.isfinite(inputs["jepa_patch_tokens"].grad).all()
    assert torch.isfinite(inputs["context_latents"].grad).all()


def test_k_is_configurable() -> None:
    for token_count in (2, 4, 8):
        output = _model(token_count).eval()(**_inputs())
        assert output.object_latent_tokens.shape == (1, token_count, 3, 32)


def test_all_invalid_objects_stay_finite_and_zero() -> None:
    model = _model().eval()
    inputs = _inputs()
    inputs["object_valid_mask"] = torch.zeros(1, 3)
    output = model(**inputs).object_latent_tokens
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0


def test_motion_encoder_compresses_observations_and_receives_gradients() -> None:
    torch.manual_seed(9)
    encoder = TrajectoryTokenEncoder(
        hidden_dim=32,
        num_motion_tokens=4,
        num_heads=4,
        fourier_bands=3,
        max_points=4,
    ).train()
    state = torch.randn(1, 8, 2, 4, 7, requires_grad=True)
    state_data = state.detach().clone()
    state_data[..., :2] = state_data[..., :2].sigmoid()
    state_data[..., 4:] = state_data[..., 4:].sigmoid()
    state = state_data.requires_grad_(True)
    valid = torch.ones(1, 8, 2, 4, dtype=torch.bool)
    tokens, token_valid, trace = encoder(state, valid)
    assert tokens.shape == (1, 2, 4, 32)
    assert token_valid.shape == (1, 2, 4)
    assert trace["motion_encoded_observations_BO_TP_H"] == [2, 32, 32]
    tokens.square().mean().backward()
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()


def test_motion_encoder_all_invalid_is_finite_and_zero() -> None:
    encoder = TrajectoryTokenEncoder(
        hidden_dim=32,
        num_motion_tokens=4,
        num_heads=4,
        max_points=4,
    ).eval()
    tokens, token_valid, _ = encoder(
        torch.zeros(1, 8, 2, 4, 7),
        torch.zeros(1, 8, 2, 4, dtype=torch.bool),
    )
    assert torch.isfinite(tokens).all()
    assert torch.count_nonzero(tokens) == 0
    assert not bool(token_valid.any())


def test_fourier_encoding_preserves_raw_coordinates() -> None:
    values = torch.tensor([[[0.25, -0.5]]])
    encoded = fourier_encode(values, num_bands=4)
    assert encoded.shape[-1] == 2 * (1 + 2 * 4)
    torch.testing.assert_close(encoded[..., 0], values[..., 0])
    torch.testing.assert_close(encoded[..., 9], values[..., 1])


def test_entity_binding_is_applied_to_every_token_in_its_object_group() -> None:
    torch.manual_seed(11)
    adapter = EntityIDBindingObjectConditionAdapter(
        dim=16,
        num_slots=2,
        max_time_steps=4,
        entity_text_dim=16,
        entity_bottleneck_dim=8,
        entity_gate_init=0.5,
        entity_dropout_prob=0.0,
        entity_residual_max_ratio=1.0,
    ).eval()
    with torch.no_grad():
        adapter.entity_text_up.weight.normal_(std=0.1)
    adapter.set_entity_binding_context(
        entity_text_by_id=torch.randn(1, 2, 16),
        entity_text_match_mask=torch.tensor([[True, True]]),
        slot_entity_ids=torch.tensor([[1, 0]]),
    )
    base = torch.randn(1, 4, 2, 16)
    bound = adapter.apply_entity_binding(
        base,
        object_valid_mask=torch.ones(1, 2),
    )
    residual = bound - base
    for token_id in range(1, 4):
        torch.testing.assert_close(residual[:, token_id], residual[:, 0])
    output = adapter(base, object_valid_mask=torch.ones(1, 2))
    assert output.shape == (1, 8, 16)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.object_cross_attn = nn.Linear(4, 4)
        self.norm4 = nn.LayerNorm(4)
        self.object_gate = nn.Parameter(torch.ones(1))


class _Dit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dim = 4
        self.blocks = nn.ModuleList([_Block() for _ in range(6)])


def test_prune_object_blocks() -> None:
    dit = _Dit()
    summary = prune_object_cross_attention_blocks(dit, (1, 4))
    assert summary["active_block_ids"] == [1, 4]
    assert parse_block_ids("4,1,4", num_blocks=6) == (1, 4)
    for block_id, block in enumerate(dit.blocks):
        if block_id in (1, 4):
            assert block.object_cross_attn is not None
            assert block.object_gate is not None
        else:
            assert block.object_cross_attn is None
            assert block.norm4 is None
            assert block.object_gate is None


def test_bottleneck_object_attention_shape_and_gradients() -> None:
    attention = BottleneckObjectCrossAttention(
        query_dim=32,
        context_dim=12,
        inner_dim=16,
        num_heads=4,
    ).train()
    query = torch.randn(2, 11, 32, requires_grad=True)
    context = torch.randn(2, 5, 12, requires_grad=True)
    output = attention(query, context)
    assert output.shape == query.shape
    output.square().mean().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert context.grad is not None and torch.isfinite(context.grad).all()


def test_install_bottleneck_attention_prunes_and_replaces() -> None:
    dit = _Dit()
    summary = install_bottleneck_object_cross_attention(
        dit,
        (1, 4),
        object_dim=3,
        inner_dim=4,
        num_heads=2,
    )
    assert summary["attention_inner_dim"] == 4
    assert dit.object_embedding is None
    for block_id, block in enumerate(dit.blocks):
        if block_id in (1, 4):
            assert isinstance(block.object_cross_attn, BottleneckObjectCrossAttention)
            assert block.object_cross_attn.q.weight.shape == (4, 4)
            assert block.object_cross_attn.k.weight.shape == (4, 3)
        else:
            assert block.object_cross_attn is None
