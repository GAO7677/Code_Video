import torch

from stage1_causal_state_probe.models import StatePredictor


def build(context, representation="full", history=4):
    return StatePredictor(
        representation=representation,
        history=history,
        context_mode=context,
        model_dim=32,
        num_heads=4,
        feedforward_dim=64,
        temporal_layers=1,
        context_layers=1,
        dropout=0.0,
    ).eval()


def test_individual_and_set_have_identical_parameter_counts():
    individual = build("individual")
    context = build("set")
    assert sum(p.numel() for p in individual.parameters()) == sum(
        p.numel() for p in context.parameters()
    )


def test_set_predictor_is_slot_permutation_equivariant():
    torch.manual_seed(0)
    model = build("set")
    value = torch.randn(2, 4, 11, 512)
    permutation = torch.randperm(11)
    with torch.inference_mode():
        reference = model(value)
        permuted = model(value[:, :, permutation])
    assert torch.allclose(permuted, reference[:, permutation], atol=1e-5, rtol=1e-5)


def test_zero_initialized_head_starts_as_copy_baseline():
    value = torch.randn(2, 4, 11, 512)
    for representation, output_dim in (("dyn", 128), ("dyn_static", 128), ("full", 512)):
        model = build("individual", representation=representation)
        with torch.inference_mode():
            output = model(value)
        expected = value[:, -1, :, -output_dim:] if output_dim == 128 else value[:, -1]
        assert output.shape == expected.shape
        assert torch.equal(output, expected)

