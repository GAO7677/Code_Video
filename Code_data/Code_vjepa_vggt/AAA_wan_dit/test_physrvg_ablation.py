"""Small CPU tests for PhysRVG runtime ablation semantics."""

from __future__ import annotations

import torch

from physrvg_ablation import (
    EXPECTED_LORA_MODULES,
    PhysRVGAblationSpec,
    get_ablation_call_count,
    install_grouped_physrvg_head_ablation,
    install_physrvg_ablation,
)


class FakeLoraLinear(torch.nn.Module):
    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = increment
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 1)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 1)})
        self.enabled = True

    def enable_adapters(self, enabled: bool) -> None:
        self.enabled = enabled

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        return x + self.increment if self.enabled else x


class FakeAttention(torch.nn.Module):
    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = increment
        self.heads = 3
        self.to_q = FakeLoraLinear(0.01)
        self.to_k = FakeLoraLinear(0.01)
        self.to_v = FakeLoraLinear(0.01)
        self.to_out = torch.nn.ModuleList([FakeLoraLinear(0.01)])

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        return x + self.increment


class FakeFFN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        first = torch.nn.Module()
        first.add_module("proj", FakeLoraLinear(0.01))
        self.net = torch.nn.ModuleList(
            [first, torch.nn.Identity(), FakeLoraLinear(0.01)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 3


class FakeBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = FakeAttention(1)
        self.attn2 = FakeAttention(2)
        self.ffn = FakeFFN()

    def forward(self, x, *args, **kwargs):
        del args, kwargs
        return x + self.attn1(x) + self.attn2(x) + self.ffn(x)


class FakeDiT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([FakeBlock() for _ in range(30)])


class FakePeft(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.add_module("model", FakeDiT())


def test_module_ablations() -> None:
    x = torch.ones(1, 2, 3)
    for mode, name in (
        ("self_attn_zero", "attn1"),
        ("text_cross_attn_zero", "attn2"),
        ("ffn_zero", "ffn"),
    ):
        model = FakeDiT()
        install_physrvg_ablation(model, PhysRVGAblationSpec(mode, 5))
        output = getattr(model.blocks[5], name)(x)
        assert torch.equal(output, torch.zeros_like(x))
        assert get_ablation_call_count(model) == 1


def test_whole_block() -> None:
    model = FakeDiT()
    install_physrvg_ablation(
        model,
        PhysRVGAblationSpec("whole_block", 29),
    )
    x = torch.randn(1, 2, 3)
    assert torch.equal(model.blocks[29](x), x)
    assert get_ablation_call_count(model) == 1


def test_single_head_zero() -> None:
    model = FakeDiT()
    metadata = install_physrvg_ablation(
        model,
        PhysRVGAblationSpec("self_attn_head_zero", 17, 1),
    )
    projection = model.blocks[17].attn1.to_out[0]
    projection_input = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    output = projection(projection_input)
    expected = projection_input.reshape(1, 2, 3, 2).clone()
    expected[..., 1, :] = 0
    expected = expected.reshape_as(projection_input) + 0.01
    assert torch.allclose(output, expected)
    assert get_ablation_call_count(model) == 1
    assert metadata["head_id"] == 1
    assert metadata["num_attention_heads"] == 3


def test_lora_off_and_peft_unwrap() -> None:
    model = FakePeft()
    metadata = install_physrvg_ablation(
        model,
        PhysRVGAblationSpec("lora_off", 0),
    )
    assert tuple(metadata["disabled_lora_modules"]) == EXPECTED_LORA_MODULES
    block = model.base_model.model.blocks[0]
    for module in block.modules():
        if hasattr(module, "lora_A"):
            assert module.enabled is False
    block(torch.ones(1, 2, 3))
    assert get_ablation_call_count(model) == 1


def test_grouped_head_zero() -> None:
    model = FakeDiT()
    metadata = install_grouped_physrvg_head_ablation(
        model,
        category="C",
        targets=[(0, 0), (29, 2)],
    )
    projection_input = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    output_0 = model.blocks[0].attn1.to_out[0](projection_input)
    output_29 = model.blocks[29].attn1.to_out[0](projection_input)
    expected_0 = projection_input.reshape(1, 2, 3, 2).clone()
    expected_0[..., 0, :] = 0
    expected_29 = projection_input.reshape(1, 2, 3, 2).clone()
    expected_29[..., 2, :] = 0
    assert torch.allclose(
        output_0, expected_0.reshape_as(projection_input) + 0.01
    )
    assert torch.allclose(
        output_29, expected_29.reshape_as(projection_input) + 0.01
    )
    assert get_ablation_call_count(model) == 2
    assert metadata["category"] == "C"
    assert metadata["num_targets"] == 2


def test_grouped_multiple_heads_in_one_block() -> None:
    model = FakeDiT()
    metadata = install_grouped_physrvg_head_ablation(
        model,
        category="S",
        targets=[(4, 0), (4, 2)],
    )
    projection_input = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    output = model.blocks[4].attn1.to_out[0](projection_input)
    expected = projection_input.reshape(1, 2, 3, 2).clone()
    expected[..., [0, 2], :] = 0
    assert torch.allclose(output, expected.reshape_as(projection_input) + 0.01)
    assert get_ablation_call_count(model) == 2
    assert metadata["num_targets"] == 2
    assert metadata["num_target_blocks"] == 1


def test_baseline() -> None:
    model = FakeDiT()
    metadata = install_physrvg_ablation(model, PhysRVGAblationSpec())
    assert metadata["disabled_module"] is None
    assert get_ablation_call_count(model) == 0


if __name__ == "__main__":
    test_module_ablations()
    test_whole_block()
    test_single_head_zero()
    test_grouped_head_zero()
    test_grouped_multiple_heads_in_one_block()
    test_lora_off_and_peft_unwrap()
    test_baseline()
    print("PhysRVG ablation tests passed")
