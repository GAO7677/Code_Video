"""Small CPU tests for PhysRVG runtime ablation semantics."""

from __future__ import annotations

import torch

from physrvg_ablation import (
    EXPECTED_LORA_MODULES,
    PhysRVGAblationSpec,
    get_ablation_call_count,
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


def test_baseline() -> None:
    model = FakeDiT()
    metadata = install_physrvg_ablation(model, PhysRVGAblationSpec())
    assert metadata["disabled_module"] is None
    assert get_ablation_call_count(model) == 0


if __name__ == "__main__":
    test_module_ablations()
    test_whole_block()
    test_lora_off_and_peft_unwrap()
    test_baseline()
    print("PhysRVG ablation tests passed")
