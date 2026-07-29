from __future__ import annotations

import unittest

import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model

import train_xssc_object_self_attn_lora as experiment


class TinyProjectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q(x)


class ExperimentComponentTests(unittest.TestCase):
    def test_merge_and_unload_preserves_forward(self) -> None:
        torch.manual_seed(7)
        model = TinyProjectionModel()
        model = inject_adapter_in_model(
            LoraConfig(r=2, lora_alpha=2, target_modules=["q"], bias="none"),
            model,
        )
        nn.init.normal_(model.q.lora_B["default"].weight)
        inputs = torch.randn(2, 3, 8)
        expected = model(inputs)

        merged = experiment.merge_and_unload_pretrained_lora(
            model, expected_module_count=1
        )

        self.assertEqual(merged, ["q"])
        self.assertIsInstance(model.q, nn.Linear)
        self.assertTrue(torch.allclose(model(inputs), expected, atol=1e-6, rtol=1e-6))
        self.assertFalse(any("lora_" in name for name, _ in model.named_parameters()))

    def test_q_adapter_changes_only_selected_output_head(self) -> None:
        torch.manual_seed(11)
        base = nn.Linear(8, 8, bias=False)
        adapter = experiment.HeadSelectiveLoRALinear(
            base,
            selected_heads=(1,),
            num_heads=2,
            rank=2,
            alpha=2,
            dropout=0.0,
            projection="q",
        )
        inputs = torch.randn(2, 5, 8)
        baseline = base(inputs)
        self.assertTrue(torch.equal(adapter(inputs), baseline))

        nn.init.normal_(adapter.head_lora_B.weight)
        output = adapter(inputs)
        delta = output - baseline
        self.assertTrue(torch.equal(delta[..., :4], torch.zeros_like(delta[..., :4])))
        self.assertGreater(float(delta[..., 4:].abs().sum()), 0.0)

        output.square().mean().backward()
        self.assertIsNone(base.weight.grad)
        self.assertIsNotNone(adapter.head_lora_A.weight.grad)
        self.assertIsNotNone(adapter.head_lora_B.weight.grad)

    def test_o_adapter_reads_only_selected_input_head(self) -> None:
        torch.manual_seed(13)
        base = nn.Linear(8, 8, bias=False)
        adapter = experiment.HeadSelectiveLoRALinear(
            base,
            selected_heads=(1,),
            num_heads=2,
            rank=2,
            alpha=2,
            dropout=0.0,
            projection="o",
        )
        nn.init.normal_(adapter.head_lora_B.weight)
        inputs = torch.randn(2, 5, 8)
        inputs[..., 4:] = 0
        self.assertTrue(
            torch.allclose(adapter(inputs), base(inputs), atol=1e-6, rtol=1e-6)
        )

    def test_same_frame_configs_and_compact_parameter_counts(self) -> None:
        matched_heads, matched_metadata = experiment.load_same_frame_head_config(
            experiment.EXPERIMENT_ROOT / "configs/same_frame_mass_heads.json",
            expected_subset_id="S_same_k32_r00_exactblock",
            expected_num_heads=32,
            num_blocks=30,
            num_heads=24,
        )
        self.assertEqual(sum(len(value) for value in matched_heads.values()), 32)
        self.assertEqual(len(matched_heads), 18)
        self.assertEqual(matched_metadata["feature_subtype"], "same_frame_mass")

        dim, rank, head_dim = 3072, 32, 128
        matched_params = sum(
            4 * dim * rank + 4 * len(heads) * head_dim * rank
            for heads in matched_heads.values()
        )
        self.assertEqual(matched_params, 7_602_176)

        full_heads, full_metadata = experiment.load_same_frame_head_config(
            experiment.EXPERIMENT_ROOT
            / "configs/same_frame_mass_heads_full59.json",
            expected_subset_id="S_same_full59",
            expected_num_heads=59,
            num_blocks=30,
            num_heads=24,
        )
        self.assertEqual(sum(len(value) for value in full_heads.values()), 59)
        self.assertEqual(len(full_heads), 21)
        self.assertEqual(full_metadata["feature_subtype"], "same_frame_mass")
        full_params = sum(
            4 * dim * rank + 4 * len(heads) * head_dim * rank
            for heads in full_heads.values()
        )
        self.assertEqual(full_params, 9_224_192)

    def test_full_sa_injection_scope(self) -> None:
        dit = nn.Module()
        dit.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": nn.ModuleDict(
                            {
                                name: nn.Linear(8, 8, bias=False)
                                for name in experiment.SELF_ATTN_PROJECTIONS
                            }
                        )
                    }
                )
                for _ in range(3)
            ]
        )
        experiment.inject_full_self_attn_lora(
            dit, rank=2, alpha=2, dropout=0.0
        )
        lora_tensors = [
            name
            for name, _ in dit.named_parameters()
            if ".lora_A." in name or ".lora_B." in name
        ]
        self.assertEqual(len(lora_tensors), 3 * 4 * 2)


if __name__ == "__main__":
    unittest.main()
