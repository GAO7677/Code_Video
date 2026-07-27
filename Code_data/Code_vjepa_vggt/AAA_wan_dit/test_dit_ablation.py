#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from dit_ablation import (
    DiTAblationSpec,
    annotate_result_files,
    get_dit_head_ablation_call_count,
    install_dynamic_grouped_head_ablator,
    install_dit_ablation,
    install_grouped_head_ablation,
)


class AddOne(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x + 1


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = AddOne()
        self.object_cross_attn = AddOne()

    def forward(self, x, *args, **kwargs):
        x = x + self.self_attn(x)
        x = x + self.object_cross_attn(x)
        return x + 1


class FakeHeadSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 3
        self.o = nn.Identity()


class FakeDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock() for _ in range(30)])


def run_block(dit):
    x = torch.ones(1, 2, 3)
    return dit.blocks[7](x, None, None, None)


def main() -> None:
    baseline = run_block(FakeDiT())

    whole = FakeDiT()
    install_dit_ablation(
        whole,
        DiTAblationSpec("whole_block", 7),
    )
    assert torch.equal(run_block(whole), torch.ones(1, 2, 3))

    self_attn_zero = FakeDiT()
    install_dit_ablation(
        self_attn_zero,
        DiTAblationSpec("self_attn_zero", 7),
    )
    self_attn_zero_out = run_block(self_attn_zero)
    assert torch.all(self_attn_zero_out < baseline)

    object_attn = FakeDiT()
    install_dit_ablation(
        object_attn,
        DiTAblationSpec("object_cross_attn", 7),
    )
    object_attn_out = run_block(object_attn)
    assert torch.all(object_attn_out < baseline)

    head_zero = FakeDiT()
    head_zero.blocks[7].self_attn = FakeHeadSelfAttention()
    metadata = install_dit_ablation(
        head_zero,
        DiTAblationSpec("self_attn_head_zero", 7, 1),
    )
    projection_input = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    projection_output = head_zero.blocks[7].self_attn.o(projection_input)
    expected = projection_input.reshape(1, 2, 3, 2).clone()
    expected[..., 1, :] = 0
    assert torch.equal(projection_output, expected.reshape_as(projection_input))
    assert get_dit_head_ablation_call_count(head_zero) == 1
    assert metadata["head_id"] == 1
    assert metadata["num_attention_heads"] == 3

    grouped = FakeDiT()
    grouped.blocks[2].self_attn = FakeHeadSelfAttention()
    grouped.blocks[7].self_attn = FakeHeadSelfAttention()
    metadata = install_grouped_head_ablation(
        grouped,
        category="T",
        targets=[(2, 0), (7, 2)],
    )
    projection_input = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    output_2 = grouped.blocks[2].self_attn.o(projection_input)
    output_7 = grouped.blocks[7].self_attn.o(projection_input)
    expected_2 = projection_input.reshape(1, 2, 3, 2).clone()
    expected_2[..., 0, :] = 0
    expected_7 = projection_input.reshape(1, 2, 3, 2).clone()
    expected_7[..., 2, :] = 0
    assert torch.equal(output_2, expected_2.reshape_as(projection_input))
    assert torch.equal(output_7, expected_7.reshape_as(projection_input))
    assert get_dit_head_ablation_call_count(grouped) == 2
    assert metadata["category"] == "T"
    assert metadata["num_targets"] == 2

    grouped_multi = FakeDiT()
    grouped_multi.blocks[2].self_attn = FakeHeadSelfAttention()
    metadata = install_grouped_head_ablation(
        grouped_multi,
        category="PREV_A",
        targets=[(2, 0), (2, 2)],
    )
    output_multi = grouped_multi.blocks[2].self_attn.o(projection_input)
    expected_multi = projection_input.reshape(1, 2, 3, 2).clone()
    expected_multi[..., [0, 2], :] = 0
    assert torch.equal(
        output_multi, expected_multi.reshape_as(projection_input)
    )
    assert get_dit_head_ablation_call_count(grouped_multi) == 2
    assert metadata["category"] == "PREV_A"
    assert metadata["num_targets"] == 2
    assert metadata["num_target_blocks"] == 1

    dynamic = FakeDiT()
    for block in dynamic.blocks:
        block.self_attn = FakeHeadSelfAttention()
    controller = install_dynamic_grouped_head_ablator(dynamic)
    baseline_metadata = controller.set_targets(category=None, targets=[])
    baseline_output = dynamic.blocks[2].self_attn.o(projection_input)
    assert torch.equal(baseline_output, projection_input)
    assert controller.call_count == 0
    assert baseline_metadata["mode"] == "baseline"

    dynamic_s = controller.set_targets(
        category="S",
        targets=[(2, 0), (2, 2)],
    )
    dynamic_s_output = dynamic.blocks[2].self_attn.o(projection_input)
    assert torch.equal(
        dynamic_s_output,
        expected_multi.reshape_as(projection_input),
    )
    assert controller.call_count == 2
    assert dynamic_s["num_targets"] == 2

    dynamic_t = controller.set_targets(category="T", targets=[(2, 1)])
    dynamic_t_output = dynamic.blocks[2].self_attn.o(projection_input)
    expected_dynamic_t = projection_input.reshape(1, 2, 3, 2).clone()
    expected_dynamic_t[..., 1, :] = 0
    assert torch.equal(
        dynamic_t_output,
        expected_dynamic_t.reshape_as(projection_input),
    )
    assert controller.call_count == 1
    assert dynamic_t["category"] == "T"

    controller.set_targets(category=None, targets=[])
    restored_output = dynamic.blocks[2].self_attn.o(projection_input)
    assert torch.equal(restored_output, projection_input)
    assert controller.call_count == 0

    untouched = FakeDiT()
    metadata = install_dit_ablation(untouched, DiTAblationSpec())
    assert torch.equal(run_block(untouched), baseline)
    assert metadata["disabled_module"] is None

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_root = Path(temporary_directory)
        json_path = output_root / "summary.json"
        jsonl_path = output_root / "per_case.jsonl"
        json_path.write_text(
            json.dumps({"status": "complete"}),
            encoding="utf-8",
        )
        jsonl_path.write_text(
            json.dumps({"case": "case_000"}) + "\n",
            encoding="utf-8",
        )
        zero_metadata = self_attn_zero._aaa_wan_dit_ablation
        negative_prompt = "low quality, artifacts"
        counts = annotate_result_files(
            [output_root],
            zero_metadata,
            negative_prompt=negative_prompt,
        )
        annotated_json = json.loads(json_path.read_text(encoding="utf-8"))
        annotated_jsonl = json.loads(jsonl_path.read_text(encoding="utf-8"))
        assert counts == {"json_files": 1, "jsonl_files": 1}
        assert annotated_json["dit_ablation"] == zero_metadata
        assert annotated_jsonl["dit_ablation"] == zero_metadata
        assert annotated_json["negative_prompt"] == negative_prompt
        assert annotated_jsonl["negative_prompt"] == negative_prompt

    print("dit_ablation tests passed")


if __name__ == "__main__":
    main()
