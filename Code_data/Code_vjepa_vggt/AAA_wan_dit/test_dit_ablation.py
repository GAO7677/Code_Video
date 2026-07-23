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
    install_dit_ablation,
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
