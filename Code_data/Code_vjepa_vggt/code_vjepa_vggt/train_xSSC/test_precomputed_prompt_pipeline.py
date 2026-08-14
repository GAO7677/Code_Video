from __future__ import annotations

import os
from types import SimpleNamespace
import unittest


os.environ.setdefault(
    "DIFFSYNTH_ROOT",
    "/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main",
)

from code_vjepa_vggt.train_xSSC.train_xssc_context_slots import (  # noqa: E402
    XSSCContextSlotsWanModule,
)
from code_vjepa_vggt.train_xSSC.object_self_attn_lora_experiments.vjepa_loss_project.train_xssc_object_self_attn_lora_vjepa_loss import (  # noqa: E402
    VJEPAFeatureLossWanModule,
)


class _Runner:
    def __init__(self) -> None:
        self.called: list[str] = []

    def __call__(self, unit, pipe, inputs_shared, inputs_posi, inputs_nega):
        name = unit.__class__.__name__
        self.called.append(name)
        if name == "WanVideoUnit_PromptEmbedder":
            raise AssertionError("precomputed context must bypass prompt embedding")
        return inputs_shared, inputs_posi, inputs_nega


class PrecomputedPromptPipelineTests(unittest.TestCase):
    def test_xssc_batch_preparation_skips_prompt_embedder(self) -> None:
        prompt_unit = type("WanVideoUnit_PromptEmbedder", (), {})()
        other_unit = type("OtherUnit", (), {})()
        runner = _Runner()
        pipe = SimpleNamespace(
            units=[prompt_unit, other_unit],
            unit_runner=runner,
            device="cpu",
            torch_dtype=None,
        )
        inputs = ({}, {"context": object()}, {})
        model = SimpleNamespace(
            pipe=pipe,
            get_pipeline_inputs=lambda sample: inputs,
            transfer_data_to_device=lambda value, device, dtype: value,
        )

        result = XSSCContextSlotsWanModule._prepare_pipeline_sample(model, {})

        self.assertIs(result[0], inputs[0])
        self.assertIs(result[1], inputs[1])
        self.assertIs(result[2], inputs[2])
        self.assertEqual(runner.called, ["OtherUnit"])

    def test_vjepa_scalar_forward_skips_prompt_embedder(self) -> None:
        prompt_unit = type("WanVideoUnit_PromptEmbedder", (), {})()
        other_unit = type("OtherUnit", (), {})()
        runner = _Runner()
        pipe = SimpleNamespace(
            units=[prompt_unit, other_unit],
            unit_runner=runner,
            device="cpu",
            torch_dtype=None,
        )
        inputs = ({}, {"context": object()}, {})
        expected_loss = object()
        expected_metrics = {"train/loss": 1.0}
        model = SimpleNamespace(
            pipe=pipe,
            get_pipeline_inputs=lambda sample: inputs,
            transfer_data_to_device=lambda value, device, dtype: value,
            _compute_object_losses=lambda pipe, shared, positive: (
                expected_loss,
                expected_metrics,
            ),
            last_train_metrics=None,
        )

        result = VJEPAFeatureLossWanModule.forward(model, {})

        self.assertIs(result, expected_loss)
        self.assertIs(model.last_train_metrics, expected_metrics)
        self.assertEqual(runner.called, ["OtherUnit"])


if __name__ == "__main__":
    unittest.main()
