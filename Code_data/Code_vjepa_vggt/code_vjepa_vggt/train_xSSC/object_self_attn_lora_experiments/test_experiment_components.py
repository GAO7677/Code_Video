from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model

import launch_from_config as launcher
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

    def test_head_configs_and_compact_parameter_counts(self) -> None:
        matched_heads, matched_metadata = experiment.load_head_selection_config(
            experiment.EXPERIMENT_ROOT / "configs/same_frame_mass_heads.json",
            expected_subset_id="S_same_k32_r00_exactblock",
            expected_role="S",
            expected_feature_subtype="same_frame_mass",
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

        full_heads, full_metadata = experiment.load_head_selection_config(
            experiment.EXPERIMENT_ROOT
            / "configs/same_frame_mass_heads_full59.json",
            expected_subset_id="S_same_full59",
            expected_role="S",
            expected_feature_subtype="same_frame_mass",
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

        t_heads, t_metadata = experiment.load_head_selection_config(
            experiment.EXPERIMENT_ROOT / "configs/common_t_heads_full70.json",
            expected_subset_id="T_common_full70",
            expected_role="T",
            expected_feature_subtype="common_t",
            expected_num_heads=70,
            num_blocks=30,
            num_heads=24,
        )
        self.assertEqual(sum(len(value) for value in t_heads.values()), 70)
        self.assertEqual(len(t_heads), 21)
        self.assertEqual(t_metadata["feature_subtype"], "common_t")
        t_params = sum(
            4 * dim * rank + 4 * len(heads) * head_dim * rank
            for heads in t_heads.values()
        )
        self.assertEqual(t_params, 9_404_416)

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

    def test_head_identity_rejects_same_shape_different_heads(self) -> None:
        expected = experiment.build_head_selection_identity(
            {4: (0, 18), 5: (9,)}
        )
        self.assertEqual(expected.dtype, torch.int32)
        digest = experiment.build_sha256_identity("12" * 32)
        state = {
            f"module.{experiment.HEAD_SELECTION_IDENTITY_KEY}": expected.clone(),
            f"module.{experiment.HEAD_SELECTION_CONFIG_SHA256_KEY}": digest.clone(),
        }
        info = experiment.validate_head_selection_checkpoint_state(
            state,
            expected_identity=expected,
            expected_config_sha256=digest,
        )
        self.assertEqual(info["num_heads"], 3)
        self.assertEqual(info["config_sha256"], "12" * 32)

        different_same_shape = experiment.build_head_selection_identity(
            {4: (0, 18), 5: (14,)}
        )
        with self.assertRaisesRegex(RuntimeError, "different heads"):
            experiment.validate_head_selection_checkpoint_state(
                state,
                expected_identity=different_same_shape,
                expected_config_sha256=digest,
            )
        with self.assertRaisesRegex(RuntimeError, "missing bound head identity"):
            experiment.validate_head_selection_checkpoint_state(
                {},
                expected_identity=expected,
                expected_config_sha256=digest,
            )

    def test_head_identity_is_exported_with_trainable_state(self) -> None:
        model = experiment.DINOv3XSSCContextSlotsWanModule.__new__(
            experiment.DINOv3XSSCContextSlotsWanModule
        )
        nn.Module.__init__(model)
        model.self_attn_adaptation_mode = "s_head"
        model.pipe = SimpleNamespace(dit=nn.Module())
        model.trainable_probe = nn.Parameter(torch.ones(1))
        model.register_buffer(
            experiment.HEAD_SELECTION_IDENTITY_KEY,
            experiment.build_head_selection_identity({4: (0, 18)}),
        )
        model.register_buffer(
            experiment.HEAD_SELECTION_CONFIG_SHA256_KEY,
            experiment.build_sha256_identity("ab" * 32),
        )

        exported = model.export_trainable_state_dict(model.state_dict())
        self.assertIn("trainable_probe", exported)
        self.assertTrue(
            torch.equal(
                exported[experiment.HEAD_SELECTION_IDENTITY_KEY],
                model.head_selection_identity,
            )
        )
        self.assertTrue(
            torch.equal(
                exported[experiment.HEAD_SELECTION_CONFIG_SHA256_KEY],
                model.head_selection_config_sha256,
            )
        )

    def test_checkpoint_saver_runs_only_on_sync_microstep(self) -> None:
        calls = []

        def save_fn(*args, **kwargs):
            calls.append((args, kwargs))

        class FakeAccelerator:
            sync_gradients = False

        accelerator = FakeAccelerator()
        wrapped = experiment.checkpoint_saver_only_on_sync(save_fn)
        wrapped(accelerator=accelerator, checkpoint_tag="step-000500")
        wrapped(accelerator=accelerator, checkpoint_tag="step-000500")
        self.assertEqual(calls, [])

        accelerator.sync_gradients = True
        wrapped(accelerator=accelerator, checkpoint_tag="step-000500")
        self.assertEqual(len(calls), 1)

    def test_launcher_snapshots_head_selection_config(self) -> None:
        source = (
            experiment.EXPERIMENT_ROOT
            / "configs"
            / "common_t_heads_full70.json"
        )
        config = {
            "adaptation": {"mode": "t_head"},
            "paths": {"head_selection_config": str(source)},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            info = launcher.snapshot_head_selection_config(
                config,
                Path(temp_dir),
            )
            snapshot = Path(config["paths"]["head_selection_config"])
            self.assertEqual(snapshot, Path(temp_dir) / "head_selection_config.json")
            self.assertEqual(snapshot.read_bytes(), source.read_bytes())
            self.assertEqual(info["num_heads"], 70)
            self.assertEqual(info["role"], "T")

    def test_single_gpu_launcher_omits_multi_gpu_flag(self) -> None:
        config_path = (
            experiment.EXPERIMENT_ROOT
            / "configs"
            / "formal_object_only_gpu1.json"
        )
        raw, _ = launcher.load_config(config_path)
        config = launcher.validate_config(raw, config_path.parent)
        command = launcher.build_command(config, Path("/tmp/test-single-gpu"))
        self.assertNotIn("--multi_gpu", command)
        self.assertEqual(config["launch"]["gpu_set"], "1")
        self.assertEqual(config["launch"]["num_processes"], 1)
        self.assertEqual(
            int(config["optimization"]["gradient_accumulation_steps"]),
            8,
        )
        self.assertIn("--pybullet0713_prompt_cache_dir", command)

    def test_official_xssc_object_only_launcher(self) -> None:
        config_path = (
            experiment.EXPERIMENT_ROOT
            / "configs"
            / "formal_official_xssc_object_only_gpu01.json"
        )
        raw, _ = launcher.load_config(config_path)
        config = launcher.validate_config(raw, config_path.parent)
        command = launcher.build_command(config, Path("/tmp/test-official-xssc"))
        self.assertIn(str(launcher.OFFICIAL_XSSC_OBJECT_ONLY_TRAIN_SCRIPT), command)
        self.assertIn("--multi_gpu", command)
        self.assertNotIn("--dinov3_checkpoint", command)
        self.assertNotIn("--xssc_box_source", command)
        self.assertNotIn("--self_attn_adaptation_mode", command)
        self.assertEqual(config["model"]["xssc_backend"], "official_dinov2")
        self.assertEqual(config["launch"]["gpu_set"], "0,1")
        self.assertEqual(config["optimization"]["max_train_steps"], 1500)


if __name__ == "__main__":
    unittest.main()
