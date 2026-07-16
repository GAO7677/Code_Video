from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

import code_vjepa_vggt.train_v_newtrain as tvn
from code_vjepa_vggt.data.mixed_replay_no_gt_box_dataset import (
    KubricReplayNoGTBoxDataset,
    OpenVidNoGTBoxDataset,
    WeightedNoGTBoxMixture,
)
from code_vjepa_vggt.data.pybullet_raw_no_gt_box_dataset import PyBulletRawNoGTBoxDataset
from code_vjepa_vggt.headonly_val_loss import HeadOnlyValConfig
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    train_stage1b_no_gt_box_replay_preserve as replay,
)
from diffsynth.diffusion import ModelLogger

from wan_phyco_train0716.dataset import PhysicalPropertyMapDataset
from wan_phyco_train0716.models import (
    controller_parameter_report,
    inject_wan_phyco_controlnet,
)


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not result:
        raise ValueError("expected at least one integer")
    return result


class WanPhyCoTrainingModule(tvn.WanTrainingModule):
    def __init__(
        self,
        *args,
        phyco_hidden_dim: int = 128,
        phyco_block_ids: str = "3,8,13,18,23,28",
        **kwargs,
    ) -> None:
        kwargs["enable_object_branch"] = False
        super().__init__(*args, **kwargs)
        self.phyco_controlnet = inject_wan_phyco_controlnet(
            self.pipe.dit,
            hidden_dim=int(phyco_hidden_dim),
            block_ids=_parse_ints(phyco_block_ids),
        )

    def trainable_modules(self):
        return [parameter for parameter in self.phyco_controlnet.parameters() if parameter.requires_grad]

    def forward(self, data, inputs=None):
        maps = data.get("phyco_control_maps")
        branch_valid = data.get("phyco_branch_valid")
        if maps is None or branch_valid is None:
            raise KeyError("dataset sample must contain phyco_control_maps and phyco_branch_valid")
        self.pipe.dit._phyco_control_maps = maps.to(
            device=self.pipe.device,
            dtype=self.pipe.torch_dtype,
        )
        self.pipe.dit._phyco_branch_valid = branch_valid.to(device=self.pipe.device)
        # Keep these tensors attached until backward finishes: Wan gradient
        # checkpointing recomputes the DiT blocks after this method returns.
        loss = super().forward(data, inputs=inputs)
        stats = self.phyco_controlnet.pop_stats()
        metrics = dict(self.last_train_metrics)
        for branch_name in self.phyco_controlnet.BRANCH_NAMES:
            selected = [item for item in stats if item.branch_name == branch_name]
            metrics[f"train/phyco_{branch_name}_active_fraction"] = float(
                branch_valid[..., self.phyco_controlnet.BRANCH_NAMES.index(branch_name)]
                .float()
                .mean()
                .item()
            )
            metrics[f"train/phyco_{branch_name}_residual_to_hidden_rms_max"] = max(
                (item.residual_to_hidden_rms for item in selected),
                default=0.0,
            )
        metrics["train/phyco_control_map_abs_mean"] = float(maps.float().abs().mean().item())
        metrics["train/phyco_control_map_abs_max"] = float(maps.float().abs().max().item())
        self.last_train_metrics = metrics
        return loss


def build_parser() -> argparse.ArgumentParser:
    parser = replay.build_parser()
    parser.description = "DiffSynth-native Wan2.2 PhyCo multi-branch ControlNet training."
    group = parser.add_argument_group("wan_phyco")
    group.add_argument("--phyco_hidden_dim", type=int, default=128)
    group.add_argument("--phyco_block_ids", default="3,8,13,18,23,28")
    group.add_argument("--phyco_map_downsample", type=int, default=16)
    group.add_argument("--phyco_init_from", default=None)
    return parser


def build_dataset(args: argparse.Namespace):
    resolution = (int(args.height), int(args.width))
    map_downsample = max(1, int(args.phyco_map_downsample))
    map_height = max(8, int(args.height) // map_downsample)
    map_width = max(8, int(args.width) // map_downsample)
    pybullet = PhysicalPropertyMapDataset(
        PyBulletRawNoGTBoxDataset(
            root=args.pybullet_raw_root,
            split=args.pybullet_raw_split,
            resolution=resolution,
            num_frames=args.num_frames,
            num_context_frames=args.fixed_num_context_frames,
            sampling_strategy=args.pybullet_raw_sampling_strategy,
            window_starts=tuple(
                int(value.strip())
                for value in args.pybullet_raw_window_starts.split(",")
                if value.strip()
            ),
            init_scan_limit=args.pybullet_raw_init_scan_limit,
        ),
        source="pybullet",
        map_height=map_height,
        map_width=map_width,
    )
    kubric = PhysicalPropertyMapDataset(
        KubricReplayNoGTBoxDataset(
            root=args.kubric_root,
            split=args.kubric_split,
            resolution=resolution,
            num_frames=args.num_frames,
            num_context_frames=args.fixed_num_context_frames,
            index_num_frames=args.kubric_replay_index_num_frames,
            index_num_context_frames=args.kubric_replay_index_num_context_frames,
            sampling_strategy=args.kubric_sampling_strategy,
            seed=42,
            scenarios=args.kubric_scenario,
            init_scan_limit=args.kubric_init_scan_limit,
            cache_root=args.kubric_cache_root,
            split_train_ratio=args.kubric_split_train_ratio,
            split_val_ratio=args.kubric_split_val_ratio,
            max_retry_samples=args.kubric_max_retry_samples,
        ),
        source="kubric",
        map_height=map_height,
        map_width=map_width,
    )
    openvid = PhysicalPropertyMapDataset(
        OpenVidNoGTBoxDataset(
            root=args.openvid_root,
            resolution=resolution,
            num_frames=args.num_frames,
            num_context_frames=args.fixed_num_context_frames,
            max_samples=args.openvid_max_samples,
        ),
        source="openvid",
        map_height=map_height,
        map_width=map_width,
    )
    return WeightedNoGTBoxMixture(
        datasets=(pybullet, kubric, openvid),
        source_names=("pybullet", "kubric", "openvid"),
        source_probabilities=(
            args.mixture_pybullet_ratio,
            args.mixture_kubric_ratio,
            args.mixture_openvid_ratio,
        ),
    )


def build_model(args: argparse.Namespace, accelerator) -> WanPhyCoTrainingModule:
    model = WanPhyCoTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        context_sampling_profile=args.context_sampling_profile,
        min_context_frames=args.min_context_frames,
        max_context_ratio=args.max_context_ratio,
        context_frame_choices=args.context_frame_choices,
        context_length_sampling=args.context_length_sampling,
        context_reference_frames=args.context_reference_frames,
        context_reference_prefixes=args.context_reference_prefixes,
        prefix_context_ratio=args.prefix_context_ratio,
        first_frame_context_ratio=args.first_frame_context_ratio,
        sparse_context_ratio=args.sparse_context_ratio,
        random_context_ratio=args.random_context_ratio,
        no_context_ratio=args.no_context_ratio,
        fixed_num_context_frames=args.fixed_num_context_frames,
        ctx_max_length=args.ctx_max_length,
        phyco_hidden_dim=args.phyco_hidden_dim,
        phyco_block_ids=args.phyco_block_ids,
    )
    if args.phyco_init_from:
        source = Path(args.phyco_init_from).expanduser().resolve()
        if source.is_dir():
            source = source / "checkpoint.safetensors"
        from safetensors.torch import load_file

        state = load_file(str(source), device="cpu")
        prefix = "phyco_controlnet."
        selected = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        info = model.phyco_controlnet.load_state_dict(selected, strict=True)
        accelerator.print(f"Loaded PhyCo controller from {source}: {info}")
    return model


def main() -> None:
    args = tvn.prepare_args(build_parser().parse_args())
    if args.enable_object_branch:
        raise ValueError("Wan-PhyCo v1 is intentionally independent from Scheme-D object branch")
    previous_handlers = tvn.install_interrupt_handlers()
    accelerator = tvn.build_accelerator(args)
    tvn.init_trackers(accelerator, args)
    dataset = build_dataset(args)
    model = build_model(args, accelerator)
    report = {
        "base_model": str(args.wan_root),
        "framework": str(args.diffsynth_root),
        "base_wan_frozen": True,
        "vae_frozen": True,
        "text_encoder_frozen": True,
        "scheme_d_object_branch": False,
        "mixture": dict(dataset.dataset_stats),
        "controller": controller_parameter_report(model.phyco_controlnet),
        "property_channels": [
            "restitution",
            "friction",
            "rigid_valid",
            "neo_hookean_mu_norm",
            "neo_hookean_lambda_norm",
            "neo_hookean_damping_norm",
            "deformation_valid",
            "action_magnitude_norm",
            "action_direction_x",
            "action_direction_y",
            "action_type",
            "action_valid",
        ],
        "action_type_values": {"initial_velocity": -1.0, "external_force": 1.0},
        "pybullet_action_semantics": "initial_velocity",
        "openvid_semantics": "null maps with disabled branches; zero controller gradient",
    }
    if accelerator.is_main_process:
        output = Path(args.output_path).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "wan_phyco_module_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    accelerator.print("[wan-phyco] " + json.dumps(report, ensure_ascii=False))
    disabled_val = HeadOnlyValConfig(enabled=False, split="val", every_steps=None, num_batches=1)
    model_logger = ModelLogger(
        tvn.get_checkpoint_dir(args),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    try:
        tvn.train_loop(
            accelerator,
            dataset,
            model,
            model_logger,
            args,
            runtime_state={},
            headonly_val_dataloader=None,
            headonly_val_config=disabled_val,
        )
    finally:
        accelerator.end_training()
        tvn.restore_interrupt_handlers(previous_handlers)


if __name__ == "__main__":
    main()
