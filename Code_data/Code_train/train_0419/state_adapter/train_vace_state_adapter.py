"""Train a small state adapter that injects future 9D object states through the VACE branch."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional


def _read_arg_value(argv, name, default=None):
    if name not in argv:
        return default
    index = argv.index(name)
    if index + 1 >= len(argv):
        return default
    return argv[index + 1]


DIFFSYNTH_ROOT = _read_arg_value(
    sys.argv,
    "--diffsynth_root",
    os.environ.get("DIFFSYNTH_ROOT", "/home/gaoya/Code_Video/DiffSynth-Studio-main"),
)
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN0419_ROOT = SCRIPT_DIR.parent
if str(TRAIN0419_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN0419_ROOT))
if DIFFSYNTH_ROOT and DIFFSYNTH_ROOT not in sys.path:
    sys.path.insert(0, DIFFSYNTH_ROOT)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from PIL import Image

from context_wan import (
    flow_match_context_sft_loss,
    resolve_context_latent_indices_from_frames,
)
from diffsynth.diffusion import DiffusionTrainingModule, ModelLogger, add_general_config, add_video_size_config
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline, model_fn_wan_video
from state_adapter_dataset import OracleStateWindowDataset
from train import (
    build_accelerator,
    checkpoint_sort_key,
    find_tokenizer_path,
    get_checkpoint_dir,
    init_trackers,
    resolve_resume_state_path,
    resolve_lora_checkpoint_for_resume,
    train_loop,
)
from vace_state_adapter import VaceStateAdapter


DEFAULT_VACE_ROOT = "/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B"


def build_vace_model_paths(vace_root: str) -> str:
    diffusion_path = os.path.join(vace_root, "diffusion_pytorch_model.safetensors")
    t5_path = os.path.join(vace_root, "models_t5_umt5-xxl-enc-bf16.pth")
    vae_path = os.path.join(vace_root, "Wan2.1_VAE.pth")
    for path in (diffusion_path, t5_path, vae_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required VACE model file not found: {path}")
    return json.dumps([diffusion_path, t5_path, vae_path])


def resolve_latest_checkpoint(root: Optional[str]) -> Optional[str]:
    if root in (None, "", "none", "None"):
        return None
    path = Path(root)
    if path.is_file():
        return str(path)
    if not path.exists():
        raise FileNotFoundError(f"Preset checkpoint root not found: {path}")
    candidates = sorted(
        [p for p in path.rglob("checkpoint.safetensors") if p.is_file()],
        key=checkpoint_sort_key,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint.safetensors found under {path}")
    return str(candidates[-1])


def model_fn_wan_video_with_state_vace(
    dit,
    motion_controller=None,
    vace=None,
    vap=None,
    animate_adapter=None,
    state_vace_adapter: Optional[VaceStateAdapter] = None,
    latents: torch.Tensor | None = None,
    timestep: torch.Tensor | None = None,
    context: torch.Tensor | None = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents=None,
    vace_context=None,
    vace_scale=1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    vap_hidden_state=None,
    vap_clip_feature=None,
    context_vap=None,
    drop_motion_frames: bool = True,
    tea_cache=None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    longcat_latents=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
    wantodance_refimage_feature=None,
    wantodance_fps: float = 30.0,
    music_feature=None,
    skip_9th_layer: bool = False,
    oracle_state: Optional[torch.Tensor] = None,
    oracle_visibility: Optional[torch.Tensor] = None,
    context_frame_count: Optional[int] = None,
    num_frames: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    **kwargs,
):
    generated_vace_context = None
    if state_vace_adapter is not None and oracle_state is not None:
        context_frame_count = int(context_frame_count or 0)
        if vace_context is not None:
            state_total_latent_frames = int(vace_context.shape[2])
        else:
            state_total_latent_frames = int(latents.shape[2])
        if context_frame_count > 0:
            context_latent_indices = resolve_context_latent_indices_from_frames(
                raw_frame_indices=list(range(context_frame_count)),
                raw_num_frames=int(num_frames or 0),
                latent_length=state_total_latent_frames,
            )
            context_latent_len = len(context_latent_indices)
        else:
            context_latent_len = 0
        generated_vace_context = state_vace_adapter.build_vace_context(
            oracle_state=oracle_state,
            total_latent_frames=state_total_latent_frames,
            clean_prefix_len=int(context_latent_len),
            latent_height=int((vace_context.shape[3] if vace_context is not None else latents.shape[3])),
            latent_width=int((vace_context.shape[4] if vace_context is not None else latents.shape[4])),
            frame_height=int(height or 1),
            frame_width=int(width or 1),
            oracle_visibility=oracle_visibility,
        )
        if vace_context is None:
            vace_context = generated_vace_context
        elif vace_context.shape == generated_vace_context.shape:
            vace_context = vace_context + generated_vace_context

    return model_fn_wan_video(
        dit=dit,
        motion_controller=motion_controller,
        vace=vace,
        vap=vap,
        animate_adapter=animate_adapter,
        latents=latents,
        timestep=timestep,
        context=context,
        clip_feature=clip_feature,
        y=y,
        reference_latents=reference_latents,
        vace_context=vace_context,
        vace_scale=vace_scale,
        audio_embeds=audio_embeds,
        motion_latents=motion_latents,
        s2v_pose_latents=s2v_pose_latents,
        vap_hidden_state=vap_hidden_state,
        vap_clip_feature=vap_clip_feature,
        context_vap=context_vap,
        drop_motion_frames=drop_motion_frames,
        tea_cache=tea_cache,
        use_unified_sequence_parallel=use_unified_sequence_parallel,
        motion_bucket_id=motion_bucket_id,
        pose_latents=pose_latents,
        face_pixel_values=face_pixel_values,
        longcat_latents=longcat_latents,
        sliding_window_size=sliding_window_size,
        sliding_window_stride=sliding_window_stride,
        cfg_merge=cfg_merge,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        control_camera_latents_input=control_camera_latents_input,
        fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        wantodance_refimage_feature=wantodance_refimage_feature,
        wantodance_fps=wantodance_fps,
        music_feature=music_feature,
        skip_9th_layer=skip_9th_layer,
        **kwargs,
    )


def build_vace_condition_video(
    full_video: list[Image.Image],
    context_video: list[Image.Image],
) -> tuple[list[Image.Image], list[Image.Image]]:
    if not full_video:
        raise ValueError("full_video cannot be empty.")
    if not context_video:
        raise ValueError("context_video cannot be empty for VACE state-adapter training.")
    width, height = full_video[0].size
    placeholder = Image.new("RGB", (width, height), (128, 128, 128))
    mask_black = Image.new("RGB", (width, height), (0, 0, 0))
    mask_white = Image.new("RGB", (width, height), (255, 255, 255))
    known_count = len(context_video)
    total_count = len(full_video)
    if known_count >= total_count:
        raise ValueError(
            f"context_video must be shorter than full video for VACE conditioning, got {known_count} >= {total_count}."
        )
    vace_video = list(context_video) + [placeholder.copy() for _ in range(total_count - known_count)]
    vace_video_mask = [mask_black.copy() for _ in range(known_count)] + [
        mask_white.copy() for _ in range(total_count - known_count)
    ]
    return vace_video, vace_video_mask


class StateAwareVaceTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        tokenizer_path=None,
        trainable_models="state_vace_adapter",
        preset_lora_path=None,
        preset_lora_model="dit",
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        device="cpu",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        state_dim=9,
        state_is_normalized=True,
        adapter_hidden_dim=128,
        condition_dropout=0.1,
        use_temporal_encoding=True,
        temporal_embed_dim=32,
        depth_log_scale=4.0,
        velocity_clip=0.5,
        depth_velocity_clip=0.1,
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is disabled. Enabling it to reduce OOM risk.")
            use_gradient_checkpointing = True

        model_configs = self.parse_model_configs(model_paths, None, device=device)
        tokenizer_config = ModelConfig(tokenizer_path or find_tokenizer_path(DEFAULT_VACE_ROOT))
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=None,
            redirect_common_files=False,
        )
        if self.pipe.vace is None:
            raise RuntimeError("Loaded pipeline does not contain a VACE module. Check --vace_root.")

        self.pipe.state_vace_adapter = VaceStateAdapter(
            state_dim=state_dim,
            hidden_dim=adapter_hidden_dim,
            vace_in_dim=int(self.pipe.vace.vace_in_dim),
            condition_dropout=condition_dropout,
            state_is_normalized=state_is_normalized,
            use_temporal_encoding=use_temporal_encoding,
            temporal_embed_dim=temporal_embed_dim,
            depth_log_scale=depth_log_scale,
            velocity_clip=velocity_clip,
            depth_velocity_clip=depth_velocity_clip,
        )
        self.pipe.model_fn = model_fn_wan_video_with_state_vace
        self.pipe.in_iteration_models = tuple(self.pipe.in_iteration_models) + ("state_vace_adapter",)
        if hasattr(self.pipe, "in_iteration_models_2"):
            self.pipe.in_iteration_models_2 = tuple(self.pipe.in_iteration_models_2) + ("state_vace_adapter",)

        self.pipe = self.split_pipeline_units("sft", self.pipe, trainable_models, None)
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models=trainable_models,
            lora_base_model=None,
            lora_target_modules="",
            lora_rank=0,
            lora_checkpoint=None,
            preset_lora_path=preset_lora_path,
            preset_lora_model=preset_lora_model,
            task="sft",
        )
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        vace_video, vace_video_mask = build_vace_condition_video(
            full_video=data["video"],
            context_video=data["context_video"],
        )
        inputs_shared = {
            "input_video": data["video"],
            "vace_video": vace_video,
            "vace_video_mask": vace_video_mask,
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "context_frame_count": len(data["context_video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "oracle_state": data["oracle_state"],
        }
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return flow_match_context_sft_loss(self.pipe, **inputs[0], **inputs[1])


def build_dataset(args):
    dataset = OracleStateWindowDataset(
        dataset_root=args.dataset_root,
        height=args.height,
        width=args.width,
        dataset_repeat=args.dataset_repeat,
        max_pixels=args.max_pixels,
        use_normalized_state=not args.use_raw_state,
        motion_complexity_filter=args.motion_complexity_filter,
        rebalance_motion_complexity=args.rebalance_motion_complexity,
        motion_complexity_rebalance_strength=args.motion_complexity_rebalance_strength,
        object_count_filter=args.object_count_filter,
        future_collision_type_filter=args.future_collision_type_filter,
        future_collision_bucket_filter=args.future_collision_bucket_filter,
    )
    print(
        "OracleStateWindowDataset summary:",
        {
            "num_windows": len(dataset.window_dirs),
            "dataset_repeat": int(dataset.dataset_repeat),
            "dataset_specs": dataset.dataset_specs,
            "dataset_source_summary": dataset.dataset_source_summary,
            "motion_complexity_filter": sorted(dataset.motion_complexity_filter),
            "motion_complexity_summary": dataset.motion_complexity_summary,
            "object_count_filter": sorted(dataset.object_count_filter),
            "object_count_summary": dataset.object_count_summary,
            "future_collision_type_filter": sorted(dataset.future_collision_type_filter),
            "future_collision_type_summary": dataset.future_collision_type_summary,
            "future_collision_bucket_filter": sorted(dataset.future_collision_bucket_filter),
            "future_collision_bucket_summary_size": len(dataset.future_collision_bucket_summary),
            "rebalance_motion_complexity": bool(dataset.rebalance_motion_complexity),
        },
    )
    return dataset


def build_model(args, accelerator):
    preset_lora_path = resolve_latest_checkpoint(args.preset_tv2v_root)
    return StateAwareVaceTrainingModule(
        model_paths=build_vace_model_paths(args.vace_root),
        trainable_models="state_vace_adapter",
        preset_lora_path=preset_lora_path,
        preset_lora_model="dit",
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        tokenizer_path=find_tokenizer_path(args.vace_root),
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        state_dim=args.state_dim,
        state_is_normalized=not args.use_raw_state,
        adapter_hidden_dim=args.adapter_hidden_dim,
        condition_dropout=args.condition_dropout,
        use_temporal_encoding=not args.disable_state_temporal_encoding,
        temporal_embed_dim=args.temporal_embed_dim,
        depth_log_scale=args.depth_log_scale,
        velocity_clip=args.velocity_clip,
        depth_velocity_clip=args.depth_velocity_clip,
    )


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a future-state adapter on top of Wan2.1-VACE-1.3B.",
        allow_abbrev=False,
    )
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    for action in parser._actions:
        if action.dest == "dataset_base_path":
            action.required = False
            action.default = ""
            break
    parser.add_argument("--diffsynth_root", type=str, default=DIFFSYNTH_ROOT)
    parser.add_argument("--vace_root", type=str, default=DEFAULT_VACE_ROOT)
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help=(
            "Single oracle-window root; or an organized summary root whose train/*/*/{no_collision,env_only}/samples.txt "
            "lists packaged window directories."
        ),
    )
    parser.add_argument("--preset_tv2v_root", type=str, default=None)
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb"])
    parser.add_argument("--wandb_project", type=str, default="wan21-vace-state-adapter")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--checkpoint_output_subdir", type=str, default="checkpoints")
    parser.add_argument("--test_output_subdir", type=str, default="test")
    parser.add_argument("--state_dim", type=int, default=9)
    parser.add_argument("--adapter_hidden_dim", type=int, default=128)
    parser.add_argument("--condition_dropout", type=float, default=0.1)
    parser.add_argument("--use_raw_state", action="store_true")
    parser.add_argument("--disable_state_temporal_encoding", action="store_true")
    parser.add_argument("--temporal_embed_dim", type=int, default=32)
    parser.add_argument("--depth_log_scale", type=float, default=4.0)
    parser.add_argument("--velocity_clip", type=float, default=0.5)
    parser.add_argument("--depth_velocity_clip", type=float, default=0.1)
    parser.add_argument(
        "--motion_complexity_filter",
        type=str,
        default="",
        help="Optional comma-separated complexity labels to keep: static,simple,moderate,complex.",
    )
    parser.add_argument(
        "--rebalance_motion_complexity",
        action="store_true",
        help="Use inverse-frequency sampling weights across motion-complexity buckets.",
    )
    parser.add_argument(
        "--motion_complexity_rebalance_strength",
        type=float,
        default=1.0,
        help="Exponent for inverse-frequency motion-complexity weights. 1.0 means exact inverse count.",
    )
    parser.add_argument(
        "--object_count_filter",
        type=str,
        default="",
        help="Optional comma-separated object counts to keep, e.g. 1,2,3.",
    )
    parser.add_argument(
        "--future_collision_type_filter",
        type=str,
        default="",
        help="Optional comma-separated future collision type buckets: none,env_only,obj_obj_only,mixed.",
    )
    parser.add_argument(
        "--future_collision_bucket_filter",
        type=str,
        default="",
        help="Optional comma-separated combined future buckets like obj1__c0__none or obj3__c2plus__mixed.",
    )
    return parser


def prepare_args(args):
    if not getattr(args, "dataset_base_path", ""):
        args.dataset_base_path = args.dataset_root
    if args.resume_from is not None:
        args.resume_from = resolve_resume_state_path(args.resume_from)
    return args


def main():
    args = prepare_args(parser().parse_args())
    accelerator = build_accelerator(args)
    init_trackers(accelerator, args)

    if args.resume_from is not None and accelerator.is_main_process:
        args.lora_checkpoint = resolve_lora_checkpoint_for_resume(args.resume_from)
        accelerator.print(f"Resuming VACE state-adapter training from {args.resume_from}")

    dataset = build_dataset(args)
    model = build_model(args, accelerator)
    model_logger = ModelLogger(
        get_checkpoint_dir(args),
        remove_prefix_in_ckpt="pipe.state_vace_adapter.",
    )
    train_loop(
        accelerator=accelerator,
        dataset=dataset,
        model=model,
        model_logger=model_logger,
        args=args,
        runtime_state={},
    )


if __name__ == "__main__":
    main()
