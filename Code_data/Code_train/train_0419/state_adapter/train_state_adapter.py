"""Train an oracle future-state adapter on top of Wan2.2-TI2V-5B context training."""

from __future__ import annotations

import argparse
import json
import os
import random
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

import accelerate
import torch

from context_wan import (
    ContextAwareWanVideoPipeline,
    WanVideoUnit_ContextVideoEmbedder,
    apply_clean_prefix_to_latents,
    flow_match_context_sft_loss,
    resolve_num_clean_prefix_latents,
)
from diffsynth.core import gradient_checkpoint_forward
from diffsynth.diffusion import (
    DiffusionTrainingModule,
    ModelLogger,
    add_general_config,
    add_video_size_config,
)
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
from diffsynth.pipelines.wan_video import ModelConfig, TemporalTiler_BCTHW, model_fn_longcat_video, model_fn_wans2v, wantodance_get_single_freqs
from diffsynth.models.longcat_video_dit import LongCatVideoTransformer3DModel
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from dataset import WAN_SPATIAL_DIVISIBILITY
from oracle_state_adapter import OracleStateAdapter
from state_adapter_dataset import OracleStateWindowDataset
from train import (
    build_accelerator,
    build_wan22_ti2v5b_model_paths,
    checkpoint_sort_key,
    find_tokenizer_path,
    get_checkpoint_dir,
    init_trackers,
    prepare_args as _unused_prepare_args,
    resolve_resume_state_path,
    resolve_lora_checkpoint_for_resume,
    save_training_state,
    training_checkpoint_file,
    training_state_file,
    train_loop,
)


DEFAULT_WAN_ROOT = "/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"


def resolve_latest_checkpoint(root: Optional[str]) -> Optional[str]:
    if root in (None, "", "none", "None"):
        return None
    path = Path(root)
    if path.is_file():
        return str(path)
    if not path.exists():
        raise FileNotFoundError(f"Preset TV2V root not found: {path}")
    candidates = sorted(
        [p for p in path.rglob("checkpoint.safetensors") if p.is_file()],
        key=checkpoint_sort_key,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint.safetensors found under {path}")
    return str(candidates[-1])


def model_fn_wan_video_with_state_context(
    dit: WanModel,
    motion_controller=None,
    vace=None,
    vap=None,
    animate_adapter: OracleStateAdapter = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
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
    clean_prefix_latents: Optional[torch.Tensor] = None,
    num_clean_prefix_latents: Optional[int] = None,
    oracle_state: Optional[torch.Tensor] = None,
    oracle_visibility: Optional[torch.Tensor] = None,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
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
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
            control_camera_latents_input=control_camera_latents_input,
            clean_prefix_latents=clean_prefix_latents,
            num_clean_prefix_latents=num_clean_prefix_latents,
            oracle_state=oracle_state,
            oracle_visibility=oracle_visibility,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video_with_state_context,
            sliding_window_size,
            sliding_window_stride,
            latents.device,
            latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1,
        )

    if isinstance(dit, LongCatVideoTransformer3DModel):
        return model_fn_longcat_video(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            longcat_latents=longcat_latents,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    if audio_embeds is not None:
        return model_fn_wans2v(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            audio_embeds=audio_embeds,
            motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents,
            drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
        )

    clean_prefix_len = resolve_num_clean_prefix_latents(
        clean_prefix_latents=clean_prefix_latents,
        num_clean_prefix_latents=num_clean_prefix_latents,
    )
    if clean_prefix_len > 0:
        latents = apply_clean_prefix_to_latents(latents, clean_prefix_latents)

    if dit.seperated_timestep and (fuse_vae_embedding_in_latents or clean_prefix_len > 0):
        clean_steps = clean_prefix_len if clean_prefix_len > 0 else 1
        token_count_per_latent = latents.shape[3] * latents.shape[4] // 4
        timestep = torch.concat(
            [
                torch.zeros(
                    (clean_steps * token_count_per_latent,),
                    dtype=latents.dtype,
                    device=latents.device,
                ),
                torch.ones(
                    ((latents.shape[2] - clean_steps) * token_count_per_latent,),
                    dtype=latents.dtype,
                    device=latents.device,
                )
                * timestep,
            ]
        ).flatten()
        t = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0)
        )
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    if hasattr(dit, "wantodance_enable_global") and dit.wantodance_enable_global and int(wantodance_fps + 0.5) != 30:
        x = dit.patchify(x, control_camera_latents_input, enable_wantodance_global=True)
    else:
        x = dit.patchify(x, control_camera_latents_input)

    f, h, w = x.shape[2:]
    future_latent_frames = max(int(f) - int(clean_prefix_len), 0)
    state_plan_tokens = None
    if animate_adapter is not None and oracle_state is not None and future_latent_frames > 0:
        state_plan_tokens = animate_adapter.encode_future_plan(
            oracle_state=oracle_state,
            target_frames=future_latent_frames,
        )

    x = torch.reshape(x, (x.shape[0], x.shape[1], f * h * w))
    x = x.transpose(1, 2).contiguous()

    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1

    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)

    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False

    if hasattr(dit, "wantodance_enable_global") and dit.wantodance_enable_global:
        if wantodance_refimage_feature is not None:
            refimage_feature_embedding = dit.img_emb_refimage(wantodance_refimage_feature)
            context = torch.cat([refimage_feature_embedding, context], dim=1)
        if (dit.wantodance_enable_dynamicfps or dit.wantodance_enable_unimodel) and int(wantodance_fps + 0.5) != 30:
            freqs_0 = wantodance_get_single_freqs(dit.freqs[0], f, wantodance_fps)
            freqs = torch.cat(
                [
                    freqs_0.view(f, 1, 1, -1).expand(f, h, w, -1),
                    dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(f * h * w, 1, -1).to(x.device)

    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        spatial_tokens_per_frame = int(h * w)
        for block_id, block in enumerate(dit.blocks):
            if skip_9th_layer and block_id == 9:
                continue
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x,
                context,
                t_mod,
                freqs,
            )
            if animate_adapter is not None and state_plan_tokens is not None:
                x = animate_adapter.apply_block_modulation(
                    block_idx=block_id,
                    hidden_states=x,
                    future_plan_tokens=state_plan_tokens,
                    total_frames=int(f),
                    clean_prefix_len=int(clean_prefix_len),
                    spatial_tokens_per_frame=spatial_tokens_per_frame,
                )
        if tea_cache is not None:
            tea_cache.store(x)

    if hasattr(dit, "wantodance_enable_unimodel") and dit.wantodance_enable_unimodel and int(wantodance_fps + 0.5) != 30:
        x = dit.head_global(x, t)
    else:
        x = dit.head(x, t)

    if reference_latents is not None:
        x = x[:, reference_latents.shape[1] :]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x


class StateAwareWanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        tokenizer_path=None,
        trainable_models="animate_adapter",
        preset_lora_path=None,
        preset_lora_model="dit",
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        device="cpu",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        state_dim=9,
        state_hidden_dim=1024,
        state_mlp_hidden_dim=512,
        state_temporal_layers=2,
        state_temporal_heads=8,
        state_pool_heads=4,
        condition_dropout=0.1,
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is disabled. Enabling it to reduce OOM risk.")
            use_gradient_checkpointing = True

        model_configs = self.parse_model_configs(model_paths, None, device=device)
        tokenizer_config = ModelConfig(tokenizer_path or find_tokenizer_path(DEFAULT_WAN_ROOT))
        self.pipe = ContextAwareWanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=None,
        )
        self.pipe.animate_adapter = OracleStateAdapter(
            dit_dim=int(self.pipe.dit.dim),
            num_layers=len(self.pipe.dit.blocks),
            state_dim=state_dim,
            hidden_dim=state_hidden_dim,
            mlp_hidden_dim=state_mlp_hidden_dim,
            temporal_layers=state_temporal_layers,
            temporal_heads=state_temporal_heads,
            pool_heads=state_pool_heads,
            condition_dropout=condition_dropout,
        )
        self.pipe.model_fn = model_fn_wan_video_with_state_context
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
        inputs_shared = {
            "input_video": data["video"],
            "context_video": data["context_video"],
            "input_image": data["video"][0],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
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
    return StateAwareWanTrainingModule(
        model_paths=build_wan22_ti2v5b_model_paths(args.wan_root),
        trainable_models="animate_adapter",
        preset_lora_path=preset_lora_path,
        preset_lora_model="dit",
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        tokenizer_path=find_tokenizer_path(args.wan_root),
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        state_dim=args.state_dim,
        state_hidden_dim=args.state_hidden_dim,
        state_mlp_hidden_dim=args.state_mlp_hidden_dim,
        state_temporal_layers=args.state_temporal_layers,
        state_temporal_heads=args.state_temporal_heads,
        state_pool_heads=args.state_pool_heads,
        condition_dropout=args.condition_dropout,
    )


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an oracle future-state adapter on top of Wan2.2-TI2V-5B.",
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
    parser.add_argument("--wan_root", type=str, default=DEFAULT_WAN_ROOT)
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help=(
            "Single oracle-window root, or a JSON file / JSON string with "
            "{datasets:[{path:'...',repeat:2,name:'genesis'}, ...]} for mixed-source ratios."
        ),
    )
    parser.add_argument("--preset_tv2v_root", type=str, default=None)
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb"])
    parser.add_argument("--wandb_project", type=str, default="wan22-oracle-state-adapter")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--checkpoint_output_subdir", type=str, default="checkpoints")
    parser.add_argument("--test_output_subdir", type=str, default="test")
    parser.add_argument("--state_dim", type=int, default=9)
    parser.add_argument("--state_hidden_dim", type=int, default=1024)
    parser.add_argument("--state_mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--state_temporal_layers", type=int, default=2)
    parser.add_argument("--state_temporal_heads", type=int, default=8)
    parser.add_argument("--state_pool_heads", type=int, default=4)
    parser.add_argument("--condition_dropout", type=float, default=0.1)
    parser.add_argument("--use_raw_state", action="store_true")
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
    if args.height is not None and args.height % WAN_SPATIAL_DIVISIBILITY != 0:
        raise ValueError(f"height must be divisible by {WAN_SPATIAL_DIVISIBILITY}, got {args.height}")
    if args.width is not None and args.width % WAN_SPATIAL_DIVISIBILITY != 0:
        raise ValueError(f"width must be divisible by {WAN_SPATIAL_DIVISIBILITY}, got {args.width}")
    if not hasattr(args, "benchmark_every_steps"):
        args.benchmark_every_steps = None
    if not hasattr(args, "benchmark_wait_timeout_seconds"):
        args.benchmark_wait_timeout_seconds = 12 * 60 * 60
    args.resume_from = resolve_resume_state_path(args.resume_from)
    return args


def main():
    args = prepare_args(parser().parse_args())
    accelerator = build_accelerator(args)
    init_trackers(accelerator, args)

    if args.resume_from is not None and accelerator.is_main_process:
        args.lora_checkpoint = resolve_lora_checkpoint_for_resume(args.resume_from)
        accelerator.print(f"Resuming adapter training from {args.resume_from}")

    dataset = build_dataset(args)
    model = build_model(args, accelerator)
    model_logger = ModelLogger(
        get_checkpoint_dir(args),
        remove_prefix_in_ckpt="pipe.animate_adapter.",
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
