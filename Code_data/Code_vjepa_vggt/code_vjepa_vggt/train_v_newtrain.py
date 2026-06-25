"""该脚本用于训练 Wan2.2 TI2V LoRA 视频生成模型；当前输入数据集路径为 /data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train，模型权重路径为 /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B，输出目录为 /data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_ctx49_736x1280_lora，产出 LoRA 检查点、训练日志和 benchmark 结果。"""
import argparse
import json
import math
import os
import random
import signal
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path


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
if DIFFSYNTH_ROOT and DIFFSYNTH_ROOT not in sys.path:
    sys.path.insert(0, DIFFSYNTH_ROOT)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import accelerate
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from code_vjepa_vggt.adapters.cotracker_adapter import CoTrackerAdapter
from code_vjepa_vggt.adapters.jepa_adapter import JEPAPatchAdapter
from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackAdapter
from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset
from code_vjepa_vggt.models.object_aux_heads import ObjectAuxHeads
from code_vjepa_vggt.models.object_condition_adapter import ObjectConditionAdapter
from code_vjepa_vggt.models.object_tokens import ObjectTubeProjector
from code_vjepa_vggt.utils.vggt_cache import VGGTDenseCache, load_vggt_cache
from code_vjepa_vggt.utils.track_supervision import align_tracks_to_boxes, track_box_iou_loss, track_box_l1_loss
from code_vjepa_vggt.context_wan_v_newtrain import (
    ContextAwareWanVideoPipeline,
    enable_object_condition_branch,
    flow_match_context_sft_loss,
)
from diffsynth.diffusion import (
    DiffusionTrainingModule,
    DirectDistillLoss,
    ModelLogger,
    add_general_config,
    add_video_size_config,
    launch_data_process_task,
)
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
from diffsynth.pipelines.wan_video import ModelConfig


DEFAULT_WAN_ROOT = "/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"
WAN_SPATIAL_DIVISIBILITY = 32
DEFAULT_BENCHMARK_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "batch_eval_lora.py",
)
DEFAULT_VALIDATION_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "run_validation_vbench.py",
)
DEFAULT_CHECKPOINT_SUBDIR = "checkpoints"
DEFAULT_TEST_SUBDIR = "test"
DEFAULT_BENCHMARK_WAIT_TIMEOUT_SECONDS = 12 * 60 * 60
DEFAULT_CONTEXT_REFERENCE_PREFIXES = (1, 4, 8, 12, 16)


def _tensor_video_to_pil_list(video_cthw: torch.Tensor) -> list[Image.Image]:
    frames = video_cthw.detach().cpu().permute(1, 2, 3, 0)
    frames = ((frames + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
    return [Image.fromarray(frame) for frame in frames]


def _sample_points_from_box(box_xyxy: torch.Tensor, points_per_object: int) -> torch.Tensor:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy.tolist()]
    if x1 <= x0 or y1 <= y0:
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        return torch.tensor([[cx, cy]] * points_per_object, dtype=torch.float32)
    cols = max(1, int(math.ceil(math.sqrt(float(points_per_object)))))
    rows = max(1, int(math.ceil(float(points_per_object) / float(cols))))
    xs = torch.linspace(x0 + 0.2 * (x1 - x0), x0 + 0.8 * (x1 - x0), cols)
    ys = torch.linspace(y0 + 0.2 * (y1 - y0), y0 + 0.8 * (y1 - y0), rows)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
    return points[:points_per_object].contiguous()


def _set_module_requires_grad(module: nn.Module | None, requires_grad: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = bool(requires_grad)


def _freeze_unused_object_pooler_geometry_projs(object_pooler: nn.Module | None) -> None:
    if object_pooler is None:
        return
    for name in ("world_proj", "track_geom_proj"):
        submodule = getattr(object_pooler, name, None)
        if submodule is not None:
            _set_module_requires_grad(submodule, False)


class TrainingInterrupted(KeyboardInterrupt):
    """Raised when the training process receives an interrupt signal."""


def install_interrupt_handlers():
    previous_handlers = {}

    def _raise_interrupt(signum, frame):
        signame = signal.Signals(signum).name
        raise TrainingInterrupted(f"Received {signame}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _raise_interrupt)
    return previous_handlers


def restore_interrupt_handlers(previous_handlers):
    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        context_sampling_profile="legacy_prefix",
        min_context_frames=1,
        max_context_ratio=0.5,
        context_reference_frames=49,
        context_reference_prefixes="1,4,8,12,16",
        prefix_context_ratio=0.55,
        first_frame_context_ratio=0.20,
        sparse_context_ratio=0.15,
        random_context_ratio=0.05,
        no_context_ratio=0.05,
        fixed_num_context_frames=8,
        enable_object_branch=False,
        object_num_queries=8,
        aux_max_objects=4,
        jepa_ckpt_path=None,
        jepa_input_size=384,
        jepa_patch_size=16,
        jepa_tubelet_size=2,
        cotracker_checkpoint=None,
        cotracker_input_h=384,
        cotracker_input_w=512,
        cotracker_window_len=60,
        vggt_model_path=None,
        vggt_input_h=420,
        vggt_input_w=728,
        vggt_cache_root=None,
        train_vggt=False,
        object_pooler_latent_dim=16,
        cond_proj_dim=4096,
        jepa_window_radius=1,
        latent_window_radius=1,
        object_track_delta_scale=0.25,
        object_track_gate_init=0.05,
        object_box_delta_scale=0.25,
        object_box_wh_log_scale=2.25,
        object_box_wh_max_scale=2.0,
        object_min_box_px=16.0,
        lambda_track_aux=0.1,
        lambda_box_aux=0.1,
        lambda_depth_aux=0.0,
        lambda_track_box_aux=0.0,
        lambda_track_iou_aux=0.0,
        lambda_object_context_reg=0.0,
        lambda_track_anchor_reg=0.0,
        lambda_box_anchor_reg=0.0,
        lambda_main=1.0,
        depth_target_state_index=None,
        object_gate_init=0.1,
        train_object_pooler=True,
        train_object_aux_heads=True,
        train_object_adapter=True,
        train_object_dit_branch=True,
        freeze_non_object_trainables=False,
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            warnings.warn(
                "Gradient checkpointing is disabled. To prevent OOM, it will be enabled."
            )
            use_gradient_checkpointing = True

        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        tokenizer_config = (
            ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-1.3B",
                origin_file_pattern="google/umt5-xxl/",
            )
            if tokenizer_path is None
            else ModelConfig(tokenizer_path)
        )
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = ContextAwareWanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
        )
        self.pipe = self.split_pipeline_units(
            task, self.pipe, trainable_models, lora_base_model
        )

        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )
        self.enable_object_branch = bool(enable_object_branch)
        self.fixed_num_context_frames = int(fixed_num_context_frames)
        self.aux_max_objects = int(aux_max_objects)
        self.object_num_queries = int(object_num_queries)
        self.total_object_queries = int(self.aux_max_objects * self.object_num_queries)
        self.lambda_track_aux = float(lambda_track_aux)
        self.lambda_box_aux = float(lambda_box_aux)
        self.lambda_depth_aux = float(lambda_depth_aux)
        self.lambda_track_box_aux = float(lambda_track_box_aux)
        self.lambda_track_iou_aux = float(lambda_track_iou_aux)
        self.lambda_object_context_reg = float(lambda_object_context_reg)
        self.lambda_track_anchor_reg = float(lambda_track_anchor_reg)
        self.lambda_box_anchor_reg = float(lambda_box_anchor_reg)
        self.lambda_main = float(lambda_main)
        self.object_track_delta_scale = float(object_track_delta_scale)
        self.object_track_gate_init = float(object_track_gate_init)
        self.object_box_delta_scale = float(object_box_delta_scale)
        self.object_box_wh_log_scale = float(object_box_wh_log_scale)
        self.object_box_wh_max_scale = float(object_box_wh_max_scale)
        self.object_min_box_px = float(object_min_box_px)
        self.object_gate_init = float(object_gate_init)
        self.train_object_pooler = bool(train_object_pooler)
        self.train_object_aux_heads = bool(train_object_aux_heads)
        self.train_object_adapter = bool(train_object_adapter)
        self.train_object_dit_branch = bool(train_object_dit_branch)
        self.freeze_non_object_trainables = bool(freeze_non_object_trainables)
        self.train_vggt = bool(train_vggt)
        self.vggt_cache_root = None if vggt_cache_root is None else str(vggt_cache_root)
        self.depth_target_state_index = (
            None if depth_target_state_index is None else int(depth_target_state_index)
        )

        if self.freeze_non_object_trainables:
            for _, param in self.pipe.dit.named_parameters():
                param.requires_grad = False

        if self.enable_object_branch:
            self.pipe.dit = enable_object_condition_branch(
                self.pipe.dit,
                object_gate_init=float(self.object_gate_init),
                reinitialize_object_branch=True,
            )
            cond_dim = int(cond_proj_dim)
            self.jepa_adapter = JEPAPatchAdapter(
                ckpt_path=str(jepa_ckpt_path),
                device=str(device),
                crop_size=int(jepa_input_size),
                num_frames=max(1, self.fixed_num_context_frames),
                patch_size=int(jepa_patch_size),
                tubelet_size=int(jepa_tubelet_size),
                trainable=False,
            )
            self.cotracker_adapter = CoTrackerAdapter(
                checkpoint_path=cotracker_checkpoint,
                num_queries=self.total_object_queries,
                device=str(device),
                input_hw=(int(cotracker_input_h), int(cotracker_input_w)),
                window_len=int(cotracker_window_len),
            )
            self.vggt_adapter = VGGTTrackAdapter(
                model_path=vggt_model_path,
                num_queries=self.total_object_queries,
                device=str(device),
                input_hw=(int(vggt_input_h), int(vggt_input_w)),
                trainable=bool(self.train_vggt),
            )
            if self.vggt_cache_root is not None and str(self.vggt_cache_root).strip():
                self.vggt_cache_root = str(Path(self.vggt_cache_root).expanduser().resolve())
            self.object_pooler = ObjectTubeProjector(
                jepa_dim=int(self.jepa_adapter.encoder.backbone.embed_dim),
                latent_dim=int(object_pooler_latent_dim),
                out_dim=cond_dim,
                vggt_dense_dim=int(self.vggt_adapter.patch_token_dim),
                jepa_window_radius=int(jepa_window_radius),
                latent_window_radius=int(latent_window_radius),
                min_box_px=float(object_min_box_px),
            )
            self.object_aux_heads = ObjectAuxHeads(
                dim=cond_dim,
                track_delta_scale=float(object_track_delta_scale),
                track_gate_init=float(object_track_gate_init),
                box_delta_scale=float(object_box_delta_scale),
                box_wh_log_scale=float(object_box_wh_log_scale),
                box_wh_max_scale=float(object_box_wh_max_scale),
            )
            self.object_adapter = ObjectConditionAdapter(
                dim=cond_dim,
                num_slots=self.aux_max_objects,
                max_time_steps=64,
            )
            _set_module_requires_grad(self.object_pooler, self.train_object_pooler)
            # Freeze only dormant geometry projections so the trainable set matches
            # the simplified object branch. `depth_proj` is still active.
            _freeze_unused_object_pooler_geometry_projs(self.object_pooler)
            _set_module_requires_grad(self.object_aux_heads, self.train_object_aux_heads)
            _set_module_requires_grad(self.object_adapter, self.train_object_adapter)
            _set_module_requires_grad(self.vggt_adapter, self.train_vggt)
            for name, param in self.pipe.dit.named_parameters():
                if (
                    "object_embedding" in name
                    or ".object_cross_attn." in name
                    or ".object_gate" in name
                    or ".norm4." in name
                ):
                    param.requires_grad = bool(self.train_object_dit_branch)
        else:
            self.jepa_adapter = None
            self.cotracker_adapter = None
            self.vggt_adapter = None
            self.object_pooler = None
            self.object_aux_heads = None
            self.object_adapter = None

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: flow_match_context_sft_loss(
                pipe, **inputs_shared, **inputs_posi
            ),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: flow_match_context_sft_loss(
                pipe, **inputs_shared, **inputs_posi
            ),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(
                pipe, **inputs_shared, **inputs_posi
            ),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(
                pipe, **inputs_shared, **inputs_posi
            ),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.context_sampling_profile = str(context_sampling_profile).strip().lower()
        self.min_context_frames = min_context_frames
        self.max_context_ratio = max_context_ratio
        self.context_reference_frames = max(1, int(context_reference_frames))
        self.context_reference_prefixes = self._parse_context_reference_prefixes(
            context_reference_prefixes
        )
        self.prefix_context_ratio = float(prefix_context_ratio)
        self.first_frame_context_ratio = float(first_frame_context_ratio)
        self.sparse_context_ratio = float(sparse_context_ratio)
        self.random_context_ratio = float(random_context_ratio)
        self.no_context_ratio = no_context_ratio
        self.last_train_metrics = {}

    def trainable_modules(self):
        params = []
        if self.enable_object_branch:
            params.extend(list(self.object_pooler.parameters()))
            params.extend(list(self.object_aux_heads.parameters()))
            params.extend(list(self.object_adapter.parameters()))
            params.extend(
                [
                    param
                    for name, param in self.pipe.dit.named_parameters()
                    if (
                        "object_embedding" in name
                        or ".object_cross_attn." in name
                        or ".object_gate" in name
                        or ".norm4." in name
                    )
                ]
            )
        else:
            params.extend(list(super().trainable_modules()))
        pipe_params = [
            param
            for name, param in self.pipe.dit.named_parameters()
            if param.requires_grad and all(
                token not in name for token in ("object_embedding", ".object_cross_attn.", ".object_gate", ".norm4.")
            )
        ]
        params.extend(pipe_params)
        unique = []
        seen = set()
        for param in params:
            if not param.requires_grad:
                continue
            key = id(param)
            if key in seen:
                continue
            seen.add(key)
            unique.append(param)
        return unique

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = {
            name
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        trainable_param_names.update(
            {
                f"pipe.dit.{name}"
                for name, param in self.pipe.dit.named_parameters()
                if param.requires_grad
            }
        )
        out = {
            name: param
            for name, param in state_dict.items()
            if name in trainable_param_names
        }
        if remove_prefix is not None:
            stripped = {}
            for name, param in out.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix) :]
                stripped[name] = param
            out = stripped
        return out

    @staticmethod
    def _parse_context_reference_prefixes(raw_value):
        if isinstance(raw_value, str):
            prefixes = [
                int(item.strip())
                for item in raw_value.split(",")
                if item.strip()
            ]
        else:
            prefixes = [int(item) for item in raw_value]
        prefixes = sorted({value for value in prefixes if value > 0})
        if not prefixes:
            raise ValueError("context_reference_prefixes must contain at least one positive integer.")
        return prefixes

    @staticmethod
    def _group_tracks_to_objects(
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        *,
        max_objects: int,
        points_per_object: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected_queries = int(max_objects) * int(points_per_object)
        if int(tracks.shape[2]) != expected_queries:
            raise ValueError(
                f"flat query count mismatch: got {int(tracks.shape[2])}, expected {expected_queries}"
            )
        return (
            tracks.view(tracks.shape[0], tracks.shape[1], int(max_objects), int(points_per_object), 2),
            visibility.view(visibility.shape[0], visibility.shape[1], int(max_objects), int(points_per_object)),
            confidence.view(confidence.shape[0], confidence.shape[1], int(max_objects), int(points_per_object)),
        )

    @staticmethod
    def _object_center_tracks_from_grouped(
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        object_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = (visibility * confidence).clamp_min(0.0)
        denom = weights.sum(dim=3, keepdim=True).clamp_min(1.0e-6)
        centers = (tracks * weights.unsqueeze(-1)).sum(dim=3) / denom
        valid = weights.sum(dim=3) > 1.0e-6
        if object_valid_mask is not None:
            valid = valid & (object_valid_mask[:, None, :] > 0.5)
        return centers, valid

    @staticmethod
    def _gather_matched_gt_features(values: torch.Tensor, matched_gt_indices: torch.Tensor) -> torch.Tensor:
        gather_idx = matched_gt_indices[:, None, :, None].expand(-1, values.shape[1], -1, values.shape[-1])
        return torch.gather(values, 2, gather_idx)

    @staticmethod
    def _group_track_summary(
        centers_xy: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        image_hw: tuple[int, int],
        latent_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group = int(centers_xy.shape[1]) // int(latent_frames)
        if int(centers_xy.shape[1]) % int(latent_frames) != 0:
            raise ValueError(
                f"context track frames {int(centers_xy.shape[1])} not divisible by latent_frames={latent_frames}"
            )
        height, width = image_hw
        centers_xy = centers_xy.view(centers_xy.shape[0], int(latent_frames), group, centers_xy.shape[2], 2)
        valid_mask = valid_mask.view(valid_mask.shape[0], int(latent_frames), group, valid_mask.shape[2])
        valid_group = valid_mask.bool().permute(0, 1, 3, 2)
        any_valid = valid_group.any(dim=-1)
        first_idx = valid_group.float().argmax(dim=-1)
        last_idx = group - 1 - valid_group.flip(dims=[-1]).float().argmax(dim=-1)
        first_idx = torch.where(any_valid, first_idx, torch.zeros_like(first_idx))
        last_idx = torch.where(any_valid, last_idx, torch.zeros_like(last_idx))
        centers_perm = centers_xy.permute(0, 1, 3, 2, 4)
        gather_first = first_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 2).long()
        gather_last = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 2).long()
        first_xy = torch.gather(centers_perm, dim=3, index=gather_first).squeeze(3)
        last_xy = torch.gather(centers_perm, dim=3, index=gather_last).squeeze(3)
        last_xy_norm = torch.stack(
            [
                last_xy[..., 0] / max(float(width - 1), 1.0),
                last_xy[..., 1] / max(float(height - 1), 1.0),
            ],
            dim=-1,
        ).clamp(0.0, 1.0)
        delta_xy_norm = torch.stack(
            [
                (last_xy[..., 0] - first_xy[..., 0]) / max(float(width - 1), 1.0),
                (last_xy[..., 1] - first_xy[..., 1]) / max(float(height - 1), 1.0),
            ],
            dim=-1,
        )
        return torch.cat([last_xy_norm, delta_xy_norm], dim=-1), valid_mask.any(dim=2)

    @staticmethod
    def _group_box_targets(
        boxes_xyxy: torch.Tensor,
        valid_mask: torch.Tensor,
        latent_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group = int(boxes_xyxy.shape[1]) // int(latent_frames)
        if int(boxes_xyxy.shape[1]) % int(latent_frames) != 0:
            raise ValueError(
                f"context box frames {int(boxes_xyxy.shape[1])} not divisible by latent_frames={latent_frames}"
            )
        boxes_grouped = boxes_xyxy.view(boxes_xyxy.shape[0], int(latent_frames), group, boxes_xyxy.shape[2], 4)
        valid_grouped = valid_mask.view(valid_mask.shape[0], int(latent_frames), group, valid_mask.shape[2])
        valid_perm = valid_grouped.bool().permute(0, 1, 3, 2)
        any_valid = valid_perm.any(dim=-1)
        last_idx = group - 1 - valid_perm.flip(dims=[-1]).float().argmax(dim=-1)
        last_idx = torch.where(any_valid, last_idx, torch.zeros_like(last_idx))
        boxes_perm = boxes_grouped.permute(0, 1, 3, 2, 4)
        gather_last = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 4).long()
        boxes = torch.gather(boxes_perm, dim=3, index=gather_last).squeeze(3)
        valid = valid_grouped.any(dim=2)
        return boxes, valid

    @staticmethod
    def _box_aux_loss(
        pred_box_xyxy: torch.Tensor,
        gt_box_xyxy: torch.Tensor,
        gt_box_valid: torch.Tensor,
    ) -> torch.Tensor:
        weights = gt_box_valid.unsqueeze(-1).to(dtype=pred_box_xyxy.dtype, device=pred_box_xyxy.device)
        denom = gt_box_valid.sum().clamp_min(1.0)

        pred_center = 0.5 * (pred_box_xyxy[..., :2] + pred_box_xyxy[..., 2:])
        gt_center = 0.5 * (gt_box_xyxy[..., :2] + gt_box_xyxy[..., 2:])
        pred_wh = (pred_box_xyxy[..., 2:] - pred_box_xyxy[..., :2]).clamp_min(1.0e-4)
        gt_wh = (gt_box_xyxy[..., 2:] - gt_box_xyxy[..., :2]).clamp_min(1.0e-4)

        center_l1 = (((pred_center - gt_center).abs()) * weights[..., :2]).sum() / (denom * 2.0)
        wh_l1 = (((pred_wh - gt_wh).abs()) * weights[..., :2]).sum() / (denom * 2.0)

        inter_x0 = torch.maximum(pred_box_xyxy[..., 0], gt_box_xyxy[..., 0])
        inter_y0 = torch.maximum(pred_box_xyxy[..., 1], gt_box_xyxy[..., 1])
        inter_x1 = torch.minimum(pred_box_xyxy[..., 2], gt_box_xyxy[..., 2])
        inter_y1 = torch.minimum(pred_box_xyxy[..., 3], gt_box_xyxy[..., 3])
        inter_w = (inter_x1 - inter_x0).clamp_min(0.0)
        inter_h = (inter_y1 - inter_y0).clamp_min(0.0)
        inter = inter_w * inter_h
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        gt_area = gt_wh[..., 0] * gt_wh[..., 1]
        union = (pred_area + gt_area - inter).clamp_min(1.0e-6)
        iou = inter / union
        iou_loss = ((1.0 - iou) * gt_box_valid.to(dtype=iou.dtype, device=iou.device)).sum() / denom
        return center_l1 + 0.5 * wh_l1 + 0.5 * iou_loss

    @staticmethod
    def _group_last(values: torch.Tensor, latent_frames: int) -> torch.Tensor:
        group = int(values.shape[1]) // int(latent_frames)
        if int(values.shape[1]) % int(latent_frames) != 0:
            raise ValueError(
                f"context value frames {int(values.shape[1])} not divisible by latent_frames={latent_frames}"
            )
        return values.view(values.shape[0], int(latent_frames), group, values.shape[2], values.shape[3])[:, :, -1]

    def _build_object_query_priors(
        self,
        sample: dict,
        *,
        image_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context_boxes = sample["context_boxes"]
        num_context_frames = int(sample["num_context_frames"])
        height, width = image_hw
        box_device = context_boxes.device if isinstance(context_boxes, torch.Tensor) else torch.device("cpu")
        grouped_points = []
        query_frame_ids = []
        valid_mask = []
        box_priors = []
        for object_idx in range(self.aux_max_objects):
            first_valid_frame = None
            box = None
            for frame_idx in range(num_context_frames):
                candidate = context_boxes[frame_idx, object_idx]
                if bool((candidate[2] - candidate[0] > 1.0e-6) and (candidate[3] - candidate[1] > 1.0e-6)):
                    first_valid_frame = frame_idx
                    box = candidate
                    break
            box_valid = box is not None
            valid_mask.append(1.0 if box_valid else 0.0)
            query_frame_ids.extend([float(first_valid_frame if first_valid_frame is not None else 0)] * self.object_num_queries)
            if box_valid:
                points = _sample_points_from_box(box, self.object_num_queries)
                points[:, 0] *= float(width)
                points[:, 1] *= float(height)
                box_priors.append(box.to(device=box_device, dtype=torch.float32))
            else:
                cx = 0.5 * float(width)
                cy = 0.5 * float(height)
                points = torch.tensor([[cx, cy]] * self.object_num_queries, dtype=torch.float32)
                box_priors.append(torch.tensor([0.45, 0.45, 0.55, 0.55], dtype=torch.float32, device=box_device))
            grouped_points.append(points)
        grouped = torch.stack(grouped_points, dim=0)
        flat = grouped.view(1, self.total_object_queries, 2)
        frame_ids = torch.tensor(query_frame_ids, dtype=torch.float32).view(1, self.total_object_queries, 1)
        box_prior_xyxy = torch.stack(box_priors, dim=0).view(1, self.aux_max_objects, 4)
        return flat, frame_ids, torch.tensor(valid_mask, dtype=torch.float32).view(1, self.aux_max_objects), box_prior_xyxy

    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        if not self.enable_object_branch:
            return flow_match_context_sft_loss(pipe, **inputs_shared, **inputs_posi), {}
        sample = inputs_shared["raw_sample"]
        context_video = sample["context_video"].unsqueeze(0).to(device=pipe.device, dtype=pipe.torch_dtype)
        image_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))
        query_points_prior, query_frame_ids, object_valid_mask, box_prior_xyxy = self._build_object_query_priors(sample, image_hw=image_hw)
        query_points_prior = query_points_prior.to(device=pipe.device, dtype=pipe.torch_dtype)
        query_frame_ids = query_frame_ids.to(device=pipe.device, dtype=pipe.torch_dtype)
        object_valid_mask = object_valid_mask.to(device=pipe.device, dtype=pipe.torch_dtype)
        box_prior_xyxy = box_prior_xyxy.to(device=pipe.device, dtype=pipe.torch_dtype)
        frames_bthwc_01 = ((context_video.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0.0, 1.0)
        cotracker_out = self.cotracker_adapter(
            frames_bthwc_01,
            query_points_prior=query_points_prior,
            query_frame_ids=query_frame_ids,
            query_image_hw=image_hw,
        )
        vggt_out: VGGTTrackAdapter | VGGTDenseCache | None = None
        if self.vggt_cache_root:
            cache = load_vggt_cache(sample, self.vggt_cache_root, allow_missing=False)
            vggt_out = cache
        if vggt_out is None:
            if self.vggt_cache_root:
                raise RuntimeError(
                    f"VGGT cache root is set but no cache found for sample {sample.get('video_path', '<unknown>')}"
                )
            vggt_out = self.vggt_adapter(
                frames_bthwc_01,
                query_points_prior=query_points_prior,
                query_image_hw=image_hw,
            )
        tracks_grouped, visibility_grouped, confidence_grouped = self._group_tracks_to_objects(
            cotracker_out.tracks,
            cotracker_out.visibility,
            cotracker_out.confidence,
            max_objects=self.aux_max_objects,
            points_per_object=self.object_num_queries,
        )
        jepa_dtype = next(self.jepa_adapter.parameters()).dtype
        jepa_out = self.jepa_adapter(context_video.to(dtype=jepa_dtype))
        context_latents = inputs_shared["clean_prefix_latents"]
        object_out = self.object_pooler(
            jepa_patch_tokens=jepa_out.patch_tokens,
            context_latents=context_latents,
            tracks=tracks_grouped,
            visibility=visibility_grouped,
            confidence=confidence_grouped,
            track_image_hw=image_hw,
            object_valid_mask=object_valid_mask,
            box_prior_xyxy=box_prior_xyxy,
            vggt_world_points=getattr(vggt_out, "world_points", None),
            vggt_world_points_conf=getattr(vggt_out, "world_points_conf", None),
            vggt_depth=getattr(vggt_out, "depth", None),
            vggt_depth_conf=getattr(vggt_out, "depth_conf", None),
            vggt_dense_patch_tokens=getattr(vggt_out, "dense_patch_tokens", None),
            vggt_patch_grid_hw=getattr(vggt_out, "patch_grid_hw", None),
            vggt_geometry_image_hw=getattr(vggt_out, "image_hw", None),
            frame_valid_mask=None,
        )
        object_aux_out = self.object_aux_heads(
            object_out.object_latent_tokens,
            object_out.active_track_summary,
            object_out.active_box_xyxy,
        )
        object_context = self.object_adapter(
            object_out.object_latent_tokens,
            object_valid_mask=object_valid_mask,
        )

        gt_boxes = sample["context_boxes"].unsqueeze(0).to(device=pipe.device, dtype=pipe.torch_dtype)
        center_tracks_native, center_track_valid = self._object_center_tracks_from_grouped(
            tracks_grouped,
            visibility_grouped,
            confidence_grouped,
            object_valid_mask=object_valid_mask,
        )
        track_alignment = align_tracks_to_boxes(
            tracks=center_tracks_native,
            gt_boxes=gt_boxes,
            image_hw=image_hw,
        )
        track_box_loss = track_box_l1_loss(
            tracks=center_tracks_native,
            matched_gt_centers=track_alignment.matched_gt_centers,
            matched_gt_valid=track_alignment.matched_gt_valid * center_track_valid.to(dtype=track_alignment.matched_gt_valid.dtype),
        )
        track_iou_loss = track_box_iou_loss(
            tracks=center_tracks_native,
            gt_boxes=gt_boxes,
            matched_gt_indices=track_alignment.matched_gt_indices,
            image_hw=image_hw,
            radius_px=12.0,
        )
        image_diag_px = math.sqrt(float(image_hw[0] ** 2 + image_hw[1] ** 2))
        track_box_loss_norm = track_box_loss / max(image_diag_px, 1.0)
        latent_frames = int(object_out.object_latent_tokens.shape[1])
        gt_valid_full = (track_alignment.matched_gt_valid > 0.5) & center_track_valid
        gt_track_summary, gt_track_valid = self._group_track_summary(
            track_alignment.matched_gt_centers,
            gt_valid_full,
            image_hw=image_hw,
            latent_frames=latent_frames,
        )
        matched_gt_boxes = self._gather_matched_gt_features(gt_boxes, track_alignment.matched_gt_indices)
        matched_gt_box_valid = ((matched_gt_boxes[..., 2] - matched_gt_boxes[..., 0]) > 1.0e-6) & (
            (matched_gt_boxes[..., 3] - matched_gt_boxes[..., 1]) > 1.0e-6
        )
        gt_box_xyxy, gt_box_valid = self._group_box_targets(
            matched_gt_boxes,
            matched_gt_box_valid,
            latent_frames,
        )

        track_valid_weights = gt_track_valid.unsqueeze(-1).to(dtype=object_aux_out.pred_track_summary.dtype)
        track_denom = track_valid_weights.sum().clamp_min(1.0)
        track_center_l1 = (((object_aux_out.pred_track_summary[..., :2] - gt_track_summary[..., :2]).abs()) * track_valid_weights[..., :2]).sum() / (
            track_denom * 2.0
        )
        track_delta_l1 = (((object_aux_out.pred_track_summary[..., 2:4] - gt_track_summary[..., 2:4]).abs()) * track_valid_weights[..., :2]).sum() / (
            track_denom * 2.0
        )
        track_aux_loss = track_center_l1 + 0.25 * track_delta_l1
        box_aux_loss = self._box_aux_loss(
            object_aux_out.pred_box_xyxy,
            gt_box_xyxy,
            gt_box_valid,
        )
        track_anchor_reg = object_aux_out.track_delta.abs().mean()
        box_anchor_reg = object_aux_out.box_center_delta.abs().mean() + object_aux_out.box_log_scale.abs().mean()
        depth_aux_loss = track_aux_loss.new_zeros(())
        if self.depth_target_state_index is not None and self.lambda_depth_aux > 0.0:
            gt_states = sample["context_states"].unsqueeze(0).to(device=pipe.device, dtype=pipe.torch_dtype)
            matched_gt_depth = self._gather_matched_gt_features(
                gt_states[..., self.depth_target_state_index : self.depth_target_state_index + 1],
                track_alignment.matched_gt_indices,
            )
            gt_depth = self._group_last(matched_gt_depth, latent_frames)
            pred_depth = object_aux_out.pred_depth
            depth_aux_loss = (pred_depth - gt_depth).abs().mean()

        if self.lambda_main > 0.0:
            loss_main = flow_match_context_sft_loss(
                pipe,
                **inputs_shared,
                **inputs_posi,
                object_context=object_context,
            )
        else:
            loss_main = track_aux_loss.new_zeros(())
        object_context_reg = object_context.square().mean()
        # `track_box_loss` / `track_iou_loss` are measured on the frozen
        # CoTracker-derived center tracks before the trainable aux heads.
        # They are useful diagnostics for track quality, but they do not
        # provide gradient to the trainable object modules in this setup.
        total = (
            self.lambda_main * loss_main
            + self.lambda_track_aux * track_aux_loss
            + self.lambda_box_aux * box_aux_loss
            + self.lambda_depth_aux * depth_aux_loss
            + self.lambda_object_context_reg * object_context_reg
            + self.lambda_track_anchor_reg * track_anchor_reg
            + self.lambda_box_anchor_reg * box_anchor_reg
        )
        object_context_abs = object_context.detach().abs()
        object_latent_tokens_abs = object_out.object_latent_tokens.detach().abs()
        metrics = {
            "train/loss_main": float(loss_main.detach().item()),
            "train/loss_track_aux": float(track_aux_loss.detach().item()),
            "train/loss_box_aux": float(box_aux_loss.detach().item()),
            "train/loss_depth_aux": float(depth_aux_loss.detach().item()),
            "train/loss_track_box_aux": float(track_box_loss_norm.detach().item()),
            "train/loss_track_iou_aux": float(track_iou_loss.detach().item()),
            "train/loss_track_center_aux": float(track_center_l1.detach().item()),
            "train/loss_track_delta_aux": float(track_delta_l1.detach().item()),
            "train/loss_object_context_reg": float(object_context_reg.detach().item()),
            "train/loss_track_anchor_reg": float(track_anchor_reg.detach().item()),
            "train/loss_box_anchor_reg": float(box_anchor_reg.detach().item()),
            "train/track_box_loss": float(track_box_loss.detach().item()),
            "train/track_box_loss_norm": float(track_box_loss_norm.detach().item()),
            "train/track_iou_loss": float(track_iou_loss.detach().item()),
            "train/object_latent_tokens_abs_max": float(object_latent_tokens_abs.max().item()),
            "train/object_context_abs_max": float(object_context_abs.max().item()),
            "train/object_context_abs_mean": float(object_context_abs.mean().item()),
            "train/track_delta_abs_mean": float(object_aux_out.track_delta.detach().abs().mean().item()),
            "train/box_center_delta_abs_mean": float(object_aux_out.box_center_delta.detach().abs().mean().item()),
            "train/box_log_scale_abs_mean": float(object_aux_out.box_log_scale.detach().abs().mean().item()),
        }
        return total, metrics

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared, enable_condition_inputs):
        video_frames = data["video"]
        if isinstance(video_frames, torch.Tensor):
            video_frames = _tensor_video_to_pil_list(data["video"])
        elif len(video_frames) > 0 and not isinstance(video_frames[0], Image.Image):
            video_frames = _tensor_video_to_pil_list(data["video"])
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                if enable_condition_inputs:
                    inputs_shared["input_image"] = video_frames[0]
            elif extra_input == "end_image":
                if enable_condition_inputs:
                    inputs_shared["end_image"] = video_frames[-1]
            elif extra_input in ("reference_image", "vace_reference_image"):
                if enable_condition_inputs:
                    inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            inputs_shared["num_frames"] = 4 * (len(data["video"]) - 1) + 1
        return inputs_shared

    def _legacy_sample_context(self, video):
        total_frames = len(video)
        max_context_frames = min(
            total_frames - 1,
            int(total_frames * self.max_context_ratio),
        )
        if max_context_frames < self.min_context_frames:
            raise ValueError(
                "Context sampling range is empty. "
                f"Got total_frames={total_frames}, min_context_frames={self.min_context_frames}, "
                f"max_context_ratio={self.max_context_ratio}."
            )

        # A small fraction of samples drop all visual conditioning so the same model
        # also learns the pure text-to-video path.
        if random.random() < self.no_context_ratio:
            return {"mode": "text_only", "frame_indices": []}

        context_frames = random.randint(self.min_context_frames, max_context_frames)
        return {
            "mode": "prefix",
            "frame_indices": list(range(context_frames)),
        }

    def _scaled_reference_counts(self, total_frames):
        counts = []
        for ref_count in self.context_reference_prefixes:
            count = math.ceil(total_frames * ref_count / self.context_reference_frames)
            count = max(1, min(int(count), total_frames - 1))
            if count not in counts:
                counts.append(count)
        return counts or [1]

    @staticmethod
    def _sparse_indices(total_frames, count):
        if count <= 1:
            return [0]
        positions = []
        for i in range(count):
            index = round(i * (total_frames - 1) / (count - 1))
            if not positions or index != positions[-1]:
                positions.append(index)
        if positions[-1] != total_frames - 1:
            positions[-1] = total_frames - 1
        cursor = 1
        while len(positions) < count and cursor < total_frames - 1:
            if cursor not in positions:
                positions.insert(-1, cursor)
            cursor += 1
        return sorted(positions[:count])

    def _sample_mixed_context(self, video):
        total_frames = len(video)
        if total_frames < 2:
            raise ValueError(f"Context sampling requires at least 2 frames, got {total_frames}.")

        counts = self._scaled_reference_counts(total_frames)
        multiframe_counts = [count for count in counts if count > 1] or [min(total_frames - 1, 2)]

        draw = random.random()
        thresholds = [
            ("prefix", self.prefix_context_ratio),
            ("first_frame", self.first_frame_context_ratio),
            ("sparse", self.sparse_context_ratio),
            ("random", self.random_context_ratio),
            ("text_only", self.no_context_ratio),
        ]

        cumulative = 0.0
        mode = "text_only"
        for candidate_mode, ratio in thresholds:
            cumulative += ratio
            if draw <= cumulative + 1e-8:
                mode = candidate_mode
                break

        if mode == "text_only":
            return {"mode": mode, "frame_indices": []}
        if mode == "first_frame":
            return {"mode": mode, "frame_indices": [0]}
        if mode == "prefix":
            count = random.choice(counts)
            return {"mode": mode, "frame_indices": list(range(count))}
        if mode == "sparse":
            count = random.choice(multiframe_counts)
            return {
                "mode": mode,
                "frame_indices": self._sparse_indices(total_frames, count),
            }
        count = random.choice(multiframe_counts)
        if count <= 1:
            return {"mode": "first_frame", "frame_indices": [0]}
        middle = sorted(random.sample(range(1, total_frames), count - 1))
        return {"mode": mode, "frame_indices": [0, *middle]}

    def sample_context_spec(self, video):
        if self.context_sampling_profile == "mixed_modes":
            return self._sample_mixed_context(video)
        return self._legacy_sample_context(video)

    def get_pipeline_inputs(self, data):
        if "prompt" in data:
            prompt = data["prompt"]
            video = data["video"]
            raw_sample = None
        else:
            prompt = data["caption"]
            video = _tensor_video_to_pil_list(data["video"])
            raw_sample = data
        inputs_posi = {"prompt": prompt}
        inputs_nega = {}
        context_spec = self.sample_context_spec(video)
        context_frame_indices = context_spec["frame_indices"]
        enable_condition_inputs = len(context_frame_indices) > 0
        inputs_shared = {
            "input_video": video,
            "context_video": None,
            "context_frame_indices": context_frame_indices,
            "sampled_context_frames": len(context_frame_indices),
            "context_sampling_mode": context_spec["mode"],
            "height": video[0].size[1],
            "width": video[0].size[0],
            "num_frames": len(video),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        if raw_sample is not None:
            inputs_shared["raw_sample"] = raw_sample
            inputs_shared["context_video"] = _tensor_video_to_pil_list(raw_sample["context_video"])
            inputs_shared["context_frame_indices"] = raw_sample["context_frame_indices"].tolist()
        inputs_shared = self.parse_extra_inputs(
            data,
            self.extra_inputs,
            inputs_shared,
            enable_condition_inputs=enable_condition_inputs,
        )
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(
            inputs, self.pipe.device, self.pipe.torch_dtype
        )
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        if self.enable_object_branch and "raw_sample" in inputs[0]:
            loss, metrics = self._compute_object_losses(self.pipe, inputs[0], inputs[1])
            self.last_train_metrics = metrics
            self.last_train_metrics["train/loss_total"] = float(loss.detach().item())
            return loss
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        self.last_train_metrics = {
            "train/loss_main": float(loss.detach().item()),
            "train/loss_total": float(loss.detach().item()),
        }
        return loss


def find_tokenizer_path(wan_root):
    candidates = [
        os.path.join(wan_root, "google", "umt5-xxl"),
        os.path.join(wan_root, "google"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"Tokenizer directory not found. Checked: {', '.join(candidates)}"
    )


def build_wan22_ti2v5b_model_paths(wan_root):
    dit_shards = [
        os.path.join(wan_root, "diffusion_pytorch_model-00001-of-00003.safetensors"),
        os.path.join(wan_root, "diffusion_pytorch_model-00002-of-00003.safetensors"),
        os.path.join(wan_root, "diffusion_pytorch_model-00003-of-00003.safetensors"),
    ]
    t5_path = os.path.join(wan_root, "models_t5_umt5-xxl-enc-bf16.pth")
    vae_path = os.path.join(wan_root, "Wan2.2_VAE.pth")

    for path in dit_shards + [t5_path, vae_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required model file not found: {path}")

    return json.dumps([dit_shards, t5_path, vae_path])


def wan_parser():
    parser = argparse.ArgumentParser(
        description="Wan2.2-TI2V-5B LoRA training script.",
        allow_abbrev=False,
    )
    parser = add_general_config(parser)
    for action in parser._actions:
        if action.dest == "dataset_base_path":
            action.required = False
            if action.default is None:
                action.default = ""
            break
    parser = add_video_size_config(parser)
    parser.add_argument(
        "--diffsynth_root",
        type=str,
        default=DIFFSYNTH_ROOT,
        help="Path to DiffSynth-Studio repository.",
    )
    parser.add_argument(
        "--wan_root",
        type=str,
        default=DEFAULT_WAN_ROOT,
        help="Local Wan2.2-TI2V-5B checkpoint directory.",
    )
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--audio_processor_path", type=str, default=None)
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    parser.add_argument("--framewise_decoding", default=False, action="store_true")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Stop training once this many optimizer steps have completed.",
    )
    parser.add_argument(
        "--context_sampling_profile",
        type=str,
        default="legacy_prefix",
        choices=["legacy_prefix", "mixed_modes"],
        help="How to sample context conditioning frames during training.",
    )
    parser.add_argument(
        "--min_context_frames",
        type=int,
        default=1,
        help="Minimum number of raw context frames when conditioning is enabled.",
    )
    parser.add_argument(
        "--max_context_ratio",
        type=float,
        default=0.5,
        help="Maximum context length as a ratio of total video frames. 0.5 means sample up to half the video.",
    )
    parser.add_argument(
        "--context_reference_frames",
        type=int,
        default=49,
        help="Reference video length used to scale the canonical prefix counts.",
    )
    parser.add_argument(
        "--context_reference_prefixes",
        type=str,
        default="1,4,8,12,16",
        help="Canonical prefix counts defined on context_reference_frames.",
    )
    parser.add_argument(
        "--prefix_context_ratio",
        type=float,
        default=0.55,
        help="Probability of using a contiguous prefix context.",
    )
    parser.add_argument(
        "--first_frame_context_ratio",
        type=float,
        default=0.20,
        help="Probability of using only the first frame as context.",
    )
    parser.add_argument(
        "--sparse_context_ratio",
        type=float,
        default=0.15,
        help="Probability of using evenly spaced multi-frame context.",
    )
    parser.add_argument(
        "--random_context_ratio",
        type=float,
        default=0.05,
        help="Probability of using randomly sampled multi-frame context.",
    )
    parser.add_argument(
        "--no_context_ratio",
        type=float,
        default=0.05,
        help="Probability of dropping all condition frames so the model also learns pure T2V.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "wandb"],
        help="Experiment tracker backend.",
    )
    parser.add_argument("--wandb_project", type=str, default="wan-train")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(
        "--benchmark_every_steps",
        type=int,
        default=None,
        help="Run the configured benchmark script every N optimizer steps. Disabled when unset.",
    )
    parser.add_argument(
        "--benchmark_script_path",
        type=str,
        default=DEFAULT_BENCHMARK_SCRIPT,
        help="Path to the benchmark script launched during training.",
    )
    parser.add_argument(
        "--benchmark_meta_list_path",
        type=str,
        default=None,
        help="Meta-list txt used for the fixed visualization benchmark.",
    )
    parser.add_argument(
        "--checkpoint_output_subdir",
        type=str,
        default=DEFAULT_CHECKPOINT_SUBDIR,
        help="Subdirectory inside output_path for persistent training checkpoints.",
    )
    parser.add_argument(
        "--max_checkpoints_keep",
        type=int,
        default=None,
        help="If set, keep only the most recent N step-* checkpoint directories under checkpoint_output_subdir.",
    )
    parser.add_argument(
        "--test_output_subdir",
        type=str,
        default=DEFAULT_TEST_SUBDIR,
        help="Subdirectory inside output_path for evaluation and other test artifacts.",
    )
    parser.add_argument(
        "--benchmark_output_subdir",
        type=str,
        default="physics_iq_benchmark",
        help="Subdirectory inside test_output_subdir for benchmark artifacts.",
    )
    parser.add_argument(
        "--benchmark_cuda_visible_devices",
        type=str,
        default="5,6,7",
        help="CUDA_VISIBLE_DEVICES used by the benchmark subprocess.",
    )
    parser.add_argument(
        "--benchmark_context_frames",
        type=int,
        default=8,
        help="Number of context frames used during benchmark generation.",
    )
    parser.add_argument(
        "--benchmark_num_frames",
        type=int,
        default=161,
        help="Number of frames generated for each benchmark sample.",
    )
    parser.add_argument(
        "--benchmark_height",
        type=int,
        default=720,
        help="Benchmark generation height.",
    )
    parser.add_argument(
        "--benchmark_width",
        type=int,
        default=1280,
        help="Benchmark generation width.",
    )
    parser.add_argument(
        "--benchmark_fps",
        type=int,
        default=30,
        help="Benchmark output FPS.",
    )
    parser.add_argument(
        "--benchmark_num_inference_steps",
        type=int,
        default=50,
        help="Benchmark sampling steps.",
    )
    parser.add_argument(
        "--benchmark_cfg_scale",
        type=float,
        default=5.0,
        help="Benchmark classifier-free guidance scale.",
    )
    parser.add_argument(
        "--benchmark_seed",
        type=int,
        default=42,
        help="Benchmark seed.",
    )
    parser.add_argument(
        "--benchmark_wait_timeout_seconds",
        type=int,
        default=DEFAULT_BENCHMARK_WAIT_TIMEOUT_SECONDS,
        help="How long non-main training ranks wait for the benchmark subprocess to finish before timing out.",
    )
    parser.add_argument(
        "--validation_every_steps",
        type=int,
        default=None,
        help="Run the validation + VBench suite every N optimizer steps.",
    )
    parser.add_argument(
        "--validation_script_path",
        type=str,
        default=DEFAULT_VALIDATION_SCRIPT,
        help="Path to the validation + VBench wrapper script.",
    )
    parser.add_argument(
        "--validation_meta_list_path",
        type=str,
        default=None,
        help="Meta-list txt used for the 100-sample validation suite.",
    )
    parser.add_argument(
        "--validation_context_frames_list",
        type=str,
        default="0,1,2,4,6,8",
        help="Comma-separated context-frame counts evaluated during validation.",
    )
    parser.add_argument(
        "--validation_output_subdir",
        type=str,
        default="validation_vbench",
        help="Subdirectory inside test_output_subdir for validation artifacts.",
    )
    parser.add_argument(
        "--validation_vbench_config_path",
        type=str,
        default=None,
        help="YAML config passed to the VBench validation wrapper.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume full training state from a .state.pt file, a .safetensors checkpoint with matching state file, or a checkpoint directory.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="wan_ti2v",
        choices=["wan_ti2v", "phys_state_episode"],
    )
    parser.add_argument("--phys_state_root", type=str, default=None)
    parser.add_argument("--phys_state_split", type=str, default="train")
    parser.add_argument("--fixed_num_context_frames", type=int, default=8)
    parser.add_argument("--enable_object_branch", action="store_true", default=False)
    parser.add_argument("--object_num_queries", type=int, default=8)
    parser.add_argument("--aux_max_objects", type=int, default=4)
    parser.add_argument("--jepa_ckpt_path", type=str, default=None)
    parser.add_argument("--jepa_input_size", type=int, default=384)
    parser.add_argument("--jepa_patch_size", type=int, default=16)
    parser.add_argument("--jepa_tubelet_size", type=int, default=2)
    parser.add_argument("--cotracker_checkpoint", type=str, default=None)
    parser.add_argument("--cotracker_input_h", type=int, default=384)
    parser.add_argument("--cotracker_input_w", type=int, default=512)
    parser.add_argument("--cotracker_window_len", type=int, default=60)
    parser.add_argument("--vggt_model_path", type=str, default="/data/gaoya/ckpt/facebook-VGGT-1B")
    parser.add_argument("--vggt_input_h", type=int, default=420)
    parser.add_argument("--vggt_input_w", type=int, default=728)
    parser.add_argument("--vggt_cache_root", type=str, default=None)
    parser.add_argument("--train_vggt", action="store_true", default=False)
    parser.add_argument("--object_pooler_latent_dim", type=int, default=16)
    parser.add_argument("--cond_proj_dim", type=int, default=4096)
    parser.add_argument("--jepa_window_radius", type=int, default=1)
    parser.add_argument("--latent_window_radius", type=int, default=1)
    parser.add_argument("--object_track_delta_scale", type=float, default=0.25)
    parser.add_argument("--object_track_gate_init", type=float, default=0.05)
    parser.add_argument("--object_box_delta_scale", type=float, default=0.25)
    parser.add_argument("--object_box_wh_log_scale", type=float, default=2.25)
    parser.add_argument("--object_box_wh_max_scale", type=float, default=2.0)
    parser.add_argument("--object_min_box_px", type=float, default=16.0)
    parser.add_argument("--object_gate_init", type=float, default=0.1)
    parser.add_argument("--lambda_main", type=float, default=1.0)
    parser.add_argument("--lambda_track_aux", type=float, default=0.1)
    parser.add_argument("--lambda_box_aux", type=float, default=0.1)
    parser.add_argument("--lambda_depth_aux", type=float, default=0.0)
    parser.add_argument("--lambda_track_box_aux", type=float, default=0.0)
    parser.add_argument("--lambda_track_iou_aux", type=float, default=0.0)
    parser.add_argument("--lambda_track_anchor_reg", type=float, default=0.0)
    parser.add_argument("--lambda_box_anchor_reg", type=float, default=0.0)
    parser.add_argument("--lambda_object_context_reg", type=float, default=0.0)
    parser.add_argument("--train_object_pooler", action="store_true", default=False)
    parser.add_argument("--train_object_aux_heads", action="store_true", default=False)
    parser.add_argument("--train_object_adapter", action="store_true", default=False)
    parser.add_argument("--train_object_dit_branch", action="store_true", default=False)
    parser.add_argument("--freeze_non_object_trainables", action="store_true", default=False)
    parser.add_argument("--depth_target_state_index", type=int, default=None)
    return parser


def prepare_args(args):
    if args.model_paths is None and args.model_id_with_origin_paths is None:
        args.model_paths = build_wan22_ti2v5b_model_paths(args.wan_root)
    if args.tokenizer_path is None:
        args.tokenizer_path = find_tokenizer_path(args.wan_root)
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError(f"max_train_steps must be positive when set, got {args.max_train_steps}.")
    if args.height is not None and args.height % WAN_SPATIAL_DIVISIBILITY != 0:
        raise ValueError(
            f"height must be divisible by {WAN_SPATIAL_DIVISIBILITY} for Wan2.2 training, got {args.height}."
        )
    if args.width is not None and args.width % WAN_SPATIAL_DIVISIBILITY != 0:
        raise ValueError(
            f"width must be divisible by {WAN_SPATIAL_DIVISIBILITY} for Wan2.2 training, got {args.width}."
        )
    if args.min_context_frames < 1:
        raise ValueError(
            f"min_context_frames must be at least 1, got {args.min_context_frames}."
        )
    if not 0.0 <= args.no_context_ratio <= 1.0:
        raise ValueError(
            f"no_context_ratio must be in [0, 1], got {args.no_context_ratio}."
        )
    ratio_total = (
        args.prefix_context_ratio
        + args.first_frame_context_ratio
        + args.sparse_context_ratio
        + args.random_context_ratio
        + args.no_context_ratio
    )
    if args.context_sampling_profile == "mixed_modes" and abs(ratio_total - 1.0) > 1e-6:
        raise ValueError(
            "Mixed context ratios must sum to 1.0, got "
            f"{ratio_total:.6f} from prefix={args.prefix_context_ratio}, "
            f"first_frame={args.first_frame_context_ratio}, sparse={args.sparse_context_ratio}, "
            f"random={args.random_context_ratio}, no_context={args.no_context_ratio}."
        )
    if not 0.0 < args.max_context_ratio <= 0.5:
        raise ValueError(
            f"max_context_ratio must be in (0, 0.5], got {args.max_context_ratio}."
        )
    if args.benchmark_every_steps is not None and args.benchmark_every_steps <= 0:
        raise ValueError(
            f"benchmark_every_steps must be positive when set, got {args.benchmark_every_steps}."
        )
    if args.benchmark_context_frames < 1:
        raise ValueError(
            f"benchmark_context_frames must be at least 1, got {args.benchmark_context_frames}."
        )
    if args.benchmark_num_frames <= args.benchmark_context_frames:
        raise ValueError(
            "benchmark_num_frames must be larger than benchmark_context_frames, "
            f"got {args.benchmark_num_frames} and {args.benchmark_context_frames}."
        )
    if args.benchmark_every_steps is not None and not os.path.isfile(
        args.benchmark_script_path
    ):
        raise FileNotFoundError(
            f"benchmark_script_path not found: {args.benchmark_script_path}"
        )
    if args.benchmark_every_steps is not None and not args.benchmark_meta_list_path:
        raise ValueError("benchmark_meta_list_path must be set when benchmark_every_steps is enabled.")
    if args.benchmark_meta_list_path is not None and not os.path.isfile(args.benchmark_meta_list_path):
        raise FileNotFoundError(
            f"benchmark_meta_list_path not found: {args.benchmark_meta_list_path}"
        )
    if args.validation_every_steps is not None and args.validation_every_steps <= 0:
        raise ValueError(
            f"validation_every_steps must be positive when set, got {args.validation_every_steps}."
        )
    if args.validation_every_steps is not None and not os.path.isfile(args.validation_script_path):
        raise FileNotFoundError(
            f"validation_script_path not found: {args.validation_script_path}"
        )
    if args.validation_every_steps is not None and not args.validation_meta_list_path:
        raise ValueError("validation_meta_list_path must be set when validation_every_steps is enabled.")
    if args.validation_every_steps is not None and not args.validation_vbench_config_path:
        raise ValueError(
            "validation_vbench_config_path must be set when validation_every_steps is enabled."
        )
    if args.validation_meta_list_path is not None and not os.path.isfile(args.validation_meta_list_path):
        raise FileNotFoundError(
            f"validation_meta_list_path not found: {args.validation_meta_list_path}"
        )
    if args.validation_vbench_config_path is not None and not os.path.isfile(
        args.validation_vbench_config_path
    ):
        raise FileNotFoundError(
            f"validation_vbench_config_path not found: {args.validation_vbench_config_path}"
        )
    validation_contexts = [
        int(item.strip())
        for item in str(args.validation_context_frames_list).split(",")
        if item.strip()
    ]
    if args.validation_every_steps is not None and not validation_contexts:
        raise ValueError("validation_context_frames_list must contain at least one integer.")
    if any(value < 0 for value in validation_contexts):
        raise ValueError(
            f"validation_context_frames_list must contain only non-negative integers, got {validation_contexts}."
        )
    max_context_frames = min(
        args.num_frames - 1,
        int(args.num_frames * args.max_context_ratio),
    )
    if max_context_frames < args.min_context_frames:
        raise ValueError(
            "Context sampling range is empty for the configured video length. "
            f"num_frames={args.num_frames}, min_context_frames={args.min_context_frames}, "
            f"max_context_ratio={args.max_context_ratio}."
        )
    if not args.checkpoint_output_subdir:
        raise ValueError("checkpoint_output_subdir must be a non-empty path segment.")
    if not args.test_output_subdir:
        raise ValueError("test_output_subdir must be a non-empty path segment.")
    if os.path.isabs(args.checkpoint_output_subdir):
        raise ValueError(
            f"checkpoint_output_subdir must be relative to output_path, got {args.checkpoint_output_subdir}."
        )
    if os.path.isabs(args.test_output_subdir):
        raise ValueError(
            f"test_output_subdir must be relative to output_path, got {args.test_output_subdir}."
        )
    if os.path.isabs(args.benchmark_output_subdir):
        raise ValueError(
            f"benchmark_output_subdir must be relative to test_output_subdir, got {args.benchmark_output_subdir}."
        )
    if os.path.isabs(args.validation_output_subdir):
        raise ValueError(
            f"validation_output_subdir must be relative to test_output_subdir, got {args.validation_output_subdir}."
        )
    args.resume_from = resolve_resume_state_path(args.resume_from)
    args.validation_context_frames_list = validation_contexts
    return args


def build_accelerator(args):
    log_with = args.report_to if args.report_to != "none" else None
    return accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
        log_with=log_with,
    )


def init_trackers(accelerator, args):
    if args.report_to == "none":
        return
    if args.wandb_mode is not None:
        os.environ["WANDB_MODE"] = args.wandb_mode
    accelerator.init_trackers(
        project_name=args.wandb_project,
        config=vars(args),
        init_kwargs={
            "wandb": {
                "entity": args.wandb_entity,
                "name": args.wandb_name
                or os.path.basename(args.output_path.rstrip("/")),
            }
        },
    )


def build_dataset(args):
    if args.dataset_type == "phys_state_episode":
        if not args.phys_state_root:
            raise ValueError("--phys_state_root is required when dataset_type=phys_state_episode")
        return PhysStateEpisodeDataset(
            root=args.phys_state_root,
            split=args.phys_state_split,
            resolution=(args.height, args.width),
            num_context_frames=args.fixed_num_context_frames,
            context_fraction=0.5,
            random_context_frames=False,
            seed=42,
        )
    return WanTI2VDataset(
        dataset_base_path=args.dataset_base_path,
        dataset_metadata_path=args.dataset_metadata_path or None,
        dataset_repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys,
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        framewise_decoding=args.framewise_decoding,
    )


def build_model(args, accelerator):
    return WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
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
        context_reference_frames=args.context_reference_frames,
        context_reference_prefixes=args.context_reference_prefixes,
        prefix_context_ratio=args.prefix_context_ratio,
        first_frame_context_ratio=args.first_frame_context_ratio,
        sparse_context_ratio=args.sparse_context_ratio,
        random_context_ratio=args.random_context_ratio,
        no_context_ratio=args.no_context_ratio,
        fixed_num_context_frames=args.fixed_num_context_frames,
        enable_object_branch=args.enable_object_branch,
        object_num_queries=args.object_num_queries,
        aux_max_objects=args.aux_max_objects,
        jepa_ckpt_path=args.jepa_ckpt_path,
        jepa_input_size=args.jepa_input_size,
        jepa_patch_size=args.jepa_patch_size,
        jepa_tubelet_size=args.jepa_tubelet_size,
        cotracker_checkpoint=args.cotracker_checkpoint,
        cotracker_input_h=args.cotracker_input_h,
        cotracker_input_w=args.cotracker_input_w,
        cotracker_window_len=args.cotracker_window_len,
        vggt_model_path=args.vggt_model_path,
        vggt_input_h=args.vggt_input_h,
        vggt_input_w=args.vggt_input_w,
        vggt_cache_root=args.vggt_cache_root,
        train_vggt=args.train_vggt,
        object_pooler_latent_dim=args.object_pooler_latent_dim,
        cond_proj_dim=args.cond_proj_dim,
        jepa_window_radius=args.jepa_window_radius,
        latent_window_radius=args.latent_window_radius,
        object_track_delta_scale=args.object_track_delta_scale,
        object_track_gate_init=args.object_track_gate_init,
        object_box_delta_scale=args.object_box_delta_scale,
        object_box_wh_log_scale=args.object_box_wh_log_scale,
        object_box_wh_max_scale=args.object_box_wh_max_scale,
        object_min_box_px=args.object_min_box_px,
        object_gate_init=args.object_gate_init,
        lambda_main=args.lambda_main,
        lambda_track_aux=args.lambda_track_aux,
        lambda_box_aux=args.lambda_box_aux,
        lambda_depth_aux=args.lambda_depth_aux,
        lambda_track_box_aux=args.lambda_track_box_aux,
        lambda_track_iou_aux=args.lambda_track_iou_aux,
        lambda_track_anchor_reg=args.lambda_track_anchor_reg,
        lambda_box_anchor_reg=args.lambda_box_anchor_reg,
        lambda_object_context_reg=args.lambda_object_context_reg,
        depth_target_state_index=args.depth_target_state_index,
        train_object_pooler=args.train_object_pooler,
        train_object_aux_heads=args.train_object_aux_heads,
        train_object_adapter=args.train_object_adapter,
        train_object_dit_branch=args.train_object_dit_branch,
        freeze_non_object_trainables=args.freeze_non_object_trainables,
    )


def should_run_benchmark(args, global_step):
    return (
        args.benchmark_every_steps is not None
        and global_step > 0
        and global_step % args.benchmark_every_steps == 0
    )


def should_run_validation(args, global_step):
    return (
        args.validation_every_steps is not None
        and global_step > 0
        and global_step % args.validation_every_steps == 0
    )


def get_checkpoint_dir(args):
    return os.path.join(args.output_path, args.checkpoint_output_subdir)


def get_test_dir(args):
    return os.path.join(args.output_path, args.test_output_subdir)


def training_checkpoint_dir(output_dir, checkpoint_tag):
    return Path(output_dir) / checkpoint_tag


def training_checkpoint_file(output_dir, checkpoint_tag):
    return training_checkpoint_dir(output_dir, checkpoint_tag) / "checkpoint.safetensors"


def training_state_file(output_dir, checkpoint_tag):
    return training_checkpoint_dir(output_dir, checkpoint_tag) / "training_state.pt"


def format_step_tag(step: int) -> str:
    return f"step-{int(step):06d}"


def checkpoint_sort_key(path):
    path = Path(path)
    name = path.name
    if name == "checkpoint.safetensors" or name == "training_state.pt":
        stem = path.parent.name
    elif name.endswith(".state.pt"):
        stem = name[: -len(".state.pt")]
    elif name.endswith(".safetensors"):
        stem = name[: -len(".safetensors")]
    else:
        stem = Path(path).stem
    if stem.startswith("step-"):
        try:
            return (1, int(stem.split("-", 1)[1]), stem)
        except ValueError:
            return (1, -1, stem)
    if stem == "interrupted-latest":
        return (3, 10**12, stem)
    if stem.startswith("interrupted-step-"):
        try:
            return (2, int(stem.split("-")[-1]), stem)
        except ValueError:
            return (2, -1, stem)
    return (0, -1, stem)


def resolve_resume_state_path(resume_from):
    if resume_from is None:
        return None

    resume_from = Path(resume_from)
    if resume_from.is_file():
        if resume_from.suffix == ".pt":
            return str(resume_from)
        if resume_from.suffix == ".safetensors":
            if resume_from.name == "checkpoint.safetensors":
                state_path = resume_from.parent / "training_state.pt"
            else:
                state_path = training_state_file(resume_from.parent, resume_from.stem)
            if not state_path.is_file():
                raise FileNotFoundError(
                    f"Resume state file not found for checkpoint: {state_path}"
                )
            return str(state_path)
        raise ValueError(
            f"Unsupported resume_from file type: {resume_from}. Use a .state.pt file, a .safetensors checkpoint, or a checkpoint directory."
        )

    if not resume_from.exists():
        raise FileNotFoundError(f"resume_from not found: {resume_from}")

    if resume_from.is_dir() and (resume_from / "training_state.pt").is_file():
        return str(resume_from / "training_state.pt")

    search_root = resume_from / "checkpoints" if (resume_from / "checkpoints").is_dir() else resume_from
    state_files = sorted(
        [
            path
            for path in search_root.rglob("training_state.pt")
            if path.is_file()
        ]
        + [
            path
            for path in search_root.rglob("*.state.pt")
            if path.is_file()
        ],
        key=checkpoint_sort_key,
    )
    if not state_files:
        raise FileNotFoundError(
            f"No resume state (*.state.pt) found under: {search_root}"
        )
    return str(state_files[-1])


def build_eval_paths(args, global_step, output_subdir, runtime_namespace):
    step_tag = format_step_tag(global_step)
    benchmark_root = os.path.join(
        get_test_dir(args),
        output_subdir,
        step_tag,
    )
    runtime_root = os.path.join(
        get_test_dir(args),
        "_benchmark_runtime",
        runtime_namespace,
        step_tag,
    )
    checkpoint_path = str(training_checkpoint_file(get_checkpoint_dir(args), step_tag))
    state_path = str(training_state_file(get_checkpoint_dir(args), step_tag))
    summary_path = os.path.join(runtime_root, "summary.json")
    stdout_path = os.path.join(runtime_root, "benchmark.stdout.log")
    stderr_path = os.path.join(runtime_root, "benchmark.stderr.log")
    done_marker_path = os.path.join(runtime_root, "benchmark.done.json")
    failed_marker_path = os.path.join(runtime_root, "benchmark.failed.json")
    return {
        "step_tag": step_tag,
        "benchmark_root": benchmark_root,
        "runtime_root": runtime_root,
        "checkpoint_path": checkpoint_path,
        "state_path": state_path,
        "summary_path": summary_path,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "done_marker_path": done_marker_path,
        "failed_marker_path": failed_marker_path,
    }


def build_benchmark_paths(args, global_step):
    return build_eval_paths(
        args,
        global_step,
        output_subdir=args.benchmark_output_subdir,
        runtime_namespace=args.benchmark_output_subdir,
    )


def build_validation_paths(args, global_step):
    return build_eval_paths(
        args,
        global_step,
        output_subdir=args.validation_output_subdir,
        runtime_namespace=args.validation_output_subdir,
    )


def capture_rng_state():
    return {
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }


def restore_rng_state(payload):
    random_state = payload.get("python_random_state")
    if random_state is not None:
        random.setstate(random_state)
    torch_state = payload.get("torch_rng_state")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_states = payload.get("torch_cuda_rng_state_all")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def build_training_state_payload(
    optimizer,
    scheduler,
    global_step,
    epoch_id,
    batch_in_epoch,
    model_logger,
):
    payload = {
        "global_step": global_step,
        "epoch_id": epoch_id,
        "batch_in_epoch": batch_in_epoch,
        "model_logger_num_steps": model_logger.num_steps,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    payload.update(capture_rng_state())
    return payload


def save_training_state(
    accelerator,
    optimizer,
    scheduler,
    global_step,
    epoch_id,
    batch_in_epoch,
    model_logger,
    state_path,
):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        payload = build_training_state_payload(
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch_id=epoch_id,
            batch_in_epoch=batch_in_epoch,
            model_logger=model_logger,
        )
        state_path = Path(state_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        accelerator.save(payload, str(state_path))
    accelerator.wait_for_everyone()


def save_training_checkpoint_bundle(
    *,
    accelerator,
    model,
    model_logger,
    optimizer,
    scheduler,
    global_step,
    epoch_id,
    batch_in_epoch,
    checkpoint_root,
    checkpoint_tag,
    max_checkpoints_keep,
):
    model_logger.save_model(
        accelerator,
        model,
        str(training_checkpoint_file(checkpoint_root, checkpoint_tag)),
    )
    save_training_state(
        accelerator=accelerator,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        epoch_id=epoch_id,
        batch_in_epoch=batch_in_epoch,
        model_logger=model_logger,
        state_path=training_state_file(
            checkpoint_root,
            checkpoint_tag,
        ),
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        prune_old_checkpoints(
            checkpoint_root,
            max_keep=max_checkpoints_keep,
            accelerator=accelerator,
        )
    accelerator.wait_for_everyone()


def load_training_state(state_path):
    print(f"💚 Loading training state from: {state_path}")
    return torch.load(state_path, map_location="cpu", weights_only=False)


def checkpoint_name_from_state_path(state_path):
    state_path = Path(state_path)
    if state_path.name == "training_state.pt":
        return state_path.parent / "checkpoint.safetensors"
    if state_path.name.endswith(".state.pt"):
        return state_path.name[: -len(".state.pt")] + ".safetensors"
    raise ValueError(f"Unsupported training state file name: {state_path}")


def resolve_lora_checkpoint_for_resume(state_path):
    state_path = Path(state_path)
    checkpoint_name = checkpoint_name_from_state_path(state_path)
    checkpoint_path = (
        checkpoint_name
        if isinstance(checkpoint_name, Path)
        else state_path.with_name(checkpoint_name)
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Matching LoRA checkpoint not found for resume state: {checkpoint_path}"
        )
    return str(checkpoint_path)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _list_step_checkpoint_dirs(checkpoint_root):
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.is_dir():
        return []
    return sorted(
        [
            path
            for path in checkpoint_root.iterdir()
            if path.is_dir() and path.name.startswith("step-")
        ],
        key=checkpoint_sort_key,
    )


def prune_old_checkpoints(checkpoint_root, *, max_keep, accelerator=None):
    if max_keep is None or int(max_keep) <= 0:
        return
    checkpoint_dirs = _list_step_checkpoint_dirs(checkpoint_root)
    if len(checkpoint_dirs) <= int(max_keep):
        return
    stale_dirs = checkpoint_dirs[: len(checkpoint_dirs) - int(max_keep)]
    for checkpoint_dir in stale_dirs:
        shutil.rmtree(checkpoint_dir)
        if accelerator is not None and getattr(accelerator, "is_main_process", False):
            accelerator.print(f"Pruned old checkpoint: {checkpoint_dir}")


def _move_tensor_tree_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = _move_tensor_tree_to_device(item, device)
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _move_tensor_tree_to_device(item, device)
        return value
    if isinstance(value, tuple):
        return tuple(_move_tensor_tree_to_device(item, device) for item in value)
    return value


def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        _move_tensor_tree_to_device(state, device)


def offload_training_state_for_eval(accelerator, model, optimizer):
    accelerator.wait_for_everyone()
    accelerator.unwrap_model(model).to("cpu")
    move_optimizer_state(optimizer, "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    accelerator.wait_for_everyone()


def restore_training_state_after_eval(accelerator, model, optimizer):
    accelerator.wait_for_everyone()
    accelerator.unwrap_model(model).to(accelerator.device)
    move_optimizer_state(optimizer, accelerator.device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    accelerator.wait_for_everyone()


def wait_for_benchmark_completion(accelerator, benchmark_paths, timeout_seconds):
    done_marker = Path(benchmark_paths["done_marker_path"])
    failed_marker = Path(benchmark_paths["failed_marker_path"])
    deadline = time.time() + timeout_seconds

    while True:
        if done_marker.is_file():
            return
        if failed_marker.is_file():
            payload = json.loads(failed_marker.read_text(encoding="utf-8"))
            raise RuntimeError(
                "Benchmark failed on the main process: "
                f"{payload.get('message', 'unknown error')} | marker={failed_marker}"
            )
        if time.time() > deadline:
            raise TimeoutError(
                "Timed out while waiting for benchmark completion marker. "
                f"Checked {done_marker} and {failed_marker}."
            )
        time.sleep(5)


def log_benchmark_results_to_wandb(args, global_step, payload):
    if args.report_to != "wandb":
        return

    try:
        import wandb
    except ImportError:
        return

    video_logs = {}
    for prefix, video_path in payload.get("selected_videos", {}).items():
        if os.path.isfile(video_path):
            video_logs[f"benchmark/video_{prefix}"] = wandb.Video(
                video_path,
                fps=args.benchmark_fps,
                caption=os.path.basename(video_path),
            )

    if video_logs:
        wandb.log(video_logs, step=global_step)


def flatten_numeric_metrics(payload, prefix):
    metrics = {}

    def _walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                _walk(value, path + [str(key)])
        elif isinstance(node, list):
            for index, value in enumerate(node):
                _walk(value, path + [str(index)])
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            metrics[f"{prefix}/{'/'.join(path)}"] = float(node)

    _walk(payload, [])
    return metrics


def run_benchmark(
    accelerator,
    model,
    model_logger,
    args,
    global_step,
    optimizer,
    scheduler,
    epoch_id,
    batch_in_epoch,
):
    benchmark_paths = build_benchmark_paths(args, global_step)
    offload_training_state_for_eval(accelerator, model, optimizer)
    if accelerator.is_main_process:
        accelerator.print(
            f"Preparing checkpoint and running evaluation at step {global_step}."
        )
        os.makedirs(benchmark_paths["benchmark_root"], exist_ok=True)
        os.makedirs(benchmark_paths["runtime_root"], exist_ok=True)
        for marker_path in (
            benchmark_paths["done_marker_path"],
            benchmark_paths["failed_marker_path"],
        ):
            if os.path.isfile(marker_path):
                os.remove(marker_path)
    checkpoint_exists = os.path.isfile(benchmark_paths["checkpoint_path"])
    state_exists = os.path.isfile(benchmark_paths["state_path"])
    if not checkpoint_exists:
        model_logger.save_model(
            accelerator,
            model,
            benchmark_paths["checkpoint_path"],
        )
    if not state_exists:
        save_training_state(
            accelerator=accelerator,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch_id=epoch_id,
            batch_in_epoch=batch_in_epoch,
            model_logger=model_logger,
            state_path=benchmark_paths["state_path"],
        )
    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        wait_for_benchmark_completion(
            accelerator,
            benchmark_paths,
            timeout_seconds=args.benchmark_wait_timeout_seconds,
        )
        restore_training_state_after_eval(accelerator, model, optimizer)
        return

    env = os.environ.copy()
    if args.benchmark_cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.benchmark_cuda_visible_devices

    command = [
        sys.executable,
        args.benchmark_script_path,
        "--wan_root",
        args.wan_root,
        "--meta_list_path",
        args.benchmark_meta_list_path,
        "--output_root",
        benchmark_paths["benchmark_root"],
        "--runtime_root",
        benchmark_paths["runtime_root"],
        "--lora_path",
        benchmark_paths["checkpoint_path"],
        "--model_name",
        benchmark_paths["step_tag"],
        "--height",
        str(args.benchmark_height),
        "--width",
        str(args.benchmark_width),
        "--fps",
        str(args.benchmark_fps),
        "--num_frames",
        str(args.benchmark_num_frames),
        "--context_frames",
        str(args.benchmark_context_frames),
        "--num_inference_steps",
        str(args.benchmark_num_inference_steps),
        "--cfg_scale",
        str(args.benchmark_cfg_scale),
        "--seed",
        str(args.benchmark_seed),
    ]
    if args.benchmark_cuda_visible_devices and "," in args.benchmark_cuda_visible_devices:
        command.append("--multi_gpu")

    try:
        with open(benchmark_paths["stdout_path"], "w", encoding="utf-8") as stdout_file, open(
            benchmark_paths["stderr_path"], "w", encoding="utf-8"
        ) as stderr_file:
            subprocess.run(
                command,
                check=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
            )
    except subprocess.CalledProcessError as exc:
        write_json(
            benchmark_paths["failed_marker_path"],
            {
                "global_step": global_step,
                "returncode": exc.returncode,
                "message": "Benchmark subprocess returned a non-zero exit code.",
                "stdout_path": benchmark_paths["stdout_path"],
                "stderr_path": benchmark_paths["stderr_path"],
            },
        )
        accelerator.print(
            f"Benchmark failed at step {global_step}. "
            f"See logs: {benchmark_paths['stdout_path']} and {benchmark_paths['stderr_path']}."
        )
        accelerator.log({"benchmark/failed": 1}, step=global_step)
        restore_training_state_after_eval(accelerator, model, optimizer)
        return

    if not os.path.isfile(benchmark_paths["summary_path"]):
        write_json(
            benchmark_paths["failed_marker_path"],
            {
                "global_step": global_step,
                "message": "Benchmark finished but summary.json was not produced.",
                "summary_path": benchmark_paths["summary_path"],
            },
        )
        accelerator.print(
            f"Benchmark did not produce summary.json at step {global_step}: "
            f"{benchmark_paths['summary_path']}"
        )
        accelerator.log({"benchmark/failed": 1}, step=global_step)
        restore_training_state_after_eval(accelerator, model, optimizer)
        return

    with open(benchmark_paths["summary_path"], "r", encoding="utf-8") as f:
        payload = json.load(f)

    metrics = {
        f"benchmark/{key}": value
        for key, value in payload.get("summary", {}).items()
        if isinstance(value, (int, float))
    }
    metrics["benchmark/failed"] = 0
    accelerator.log(metrics, step=global_step)
    log_benchmark_results_to_wandb(args, global_step, payload)
    accelerator.print(
        f"Benchmark finished at step {global_step}: "
        f"{payload.get('summary', {})}"
    )
    write_json(
        benchmark_paths["done_marker_path"],
        {
            "global_step": global_step,
            "summary_path": benchmark_paths["summary_path"],
        },
    )
    restore_training_state_after_eval(accelerator, model, optimizer)


def run_validation_suite(
    accelerator,
    model,
    model_logger,
    args,
    global_step,
    optimizer,
    scheduler,
    epoch_id,
    batch_in_epoch,
):
    validation_paths = build_validation_paths(args, global_step)
    offload_training_state_for_eval(accelerator, model, optimizer)
    if accelerator.is_main_process:
        accelerator.print(
            f"Preparing checkpoint and running validation at step {global_step}."
        )
        os.makedirs(validation_paths["benchmark_root"], exist_ok=True)
        os.makedirs(validation_paths["runtime_root"], exist_ok=True)
        for marker_path in (
            validation_paths["done_marker_path"],
            validation_paths["failed_marker_path"],
        ):
            if os.path.isfile(marker_path):
                os.remove(marker_path)
    checkpoint_exists = os.path.isfile(validation_paths["checkpoint_path"])
    state_exists = os.path.isfile(validation_paths["state_path"])
    if not checkpoint_exists:
        model_logger.save_model(
            accelerator,
            model,
            validation_paths["checkpoint_path"],
        )
    if not state_exists:
        save_training_state(
            accelerator=accelerator,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch_id=epoch_id,
            batch_in_epoch=batch_in_epoch,
            model_logger=model_logger,
            state_path=validation_paths["state_path"],
        )
    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        wait_for_benchmark_completion(
            accelerator,
            validation_paths,
            timeout_seconds=args.benchmark_wait_timeout_seconds,
        )
        restore_training_state_after_eval(accelerator, model, optimizer)
        return

    env = os.environ.copy()
    if args.benchmark_cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.benchmark_cuda_visible_devices

    command = [
        sys.executable,
        args.validation_script_path,
        "--wan_root",
        args.wan_root,
        "--meta_list_path",
        args.validation_meta_list_path,
        "--output_root",
        validation_paths["benchmark_root"],
        "--runtime_root",
        validation_paths["runtime_root"],
        "--lora_path",
        validation_paths["checkpoint_path"],
        "--model_name",
        validation_paths["step_tag"],
        "--batch_eval_script_path",
        args.benchmark_script_path,
        "--vbench_config_path",
        args.validation_vbench_config_path,
        "--height",
        str(args.benchmark_height),
        "--width",
        str(args.benchmark_width),
        "--fps",
        str(args.benchmark_fps),
        "--num_frames",
        str(args.benchmark_num_frames),
        "--num_inference_steps",
        str(args.benchmark_num_inference_steps),
        "--cfg_scale",
        str(args.benchmark_cfg_scale),
        "--seed",
        str(args.benchmark_seed),
        "--context_frames_list",
        ",".join(str(item) for item in args.validation_context_frames_list),
    ]
    if args.benchmark_cuda_visible_devices and "," in args.benchmark_cuda_visible_devices:
        command.append("--multi_gpu")

    try:
        with open(validation_paths["stdout_path"], "w", encoding="utf-8") as stdout_file, open(
            validation_paths["stderr_path"], "w", encoding="utf-8"
        ) as stderr_file:
            subprocess.run(
                command,
                check=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
            )
    except subprocess.CalledProcessError as exc:
        write_json(
            validation_paths["failed_marker_path"],
            {
                "global_step": global_step,
                "returncode": exc.returncode,
                "message": "Validation subprocess returned a non-zero exit code.",
                "stdout_path": validation_paths["stdout_path"],
                "stderr_path": validation_paths["stderr_path"],
            },
        )
        accelerator.print(
            f"Validation failed at step {global_step}. "
            f"See logs: {validation_paths['stdout_path']} and {validation_paths['stderr_path']}."
        )
        accelerator.log({"validation/failed": 1}, step=global_step)
        restore_training_state_after_eval(accelerator, model, optimizer)
        return

    if not os.path.isfile(validation_paths["summary_path"]):
        write_json(
            validation_paths["failed_marker_path"],
            {
                "global_step": global_step,
                "message": "Validation finished but summary.json was not produced.",
                "summary_path": validation_paths["summary_path"],
            },
        )
        accelerator.print(
            f"Validation did not produce summary.json at step {global_step}: "
            f"{validation_paths['summary_path']}"
        )
        accelerator.log({"validation/failed": 1}, step=global_step)
        restore_training_state_after_eval(accelerator, model, optimizer)
        return

    try:
        with open(validation_paths["summary_path"], "r", encoding="utf-8") as f:
            payload = json.load(f)

        metrics = flatten_numeric_metrics(payload.get("contexts", {}), "validation")
        summary_metrics = {
            f"validation/{key}": float(value)
            for key, value in payload.get("summary", {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        metrics.update(summary_metrics)
        metrics["validation/failed"] = 0
        accelerator.log(metrics, step=global_step)
        accelerator.print(
            f"Validation finished at step {global_step}: "
            f"{payload.get('summary', {})}"
        )
        write_json(
            validation_paths["done_marker_path"],
            {
                "global_step": global_step,
                "summary_path": validation_paths["summary_path"],
            },
        )
    except Exception as exc:
        write_json(
            validation_paths["failed_marker_path"],
            {
                "global_step": global_step,
                "message": "Validation summary parsing/logging failed.",
                "summary_path": validation_paths["summary_path"],
                "error": repr(exc),
            },
        )
        accelerator.print(
            f"Validation summary processing failed at step {global_step}: {exc}. "
            f"See {validation_paths['summary_path']}."
        )
        accelerator.log({"validation/failed": 1}, step=global_step)
    restore_training_state_after_eval(accelerator, model, optimizer)


def train_loop(accelerator, dataset, model, model_logger, args, runtime_state=None):
    optimizer = torch.optim.AdamW(
        model.trainable_modules(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    sampler = None
    shuffle = True
    sample_weights = getattr(dataset, "sample_weights", None)
    if sample_weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
        accelerator.print(
            "Using WeightedRandomSampler from dataset.sample_weights "
            f"(num_samples={len(sample_weights)})."
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=lambda batch: batch[0],
        num_workers=args.dataset_num_workers,
    )

    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    initialize_deepspeed_gradient_checkpointing(accelerator)
    if runtime_state is not None:
        runtime_state["optimizer"] = optimizer
        runtime_state["scheduler"] = scheduler
    optimizer.zero_grad(set_to_none=True)
    dataset_load_from_cache = bool(getattr(dataset, "load_from_cache", False))

    start_epoch = 0
    resume_batch_in_epoch = 0
    global_step = 0
    if args.resume_from is not None:
        resume_payload = load_training_state(args.resume_from)
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        global_step = resume_payload.get("global_step", 0)
        start_epoch = resume_payload.get("epoch_id", 0)
        resume_batch_in_epoch = resume_payload.get("batch_in_epoch", 0)
        model_logger.num_steps = resume_payload.get(
            "model_logger_num_steps", global_step
        )
        restore_rng_state(resume_payload)
        accelerator.wait_for_everyone()
        accelerator.print(
            "Restored training state: "
            f"global_step={global_step}, epoch_id={start_epoch}, batch_in_epoch={resume_batch_in_epoch}, "
            f"model_logger_num_steps={model_logger.num_steps}"
        )
        if resume_batch_in_epoch > 0 and not dataset_load_from_cache:
            accelerator.print(
                "Resume fast-path enabled for non-cached dataset loading: "
                f"ignoring batch_in_epoch={resume_batch_in_epoch} to avoid replaying and re-decoding "
                "all skipped video batches. Training will resume from the start of the current epoch."
            )
            resume_batch_in_epoch = 0

    progress = {
        "global_step": global_step,
        "epoch_id": start_epoch,
        "batch_in_epoch": resume_batch_in_epoch,
        "model_logger_num_steps": model_logger.num_steps,
    }
    if runtime_state is not None:
        runtime_state["progress"] = progress

    for epoch_id in range(start_epoch, args.num_epochs):
        model.train()
        skip_batches = resume_batch_in_epoch if epoch_id == start_epoch else 0
        progress_bar = tqdm(
            total=len(dataloader),
            initial=skip_batches,
            disable=not accelerator.is_local_main_process,
            desc=f"epoch {epoch_id} | global_step {global_step}",
        )
        if skip_batches > 0:
            accelerator.print(
                f"Resuming epoch {epoch_id}: skipping the first {skip_batches} batches before continuing training."
            )
        for batch_index, data in enumerate(dataloader):
            if batch_index < skip_batches:
                if accelerator.is_local_main_process:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(model):
                loss = model({}, inputs=data) if dataset_load_from_cache else model(data)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    global_step += 1
                    model_logger.num_steps = global_step
                    metrics = {
                        "train/loss": loss.detach().float().item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch_id,
                    }
                    extra_metrics = getattr(accelerator.unwrap_model(model), "last_train_metrics", {})
                    metrics.update(extra_metrics)
                    accelerator.log(metrics, step=global_step)

                progress["global_step"] = global_step
                progress["epoch_id"] = epoch_id
                progress["batch_in_epoch"] = batch_index + 1
                progress["model_logger_num_steps"] = model_logger.num_steps

                if accelerator.is_local_main_process:
                    progress_bar.set_description(
                        f"epoch {epoch_id} | global_step {global_step}"
                    )
                    progress_bar.set_postfix(
                        model_step=model_logger.num_steps,
                        refresh=False,
                    )

                if (
                    args.save_steps is not None
                    and model_logger.num_steps > 0
                    and model_logger.num_steps % args.save_steps == 0
                ):
                    checkpoint_tag = format_step_tag(model_logger.num_steps)
                    save_training_checkpoint_bundle(
                        accelerator=accelerator,
                        model=model,
                        model_logger=model_logger,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        epoch_id=epoch_id,
                        batch_in_epoch=batch_index + 1,
                        checkpoint_root=get_checkpoint_dir(args),
                        checkpoint_tag=checkpoint_tag,
                        max_checkpoints_keep=args.max_checkpoints_keep,
                    )

                if accelerator.sync_gradients and should_run_benchmark(args, global_step):
                    run_benchmark(
                        accelerator,
                        model,
                        model_logger,
                        args,
                        global_step,
                        optimizer,
                        scheduler,
                        epoch_id,
                        batch_index + 1,
                    )
                if accelerator.sync_gradients and should_run_validation(args, global_step):
                    run_validation_suite(
                        accelerator,
                        model,
                        model_logger,
                        args,
                        global_step,
                        optimizer,
                        scheduler,
                        epoch_id,
                        batch_index + 1,
                    )
            progress_bar.update(1)
            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                break
        progress_bar.close()

        accelerator.log({"train/epoch_end": epoch_id}, step=global_step)
        progress["global_step"] = global_step
        progress["epoch_id"] = epoch_id + 1
        progress["batch_in_epoch"] = 0
        progress["model_logger_num_steps"] = model_logger.num_steps
        resume_batch_in_epoch = 0
        if args.save_steps is None:
            model_logger.save_model(
                accelerator,
                model,
                str(
                    training_checkpoint_file(
                        get_checkpoint_dir(args),
                        f"epoch-{epoch_id}",
                    )
                ),
            )
        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            break

    if args.save_steps is not None and model_logger.num_steps % args.save_steps != 0:
        checkpoint_tag = format_step_tag(model_logger.num_steps)
        save_training_checkpoint_bundle(
            accelerator=accelerator,
            model=model,
            model_logger=model_logger,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch_id=progress["epoch_id"],
            batch_in_epoch=progress["batch_in_epoch"],
            checkpoint_root=get_checkpoint_dir(args),
            checkpoint_tag=checkpoint_tag,
            max_checkpoints_keep=args.max_checkpoints_keep,
        )
    return progress


def main():
    parser = wan_parser()
    args = prepare_args(parser.parse_args())
    previous_handlers = install_interrupt_handlers()

    accelerator = build_accelerator(args)
    init_trackers(accelerator, args)

    if args.resume_from is not None:
        args.lora_checkpoint = resolve_lora_checkpoint_for_resume(args.resume_from)
        if accelerator.is_main_process:
            accelerator.print(
                f"👉 Resuming training from state {args.resume_from} with checkpoint {args.lora_checkpoint}."
            )

    dataset = build_dataset(args)
    model = build_model(args, accelerator)
    model_logger = ModelLogger(
        get_checkpoint_dir(args),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    runtime_state = {}

    try:
        if args.task in ("sft:data_process", "direct_distill:data_process"):
            launch_data_process_task(accelerator, dataset, model, model_logger, args=args)
        else:
            train_loop(
                accelerator,
                dataset,
                model,
                model_logger,
                args,
                runtime_state=runtime_state,
            )
    except (KeyboardInterrupt, TrainingInterrupted) as exc:
        interrupted_step = model_logger.num_steps
        interrupted_checkpoint_path = training_checkpoint_file(
            get_checkpoint_dir(args), "interrupted-latest"
        )
        accelerator.print(
            f"Training interrupted at step {interrupted_step}. Saving interrupt checkpoint."
        )
        model_logger.save_model(
            accelerator,
            model,
            interrupted_checkpoint_path,
        )
        optimizer = runtime_state.get("optimizer")
        scheduler = runtime_state.get("scheduler")
        progress = runtime_state.get(
            "progress",
            {
                "global_step": 0,
                "epoch_id": 0,
                "batch_in_epoch": 0,
            },
        )
        if optimizer is not None and scheduler is not None:
            save_training_state(
                accelerator=accelerator,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=progress.get("global_step", 0),
                epoch_id=progress.get("epoch_id", 0),
                batch_in_epoch=progress.get("batch_in_epoch", 0),
                model_logger=model_logger,
                state_path=training_state_file(
                    get_checkpoint_dir(args),
                    "interrupted-latest",
                ),
            )
        accelerator.end_training()
        restore_interrupt_handlers(previous_handlers)
        raise SystemExit(130) from exc

    accelerator.end_training()
    restore_interrupt_handlers(previous_handlers)


if __name__ == "__main__":
    main()
