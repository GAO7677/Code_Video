"""Train Wan VACE with frozen official xSSC as the condition extractor.

This is copied from DiffSynth's Wan training entry point at the level of the
training loop, but the VACE condition unit is replaced locally:

    input video first N ctx frames -> frozen official xSSC -> vace_context
    input video first N ctx frames -> VACE reference-video latents

The official VACE model and its residual hint injection remain unchanged.  This
version intentionally does not add custom per-layer xSSC hooks.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import accelerate
import torch


THIS_DIR = Path(__file__).resolve().parent
DIFFSYNTH_ROOT = Path(
    os.environ.get("DIFFSYNTH_ROOT", "/home/gaoya/Code_Video/DiffSynth-Studio-main")
).expanduser()
UPSTREAM_TRAIN_DIR = THIS_DIR / "upstream_vace_scripts" / "model_training"
for path in (DIFFSYNTH_ROOT, THIS_DIR, UPSTREAM_TRAIN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train import WanTrainingModule, wan_parser  # noqa: E402
from xssc_vace_condition import (  # noqa: E402
    DEFAULT_XSSC_CHECKPOINT,
    DEFAULT_XSSC_CONFIG,
    DEFAULT_XSSC_CONTEXT_FRAMES,
    DEFAULT_XSSC_ROOT,
    DEFAULT_WAN_VAE_TEMPORAL_STRIDE,
    XSSCVACEContextConditioner,
    XSSCVACEContextUnit,
    XSSCVACEReferenceVideoEmbedder,
)

from diffsynth.core import UnifiedDataset  # noqa: E402
from diffsynth.core.data.operators import (  # noqa: E402
    ImageCropAndResize,
    LoadAudio,
    LoadVideo,
    ToAbsolutePath,
)
from diffsynth.diffusion import (  # noqa: E402
    ModelLogger,
    launch_data_process_task,
    launch_training_task,
)


class XSSCVACEWanTrainingModule(WanTrainingModule):
    """Wan training module with xSSC-generated VACE conditions."""

    def __init__(
        self,
        *args,
        xssc_root: str = DEFAULT_XSSC_ROOT,
        xssc_config: str = DEFAULT_XSSC_CONFIG,
        xssc_checkpoint: str = DEFAULT_XSSC_CHECKPOINT,
        xssc_input_size: int = 256,
        xssc_condition_frames: int = DEFAULT_XSSC_CONTEXT_FRAMES,
        xssc_reference_frames: int = DEFAULT_XSSC_CONTEXT_FRAMES,
        xssc_vae_temporal_stride: int = DEFAULT_WAN_VAE_TEMPORAL_STRIDE,
        xssc_slot_dropout: float = 0.0,
        xssc_query_dim: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if str(self.task).endswith(":data_process"):
            raise NotImplementedError(
                "xSSC-VACE condition generation is train-time only in this first "
                "prototype; use task=sft for smoke training."
            )
        if self.pipe.vace is None:
            raise ValueError(
                "The loaded Wan pipeline has no VACE model. Use a VACE model id, "
                "for example Wan-AI/Wan2.1-VACE-1.3B."
            )
        self.xssc_reference_frames = int(xssc_reference_frames)
        if self.xssc_reference_frames < 0:
            raise ValueError("--xssc_reference_frames must be non-negative")
        if self.xssc_reference_frames > int(xssc_condition_frames):
            raise ValueError(
                "--xssc_reference_frames cannot exceed --xssc_condition_frames when "
                "using ctx video as the reference video."
            )

        vace_patch = self.pipe.vace.vace_patch_embedding
        vace_in_dim = int(getattr(self.pipe.vace, "vace_in_dim", vace_patch.in_channels))
        device = vace_patch.weight.device
        dtype = vace_patch.weight.dtype
        self.xssc_conditioner = XSSCVACEContextConditioner(
            xssc_root=xssc_root,
            xssc_config=xssc_config,
            xssc_checkpoint=xssc_checkpoint,
            xssc_input_size=xssc_input_size,
            xssc_condition_frames=xssc_condition_frames,
            temporal_stride=xssc_vae_temporal_stride,
            vace_in_dim=vace_in_dim,
            query_dim=xssc_query_dim,
            slot_dropout=xssc_slot_dropout,
            device=device,
            dtype=dtype,
        )
        self._replace_pipeline_units()

    def _replace_pipeline_units(self) -> None:
        replaced_vace = False
        replaced_embedder = False
        for index, unit in enumerate(self.pipe.units):
            if unit.__class__.__name__ == "WanVideoUnit_InputVideoEmbedder":
                self.pipe.units[index] = XSSCVACEReferenceVideoEmbedder()
                replaced_embedder = True
            if unit.__class__.__name__ == "WanVideoUnit_VACE":
                self.pipe.units[index] = XSSCVACEContextUnit(self.xssc_conditioner)
                replaced_vace = True
        if not replaced_vace:
            raise RuntimeError("Could not find WanVideoUnit_VACE in the pipeline units")
        if self.xssc_reference_frames > 1 and not replaced_embedder:
            raise RuntimeError(
                "Could not find WanVideoUnit_InputVideoEmbedder; multi-frame ctx "
                "reference requires the xSSC-VACE reference-video embedder."
            )

    def train(self, mode: bool = True):
        super().train(mode)
        self.xssc_conditioner.xssc.eval()
        return self

    def get_pipeline_inputs(self, data):
        inputs_shared, inputs_posi, inputs_nega = super().get_pipeline_inputs(data)
        # Keep VACE's official reference marker, but source it only from ctx video.
        # A list means each ctx frame is prepended as one independent reference
        # latent by XSSCVACEReferenceVideoEmbedder.
        if self.xssc_reference_frames > 0:
            if len(data["video"]) < self.xssc_reference_frames:
                raise ValueError(
                    f"Need at least {self.xssc_reference_frames} frames for ctx reference, "
                    f"got {len(data['video'])}"
                )
            inputs_shared["vace_reference_image"] = data["video"][: self.xssc_reference_frames]
        else:
            inputs_shared["vace_reference_image"] = None
        return inputs_shared, inputs_posi, inputs_nega


def add_xssc_vace_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("xssc_vace_condition")
    group.add_argument("--xssc_root", default=DEFAULT_XSSC_ROOT)
    group.add_argument("--xssc_config", default=DEFAULT_XSSC_CONFIG)
    group.add_argument("--xssc_checkpoint", default=DEFAULT_XSSC_CHECKPOINT)
    group.add_argument("--xssc_input_size", type=int, default=256)
    group.add_argument("--xssc_condition_frames", type=int, default=DEFAULT_XSSC_CONTEXT_FRAMES)
    group.add_argument("--xssc_reference_frames", type=int, default=DEFAULT_XSSC_CONTEXT_FRAMES)
    group.add_argument("--xssc_vae_temporal_stride", type=int, default=DEFAULT_WAN_VAE_TEMPORAL_STRIDE)
    group.add_argument("--xssc_slot_dropout", type=float, default=0.0)
    group.add_argument("--xssc_query_dim", type=int, default=256)
    return parser


def build_dataset(args):
    return UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path)
            >> LoadVideo(
                args.num_frames,
                4,
                1,
                frame_processor=ImageCropAndResize(512, 512, None, 16, 16),
            ),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
        },
    )


def build_model(args, accelerator):
    return XSSCVACEWanTrainingModule(
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
        xssc_root=args.xssc_root,
        xssc_config=args.xssc_config,
        xssc_checkpoint=args.xssc_checkpoint,
        xssc_input_size=args.xssc_input_size,
        xssc_condition_frames=args.xssc_condition_frames,
        xssc_reference_frames=args.xssc_reference_frames,
        xssc_vae_temporal_stride=args.xssc_vae_temporal_stride,
        xssc_slot_dropout=args.xssc_slot_dropout,
        xssc_query_dim=args.xssc_query_dim,
    )


def main() -> None:
    parser = add_xssc_vace_args(wan_parser())
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )
    dataset = build_dataset(args)
    model = build_model(args, accelerator)
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft": launch_training_task,
        "sft:train": launch_training_task,
    }
    if args.task not in launcher_map:
        raise NotImplementedError(
            f"xSSC-VACE prototype currently supports task=sft or sft:train, got {args.task!r}"
        )
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
