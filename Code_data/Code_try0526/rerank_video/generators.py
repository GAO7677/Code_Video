from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import torch
from PIL import Image

from .schemas import CandidateRecord, GeneratorConfig, InputSpec
from .video_utils import ensure_dir, extract_first_frame, load_context_frames, pil_list_to_numpy, save_video_frames


DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/DiffSynth-Studio-main")
TRAIN0419_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419")

if str(TRAIN0419_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN0419_ROOT))
if str(DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_ROOT))


class CandidateGenerator(Protocol):
    def generate(
        self,
        *,
        input_spec: InputSpec,
        config: GeneratorConfig,
        output_dir: Path,
    ) -> list[CandidateRecord]:
        ...


def _build_model_configs_for_wan(wan_root: Path):
    from diffsynth import ModelConfig

    dit_shards = [
        wan_root / "diffusion_pytorch_model-00001-of-00003.safetensors",
        wan_root / "diffusion_pytorch_model-00002-of-00003.safetensors",
        wan_root / "diffusion_pytorch_model-00003-of-00003.safetensors",
    ]
    t5_path = wan_root / "models_t5_umt5-xxl-enc-bf16.pth"
    vae_path = wan_root / "Wan2.2_VAE.pth"
    for path in dit_shards + [t5_path, vae_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required Wan file not found: {path}")
    return [
        ModelConfig(path=[str(path) for path in dit_shards]),
        ModelConfig(path=str(t5_path)),
        ModelConfig(path=str(vae_path)),
    ]


def _find_tokenizer_path(root: Path) -> Path:
    candidates = [
        root / "google" / "umt5-xxl",
        root / "umt5-xxl",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Tokenizer directory not found under {root}")


class WanContextGenerator:
    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.pipe = self._build_pipeline(config)

    def _build_pipeline(self, config: GeneratorConfig):
        from context_wan import ContextAwareWanVideoPipeline
        from diffsynth import ModelConfig

        if config.model_root is None:
            raise ValueError("Wan generator requires model_root")
        pipe = ContextAwareWanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=config.device,
            model_configs=_build_model_configs_for_wan(config.model_root),
            tokenizer_config=ModelConfig(path=str(_find_tokenizer_path(config.model_root))),
        )
        if config.lora_path is not None:
            pipe.load_lora(pipe.dit, str(config.lora_path), alpha=1.0)
        return pipe

    def generate(self, *, input_spec: InputSpec, config: GeneratorConfig, output_dir: Path) -> list[CandidateRecord]:
        ensure_dir(output_dir)
        context_frames = load_context_frames(
            input_spec.context_video_path,
            context_frames=config.context_frames,
            width=config.width,
            height=config.height,
            resize_mode="crop",
        )
        records: list[CandidateRecord] = []
        for candidate_index in range(config.num_candidates):
            seed = config.base_seed + candidate_index
            candidate_id = f"{config.key}_{candidate_index:03d}_seed{seed}"
            output_path = output_dir / f"{candidate_id}.mp4"
            with torch.no_grad():
                video = self.pipe(
                    prompt=input_spec.prompt,
                    negative_prompt=config.negative_prompt,
                    input_image=context_frames[0],
                    context_video=context_frames,
                    height=config.height,
                    width=config.width,
                    num_frames=config.num_frames,
                    seed=seed,
                    cfg_scale=config.cfg_scale,
                    num_inference_steps=config.num_inference_steps,
                    tiled=True,
                )
            save_video_frames(output_path, pil_list_to_numpy(video[: config.num_frames]), fps=config.fps, quality=config.quality)
            records.append(
                CandidateRecord(
                    candidate_id=candidate_id,
                    generator_key=config.key,
                    generator_type=config.type,
                    seed=seed,
                    video_path=output_path,
                    used_context_frames=config.context_frames,
                    metadata={
                        "conditioning_mode": config.conditioning_mode,
                        "generator_config": asdict(config),
                    },
                )
            )
        return records


class VaceGenerator:
    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.pipe = self._build_pipeline(config)

    def _build_pipeline(self, config: GeneratorConfig):
        from diffsynth import ModelConfig
        from diffsynth.pipelines.wan_video import WanVideoPipeline

        if config.model_root is None:
            raise ValueError("VACE generator requires model_root")
        return WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=config.device,
            model_configs=[
                ModelConfig(path=str(config.model_root / "diffusion_pytorch_model.safetensors")),
                ModelConfig(path=str(config.model_root / "models_t5_umt5-xxl-enc-bf16.pth")),
                ModelConfig(path=str(config.model_root / "Wan2.1_VAE.pth")),
            ],
            tokenizer_config=ModelConfig(
                path=str(config.model_root / "google" / "umt5-xxl"),
                skip_download=True,
            ),
            redirect_common_files=False,
        )

    def generate(self, *, input_spec: InputSpec, config: GeneratorConfig, output_dir: Path) -> list[CandidateRecord]:
        ensure_dir(output_dir)
        context_frames = load_context_frames(
            input_spec.context_video_path,
            context_frames=config.context_frames,
            width=config.width,
            height=config.height,
            resize_mode="crop",
        )
        placeholder = Image.new("RGB", (config.width, config.height), (128, 128, 128))
        mask_black = Image.new("RGB", (config.width, config.height), (0, 0, 0))
        mask_white = Image.new("RGB", (config.width, config.height), (255, 255, 255))
        records: list[CandidateRecord] = []
        for candidate_index in range(config.num_candidates):
            seed = config.base_seed + candidate_index
            candidate_id = f"{config.key}_{candidate_index:03d}_seed{seed}"
            output_path = output_dir / f"{candidate_id}.mp4"
            known_frames = list(context_frames)
            video_input = known_frames + [placeholder.copy() for _ in range(max(config.num_frames - len(known_frames), 0))]
            video_mask = [mask_black.copy() for _ in range(len(known_frames))] + [
                mask_white.copy() for _ in range(max(config.num_frames - len(known_frames), 0))
            ]
            with torch.no_grad():
                video = self.pipe(
                    prompt=input_spec.prompt,
                    negative_prompt=config.negative_prompt,
                    vace_video=video_input,
                    vace_video_mask=video_mask,
                    height=config.height,
                    width=config.width,
                    num_frames=config.num_frames,
                    seed=seed,
                    cfg_scale=config.cfg_scale,
                    num_inference_steps=config.num_inference_steps,
                    tiled=True,
                )
            save_video_frames(output_path, pil_list_to_numpy(video[: config.num_frames]), fps=config.fps, quality=config.quality)
            records.append(
                CandidateRecord(
                    candidate_id=candidate_id,
                    generator_key=config.key,
                    generator_type=config.type,
                    seed=seed,
                    video_path=output_path,
                    used_context_frames=config.context_frames,
                    metadata={
                        "conditioning_mode": "vace_video_mask",
                        "generator_config": asdict(config),
                    },
                )
            )
        return records


class CogVideoXGenerator:
    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.pipe = self._build_pipeline(config)

    def _build_pipeline(self, config: GeneratorConfig):
        from diffusers import CogVideoXImageToVideoPipeline

        if config.model_root is None:
            raise ValueError("CogVideoX generator requires model_root")
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            str(config.model_root),
            torch_dtype=torch.bfloat16,
        )
        pipe = pipe.to(config.device)
        if hasattr(pipe, "enable_model_cpu_offload") and not str(config.device).startswith("cuda"):
            pipe.enable_model_cpu_offload()
        return pipe

    def generate(self, *, input_spec: InputSpec, config: GeneratorConfig, output_dir: Path) -> list[CandidateRecord]:
        ensure_dir(output_dir)
        first_frame = extract_first_frame(input_spec.context_video_path)
        first_frame = first_frame.resize((config.width, config.height), Image.Resampling.BILINEAR)
        records: list[CandidateRecord] = []
        for candidate_index in range(config.num_candidates):
            seed = config.base_seed + candidate_index
            candidate_id = f"{config.key}_{candidate_index:03d}_seed{seed}"
            output_path = output_dir / f"{candidate_id}.mp4"
            generator = torch.Generator(device="cpu").manual_seed(seed)
            result = self.pipe(
                prompt=input_spec.prompt,
                negative_prompt=config.negative_prompt,
                image=first_frame,
                height=config.height,
                width=config.width,
                num_frames=config.num_frames,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.cfg_scale,
                generator=generator,
            )
            frames = result.frames[0] if hasattr(result, "frames") else result[0]
            save_video_frames(output_path, pil_list_to_numpy(frames[: config.num_frames]), fps=config.fps, quality=config.quality)
            records.append(
                CandidateRecord(
                    candidate_id=candidate_id,
                    generator_key=config.key,
                    generator_type=config.type,
                    seed=seed,
                    video_path=output_path,
                    used_context_frames=1,
                    metadata={
                        "conditioning_mode": "first_frame_only",
                        "generator_config": asdict(config),
                    },
                )
            )
        return records


def build_generator(config: GeneratorConfig) -> CandidateGenerator:
    if config.type == "wan":
        return WanContextGenerator(config)
    if config.type == "vace":
        return VaceGenerator(config)
    if config.type == "cogvideox":
        return CogVideoXGenerator(config)
    raise ValueError(f"Unsupported generator type: {config.type}")

