"""该脚本用于训练 Wan2.2 TI2V LoRA 视频生成模型；当前输入数据集路径为 /data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train，模型权重路径为 /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B，输出目录为 /data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_ctx49_736x1280_lora，产出 LoRA 检查点、训练日志和 benchmark 结果。"""
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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
from tqdm import tqdm

from dataset import WanTI2VDataset
from context_wan import ContextAwareWanVideoPipeline, flow_match_context_sft_loss
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

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared, enable_condition_inputs):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                if enable_condition_inputs:
                    inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                if enable_condition_inputs:
                    inputs_shared["end_image"] = data["video"][-1]
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
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        context_spec = self.sample_context_spec(data["video"])
        context_frame_indices = context_spec["frame_indices"]
        enable_condition_inputs = len(context_frame_indices) > 0
        inputs_shared = {
            "input_video": data["video"],
            "context_video": None,
            "context_frame_indices": context_frame_indices,
            "sampled_context_frames": len(context_frame_indices),
            "context_sampling_mode": context_spec["mode"],
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
        }
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
        return self.task_to_loss[self.task](self.pipe, *inputs)


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


def build_wan_model_paths(wan_root):
    dit_shards = [
        os.path.join(wan_root, "diffusion_pytorch_model-00001-of-00003.safetensors"),
        os.path.join(wan_root, "diffusion_pytorch_model-00002-of-00003.safetensors"),
        os.path.join(wan_root, "diffusion_pytorch_model-00003-of-00003.safetensors"),
    ]
    dit_single = os.path.join(wan_root, "diffusion_pytorch_model.safetensors")
    t5_path = os.path.join(wan_root, "models_t5_umt5-xxl-enc-bf16.pth")
    vae22_path = os.path.join(wan_root, "Wan2.2_VAE.pth")
    vae21_path = os.path.join(wan_root, "Wan2.1_VAE.pth")

    if all(os.path.isfile(path) for path in dit_shards) and os.path.isfile(vae22_path):
        for path in dit_shards + [t5_path, vae22_path]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Required model file not found: {path}")
        return json.dumps([dit_shards, t5_path, vae22_path])

    if os.path.isfile(dit_single) and os.path.isfile(vae21_path):
        for path in [dit_single, t5_path, vae21_path]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Required model file not found: {path}")
        return json.dumps([dit_single, t5_path, vae21_path])

    raise FileNotFoundError(
        "Unsupported Wan checkpoint layout under "
        f"{wan_root}. Expected either Wan2.2 3-shard DiT + Wan2.2_VAE.pth "
        "or Wan2.1 single-file DiT + Wan2.1_VAE.pth."
    )


def wan_parser():
    parser = argparse.ArgumentParser(
        description="Wan2.2-TI2V-5B LoRA training script.",
        allow_abbrev=False,
        conflict_handler="resolve",
    )
    parser = add_general_config(parser)
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
    return parser


def prepare_args(args):
    if args.model_paths is None and args.model_id_with_origin_paths is None:
        args.model_paths = build_wan_model_paths(args.wan_root)
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
        if resume_batch_in_epoch > 0 and not dataset.load_from_cache:
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
                loss = model({}, inputs=data) if dataset.load_from_cache else model(data)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    global_step += 1
                    model_logger.num_steps = global_step
                    accelerator.log(
                        {
                            "train/loss": loss.detach().float().item(),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/epoch": epoch_id,
                        },
                        step=global_step,
                    )

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
                    model_logger.save_model(
                        accelerator,
                        model,
                        str(
                            training_checkpoint_file(
                                get_checkpoint_dir(args),
                                checkpoint_tag,
                            )
                        ),
                    )
                    save_training_state(
                        accelerator=accelerator,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        epoch_id=epoch_id,
                        batch_in_epoch=batch_index + 1,
                        model_logger=model_logger,
                        state_path=training_state_file(
                            get_checkpoint_dir(args),
                            checkpoint_tag,
                        ),
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
        model_logger.save_model(
            accelerator,
            model,
            str(
                training_checkpoint_file(
                    get_checkpoint_dir(args),
                    checkpoint_tag,
                )
            ),
        )
        save_training_state(
            accelerator=accelerator,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch_id=progress["epoch_id"],
            batch_in_epoch=progress["batch_in_epoch"],
            model_logger=model_logger,
            state_path=training_state_file(
                get_checkpoint_dir(args),
                checkpoint_tag,
            ),
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
