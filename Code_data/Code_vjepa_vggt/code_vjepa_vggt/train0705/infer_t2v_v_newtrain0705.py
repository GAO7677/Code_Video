from __future__ import annotations
"""
Run command example:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/infer_t2v_v_newtrain0705.py \
  --checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-002500 \
  --prompt "Two pillows on a table and two grabber tools hanging above them from which a brown tennis ball and an orange block are suspended. The grabber tools let go of the ball and block. Static shot with no camera movement." \
  --output-dir /data/gaoya/agent-data/outputs/train0705_t2v_demo \
  --sampling-steps 40


  
  
  
  
  
  
  
  
  
  
  
  
  
  Guided example:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=3 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/infer_t2v_v_newtrain0705.py \
  --checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-002500 \
  --prompt "A metal ball drops onto a block on a table. Static shot with no camera movement." \
  --output-dir /data/gaoya/agent-data/outputs/train0705_t2v_vjepa_demo \
  --sampling-steps 40 \
  --vjepa-preset ladder_s20 \
  --enable-vjepa-guidance \
  --vjepa-device cuda:0


Pure text-to-video inference entry for the train0705 runtime stack.

This script reuses the same DiffSynth-native Wan runtime, base LoRA loading, and
optional V-JEPA guidance plumbing as the stage1b train0705 inference script, but
it intentionally disables both:

  1. context-video conditioning
  2. object-branch conditioning

So the actual generation path is:

  prompt -> Wan text encoder / DiT / VAE -> video

The provided --checkpoint is still accepted for bookkeeping and overlap loading,
but the stage1b checkpoints inspected so far are object-branch-only. In that
case the overlap load count will be zero, and the effective generation weights
come from the frozen base LoRA passed by --lora-checkpoint.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import MethodType
from types import SimpleNamespace

import numpy as np
import torch


def _read_cli_arg_value(argv: list[str], names: tuple[str, ...], default: str | None = None) -> str | None:
    for name in names:
        if name not in argv:
            continue
        index = argv.index(name)
        if index + 1 < len(argv):
            return argv[index + 1]
    return default


_DEFAULT_DIFFSYNTH_ROOT = "/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main"
_SELECTED_DIFFSYNTH_ROOT = _read_cli_arg_value(
    sys.argv,
    ("--diffsynth-root", "--diffsynth_root"),
    os.environ.get("DIFFSYNTH_ROOT", _DEFAULT_DIFFSYNTH_ROOT),
)
if _SELECTED_DIFFSYNTH_ROOT:
    os.environ["DIFFSYNTH_ROOT"] = _SELECTED_DIFFSYNTH_ROOT
    if _SELECTED_DIFFSYNTH_ROOT not in sys.path:
        sys.path.insert(0, _SELECTED_DIFFSYNTH_ROOT)

from diffsynth.utils.data import save_video

import code_vjepa_vggt.train_v_newtrain as tvn
from code_vjepa_vggt.train0705 import infer_stage1b_context_only_no_gt_box_v_newtrain0705 as infer0705
from code_vjepa_vggt.train0705 import train_stage1b_context_only_no_gt_box_v_newtrain as t0705


DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
DEFAULT_BASE_LORA = Path(
    "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/"
    "raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors"
)


def _resolve_launch_device() -> str:
    return infer0705._resolve_launch_device()


def _summarize_string_list(values, *, sample_limit: int = 8):
    values = list(values or [])
    return {
        "count": len(values),
        "sample": values[:sample_limit],
    }


def _summarize_load_info(load_info: dict) -> dict:
    summarized = {}
    for name, info in load_info.items():
        summarized[name] = {
            "loaded_count": int(info.get("loaded_count", 0)),
            "selected_source_keys": int(info.get("selected_source_keys", 0)),
            "missing_keys": _summarize_string_list(info.get("missing_keys", [])),
            "unexpected_keys": _summarize_string_list(info.get("unexpected_keys", [])),
            "skipped_shape_mismatch_count": len(info.get("skipped_shape_mismatch", [])),
            "skipped_shape_mismatch_sample": list(info.get("skipped_shape_mismatch", []))[:4],
        }
    return summarized


def _load_overlapping_checkpoint_into_model(model, checkpoint_path):
    state_dict = tvn._load_trainable_state(checkpoint_path)
    model_state = model.state_dict()
    normalized_model_keys = {tvn._normalize_checkpoint_key(key): key for key in model_state.keys()}
    normalized_checkpoint_keys = {tvn._normalize_checkpoint_key(key): key for key in state_dict.keys()}
    overlapping = sorted(set(normalized_model_keys.keys()) & set(normalized_checkpoint_keys.keys()))

    if not overlapping:
        return {
            "loaded_count": 0,
            "missing_keys": [],
            "unexpected_keys": sorted(state_dict.keys()),
            "skipped_shape_mismatch": [],
            "selected_source_keys": len(state_dict),
        }

    remapped_state = {}
    skipped_shape_mismatch = []
    for normalized_key in overlapping:
        model_key = normalized_model_keys[normalized_key]
        source_key = normalized_checkpoint_keys[normalized_key]
        source_value = state_dict[source_key]
        target_value = model_state[model_key]
        if tuple(source_value.shape) != tuple(target_value.shape):
            skipped_shape_mismatch.append(
                {
                    "normalized_key": normalized_key,
                    "checkpoint_key": source_key,
                    "model_key": model_key,
                    "checkpoint_shape": list(source_value.shape),
                    "model_shape": list(target_value.shape),
                }
            )
            continue
        remapped_state[model_key] = source_value

    missing_keys, unexpected_keys = model.load_state_dict(remapped_state, strict=False)
    return {
        "loaded_count": len(remapped_state),
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "skipped_shape_mismatch": skipped_shape_mismatch,
        "selected_source_keys": len(state_dict),
    }


def _patch_dit_blocks_for_optional_object_context(dit) -> None:
    for block in getattr(dit, "blocks", []):
        if getattr(block, "object_cross_attn", None) is not None:
            continue
        original_forward = block.forward

        def _forward_with_optional_object_context(
            self,
            x,
            context,
            t_mod,
            freqs,
            object_context=None,
            _original_forward=original_forward,
        ):
            return _original_forward(x, context, t_mod, freqs)

        block.forward = MethodType(_forward_with_optional_object_context, block)


def _build_model_args(args: argparse.Namespace) -> argparse.Namespace:
    parser = t0705.build_parser()
    model_args = parser.parse_args([])

    model_args.diffsynth_root = str(args.diffsynth_root)
    model_args.wan_root = str(args.wan_root)
    model_args.height = int(args.height)
    model_args.width = int(args.width)
    model_args.num_frames = int(args.num_frames)
    model_args.max_train_steps = 1
    model_args.num_epochs = 1
    model_args.output_path = str(args.output_dir)
    model_args.dataset_type = "wan_ti2v"
    model_args.report_to = "none"
    model_args.initialize_model_on_cpu = bool(getattr(args, "initialize_model_on_cpu", False))

    model_args.lora_base_model = "dit"
    model_args.lora_target_modules = "q,k,v,o,ffn.0,ffn.2"
    model_args.lora_rank = int(args.lora_rank)
    model_args.lora_alpha = int(args.lora_alpha)
    model_args.lora_checkpoint = str(args.lora_checkpoint)
    model_args.extra_inputs = None

    model_args.enable_object_branch = False
    model_args.freeze_non_object_trainables = False
    model_args.train_object_adapter = False
    model_args.train_object_dit_branch = False
    model_args.train_object_pooler = False
    model_args.train_object_aux_heads = False

    model_args.fixed_num_context_frames = 8

    return tvn.prepare_args(model_args)


def _build_runtime_model(args: argparse.Namespace):
    infer0705.apply_vjepa_preset_if_requested(args)
    model_args = _build_model_args(args)
    accelerator = SimpleNamespace(device=torch.device(args.device))
    model = tvn.build_model(model_args, accelerator)

    checkpoint_info = _load_overlapping_checkpoint_into_model(model, args.checkpoint)

    target_device = torch.device(args.device)
    model.to(target_device)
    model.pipe.to(device=target_device, dtype=model.pipe.torch_dtype)
    _patch_dit_blocks_for_optional_object_context(model.pipe.dit)
    model.eval()
    infer0705.configure_runtime_pipe_vjepa(model.pipe, args)
    return model, model_args, {
        "checkpoint_overlap_info": checkpoint_info,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run train0705 pure text-to-video inference. "
            "This variant disables context-video and object-branch inputs."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir or checkpoint.safetensors")
    parser.add_argument("--prompt", required=True, help="Prompt / caption for generation")
    parser.add_argument("--output-dir", required=True, help="Directory for video + json outputs")
    parser.add_argument("--negative-prompt", default="", help="Optional negative prompt")
    parser.add_argument("--wan-root", default=str(DEFAULT_WAN_ROOT))
    parser.add_argument("--diffsynth-root", default=str(DEFAULT_DIFFSYNTH_ROOT))
    parser.add_argument("--lora-checkpoint", default=str(DEFAULT_BASE_LORA))
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--initialize-model-on-cpu", action="store_true")
    infer0705.add_vjepa_cli_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer0705.apply_vjepa_preset_if_requested(args)
    args.device = _resolve_launch_device()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, model_args, load_info = _build_runtime_model(args)
    pipe = model.pipe
    pipe.dit.eval()

    with torch.no_grad():
        video = pipe(
            prompt=str(args.prompt),
            negative_prompt=str(args.negative_prompt),
            seed=int(args.seed),
            tiled=True,
            height=int(args.height),
            width=int(args.width),
            num_frames=int(args.num_frames),
            num_inference_steps=int(args.sampling_steps),
            cfg_scale=float(args.cfg_scale),
        )

    checkpoint_path = Path(tvn._resolve_checkpoint_file(args.checkpoint)).resolve()
    checkpoint_tag = checkpoint_path.parent.name
    output_video = output_dir / f"{checkpoint_tag}.mp4"
    save_video(video, str(output_video), fps=int(args.fps), quality=int(args.quality))

    result = {
        "checkpoint": str(checkpoint_path),
        "output_video": str(output_video),
        "prompt": str(args.prompt),
        "negative_prompt": str(args.negative_prompt),
        "mode": "text_to_video",
        "object_branch_enabled": False,
        "context_video_used": False,
        "model_device": str(args.device),
        "model_args": {
            "height": int(model_args.height),
            "width": int(model_args.width),
            "num_frames": int(model_args.num_frames),
            "lora_checkpoint": str(model_args.lora_checkpoint),
        },
        "load_info": _summarize_load_info(load_info),
        "vjepa": infer0705.summarize_vjepa_args(args),
    }
    (output_dir / f"{checkpoint_tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
