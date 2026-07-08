# Run command example:
# env PYTHONPATH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main" \
# CUDA_VISIBLE_DEVICES="2,3" \
# /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
# /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/infer_stage1b_context_only_no_gt_box_v_newtrain_kubric_two_gpu.py \
#   --checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
#   --context-video /path/to/context.mp4 \
#   --prompt "your prompt" \
#   --output-dir /data/gaoya/agent-data/outputs/kubric_two_gpu_infer \
#   --inference-devices cuda:0,cuda:1 \
#   --output-num-frames 49
#
from __future__ import annotations

"""
Kubric stage1b context-only no-GT-box two-GPU inference wrapper.

This keeps the Kubric no-GT-box object-conditioning path identical to the
existing inference wrapper, but adds a lightweight two-device runtime mode:

  --inference-devices cuda:0,cuda:1
      First device: main Wan inference / DiT / VAE / object adapter
      Second device: JEPA / CoTracker / VGGT auxiliary stack

  --output-num-frames N
      Alias of the base script's --num-frames so callers can specify output
      frames explicitly from the command line.

Example:
  env PYTHONPATH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main" \
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/infer_stage1b_context_only_no_gt_box_v_newtrain_kubric_two_gpu.py \
    --checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
    --context-video /path/to/context.mp4 \
    --prompt "your prompt" \
    --output-dir /data/gaoya/agent-data/outputs/kubric_two_gpu_infer \
    --inference-devices cuda:0,cuda:1 \
    --output-num-frames 49
"""

import os
import sys
from typing import Sequence

from code_vjepa_vggt.train0705 import (
    infer_stage1b_context_only_no_gt_box_v_newtrain0705 as base,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    infer_stage1b_context_only_no_gt_box_v_newtrain_kubric as kubric_base,
)


_ORIG_RESOLVE_LAUNCH_DEVICE = base._resolve_launch_device
_ORIG_RESOLVE_AUX_DEVICE = base._resolve_aux_device

_TWO_GPU_MAIN_DEVICE: str | None = None
_TWO_GPU_AUX_DEVICE: str | None = None


def _normalize_device_token(raw_value: str) -> str:
    value = str(raw_value).strip()
    if not value:
        raise ValueError("empty device token")
    if value.lower() == "cpu":
        return "cpu"
    if value.isdigit():
        return f"cuda:{int(value)}"
    if value.startswith("cuda:"):
        suffix = value.split(":", 1)[1].strip()
        if suffix.isdigit():
            return f"cuda:{int(suffix)}"
    raise ValueError(
        f"unsupported device token: {raw_value!r}; expected forms like '0', '1', 'cuda:0', 'cuda:1'"
    )


def _parse_two_gpu_devices(raw_value: str) -> tuple[str, str]:
    parts = [part.strip() for part in str(raw_value).split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(
            f"--inference-devices expects exactly two devices, got {raw_value!r}"
        )
    main_device = _normalize_device_token(parts[0])
    aux_device = _normalize_device_token(parts[1])
    return main_device, aux_device


def _extract_option_value(argv: Sequence[str], option_name: str) -> tuple[str | None, list[str]]:
    remaining: list[str] = [str(argv[0])]
    found_value: str | None = None
    index = 1
    while index < len(argv):
        token = str(argv[index])
        if token == option_name:
            if index + 1 >= len(argv):
                raise ValueError(f"{option_name} requires a value")
            found_value = str(argv[index + 1])
            index += 2
            continue
        prefix = f"{option_name}="
        if token.startswith(prefix):
            found_value = token[len(prefix) :]
            index += 1
            continue
        remaining.append(token)
        index += 1
    return found_value, remaining


def _rewrite_cli(argv: Sequence[str]) -> list[str]:
    global _TWO_GPU_MAIN_DEVICE, _TWO_GPU_AUX_DEVICE

    inference_devices_raw, stripped = _extract_option_value(argv, "--inference-devices")
    if inference_devices_raw is not None:
        _TWO_GPU_MAIN_DEVICE, _TWO_GPU_AUX_DEVICE = _parse_two_gpu_devices(inference_devices_raw)

    output_num_frames_raw, stripped = _extract_option_value(stripped, "--output-num-frames")
    if output_num_frames_raw is not None:
        stripped = list(stripped)
        stripped.extend(["--num-frames", str(int(output_num_frames_raw))])

    return list(stripped)


def _resolve_launch_device_two_gpu() -> str:
    if _TWO_GPU_MAIN_DEVICE is not None:
        return _TWO_GPU_MAIN_DEVICE
    return _ORIG_RESOLVE_LAUNCH_DEVICE()


def _resolve_aux_device_two_gpu(args) -> str | None:
    explicit_value = _ORIG_RESOLVE_AUX_DEVICE(args)
    if explicit_value is not None:
        return explicit_value
    return _TWO_GPU_AUX_DEVICE


def main() -> None:
    sys.argv = _rewrite_cli(sys.argv)
    base.t0705 = kubric_base.trainmod
    base._build_object_context = kubric_base._build_object_context
    base._build_model_args = kubric_base._build_model_args
    base._resolve_launch_device = _resolve_launch_device_two_gpu
    base._resolve_aux_device = _resolve_aux_device_two_gpu
    os.environ.setdefault("KUBRIC_TWO_GPU_INFER", "1")
    base.main()


if __name__ == "__main__":
    main()
