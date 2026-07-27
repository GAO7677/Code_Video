#!/usr/bin/env python3
"""Capture all-block attention and the exact step input latent in one Wan run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from allblock_ball_query_utils import (
    build_case_recorder_group,
    install_diffsynth_group,
    load_case_query_map,
)
from self_attention_matrix import DiffSynthAttentionScope


def _extract_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attention-output-root", type=Path, required=True)
    parser.add_argument("--attention-blocks", required=True)
    parser.add_argument("--attention-step", type=int, default=25)
    parser.add_argument("--attention-query-map", type=Path, required=True)
    parser.add_argument("--attention-map-heads", required=True)
    parser.add_argument("--latent-output-root", type=Path, required=True)
    return parser.parse_known_args(argv)


def _cli_path(argv: list[str], name: str) -> Path:
    for index, token in enumerate(argv):
        if token == name:
            return Path(argv[index + 1]).expanduser().resolve()
        if token.startswith(f"{name}="):
            return Path(token.split("=", 1)[1]).expanduser().resolve()
    raise ValueError(f"missing required base option {name}")


def _case_map(input_list: Path) -> dict[Path, str]:
    output: dict[Path, str] = {}
    for line in input_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        path = Path(line.strip()).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        output[Path(payload["input_video"]).expanduser().resolve()] = path.stem
    return output


class LatentCaptureScope:
    """Capture model_fn input on the selected positive CFG call."""

    def __init__(self, *, pipe: Any, cfg_scale: float, step: int) -> None:
        self.pipe = pipe
        self.step = int(step)
        self.calls_per_step = 1 if abs(float(cfg_scale) - 1.0) < 1.0e-8 else 2
        self.call_index = 0
        self.original = pipe.model_fn
        self.latent: torch.Tensor | None = None

    def install(self) -> None:
        self.pipe.model_fn = self

    def restore(self) -> None:
        self.pipe.model_fn = self.original

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        step_number = self.call_index // self.calls_per_step + 1
        branch_index = self.call_index % self.calls_per_step
        if step_number == self.step and branch_index == 0:
            if self.latent is not None:
                raise RuntimeError(f"step {self.step} positive latent captured twice")
            latent = kwargs.get("latents")
            if latent is None:
                raise RuntimeError("model_fn did not expose latents")
            self.latent = latent.detach().to(device="cpu").clone()
        try:
            return self.original(*args, **kwargs)
        finally:
            self.call_index += 1


@torch.no_grad()
def _save_and_decode_latent(
    *,
    pipe: Any,
    latent: torch.Tensor,
    output_root: Path,
    case_key: str,
    step: int,
) -> Path:
    case_dir = output_root.expanduser().resolve() / case_key / f"step_{step:02d}"
    frame_dir = case_dir / "vae_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    latent_path = case_dir / "model_input_latent.pt"
    torch.save(
        {
            "latents": latent,
            "denoise_step_one_based": int(step),
            "cfg_branch": "positive",
            "capture_point": "pipe.model_fn input",
        },
        latent_path,
    )

    pipe.load_models_to_device(["vae"])
    try:
        decoded = pipe.vae.decode(
            latent.to(dtype=pipe.torch_dtype),
            device=pipe.device,
            tiled=True,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )
        frames = pipe.vae_output_to_video(decoded)
    finally:
        pipe.load_models_to_device([])

    if latent.ndim != 5 or tuple(latent.shape) != (1, 48, 13, 32, 56):
        raise ValueError(f"unexpected captured latent shape: {tuple(latent.shape)}")
    if len(frames) != 49:
        raise ValueError(f"joint VAE decode returned {len(frames)} frames, expected 49")
    for index, frame in enumerate(frames):
        if frame.size != (896, 512):
            raise ValueError(f"decoded frame {index} has size {frame.size}")
        frame.save(frame_dir / f"frame_{index:03d}.jpg", quality=92)

    groups = []
    for latent_time in range(13):
        rgb_indices = (
            [0]
            if latent_time == 0
            else list(range(1 + 4 * (latent_time - 1), 1 + 4 * latent_time))
        )
        groups.append(
            {
                "latent_time": latent_time,
                "decoded_rgb_frame_indices": rgb_indices,
            }
        )
    manifest = {
        "case": case_key,
        "denoise_step_one_based": int(step),
        "cfg_branch": "positive",
        "capture_point": "exact pipe.model_fn input used by captured attention",
        "latent_path": latent_path.name,
        "latent_shape": list(latent.shape),
        "latent_dtype": str(latent.dtype),
        "vae_decode": "full 13-slice sequence jointly decoded with WanVideoVAE.decode",
        "decoded_frame_count": len(frames),
        "decoded_frame_shape_hw": [512, 896],
        "latent_to_rgb_groups": groups,
        "attention_grid": [13, 16, 28],
        "dit_patch_size": [1, 2, 2],
        "spatial_mapping": {
            "heatmap_to_rgb": "integer cell replication only",
            "repeat_y": 32,
            "repeat_x": 32,
            "interpolation": "none",
        },
    }
    manifest_path = case_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[latent] wrote {manifest_path}", flush=True)
    return manifest_path


def main() -> None:
    custom, remaining = _extract_args(sys.argv[1:])
    if custom.attention_step < 1:
        raise ValueError("--attention-step must be one-based and positive")
    mapping = _case_map(_cli_path(remaining, "--input-json-list-path"))
    query_map = load_case_query_map(custom.attention_query_map)

    from code_vjepa_vggt.AAAinfer import wan_openvid_0613pybullet_lorav2v as base

    original_generate = base.core.generate_one_video

    def generate_with_capture(*args, **kwargs):
        pipe = kwargs["pipe"]
        context_path = Path(kwargs["context_path"]).expanduser().resolve()
        case_key = mapping.get(context_path, context_path.parent.name)
        group = build_case_recorder_group(
            blocks_text=custom.attention_blocks,
            steps_text=str(custom.attention_step),
            model_label="wan_lora",
            output_root=custom.attention_output_root,
            case_key=case_key,
            query_map=query_map,
            map_heads_text=custom.attention_map_heads,
            query_mode="moving",
        )
        group.begin_case(case_key, metadata={"input_video": str(context_path)})
        restore_blocks = install_diffsynth_group(pipe.dit, group)
        attention_scope = DiffSynthAttentionScope(
            pipe=pipe,
            recorder=group,
            cfg_scale=float(kwargs["cfg_scale"]),
        )
        attention_scope.install()
        latent_scope = LatentCaptureScope(
            pipe=pipe,
            cfg_scale=float(kwargs["cfg_scale"]),
            step=custom.attention_step,
        )
        latent_scope.install()
        try:
            result = original_generate(*args, **kwargs)
        finally:
            latent_scope.restore()
            attention_scope.restore()
            restore_blocks()
        if latent_scope.latent is None:
            raise RuntimeError(
                f"did not capture positive CFG latent at step {custom.attention_step}"
            )
        group.finalize_case()
        _save_and_decode_latent(
            pipe=pipe,
            latent=latent_scope.latent,
            output_root=custom.latent_output_root,
            case_key=case_key,
            step=custom.attention_step,
        )
        return result

    base.core.generate_one_video = generate_with_capture
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
