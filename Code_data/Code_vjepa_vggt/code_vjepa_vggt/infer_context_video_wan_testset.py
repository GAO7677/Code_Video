
'''

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan_testset.py \
  --checkpoint-dir /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/pybullet0613_wan_lora_gpu67 \
  --split test \
  --dataset-root /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500 \
  --output-dir /data/gaoya/AAA_test_video/0529/vjepa_vggt/test \
  --num-cases 4 \
  --save-raw

'''
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset
from code_vjepa_vggt.infer_context_video_wan import (
    _build_cond_context,
    _load_trainable_state_into_model,
    _resolve_launch_device,
    _run_sampling,
)
from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.infer_context_video_wan import _infer_object_pooler_latent_dim, _load_trainable_state


def _maybe_limit_indices(total: int, start_index: int, num_cases: int) -> range:
    return range(start_index, min(total, start_index + num_cases))


def _video_bcthw_to_uint8_thwc(video_bcthw: torch.Tensor) -> np.ndarray:
    video = video_bcthw.detach().cpu().clamp(-1.0, 1.0)
    video = ((video + 1.0) * 127.5).to(torch.uint8)
    if video.ndim == 5 and video.shape[0] == 1:
        video = video[0]
    return video.permute(1, 2, 3, 0).contiguous().numpy()


def _write_mp4(path: Path, frames_thwc_uint8: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames_thwc_uint8.shape[1]), int(frames_thwc_uint8.shape[2])
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        tmp_path = path.with_suffix(".tmp.mp4")
        writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"failed to open writer for {tmp_path}")
        try:
            for frame in frames_thwc_uint8:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        import subprocess

        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(tmp_path),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tmp_path.unlink(missing_ok=True)
        return

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer for {path}")
    try:
        for frame in frames_thwc_uint8:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _resolve_input_videos(
    sample: dict[str, object],
    *,
    default_num_context_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    video = sample["video"]
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"sample['video'] must be a tensor, got {type(video)}")

    context_video = sample.get("context_video")
    if isinstance(context_video, torch.Tensor) and context_video.numel() > 0:
        context_video = context_video.contiguous()
        context_indices = sample.get("context_frame_indices")
        if isinstance(context_indices, torch.Tensor):
            context_indices = context_indices.long()
        else:
            context_indices = torch.arange(context_video.shape[1], dtype=torch.long)
        return video.contiguous(), context_video, context_indices

    total_frames = int(video.shape[1])
    context_len = min(default_num_context_frames, total_frames)
    context_indices = torch.arange(context_len, dtype=torch.long)
    context_video = video[:, context_indices].contiguous()
    return video.contiguous(), context_video, context_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, help="checkpoint folder containing step_*.pt")
    parser.add_argument("--config", default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0613pybullet_wan_lora_gpu67.yaml")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--dataset-root", default="/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--output-dir", default="/data/gaoya/AAA_test_video/0529/vjepa_vggt/test")
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-fraction", type=float, default=0.5)
    parser.add_argument("--random-context-frames", action="store_true")
    parser.add_argument("--save-raw", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    device = _resolve_launch_device()
    device_obj = torch.device(device)

    dataset = PhysStateEpisodeDataset(
        root=args.dataset_root,
        split=args.split,
        resolution=tuple(config["data"]["resolution"]),
        num_context_frames=int(config["data"]["num_context_frames"]),
        context_fraction=float(args.context_fraction),
        random_context_frames=bool(args.random_context_frames),
        seed=int(args.seed),
    )

    if args.num_frames <= 0:
        raise ValueError(f"--num-frames must be positive, got {args.num_frames}")

    checkpoint_state = _load_trainable_state(Path(args.checkpoint_dir))
    object_pooler_latent_dim = _infer_object_pooler_latent_dim(
        checkpoint_state,
        int(config["model"].get("object_pooler_latent_dim", 16)),
    )
    config["model"]["object_pooler_latent_dim"] = int(object_pooler_latent_dim)

    trainer = ContextVideoTrainer(config, build_optimizer=True, device=device)
    trainer.build_optimizer = False
    state_info = _load_trainable_state_into_model(trainer, Path(args.checkpoint_dir))
    if trainer.bundle.dit is not None:
        trainer.bundle.dit.eval()
    if trainer.bundle.vae is not None:
        trainer.bundle.vae.model.eval()
    if trainer.bundle.text_encoder is not None:
        trainer.bundle.text_encoder.model.eval()

    checkpoint_name = Path(args.checkpoint_dir).name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for local_idx, dataset_idx in enumerate(_maybe_limit_indices(len(dataset), args.start_index, args.num_cases)):
        sample = dataset[dataset_idx]
        sample_stem = Path(str(sample["video_path"])).stem
        sample_video = sample["video"]
        if not isinstance(sample_video, torch.Tensor):
            raise TypeError(f"sample['video'] must be a tensor, got {type(sample_video)}")
        total_frames = min(int(args.num_frames), int(sample_video.shape[1]))
        source_video = sample_video[:, :total_frames].contiguous()
        context_len = min(int(config["data"]["num_context_frames"]), int(source_video.shape[1]))
        context_indices = torch.arange(context_len, dtype=torch.long)
        input_context_video = source_video[:, context_indices].contiguous()
        num_context_frames = torch.tensor([int(context_indices.numel())], dtype=torch.long, device=device_obj)
        captions = [str(sample["caption"])]

        source_video_path = output_dir / f"{sample_stem}_source_video.mp4"
        input_video_path = output_dir / f"{sample_stem}_input_video.mp4"
        _write_mp4(source_video_path, _video_bcthw_to_uint8_thwc(source_video.unsqueeze(0)), fps=int(args.fps))
        _write_mp4(input_video_path, _video_bcthw_to_uint8_thwc(input_context_video.unsqueeze(0)), fps=int(args.fps))

        fused_context, context_latents, prep_debug = _build_cond_context(
            trainer=trainer,
            config=config,
            context_video=input_context_video.unsqueeze(0).to(device_obj),
            captions=captions,
            num_context_frames=num_context_frames,
            device_obj=device_obj,
        )
        pred_latent, sample_debug = _run_sampling(
            bundle=trainer.bundle,
            fused_context=fused_context,
            context_latents=context_latents,
            total_frames=int(source_video.shape[1]),
            num_context_frames=int(num_context_frames.item()),
            num_inference_steps=int(args.sampling_steps),
        )

        output_video_path = output_dir / f"{sample_stem}.mp4"
        if args.save_raw:
            with torch.no_grad():
                decoded = trainer.bundle.vae.decode([pred_latent.to(next(trainer.bundle.vae.model.parameters()).device if hasattr(trainer.bundle.vae, "model") else device_obj)])
            if isinstance(decoded, list):
                decoded = decoded[0]
            _write_mp4(output_video_path, _video_bcthw_to_uint8_thwc(decoded), fps=int(args.fps))

        result = {
            "checkpoint_dir": str(args.checkpoint_dir),
            "seed": int(args.seed),
            "input_caption": str(sample["caption"]),
            "source_video": str(source_video_path),
            "input_video": str(input_video_path),
            "output_video": str(output_video_path),
            "sample_debug": sample_debug,
        }
        with open(output_dir / f"{sample_stem}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps({"output_dir": str(output_dir), "checkpoint_dir": str(args.checkpoint_dir), "seed": int(args.seed)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
