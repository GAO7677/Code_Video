
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
from pathlib import Path

import torch

from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset
from code_vjepa_vggt.infer_context_video_wan import (
    _build_cond_context,
    _load_trainable_state_into_model,
    _resolve_launch_device,
    _run_sampling,
)
from code_vjepa_vggt.models.wan_context_model import WanContextVideoModel
from code_vjepa_vggt.utils.config import load_yaml_config


def _maybe_limit_indices(total: int, start_index: int, num_cases: int) -> range:
    return range(start_index, min(total, start_index + num_cases))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, help="checkpoint folder containing step_*.pt")
    parser.add_argument("--config", default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0613pybullet_wan_lora_gpu67.yaml")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--dataset-root", default="/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--output-dir", default="/data/gaoya/AAA_test_video/0529/vjepa_vggt/test")
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-fraction", type=float, default=0.5)
    parser.add_argument("--random-context-frames", action="store_true")
    parser.add_argument("--save-raw", action="store_true")
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

    trainer = WanContextVideoModel(
        ckpt_dir=str(config["model"]["wan_ckpt_dir"]),
        task=str(config["model"]["wan_task"]),
        device=device,
        load_dit=True,
        lora_rank=int(config["model"].get("wan_lora_rank", 0)),
        lora_alpha=int(config["model"].get("wan_lora_alpha", 0)),
        lora_dropout=float(config["model"].get("wan_lora_dropout", 0.0)),
        lora_init=str(config["model"].get("wan_lora_init", "gaussian")),
    )
    trainer.freeze_parts(
        freeze_vae=bool(config["model"]["freeze_vae"]),
        freeze_text_encoder=bool(config["model"]["freeze_text_encoder"]),
        freeze_dit=bool(config["model"]["freeze_wan_dit"]),
    )
    if trainer.dit is not None:
        trainer.dit.eval()
    state_info = _load_trainable_state_into_model(trainer, Path(args.checkpoint_dir))

    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for local_idx, dataset_idx in enumerate(_maybe_limit_indices(len(dataset), args.start_index, args.num_cases)):
        sample = dataset[dataset_idx]
        context_video = sample["context_video"].unsqueeze(0).to(device_obj)
        num_context_frames = torch.tensor([int(sample["num_context_frames"])], dtype=torch.long, device=device_obj)
        captions = [str(sample["caption"])]
        fused_context, context_latents, prep_debug = _build_cond_context(
            trainer=trainer,
            config=config,
            context_video=context_video,
            captions=captions,
            num_context_frames=num_context_frames,
            device_obj=device_obj,
        )
        pred_latent, sample_debug = _run_sampling(
            bundle=trainer.bundle,
            fused_context=fused_context,
            context_latents=context_latents,
            total_frames=int(sample["video"].shape[1]),
            num_context_frames=int(num_context_frames.item()),
            num_inference_steps=int(args.sampling_steps),
        )

        case_dir = output_dir / f"case_{dataset_idx:06d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "case_id": int(local_idx),
            "dataset_index": int(dataset_idx),
            "split": args.split,
            "caption": sample["caption"],
            "video_path": sample["video_path"],
            "context_frame_indices": sample["context_frame_indices"].tolist(),
            "frame_indices": sample["frame_indices"].tolist(),
            "prep_debug": prep_debug,
            "sample_debug": sample_debug,
            "load_state_info": state_info,
        }
        with open(case_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if args.save_raw:
            with torch.no_grad():
                decoded = trainer.bundle.vae.decode([pred_latent.to(next(trainer.bundle.vae.model.parameters()).device if hasattr(trainer.bundle.vae, "model") else device_obj)])
            if isinstance(decoded, list):
                decoded = decoded[0]
            video_out = decoded.detach().cpu().permute(1, 0, 2, 3).contiguous()
            video_out = ((video_out.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8).permute(0, 2, 3, 1).numpy()
            import cv2

            raw_path = case_dir / "prediction.mp4"
            writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), int(args.fps), (video_out.shape[2], video_out.shape[1]))
            try:
                for frame in video_out:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            result["prediction_video"] = str(raw_path)
            with open(case_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        results.append(result)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint_dir": str(args.checkpoint_dir),
                "dataset_root": str(args.dataset_root),
                "split": args.split,
                "num_cases": len(results),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps({"output_dir": str(output_dir), "num_cases": len(results)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
