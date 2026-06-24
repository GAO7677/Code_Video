from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset
from code_vjepa_vggt.infer_v_newtrain_context_video_wan import (
    _build_object_context,
    _load_v_newtrain_state_into_model,
    _resolve_checkpoint_file,
    _resolve_launch_device,
    _tensor_video_to_pil_list,
    build_model,
)
from code_vjepa_vggt.utils.config import load_yaml_config


def _resolve_case_index(dataset: PhysStateEpisodeDataset, case_name: str | None, case_index: int | None) -> int:
    if case_name is not None:
        target = str(case_name).strip()
        if not target:
            raise ValueError("--case-name cannot be empty")
        for idx, meta_path in enumerate(dataset.samples):
            if meta_path.stem == target:
                return idx
        raise ValueError(f"case_name={target!r} not found under split={dataset.split}")
    if case_index is None:
        raise ValueError("one of --case-index or --case-name is required")
    if not (0 <= int(case_index) < len(dataset)):
        raise IndexError(f"case_index={case_index} out of range [0, {len(dataset) - 1}]")
    return int(case_index)


def _sample_stem(sample: dict[str, object]) -> str:
    video_path = sample.get("video_path")
    if isinstance(video_path, str) and video_path:
        return Path(video_path).stem
    return "case"


def _tensor_video_to_np_uint8(video_bcthw: torch.Tensor) -> np.ndarray:
    frames = video_bcthw.detach().cpu().permute(1, 2, 3, 0)
    return ((frames + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run v_newtrain inference on a single dataset case by case index or sample stem."
    )
    parser.add_argument("--checkpoint", required=True, help="checkpoint .pt/.safetensors file or step-xxxx directory")
    parser.add_argument(
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_freeze_lora_other_modules_gpu67.yaml",
    )
    parser.add_argument(
        "--dataset-root",
        default="/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--case-index", type=int, default=None)
    parser.add_argument("--case-name", default=None, help="sample stem such as sample_000339_w000")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--prompt", default=None, help="override dataset prompt if provided")
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--export-context-video", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    height, width = tuple(int(v) for v in data_cfg.get("resolution", [512, 896]))
    num_frames = int(args.num_frames if args.num_frames is not None else data_cfg.get("num_frames", 24))
    fps = int(args.fps if args.fps is not None else data_cfg.get("fps", 30))
    context_frames = int(data_cfg.get("num_context_frames", data_cfg.get("fixed_num_context_frames", 8)))

    dataset = PhysStateEpisodeDataset(
        root=args.dataset_root,
        split=str(args.split),
        resolution=(height, width),
        num_context_frames=context_frames,
        context_fraction=0.5,
        random_context_frames=False,
        seed=42,
    )
    dataset_idx = _resolve_case_index(dataset, args.case_name, args.case_index)
    sample = dataset[dataset_idx]
    sample_stem = _sample_stem(sample)

    prompt = str(args.prompt) if args.prompt is not None else str(sample["caption"])
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = (
        Path(args.output_video).expanduser().resolve()
        if args.output_video is not None
        else output_dir / f"{sample_stem}.mp4"
    )
    output_video.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_launch_device()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    class _Args:
        pass

    model_args = _Args()
    model_args.device = device
    model_args.wan_root = str(model_cfg.get("wan_ckpt_dir", "/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"))
    model_args.lora_rank = int(model_cfg.get("wan_lora_rank", train_cfg.get("lora_rank", 32)))
    model_args.context_frames = context_frames
    model_args.disable_object_branch = False
    model_args.object_num_queries = int(model_cfg.get("object_num_queries", 8))
    model_args.aux_max_objects = int(model_cfg.get("aux_max_objects", model_cfg.get("sam2_max_objects", 4)))
    model_args.jepa_ckpt_path = str(model_cfg.get("jepa_ckpt_path", "/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth"))
    model_args.jepa_input_size = int(model_cfg.get("jepa_input_size", 384))
    model_args.jepa_patch_size = int(model_cfg.get("jepa_patch_size", 16))
    model_args.jepa_tubelet_size = int(model_cfg.get("jepa_tubelet_size", 2))
    model_args.cotracker_checkpoint = str(model_cfg.get("cotracker_checkpoint", "/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth"))
    cotracker_input_hw = model_cfg.get("cotracker_input_hw", [384, 512])
    model_args.cotracker_input_h = int(cotracker_input_hw[0])
    model_args.cotracker_input_w = int(cotracker_input_hw[1])
    model_args.cotracker_window_len = int(model_cfg.get("cotracker_window_len", 60))
    model_args.object_pooler_latent_dim = int(model_cfg.get("object_pooler_latent_dim", 16))
    model_args.cond_proj_dim = int(model_cfg.get("cond_proj_dim", 4096))
    model_args.jepa_window_radius = int(model_cfg.get("jepa_window_radius", 1))
    model_args.latent_window_radius = int(model_cfg.get("latent_window_radius", 1))

    model = build_model(model_args)
    model.to(torch.device(device))
    model.eval()
    load_info = _load_v_newtrain_state_into_model(model, Path(args.checkpoint))
    pipe = model.pipe
    pipe.dit.eval()

    context_video_single = sample["context_video"]
    if not isinstance(context_video_single, torch.Tensor):
        raise TypeError(f"sample['context_video'] must be a tensor, got {type(context_video_single)}")
    context_video_single = context_video_single[:, :context_frames].contiguous()
    if int(context_video_single.shape[1]) != int(context_frames):
        raise RuntimeError(
            f"case {sample_stem} only has {int(context_video_single.shape[1])} context frames, expected {context_frames}"
        )

    context_pil = _tensor_video_to_pil_list(context_video_single)
    object_context, object_debug = _build_object_context(
        model=model,
        context_video_single=context_video_single,
    )

    from diffsynth.utils.data import save_video

    with torch.no_grad():
        video = pipe(
            prompt=prompt,
            negative_prompt="",
            context_video=context_pil,
            seed=int(args.seed),
            tiled=True,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=int(args.sampling_steps),
            cfg_scale=float(args.cfg_scale),
            object_context=object_context,
        )

    save_video(video, str(output_video), fps=fps, quality=int(args.quality))

    result = {
        "checkpoint": str(_resolve_checkpoint_file(Path(args.checkpoint))),
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "split": str(args.split),
        "case_index": int(dataset_idx),
        "case_name": sample_stem,
        "prompt": prompt,
        "output_video": str(output_video),
        "source_npz": str(sample["video_path"]),
        "context_frame_indices": sample["context_frame_indices"].tolist()
        if isinstance(sample.get("context_frame_indices"), torch.Tensor)
        else None,
        "load_info": load_info,
        "object_debug": object_debug,
    }

    if args.export_context_video:
        context_video_path = output_dir / f"{sample_stem}_context.mp4"
        save_video(
            list(_tensor_video_to_np_uint8(context_video_single)),
            str(context_video_path),
            fps=fps,
            quality=int(args.quality),
        )
        result["context_video"] = str(context_video_path)

    with (output_dir / f"{sample_stem}.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
