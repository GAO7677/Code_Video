from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from decord import VideoReader, cpu

from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackAdapter
from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, read_video_prefix, read_video_uniform


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _resolve_device(device_arg: str) -> str:
    if device_arg and device_arg != "auto":
        if device_arg.startswith("cuda") and torch.cuda.is_available():
            index = 0
            if ":" in device_arg:
                index = int(device_arg.split(":", 1)[1])
            torch.cuda.set_device(index)
        return device_arg
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


def _read_all_frames(video_path: Path) -> tuple[np.ndarray, np.ndarray]:
    vr = VideoReader(str(video_path), ctx=cpu(0))
    frame_indices = np.arange(len(vr), dtype=np.int64)
    frames = vr.get_batch(frame_indices).asnumpy()
    return frames, frame_indices


def _load_video_frames(
    video_path: Path,
    *,
    num_frames: int | None,
    sampling_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if num_frames is None:
        return _read_all_frames(video_path)
    if sampling_mode == "prefix":
        return read_video_prefix(video_path, int(num_frames))
    return read_video_uniform(video_path, int(num_frames))


def _iter_input_videos(args: argparse.Namespace) -> list[Path]:
    videos: list[Path] = []
    if args.input_video is not None:
        videos.append(Path(args.input_video).expanduser().resolve())
    if args.input_list is not None:
        list_path = Path(args.input_list).expanduser().resolve()
        with list_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                videos.append(Path(line).expanduser().resolve())
    if args.input_dir is not None:
        input_dir = Path(args.input_dir).expanduser().resolve()
        for ext in sorted(VIDEO_EXTS):
            videos.extend(sorted(input_dir.rglob(f"*{ext}")))
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in videos:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    if not deduped:
        raise ValueError("please provide one of --input-video, --input-list, or --input-dir")
    return deduped


def _resolve_model_config(args: argparse.Namespace) -> None:
    if not args.config:
        return
    config = load_yaml_config(args.config)
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}

    if "vggt_model_path" in model_cfg and args.vggt_model_path == args._parser_defaults["vggt_model_path"]:
        args.vggt_model_path = str(model_cfg["vggt_model_path"])
    if "vggt_input_hw" in model_cfg and args.vggt_input_h == args._parser_defaults["vggt_input_h"]:
        vggt_input_hw = model_cfg["vggt_input_hw"]
        if isinstance(vggt_input_hw, (list, tuple)) and len(vggt_input_hw) == 2:
            args.vggt_input_h = int(vggt_input_hw[0])
            args.vggt_input_w = int(vggt_input_hw[1])
    if "vggt_input_h" in model_cfg and args.vggt_input_h == args._parser_defaults["vggt_input_h"]:
        args.vggt_input_h = int(model_cfg["vggt_input_h"])
    if "vggt_input_w" in model_cfg and args.vggt_input_w == args._parser_defaults["vggt_input_w"]:
        args.vggt_input_w = int(model_cfg["vggt_input_w"])
    if args.num_frames is None:
        if "num_context_frames" in data_cfg:
            args.num_frames = int(data_cfg["num_context_frames"])
        elif "fixed_num_context_frames" in data_cfg:
            args.num_frames = int(data_cfg["fixed_num_context_frames"])


def _prepare_vggt_input(frames_thwc_uint8: np.ndarray, input_hw: tuple[int, int]) -> torch.Tensor:
    video_cthw_01 = preprocess_video_rgb_uint8(frames_thwc_uint8, input_hw, value_range="zero_to_one")
    return video_cthw_01.permute(1, 0, 2, 3).contiguous().unsqueeze(0)


def _cache_one_video(
    adapter: VGGTTrackAdapter,
    video_path: Path,
    output_path: Path,
    *,
    num_frames: int | None,
    sampling_mode: str,
) -> dict[str, object]:
    frames, frame_indices = _load_video_frames(video_path, num_frames=num_frames, sampling_mode=sampling_mode)
    if frames.ndim != 4 or int(frames.shape[-1]) != 3:
        raise ValueError(f"expected video frames in [T,H,W,3], got {list(frames.shape)} from {video_path}")

    frames_bthwc_01 = _prepare_vggt_input(frames, adapter.input_hw)
    device = next(adapter.model.parameters()).device if adapter.model is not None else torch.device("cpu")
    dtype = next(adapter.model.parameters()).dtype if adapter.model is not None else torch.float32
    frames_bthwc_01 = frames_bthwc_01.to(device=device, dtype=dtype)

    with torch.no_grad():
        aggregated_tokens_list, patch_start_idx = adapter.model.shortcut_forward(frames_bthwc_01)  # type: ignore[union-attr]
        dense_patch_tokens, patch_grid_hw = adapter._dense_patch_tokens_from_aggregated(
            aggregated_tokens_list,
            patch_start_idx,
            batch_size=1,
            frames=int(frames_bthwc_01.shape[1]),
        )

    dense_patch_tokens_cpu = dense_patch_tokens.squeeze(0).detach().cpu().to(torch.float16).contiguous()
    payload = {
        "source_video": str(video_path),
        "output_file": str(output_path),
        "frame_indices": frame_indices.tolist(),
        "num_frames": int(frames_bthwc_01.shape[1]),
        "input_hw": [int(adapter.input_hw[0]), int(adapter.input_hw[1])],
        "patch_size": int(adapter.patch_size),
        "patch_grid_hw": [int(patch_grid_hw[0]), int(patch_grid_hw[1])],
        "dense_patch_tokens": dense_patch_tokens_cpu,
        "dense_patch_tokens_shape": list(dense_patch_tokens_cpu.shape),
        "aggregated_last_shape": list(aggregated_tokens_list[-1].shape),
        "patch_start_idx": int(patch_start_idx),
        "dtype": "torch.float16",
        "model_path": str(adapter.model_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return payload


def _cache_one_dataset_sample(
    adapter: VGGTTrackAdapter,
    sample: dict[str, object],
    output_path: Path,
) -> dict[str, object]:
    context_video = sample.get("context_video")
    if not isinstance(context_video, torch.Tensor):
        raise TypeError(f"sample['context_video'] must be a tensor, got {type(context_video)}")
    if context_video.ndim != 4:
        raise ValueError(f"context_video must have shape [C,T,H,W], got {list(context_video.shape)}")
    context_video_thwc = context_video.permute(1, 2, 3, 0).contiguous().cpu().numpy()
    frames_bthwc_01 = _prepare_vggt_input(context_video_thwc, adapter.input_hw).to(
        device=next(adapter.model.parameters()).device,  # type: ignore[union-attr]
        dtype=next(adapter.model.parameters()).dtype,  # type: ignore[union-attr]
    )
    with torch.no_grad():
        aggregated_tokens_list, patch_start_idx = adapter.model.shortcut_forward(frames_bthwc_01)  # type: ignore[union-attr]
        dense_patch_tokens, patch_grid_hw = adapter._dense_patch_tokens_from_aggregated(
            aggregated_tokens_list,
            patch_start_idx,
            batch_size=1,
            frames=int(frames_bthwc_01.shape[1]),
        )
    source_video = str(sample.get("video_path", ""))
    dense_patch_tokens_cpu = dense_patch_tokens.squeeze(0).detach().cpu().to(torch.float16).contiguous()
    payload = {
        "source_video": source_video,
        "output_file": str(output_path),
        "frame_indices": sample.get("context_frame_indices", torch.arange(frames_bthwc_01.shape[1])).tolist()
        if isinstance(sample.get("context_frame_indices"), torch.Tensor)
        else list(range(int(frames_bthwc_01.shape[1]))),
        "num_frames": int(frames_bthwc_01.shape[1]),
        "input_hw": [int(adapter.input_hw[0]), int(adapter.input_hw[1])],
        "patch_size": int(adapter.patch_size),
        "patch_grid_hw": [int(patch_grid_hw[0]), int(patch_grid_hw[1])],
        "dense_patch_tokens": dense_patch_tokens_cpu,
        "dense_patch_tokens_shape": list(dense_patch_tokens_cpu.shape),
        "aggregated_last_shape": list(aggregated_tokens_list[-1].shape),
        "patch_start_idx": int(patch_start_idx),
        "dtype": "torch.float16",
        "model_path": str(adapter.model_path),
        "sample_name": Path(source_video).stem if source_video else None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return payload


def _json_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache VGGT dense patch features from videos.")
    parser.add_argument("--config", default=None, help="optional yaml config for default model/input settings")
    parser.add_argument("--vggt-model-path", default="/data/gaoya/ckpt/facebook-VGGT-1B")
    parser.add_argument("--vggt-input-h", type=int, default=420)
    parser.add_argument("--vggt-input-w", type=int, default=728)
    parser.add_argument("--device", default="auto", help="cuda:0 / cpu / auto")
    parser.add_argument("--dataset-root", default=None, help="phys_state dataset root")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--dataset-start", type=int, default=0)
    parser.add_argument("--dataset-end", type=int, default=None)
    parser.add_argument("--input-video", default=None, help="single video path")
    parser.add_argument("--input-list", default=None, help="text file with one video path per line")
    parser.add_argument("--input-dir", default=None, help="directory to scan for video files")
    parser.add_argument("--output-dir", required=True, help="directory for cached .pt files")
    parser.add_argument("--num-frames", type=int, default=None, help="optional sampling length; default uses all frames")
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cudnn.enabled = False
    cudnn.benchmark = False
    cudnn.deterministic = False

    args._parser_defaults = {  # type: ignore[attr-defined]
        "vggt_model_path": parser.get_default("vggt_model_path"),
        "vggt_input_h": parser.get_default("vggt_input_h"),
        "vggt_input_w": parser.get_default("vggt_input_w"),
    }
    _resolve_model_config(args)

    device = _resolve_device(str(args.device))
    if args.vggt_model_path is None or not str(args.vggt_model_path).strip():
        raise ValueError("--vggt-model-path is required")

    adapter = VGGTTrackAdapter(
        model_path=str(args.vggt_model_path),
        num_queries=8,
        device=device,
        input_hw=(int(args.vggt_input_h), int(args.vggt_input_w)),
        trainable=False,
    )
    if adapter.model is None:
        raise RuntimeError(f"failed to load VGGT model from {args.vggt_model_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    results: list[dict[str, object]] = []
    if args.dataset_root is not None:
        dataset = PhysStateEpisodeDataset(
            root=args.dataset_root,
            split=str(args.dataset_split),
            resolution=(512, 896),
            num_context_frames=int(args.num_frames) if args.num_frames is not None else 8,
            context_fraction=0.5,
            random_context_frames=False,
            seed=42,
        )
        start_idx = max(0, int(args.dataset_start))
        end_idx = len(dataset) if args.dataset_end is None else min(len(dataset), int(args.dataset_end))
        if start_idx >= end_idx:
            raise ValueError(f"invalid dataset range [{start_idx}, {end_idx}) for dataset size {len(dataset)}")
        limit = end_idx if args.dataset_limit is None else min(end_idx, start_idx + int(args.dataset_limit))
        for idx in range(start_idx, limit):
            sample = dataset[idx]
            video_path = str(sample.get("video_path", f"sample_{idx:06d}"))
            stem = Path(video_path).stem
            output_path = output_dir / f"{stem}.vggt.pt"
            if output_path.exists() and not args.overwrite:
                results.append({"source_video": video_path, "output_file": str(output_path), "skipped": True})
                continue
            payload = _cache_one_dataset_sample(adapter, sample, output_path)
            results.append(payload)
            print(
                json.dumps(
                    {
                        "source_video": payload["source_video"],
                        "output_file": payload["output_file"],
                        "num_frames": payload["num_frames"],
                        "patch_grid_hw": payload["patch_grid_hw"],
                        "dense_patch_tokens_shape": payload["dense_patch_tokens_shape"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    else:
        video_paths = _iter_input_videos(args)
        for video_path in video_paths:
            stem = video_path.stem
            output_path = output_dir / f"{stem}.vggt.pt"
            if output_path.exists() and not args.overwrite:
                results.append(
                    {
                        "source_video": str(video_path),
                        "output_file": str(output_path),
                        "skipped": True,
                    }
                )
                continue
            payload = _cache_one_video(
                adapter,
                video_path,
                output_path,
                num_frames=args.num_frames,
                sampling_mode=args.sampling_mode,
            )
            results.append(payload)
            print(
                json.dumps(
                    {
                        "source_video": payload["source_video"],
                        "output_file": payload["output_file"],
                        "num_frames": payload["num_frames"],
                        "patch_grid_hw": payload["patch_grid_hw"],
                        "dense_patch_tokens_shape": payload["dense_patch_tokens_shape"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(_json_safe(item), ensure_ascii=False) + "\n")

    summary = {
        "device": device,
        "num_items": len(results),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
