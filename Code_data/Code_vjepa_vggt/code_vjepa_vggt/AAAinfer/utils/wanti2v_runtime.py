from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from diffusers import WanImageToVideoPipeline
from PIL import Image


DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards"
)


@dataclass
class WanTI2VArgs:
    input_list: Path
    output_root: Path
    wan_root: Path
    size: str
    frame_num: int
    fps: int
    seed: int
    sample_solver: str
    sampling_steps: int
    sample_shift: float
    cfg_scale: float
    offload_model: bool
    t5_cpu: bool
    convert_model_dtype: bool
    force: bool


def read_list_file(list_path: Path) -> list[Path]:
    items: list[Path] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            items.append(Path(line).expanduser().resolve())
    return items


def load_json(json_path: Path) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"input json must be an object: {json_path}")
    return payload


def write_json(json_path: Path, payload: dict[str, Any]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def ensure_str_field(payload: dict[str, Any], key: str, json_path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or empty {key!r} in {json_path}")
    return value.strip()


def derive_firstframe_png_path(source_video: Path) -> Path:
    parent = source_video.parent
    stem = source_video.stem
    return parent / f"{stem}_firstframe.png"


def extract_first_frame_png(input_video: Path, output_png: Path) -> Path:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    frame = iio.imread(input_video, index=0)
    if frame.ndim != 3:
        raise ValueError(f"unexpected first-frame shape from {input_video}: {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    Image.fromarray(frame).save(output_png)
    return output_png


def ensure_firstframe_image(json_path: Path, payload: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    source_video = Path(ensure_str_field(payload, "source_video", json_path)).expanduser().resolve()
    input_video = Path(ensure_str_field(payload, "input_video", json_path)).expanduser().resolve()

    firstframe_key = "input_image"
    existing = payload.get(firstframe_key)
    if isinstance(existing, str) and existing.strip():
        firstframe_path = Path(existing).expanduser().resolve()
    else:
        firstframe_path = derive_firstframe_png_path(source_video)

    if not firstframe_path.exists():
        extract_first_frame_png(input_video, firstframe_path)

    payload[firstframe_key] = str(firstframe_path)
    write_json(json_path, payload)
    return payload, firstframe_path


def _parse_size(size: str) -> tuple[int, int]:
    try:
        height_str, width_str = size.split("*", maxsplit=1)
        height = int(height_str)
        width = int(width_str)
    except Exception as exc:
        raise ValueError(f"invalid size string: {size}") from exc
    return height, width


def build_run_manifest(args: WanTI2VArgs, json_paths: list[Path]) -> dict[str, Any]:
    height, width = _parse_size(args.size)
    return {
        "model_type": "wan_ti2v_5b_diffusers",
        "input_list": str(args.input_list),
        "num_items": len(json_paths),
        "wan_root": str(args.wan_root),
        "height": int(height),
        "width": int(width),
        "frame_num": int(args.frame_num),
        "fps": int(args.fps),
        "seed": int(args.seed),
        "sampling_steps": int(args.sampling_steps),
        "cfg_scale": float(args.cfg_scale),
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "input_field": "input_video",
        "image_field": "input_image",
        "single_process": True,
        "backend": "diffusers.WanImageToVideoPipeline",
    }


class OfficialWanTI2VWrapper:
    def __init__(self, pipe: WanImageToVideoPipeline):
        self.pipe = pipe

    def __call__(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        seed: int,
        input_image: Image.Image,
        height: int,
        width: int,
        num_frames: int,
        cfg_scale: float,
        num_inference_steps: int,
    ) -> np.ndarray:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(int(seed))
        output = self.pipe(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(cfg_scale),
            generator=generator,
            output_type="np",
            return_dict=True,
        )
        frames = output.frames
        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu().numpy()
        frames = np.asarray(frames)
        if frames.ndim != 5:
            raise ValueError(f"unexpected official Wan output shape: {frames.shape}")
        return frames[0]


def build_wan_ti2v_pipeline(args: WanTI2VArgs):
    pipe = WanImageToVideoPipeline.from_pretrained(
        str(args.wan_root),
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    if hasattr(pipe, "enable_model_cpu_offload") and args.offload_model:
        pipe.enable_model_cpu_offload()
    return OfficialWanTI2VWrapper(pipe)


def save_video_np(frames: np.ndarray, save_path: Path, fps: int) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"expected video array [T,H,W,C], got {frames.shape}")
    if frames.dtype != np.uint8:
        if frames.max() <= 1.0:
            frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
        else:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
    iio.imwrite(save_path, frames, fps=int(fps), codec="libx264")


def run_single_case(
    *,
    pipe,
    args: WanTI2VArgs,
    input_json_path: Path,
    payload: dict[str, Any],
    firstframe_path: Path,
    output_video: Path,
) -> tuple[dict[str, Any], list[str]]:
    input_caption = ensure_str_field(payload, "input_caption", input_json_path)
    height, width = _parse_size(args.size)

    image = Image.open(firstframe_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)

    logs = [
        f"[case] input_json={input_json_path}",
        f"[case] input_image={firstframe_path}",
        f"[case] input_caption={input_caption}",
        f"[case] wan_root={args.wan_root}",
    ]

    with torch.no_grad():
        video = pipe(
            prompt=input_caption,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            seed=int(args.seed),
            input_image=image,
            height=int(height),
            width=int(width),
            num_frames=int(args.frame_num),
            cfg_scale=float(args.cfg_scale),
            num_inference_steps=int(args.sampling_steps),
        )

    save_video_np(video, output_video, fps=int(args.fps))

    result = {
        "input_json": str(input_json_path),
        "input_image": str(firstframe_path),
        "input_caption": str(input_caption),
        "output_video": str(output_video),
        "seed": int(args.seed),
        "step": int(args.sampling_steps),
        "guidance": float(args.cfg_scale),
        "ckpt": str(args.wan_root),
    }
    return result, logs


def cleanup_pipeline(pipe) -> None:
    if hasattr(pipe, "pipe"):
        del pipe.pipe
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_default_frame_num(frame_num: int | None) -> int:
    if frame_num is not None:
        return int(frame_num)
    return 25


def resolve_default_sample_shift(sample_shift: float | None) -> float:
    if sample_shift is not None:
        return float(sample_shift)
    return 5.0


def resolve_default_sampling_steps(sampling_steps: int | None) -> int:
    if sampling_steps is not None:
        return int(sampling_steps)
    return 40


def resolve_default_cfg_scale(cfg_scale: float | None) -> float:
    if cfg_scale is not None:
        return float(cfg_scale)
    return 5.0


def ensure_cuda_env() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("WanTI2V batch inference requires CUDA, but no GPU is visible.")
    torch.cuda.set_device(0)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
