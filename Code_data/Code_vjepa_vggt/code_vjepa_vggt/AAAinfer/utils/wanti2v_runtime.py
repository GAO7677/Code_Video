from __future__ import annotations

import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image


WAN_REPO_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main").resolve()
if str(WAN_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_REPO_ROOT))

import wan  # type: ignore  # noqa: E402
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS  # type: ignore  # noqa: E402
from wan.modules.model import WanModel  # type: ignore  # noqa: E402
from wan.utils.utils import save_video  # type: ignore  # noqa: E402


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


def build_run_manifest(args: WanTI2VArgs, json_paths: list[Path]) -> dict[str, Any]:
    return {
        "model_type": "wan_ti2v_5b",
        "input_list": str(args.input_list),
        "num_items": len(json_paths),
        "wan_root": str(args.wan_root),
        "size": str(args.size),
        "frame_num": int(args.frame_num),
        "fps": int(args.fps),
        "seed": int(args.seed),
        "sample_solver": str(args.sample_solver),
        "sampling_steps": int(args.sampling_steps),
        "sample_shift": float(args.sample_shift),
        "cfg_scale": float(args.cfg_scale),
        "input_field": "input_video",
        "image_field": "input_image",
        "single_process": True,
    }


def build_wan_ti2v_pipeline(args: WanTI2VArgs):
    cfg = WAN_CONFIGS["ti2v-5B"]
    original_from_pretrained = WanModel.from_pretrained

    def _patched_from_pretrained(pretrained_model_name_or_path, **kwargs):
        kwargs.setdefault("low_cpu_mem_usage", False)
        kwargs.setdefault("device_map", None)
        return original_from_pretrained(pretrained_model_name_or_path, **kwargs)

    WanModel.from_pretrained = _patched_from_pretrained
    pipe = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=str(args.wan_root),
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else 0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=bool(args.t5_cpu),
        convert_model_dtype=bool(args.convert_model_dtype),
    )
    WanModel.from_pretrained = original_from_pretrained
    return pipe


def write_text_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            if not line.endswith("\n"):
                handle.write("\n")


def run_single_case(
    *,
    pipe,
    args: WanTI2VArgs,
    input_json_path: Path,
    payload: dict[str, Any],
    firstframe_path: Path,
    output_video: Path,
) -> tuple[dict[str, Any], list[str]]:
    input_video = ensure_str_field(payload, "input_video", input_json_path)
    input_caption = ensure_str_field(payload, "input_caption", input_json_path)

    image = Image.open(firstframe_path).convert("RGB")

    logs = [
        f"[case] input_json={input_json_path}",
        f"[case] input_image={firstframe_path}",
        f"[case] input_caption={input_caption}",
        f"[case] wan_root={args.wan_root}",
    ]

    with torch.no_grad():
        video = pipe.generate(
            input_prompt=input_caption,
            img=image,
            size=SIZE_CONFIGS[str(args.size)],
            max_area=MAX_AREA_CONFIGS[str(args.size)],
            frame_num=int(args.frame_num),
            shift=float(args.sample_shift),
            sample_solver=str(args.sample_solver),
            sampling_steps=int(args.sampling_steps),
            guide_scale=float(args.cfg_scale),
            seed=int(args.seed),
            offload_model=bool(args.offload_model),
        )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(output_video), fps=int(args.fps))

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
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_default_frame_num(frame_num: int | None) -> int:
    if frame_num is not None:
        return int(frame_num)
    return 24


def resolve_default_sample_shift(sample_shift: float | None) -> float:
    if sample_shift is not None:
        return float(sample_shift)
    return float(WAN_CONFIGS["ti2v-5B"].sample_shift)


def resolve_default_sampling_steps(sampling_steps: int | None) -> int:
    if sampling_steps is not None:
        return int(sampling_steps)
    return int(WAN_CONFIGS["ti2v-5B"].sample_steps)


def resolve_default_cfg_scale(cfg_scale: float | None) -> float:
    if cfg_scale is not None:
        return float(cfg_scale)
    return float(WAN_CONFIGS["ti2v-5B"].sample_guide_scale)


def ensure_cuda_env() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("WanTI2V batch inference requires CUDA, but no GPU is visible.")
    torch.cuda.set_device(0)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
