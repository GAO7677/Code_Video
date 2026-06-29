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


DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards"
)
DEFAULT_OFFICIAL_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
OFFICIAL_WAN_REPO = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main")
LEGACY_WAN_REPO = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")


@dataclass
class WanTI2VArgs:
    input_list: Path
    output_root: Path
    model_name: str
    wan_root: Path
    backend: str
    size: str
    frame_num: int
    fps: int
    seed: int
    sample_solver: str
    sampling_steps: int
    sample_shift: float
    cfg_scale: float
    negative_prompt: str
    offload_model: bool
    t5_cpu: bool
    convert_model_dtype: bool
    force: bool


def _ensure_official_wan_imports():
    repo_root = OFFICIAL_WAN_REPO.resolve()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    import wan  # type: ignore
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS  # type: ignore

    return wan, WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS


def _ensure_legacy_wan_imports():
    repo_root = LEGACY_WAN_REPO.resolve()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from rerank_video.pdi_proxy_eval import WanTI2VRunner  # type: ignore

    return WanTI2VRunner


def patch_wanmodel_from_pretrained_defaults() -> None:
    _ensure_official_wan_imports()
    from wan.modules.model import WanModel  # type: ignore

    if getattr(WanModel, "_codex_low_cpu_patch", False):
        return

    original = WanModel.from_pretrained.__func__

    def patched(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs.setdefault("low_cpu_mem_usage", False)
        return original(cls, pretrained_model_name_or_path, *model_args, **kwargs)

    WanModel.from_pretrained = classmethod(patched)
    WanModel._codex_low_cpu_patch = True


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


def derive_method_name(output_video: Path) -> str:
    return output_video.parent.name


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


def normalize_sample_solver(sample_solver: str) -> str:
    solver = str(sample_solver).strip().lower()
    if solver in {"official_diffusers", "unipc", "official", "default"}:
        return "unipc"
    if solver in {"dpm++", "dpmpp", "dpm-solver", "dpm_solver"}:
        return "dpm++"
    return solver


def resolve_official_wan_root(wan_root: Path) -> Path:
    candidate = wan_root.expanduser()
    if not candidate.is_absolute():
        candidate = candidate.resolve()
    else:
        candidate = candidate.resolve()

    if candidate.is_dir() and (candidate / "Wan2.2_VAE.pth").exists():
        return candidate

    if candidate.name.endswith("-Diffusers"):
        sibling = candidate.with_name(candidate.name.removesuffix("-Diffusers"))
        if sibling.is_dir() and (sibling / "Wan2.2_VAE.pth").exists():
            return sibling.resolve()

    if DEFAULT_OFFICIAL_WAN_ROOT.is_dir() and (DEFAULT_OFFICIAL_WAN_ROOT / "Wan2.2_VAE.pth").exists():
        return DEFAULT_OFFICIAL_WAN_ROOT.resolve()

    raise FileNotFoundError(
        "unable to resolve official Wan2.2 TI2V checkpoint directory from "
        f"{wan_root}; expected a directory containing Wan2.2_VAE.pth"
    )


def build_run_manifest(args: WanTI2VArgs, json_paths: list[Path]) -> dict[str, Any]:
    height, width = _parse_size(args.size)
    resolved_wan_root = resolve_official_wan_root(args.wan_root)
    return {
        "model_name": str(args.model_name),
        "model_type": f"wan_ti2v_5b_{args.backend}",
        "input_list": str(args.input_list),
        "num_items": len(json_paths),
        "wan_root": str(args.wan_root),
        "resolved_wan_root": str(resolved_wan_root),
        "backend": str(args.backend),
        "height": int(height),
        "width": int(width),
        "frame_num": int(args.frame_num),
        "fps": int(args.fps),
        "seed": int(args.seed),
        "sampling_steps": int(args.sampling_steps),
        "sample_shift": float(args.sample_shift),
        "cfg_scale": float(args.cfg_scale),
        "negative_prompt": str(args.negative_prompt),
        "input_field": "input_video",
        "image_field": "input_image",
        "single_process": True,
        "backend_impl": "official_wan.WanTI2V" if args.backend == "official" else "legacy_diffsynth.WanVideoPipeline",
    }


def convert_official_video_to_thwc(video: Any) -> np.ndarray:
    if isinstance(video, torch.Tensor):
        tensor = video.detach().cpu()
    else:
        tensor = torch.as_tensor(video).cpu()

    if tensor.ndim != 4:
        raise ValueError(f"unexpected official Wan output shape: {tuple(tensor.shape)}")

    if tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 3, 0)
    elif tensor.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"cannot infer channel dimension from official Wan output shape: {tuple(tensor.shape)}")

    frames = tensor.numpy()
    if frames.dtype != np.uint8:
        frames = np.clip(((frames + 1.0) / 2.0) * 255.0, 0, 255).astype(np.uint8)
    return frames


class OfficialWanTI2VWrapper:
    def __init__(self, model: Any, resolved_wan_root: Path, max_area: int):
        self.model = model
        self.resolved_wan_root = resolved_wan_root
        self.max_area = int(max_area)
        self.sample_solver = "unipc"

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
        sample_shift: float,
        sample_solver: str,
        offload_model: bool,
    ) -> np.ndarray:
        solver = normalize_sample_solver(sample_solver or self.sample_solver)
        video = self.model.generate(
            input_prompt=prompt,
            img=input_image,
            size=(int(width), int(height)),
            max_area=self.max_area,
            frame_num=int(num_frames),
            shift=float(sample_shift),
            sample_solver=solver,
            sampling_steps=int(num_inference_steps),
            guide_scale=float(cfg_scale),
            n_prompt=str(negative_prompt),
            seed=int(seed),
            offload_model=bool(offload_model),
        )
        return convert_official_video_to_thwc(video)


class LegacyWanTI2VWrapper:
    def __init__(self, pipe: Any, resolved_wan_root: Path):
        self.pipe = pipe
        self.resolved_wan_root = resolved_wan_root

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
        sample_shift: float,
        sample_solver: str,
        offload_model: bool,
    ) -> np.ndarray:
        del sample_shift, sample_solver, offload_model
        with torch.no_grad():
            generation_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": int(height),
                "width": int(width),
                "num_frames": int(num_frames),
                "seed": int(seed),
                "cfg_scale": float(cfg_scale),
                "num_inference_steps": int(num_inference_steps),
                "tiled": True,
                "input_image": input_image,
            }
            video = self.pipe(
                **generation_kwargs,
            )
        frames = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in video[: int(num_frames)]]
        return np.stack(frames, axis=0)


def build_wan_ti2v_pipeline(args: WanTI2VArgs):
    resolved_wan_root = resolve_official_wan_root(args.wan_root)
    if args.backend == "legacy":
        WanTI2VRunner = _ensure_legacy_wan_imports()
        runner = WanTI2VRunner(model_root=resolved_wan_root, device="cuda")
        return LegacyWanTI2VWrapper(pipe=runner.pipe, resolved_wan_root=resolved_wan_root)

    wan, WAN_CONFIGS, _, MAX_AREA_CONFIGS = _ensure_official_wan_imports()
    patch_wanmodel_from_pretrained_defaults()
    cfg = WAN_CONFIGS["ti2v-5B"]
    max_area = int(MAX_AREA_CONFIGS[args.size])

    model = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=str(resolved_wan_root),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=bool(args.t5_cpu),
        convert_model_dtype=bool(args.convert_model_dtype),
    )
    return OfficialWanTI2VWrapper(model=model, resolved_wan_root=resolved_wan_root, max_area=max_area)


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


def _run_pipe_once(
    *,
    pipe,
    prompt: str,
    negative_prompt: str,
    seed: int,
    input_image: Image.Image,
    height: int,
    width: int,
    num_frames: int,
    cfg_scale: float,
    num_inference_steps: int,
    sample_shift: float,
    sample_solver: str,
    offload_model: bool,
) -> np.ndarray:
    with torch.no_grad():
        return pipe(
            prompt=prompt,
            negative_prompt=str(negative_prompt),
            seed=int(seed),
            input_image=input_image,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            cfg_scale=float(cfg_scale),
            num_inference_steps=int(num_inference_steps),
            sample_shift=float(sample_shift),
            sample_solver=normalize_sample_solver(sample_solver),
            offload_model=bool(offload_model),
        )


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
        f"[case] resolved_wan_root={pipe.resolved_wan_root}",
        f"[case] backend={args.backend}",
        f"[case] sample_solver={normalize_sample_solver(args.sample_solver)}",
        f"[case] negative_prompt={args.negative_prompt}",
    ]

    used_offload = bool(args.offload_model)
    try:
        video = _run_pipe_once(
            pipe=pipe,
            prompt=input_caption,
            negative_prompt=str(args.negative_prompt),
            seed=int(args.seed),
            input_image=image,
            height=int(height),
            width=int(width),
            num_frames=int(args.frame_num),
            cfg_scale=float(args.cfg_scale),
            num_inference_steps=int(args.sampling_steps),
            sample_shift=float(args.sample_shift),
            sample_solver=str(args.sample_solver),
            offload_model=used_offload,
        )
    except RuntimeError as exc:
        if used_offload or "out of memory" not in str(exc).lower():
            raise
        logs.append("[case] retry=oom_with_offload_model")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        video = _run_pipe_once(
            pipe=pipe,
            prompt=input_caption,
            negative_prompt=str(args.negative_prompt),
            seed=int(args.seed),
            input_image=image,
            height=int(height),
            width=int(width),
            num_frames=int(args.frame_num),
            cfg_scale=float(args.cfg_scale),
            num_inference_steps=int(args.sampling_steps),
            sample_shift=float(args.sample_shift),
            sample_solver=str(args.sample_solver),
            offload_model=True,
        )
        used_offload = True

    save_video_np(video, output_video, fps=int(args.fps))

    result = {
        "input_json": str(input_json_path),
        "input_image": str(firstframe_path),
        "input_caption": str(input_caption),
        "output_video": str(output_video),
        "method": derive_method_name(output_video),
        "seed": int(args.seed),
        "step": int(args.sampling_steps),
        "guidance": float(args.cfg_scale),
        "sample_shift": float(args.sample_shift),
        "sample_solver": normalize_sample_solver(args.sample_solver),
        "backend": str(args.backend),
        "negative_prompt": str(args.negative_prompt),
        "offload_model": bool(used_offload),
        "ckpt": str(pipe.resolved_wan_root),
    }
    return result, logs


def cleanup_pipeline(pipe) -> None:
    if hasattr(pipe, "pipe"):
        del pipe.pipe
    if hasattr(pipe, "model"):
        del pipe.model
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
