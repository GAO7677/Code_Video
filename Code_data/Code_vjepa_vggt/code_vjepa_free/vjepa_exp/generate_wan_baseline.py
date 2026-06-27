from __future__ import annotations

import argparse
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

"""
Examples



Override all outputs to a new root directory:
CUDA_VISIBLE_DEVICES=5 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_exp/generate_wan_baseline.py \
    --manifest /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42/manifest.json \
    --output-root /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42_ti2v_5b \
    --cuda-visible-devices 5 \
    --force

Run only one case:
CUDA_VISIBLE_DEVICES=5 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_exp/generate_wan_baseline.py \
    --manifest /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42/manifest.json \
    --output-root /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42_ti2v_5b \
    --cuda-visible-devices 5 \
    --limit 1 \
    --force
"""

PROJECT_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
OFFICIAL_WAN_REPO = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main")
LEGACY_WAN_REPO = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")


@dataclass
class WanTI2VArgs:
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
    output_root: Path | None
    force: bool


def _ensure_official_wan_imports():
    repo_root_str = str(OFFICIAL_WAN_REPO.resolve())
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    import wan  # type: ignore
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS  # type: ignore

    return wan, WAN_CONFIGS, MAX_AREA_CONFIGS


def _ensure_legacy_wan_imports():
    repo_root_str = str(LEGACY_WAN_REPO.resolve())
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


def write_json(json_path: Path, payload: dict[str, Any]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


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
    candidate = wan_root.expanduser().resolve()
    if candidate.is_dir() and (candidate / "Wan2.2_VAE.pth").exists():
        return candidate
    if DEFAULT_WAN_ROOT.is_dir() and (DEFAULT_WAN_ROOT / "Wan2.2_VAE.pth").exists():
        return DEFAULT_WAN_ROOT.resolve()
    raise FileNotFoundError(
        "unable to resolve official Wan2.2 TI2V checkpoint directory; "
        f"expected Wan2.2_VAE.pth under {wan_root}"
    )


def convert_official_video_to_thwc(video: Any) -> np.ndarray:
    if isinstance(video, torch.Tensor):
        tensor = video.detach().cpu()
    else:
        tensor = torch.as_tensor(video).cpu()

    if tensor.ndim != 4:
        raise ValueError(f"unexpected official Wan output shape: {tuple(tensor.shape)}")

    if tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 3, 0)
    elif tensor.shape[-1] != 3:
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
            video = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=int(height),
                width=int(width),
                num_frames=int(num_frames),
                seed=int(seed),
                cfg_scale=float(cfg_scale),
                num_inference_steps=int(num_inference_steps),
                tiled=True,
                input_image=input_image,
            )
        frames = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in video[: int(num_frames)]]
        return np.stack(frames, axis=0)


def build_wan_ti2v_pipeline(args: WanTI2VArgs):
    resolved_wan_root = resolve_official_wan_root(args.wan_root)
    if args.backend == "legacy":
        WanTI2VRunner = _ensure_legacy_wan_imports()
        runner = WanTI2VRunner(model_root=resolved_wan_root, device="cuda")
        return LegacyWanTI2VWrapper(pipe=runner.pipe, resolved_wan_root=resolved_wan_root)

    wan, WAN_CONFIGS, MAX_AREA_CONFIGS = _ensure_official_wan_imports()
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
    case: dict[str, Any],
    output_video: Path,
) -> tuple[dict[str, Any], list[str]]:
    case_id = str(case["case_id"])
    prompt = str(case.get("input_video_prompt") or case.get("prompt") or "")
    if not prompt:
        raise ValueError(f"missing input_video_prompt/prompt for case {case_id}")
    image_path_value = case.get("input_image") or case.get("image_path")
    if not image_path_value:
        raise ValueError(f"missing input_image/image_path for case {case_id}")
    image_path = Path(str(image_path_value)).expanduser().resolve()
    seed = int(case.get("seed", args.seed))
    height, width = _parse_size(args.size)

    image = Image.open(image_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)

    logs = [
        f"[case] case_id={case_id}",
        f"[case] input_image={image_path}",
        f"[case] input_video_prompt={prompt}",
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
            prompt=prompt,
            negative_prompt=str(args.negative_prompt),
            seed=seed,
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
            prompt=prompt,
            negative_prompt=str(args.negative_prompt),
            seed=seed,
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
        "case_id": case_id,
        "input_image": str(image_path),
        "input_video_prompt": prompt,
        "output_video": str(output_video),
        "method": output_video.parent.name,
        "seed": seed,
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


def ensure_cuda_env(cuda_visible_devices: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    project_root_str = str(PROJECT_ROOT.resolve())
    official_repo_str = str(OFFICIAL_WAN_REPO.resolve())
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    paths = [p for p in current_pythonpath.split(":") if p]
    for required in [project_root_str, official_repo_str]:
        if required not in paths:
            paths.insert(0, required)
    os.environ["PYTHONPATH"] = ":".join(paths)
    if not torch.cuda.is_available():
        raise RuntimeError("WanTI2V baseline requires CUDA, but no GPU is visible.")
    torch.cuda.set_device(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Wan2.2 TI2V-5B baseline videos directly from the vjepa_exp manifest format."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", default="legacy", choices=["legacy", "official"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_output_video_path(case: dict[str, Any], output_root: Path | None) -> Path:
    if output_root is None:
        return Path(str(case["video_path"])).expanduser().resolve()

    case_id = str(case["case_id"])
    original_video_name = Path(str(case["video_path"])).name
    return output_root.expanduser().resolve() / case_id / original_video_name


def main() -> None:
    cli_args = parse_args()
    ensure_cuda_env(cli_args.cuda_visible_devices)

    manifest_path = cli_args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wan_args_raw = manifest["wan_args"]
    cases = manifest["cases"]
    if cli_args.limit is not None:
        cases = cases[: cli_args.limit]

    args = WanTI2VArgs(
        wan_root=Path(manifest.get("wan_root", str(DEFAULT_WAN_ROOT))).expanduser().resolve(),
        backend=str(cli_args.backend),
        size=str(wan_args_raw["size"]),
        frame_num=int(wan_args_raw["frame_num"]),
        fps=int(wan_args_raw["fps"]),
        seed=int(cases[0]["seed"]) if cases else 42,
        sample_solver=str(wan_args_raw["sample_solver"]),
        sampling_steps=int(wan_args_raw["sampling_steps"]),
        sample_shift=float(wan_args_raw["sample_shift"]),
        cfg_scale=float(wan_args_raw["cfg_scale"]),
        negative_prompt=str(wan_args_raw["negative_prompt"]),
        offload_model=bool(wan_args_raw["offload_model"]),
        t5_cpu=bool(wan_args_raw["t5_cpu"]),
        convert_model_dtype=bool(wan_args_raw["convert_model_dtype"]),
        output_root=cli_args.output_root.expanduser().resolve() if cli_args.output_root is not None else None,
        force=bool(cli_args.force),
    )

    pipe = build_wan_ti2v_pipeline(args)
    try:
        for case in cases:
            output_video = resolve_output_video_path(case, args.output_root)
            output_json = output_video.with_suffix(".json")
            if output_video.exists() and output_json.exists() and not args.force:
                print(f"[skip] {case['case_id']}")
                continue

            result, case_logs = run_single_case(
                pipe=pipe,
                args=args,
                case=case,
                output_video=output_video,
            )
            write_json(output_json, result)
            for log_line in case_logs:
                print(log_line)
            print(f"[done] {case['case_id']}")
    finally:
        cleanup_pipeline(pipe)


if __name__ == "__main__":
    main()
