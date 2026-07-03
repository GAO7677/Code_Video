#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image


WAN_DIFFUSER_ROOT = "/home/gaoya/diffuser_code/src"


@dataclass
class ProbeConfig:
    model_root: str
    manifest_csv: str
    output_root: str
    limit: Optional[int]
    overwrite: bool
    device: str
    dtype: str
    num_inference_steps: int
    guidance_scale: float
    height: int
    width: int
    num_frames: int
    prompt_max_length: int
    seed_mode: str
    fixed_seed: int
    negative_prompt: Optional[str]
    capture_step_indices: List[int]
    capture_layers: List[int]
    capture_branches: str
    save_final_latents: bool
    use_image_cond: bool


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Wan2.1 T2V 1.3B probing features for a manifest of prompts.")
    parser.add_argument(
        "--model_root",
        default="/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B-Diffusers",
        help="Diffusers-format Wan2.1 T2V 1.3B model directory",
    )
    parser.add_argument(
        "--manifest_csv",
        default="/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/subsets/subset16_smoke.csv",
    )
    parser.add_argument(
        "--output_root",
        default="/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan21/extracted",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--prompt_max_length", type=int, default=256)
    parser.add_argument("--seed_mode", default="source", choices=["source", "fixed"])
    parser.add_argument("--fixed_seed", type=int, default=42)
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--capture_steps", default="10,25,40")
    parser.add_argument("--capture_layers", default="2,8,14,20,29")
    parser.add_argument("--capture_branches", default="cond", choices=["cond", "both"])
    parser.add_argument("--save_final_latents", action="store_true")
    parser.add_argument(
        "--no_image_cond",
        action="store_true",
        help="Disable image conditioning (pure T2V mode, default for Wan2.1 T2V 1.3B).",
    )
    return parser.parse_args()


def torch_dtype_from_name(name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def ensure_sys_path():
    if WAN_DIFFUSER_ROOT not in sys.path:
        sys.path.insert(0, WAN_DIFFUSER_ROOT)


def load_manifest(path: str, limit: Optional[int]) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit is not None else rows


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_prompt(source_record: Dict, manifest_row: Dict[str, str]) -> str:
    prompt = source_record.get("input_caption") or source_record.get("caption") or source_record.get("prompt")
    if prompt:
        return str(prompt)
    fallback = manifest_row["basename"].replace(".mp4", "").replace("_", " ")
    print(f"WARNING: no caption in source JSON for {manifest_row['basename']}, using basename fallback", flush=True)
    return fallback


def choose_negative_prompt(source_record: Dict, cli_negative_prompt: Optional[str]) -> str:
    if cli_negative_prompt is not None:
        return cli_negative_prompt
    value = source_record.get("negative_prompt", "")
    return "" if value is None else str(value)


def choose_seed(source_record: Dict, config: "ProbeConfig") -> int:
    if config.seed_mode == "fixed":
        return config.fixed_seed
    value = source_record.get("seed")
    if value is None:
        print(f"WARNING: no seed in source JSON, using fixed_seed={config.fixed_seed}", flush=True)
        return config.fixed_seed
    return int(value)


def choose_image_path(source_record: Dict) -> Optional[str]:
    """Return an image path for I2V conditioning, or None if not available."""
    img = source_record.get("input_image")
    if img and Path(img).exists():
        return str(img)
    # Fallback: extract first frame path from input_video if present
    vid = source_record.get("input_video")
    if vid and Path(vid).exists():
        return str(vid)
    return None


def load_image_for_cond(image_path: str, height: int, width: int) -> Image.Image:
    """Load a PIL image (or first frame of video) resized to (width, height)."""
    p = Path(image_path)
    if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        import cv2
        cap = cv2.VideoCapture(str(p))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Cannot read first frame from video: {image_path}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
    else:
        img = Image.open(image_path).convert("RGB")
    return img.resize((width, height), Image.LANCZOS)


def encode_image_latent(pipe, pil_image: Image.Image, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """VAE-encode a PIL image into a latent tensor of shape [1, z_dim, 1, H_lat, W_lat]."""
    import numpy as np
    arr = np.array(pil_image, dtype=np.float32) / 255.0  # [H, W, 3]
    arr = arr * 2.0 - 1.0  # [-1, 1]
    img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
    img_tensor = img_tensor.to(device=device, dtype=pipe.vae.dtype)

    with torch.no_grad():
        latent = pipe.vae.encode(img_tensor).latent_dist.mode()  # [1, z_dim, 1, H_lat, W_lat]

    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(device, dtype)
    )
    latents_std = 1.0 / (
        torch.tensor(pipe.vae.config.latents_std)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(device, dtype)
    )
    latent = latent.to(dtype)
    latent = (latent - latents_mean) * latents_std
    return latent


def compute_token_grid(latent_shape: Tuple[int, ...], patch_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
    _, _, num_frames, height, width = latent_shape
    p_t, p_h, p_w = patch_size
    return num_frames // p_t, height // p_h, width // p_w


def reshape_tokens(hidden_states: torch.Tensor, token_grid: Tuple[int, int, int]) -> Optional[torch.Tensor]:
    batch_size, token_count, hidden_dim = hidden_states.shape
    grid_tokens = token_grid[0] * token_grid[1] * token_grid[2]
    if token_count != grid_tokens:
        return None
    return hidden_states.reshape(batch_size, token_grid[0], token_grid[1], token_grid[2], hidden_dim)


class BlockFeatureRecorder:
    def __init__(self, capture_layers: List[int], capture_step_indices: List[int], capture_branches: str):
        self.capture_layers = set(capture_layers)
        self.capture_step_indices = set(capture_step_indices)
        self.capture_branches = capture_branches
        self.active_step_idx: Optional[int] = None
        self.active_timestep: Optional[int] = None
        self.active_branch: Optional[str] = None
        self.active_token_grid: Optional[Tuple[int, int, int]] = None
        self.data: Dict[int, Dict] = {}
        self._handles = []

    def should_capture_branch(self, branch_name: str) -> bool:
        return self.capture_branches == "both" or branch_name == "cond"

    def activate(
        self,
        step_idx: int,
        timestep_value: int,
        branch_name: str,
        token_grid: Tuple[int, int, int],
        latent_shape: Tuple[int, ...],
    ):
        if step_idx not in self.capture_step_indices or not self.should_capture_branch(branch_name):
            self.active_step_idx = None
            self.active_timestep = None
            self.active_branch = None
            self.active_token_grid = None
            return

        self.active_step_idx = step_idx
        self.active_timestep = timestep_value
        self.active_branch = branch_name
        self.active_token_grid = token_grid
        step_payload = self.data.setdefault(
            step_idx,
            {
                "timestep": timestep_value,
                "token_grid": list(token_grid),
                "latent_shape_pre": list(latent_shape),
                "branches": {},
            },
        )
        step_payload["branches"].setdefault(branch_name, {})

    def deactivate(self):
        self.active_step_idx = None
        self.active_timestep = None
        self.active_branch = None
        self.active_token_grid = None

    def finalize_step(self, step_idx: int, latent_shape_post: Tuple[int, ...]):
        if step_idx in self.data:
            self.data[step_idx]["latent_shape_post"] = list(latent_shape_post)

    def _tensor_to_cpu(self, value: torch.Tensor) -> torch.Tensor:
        return value.detach().float().cpu()

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if self.active_step_idx is None or self.active_branch is None:
                return

            hidden_in = inputs[0]
            hidden_out = output
            hidden_delta = hidden_out - hidden_in

            layer_payload = {
                "token_count": int(hidden_out.shape[1]),
                "hidden_dim": int(hidden_out.shape[2]),
                "h_post_global_mean": self._tensor_to_cpu(hidden_out.mean(dim=1).squeeze(0)),
                "delta_h_global_mean": self._tensor_to_cpu(hidden_delta.mean(dim=1).squeeze(0)),
                "h_post_token_l2_mean": float(hidden_out.float().norm(dim=-1).mean().item()),
                "delta_h_token_l2_mean": float(hidden_delta.float().norm(dim=-1).mean().item()),
            }

            if self.active_token_grid is not None:
                token_grid_states = reshape_tokens(hidden_out, self.active_token_grid)
                token_grid_delta = reshape_tokens(hidden_delta, self.active_token_grid)
                if token_grid_states is not None and token_grid_delta is not None:
                    layer_payload["h_post_frame_mean"] = self._tensor_to_cpu(
                        token_grid_states.mean(dim=(2, 3)).squeeze(0)
                    )
                    layer_payload["delta_h_frame_mean"] = self._tensor_to_cpu(
                        token_grid_delta.mean(dim=(2, 3)).squeeze(0)
                    )

            self.data[self.active_step_idx]["branches"][self.active_branch][layer_idx] = layer_payload

        return hook

    def register(self, transformer):
        for idx, block in enumerate(transformer.blocks):
            if idx in self.capture_layers:
                self._handles.append(block.register_forward_hook(self._make_hook(idx)))

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []


def load_pipe(config: ProbeConfig):
    ensure_sys_path()
    from diffusers import WanPipeline

    dtype = torch_dtype_from_name(config.dtype)
    pipe = WanPipeline.from_pretrained(
        config.model_root,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.to(config.device)
    return pipe


def build_sample_runtime(manifest_row: Dict[str, str], config: "ProbeConfig") -> Dict:
    source_record = load_json(manifest_row["json_path"])
    sample_id = f"{manifest_row['pair_id']}__{manifest_row['role']}"
    image_path = choose_image_path(source_record) if config.use_image_cond else None
    return {
        "sample_id": sample_id,
        "manifest_row": manifest_row,
        "source_record": source_record,
        "prompt": choose_prompt(source_record, manifest_row),
        "negative_prompt": choose_negative_prompt(source_record, config.negative_prompt),
        "seed": choose_seed(source_record, config),
        "image_path": image_path,
    }


def run_single(pipe, config: ProbeConfig, runtime: Dict) -> Dict[str, str]:
    manifest_row = runtime["manifest_row"]
    source_record = runtime["source_record"]
    sample_id = runtime["sample_id"]
    out_dir = Path(config.output_root) / sample_id
    feature_path = out_dir / "probe_features.pt"
    meta_path = out_dir / "meta.json"

    if feature_path.exists() and meta_path.exists() and not config.overwrite:
        return {"sample_id": sample_id, "status": "skipped_existing", "output_dir": str(out_dir)}

    out_dir.mkdir(parents=True, exist_ok=True)

    recorder = BlockFeatureRecorder(
        capture_layers=config.capture_layers,
        capture_step_indices=config.capture_step_indices,
        capture_branches=config.capture_branches,
    )
    recorder.register(pipe.transformer)

    do_cfg = config.guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=runtime["prompt"],
        negative_prompt=runtime["negative_prompt"],
        do_classifier_free_guidance=do_cfg,
        num_videos_per_prompt=1,
        max_sequence_length=config.prompt_max_length,
        device=pipe._execution_device,
    )

    transformer_dtype = pipe.transformer.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    pipe.scheduler.set_timesteps(config.num_inference_steps, device=pipe._execution_device)
    timesteps = pipe.scheduler.timesteps
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.transformer.config.in_channels,
        height=config.height,
        width=config.width,
        num_frames=config.num_frames,
        dtype=torch.float32,
        device=pipe._execution_device,
        generator=torch.Generator(device=config.device).manual_seed(runtime["seed"]),
        latents=None,
    )
    patch_size = tuple(pipe.transformer.config.patch_size)

    # Build image conditioning if available (I2V mode for Wan2.2 TI2V).
    # first_frame_mask is 0 at the first temporal latent frame and 1 elsewhere.
    # The conditioned latent_model_input replaces first-frame noise with VAE(image),
    # and the per-token timestep is set to 0 for already-denoised (first-frame) positions.
    image_path = runtime.get("image_path")
    image_cond_active = False
    latent_condition = None
    first_frame_mask = None
    if image_path is not None:
        try:
            pil_img = load_image_for_cond(image_path, config.height, config.width)
            dtype_t = torch_dtype_from_name(config.dtype)
            latent_condition = encode_image_latent(pipe, pil_img, config.height, config.width, pipe._execution_device, dtype_t)
            # shape: [1, z_dim, 1, H_lat, W_lat] -> broadcast to [1, z_dim, T_lat, H_lat, W_lat]
            num_latent_frames = latents.shape[2]
            latent_condition = latent_condition.expand(-1, -1, num_latent_frames, -1, -1).clone()
            first_frame_mask = torch.ones(1, 1, num_latent_frames, latents.shape[3], latents.shape[4],
                                          dtype=torch.float32, device=pipe._execution_device)
            first_frame_mask[:, :, 0] = 0.0
            image_cond_active = True
            print(f"[I2V] image conditioning active: {image_path}", flush=True)
        except Exception as e:
            print(f"WARNING: failed to load/encode image {image_path}: {e}, falling back to T2V mode", flush=True)

    if not image_cond_active:
        # Pure T2V: all-ones mask, latent_model_input = latents
        first_frame_mask = torch.ones(latents.shape, dtype=torch.float32, device=pipe._execution_device)

    try:
        for step_idx, t in enumerate(timesteps):
            if image_cond_active:
                # I2V: blend conditioned first frame with noisy latents
                latent_model_input = (
                    (1.0 - first_frame_mask) * latent_condition + first_frame_mask * latents
                ).to(transformer_dtype)
            else:
                latent_model_input = latents.to(transformer_dtype)

            token_grid = compute_token_grid(tuple(latent_model_input.shape), patch_size)

            if pipe.config.expand_timesteps:
                # Use first_frame_mask's spatial downsampled version for per-token timestep.
                # first_frame_mask shape: [1, 1, T, H, W] (I2V) or [1, C, T, H, W] (T2V all-ones).
                # For T2V all-ones mask: take [0][0] -> [T, H, W], then spatial downsample.
                # For I2V first_frame_mask: take [0][0] -> [T, H, W].
                ts_mask = first_frame_mask[0][0]  # [T, H, W]
                temp_ts = (ts_mask[:, ::2, ::2] * t).flatten()
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                timestep = t.expand(latents.shape[0])

            timestep_value = int(t.item()) if torch.is_tensor(t) else int(t)

            recorder.activate(step_idx, timestep_value, "cond", token_grid, tuple(latent_model_input.shape))
            with pipe.transformer.cache_context("cond"):
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
            recorder.deactivate()

            if do_cfg:
                recorder.activate(step_idx, timestep_value, "uncond", token_grid, tuple(latent_model_input.shape))
                with pipe.transformer.cache_context("uncond"):
                    noise_uncond = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        return_dict=False,
                    )[0]
                recorder.deactivate()
                noise_pred = noise_uncond + config.guidance_scale * (noise_pred - noise_uncond)

            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            recorder.finalize_step(step_idx, tuple(latents.shape))

        payload = {
            "meta": {
                "sample_id": sample_id,
                "pair_id": manifest_row["pair_id"],
                "role": manifest_row["role"],
                "basename": manifest_row["basename"],
                "manifest_video_path": manifest_row["video_path"],
                "manifest_json_path": manifest_row["json_path"],
                "relative_path": manifest_row["relative_path"],
                "source_surprise_score": float(manifest_row["surprise_score"]),
                "source_tag": manifest_row["source_tag"],
                "prompt": runtime["prompt"],
                "negative_prompt": runtime["negative_prompt"],
                "seed": runtime["seed"],
                "source_input_json": source_record.get("input_json"),
                "source_input_image": source_record.get("input_image"),
                "source_input_caption": source_record.get("input_caption"),
                "source_generation_method": source_record.get("method"),
                "source_generation_backend": source_record.get("backend"),
                "height": config.height,
                "width": config.width,
                "num_frames": config.num_frames,
                "num_inference_steps": config.num_inference_steps,
                "guidance_scale": config.guidance_scale,
                "capture_step_indices": config.capture_step_indices,
                "capture_layers": config.capture_layers,
                "capture_branches": config.capture_branches,
                "model_root": config.model_root,
                "image_cond_active": image_cond_active,
                "image_path_used": image_path if image_cond_active else None,
                "final_latents_shape": list(latents.shape),
            },
            "features": recorder.data,
        }

        if config.save_final_latents:
            payload["final_latents"] = latents.detach().float().cpu()

        torch.save(payload, feature_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload["meta"], f, indent=2, ensure_ascii=False)
        return {"sample_id": sample_id, "status": "ok", "output_dir": str(out_dir)}
    finally:
        recorder.remove()
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    config = ProbeConfig(
        model_root=args.model_root,
        manifest_csv=args.manifest_csv,
        output_root=args.output_root,
        limit=args.limit,
        overwrite=args.overwrite,
        device=args.device,
        dtype=args.dtype,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        prompt_max_length=args.prompt_max_length,
        seed_mode=args.seed_mode,
        fixed_seed=args.fixed_seed,
        negative_prompt=args.negative_prompt,
        capture_step_indices=[int(x) for x in args.capture_steps.split(",") if x.strip()],
        capture_layers=[int(x) for x in args.capture_layers.split(",") if x.strip()],
        capture_branches=args.capture_branches,
        save_final_latents=args.save_final_latents,
        use_image_cond=not args.no_image_cond,
    )

    os.makedirs(config.output_root, exist_ok=True)
    samples = load_manifest(config.manifest_csv, config.limit)
    pipe = load_pipe(config)

    results = []
    for manifest_row in samples:
        runtime = build_sample_runtime(manifest_row, config)
        sample_output_dir = Path(config.output_root) / runtime["sample_id"]
        if (
            not config.overwrite
            and (sample_output_dir / "probe_features.pt").is_file()
            and (sample_output_dir / "meta.json").is_file()
        ):
            result = {
                "sample_id": runtime["sample_id"],
                "status": "skipped_existing",
                "output_dir": str(sample_output_dir),
            }
        else:
            result = run_single(pipe, config, runtime)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    summary_path = Path(config.output_root) / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
