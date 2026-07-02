from __future__ import annotations

import argparse
import gc
import logging
import math
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm


THIS_FILE = Path(__file__).resolve()
GUIDANCE_DIR = THIS_FILE.parent
WAN_REPO_DIR = Path("/home/gaoya/Code_Video/phaselock-main/Wan2.2-main")

for path in (str(GUIDANCE_DIR), str(WAN_REPO_DIR)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.textimage2video import WanTI2V  # noqa: E402
from wan.utils.fm_solvers import (  # noqa: E402
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402
from wan.utils.utils import best_output_size, masks_like, save_video  # noqa: E402

from vjepa_surprise import VJEPASurpriseEnergy  # noqa: E402
from wan_latent_guidance import (  # noqa: E402
    WanVJEPAConfig,
    apply_vjepa_latent_guidance,
    pick_guidance_step_indices,
)


DEFAULT_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves "
    "fight intensely on a spotlighted stage."
)
DEFAULT_OUTPUT_PATH = Path("/data/gaoya/agent-data/outputs/vjepa_guidance/wan_ti2v_vjepa.mp4")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WanTI2VVJEPA(WanTI2V):
    def __init__(
        self,
        *args,
        vjepa_model_name: str = "vitg",
        vjepa_checkpoint_path: Optional[str] = None,
        vjepa_config: Optional[WanVJEPAConfig] = None,
        **kwargs,
    ):
        checkpoint_dir = kwargs.get("checkpoint_dir")
        if checkpoint_dir is None and len(args) >= 2:
            checkpoint_dir = args[1]
        convert_model_dtype = kwargs.get("convert_model_dtype", False)
        super().__init__(*args, **kwargs)
        self.vjepa_model_name = vjepa_model_name
        self.vjepa_checkpoint_path = vjepa_checkpoint_path
        self.vjepa_config = vjepa_config or WanVJEPAConfig()
        self._vjepa_energy: VJEPASurpriseEnergy | None = None

        if any(parameter.is_meta for parameter in self.model.parameters()):
            if checkpoint_dir is None:
                raise ValueError(
                    "Wan model contains meta tensors, but checkpoint_dir is unavailable for reload."
                )
            logging.info(
                "Reloading WanModel with low_cpu_mem_usage=False to materialize meta tensors."
            )
            reloaded_model = self.model.__class__.from_pretrained(
                checkpoint_dir,
                low_cpu_mem_usage=False,
            )
            reloaded_model.eval().requires_grad_(False)
            if convert_model_dtype:
                reloaded_model.to(self.param_dtype)
            self.model = reloaded_model

        # Guidance uses gradients through the current latent and VAE decode only.
        vae_module = getattr(self.vae, "model", None)
        if vae_module is not None:
            for parameter in vae_module.parameters():
                parameter.requires_grad_(False)
            vae_module.eval()

    def _ensure_vjepa_energy(self) -> VJEPASurpriseEnergy:
        if self._vjepa_energy is None:
            logging.info("Loading V-JEPA energy model: %s", self.vjepa_model_name)
            self._vjepa_energy = VJEPASurpriseEnergy(
                model_name=self.vjepa_model_name,
                device=self.device,
                local_torchhub=True,
                checkpoint_path=self.vjepa_checkpoint_path,
            )
        return self._vjepa_energy

    def generate_vjepa(
        self,
        input_prompt: str,
        img: Optional[Image.Image] = None,
        size=(1280, 704),
        max_area=704 * 1280,
        frame_num=121,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
    ):
        self._ensure_vjepa_energy()

        if img is None:
            final_latents = self._sample_t2v_latents(
                input_prompt=input_prompt,
                size=size,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model,
            )
        else:
            final_latents = self._sample_i2v_latents(
                input_prompt=input_prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model,
            )

        videos = self._decode_latents([final_latents])
        return videos[0]

    def _sample_t2v_latents(
        self,
        input_prompt,
        size,
        frame_num,
        shift,
        sample_solver,
        sampling_steps,
        guide_scale,
        n_prompt,
        seed,
        offload_model,
    ):
        frames = frame_num
        target_shape = (
            self.vae.model.z_dim,
            (frames - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (self.patch_size[1] * self.patch_size[2])
            * target_shape[1]
            / self.sp_size
        ) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        context, context_null, seed_g = self._prepare_sampling_context(
            input_prompt=input_prompt,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model,
        )
        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g,
            )
        ]

        _, mask2 = masks_like(noise, zero=False)
        latents = noise
        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        scheduler, timesteps = self._create_scheduler(
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            shift=shift,
        )
        latents = self._run_sampling_loop(
            latents=latents,
            scheduler=scheduler,
            timesteps=timesteps,
            mask2=mask2,
            seq_len=seq_len,
            arg_c=arg_c,
            arg_null=arg_null,
            guide_scale=guide_scale,
            seed_g=seed_g,
            offload_model=offload_model,
        )
        return latents[0]

    def _sample_i2v_latents(
        self,
        input_prompt,
        img,
        max_area,
        frame_num,
        shift,
        sample_solver,
        sampling_steps,
        guide_scale,
        n_prompt,
        seed,
        offload_model,
    ):
        input_height, input_width = img.height, img.width
        down_h = self.patch_size[1] * self.vae_stride[1]
        down_w = self.patch_size[2] * self.vae_stride[2]
        output_width, output_height = best_output_size(input_width, input_height, down_w, down_h, max_area)

        scale = max(output_width / input_width, output_height / input_height)
        img = img.resize((round(input_width * scale), round(input_height * scale)), Image.LANCZOS)
        x1 = (img.width - output_width) // 2
        y1 = (img.height - output_height) // 2
        img = img.crop((x1, y1, x1 + output_width, y1 + output_height))
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

        frames = frame_num
        seq_len = ((frames - 1) // self.vae_stride[0] + 1) * (
            output_height // self.vae_stride[1]
        ) * (output_width // self.vae_stride[2]) // (self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        context, context_null, seed_g = self._prepare_sampling_context(
            input_prompt=input_prompt,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model,
        )

        noise = torch.randn(
            self.vae.model.z_dim,
            (frames - 1) // self.vae_stride[0] + 1,
            output_height // self.vae_stride[1],
            output_width // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device,
        )
        fixed_latent = self.vae.encode([img])[0]
        _, mask2 = masks_like([noise], zero=True)
        latent = (1.0 - mask2[0]) * fixed_latent + mask2[0] * noise

        arg_c = {"context": [context[0]], "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        scheduler, timesteps = self._create_scheduler(
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            shift=shift,
        )
        out = self._run_sampling_loop(
            latents=[latent],
            scheduler=scheduler,
            timesteps=timesteps,
            mask2=mask2,
            seq_len=seq_len,
            arg_c=arg_c,
            arg_null=arg_null,
            guide_scale=guide_scale,
            seed_g=seed_g,
            offload_model=offload_model,
            fixed_latent=fixed_latent,
        )
        return out[0]

    def _prepare_sampling_context(self, input_prompt, n_prompt, seed, offload_model):
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device("cpu"))
            context_null = self.text_encoder([n_prompt], torch.device("cpu"))
            context = [tensor.to(self.device) for tensor in context]
            context_null = [tensor.to(self.device) for tensor in context_null]

        return context, context_null, seed_g

    def _create_scheduler(self, sample_solver, sampling_steps, shift):
        if sample_solver == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = scheduler.timesteps
        elif sample_solver == "dpm++":
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                scheduler,
                device=self.device,
                sigmas=sampling_sigmas,
            )
        else:
            raise NotImplementedError(f"Unsupported solver: {sample_solver}")
        return scheduler, timesteps

    def _run_sampling_loop(
        self,
        latents,
        scheduler,
        timesteps,
        mask2,
        seq_len,
        arg_c,
        arg_null,
        guide_scale,
        seed_g,
        offload_model,
        fixed_latent=None,
    ):
        selected_steps = set(
            pick_guidance_step_indices(
                total_steps=len(timesteps),
                count=self.vjepa_config.guidance_steps,
                min_step_percent=self.vjepa_config.min_step_percent,
                max_step_percent=self.vjepa_config.max_step_percent,
            )
        )
        energy_fn = self._ensure_vjepa_energy()

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)

        with torch.amp.autocast("cuda", dtype=self.param_dtype), no_sync():
            for step_index, timestep_value in enumerate(tqdm(timesteps)):
                if (offload_model or self.init_on_cpu) and next(self.model.parameters()).device.type != "cuda":
                    self.model.to(self.device)
                    torch.cuda.empty_cache()

                with torch.no_grad():
                    latent_model_input = [latents[0].to(self.device)]
                    timestep = torch.stack([timestep_value]).to(self.device)

                    temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                    temp_ts = torch.cat(
                        [temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep]
                    )
                    timestep = temp_ts.unsqueeze(0)

                    noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                    if offload_model:
                        torch.cuda.empty_cache()
                    noise_pred_uncond = self.model(
                        latent_model_input,
                        t=timestep,
                        **arg_null,
                    )[0]
                    if offload_model:
                        torch.cuda.empty_cache()

                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )

                latent_xt = latents[0]
                if step_index in selected_steps:
                    if next(self.model.parameters()).device.type == "cuda":
                        self.model.cpu()
                        torch.cuda.empty_cache()
                    with torch.enable_grad():
                        latent_xt, stats = apply_vjepa_latent_guidance(
                            latent_xt=latent_xt,
                            model_output=noise_pred,
                            timestep=timestep_value,
                            scheduler=scheduler,
                            vae=self.vae,
                            energy_fn=energy_fn,
                            config=self.vjepa_config,
                        )
                    logging.info(
                        "V-JEPA step=%d timestep=%d energy=%.6f grad_rms=%.6f preview=%dx%dx%d",
                        step_index,
                        int(timestep_value.item()),
                        stats["energy"],
                        stats["grad_rms"],
                        int(stats["preview_frames"]),
                        int(stats["preview_height"]),
                        int(stats["preview_width"]),
                    )

                with torch.no_grad():
                    temp_x0 = scheduler.step(
                        noise_pred.unsqueeze(0),
                        timestep_value,
                        latent_xt.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g,
                    )[0]
                    latent = temp_x0.squeeze(0)

                    if fixed_latent is not None:
                        latent = (1.0 - mask2[0]) * fixed_latent + mask2[0] * latent

                    latents = [latent]
                    del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        del scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        return latents

    def _decode_latents(self, zs):
        if self.rank == 0:
            return self.vae.decode(zs)
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Wan2.2-TI2V-5B with V-JEPA latent guidance."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B",
        help="Path to the Wan2.2-TI2V-5B checkpoint directory.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1280*704",
        choices=list(SIZE_CONFIGS.keys()),
        help="Video size config.",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional image path for TI2V image-conditioned generation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output video path.",
    )
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=None)
    parser.add_argument(
        "--sample_solver",
        type=str,
        default="unipc",
        choices=["unipc", "dpm++"],
    )
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offload_model", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--convert_model_dtype", action="store_true", default=False)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--vjepa_model", type=str, default="vitg", choices=["vith", "vitg", "vitg384"])
    parser.add_argument(
        "--vjepa_ckpt",
        type=str,
        default=None,
        help="Optional local V-JEPA checkpoint path. If omitted, a known local default is used when available.",
    )
    parser.add_argument("--vjepa_guidance_steps", type=int, default=6)
    parser.add_argument("--vjepa_min_step_percent", type=float, default=0.2)
    parser.add_argument("--vjepa_max_step_percent", type=float, default=0.8)
    parser.add_argument("--vjepa_latent_step_size", type=float, default=0.02)
    parser.add_argument("--vjepa_preview_downsample_factor", type=int, default=2)
    parser.add_argument("--vjepa_preview_frame_stride", type=int, default=1)
    parser.add_argument("--vjepa_window_size", type=int, default=16)
    parser.add_argument("--vjepa_context_frames", type=int, default=8)
    parser.add_argument("--vjepa_stride", type=int, default=4)
    parser.add_argument("--vjepa_reduction", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--vjepa_grad_norm_mode", type=str, default="rms", choices=["none", "rms", "l2"])
    parser.add_argument("--vjepa_max_grad_norm", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    set_seed(args.seed)

    cfg = WAN_CONFIGS["ti2v-5B"]
    sample_shift = args.sample_shift if args.sample_shift is not None else cfg.sample_shift
    sample_guide_scale = (
        args.sample_guide_scale
        if args.sample_guide_scale is not None
        else cfg.sample_guide_scale
    )
    sample_steps = args.sample_steps if args.sample_steps is not None else 50
    frame_num = args.frame_num if args.frame_num is not None else cfg.frame_num
    if (frame_num - 1) % 4 != 0:
        raise ValueError(f"frame_num must satisfy 4n+1, got {frame_num}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = None
    if args.image:
        img = Image.open(args.image).convert("RGB")

    vjepa_config = WanVJEPAConfig(
        guidance_steps=args.vjepa_guidance_steps,
        min_step_percent=args.vjepa_min_step_percent,
        max_step_percent=args.vjepa_max_step_percent,
        latent_step_size=args.vjepa_latent_step_size,
        preview_downsample_factor=args.vjepa_preview_downsample_factor,
        preview_frame_stride=args.vjepa_preview_frame_stride,
        window_size=args.vjepa_window_size,
        context_frames=args.vjepa_context_frames,
        stride=args.vjepa_stride,
        reduction=args.vjepa_reduction,
        gradient_normalization=args.vjepa_grad_norm_mode,
        max_grad_norm=args.vjepa_max_grad_norm,
    )

    logging.info("Loading WanTI2V with V-JEPA wrapper.")
    pipe = WanTI2VVJEPA(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        vjepa_model_name=args.vjepa_model,
        vjepa_checkpoint_path=args.vjepa_ckpt,
        vjepa_config=vjepa_config,
    )

    logging.info(
        "Generating video with sample_steps=%s, vjepa_model=%s, guidance_steps=%s, latent_step_size=%s",
        sample_steps,
        args.vjepa_model,
        args.vjepa_guidance_steps,
        args.vjepa_latent_step_size,
    )
    result = pipe.generate_vjepa(
        input_prompt=args.prompt,
        img=img,
        size=SIZE_CONFIGS[args.size],
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=frame_num,
        shift=sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=sample_steps,
        guide_scale=sample_guide_scale,
        seed=args.seed,
        offload_model=args.offload_model,
    )

    logging.info("Saving video to %s", output_path)
    save_video(
        tensor=result,
        save_file=str(output_path),
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
