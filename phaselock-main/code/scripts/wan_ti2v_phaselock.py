#!/usr/bin/env python
"""
Run Wan2.2-TI2V-5B with PhaseLock-style latent delta guidance.

This script reuses the upstream Wan2.2 implementation and inserts a two-stage
sampling procedure:
1. Run a few-step pass to extract a motion prior directly from Wan latents.
2. Re-run full sampling while guiding latent frame deltas toward that prior.
"""

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

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm


THIS_FILE = Path(__file__).resolve()
CODE_DIR = THIS_FILE.parents[1]
REPO_ROOT = CODE_DIR.parent
WAN_REPO_DIR = REPO_ROOT / "Wan2.2-main"

# Force local repository modules ahead of any site-packages install.
for path in (str(CODE_DIR), str(WAN_REPO_DIR)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

from phaselock.utils import set_seed  # noqa: E402
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.textimage2video import WanTI2V  # noqa: E402
from wan.utils.fm_solvers import (  # noqa: E402
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402
from wan.utils.utils import best_output_size, masks_like, save_video  # noqa: E402


DEFAULT_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves "
    "fight intensely on a spotlighted stage."
)


def extract_wan_motion_prior(latents: torch.Tensor) -> torch.Tensor:
    """Extract frame-to-frame latent deltas from Wan latents shaped as (C, T, H, W)."""
    return latents[:, 1:] - latents[:, :-1]


class WanTI2VPhaseLock(WanTI2V):
    """WanTI2V wrapper with PhaseLock's two-stage latent delta guidance."""

    def generate_phaselock(
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
        few_steps=2,
        guidance_strength=0.05,
        guide_start=0,
        guide_end: Optional[int] = None,
        return_few_result=False,
    ):
        if guide_end is None:
            guide_end = sampling_steps // 2

        if img is None:
            few_latents = self._sample_t2v_latents(
                input_prompt=input_prompt,
                size=size,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=few_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model,
            )
            motion_prior = extract_wan_motion_prior(few_latents)
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
                motion_prior=motion_prior,
                guidance_strength=guidance_strength,
                guide_start=guide_start,
                guide_end=guide_end,
            )
        else:
            few_latents = self._sample_i2v_latents(
                input_prompt=input_prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=few_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model,
            )
            motion_prior = extract_wan_motion_prior(few_latents)
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
                motion_prior=motion_prior,
                guidance_strength=guidance_strength,
                guide_start=guide_start,
                guide_end=guide_end,
            )

        videos = self._decode_latents([final_latents])
        if return_few_result:
            few_videos = self._decode_latents([few_latents])
            return videos[0], few_videos[0]
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
        motion_prior=None,
        guidance_strength=0.05,
        guide_start=0,
        guide_end=None,
    ):
        F = frame_num
        target_shape = (
            self.vae.model.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
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

        mask1, mask2 = masks_like(noise, zero=False)
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
            motion_prior=motion_prior,
            guidance_strength=guidance_strength,
            guide_start=guide_start,
            guide_end=guide_end,
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
        motion_prior=None,
        guidance_strength=0.05,
        guide_start=0,
        guide_end=None,
    ):
        ih, iw = img.height, img.width
        dh = self.patch_size[1] * self.vae_stride[1]
        dw = self.patch_size[2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

        F = frame_num
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]
        ) * (ow // self.vae_stride[2]) // (self.patch_size[1] * self.patch_size[2])
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
            (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device,
        )
        z = self.vae.encode([img])
        mask1, mask2 = masks_like([noise], zero=True)
        latent = (1.0 - mask2[0]) * z[0] + mask2[0] * noise

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
            fixed_latent=z[0],
            motion_prior=motion_prior,
            guidance_strength=guidance_strength,
            guide_start=guide_start,
            guide_end=guide_end,
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
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

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
        motion_prior=None,
        guidance_strength=0.05,
        guide_start=0,
        guide_end=None,
    ):
        if guide_end is None:
            guide_end = len(timesteps) // 2

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)

        with (
            torch.amp.autocast("cuda", dtype=self.param_dtype),
            torch.no_grad(),
            no_sync(),
        ):
            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            for step_index, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latents[0].to(self.device)]
                timestep = torch.stack([t]).to(self.device)

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

                temp_x0 = scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
                latent = temp_x0.squeeze(0)

                if motion_prior is not None:
                    latent = self._apply_phaselock_guidance(
                        latent=latent,
                        motion_prior=motion_prior,
                        step_index=step_index,
                        guide_start=guide_start,
                        guide_end=guide_end,
                        guidance_strength=guidance_strength,
                    )

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

    @staticmethod
    def _apply_phaselock_guidance(
        latent,
        motion_prior,
        step_index,
        guide_start,
        guide_end,
        guidance_strength,
    ):
        if step_index < guide_start or step_index >= guide_end:
            return latent
        if latent.shape[1] <= 1:
            return latent

        progress = (step_index - guide_start) / max(guide_end - guide_start, 1)
        strength = guidance_strength * (1.0 - progress)
        current_motion = latent[:, 1:] - latent[:, :-1]
        prior = motion_prior.to(device=latent.device, dtype=latent.dtype)

        if prior.shape != current_motion.shape:
            min_t = min(prior.shape[1], current_motion.shape[1])
            prior = prior[:, :min_t]
            current_motion = current_motion[:, :min_t]

        guidance_signal = prior - current_motion
        guided = latent.clone()
        guided[:, 1 : 1 + guidance_signal.shape[1]] = (
            latent[:, 1 : 1 + guidance_signal.shape[1]] + strength * guidance_signal
        )
        return guided

    def _decode_latents(self, zs):
        if self.rank == 0:
            return self.vae.decode(zs)
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Wan2.2-TI2V-5B with PhaseLock latent delta guidance."
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
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Text prompt.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional image path for TI2V image-conditioned generation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "outputs" / "wan_ti2v_phaselock.mp4"),
        help="Output video path.",
    )
    parser.add_argument("--few_steps", type=int, default=2)
    parser.add_argument("--full_steps", type=int, default=50)
    parser.add_argument("--guidance_strength", type=float, default=0.05)
    parser.add_argument("--guide_start", type=int, default=0)
    parser.add_argument("--guide_end", type=int, default=None)
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
    parser.add_argument("--save_few", action="store_true", default=False)
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
    frame_num = args.frame_num if args.frame_num is not None else cfg.frame_num
    if (frame_num - 1) % 4 != 0:
        raise ValueError(f"frame_num must satisfy 4n+1, got {frame_num}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = None
    if args.image:
        img = Image.open(args.image).convert("RGB")

    logging.info("Loading WanTI2V with PhaseLock wrapper.")
    pipe = WanTI2VPhaseLock(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    logging.info(
        "Generating video with few_steps=%s, full_steps=%s, guidance_strength=%s",
        args.few_steps,
        args.full_steps,
        args.guidance_strength,
    )
    result = pipe.generate_phaselock(
        input_prompt=args.prompt,
        img=img,
        size=SIZE_CONFIGS[args.size],
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=frame_num,
        shift=sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.full_steps,
        guide_scale=sample_guide_scale,
        seed=args.seed,
        offload_model=args.offload_model,
        few_steps=args.few_steps,
        guidance_strength=args.guidance_strength,
        guide_start=args.guide_start,
        guide_end=args.guide_end,
        return_few_result=args.save_few,
    )

    if args.save_few:
        video, few_video = result
        few_path = output_path.with_name(output_path.stem + "_few" + output_path.suffix)
        save_video(
            tensor=few_video[None],
            save_file=str(few_path),
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        logging.info("Saved few-step preview to %s", few_path)
    else:
        video = result

    save_video(
        tensor=video[None],
        save_file=str(output_path),
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    logging.info("Saved video to %s", output_path)


if __name__ == "__main__":
    main()
