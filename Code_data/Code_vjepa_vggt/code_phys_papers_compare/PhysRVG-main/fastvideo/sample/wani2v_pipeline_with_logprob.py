from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
import math
import numpy as np
from PIL import Image

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents
from diffusers.utils import export_to_video, load_image
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan, ModularPipeline

from fastvideo.utils.logging_ import main_print

def sde_step_with_logprob(
    scheduler: UniPCMultistepScheduler,
    timestep: torch.Tensor,
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float, 
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool,
):  
    # # type
    # model_output = model_output.to(torch.float32)
    # latents = latents.to(torch.float32)
    # if prev_sample is not None:
    #   prev_sample = prev_sample.to(torch.float32)

    # sigma & timestep
    index = scheduler.index_for_timestep(timestep) # (steps,)
    sigmas = scheduler.sigmas.to(model_output.device) # (steps+1,)

    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    delta_t = sigma - sigmas[index + 1]

    # sde logic
    prev_sample_mean = latents + dsigma * model_output
    pred_original_sample = latents - sigma * model_output


    # if args.is_decay:
    #     sigma_ = min(sigma, 0.92)
    #     eta = torch.sqrt(sigma / (1 - sigma_))*0.7

    # noise control signal
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver:
        score_estimate = -(latents-pred_original_sample*(1 - sigma))/sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if sde_solver and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 

        

    if grpo:
        if eta == 0.0:
            log_prob = torch.zeros(prev_sample_mean.shape[0], dtype=torch.float32).to(prev_sample_mean.device)
        else:
            # log prob of prev_sample given prev_sample_mean and std_dev_t
            log_prob = (
                -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
            )
            - math.log(std_dev_t)- torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))

            # mean along all but batch dimension
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        log_prob = torch.zeros(prev_sample_mean.shape[0], dtype=torch.float32).to(prev_sample_mean.device)
        return prev_sample_mean,pred_original_sample,log_prob


def wani2v_prepare_latents_with_gt_video(
    self,
    image: torch.Tensor, # BCHW
    video: torch.Tensor, # BCTHW
    timestep : torch.Tensor, #(1,)
    batch_size: int,
    num_channels_latents: int = 16,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    last_image: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
    latent_height = height // self.vae_scale_factor_spatial
    latent_width = width // self.vae_scale_factor_spatial

    shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

    latents_mean = (
        torch.tensor(self.vae.config.latents_mean)
        .view(1, self.vae.config.z_dim, 1, 1, 1)
        .to(device, dtype)
    )
    latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
        device, dtype
    )

    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    # ------------------------------- get latents from  gt_video ----------------------------------------
    # if latents is None:
    #     latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    # else:
    #     latents = latents.to(device=device, dtype=dtype)

    # vae encode
    video_latents = retrieve_latents(self.vae.encode(video), sample_mode="argmax")
    video_latents = video_latents.to(dtype)
    video_latents = (video_latents - latents_mean) * latents_std # BCTHW
    assert video_latents.shape == shape , f"shape {shape} != video_latents's shape {video_latents.shape}"

    # add noise
    assert timestep in self.scheduler.timesteps , f"your train_timestep{timestep} not in scheduler's timesteps {self.scheduler.timesteps}"
    noise = torch.randn_like(video_latents).to(device)
    latents = self.scheduler.add_noise(video_latents,noise,timestep) # BCTHW
    

    # ------------------------------- get latents from  gt_video ----------------------------------------


    image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]

    if self.config.expand_timesteps:
        video_condition = image

    elif last_image is None:
        video_condition = torch.cat(
            [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
        )
    else:
        last_image = last_image.unsqueeze(2)
        video_condition = torch.cat(
            [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width), last_image],
            dim=2,
        )
    video_condition = video_condition.to(device=device, dtype=self.vae.dtype)



    if isinstance(generator, list):
        latent_condition = [
            retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
        ]
        latent_condition = torch.cat(latent_condition)
    else:
        latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
        latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

    latent_condition = latent_condition.to(dtype)
    latent_condition = (latent_condition - latents_mean) * latents_std

    if self.config.expand_timesteps:
        first_frame_mask = torch.ones(
            1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device
        )
        first_frame_mask[:, :, 0] = 0
        return latents, latent_condition, first_frame_mask

    mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)

    if last_image is None:
        mask_lat_size[:, :, list(range(1, num_frames))] = 0
    else:
        mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
    first_frame_mask = mask_lat_size[:, :, 0:1]
    first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
    mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
    mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
    mask_lat_size = mask_lat_size.transpose(1, 2)
    mask_lat_size = mask_lat_size.to(latent_condition.device)

    return latents, torch.concat([mask_lat_size, latent_condition], dim=1)



@torch.no_grad()
def wani2v_sample_with_logprob(
    self: WanImageToVideoPipeline,
    args,
    image: PipelineImageInput,
    video:List[Image.Image],
    train_timestep_idx:int,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    guidance_scale_2: Optional[float] = None,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    image_embeds: Optional[torch.Tensor] = None,
    last_image: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "np",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    determistic: bool = False,
    kl_reward: float = 0.0,
    return_pixel_log_prob: bool = False,
    device = "cuda",
    dtype = torch.bfloat16,
):
    r"""
    The call function to the pipeline for generation.

    Args:
        image (`PipelineImageInput`):
            The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        height (`int`, defaults to `480`):
            The height of the generated video.
        width (`int`, defaults to `832`):
            The width of the generated video.
        num_frames (`int`, defaults to `81`):
            The number of frames in the generated video.
        num_inference_steps (`int`, defaults to `50`):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, defaults to `5.0`):
            Guidance scale as defined in [Classifier-Free Diffusion
            Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
            of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
            `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
            the text `prompt`, usually at the expense of lower image quality.
        guidance_scale_2 (`float`, *optional*, defaults to `None`):
            Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
            `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
            and the pipeline's `boundary_ratio` are not None.
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
            generation deterministic.
        latents (`torch.Tensor`, *optional*):
            Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor is generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
            provided, text embeddings are generated from the `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
            provided, text embeddings are generated from the `negative_prompt` input argument.
        image_embeds (`torch.Tensor`, *optional*):
            Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided,
            image embeddings are generated from the `image` input argument.
        output_type (`str`, *optional*, defaults to `"np"`):
            The output format of the generated image. Choose between `PIL.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
        attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
            A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
            each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
            DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
            list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        max_sequence_length (`int`, defaults to `512`):
            The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
            truncated. If the prompt is shorter, it will be padded to this length.

    Examples:

    Returns:
        [`~WanPipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
            the first element is a list with the generated images and the second element is a list of `bool`s
            indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
    """

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds,
        negative_prompt_embeds,
        image_embeds,
        callback_on_step_end_tensor_inputs,
        guidance_scale_2,
    )

    if num_frames % self.vae_scale_factor_temporal != 1:
        print(
            f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
        )
        num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
    num_frames = max(num_frames, 1)

    if self.config.boundary_ratio is not None and guidance_scale_2 is None:
        guidance_scale_2 = guidance_scale

    self._guidance_scale = guidance_scale
    self._guidance_scale_2 = guidance_scale_2
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    device = device

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )# prompt_embeds torch.Size([1, 512, 4096])  negative_prompt_embeds torch.Size([1, 512, 4096])

    # Encode image embedding
    transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    # only wan 2.1 i2v transformer accepts image_embeds 不执行
    if self.transformer is not None and self.transformer.config.image_dim is not None:
        if image_embeds is None:
            if last_image is None:
                image_embeds = self.encode_image(image, device)
            else:
                image_embeds = self.encode_image([image, last_image], device)
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    # ===========> start timestep index
    start_idx = train_timestep_idx
    train_timestep = self.scheduler.timesteps[start_idx:start_idx+1]
    timesteps = self.scheduler.timesteps[start_idx:] # torch.Size([samplestep])

    # 5. Prepare latent variables
    num_channels_latents = self.vae.config.z_dim
    # image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32) # torch.Size([1, 3, height, width])
    # if last_image is not None:
    #     last_image = self.video_processor.preprocess(last_image, height=height, width=width).to(
    #         device, dtype=torch.float32
    #     )

    # ------------------------ prepare latents from gt_video ----------------------
    video = self.video_processor.preprocess(video, height=height, width=width).to(device,transformer_dtype)  # TCHW
    video = video.unsqueeze(0).permute(0,2,1,3,4) # BCTHW
    image = video[:,:,0,:,:] # BCHW

    latents_outputs = wani2v_prepare_latents_with_gt_video(
        self,
        image,# BCHW
        video,# BCTHW
        train_timestep,
        batch_size * num_videos_per_prompt,
        num_channels_latents,
        height,
        width,
        num_frames,
        torch.float32,
        device,
        generator,
        latents,
        last_image,
    )
    # ------------------------ prepare latents from gt_video ----------------------


    # latents_outputs = self.prepare_latents(
    #     image,
    #     batch_size * num_videos_per_prompt,
    #     num_channels_latents,
    #     height,
    #     width,
    #     num_frames,
    #     torch.float32,
    #     device,
    #     generator,
    #     latents,
    #     last_image,
    # )
    if self.config.expand_timesteps:
        # wan 2.2 5b i2v use firt_frame_mask to mask timesteps
        # latents BCTHW ; condition BC1HW ; first_frame_mask B1THW --> allzero [:,:,:1], allones [:,:,1:]
        latents, condition, first_frame_mask = latents_outputs
    else:
        # latents BCTHW ; condition BCTHW
        latents, condition = latents_outputs


    # ===========> record latents
    all_latents = [latents]
    all_log_probs = []
    all_kl = []

    # 6. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)

    if self.config.boundary_ratio is not None:
        boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
    else:
        boundary_timestep = None

    # ----------------------------------- denoise ------------------------
    with self.progress_bar(total=len(timesteps)) as progress_bar:
        for i, t in enumerate(timesteps):

            if self.interrupt:
                continue

            self._current_timestep = t


            if boundary_timestep is None or t >= boundary_timestep:
                # wan2.1 or high-noise stage in wan2.2
                current_model = self.transformer
                current_guidance_scale = guidance_scale
            else:
                # low-noise stage in wan2.2
                current_model = self.transformer_2
                current_guidance_scale = guidance_scale_2

            if self.config.expand_timesteps:
                latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents # BCTHW torch.Size([1, 48, 13, 68, 50])
                latent_model_input = latent_model_input.to(transformer_dtype)

                # seq_len: num_latent_frames * (latent_height // patch_size) * (latent_width // patch_size)
                temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten() # T*H/p*W/p
                # batch_size, seq_len
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1) # B,T*H/p*W/p
            else:
                latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype) # B,C_image+ C_condition,THW
                timestep = t.expand(latents.shape[0]) # B,1

            with torch.no_grad(),torch.autocast("cuda", torch.bfloat16):
                # forward
                with current_model.cache_context("cond"):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

            # sde sampler
            latents, pred_original, log_prob = sde_step_with_logprob(
                scheduler = self.scheduler,
                timestep = t,
                model_output = noise_pred,
                latents = latents.to(torch.float32),
                eta = args.eta,
                prev_sample = None,
                grpo = True,
                sde_solver = True,
            )
            all_latents.append(latents.to(transformer_dtype))
            all_log_probs.append(log_prob)


            # # compute the previous noisy sample x_t -> x_t-1
            # latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0] # BCTHW

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            # if XLA_AVAILABLE:
            #     xm.mark_step()

    self._current_timestep = None

    if self.config.expand_timesteps:
        latents = (1 - first_frame_mask) * condition + first_frame_mask * pred_original


    if not output_type == "latent":
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
    else:
        video = latents

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (video,)

    # return WanPipelineOutput(frames=video)


    all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, c,t,h,w)
    all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
    return video, all_latents, all_log_probs, prompt_embeds, negative_prompt_embeds, condition

@torch.no_grad()
def wani2v_sample_with_logprob_fast(
    self: WanImageToVideoPipeline,
    args,
    image: PipelineImageInput,
    video:List[Image.Image],
    train_timestep_idx:int,
    sde_idxs:List[int],
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    guidance_scale_2: Optional[float] = None,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    image_embeds: Optional[torch.Tensor] = None,
    last_image: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "np",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    determistic: bool = False,
    kl_reward: float = 0.0,
    return_pixel_log_prob: bool = False,
    device = "cuda",
    dtype = torch.bfloat16,
):
    r"""
    The call function to the pipeline for generation.

    Args:
        image (`PipelineImageInput`):
            The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        height (`int`, defaults to `480`):
            The height of the generated video.
        width (`int`, defaults to `832`):
            The width of the generated video.
        num_frames (`int`, defaults to `81`):
            The number of frames in the generated video.
        num_inference_steps (`int`, defaults to `50`):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, defaults to `5.0`):
            Guidance scale as defined in [Classifier-Free Diffusion
            Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
            of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
            `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
            the text `prompt`, usually at the expense of lower image quality.
        guidance_scale_2 (`float`, *optional*, defaults to `None`):
            Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
            `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
            and the pipeline's `boundary_ratio` are not None.
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
            generation deterministic.
        latents (`torch.Tensor`, *optional*):
            Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor is generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
            provided, text embeddings are generated from the `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
            provided, text embeddings are generated from the `negative_prompt` input argument.
        image_embeds (`torch.Tensor`, *optional*):
            Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided,
            image embeddings are generated from the `image` input argument.
        output_type (`str`, *optional*, defaults to `"np"`):
            The output format of the generated image. Choose between `PIL.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
        attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
            A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
            each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
            DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
            list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        max_sequence_length (`int`, defaults to `512`):
            The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
            truncated. If the prompt is shorter, it will be padded to this length.

    Examples:

    Returns:
        [`~WanPipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
            the first element is a list with the generated images and the second element is a list of `bool`s
            indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
    """

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds,
        negative_prompt_embeds,
        image_embeds,
        callback_on_step_end_tensor_inputs,
        guidance_scale_2,
    )

    if num_frames % self.vae_scale_factor_temporal != 1:
        print(
            f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
        )
        num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
    num_frames = max(num_frames, 1)

    if self.config.boundary_ratio is not None and guidance_scale_2 is None:
        guidance_scale_2 = guidance_scale

    self._guidance_scale = guidance_scale
    self._guidance_scale_2 = guidance_scale_2
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    device = device

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )# prompt_embeds torch.Size([1, 512, 4096])  negative_prompt_embeds torch.Size([1, 512, 4096])

    # Encode image embedding
    transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    # only wan 2.1 i2v transformer accepts image_embeds 不执行
    if self.transformer is not None and self.transformer.config.image_dim is not None:
        if image_embeds is None:
            if last_image is None:
                image_embeds = self.encode_image(image, device)
            else:
                image_embeds = self.encode_image([image, last_image], device)
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    # ===========> start timestep index
    start_idx = train_timestep_idx
    train_timestep = self.scheduler.timesteps[start_idx:start_idx+1]
    sde_timestep = self.scheduler.timesteps[sde_idxs[0]:sde_idxs[0]+len(sde_idxs)]
    timesteps = self.scheduler.timesteps[start_idx:] # torch.Size([samplestep])

    # 5. Prepare latent variables
    num_channels_latents = self.vae.config.z_dim
    # image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32) # torch.Size([1, 3, height, width])
    # if last_image is not None:
    #     last_image = self.video_processor.preprocess(last_image, height=height, width=width).to(
    #         device, dtype=torch.float32
    #     )

    # ------------------------ prepare latents from gt_video ----------------------
    video = self.video_processor.preprocess(video, height=height, width=width).to(device,transformer_dtype)  # TCHW
    video = video.unsqueeze(0).permute(0,2,1,3,4) # BCTHW
    image = video[:,:,0,:,:] # BCHW

    latents_outputs = wani2v_prepare_latents_with_gt_video(
        self,
        image,# BCHW
        video,# BCTHW
        train_timestep,
        batch_size * num_videos_per_prompt,
        num_channels_latents,
        height,
        width,
        num_frames,
        torch.float32,
        device,
        generator,
        latents,
        last_image,
    )
    # ------------------------ prepare latents from gt_video ----------------------


    # latents_outputs = self.prepare_latents(
    #     image,
    #     batch_size * num_videos_per_prompt,
    #     num_channels_latents,
    #     height,
    #     width,
    #     num_frames,
    #     torch.float32,
    #     device,
    #     generator,
    #     latents,
    #     last_image,
    # )
    if self.config.expand_timesteps:
        # wan 2.2 5b i2v use firt_frame_mask to mask timesteps
        # latents BCTHW ; condition BC1HW ; first_frame_mask B1THW --> allzero [:,:,:1], allones [:,:,1:]
        latents, condition, first_frame_mask = latents_outputs
    else:
        # latents BCTHW ; condition BCTHW
        latents, condition = latents_outputs


    # ===========> record latents
    all_latents = [latents]
    all_log_probs = []
    all_kl = []

    # 6. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)

    if self.config.boundary_ratio is not None:
        boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
    else:
        boundary_timestep = None

    # ----------------------------------- denoise ------------------------
    with self.progress_bar(total=len(timesteps)) as progress_bar:
        for i, t in enumerate(timesteps):

            if self.interrupt:
                continue

            self._current_timestep = t


            if boundary_timestep is None or t >= boundary_timestep:
                # wan2.1 or high-noise stage in wan2.2
                current_model = self.transformer
                current_guidance_scale = guidance_scale
            else:
                # low-noise stage in wan2.2
                current_model = self.transformer_2
                current_guidance_scale = guidance_scale_2

            if self.config.expand_timesteps:
                latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents # BCTHW torch.Size([1, 48, 13, 68, 50])
                latent_model_input = latent_model_input.to(transformer_dtype)

                # seq_len: num_latent_frames * (latent_height // patch_size) * (latent_width // patch_size)
                temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten() # T*H/p*W/p
                # batch_size, seq_len
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1) # B,T*H/p*W/p
            else:
                latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype) # B,C_image+ C_condition,THW
                timestep = t.expand(latents.shape[0]) # B,1

            with torch.no_grad(),torch.autocast("cuda", torch.bfloat16):
                # forward
                with current_model.cache_context("cond"):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

            # sde sampler
            if t in sde_timestep:
                latents, pred_original, log_prob = sde_step_with_logprob(
                    scheduler = self.scheduler,
                    timestep = t,
                    model_output = noise_pred,
                    latents = latents.to(torch.float32),
                    eta = args.eta,
                    prev_sample = None,
                    grpo = True,
                    sde_solver = True,
                )
            else:
                latents, pred_original, log_prob = sde_step_with_logprob(
                    scheduler = self.scheduler,
                    timestep = t,
                    model_output = noise_pred,
                    latents = latents.to(torch.float32),
                    eta = args.eta,
                    prev_sample = None,
                    grpo = False,
                    sde_solver = False,
                )                
            all_latents.append(latents.to(transformer_dtype))
            all_log_probs.append(log_prob)


            # # compute the previous noisy sample x_t -> x_t-1
            # latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0] # BCTHW

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            # if XLA_AVAILABLE:
            #     xm.mark_step()

    self._current_timestep = None

    if self.config.expand_timesteps:
        latents = (1 - first_frame_mask) * condition + first_frame_mask * pred_original


    if not output_type == "latent":
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
    else:
        video = latents

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (video,)

    # return WanPipelineOutput(frames=video)


    all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, c,t,h,w)
    all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
    return video, all_latents, all_log_probs, prompt_embeds, negative_prompt_embeds, condition



def wani2v_train_onestep_with_logprob(
    self: WanImageToVideoPipeline,
    args,
    timestep: torch.Tensor, # B
    latents: torch.Tensor, # BCTHW
    prev_latents: torch.Tensor, # BCTHW
    condition: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    guidance_scale: float = 5.0,
    device = "cuda",
    
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale_2: Optional[float] = None,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    image_embeds: Optional[torch.Tensor] = None,
    last_image: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "np",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    determistic: bool = False,
    kl_reward: float = 0.0,
    return_pixel_log_prob: bool = False,
):
    shape = latents.shape
    transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
    device = device

    # wan2.2 5b first_frame_mask
    if self.config.expand_timesteps:
        first_frame_mask = torch.ones(
                1, 1, shape[2],  shape[3], shape[4], dtype=transformer_dtype, device=device
            )
        first_frame_mask[:, :, 0] = 0


    if self.config.boundary_ratio is not None:
        boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
    else:
        boundary_timestep = None
  
    t = timestep
    self._current_timestep = t
    image_embeds = None


    if boundary_timestep is None or t >= boundary_timestep:
        # wan2.1 or high-noise stage in wan2.2
        current_model = self.transformer
        current_guidance_scale = guidance_scale
    else:
        # low-noise stage in wan2.2
        current_model = self.transformer_2
        current_guidance_scale = guidance_scale_2

    if self.config.expand_timesteps:
        latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents # BCTHW torch.Size([1, 48, 13, 68, 50])
        latent_model_input = latent_model_input.to(transformer_dtype)

        # seq_len: num_latent_frames * (latent_height // patch_size) * (latent_width // patch_size)
        temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten() # T*H/p*W/p
        # batch_size, seq_len
        timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1) # B,T*H/p*W/p
    else:
        latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype) # B,C_image+ C_condition,THW
        timestep = t.expand(latents.shape[0]) # B,1

    with torch.autocast("cuda", torch.bfloat16):
        # forward
        with current_model.cache_context("cond"):
            noise_pred = current_model(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

        if self.do_classifier_free_guidance:
            with current_model.cache_context("uncond"):
                noise_uncond = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

    # sde sampler
    latents, pred_original, log_prob = sde_step_with_logprob(
        scheduler = self.scheduler,
        timestep = t,
        model_output = noise_pred,
        latents = latents.to(torch.float32),
        eta = args.eta,
        prev_sample = prev_latents.to(torch.float32),
        grpo = True,
        sde_solver = True,
    )



    return log_prob
