# V-JEPA Guidance Prototype

This directory is a standalone prototype for the training-free idea:

1. Run normal video diffusion sampling with a single sample.
2. At a small number of selected denoising steps, recover a preview `x0`.
3. Decode a low-resolution preview video from that `x0`.
4. Compute a V-JEPA past-to-future masked predictive surprise energy.
5. Apply one small latent correction step with `-∇E_JEPA`.

The code here is intentionally independent from:

- `/home/gaoya/Code_Video/vjepa2-main`
- `/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main`
- `/home/gaoya/Code_Video/WMReward-main1/WMReward-main`

The goal is to validate feasibility first, without forking or rewriting the
upstream repositories.

## How The Two Repos Are Related

`WMReward` is already an inference-time guidance method built on top of
`V-JEPA 2`.

- `WMReward-main/utils.py` reuses V-JEPA encoder / predictor logic and defines
  the masked predictive surprise used as a physics-oriented reward.
- `WMReward-main/compute_wmreward.py` loads pretrained V-JEPA models and
  computes a sliding-window surprise score for a video.
- `WMReward-main/generate_magi1.py` and
  `WMReward-main/generator_i2v_multinode.py` pass guidance settings into a
  video generator.

So the relationship is direct:

- `vjepa2-main` is the upstream world-model codebase.
- `WMReward-main` is a downstream inference-time physics-alignment method that
  uses V-JEPA as the reward / energy model.

## What Is Different In This Prototype

This prototype is not a BoN or rejection method.

- Only one video is sampled.
- No re-ranking across multiple candidates.
- The V-JEPA signal is injected during denoising, not after generation.
- The correction acts on the latent state, not on the final decoded video.

Compared with the public `WMReward` wrapper, this prototype is closer to
"energy guidance" than "selection":

- `WMReward` public scripts expose V-JEPA-guided generation for MAGI, but the
  local MAGI submodule in this workspace is empty, so that path is not
  reproducible here.
- Your local Wan runtime is available and exposes a denoising loop in the
  upstream repo, which is the more practical integration target.

## Feasibility Assessment

The proposal is technically feasible.

Why it is feasible:

- Official Wan I2V / TI2V sampling loops explicitly iterate over denoising
  timesteps and keep the current latent in Python.
- Wan schedulers expose enough information to reconstruct a clean preview
  sample with `x0_pred = x_t - sigma_t * model_output` for
  `flow_prediction`.
- Wan VAE decoding is callable inside the loop, so a preview branch can decode
  a lower-resolution latent.
- V-JEPA surprise is differentiable with respect to the decoded preview video,
  and therefore with respect to the latent if the VAE decode path is kept in
  graph.

Main risks:

- VAE decode inside the loop is expensive. This is why the preview branch
  should use only 5-8 selected timesteps and spatially downsampled latents.
- V-JEPA surprise was designed as a video-level plausibility signal, not as a
  local corrective oracle. The gradient may be weak or noisy on very early
  timesteps.
- The best guidance steps are likely mid-to-late denoising steps, not the
  earliest ones.
- Guidance can improve gross physical plausibility while slightly hurting
  sharpness or prompt faithfulness if the step size is too large.

Expected practical starting point:

- Select 6 denoising steps uniformly from the middle 60 percent of the schedule.
- Use a preview latent downsample factor of `2`.
- Use V-JEPA `vitg` or `vitg384`.
- Keep a small normalized latent step such as `0.01` to `0.05`.

## Files

- `vjepa_surprise.py`
  Differentiable V-JEPA surprise energy on a decoded video tensor.
- `wan_latent_guidance.py`
  Utilities to reconstruct `x0`, decode a preview video, choose guidance
  steps, and apply one latent correction step.
- `wan_ti2v_vjepa.py`
  Standalone Wan 2.2 TI2V/I2V wrapper that reuses the PhaseLock-style
  sampling-loop structure, but swaps the guidance rule to V-JEPA energy
  guidance.
- `wan21_t2v_1_3b_vjepa.py`
  Standalone Wan 2.1 T2V 1.3B Diffusers wrapper with the same V-JEPA latent
  guidance idea. This variant is text-to-video only because the 1.3B
  checkpoint is not an image-conditioned model.

## Integration Target In Wan

The cleanest current hook point is the official Wan TI2V I2V loop in:

- `/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main/wan/textimage2video.py`

Relevant stages in that loop:

1. predict conditional / unconditional model outputs
2. form guided model output
3. reconstruct preview `x0`
4. optionally decode low-resolution preview and compute V-JEPA energy
5. update latent with one small correction
6. run the scheduler step

For a first pass, apply V-JEPA correction before the scheduler step.

## Suggested First Experiment

1. Keep Wan sampling unchanged except for 6 guidance calls.
2. Use `guide_scale_cfg` exactly as before; do not replace CFG.
3. Add a second small V-JEPA step on top of CFG.
4. Log per-step:
   - timestep
   - V-JEPA surprise
   - latent gradient norm
   - preview decode resolution
5. Compare:
   - baseline Wan
   - Wan + V-JEPA guidance
6. Evaluate with your existing physics-focused cases first.

## Minimal Integration Sketch

Inside the Wan I2V loop, after you already formed the CFG-combined model
output:

```python
from vjepa_guidance import (
    VJEPASurpriseEnergy,
    WanVJEPAConfig,
    apply_vjepa_latent_guidance,
    pick_guidance_step_indices,
)

energy = VJEPASurpriseEnergy(model_name="vitg", device="cuda")
cfg = WanVJEPAConfig(
    guidance_steps=6,
    latent_step_size=0.02,
    preview_downsample_factor=2,
    context_frames=8,
    window_size=16,
    stride=4,
)
selected_steps = set(
    pick_guidance_step_indices(
        total_steps=len(timesteps),
        count=cfg.guidance_steps,
        min_step_percent=cfg.min_step_percent,
        max_step_percent=cfg.max_step_percent,
    )
)

for step_idx, t in enumerate(timesteps):
    # ... compute noise_pred from Wan CFG path first ...

    if step_idx in selected_steps:
        latent, stats = apply_vjepa_latent_guidance(
            latent_xt=latent,
            model_output=noise_pred,
            timestep=t,
            scheduler=sample_scheduler,
            vae=self.vae,
            energy_fn=energy,
            config=cfg,
        )
        print(step_idx, int(t.item()), stats)

    latent = sample_scheduler.step(
        noise_pred.unsqueeze(0),
        t,
        latent.unsqueeze(0),
        return_dict=False,
        generator=seed_g,
    )[0].squeeze(0)
```

That keeps the method single-sample and adds only one lightweight correction on
selected steps.

## Minimal Run Command

```bash
CUDA_VISIBLE_DEVICES=0 python3 /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan_ti2v_vjepa.py \
  --ckpt_dir /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \
  --prompt "A ball falls from a platform and bounces on the floor" \
  --image /path/to/first_frame.png \
  --output /data/gaoya/agent-data/outputs/vjepa_guidance/ball_bounce_vjepa.mp4 \
  --sample_steps 50 \
  --vjepa_model vitg \
  --vjepa_guidance_steps 6 \
  --vjepa_latent_step_size 0.02 \
  --offload_model
```

This is the foreground startup command for the current prototype script.

## Wan2.1 1.3B Command

The `Wan-AI-Wan2.1-T2V-1.3B-Diffusers` checkpoint is `T2V` only, so this
script does not accept a first-frame image.

```bash
CUDA_VISIBLE_DEVICES=0,5 /data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan21_t2v_1_3b_vjepa.py \
  --ckpt_dir /data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A ball falls from a platform and bounces on the floor" \
  --output /data/gaoya/agent-data/outputs/vjepa_guidance/wan21_t2v_1_3b_vjepa.mp4 \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --num_inference_steps 50 \
  --guidance_scale 6 \
  --flow_shift 8 \
  --device_id 1 \
  --vjepa_device_id 0 \
  --vjepa_model vith \
  --vjepa_ckpt /data/gaoya/ckpt/VJEPA2/vith.pt
```

For a pure baseline run with default Diffusers sampling:

```bash
CUDA_VISIBLE_DEVICES=0,5 /data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan21_t2v_1_3b_vjepa.py \
  --ckpt_dir /data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A ball falls from a platform and bounces on the floor" \
  --output /data/gaoya/agent-data/outputs/vjepa_guidance/wan21_t2v_1_3b_baseline.mp4 \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --num_inference_steps 50 \
  --guidance_scale 6 \
  --flow_shift 8 \
  --device_id 1 \
  --disable_vjepa_guidance
```

## What Is Not Implemented Here

- End-to-end patching of the official Wan repo
- A launcher that instantiates Wan weights and runs full generation
- MAGI integration, because the local MAGI submodule is currently empty

This is deliberate. The prototype isolates the energy and latent-update logic
first, so you can test the method before committing to a full generator fork.
