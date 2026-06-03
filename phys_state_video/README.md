# phys-state-video

`phys-state-video` is a standalone MVP for object-state-conditioned future video generation.

This package implements the project structure discussed earlier:

- pseudo object-state extraction from tracked annotations or perception outputs
- future object-state prediction with uncertainty-aware rollout
- confidence-aware consistency projection
- conversion from object states to spatial condition maps and memory tokens
- a minimal trainable state-conditioned video model
- training and inference entrypoints

The code is intentionally backbone-agnostic. The default model is a small PyTorch baseline that makes the pipeline runnable. Later, the same state extraction, predictor, projection, and conditioning modules can be attached to a larger base model such as Wan or VACE.

## Dataset format

The training scripts expect `.npz` episode files with these keys:

- `context_frames`: `[K, 3, H, W]`
- `future_frames`: `[T, 3, H, W]`
- `context_states`: `[K, N, D]`
- `future_states`: `[T, N, D]`
- `context_boxes`: `[K, N, 4]`
- `future_boxes`: `[T, N, 4]`
- `appearance`: `[N, A]`
- `camera`: `[K, C]`

Optional metadata:

- `prompt`: UTF-8 string saved under a sidecar `.json`, or omitted

`D` defaults to `10` with the layout:

1. `center_x`
2. `center_y`
3. `relative_depth`
4. `log_scale`
5. `vel_x`
6. `vel_y`
7. `depth_vel`
8. `visibility`
9. `existence`
10. `confidence`

## Environment

The default system Python in this workspace does not include PyTorch. Use the existing `wan` environment when running the model code:

```bash
/data/gaoya/miniconda3/envs/wan/bin/python -m pip install -e /home/gaoya/Code_Video/phys_state_video
```

## Quick start

Train the future state predictor:

```bash
/data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/phys_state_video/scripts/train_predictor.py \
  --data /path/to/episodes \
  --output /path/to/checkpoints/predictor.pt
```

Train the state-conditioned video model:

```bash
/data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/phys_state_video/scripts/train_adapter.py \
  --data /path/to/episodes \
  --output /path/to/checkpoints/adapter.pt
```

Run inference:

```bash
/data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/phys_state_video/scripts/run_inference.py \
  --episode /path/to/episode.npz \
  --predictor /path/to/checkpoints/predictor.pt \
  --adapter /path/to/checkpoints/adapter.pt \
  --output /path/to/output_dir
```

## Wan state predictor v2

The repository now also contains a `wan_state_v2_latent_time` predictor path. This version keeps the predictor fully on the Wan latent time axis instead of resampling latents back to the original video frame count.

### v2 predictor design

- `context_frames ∈ R^{B×K×3×H×W}` are encoded into raw Wan latents
- predictor input becomes `context_latents_raw ∈ R^{B×L_ctx×C_w×H_w×W_w}`
- predictor output becomes `future_state_latents ∈ R^{B×L_future×D_s}`
- explicit physics heads are grouped into:
  - `geom`: `center_x, center_y, depth, log_scale`
  - `motion`: `vel_x, vel_y, depth_vel`
  - `vis`: `visibility, existence, confidence`
- training uses staged optimization:
  - `context_only`
  - `future_only`
  - `joint_finetune`

### v2 latent sources

There are two supported ways to build predictor inputs:

- `mock latent`
  - Uses `MockLatentExtractor`
  - Does not require Wan CUDA runtime or Wan checkpoints
  - Intended for unit tests, CPU smoke tests, and training loop validation
- `real Wan latent`
  - Uses `WanLatentExtractor.encode_context_frames_raw()`
  - Requires Wan VAE checkpoint access
  - Intended for real latent-time training once the local Wan runtime is available

### v2 smoke path on this machine

Because the active `wan` environment currently cannot initialize CUDA for the installed PyTorch build, the runnable end-to-end smoke path in this workspace is:

- `toy dataset -> train_predictor_wan_state_v2.py --latent-source mock`
- save checkpoint
- load checkpoint
- `run_inference_wan_state_v2.py --latent-source auto`
- optional `export_wan_state_condition_dataset.py --future-state-source wan_predictor`

Train `wan_state_v2_latent_time` with the mock latent path:

```bash
/data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/phys_state_video/scripts/train_predictor_wan_state_v2.py \
  --data /path/to/episodes \
  --output /path/to/checkpoints/predictor_v2.pt \
  --device cpu \
  --latent-source mock
```

Run v2 inference from the saved checkpoint:

```bash
/data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/phys_state_video/scripts/run_inference_wan_state_v2.py \
  --episode /path/to/episode.npz \
  --predictor /path/to/checkpoints/predictor_v2.pt \
  --output /path/to/output_dir \
  --device cpu \
  --latent-source auto
```

Export `state_tokens` from a v2 checkpoint:

```bash
/data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/phys_state_video/scripts/export_wan_state_condition_dataset.py \
  --episodes /path/to/episodes \
  --output /path/to/wan_state_condition_predictor_v2 \
  --future-state-source wan_predictor \
  --predictor /path/to/checkpoints/predictor_v2.pt \
  --predictor-latent-source auto \
  --device cpu
```

## Wan state-condition export

To connect `phys_state_video` episodes with the external Wan `state_condition` interface, export per-sample bundles containing:

- `input_image.png`: the first context frame
- `state_condition.npz`: Wan-readable condition payload
- `meta.json`: shape/source metadata
- `prompt.txt`: prompt text
- `manifest.jsonl`: dataset manifest

Ground-truth future states can be exported directly:

```bash
/data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/phys_state_video/scripts/export_wan_state_condition_dataset.py \
  --episodes /path/to/episodes \
  --output /path/to/wan_state_condition_gt \
  --future-state-source ground_truth
```

When a trained `wan_state_v1` predictor checkpoint is available, export predictor-produced `state_tokens`:

```bash
/data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/phys_state_video/scripts/export_wan_state_condition_dataset.py \
  --episodes /path/to/episodes \
  --output /path/to/wan_state_condition_predictor \
  --future-state-source wan_predictor \
  --predictor /path/to/wan_state_predictor.pt \
  --wan-ckpt-dir /path/to/wan_ckpt_dir
```

## Current local checkpoint note

This workspace currently has a local `Wan2.2-TI2V-5B` checkpoint under `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`. It does not currently include a local `i2v-A14B` checkpoint, so the `Wan` smoke test that can be run end-to-end here is limited to the generic external `state_condition` path rather than the full local `clean_prefix_latents + WanI2V` continuation bridge.

## Current CUDA note

`nvidia-smi` is healthy on this machine, but the active `wan` environment currently uses `torch 2.11.0+cu130`, while `torch.cuda.is_available()` reports a driver/runtime mismatch and returns `False`. In practice this means:

- CPU predictor smoke tests work
- mock-latent v2 training/inference works
- local Wan adapter training is blocked
- real Wan latent extraction and Wan sampling should be treated as environment-blocked until the PyTorch/CUDA build is aligned with the installed driver stack
