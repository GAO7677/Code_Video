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

## Current mainline

As of `2026-06-04`, the recommended mainline is:

- `wan_state_v2_latent_time` predictor training on the Wan latent time axis
- exported `state_tokens ∈ R^{L_future×D_s}` as the video-side condition interface
- a trained Wan state adapter checkpoint as a required dependency for formal Wan inference
- `WanImageToVideoBackend.generate()` with clean prefix latents held fixed and only future latents denoised

The old `wan_state_v1` predictor and older TI2V-first adapter path are still kept for reproducibility, but they are no longer the recommended default path for new experiments.

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
- optional adapter-space alignment can be added during `future_only` and `joint_finetune`
  - freeze a trained Wan state adapter
  - optionally freeze a teacher `wan_state_v2_latent_time` predictor
  - align predictor-produced `future_state_latents` after the adapter encoder

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

### Recommended predictor training path

The recommended predictor training path is now:

1. train a baseline `wan_state_v2_latent_time` predictor
2. train a Wan state adapter from exported `state_tokens`
3. continue predictor training with `--adapter-align-ckpt` and optional `--teacher-predictor`
4. run formal Wan inference through `run_inference_wan_state.py`

The adapter-space alignment stage is intentionally lightweight in the current repo:

- it freezes a trained Wan state adapter encoder
- it compares the encoded `state_context` from the current predictor against a frozen teacher predictor
- it adds `adapter_align_scale * L2(pred_state_context, teacher_state_context)` during non-`context_only` stages

This is not yet full future-latent diffusion supervision, but it directly ties the predictor token space to a trained adapter space and is much stronger than state-head-only supervision.

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

### v2 predictor pipeline summary

From the dataset, each sample starts as:

- `context_frames ∈ R^{B×K×3×H×W}`
- `camera ∈ R^{B×K×C_cam}`
- `context_states ∈ R^{B×K×N×10}`
- `future_states ∈ R^{B×T×N×10}`

For `mock latent`, `context_frames` are temporally compressed by `MockLatentExtractor` into `context_latents_raw ∈ R^{B×L_ctx×C_w×H_w×W_w}`.

For `real Wan latent`, `context_frames` are encoded by `WanLatentExtractor.encode_context_frames_raw()` into the same latent-time shape `context_latents_raw ∈ R^{B×L_ctx×C_w×H_w×W_w}`.

The predictor runs entirely on the Wan latent time axis:

- input: `context_latents_raw ∈ R^{B×L_ctx×C_w×H_w×W_w}`
- input: resampled camera `camera_latent ∈ R^{B×L_ctx×C_cam}`
- output: `context_state_latents ∈ R^{B×L_ctx×D_s}`
- output: `future_state_latents ∈ R^{B×L_future×D_s}`
- output: `context_state_predictions ∈ R^{B×L_ctx×N×10}`
- output: `future_state_predictions ∈ R^{B×L_future×N×10}`

The explicit state head is grouped instead of using one flat head:

- `geom`: 4 dims
- `motion`: 3 dims
- `vis`: 3 dims

Training is staged:

- `context_only`: train grouped state heads using context supervision first
- `future_only`: freeze state heads and train the future latent rollout
- `joint_finetune`: unfreeze all predictor modules and fine-tune jointly

Optional adapter alignment:

- `--teacher-predictor /path/to/predictor_v2_teacher.pt`
- `--adapter-align-ckpt /path/to/trained_state_adapter.pt`
- `--adapter-align-scale 1.0`

This adds an adapter-space alignment loss while keeping the teacher predictor and adapter frozen.

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

The exported bundle layout is:

- `input_image.png`: first context frame
- `state_condition.npz`: usually `state_tokens ∈ R^{L_future×D_s}` for predictor export, or `predicted_states ∈ R^{T×N×10}` for ground-truth export
- `meta.json`: includes `episode_path`, original sample shapes, and latent-step metadata
- `prompt.txt`: prompt text
- `manifest.jsonl`: dataset index

For `wan_state_v2_latent_time`, the export path keeps the predictor on latent time:

- predictor inference produces `future_state_latents ∈ R^{L_future×D_s}`
- export writes them into `state_condition.npz` as `state_tokens`
- `meta.json` stores `context_latent_steps`, `future_latent_steps`, and `temporal_stride`

## Local Wan adapter training

The repository now includes two local state-adapter trainers:

- script: `/home/gaoya/Code_Video/phys_state_video/scripts/train_wan_state_adapter_local.py`
- script: `/home/gaoya/Code_Video/phys_state_video/scripts/train_wan_state_adapter_prefix_local.py`
- helpers: `/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/wan_adapter_training.py`

`train_wan_state_adapter_local.py` is the older TI2V-aligned path:

- it reads `manifest.jsonl` or per-bundle directories under `--state-condition-root`
- it loads `state_condition.npz`
- it follows `meta.json["episode_path"]` back to the original episode `.npz`
- for TI2V training, it builds the supervision video as `first context frame + future_frames`
- it pads the target video to Wan's `4n+1` frame convention by repeating the last frame when needed
- it trains only the Wan `state_adapter` parameters and the model's internal `state_adapter_*` weights
- it saves checkpoints in the format expected by `WanTI2V.load_state_adapter()`

`train_wan_state_adapter_prefix_local.py` is the new recommended prefix-infill-aligned path:

- it reads the same exported `state_tokens` bundles
- it rebuilds the full `context + future` training video
- it encodes the full clip and also separately encodes the clean context prefix
- it keeps the entire prefix latent segment clean
- it only adds noise to future latent steps
- it computes training loss only on future latent steps
- it saves checkpoints in the format expected by `WanI2V.load_state_adapter()`

This second path is important because it matches the semantics of formal prefix infill inference much more closely than the older “clean first frame only” TI2V trainer.

Example command:

```bash
/data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/phys_state_video/scripts/train_wan_state_adapter_local.py \
  --state-condition-root /path/to/wan_state_condition_predictor_v2 \
  --wan-ckpt-dir /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \
  --task ti2v-5B \
  --size 704*1280 \
  --output /path/to/checkpoints/wan_ti2v_state_adapter.pt \
  --device cuda:0
```

Recommended prefix-infill-aligned adapter training command:

```bash
/data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/phys_state_video/scripts/train_wan_state_adapter_prefix_local.py \
  --state-condition-root /path/to/wan_state_condition_predictor_v2 \
  --wan-ckpt-dir /path/to/Wan-I2V-checkpoint \
  --task i2v-A14B \
  --size 480*832 \
  --output /path/to/checkpoints/wan_i2v_prefix_state_adapter.pt \
  --device cuda:0
```

Formal Wan prefix infill inference now requires a trained state adapter checkpoint:

```bash
/data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/phys_state_video/scripts/run_inference_wan_state.py \
  --episode /path/to/episode.npz \
  --predictor /path/to/checkpoints/predictor_v2.pt \
  --wan-ckpt-dir /path/to/Wan-I2V-checkpoint \
  --wan-state-adapter-ckpt /path/to/checkpoints/wan_i2v_prefix_state_adapter.pt \
  --output /path/to/output_dir \
  --predictor-latent-source auto \
  --state-guidance-scale 1.0
```

At inference time, the saved adapter checkpoint can be loaded by:

- `/home/gaoya/Code_Video/phys_state_video/scripts/run_wan_state_condition_bundle.py --state-adapter-ckpt ...`
- `/home/gaoya/Code_Video/phys_state_video/scripts/run_wan_ti2v_state_condition_smoke.py --state-adapter-ckpt ...` as a compatibility alias
- the native `WanTI2V.load_state_adapter(...)` interface

## Formal tests

Formal unit tests now cover:

- v1 predictor helpers and regression cases
- `wan_state_v2_latent_time` predictor shapes and staged losses
- latent-time helper utilities
- local TI2V adapter-training helpers such as bundle discovery, `4n+1` alignment, and checkpoint format checks
- prefix-infill adapter helper utilities such as full-context training-video assembly, prefix masking, and prefix latent overwrite helpers

Run:

```bash
cd /home/gaoya/Code_Video/phys_state_video
/data/gaoya/miniconda3/envs/wan/bin/python -m pytest tests/test_wan_state_predictor.py tests/test_wan_adapter_training.py
```

## Current local checkpoint note

This workspace currently has a local `Wan2.2-TI2V-5B` checkpoint under `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`. It does not currently include a local `i2v-A14B` checkpoint, so the `Wan` smoke test that can be run end-to-end here is limited to the generic external `state_condition` path rather than the full local `clean_prefix_latents + WanI2V` continuation bridge.

## Current CUDA note

`nvidia-smi` is healthy on this machine, but the active `wan` environment currently uses `torch 2.11.0+cu130`, while `torch.cuda.is_available()` reports a driver/runtime mismatch and returns `False`. The machine reports driver `570.124.06` / CUDA `12.8`, while PyTorch in the `wan` env was built for CUDA `13.0`. In practice this means:

- CPU predictor smoke tests work
- mock-latent v2 training/inference works
- local Wan adapter training code for both TI2V and I2V-prefix paths is present but real optimization is blocked in this environment
- real Wan latent extraction and Wan sampling should be treated as environment-blocked until the PyTorch/CUDA build is aligned with the installed driver stack
