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
