# TextOCVP Stage 1 in V-JEPA and Wan VAE Spaces

## Decision

Both feature-space variants are technically valid experiments, but they test
different hypotheses and should not replace pixel-space SAVi before objectness
is measured.

- V-JEPA space is the stronger object-centric candidate. Its frozen tokens are
  semantically and temporally structured, so slot masks are more likely to
  follow persistent entities instead of colors or local textures.
- Wan VAE space is the stronger integration candidate. The reconstructed
  features already use Wan 2.2's normalized latent convention, so learned slots
  can be consumed by a later Wan adapter without an additional RGB encoder.
- Wan VAE latents are not explicitly object-centric. A low latent reconstruction
  loss can be achieved with spatial partitions that do not correspond to
  objects. Mask overlays, temporal consistency, and slot-ablation tests are
  required before treating this branch as successful.

The frozen tokenizers are never added to the optimizer or checkpoint. Only the
SAVi-style slot initializer, Slot Attention, temporal transition, positional
MLP, and per-slot feature decoder are trained.

## Data Flow

### V-JEPA

```text
RGB [B,10,3,216,384]
  -> replicate-pad height to 224 and ImageNet normalize
  -> frozen V-JEPA2 ViT-G, patch=16, tubelet=2
  -> tokens [B,5,14,24,1408]
  -> recurrent Slot Attention [B,5,8,256]
  -> per-slot spatial feature decoder
  -> masks [B,5,8,14,24,1]
  -> reconstructed tokens [B,5,14,24,1408]
  -> loss = feature MSE + 0.1 * cosine distance
```

### Wan VAE

```text
RGB [B,9,3,216,384]
  -> replicate-pad height to 224 and map [0,1] to [-1,1]
  -> frozen Wan 2.2 causal VAE with official channel mean/std
  -> latents [B,3,14,24,48]
  -> recurrent Slot Attention [B,3,8,256]
  -> per-slot spatial latent decoder
  -> masks [B,3,8,14,24,1]
  -> reconstructed latents [B,3,14,24,48]
  -> loss = normalized latent MSE
```

Wan 2.2's temporal VAE consumes `4n+1` frames. Nine frames are used because ten
frames would silently leave the final frame outside the encoder's causal chunks.

## Reused Components

- Dataset and hand-off split: `TextOCVP-master/src/data/Stage1Indexed.py`
- Slot Attention: `TextOCVP-master/src/models/Blocks/attention.py`
- Learned-random slot initialization and transformer transition: TextOCVP Stage 1
- Fixed PyBullet/Kubric/mixed indices under `/data/gaoya/AAA_test_video/0623_savi/indices`
- Step validation, source-specific metrics, overfit monitor, W&B logging, and
  500-step checkpoint semantics from the existing pixel-space runner

## Recommended Execution

1. Run one-GPU two-step shape smokes separately. Do not run both frozen models
   on a GPU already occupied by the pixel-space comparison.
2. Run 100 optimizer steps on the mixed split and inspect mask entropy, minimum
   slot usage, validation loss, and overlay videos.
3. Run 500 steps only if multiple slots are used and masks are temporally stable.
4. Compare pixel, V-JEPA, and VAE branches on identical hand-off videos using
   foreground IoU when masks exist, temporal mask consistency, slot utilization,
   reconstruction loss normalized to step 0, and slot-ablation sensitivity.
5. Promote the best branch to 1000 epochs only after the objectness gate passes.

## Commands

V-JEPA two-step smoke on GPU 4:

```bash
GPU_IDS=4 DATASET_MODE=mixed \
OUTPUT_DIR=/data/gaoya/AAA_test_video/0623_savi/outputs/vjepa_space_shape_smoke \
bash /home/gaoya/Code_Video/TextOCVP-PyBullet-smoke/run_stage1_vjepa_space.sh \
  --per-gpu-batch-size 1 \
  --effective-batch-size 1 \
  --max-optimizer-steps 2 \
  --validation-frequency-steps 1 \
  --max-train-samples 4 \
  --max-valid-samples 2 \
  --num-workers 0 \
  --disable-wandb
```

Wan VAE two-step smoke on GPU 5:

```bash
GPU_IDS=5 DATASET_MODE=mixed \
OUTPUT_DIR=/data/gaoya/AAA_test_video/0623_savi/outputs/vae_space_shape_smoke \
bash /home/gaoya/Code_Video/TextOCVP-PyBullet-smoke/run_stage1_vae_space.sh \
  --per-gpu-batch-size 1 \
  --effective-batch-size 1 \
  --max-optimizer-steps 2 \
  --validation-frequency-steps 1 \
  --max-train-samples 4 \
  --max-valid-samples 2 \
  --num-workers 0 \
  --disable-wandb
```

Formal four-GPU commands are the same launchers without smoke overrides. The
default formal effective batch is 16 and validation/checkpoint interval is 500
optimizer steps.
