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
native RGB: PyBullet [B,10,3,540,960] or Kubric [B,10,3,432,768]
  -> dataset preserves aspect ratio and resizes short side to 438
  -> dataset center crops to [B,10,3,384,384]
  -> ImageNet normalize without an intermediate 216x384 resize
  -> frozen V-JEPA2 ViT-G, patch=16, tubelet=2
  -> tokens [B,5,24,24,1408]
  -> per-token unit-variance normalization after V-JEPA's affine final LayerNorm
  -> official-style feature projector 1408 -> 1408 -> 512
  -> recurrent Slot Attention [B,5,8,512]
  -> per-slot spatial feature decoder
  -> masks [B,5,8,24,24,1]
  -> reconstructed tokens [B,5,24,24,1408]
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

The dedicated PyBullet:Kubric `1:8` V-JEPA launcher uses the full 1200+9600
candidate pool and GPU 5,6:

```bash
bash /home/gaoya/Code_Video/TextOCVP-PyBullet-smoke/run_stage1_vjepa_space_pybullet1_kubric8_gpu56.sh
```

### Historical V-JEPA batch capacity on GPU 5,6

The old direct-feature, 256-dimensional slot path was tested on the two 48 GB GPUs with
the full frozen V-JEPA forward, slot-model reconstruction loss, DDP backward,
and optimizer update. The observed DDP boundary is:

```text
per-GPU 112, global 224: pass for five consecutive optimizer steps
per-GPU 113, global 226: CUDA OOM during backward
```

These numbers no longer apply after adding the official-style feature projector
and increasing the slot dimension to 512. Re-run the batch probe before choosing
a formal-training micro-batch.

The reviewed 512-dimensional V-JEPA configuration uses per-GPU batch 48. On GPU
5,6 this gives a regular global batch of 96. Since each rank receives 5400 unique
samples per epoch, every epoch has 112 full steps followed by one partial step
with 24 samples per rank (global batch 48), preserving all 10800 unique samples.

### Known architecture risks

- V-JEPA jointly contextualizes all ten input frames. Its five temporal token
  slices are not equivalent to TextOCVP's frame-local `h_t`, so an early slot can
  contain information from later frames in the sampled clip. This is valid for
  clip decomposition but is not a causal implementation of TextOCVP scene parsing.
- The current decoder reconstructs normalized V-JEPA features only. Official
  ExtendedDINOSAUR also reconstructs RGB. Decoder-mask entropy and per-slot usage
  must therefore be monitored for slot collapse before the decomposition model is
  accepted for Stage 2.
