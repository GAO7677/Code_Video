# Scheme-D Object Tube Resampler

This directory is an independent replacement for the Scheme-C object-token
builder. Existing Scheme-C source files and checkpoints are not modified.

## Architecture

For every grounded object slot, the resampler receives three variable-length
source sequences:

```text
Wan VAE samples:    T_latent x P query points
V-JEPA samples:     T_jepa   x P query points
CoTracker states:   T_ctx    x P query points
```

Each source token contains modality, normalized time, point-index, and slot
embeddings. Objects are flattened into the batch dimension before attention,
so one object's resampler cannot attend another object's source tokens.

`K` learned queries independently compress each object tube:

```text
[B, source_time_and_points, O, feature]
    -> per-object cross-attention resampler
[B, K, O, 4096]
    -> noun phrase + instance-ID residual + bbox side information
[B, K * O_valid, 4096]
    -> selected Wan object cross-attention blocks
```

Defaults:

```text
context frames                 8
tracked points per object      8
maximum objects                4
learned tokens per object K    4
resampler hidden dimension     512
resampler layers               2
Wan object blocks              8,11,14,17,20,23
VGGT                            disabled
legacy Stage1A checkpoint      disabled
```

The seven scalar track/box values are source features only. They do not replace
visual VAE/V-JEPA features and are not the final object representation.

## Files

```text
models.py       learned object-tube resampler and DiT block pruning
train.py        replay-preserving training entrypoint
infer.py        JSON-native batch v2v inference entrypoint
run_smoke.sh    short two-step data/model smoke
run_train.sh    formal 3500-step training configuration
run_infer.sh    batch inference command
test_models.py  CPU unit tests for shapes, isolation, gradients, K, and pruning
```

## Smoke

The smoke starts from the base Wan LoRA. It does not resume Scheme-C and does
not load `stage1a_full_token_old`.

```bash
GPU_PAIR=0,6 MAX_TRAIN_STEPS=2 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0715_scheme_d_object_tube_resampler/run_smoke.sh
```

Large outputs are written below `/data/gaoya/agent-data/checkpoints` by default.

## Formal Training

Run in the foreground:

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0715_scheme_d_object_tube_resampler/run_train.sh
```

Useful overrides:

```bash
MAX_TRAIN_STEPS=3500
SAVE_STEPS=500
LEARNING_RATE=1e-5
TUBE_NUM_TOKENS=4
TUBE_HIDDEN_DIM=512
OUTPUT_DIR=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/<run>
```

`--stage2_resume_from` is supported only for a checkpoint produced by the same
Scheme-D architecture and the same `K`, hidden dimension, and block IDs.

## Inference

```bash
GPU_PAIR=7,7 \
WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/<run>/checkpoints/step-000500 \
INPUT_JSON_LIST=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt \
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/AAA_physv/scheme_d_object_tube_step500 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0715_scheme_d_object_tube_resampler/run_infer.sh
```

Inference architecture variables must match training:

```text
TUBE_NUM_TOKENS
TUBE_HIDDEN_DIM
OBJECT_BLOCK_IDS
```

## Checkpoint Contract

A valid Scheme-D checkpoint includes:

```text
object_pooler.*
object_adapter.*
object_embedding.*
blocks.{8,11,14,17,20,23}.object_cross_attn.*
blocks.{8,11,14,17,20,23}.norm4.*
blocks.{8,11,14,17,20,23}.object_gate
```

Scheme-C checkpoints do not contain the learned resampler and must not be used
as Scheme-D resume checkpoints.

## Deliberate Exclusion

This first implementation does not add a future spatial attention mask. Future
object locations are unknown at inference, so such a mask would require a
separately trained future-location predictor. A context mask-to-slot routing
loss is also left out until the Wan attention implementation exposes stable
per-head logits; the current Flash Attention wrapper returns only outputs.
