# Scheme-D v3 Object Tube Resampler

This directory is an independent replacement for the Scheme-C object-token
builder. Existing Scheme-C source files and checkpoints are not modified.

## Architecture

For every grounded object slot, the resampler receives two visual token
sequences and one compact motion sequence:

```text
Wan VAE samples:    T_latent x P query points
V-JEPA samples:     T_jepa   x P query points
CoTracker states:   T_ctx x P points -> Fourier trajectory encoder -> M=4 tokens
```

VAE and V-JEPA have independent LayerNorm/projection adapters and remain
separate token streams because their temporal grids differ. Visual tokens use
fixed sinusoidal time encodings plus Fourier encodings of their actual sampled
`[x,y]` coordinates; they do not use arbitrary point-index embeddings.
CoTracker `[x,y,dx,dy,t]` values use fixed
Fourier features plus visibility/confidence, then four learned motion queries
compress all point observations. Objects are flattened into the batch
dimension before attention, so one object's resampler cannot attend another
object's source tokens.

`K` learned queries independently compress each object tube:

```text
[B, O, 16 VAE + 32 V-JEPA + 4 motion, 256]
    -> per-object cross-attention resampler
[B, K, O, 256]
    -> noun phrase + instance-ID residual + bbox side information
[B, K * O_valid, 256]
    -> selected 3072 -> 256 -> 3072 bottleneck object cross-attention blocks
```

Defaults:

```text
context frames                 8
tracked points per object      8
maximum objects                4
learned tokens per object K    4
resampler/object dimension     256
motion tokens per object       4
motion Fourier bands           4
object attention dimension     256
object attention heads         8
resampler layers               2
Wan object blocks              8,11,14,17,20,23
VGGT                            disabled
legacy Stage1A checkpoint      disabled
training timestep source IDs   10..999 (BF16-zero-weight endpoint excluded)
```

The seven trajectory values `[x,y,dx,dy,t,visibility,confidence]` are compact
source features only. They do not replace visual VAE/V-JEPA features and do
not each become a 256-dimensional output token.

Compared with v1, the source sequence decreases from 112 to 52 tokens/object
and object memory decreases from 4096 to 256 channels. Compared with v2, v3
removes the global `256 -> 3072` object expansion and replaces six full-width
3072-dimensional attention branches with bottleneck attention:

```text
Q: video  3072 -> 256
K: object  256 -> 256
V: object  256 -> 256
O: delta   256 -> 3072
```

This reduces trainable parameters from 231.24M in v2 to 14.07M in v3.

## Files

```text
models.py       learned object-tube resampler and DiT block pruning
train.py        replay-preserving training entrypoint
infer.py        JSON-native batch v2v inference entrypoint
run_smoke.sh    short two-step data/model smoke
run_train.sh    formal 3500-step training configuration
run_infer.sh    batch inference command
test_models.py  CPU unit tests for shapes, isolation, gradients, K, and pruning
monitor_training_health.py  rolling loss/gradient/residual health monitor
audit_checkpoint.py  finite/shape/update audit without loading Wan
watch_checkpoint_audits.py  automatically audit each completed step-* bundle
build_validation_contact_sheets.py  aligned four-variant visual gate
```

## Smoke

The smoke starts from the base Wan LoRA. It does not resume Scheme-C and does
not load `stage1a_full_token_old`.

```bash
GPU_PAIR=0,6 MAX_TRAIN_STEPS=2 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0715_scheme_d_object_tube_resampler/run_smoke.sh
```

Large outputs are written below `/data/gaoya/agent-data/checkpoints` by default.

## Health Monitoring

Run against a formal training log:

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0715_scheme_d_object_tube_resampler/monitor_training_health.py \
  --log /data/gaoya/agent-data/checkpoints/<run>/train_<timestamp>.log \
  --output /data/gaoya/agent-data/checkpoints/<run>/training_health_latest.json \
  --history /data/gaoya/agent-data/checkpoints/<run>/training_health_history.jsonl \
  --poll-seconds 30
```

The monitor is read-only with respect to training. It reports non-finite
values, post-clip gradient anomalies, stale progress, object residual guard
activation, adapter cap activation, and entity residual growth.

Audit a saved checkpoint and optionally compare it with an earlier checkpoint:

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0715_scheme_d_object_tube_resampler/audit_checkpoint.py \
  /data/gaoya/agent-data/checkpoints/<run>/checkpoints/step-001000 \
  --compare /data/gaoya/agent-data/checkpoints/<run>/checkpoints/step-000500 \
  --output /data/gaoya/agent-data/checkpoints/<run>/checkpoint_audit_step1000.json
```

Continuously watch a formal run:

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0715_scheme_d_object_tube_resampler/watch_checkpoint_audits.py \
  --checkpoint-root /data/gaoya/agent-data/checkpoints/<run>/checkpoints \
  --output-dir /data/gaoya/agent-data/checkpoints/<run>/checkpoint_audits \
  --poll-seconds 60 --max-step 3500
```

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
TUBE_HIDDEN_DIM=256
TUBE_MOTION_TOKENS=4
TUBE_MOTION_FOURIER_BANDS=4
TUBE_OBJECT_ATTN_DIM=256
OUTPUT_DIR=/data/gaoya/agent-data/checkpoints/<run>
```

`--stage2_resume_from` is supported only for a v3 checkpoint with the same
`K`, hidden dimension, motion-token count, object-attention dimension, and
block IDs. Scheme-D v1/v2 checkpoints are intentionally rejected because they
use full-width object attention and an object embedding absent from v3.

## Inference

```bash
GPU_PAIR=7,7 \
INFERENCE_DEVICES=cuda:0,cuda:1 \
WEIGHTS_ROOT=/data/gaoya/agent-data/checkpoints/<run>/checkpoints/step-000500 \
INPUT_JSON_LIST=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt \
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/AAA_physv/scheme_d_object_tube_step500 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0715_scheme_d_object_tube_resampler/run_infer.sh
```

Inference architecture variables must match training:

```text
TUBE_NUM_TOKENS
TUBE_HIDDEN_DIM
TUBE_MOTION_TOKENS
TUBE_MOTION_FOURIER_BANDS
TUBE_OBJECT_ATTN_DIM
OBJECT_BLOCK_IDS
```

For one physical GPU, expose it once and map both runtime roles to the same
logical device. Repeating an ID in `CUDA_VISIBLE_DEVICES` is invalid on CUDA:

```bash
GPU_PAIR=4 INFERENCE_DEVICES=cuda:0,cuda:0 ... bash run_infer.sh
```

## Checkpoint Contract

A valid Scheme-D checkpoint includes:

```text
object_pooler.*
object_adapter.*
blocks.{8,11,14,17,20,23}.object_cross_attn.*
blocks.{8,11,14,17,20,23}.norm4.*
blocks.{8,11,14,17,20,23}.object_gate
```

Scheme-D v3 intentionally has no global `object_embedding.*`: bottleneck
object cross-attention consumes the 256-dimensional object memory directly.
Scheme-C and Scheme-D v1/v2 checkpoints do not satisfy this contract and must
not be used as Scheme-D v3 resume checkpoints.

## Deliberate Exclusion

This first implementation does not add a future spatial attention mask. Future
object locations are unknown at inference, so such a mask would require a
separately trained future-location predictor. A context mask-to-slot routing
loss is also left out until the Wan attention implementation exposes stable
per-head logits; the current Flash Attention wrapper returns only outputs.
