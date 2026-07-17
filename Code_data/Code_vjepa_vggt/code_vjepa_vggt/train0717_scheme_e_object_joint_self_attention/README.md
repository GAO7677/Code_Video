# Scheme-E Gated Object Joint Self-Attention

Scheme-E keeps Scheme-D's per-object tube resampler and entity binding, but
replaces the independent Wan object cross-attention residual with a GLIGEN-like
low-width joint self-attention adapter.

Architecture version 2 routes each sample-local entity ID to both sides of the
binding: the tracked object slot and one explicit noun-phrase span in the
positive T5 context. Repeated nouns consume distinct spans in prompt order.

## Forward

```text
Wan native self-attention (frozen)
  -> project video 3072 -> 256
  -> concatenate [video tokens; object tokens]
  -> joint self-attention at width 256
  -> retain video-token output only
  -> project 256 -> 3072
  -> scalar tanh gate, initialized to zero
  -> Wan text cross-attention (frozen)
  -> Wan FFN (frozen)
```

The adapter is installed only in blocks `8,14,20`. Object tokens do not use
Wan's video 3D RoPE; their existing entity, visual, trajectory, and geometry
encodings remain additive inputs to joint attention. A zero object context
produces an exact zero adapter residual, so full object dropout remains a true
object-branch dropout rather than training an unconditional extra video path.

## Trainable Modules

```text
ObjectTubeResampler
EntityIDBindingObjectConditionAdapter
  - shared entity ID embedding
  - object-side entity projection
  - positive T5 noun-span entity projection
blocks.{8,14,20}.norm4
blocks.{8,14,20}.object_cross_attn
blocks.{8,14,20}.object_gate
```

Wan DiT, base LoRA, VAE, T5, V-JEPA, CoTracker, GroundingDINO, and SAM2 remain
frozen. Despite the compatibility parameter name `object_cross_attn`, the
actual module class is `BottleneckObjectJointSelfAttention`.

Scheme-D, Scheme-E v1, and Scheme-E v2 Stage1B checkpoints are intentionally
incompatible.
Scheme-E starts from the same frozen Wan base LoRA or resumes a Scheme-E
checkpoint with matching dimensions and block IDs.

## Smoke

```bash
GPU_PAIR=0,6 MAX_TRAIN_STEPS=2 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0717_scheme_e_object_joint_self_attention/run_smoke.sh
```

## Formal Training

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0717_scheme_e_object_joint_self_attention/run_train.sh
```

Large checkpoints and generated outputs are written under
`/data/gaoya/agent-data`.

## Inference

```bash
GPU_PAIR=7 \
INFERENCE_DEVICES=cuda:0,cuda:0 \
WEIGHTS_ROOT=/data/gaoya/agent-data/checkpoints/<scheme-e-run>/checkpoints/step-000500 \
INPUT_JSON_LIST=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt \
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/AAA_physv/scheme_e_step500 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0717_scheme_e_object_joint_self_attention/run_infer.sh
```

`OBJECT_BRANCH_RESIDUAL_SCALE=0.0` is the strict functional ablation for the
joint adapter. `OBJECT_CONTEXT_ABLATION=zero` tests zero object memory while
retaining the same model path.
