# Scheme-E Gated Masked Object Joint Attention

Scheme-E keeps Scheme-D's per-object tube resampler and entity binding, but
replaces the independent Wan object cross-attention residual with a low-width,
block-sparse masked joint-attention adapter.

Architecture version 3 routes each sample-local entity ID to both sides of the
binding: the tracked object slot and one explicit noun-phrase span in the
positive T5 context. Repeated nouns consume distinct spans in prompt order.

## Forward

```text
Wan native self-attention (frozen)
  -> project video 3072 -> 256
  -> object queries read [video tokens; object tokens]
  -> update compact object memory
  -> video queries read updated object tokens only
  -> project 256 -> 3072
  -> scalar tanh gate, initialized to zero
  -> Wan text cross-attention (frozen)
  -> Wan FFN (frozen)
```

The adapter is installed only in blocks `8,14,20`. The added branch has no
video-to-video attention edges: every path from an added video residual passes
through compact object memory. Its attention complexity is
`O(M*(N+M) + N*M)` instead of `O((N+M)^2)`. A zero object context exits before
attention and produces an exact zero adapter residual.

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

Scheme-D and Scheme-E v1/v2/v3 Stage1B checkpoints are intentionally
incompatible with each other.
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
