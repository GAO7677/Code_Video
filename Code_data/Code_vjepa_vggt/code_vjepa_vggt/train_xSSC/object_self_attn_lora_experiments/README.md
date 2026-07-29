# Object/Self-Attention LoRA Experiments

This directory is independent of the original `train_xSSC` entry points. It
implements three experiments from one training script:

- `object_only`: object cross-attention LoRA, object gates, xSSC projection,
  and time embedding.
- `full_sa`: `object_only` plus standard q/k/v/o LoRA in all 30 Wan
  self-attention layers.
- `s_head`: `object_only` plus compact q/k/v/o LoRA supported only on all 59
  configured same-frame-mass S heads.

All modes load the same OpenVid/MOVi-D/Genesis `step-010000` LoRA, merge it
into the frozen Wan weights, and unload the original PEFT modules before
constructing the experiment adapters. New self-attention adapters are
zero-delta initialized, so all modes share the same step-0 Wan forward.

## Configuration

Common settings live in `configs/base.json`. Each experiment config extends
that file and changes only its name and `adaptation.mode`. Paths, data mixture,
GPU selection, batch size, LoRA ranks, optimization, checkpointing, and W&B
settings are all configuration values.

Validate without allocating a model:

```bash
bash run_train_from_config.sh configs/object_only.json --validate-only
bash run_train_from_config.sh configs/full_sa.json --validate-only
bash run_train_from_config.sh configs/s_head.json --validate-only
```

Print the exact launch command:

```bash
bash run_train_from_config.sh configs/s_head.json --dry-run
```

Start a foreground training run:

```bash
bash run_train_from_config.sh configs/object_only.json
```

Every run stores its fully resolved configuration and exact launch command in
the checkpoint output directory.

## S-head parameterization

For q/k/v, each block-level compact adapter maps the full input through a
shared rank-r basis and stores output rows only for selected heads. For the
output projection, it reads only selected head channels and maps them back to
the model dimension. Heads selected within one block share the low-rank basis;
no parameters or residual output are allocated to non-selected heads.

The default full head list was discovered on the common-S population measured
using the raw-phys-state Wan LoRA, xSSC, and PhysRVG. Its provenance and hashes
are recorded in `configs/same_frame_mass_heads_full59.json`. The previous
32-head exact-block-matched subset remains in
`configs/same_frame_mass_heads.json` for controlled replication. Role stability
should be validated on the merged OpenVid initialization before making a
causal claim.

With rank 32, the full 59-head adapter spans 21 blocks and adds 9,224,192
self-attention parameters. Together with the 25,458,688 object-branch
parameters, `s_head` trains 34,682,880 parameters.
