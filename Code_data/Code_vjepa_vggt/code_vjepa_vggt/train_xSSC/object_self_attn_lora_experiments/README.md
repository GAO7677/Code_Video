# Object/Self-Attention LoRA Experiments

This directory is independent of the original `train_xSSC` entry points. It
implements four experiments from one training script:

- `object_only`: object cross-attention LoRA, object gates, xSSC projection,
  and time embedding.
- `full_sa`: `object_only` plus standard q/k/v/o LoRA in all 30 Wan
  self-attention layers.
- `s_head`: `object_only` plus compact q/k/v/o LoRA supported only on all 59
  configured same-frame-mass S heads.
- `t_head`: `object_only` plus compact q/k/v/o LoRA supported only on all 70
  configured common T heads.

All modes load the same OpenVid/MOVi-D/Genesis `step-010000` LoRA, merge it
into the frozen Wan weights, and unload the original PEFT modules before
constructing the experiment adapters. New self-attention adapters are
zero-delta initialized, so all modes share the same step-0 Wan forward.
Here, "unload" removes only the old PEFT wrapper and its A/B parameter objects
after their delta has been added to the Wan base weights. It does not discard
the learned OpenVid update. Consequently, the new experiment adapters learn a
fresh delta on top of the same baked OpenVid initialization; they do not
continue optimizing the original OpenVid A/B factors.

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
bash run_train_from_config.sh configs/t_head.json --validate-only
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

For `s_head` and `t_head`, the launcher also stores the exact input JSON as
`head_selection_config.json`. Each model checkpoint contains both the sorted
`[block, head]` tensor and the SHA256 of that JSON. Resume validates both
before loading any trainable tensor, so a checkpoint cannot silently move a
same-shaped adapter onto a different head list. Legacy S/T checkpoints without
this identity metadata are intentionally rejected for resume.

Periodic checkpoints are emitted only when `accelerator.sync_gradients` is
true, immediately after a complete optimizer update. With gradient
accumulation enabled, the intervening micro-steps therefore cannot repeatedly
overwrite the same `step-xxxxxx` checkpoint.

## Head-selective parameterization

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

The zero-based T-head list is stored in `configs/common_t_heads_full70.json`.
For example, `B04H00` maps to `blocks[4].self_attn` head 0. Its 70 heads span
21 blocks and add 9,404,416 rank-32 self-attention parameters; together with
the object branch, `t_head` trains 34,863,104 parameters.
