# Wan DiT Block Ablation

This directory contains runtime-only DiT ablation scripts. Existing Wan,
DiffSynth, LoRA, xSSC training, and inference source files are not modified.

## Models

- `wan_lora`: Wan2.2-TI2V-5B plus the physical-state LoRA at step 500.
- `xssc`: the same Wan and physical-state LoRA plus the xSSC object
  cross-attention checkpoint at step 1500.

Both paths use 8 context frames, 49 output frames, 512x896, 40 denoising
steps, CFG 5.0, seed 42, and context-aware video conditioning by default.

## Ablations

- `baseline`: no DiT ablation.
- `whole_block`: target block returns its input, `x_out = x_in`.
- `self_attn`: target self-attention returns `zeros_like(query)`.
- `object_cross_attn`: target xSSC object cross-attention returns
  `zeros_like(query)`; valid only for `xssc`.

Wan2.2-TI2V-5B has 30 blocks, indexed from 0 through 29.

## One Experiment

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_physiciq_one.sh \
  wan_lora self_attn 12 0
```

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_physiciq_one.sh \
  xssc object_cross_attn 12 1
```

Use `none` as the block argument for a baseline:

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_physiciq_one.sh \
  xssc baseline none 1
```

Default outputs are written under:

```text
/data/gaoya/agent-data/outputs/wan_dit_block_ablation/physicIQ
```

Set `OUTPUT_BASE` to override this root.

## Sweep

The default sweep runs all 30 blocks for both models and includes the xSSC
object-cross-attention ablation:

```bash
GPU_IDS=0,1,2,3 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_physiciq_sweep.sh
```

For a smaller pilot:

```bash
GPU_IDS=0,1 \
BLOCK_IDS="0 5 10 15 20 25 29" \
INCLUDE_OBJECT_CROSS_ATTN=0 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_physiciq_sweep.sh
```

The full default sweep produces 152 configurations and, for the current 67
PhysicIQ cases, up to 10,184 videos. Existing complete outputs are skipped.
