# Wan DiT Block Ablation

This directory contains runtime-only DiT ablation scripts. Existing Wan,
DiffSynth, LoRA, xSSC training, and inference source files are not modified.

## PhysRVG

PhysRVG ablations use the official model-loading order: the Wan2.2 TI2V 5B
base, full PhysRVG DiT checkpoint, then official rank-32 LoRA. Evaluation is
matched to the previous xSSC ablation run at 512x896, 49 output frames, 40
denoising steps, guidance scale 5, seed 42, and 30 fps. The one intentional
difference is that PhysRVG keeps its official `do_cfg=False` behavior. Every
PhysicIQ JSON uses its own `input_caption` and all eight frames from its
`input_video`; context frames use aspect-preserving resize plus center crop.

Supported PhysRVG modes:

- `baseline`: no ablation.
- `whole_block`: the selected block returns its input.
- `self_attn_zero`: `attn1` output is zero.
- `text_cross_attn_zero`: `attn2` output is zero.
- `ffn_zero`: FFN output is zero.
- `lora_off`: disable all ten official LoRA modules in the selected block.

One-case checks:

```bash
LIMIT=1 bash run_physrvg_physiciq_one.sh baseline none 0
LIMIT=1 bash run_physrvg_physiciq_one.sh self_attn_zero 5 0
```

Recommended sparse-layer sweep:

```bash
LIMIT=1 GPU_IDS=0,1 bash run_physrvg_physiciq_sweep.sh
```

The default sparse layers are `0 5 11 17 19 29`. Remove `LIMIT=1` only after
the pilot outputs have been checked.

## Models

- `wan_lora`: Wan2.2-TI2V-5B plus the physical-state LoRA at step 500.
- `xssc`: the same Wan and physical-state LoRA plus the xSSC object
  cross-attention checkpoint at step 1500.

Both paths use 8 context frames, 49 output frames, 512x896, 40 denoising
steps, CFG 5.0, seed 42, and context-aware video conditioning by default.

## Ablations

- `baseline`: no DiT ablation.
- `whole_block`: target block returns its input, `x_out = x_in`.
- `self_attn_zero`: target self-attention returns `zeros_like(query)`.
- `object_cross_attn`: target xSSC object cross-attention returns
  `zeros_like(query)`; valid only for `xssc`.

Wan2.2-TI2V-5B has 30 blocks, indexed from 0 through 29.

## One Experiment

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_physiciq_one.sh \
  wan_lora self_attn_zero 12 0
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
/data/gaoya/AAA_test_video/0623/test/v2v_wan
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
Every generated JSON/JSONL artifact records the exact mode and block under
the top-level `dit_ablation` field.
