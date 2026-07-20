# xSSC RandSFQ2 with DINOv3 ViT-L/16 at 256x256

This is an isolated derivative of the official xSSC `rsfq2_r-ytvis`
experiment. It keeps the 16x16 spatial token grid and replaces only the visual
backbone family and the dimensions that must follow from that replacement.

## Fixed Sources

- xSSC: `Genera1Z/xSSC` commit
  `90a0ef1c3cc02c05e7a6abcee7b1adeaca107967`
- DINOv3: `facebookresearch/dinov3` commit
  `6876159a11b4df116f30f667f8c9888617df0751`
- DINOv3 weights:
  `/data/gaoya/ckpt/facebook-dinov3-vitl16-pretrain-sat493m/model.safetensors`
- Target config: `upstream/config-randsfq/rsfq2_r-ytvis.py`

Meta's official DINOv3 implementation is vendored under `third_party/dinov3`.
The local Hugging Face SAT-493M checkpoint is converted in memory into the Meta
module layout. All 415 source tensors must be consumed, and any unknown key or
shape mismatch is fatal.

## Controlled Backbone Change

Both experiments retain a 16x16 spatial grid:

```text
Official xSSC: 256 -> bicubic 224 -> DINOv2-S/14 -> 16x16x384
This variant:  256 ----------------> DINOv3-L/16 -> 16x16x1024
```

The DINOv3 variant uses the SAT-493M processor normalization:

```text
mean = [0.430, 0.411, 0.296]
std  = [0.213, 0.156, 0.143]
```

Seven 256-dimensional slots, five-frame training clips, the xSSC recurrence,
the 16x16 decoder grid, losses, optimizer, learning rate, batch size, training
steps, augmentation policy, and frozen-backbone policy remain unchanged.

See `AFFECTED_COMPONENTS.md` for the complete impact list.

## Data

The official converted dataset layout is required:

```text
${DATA_DIR}/ytvis_2022/train.lmdb
${DATA_DIR}/ytvis_2022/val.lmdb
```

The converted LMDB dataset is not currently available on this machine.

## Verification

```bash
bash verify_experiment.sh
CUDA_VISIBLE_DEVICES=6 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python smoke_test_dinov3_xssc.py
```

## Training

```bash
DATA_DIR=/data/gaoya/dataset/xssc_converted \
GPU_ID=6 \
bash run_train_rsfq2_ytvis_dinov3_vitl16_256.sh
```

Checkpoints default to
`/data/gaoya/agent-data/checkpoints/xssc_rsfq2_ytvis_dinov3_vitl16_256`.
