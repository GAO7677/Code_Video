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
  `/data/gaoya/ckpt/facebook-dinov3-vitl16-pretrain-lvd1689m/model.safetensors`
- YTVIS-2022 config: `upstream/config-randsfq/rsfq2_r-ytvis.py`
- YTVIS-HQ config:
  `upstream/config-randsfq/rsfq2_r-ytvis_hq-dinov3_vitl16_256.py`
- YTVIS-HQ slot-512 enhanced config:
  `upstream/config-randsfq/rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512.py`

Meta's official DINOv3 implementation is vendored under `third_party/dinov3`.
The local Hugging Face LVD-1689M checkpoint is converted in memory into the Meta
module layout. All 415 source tensors must be consumed, and any unknown key or
shape mismatch is fatal.

## Controlled Backbone Change

Both experiments retain a 16x16 spatial grid:

```text
Official xSSC: 256 -> bicubic 224 -> DINOv2-S/14 -> 16x16x384
This variant:  256 ----------------> DINOv3-L/16 -> 16x16x1024
```

The DINOv3 LVD checkpoint uses the same ImageNet normalization as the official
DINOv2 xSSC baseline:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Seven 256-dimensional slots, five-frame training clips, the xSSC recurrence,
the 16x16 decoder grid, losses, optimizer, learning rate, batch size, training
steps, augmentation policy, and frozen-backbone policy remain unchanged.

See `AFFECTED_COMPONENTS.md` for the complete impact list.

## Data

The official converted YTVIS-HQ release is installed at:

```text
/data/gaoya/dataset/ytvis_hq/train.lmdb
/data/gaoya/dataset/ytvis_hq/val.lmdb
/data/gaoya/dataset/ytvis_hq/test.lmdb
```

The three downloaded release volumes and their published SHA-256 values are
kept under `/data/gaoya/dataset/ytvis_hq_download`.

## Verification

```bash
bash verify_experiment.sh
CUDA_VISIBLE_DEVICES=6 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python smoke_test_dinov3_xssc.py

CUDA_VISIBLE_DEVICES=1 \
DINOV3_CHECKPOINT=/data/gaoya/ckpt/facebook-dinov3-vitl16-pretrain-lvd1689m/model.safetensors \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python smoke_train_ytvis_hq.py \
  --data-dir /data/gaoya/dataset --batch-size 1
```

## Training

```bash
DATA_DIR=/data/gaoya/dataset/xssc_converted \
GPU_ID=6 \
bash run_train_rsfq2_ytvis_dinov3_vitl16_256.sh
```

Checkpoints default to
`/data/gaoya/agent-data/checkpoints/xssc_rsfq2_ytvis_dinov3_vitl16_256`.

For YTVIS-HQ training:

```bash
DATA_DIR=/data/gaoya/dataset \
GPU_ID=1 \
bash run_train_rsfq2_ytvis_hq_dinov3_vitl16_256.sh
```

YTVIS-HQ checkpoints default to
`/data/gaoya/agent-data/checkpoints/xssc_rsfq2_ytvis_hq_dinov3_vitl16_256`.

## Slot-512 Enhanced Variant

The slot-512 config keeps the slot-256 config unchanged for controlled
DINOv2/DINOv3 comparisons. It centralizes all ablation controls at the top of
the config, uses 512-dimensional slots, and scales the transition to eight
64-dimensional heads. The 1024-dimensional decoder is unchanged except for
the slot projection from 512 to 1024.

```bash
DATA_DIR=/data/gaoya/dataset \
GPU_IDS=0,1,2,3 \
WANDB_PROJECT=<project> \
bash run_train_rsfq2_ytvis_hq_dinov3_vitl16_256_slot512.sh
```

Slot-512 checkpoints default to
`/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC`.
The rank-0 trainer writes per-step reconstruction loss, pre-clip gradient norm,
clip coefficient, learning rate, and peak reserved memory to W&B and
`step_metrics.jsonl`. Full validation reconstruction loss and segmentation
metrics are logged every 1,250 optimizer steps.

### Four-GPU smoke train and validation

The enhanced config uses a per-GPU batch size of 96. The DDP smoke launcher
runs the formal BF16 trainer for 10 optimizer steps on four GPUs (global batch
384), averages gradients through DDP, clips the synchronized gradient norm,
runs the full validation set, and saves the complete final checkpoint:

```bash
GPU_IDS=0,1,2,3 bash run_smoke_train_slot512_ddp_4gpu.sh
```

The matching inference launcher reconstructs the frozen LVD-1689M backbone,
strictly checks every non-backbone checkpoint key, and evaluates all 280
variable-length YTVIS-HQ validation videos:

```bash
GPU_IDS=0,1,2,3 bash run_infer_val_slot512_ddp_4gpu.sh
```

Smoke artifacts are stored outside the source tree:

```text
/data/gaoya/agent-data/checkpoints/xssc_slot512_formal_smoke/rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512/42/last.pth
/data/gaoya/agent-data/checkpoints/xssc_slot512_formal_smoke/rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512/42/run_summary.json
/data/gaoya/agent-data/outputs/xssc_slot512_formal_smoke/ytvis_hq_val_all_loss.json
```

### Training batch viewer

Reconstruct a per-rank batch with the formal sampler, temporal/spatial
transforms, worker seeds, and collate function:

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python visualize_training_batch.py \
  --epoch 52 --rank 0 --world-size 4 --batch-index 0 \
  --data-dir /data/gaoya/dataset \
  --output-dir /data/gaoya/agent-data/outputs/xssc_training_batch_epoch52_rank0_batch0
```
