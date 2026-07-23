# DINOv3-S/16 xSSC with official MOVi-C dimensions

This is an independent controlled-backbone experiment. It keeps DINOv3 and
restores the official `rsfq2_c-movi_c.py` model dimensions:

- input: `256x256`
- patch grid: `16x16`
- VFM and decoder dimension: `384`
- slot dimension: `256`
- decoder static/dynamic split: `288/96`
- transition: 4 heads
- decoder: 4 layers, 4 heads, FFN dimension 1536

The xSSC method remains unchanged: six-frame MOVi-C clips, bbox-conditioned
initialization, dynamic ratio 1/4, previous/current cross-temporal
reconstruction, relative time embedding, and reconstruction MSE only.

DINOv3 uses `norm_out=False` to match the official xSSC feature-target choice.
No feature normalization, explicit consistency loss, branch-specific
projection, or transfer checkpoint is added.

The verifier also compares every non-backbone state-dict entry against the
official MOVi-C `42-0035.pth`. All 84 keys and all tensor shapes must match.
The resulting model has 34,086,912 parameters: 21,601,152 frozen DINOv3
parameters and 12,485,760 trainable xSSC parameters.

## Before formal training

- The current launcher uses four GPUs with batch 32 per GPU and no gradient
  accumulation, for an effective batch of 128. Run a GPU memory smoke before
  increasing it.
- The local MOVi-C train split currently reports 669/1024 shards and
  6361/9737 samples. Complete the missing shards before treating a formal run
  as the full-dataset result.
- A real six-frame CPU forward passed. With the intentionally unnormalized
  DINOv3 pre-final-norm target, the untrained sample had feature RMS 29.95 and
  reconstruction MSE 899.64; monitor loss scale and gradient clipping in smoke.

Weights:

`/data/gaoya/ckpt/facebook-dinov3-vits16-pretrain-lvd1689m/model.safetensors`

Dimension verification:

```bash
cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_movic_dinov3_vits16_official_dims
DINOV3_CHECKPOINT=/data/gaoya/ckpt/facebook-dinov3-vits16-pretrain-lvd1689m/model.safetensors \
/data/gaoya/miniconda3/envs/vjepa2/bin/python verify_official_dimensions.py
```

Formal DDP launcher:

```bash
bash run_train_rsfq2_movi_c_dinov3_vits16_official_dims.sh
```

YTVIS-HQ reproduction stage (`0 -> 15000` on GPU 5,6): batch 192 per
GPU with one accumulation step, for an effective batch of 384.

```bash
bash run_train_rsfq2_ytvis_hq_dinov3_vits16_official_dims_gpu56.sh
```
