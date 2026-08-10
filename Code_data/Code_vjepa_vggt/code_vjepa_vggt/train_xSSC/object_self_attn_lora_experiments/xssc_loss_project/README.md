# Full-SA + No-Object + xSSC feature loss

Common training path:

- Wan2.2-TI2V-5B DiT initialized by merging the existing OpenVid rank-32 LoRA.
- No object-conditioning branch.
- Trainable parameters remain the rank-32 Q/K/V/O LoRA modules in all 30
  self-attention blocks (23,592,960 parameters).
- The frozen Wan VAE decodes reconstructed latent `pred x0` and GT latent `x0`.
- A frozen xSSC encoder extracts aligned slots from all 49 frames; cosine loss
  is averaged only over future frames 8--48 and weighted by 0.1.
- DINOv3 MOVi-C uses GT first-frame SAM2 AMG boxes for both branches. Official
  DINOv2 xSSC shares the exact same stochastic initialization query between
  prediction and target so slot identities are comparable.

Two configs select the frozen encoder while leaving the rest of training equal:

- `configs/full_sa_no_object_xssc_loss_dinov3_movic_step50000.json`
- `configs/full_sa_no_object_xssc_loss_official_dinov2.json`

Validate without launching training:

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  ../launch_from_config.py \
  configs/full_sa_no_object_xssc_loss_dinov3_movic_step50000.json \
  --validate-only
```

Formal training is intentionally not started by the implementation/diagnostic
step. Remove `--validate-only` only after choosing the GPUs and approving the
formal run.
