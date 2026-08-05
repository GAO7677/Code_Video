# V-JEPA Auxiliary Loss Training Requirements

## Requested behavior

- Keep the formal Full-SA/no-object experiment configuration unchanged except
  for the new auxiliary loss and experiment name.
- Reconstruct `pred_x0 = x_t - sigma_t * pred_v` from the same DiT forward used
  by the original flow-matching loss.
- Restore clean conditioning latents before feature comparison.
- Decode prediction and GT latents with the same frozen `taew2_2` Tiny VAE.
- Uniformly select the same video frames and apply identical V-JEPA transforms.
- Compare frozen V-JEPA2.1 ViT-L token features with normalized feature MSE.
- Preserve gradient flow from V-JEPA through Tiny VAE to DiT LoRA only.
- Never update or checkpoint Tiny VAE/V-JEPA parameters and never use MP4 I/O.

## Effective defaults

- Auxiliary weight: `0.01`.
- Noise range: `sigma in [0.2, 0.8]`.
- Cadence: every 2 micro-forwards, not every 2 optimizer updates.
- Clip: 16 uniformly sampled frames at V-JEPA input size 384.
- Tiny VAE sequential decode to reduce peak memory.
- V-JEPA runs in FP32 because the current V-JEPA2.1 attention path showed a
  query/key/value dtype mismatch in BF16.
- Tiny VAE RGB uses clamp in the forward pass with a straight-through gradient,
  plus a `0.1` range-violation penalty inside the auxiliary objective.
- V-JEPA auxiliary loss uses Wan's native timestep weight normalized over the
  configured sigma gate.
- Only future-only temporal tubelets contribute feature loss; the full clip is
  still encoded so future tokens can attend to clean context.
- Frame sampling mixes global uniform clips and local context-boundary clips at
  equal probability.
- Main/auxiliary `pred_v` output-gradient norms and cosine are measured every
  400 micro-forwards when the auxiliary branch is active.

## Strict review findings and safeguards

- Re-running DiT to obtain `pred_v` would change stochastic behavior and double
  the dominant compute. The implementation captures the original single forward.
- Saving and reopening videos is non-differentiable. Training stays tensor-only.
- Context positions are not supervised by the original DiT loss. They are
  restored from GT before decoding so arbitrary context `pred_v` cannot pollute
  the feature target.
- Frozen encoders still need activations for gradients to their inputs. Expect a
  substantial VRAM and runtime increase even though their parameters are frozen.
- Frozen auxiliary modules are intentionally not registered as children of the
  training module, preventing DDP broadcasts and accidental checkpoint bloat.
- Raw feature MSE depends strongly on feature width and norm. Token features are
  L2-normalized and squared distance is summed over channels before averaging.
- The cadence counter is local and resets after resume. This only shifts which
  micro-forward receives the auxiliary loss; it does not affect model state.
- GT V-JEPA features are computed online because training samples and frame
  transforms are generated in the existing data path. They are not globally
  cached.

## Foreground launch

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/run_train_from_config_vjepa_loss.sh \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/configs/formal_full_sa_no_object_gpu27_vjepa_loss.json
```

Use `--dry-run` first if command/config validation is desired without starting
the two-GPU training job.
