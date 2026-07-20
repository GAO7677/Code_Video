# DINOv2 to DINOv3 Impact

## Changed

1. **Backbone architecture**: DINOv2-S/14 is replaced by the frozen
   DINOv3-L/16 LVD-1689M backbone. Depth changes from 12 to 24, hidden width from
   384 to 1024, attention heads from 6 to 16, and backbone parameters increase
   to 303,154,176.
2. **Backbone input path**: the DINOv2-only 0.875 bicubic resize is removed.
   DINOv3 receives the full 256x256 crop, so patch size 16 still produces a
   16x16 grid. This is intentionally different from the checkpoint's default
   224x224 processor size.
3. **Normalization**: the LVD-1689M checkpoint uses the same ImageNet mean/std
   as the official DINOv2 xSSC baseline, so normalization is unchanged.
4. **Feature width**: `vfm_dim` changes from 384 to 1024. This propagates to the
   encoder projection input/output, SlotAttention `kv_dim`, transition input
   projection, decoder feature width, learned decoder positional embeddings,
   decoder projections, Transformer decoder `d_model` and feed-forward width,
   and the reconstruction target.
5. **Weights loader**: Hugging Face Q/K/V tensors are merged in Q-K-V order for
   Meta's official implementation. K bias remains zero, matching
   `key_bias=false`. Register tokens map to storage tokens. The LVD checkpoint
   does not use a separate local CLS norm, and the frozen single-crop feature
   path consumes pre-final-norm patch tokens only.
6. **Resource cost**: the complete model has 376,783,360 parameters and
   73,629,184 trainable parameters after freezing the backbone. The official
   DINOv2 xSSC downstream modules contain 12,418,944 trainable parameters, so
   widening `vfm_dim` increases the trainable count by 61,210,240 (5.93x).
   Training memory and runtime will be substantially higher than DINOv2-S/14,
   even though the spatial token count remains 256.
7. **Feature statistics**: the configuration keeps the original xSSC
   `norm_out=False` behavior. The encoder projection begins with LayerNorm, but
   the reconstruction target uses pre-final-norm DINOv3 features directly, so
   reconstruction-loss scale changes substantially. With LVD-1689M, the
   synthetic five-frame smoke test produced MSE around 5,694 with finite
   gradients. The one-batch YTVIS-HQ smoke produced MSE around 7,155 and a
   pre-clip gradient norm around 452. The frequency of activation of the
   official `0.05` gradient clip must be monitored before the full 50,000-step
   run.

## Unchanged

- YTVIS-2022 LMDB dataset and train/validation splits
- Random square crop, 256x256 resize, and random horizontal flip policy
- Five-frame training clips and temporal stride sampling
- 16x16 spatial feature grid and 256 spatial positions
- Seven slots with 256-dimensional slot embeddings
- SlotAttention iterations and RandSFQ/xSSC temporal transition design
- Reconstruction objective and object-discovery metrics
- Frozen visual backbone policy
- 50,000 steps, batch size 8, Adam, learning rate `5e-5`, and grad clip `0.05`
