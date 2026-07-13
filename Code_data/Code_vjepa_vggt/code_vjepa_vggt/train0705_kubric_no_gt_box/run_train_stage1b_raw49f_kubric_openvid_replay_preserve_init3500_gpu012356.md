# Stage1B raw49f + Kubric + OpenVid replay-preserve training

## Purpose

This run starts from the selected stability-v3 `step-003500` Stage1B weights but
does not restore optimizer, scheduler, RNG, epoch, or global-step state. It is a
new W&B run with a fresh optimizer at learning rate `2e-5`.

## Data contract

- PyBullet: 1,200 raw H.264 videos, `start0` only, raw frames `0..48`.
- Kubric/PhyCo: 114,276 stability-v3 indexed training samples, decoded as prefix
  49-frame clips. The existing 69f/20ctx index is reused as a candidate list and
  the returned contract is then set to 49f/8ctx; this preserves the old replay
  distribution and avoids rescanning the full directory tree.
- OpenVid: 53,500 parquet rows, a random contiguous 49-frame clip per access.
- Every source is converted to `[C=3,T=49,H=512,W=896]` in `[-1,1]`.
- Context is explicitly fixed to frames `0..7`; the remaining 41 frames are
  targets. `replay_fixed_context_frames=8` is a frame count, not a maximum frame
  index.

The source sampling probabilities are `30% PyBullet / 30% Kubric / 40% OpenVid`.
Weights are assigned per item as `source_probability / source_length`, so the
large source sizes do not control the mixture. PyBullet supplies controlled long
physics, Kubric replays the distribution that produced the selected checkpoint,
and OpenVid supplies real appearance/background/text replay for TI2V retention.

## Retention controls

- Existing non-empty slot dropout remains at `0.35`.
- PyBullet/Kubric null-object probability is `0.20`; OpenVid uses a source-aware
  `0.50` probability. Null-object mode uses zero object tokens instead of `None`,
  so the otherwise frozen model still gives gradients to the object branch.
- OpenVid's remaining `0.50` detected-object samples run the normal grounding and
  object-conditioning path, preserving broad real-video object appearance.
- A no-object Wan forward is used as a frozen teacher at exactly the same noisy
  latent and timestep as the student. Prediction MSE weight is `0.05`. OpenVid
  evaluates it on every sample; PyBullet/Kubric evaluate it every 4 source samples
  and multiply active losses by 4 for an unbiased expected coefficient.
- DiT `object_gate` parameters remain FP32 while their forward activations are cast
  to the object branch dtype. This prevents `2e-5` updates from disappearing at
  BF16 precision.
- Existing gate, adapter residual, and per-block object-ratio guards remain active.

## First training gate

The default run is 300 new optimizer steps and saves at steps 150 and 300. Do not
extend directly to five PyBullet epochs. First compare both checkpoints on the
PhysicsIQ context/no-context split and inspect appearance drift, teacher delta,
object residual ratio, full-dropout frequency, and source sampling frequency.

## Command

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_train_stage1b_raw49f_kubric_openvid_replay_preserve_init3500_gpu012356.sh
```

Ratios and run length can be overridden without editing the script, for example:

```bash
MAX_TRAIN_STEPS=600 SAVE_STEPS=150 \
PYBULLET_RATIO=0.25 KUBRIC_RATIO=0.25 OPENVID_RATIO=0.50 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_train_stage1b_raw49f_kubric_openvid_replay_preserve_init3500_gpu012356.sh
```
