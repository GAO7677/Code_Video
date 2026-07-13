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
- Context is the first 8 sampled frames; the remaining 41 frames are targets.

The source sampling probabilities are `30% PyBullet / 30% Kubric / 40% OpenVid`.
Weights are assigned per item as `source_probability / source_length`, so the
large source sizes do not control the mixture. PyBullet supplies controlled long
physics, Kubric replays the distribution that produced the selected checkpoint,
and OpenVid supplies real appearance/background/text replay for TI2V retention.

## Retention controls

- Existing non-empty slot dropout remains at `0.35`.
- Full object-condition dropout is `0.20`. It uses zero object tokens instead of
  `None`, so the otherwise frozen model still gives gradients to the object branch.
- A no-object Wan forward is used as a frozen teacher at exactly the same noisy
  latent and timestep as the student. Prediction MSE weight is `0.05`, evaluated
  every 4 training steps to control memory and runtime overhead.
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
