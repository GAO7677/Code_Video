# Wan-PhyCo Train 0716

This project ports the physical-property conditioning principle from
`nnsriram97/phyco` to Wan2.2-TI2V-5B using the local DiffSynth-Studio training
stack. It does not use the Scheme-D object branch.

## Architecture

The original PhyCo implementation freezes Cosmos-Predict2 and trains three
full ControlNet branches. Copying three groups of full 3072-wide Wan blocks is
not practical for the 5B Wan checkpoint, so this implementation uses the same
three-branch and zero-initialized-residual contract with ControlNet-XS
bottlenecks:

```text
property maps [B, 9, 1, H/16, W/16]
  rigid       [restitution, friction, valid]
  deformation [lambda, mu, valid]
  force       [strength, direction-x, direction-y]
    -> independent Conv3D encoders
    -> independent 128-wide residual blocks
    -> zero-initialized projections
    -> frozen Wan blocks 3,8,13,18,23,28
```

Wan DiT, VAE and T5 remain frozen. Only `pipe.dit.phyco_controlnet.*` is
trainable and saved.

## Data semantics

- PyBullet: direct friction/restitution supervision. Initial velocity provides
  a movement-direction proxy, not a force magnitude. No deformation labels.
- PhyCo Kubric: segmentation and metadata supervise all available branches.
- OpenVid: zero maps with all branches disabled. Because Wan is frozen, these
  samples preserve behavior but produce zero controller gradient.

The formal source ratio matches Scheme-D: PyBullet 0.30, Kubric 0.30, OpenVid
0.40. With six DDP workers and one sample per worker, 12000 optimizer steps
process roughly 72000 global samples. The expected PyBullet share is 21600
samples, or 18 expected passes over the 1200-sample training split. Use about
2000 optimizer steps when the target is three expected PyBullet passes.

## Commands

CPU tests:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python -m py_compile \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/wan_phyco_train0716/*.py
```

Two-step smoke:

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/wan_phyco_train0716/run_smoke.sh
```

Formal training, after smoke validation:

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/wan_phyco_train0716/run_train.sh
```

Large outputs are written below `/data/gaoya/agent-data`.

## Provenance

The method is adapted from PhyCo, whose repository is licensed CC BY-NC 4.0.
The controller code here is a new Wan/DiffSynth implementation and does not
copy Cosmos model code. Use is limited to research/non-commercial contexts
consistent with the upstream license.
