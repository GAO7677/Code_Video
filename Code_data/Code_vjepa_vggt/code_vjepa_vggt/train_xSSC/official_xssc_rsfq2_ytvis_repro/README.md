# Official xSSC RandSFQ2/YTVIS Reproduction

This directory vendors the training code for the official `RandSFQ2 + xSSC`
YTVIS-2022 object-discovery experiment used by the Wan integration in the
parent directory.

## Provenance

- Upstream: `https://github.com/Genera1Z/xSSC.git`
- Commit: `90a0ef1c3cc02c05e7a6abcee7b1adeaca107967`
- Official source snapshot: `upstream/`
- Target config: `upstream/config-randsfq/rsfq2_r-ytvis.py`
- Snapshot integrity: `SHA256SUMS`

Files under `upstream/` are copied verbatim from the commit above. The launcher
and this document are local reproduction aids and are not upstream files.

## Official Experiment

The target config trains for 50,000 steps on converted YTVIS-2022 LMDB data.
It samples five-frame training clips, applies a random square crop with scale
`[0.75, 1]`, resizes to 256x256, and randomly flips horizontally. Validation
uses a center square crop followed by a 256x256 resize. The model uses a frozen
DINOv2-S/14 backbone, seven 256-dimensional slots, RandSFQ aggregation, the
xSSC temporal transition, and a reconstruction objective over backbone
features. The official per-process training batch size is 8, learning rate is
`5e-5`, gradient clipping is `0.05`, and the default seed is 42.

## Required Data

Set `DATA_DIR` to a directory containing exactly:

```text
${DATA_DIR}/ytvis_2022/train.lmdb
${DATA_DIR}/ytvis_2022/val.lmdb
```

The converted LMDB dataset is not currently present on this machine. It must
come from the converted datasets referenced by the official xSSC/RandSF.Q
documentation; ordinary YTVIS image folders are not a drop-in replacement.

## Launch

The wrapper changes only paths and process environment. It invokes the copied
official `train.py` and official config without patching either file.

```bash
DATA_DIR=/data/gaoya/dataset/xssc_converted \
GPU_ID=1 \
bash run_official_rsfq2_ytvis.sh
```

Outputs default to
`/data/gaoya/agent-data/checkpoints/xssc_official_reproduction`. Set
`SAVE_DIR`, `SEED`, `PYTHON_BIN`, or `WANDB_PROJECT` when needed.

Verify the source snapshot before training:

```bash
bash verify_upstream_snapshot.sh
```

The upstream repository does not provide a fully pinned environment lockfile;
its `requirements.txt` lists package names and version comments. Reproducing
the official code path is therefore exact here, while bitwise numerical
reproduction also depends on matching the original CUDA, PyTorch, timm, and
dataset versions.
