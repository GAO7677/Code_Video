# TextOCVP PyBullet Smoke

This directory adapts the existing PyBullet episode dataset to the official
TextOCVP implementation without modifying `TextOCVP-master`.

The smoke performs both required stages:

1. Train SAVi object-centric decomposition for a few steps.
2. Freeze SAVi and train a compact TextOCVP-T5 predictor for a few steps.

Default data and model contract:

```text
source episode       full_frames [24,3,144,256]
sampled video        [10,3,64,112]
SAVi slots           6 x 128
predictor context    1 frame
predictor targets    9 frames
captions             deterministic English physical-motion templates
```

Run:

```bash
bash /home/gaoya/Code_Video/TextOCVP-PyBullet-smoke/run_smoke.sh
```

Useful overrides:

```bash
GPU_ID=6 DECOMP_STEPS=6 PREDICTOR_STEPS=3 DATASET_LIMIT=32 \
bash /home/gaoya/Code_Video/TextOCVP-PyBullet-smoke/run_smoke.sh
```

Large artifacts are written under `/data/gaoya/agent-data/checkpoints`.
The frozen `t5-small` encoder is loaded from
`/data/gaoya/agent-data/cache/textocvp/t5-small`.
