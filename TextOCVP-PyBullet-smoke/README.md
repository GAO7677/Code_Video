# TextOCVP PyBullet Stage 1

This directory is a thin launcher for the official TextOCVP SAVi Stage 1 trainer.
It no longer trains the decomposition model and text predictor in one process.

## Stage 1 contract

```text
source                  raw H.264 PyBullet video.mp4
raw video               90 frames, 960x540, 30 FPS
sampling range           raw frame 0 through 49, inclusive
sampled clip             10 contiguous frames, stride 1; train start 0 through 40
training tensor          [B,10,3,64,112]
SAVi slots               8 x 128
loss                     official TextOCVP reconstruction MSE
trainer/checkpoint       official TextOCVP implementation and schema
caption/T5/predictor     not loaded during Stage 1
```

Training clips use a random contiguous start in `[0, 40]`, so every selected frame
is inside `[0, 49]`. Validation maps TextOCVP's `valid` split to the dataset's
`val` directory and uses the deterministic centered start frame `20`.

Run the Stage 1 smoke:

```bash
GPU_ID=6 DATASET_LIMIT=32 NUM_EPOCHS=1 \
bash /home/gaoya/Code_Video/TextOCVP-PyBullet-smoke/run_smoke.sh
```

Large artifacts are written under `/data/gaoya/agent-data/checkpoints`.

Stage 2 must be created as a separate official TextOCVP predictor experiment and
must explicitly load a selected Stage 1 checkpoint from its `models/` directory.
It is intentionally not started by this smoke launcher.
