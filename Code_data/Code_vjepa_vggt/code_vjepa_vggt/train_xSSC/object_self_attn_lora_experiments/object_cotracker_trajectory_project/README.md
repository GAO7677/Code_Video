# Object CoTracker trajectory loss

Training-case diagnostics for a proposed PyBullet-only auxiliary objective:

```text
x_t -> Full-SA No-Object Wan -> x0_pred -> frozen Tiny-VAE
    -> frozen CoTracker3 object-point trajectories
    -> GT-relative displacement Huber loss
```

The GT trajectory is extracted from the original PyBullet RGB video. Twenty-four
query points are sampled from the cached F04 SAM2 identity mask and reused for
the predicted video. The report compares loss over all selected object tracks
with a stricter GT-visible-only audit. Prediction visibility is recorded but
never removes a primary loss term.

Run the three cached F1/F2/F3 training cases on GPU 0:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/gaoya/Code_Video/DiffTrack-main:/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main:/home/gaoya/Code_Video/co-tracker-main \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
run_training_case_diagnostics.py all --device cuda:0
```

Run the high-noise `t=900` comparison in a separate output directory:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/gaoya/Code_Video/DiffTrack-main:/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main:/home/gaoya/Code_Video/co-tracker-main \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
run_training_case_diagnostics.py all \
  --device cuda:0 \
  --training-timestep 900 \
  --num-points 24 \
  --gradient-audit \
  --output-root /data/gaoya/agent-data/outputs/object_cotracker_trajectory_diagnostics_t900_p24
```

Large outputs are written under:

```text
/data/gaoya/agent-data/outputs/object_cotracker_trajectory_diagnostics
```

Serve the static report in the foreground:

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python -m http.server 8950 \
  --bind 0.0.0.0 \
  --directory /data/gaoya/agent-data/outputs/object_cotracker_trajectory_diagnostics
```
