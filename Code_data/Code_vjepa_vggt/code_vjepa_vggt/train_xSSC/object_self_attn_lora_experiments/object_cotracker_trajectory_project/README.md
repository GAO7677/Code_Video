# Object CoTracker trajectory loss

Training-case diagnostics for a proposed PyBullet-only auxiliary objective:

```text
x_t -> Full-SA No-Object Wan -> x0_pred -> frozen Tiny-VAE
    -> frozen CoTracker3 object-point trajectories
    -> GT-relative displacement Smooth L1 loss (beta=0.01)
```

The GT trajectory is extracted from the original PyBullet RGB video. Twenty-four
query points are sampled from the cached F04 SAM2 identity mask and reused for
the predicted video. The report keeps the old all-point objective and compares
it with the visibility-aware objective:

```text
w_gt = 1(gt_track_xy in per-frame object mask) * gt_confidence
L_coord = weighted SmoothL1(predicted displacement, GT displacement; w_gt)
L_vis = weighted -log(pred_visibility)
L_new = L_coord + 0.05 * L_vis
```

The per-frame object mask is the physical GT visibility gate; CoTracker
confidence is only a detached soft reliability weight. CoTracker visibility is
reported as a point-identity diagnostic, not as object occlusion. Predicted
visibility never masks coordinate supervision, so hiding an object cannot evade
the coordinate term; it is penalized separately by `L_vis`. For multiple
objects, point/time means are computed within each object and then averaged
equally across objects.

The training-scale coordinate loss is Smooth L1, equivalent to raw Huber divided
by its beta. Reports retain the old raw Huber/ADE diagnostics and overlay old
loss, new coordinate loss, the weighted visibility penalty, and new total loss
on the trajectory video. Raw CoTracker visibility and confidence probabilities
are saved in `trajectories.npz`.

Run the three cached F1/F2/F3 training cases on GPU 0:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/gaoya/Code_Video/DiffTrack-main:/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main:/home/gaoya/Code_Video/co-tracker-main \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
run_training_case_diagnostics.py all --device cuda:0
```

Run the high-noise `t=900` comparison in a separate output directory (the
command below maps physical GPU 2 to `cuda:0`; GPU 4 is intentionally unused):

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/gaoya/Code_Video/DiffTrack-main:/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main:/home/gaoya/Code_Video/co-tracker-main \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
run_training_case_diagnostics.py all \
  --device cuda:0 \
  --training-timestep 900 \
  --num-points 24 \
  --gradient-audit \
  --visibility-threshold 0.9 \
  --visibility-loss-weight 0.05 \
  --multiobject-cache /data/gaoya/agent-data/cache/uniform_multiobject_correspondence_diagnostics \
  --output-root /data/gaoya/agent-data/outputs/object_cotracker_trajectory_diagnostics_t900_p24_visibility_compare
```

The completed comparison report is written under:

```text
/data/gaoya/agent-data/outputs/object_cotracker_trajectory_diagnostics_t900_p24_visibility_compare
```

Serve the static report in the foreground:

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python -m http.server 8769 \
  --bind 0.0.0.0 \
  --directory /data/gaoya/agent-data/outputs/object_cotracker_trajectory_diagnostics_t900_p24_visibility_compare
```
