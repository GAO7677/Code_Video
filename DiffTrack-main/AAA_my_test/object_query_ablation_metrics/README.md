# Object Query Ablation Metrics · 001460 / six-seed common cohort

This directory contains the code for an auditable evaluation of each 49-video
Object Query ablation set: one no-intervention baseline, 24 Fixed Top100
ablations and 24 Tube Top100 ablations. The dashboard table is a strict macro
mean over seeds `13248 32466 35075 47326 68613 90094`.

Large caches and generated media are written to:

`/data/gaoya/agent-data/outputs/object_query_ablation_metrics/0613pybullet_sample_001460_w002/`

The evaluation has two references:

- `baseline`: same-seed no-intervention generated video, measuring intervention effect.
- `source_gt_video` plus `states.npz`: source render and projected simulator state,
  measuring physical/source fidelity.

The web page is exposed by the existing 8092 viewer at
`/object-query-ablation-metrics` after all stages have generated `report.json`.

## Stages

Set `OBJECT_QUERY_ABLATION_SEED` for every single-seed stage. For example:

```bash
# Do not use GPU 4. These examples use physical GPU 5.
OBJECT_QUERY_ABLATION_SEED=47326 CUDA_VISIBLE_DEVICES=5 \
  /data/gaoya/miniconda3/envs/wan/bin/python \
  AAA_my_test/object_query_ablation_metrics/extract_tracks.py

OBJECT_QUERY_ABLATION_SEED=47326 CUDA_VISIBLE_DEVICES=5 \
  /data/gaoya/miniconda3/envs/physxnet_mpm_env/bin/python \
  AAA_my_test/object_query_ablation_metrics/extract_masks.py

OBJECT_QUERY_ABLATION_SEED=47326 CUDA_VISIBLE_DEVICES=5 \
  /data/gaoya/miniconda3/envs/wan/bin/python \
  AAA_my_test/object_query_ablation_metrics/extract_source_raft.py

OBJECT_QUERY_ABLATION_SEED=47326 CUDA_VISIBLE_DEVICES=5 \
  /data/gaoya/miniconda3/envs/physxnet_mpm_env/bin/python \
  AAA_my_test/object_query_ablation_metrics/compute_perceptual.py

OBJECT_QUERY_ABLATION_SEED=47326 /data/gaoya/miniconda3/envs/wan/bin/python \
  AAA_my_test/object_query_ablation_metrics/compute_metrics_and_overlays.py

OBJECT_QUERY_ABLATION_SEED=47326 /data/gaoya/miniconda3/envs/wan/bin/python \
  AAA_my_test/object_query_ablation_metrics/validate_outputs.py

/data/gaoya/miniconda3/envs/wan/bin/python \
  AAA_my_test/object_query_ablation_metrics/aggregate_reports.py
```

The aggregate report uses one invariant cohort for every displayed scalar. A
value is emitted only when all six seed reports contain a finite value; otherwise
the table receives `N/A` instead of averaging a smaller subset.

Every metric record states its reference, exact formula, validity mask and media
asset. Overlay media are rendered from the cached arrays used by the metric
calculation; the visualization does not rerun tracking or segmentation.

The model-backed measurements deliberately use their canonical local
implementations and checkpoints: CoTracker3 offline scaled, SAM2.1 Hiera Large,
torchvision RAFT Large `C_T_SKHT_V2`, official DINOv2 ViT-L/14 and LPIPS v0.1
AlexNet through torchmetrics. Existing official VBench scores are read from each
generation manifest rather than approximated here. Simulator center trajectories
and sphere-to-oriented-box contact are calculated directly from `states.npz`.
