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

The M1/M2/M3 Head-Scope grid also has an incremental, CPU-only Baseline-effect
stage.  It is deliberately separate from the 25-metric model-backed report: it
measures full-frame, frozen target-ROI and outside-object image/temporal change,
then writes the values into a collapsed panel below each generated video.

## One-command evaluation

Use `bench.sh` with either one `seed_XXXXX` result directory or its parent case
directory. The seed directory must contain `video_similarity_top100.json` with
exactly one baseline and 48 Fixed/Tube ablations. A case directory runs every
direct `seed_*` child in sequence.

```bash
cd /home/gaoya/Code_Video/DiffTrack-main

# One seed. GPU is a physical index; GPU 4 is forbidden.
GPU=5 bash AAA_my_test/object_query_ablation_metrics/bench.sh \
  /data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1/0613pybullet_sample_001460_w002/seed_47326

# All seed_* directories directly under the case directory.
GPU=5 bash AAA_my_test/object_query_ablation_metrics/bench.sh \
  /data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1/0613pybullet_sample_001460_w002

# Validate input resolution and print the complete pipeline without inference.
bash AAA_my_test/object_query_ablation_metrics/bench.sh \
  /path/to/CASE/seed_47326 --dry-run
```

The entry point runs, in order: the seven official VBench dimensions
(cache-aware), CoTracker, SAM2, candidate and source RAFT, DINOv2/LPIPS,
non-neural metrics plus every audit overlay, strict output validation, and the
common-seed aggregate. It derives `case`, `seed`, source render and
`states.npz` from the inventory and baseline manifest, while large caches and
reports remain under `/data/gaoya/agent-data`.

Use `--overwrite` only when cached model outputs must be regenerated.
`--skip-vbench` and `--no-aggregate` are debugging controls and therefore do
not represent the default complete evaluation. Run `bench.sh --help` for the
full interface.

## Incremental M1/M2/M3 Head-Scope metrics

This mode accepts one `seed_*` directory, a case directory, or the complete
temporal-tube experiment root.  It discovers completed Top100, Bottom100 and
All-Heads M1/M2/M3 variants from their manifests and skips unchanged videos by
file signature.

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
bash AAA_my_test/object_query_ablation_metrics/bench.sh \
  /data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1 \
  --head-scope-baseline --workers 6 --watch-seconds 60
```

Reports are written to
`/data/gaoya/agent-data/outputs/object_query_ablation_metrics/head_scope_baseline_fast/<case>/seed_<seed>/report.json`;
the cross-case/seed ranking is `head_scope_baseline_fast/ranking.json`.
`impact_score_0_100` is an absolute visual intervention-strength score versus
the same-seed Baseline.  Larger means more visible change, not better/worse
generation and not greater physical error.  This fast report must not be called
CoTracker trajectory, SAM2 shape, RAFT, DINOv2, LPIPS, VBench or simulator-GT
evaluation; those remain in the complete pipeline above.

The page and JSON report also separate the quick measurements into four
independently ranked effect categories.  All scores use a `0–100` scale and
larger means a stronger change versus the same-seed Baseline:

| Category | Exact quick score | What it ranks |
|---|---|---|
| Global appearance | `100 * [0.50 * (1 - global_SSIM) + 0.50 * global_MAE]` | Whole-frame structural and pixel change |
| Target-local | `100 * target_ROI_MAE` | Change inside the frozen Baseline object tube |
| Temporal appearance | `100 * [0.40 * global_delta_MAE + 0.60 * target_ROI_delta_MAE]` | Frame-to-frame pixel-change patterns, mixing motion, appearance, deformation and flicker; never rank trajectory with this score |
| Outside-object spillover | `100 * mean(outside_object_MAE, outside_object_delta_MAE)` | Static and temporal change outside all frozen object tubes |

Each video record contains `category_scores_0_100` and
`category_ranks_within_case_seed`.  `ranking.json` additionally stores the
corresponding cross-seed means.  Categories are ranked separately; a variant
can therefore lead the target-local list without leading the spillover list.

### True Head-Scope trajectory ranking

Temporal Delta-MAE is a temporal-appearance measurement, not a trajectory
measurement.  Use the separate CoTracker mode for real object-motion ranking:

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
GPU=2 bash AAA_my_test/object_query_ablation_metrics/bench.sh \
  /data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1/0613pybullet_sample_001460_w002/seed_47326 \
  --head-scope-trajectory
```

The primary score is
`100 * mean_selected_objects(center_ADE_norm)`, where the center is the median
of at least four visible CoTracker points and the normalizer is the F00 object
bbox diagonal.  Center-FDE, four-frame velocity-vector error, PCK@5/10/20% and
common-visible coverage are reported alongside it.  A selected object must
have at least four common center frames and at least 80% coverage relative to
the Baseline-valid center frames; otherwise its score is `N/A` and it is not
ranked.  Every record includes a Baseline/Ablation point-and-trajectory overlay.
Artifacts are written under
`/data/gaoya/agent-data/outputs/object_query_ablation_metrics/head_scope_trajectory`.

The same report also contains a quality-gate-independent tracking-loss
measurement for every generated ablation:

`Track Loss = 100 * (1 - common_center_coverage)`.

It is ranked for all 108 videos. A larger value means that fewer Baseline-valid
object-center frames remain jointly observable by CoTracker. This closes the
coverage gap left by Center-ADE, but it is a tracker observability score: track
loss can be caused by disappearance, identity/appearance corruption, severe
deformation or tracker failure, so it must not be labeled true disappearance.

### Head-Scope object retention / disappearance

Use the model-backed survival mode to distinguish object-retention failure from
ordinary trajectory displacement:

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
GPU=0 bash AAA_my_test/object_query_ablation_metrics/bench.sh \
  /data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1/0613pybullet_sample_001460_w002/seed_47326 \
  --head-scope-object-survival
```

For object `o` and frame `t`, `alive(o,t)=1` exactly when all three conditions
hold: the SAM2 mask is nonempty; the mask-pooled DINOv2 cosine to the same-frame
Baseline object is above its per-object calibrated identity threshold; and the
candidate/Baseline mask-area ratio lies in `[0.25, 4.0]`. Then:

`Object Retention Failure = 100 * (1 - mean_t alive(o,t))`.

The stricter disappearance proxy is reported separately as
`SAM2 Mask Absence = 100 * mean_t 1[mask area = 0]`. It does not count an
identity replacement or size corruption as literal absence, although SAM2
tracking failure can still create a false positive and therefore requires the
overlay audit.

Single-object experiments rank that selected object. For `all_objects`, the
primary rank uses the worse of A/B; the report also preserves their mean. A
larger score means weaker object retention and is therefore an explicit
generation-failure indicator when the requested object should persist. It
includes true disappearance, identity replacement and extreme size corruption;
the per-frame overlay shows which frames fail. SAM2 F00 prompt IoU must be at
least `0.50`, the first sustained loss means three consecutive failed frames,
and terminal loss is measured over the final eight frames.

The survival report, bit-packed SAM2 masks, DINOv2 features and audit overlays
are written next to the trajectory report. Cached video signatures make the
mode incremental and safe to rerun.

Build a complete Markdown report from any generated trajectory `report.json`
without rerunning CoTracker:

```bash
/data/gaoya/miniconda3/envs/wan/bin/python \
  AAA_my_test/object_query_ablation_metrics/build_head_scope_trajectory_ranking_md.py \
  /data/gaoya/agent-data/outputs/object_query_ablation_metrics/head_scope_trajectory/0613pybullet_sample_001460_w002/seed_47326/report.json
```

The Markdown contains the exact metric definitions, M1/M2/M3 information-flow
semantics, all quality-passing scalar values, independent ranks for every
metric, target-specific Center-ADE rankings, and a separate audit table for
quality-gated `N/A` records.

## Manual stages (advanced)

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
AlexNet through torchmetrics. Official VBench scores are read from each
generation manifest rather than approximated here; `bench.sh` first fills any
missing scores with the official local VBench runner. Simulator center
trajectories and sphere-to-oriented-box contact are calculated directly from
`states.npz`.
