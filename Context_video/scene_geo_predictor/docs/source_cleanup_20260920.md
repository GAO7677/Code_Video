# Cleanup Record · 2026-09-20

This is an audit record for the first approved cleanup pass. No training or
evaluation output under `validation_20260920/` was touched.

## Removed

```text
bank20_full_render_20260919/renders/              876 MiB
bank20_full_render_20260919/context_inputs/       143 MiB
bank20_full_render_20260919/videos/                 2.8 MiB
raw_probe/                                          77 MiB
utonia_feature_pilot/                               13 MiB
validation_20260919_v3/diagnostic_cpu_fit_door6/   6.8 MiB
validation_20260919_v3/alignment_comparison/       1.4 MiB
validation_20260919_v3/alignment_comparison_v2/    1.9 MiB
validation_20260919_v3/alignment_comparison_v3/    1.9 MiB
__pycache__/                                       412 KiB
```

The bank20 metadata was retained:
`bank20_full_render_20260919/{README.md,selection.json,render_manifest.json,samples/}`
and `view/{index.html,data.json}`.  The three bank20 asset symlinks under
`view/` were removed so that no dangling links remain.  The final alignment
comparison remains at `validation_20260919_v3/alignment_comparison_final/`.

## Retained and protected

- `small_trial_120/`, `controlled_scene_eval_12/` and `two_round_release/`
  remain intact because round1/round2 use symlinks into these directories.
- `validation_20260920/` remains intact; its paired training job was active
  during cleanup.
- Final release weights, runtime files, reports, manifests and documentation
  remain intact.

The removed raw probe and pilot assets can be regenerated from the preserved
scripts and source inputs, but their old preview links are no longer live.
