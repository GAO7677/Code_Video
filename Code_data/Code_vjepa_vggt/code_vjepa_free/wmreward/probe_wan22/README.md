# Wan2.2 Probe Scaffold

This directory contains the first-stage probing code for testing whether
`WMReward / V-JEPA surprise` can be decoded from `wan2.2-ti2v-5b`
intermediate representations.

Scripts:

- `extract_probe_features.py`
  - Generates a small set of Wan videos from manifest prompts.
  - Captures intermediate transformer block features at selected denoising
    steps and layers.
  - Saves one `probe_features.pt` plus `meta.json` per sample.
- `run_probe_smoke.py`
  - Thin wrapper around `extract_probe_features.py`.
  - Intended for single-sample smoke tests.
- `build_probe_index.py`
  - Builds a flat CSV/JSONL index over extracted feature files.

Current pooling saved per `(step, branch, layer)`:

- `h_post_global_mean`
- `delta_h_global_mean`
- `h_post_frame_mean`
- `delta_h_frame_mean`
- token-level L2 summary stats

Default output location for large artifacts:

- `/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22`
