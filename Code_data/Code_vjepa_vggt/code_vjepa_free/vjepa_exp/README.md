# V-JEPA -> Wan Pre-Validation

This directory contains the staged, pre-validation-first experiment code for
testing whether V-JEPA features are useful as training-free guidance for
Wan2.2-TI2V-5B.

Scope of this stage:

- Generate Wan baseline videos for a fixed prompt/seed set
- Extract multi-layer V-JEPA features from baseline videos
- Build offline V-JEPA signals
- Log lightweight Wan attention statistics
- Produce a go/no-go analysis report before any attention injection work

Guardrails:

- Do not modify the official V-JEPA repo in
  `/home/gaoya/Code_Video/vjepa2-main`
- Do not store large outputs under `/home/gaoya`
- Write generated artifacts under `/data/gaoya`

Suggested layout for outputs:

- Baseline videos:
  `/data/gaoya/AAA_test_video/0626vjepa_free/test/<run_name>`
- Analysis artifacts:
  `/data/gaoya/agent-data/outputs/vjepa_wan_precheck/<run_name>`

Current scripts:

- `baseline_plan.py`
  Writes the original vjepa_exp TI2V manifest format:
  prompts, seeds, image paths, output paths, and shared Wan inference args.
- `generate_wan_baseline.py`
  Keeps the existing vjepa_exp manifest input format, but swaps the underlying
  base model execution to the same runtime used by
  `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wanti2v.py`,
  via `code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime`.
- `extract_vjepa_features.py`
  Extracts per-video multi-layer V-JEPA features and simple derived signals.
- `run_manifest_extract.py`
  Runs feature extraction over every generated video listed in a manifest.
- `analyze_signals.py`
  Aggregates extracted features and optional evaluation signals into a compact
  summary report.

Typical execution order:

1. Create a manifest with `baseline_plan.py`
2. Generate baseline videos with `generate_wan_baseline.py`
3. Extract V-JEPA features with `run_manifest_extract.py`
4. Aggregate reports with `analyze_signals.py`

Environment notes:

- Wan generation should run in the `wan` environment.
- V-JEPA extraction should run in the `vjepa2` environment.
- The baseline runner now uses the official-style runtime under
  `/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main`
  through `code_vjepa_vggt.AAAinfer`, while preserving the original
  `vjepa_exp` manifest input structure.
- V-JEPA extraction assumes the local repo at
  `/home/gaoya/Code_Video/vjepa2-main`.

Notes:

- This is intentionally the minimum viable experiment scaffold.
- Wan attention logging and direct injection are not implemented yet in this
  first pass.
