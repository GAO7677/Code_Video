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
  Writes the prompt/seed experiment manifest and output layout.
- `extract_vjepa_features.py`
  Extracts per-video multi-layer V-JEPA features and simple derived signals.
- `analyze_signals.py`
  Aggregates extracted features and optional evaluation signals into a compact
  summary report.

Notes:

- This is intentionally the minimum viable experiment scaffold.
- Wan attention logging and direct injection are not implemented yet in this
  first pass.
