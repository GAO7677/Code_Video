# Geometry Diagnostics

Minimal per-case geometry diagnostics for benchmark outputs produced by
`batch_eval_lora.py` and `batch_eval_vace.py`.

This tool is designed to answer a narrow question:

- when an object suddenly becomes larger in generated video, is the change more
  consistent with normal perspective motion, camera zoom/drift, object-only
  scale drift, or tracking/visibility failure?

## Scope

Current implementation favors simple, verifiable behavior over broad coverage.

- Synthetic cases with ground-truth supervision:
  - `kubric_tfds_movi-d`
  - `GenesisRigid` / `version_1_genesis_rigid_data_all_cases`
- Generic fallback for other datasets:
  - estimate a foreground proxy from frame differencing against the last context
    frame
  - estimate relative depth from grayscale luminance proxy
  - keep results marked as approximate

## Outputs

For each case the script writes:

- `diagnostics.json`
- `curves.csv`
- `curves.png`

## Notes

- The current target-object selection is intentionally simple:
  - MOVI-D: largest visible non-background segmentation id on the last context
    frame
  - Genesis: `role=target` if available, otherwise largest visible non-zero
    segmentation id on the last context frame
- This is enough to validate the pipeline on a few benchmark cases before
  scaling to all 300 outputs.
