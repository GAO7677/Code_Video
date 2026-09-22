# Initial-contact gate and targeted replay

The RGB-to-Bullet evaluator now admits initial contacts before any simulation step. Strict mode rejects penetration exceeding 1 mm with an auditable `InitialContactError`. Existing valid states are unchanged. No settling, time-step change, GT geometry substitution, or velocity adjustment is performed.

The explicit `support_aligned` diagnostic admits only confirmed upward horizontal support contacts and normal velocity at most 0.01 m/s. It moves the sphere upward by at most half its known radius (55 mm), preserves horizontal position/orientation/velocities, refreshes collision detection, and rejects remaining intersections or lost support. The original and adjusted positions and every initialization reset are recorded. This is not strict GT-state C.

The frozen evaluation contains 27 affected C cases, each with 3 omega policies. All 81 strict runs were instrumented and rejected before `stepSimulation`. 54 aligned runs across 18 cases completed; 27 runs across 9 cases were blocked (6 cases exceed displacement limit; 3 have non-horizontal platform contact normals). No blocked run generated a trajectory. Maximum admitted initial penetration: 0.010 mm; maximum sampled rollout penetration: 0.01005 mm. Contact samples observe 240 Hz API boundaries, not hidden engine substeps.

For aperture_g01_v0460, GT-omega C moved upward 7.762 mm. Maximum height changed from 0.56014 m to 0.117855 m. New diagnostic ADE/FDE: 0.01811 / 0.05945 m; these must not be merged with strict C statistics. A/B/D results remain frozen, with their original provenance.

Outputs: `/data/gaoya/agent-data/outputs/physvideo_context_rgb_to_pybullet_20260921_v1/contact_alignment_v1`; viewer: sibling `overlay_viewer_v2`. Original outputs and `overlay_viewer_v1` are retained. Only a navigation banner is added to the old HTML.

Validation: `tests/test_initial_contact_gate.py` passes all 6 real Bullet tests. Browser check loaded 36 cases and all 6 stages without runtime errors or horizontal overflow at desktop/mobile widths.

Reproduction (from project root; output destinations must not exist):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B tests/test_initial_contact_gate.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B code/rerun_context_contact_alignment.py --root /data/gaoya/agent-data/outputs/physvideo_context_rgb_to_pybullet_20260921_v1 --output /data/gaoya/agent-data/outputs/physvideo_context_rgb_to_pybullet_20260921_v1/contact_alignment_v1 --viewer /data/gaoya/agent-data/outputs/physvideo_context_rgb_to_pybullet_20260921_v1/overlay_viewer_v2
```

The original batch evaluator fails closed on invalid initialization. Use the targeted replay runner for explicit aligned diagnostics and structured blocked-case outputs; do not disable the gate to continue a batch.
