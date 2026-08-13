# GT-STC Guidance Hyperparameter Search

## Objective

Find a low-dose inference-time GT-STC configuration that improves GT-relative
object motion without destroying object trackability.  Pixel MSE and CoTracker
trajectory errors are **offline selection metrics only**; neither is
back-propagated through Wan.

## Frozen controls

- Wan2.2 TI2V, 40 denoising steps, CFG 5, sigma shift 5.
- Seed 47326, 49 frames, 704 x 1280.
- latest3350 S039 Top100 heads.
- RMS gradient normalization, maximum gradient/noise RMS ratio 1.
- Gaussian sigma 1.5 tokens in Stage 1.
- Conditioned first latent is immutable.

## Stage 1A: safe-dose screen

- Calibration case: `0613pybullet_sample_001460_w002`, target `object_A`.
- Guidance window: steps 0-9.
- Loss modes: Region, Point, Combined (equal Region/Point gradient mean).
- Guidance scales: 0.005, 0.01, 0.02, 0.05.
- Total: 12 guided videos plus the frozen same-seed Baseline.

Hypothesis: if the current failures are primarily excessive intervention dose,
then lowering lambda below 0.05 will reduce Track Loss while retaining or
improving GT-relative trajectory error.

## Pre-registered selection order

1. Trajectory quality gate pass is mandatory for an acceptable winner.
2. Minimize future Track Loss.
3. Minimize gated Center-ADE / D0; raw ADE is diagnostic only when the gate fails.
4. Minimize target-tube GT MSE.
5. Use outside-object GT MSE as a spillover guardrail.

MSE is measured on future latent-anchor RGB frames 4, 8, ..., 48 in `[0,1]`
intensity units. Target MSE uses the selected GT mask dilated by 16 pixels.
Outside-object MSE excludes the union of all GT object masks dilated by 16
pixels. Full-frame MSE is reported only as a sanity check.

## Adaptive continuation

- If at least two Stage 1A configurations pass the trajectory gate, retain the
  Pareto candidates and compare windows 0-4, 0-19, and 0-39.
- If none pass, do not expand the grid blindly: add lambda 0.001 and 0.002, then
  rerun the same First10 screen.
- Refine Region/Point weight and point Gaussian sigma only after a safe
  `(lambda, window)` has been identified.
- Re-test finalists on the other two 0613 cases, then on the six eligible
  non-0613 GT cases.  Results on the calibration case alone are not a general
  optimum.

