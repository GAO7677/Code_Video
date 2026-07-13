# PhysicIQ Wan v4

This package is the implementation boundary for the reviewed v4 design. It is
separate from `train0705_kubric_no_gt_box` so v3 remains reproducible.

The full audit and design documents are stored at:

`/data/gaoya/agent-data/outputs/physiq_wan_conditioning_v4_20260713`

## Architecture boundary

The implementation must preserve two condition paths:

1. Dense latent-aligned physics controls for trajectories, depth, occupancy,
   contact, and confidence.
2. Sparse object, material, event, and relation tokens with regional supports.

Do not expose a single untyped `object_context` as the main v4 API. The tensor
contracts in `contracts.py` are the required boundary between perception,
future prediction, rasterization, and Wan injection.

## Planned modules

- `future_state_predictor.py`
- `physics_control_rasterizer.py`
- `dense_context_adapter.py`
- `regional_relation_attention.py`
- `condition_scheduler.py`
- `train_stage_a_future_state.py`
- `train_stage_b_wan_adapter.py`
- `infer_physiq_wan_v4.py`

Modules are added only after their matched ablation and contract tests are
defined in the design review.

## F4 training-free experiment

`conditioned_residual.py` implements a two-pass denoising controller. At each
step it computes CFG once with the v3 object branch and once with that branch
disabled, then applies only the object-conditioned difference inside a
prefix-derived spatial support. `infer_localized_residual.py` installs the
controller without changing the v3 inference entrypoint or checkpoint format.

The initial support is a deliberately conservative union of all object boxes
observed in the context. Dilation is relative to each observed box's width and
height, not to the full image. The union is repeated only over future latent
frames; clean context latents are always excluded. No future box, target frame,
or benchmark metric is consumed by the controller.
