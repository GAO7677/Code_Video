# Simulation Dataset Plan

## Goal

Replace the current real-video-heavy training source with a controllable simulation dataset built on top of the old `0526dp` PyBullet pipeline. The new dataset should be suitable for training object-state-conditioned future video generation rather than only for evaluation demos.

The old generators under `/home/gaoya/Code_Video/Code_data/Code_try0526/physics_sim/` already provide:

- rigid-body simulation with `PyBullet`
- consistent rendering with `Pyrender`
- scenario-level metadata such as physical parameters and rendering config

The main limitation is that the old format only stores `mp4 + summary json`. It does not preserve per-frame object-level ground truth needed by the current `phys_state_video` training pipeline.

## Design Principles

1. Keep the simulator simple and stable.
2. Increase motion diversity before increasing photorealism.
3. Preserve exact object states from the simulator and derive pseudo-state targets from them.
4. Separate training truth from evaluation summaries.
5. Keep the rendered videos visually clean enough that motion is easy to learn.
6. Start with rigid bodies only; defer cloth, fluid, articulated hands, and human-centric scenes.

## Recommended Simulator Stack

- Physics: `PyBullet`
- Renderer: `Pyrender` for fast offline rendering
- Scene description: Python scenario configs
- Export format:
  - raw simulation package for full truth
  - derived `phys_state_video` episodes in `.npz`

This is the lowest-risk path because it extends the known-working `0526dp` code instead of switching engines.

## Dataset Scope

### Phase S1: single-room rigid-body interactions

This phase is intended to support the current project end-to-end.

Object categories:

- sphere
- cube
- rectangular block
- cylinder
- capsule
- cone

Appearance families:

- rubber
- wood
- metal
- plastic
- matte painted

Motion families:

- straight translation
- ballistic arc
- rolling
- sliding
- bouncing
- collision transfer
- multi-bounce wall/floor interaction
- object entering or leaving view
- partial occlusion behind another object
- chain reaction with 3 to 6 objects

Object count strata:

- 1 object
- 2 objects
- 3 to 4 objects
- 5 to 6 objects

We should deliberately overweight `2-4` objects because this is the most useful regime for state-conditioned generation and tracking.

## Scenario Families

Each sample should belong to exactly one primary scenario family so the dataset remains analyzable.

### F1. Single-object kinematics

Purpose:

- teach smooth trajectory continuation
- teach depth-scale consistency
- teach visibility changes at borders

Examples:

- thrown ball with gravity
- rolling cylinder
- sliding block with friction
- bouncing sphere with different restitution

### F2. Two-object interaction

Purpose:

- teach contact events
- teach identity persistence after collision
- teach relative motion induced by impact

Examples:

- ball hits block
- sphere-sphere collision
- cylinder hits cube
- glancing collision with deflection

### F3. Multi-object chain reaction

Purpose:

- teach multi-body causal propagation
- force the predictor to handle several active tracks

Examples:

- Newton-cradle-like line
- one object hits a small stack
- one rolling object triggers several blocks

### F4. Occlusion and reappearance

Purpose:

- teach visibility and existence separately
- improve non-flickering state prediction

Examples:

- moving object passes behind static foreground block
- two moving objects cross and occlude each other
- object leaves frame and re-enters

### F5. Support and drop events

Purpose:

- teach discrete event transitions
- diversify vertical motion

Examples:

- object dropped onto floor
- object falls onto another object and bounces
- stacked objects lose support and topple

## Parameter Randomization

Each sample should randomize within bounded ranges instead of enumerating a tiny grid.

Physical parameters:

- mass
- restitution
- lateral friction
- rolling friction
- linear damping
- angular damping
- gravity magnitude

Geometric parameters:

- shape type
- object size
- aspect ratio
- initial position
- initial orientation
- initial linear velocity
- initial angular velocity

Scene parameters:

- number of active objects
- number of static occluders
- floor material
- wall layout

Camera parameters:

- azimuth
- elevation
- distance
- focal length or field of view
- small camera jitter per sequence, but no violent hand-held motion in phase S1

Rendering parameters:

- light direction
- light intensity
- background tone
- material color

The appearance randomization should be moderate. The old `ball_block_appearance` study showed that overly different lighting and background can strongly perturb downstream visual metrics. For training, some variation is useful, but extreme appearance gaps should be avoided in the first pass.

## Ground Truth To Preserve

This is the critical upgrade relative to the old `0526dp` data.

For every frame and every tracked object, preserve:

- `track_id`
- `category_id`
- `shape_type`
- `material_type`
- `is_dynamic`
- `is_occluder`
- `position_world`: `[3]`
- `rotation_quat_world`: `[4]`
- `linear_velocity_world`: `[3]`
- `angular_velocity_world`: `[3]`
- `projected_center_xy`: `[2]`, normalized to `[0, 1]`
- `bbox_xyxy`: `[4]`, normalized to `[0, 1]`
- `mask_rle` or compact binary mask path
- `depth_value`
- `visibility`
- `existence`
- `contact_flag`
- `contact_impulse` if available

Per-sequence truth:

- prompt or caption
- scenario family
- simulator parameters
- camera intrinsics
- camera extrinsics
- render settings
- split
- seed

Optional but recommended:

- depth map
- object id map
- surface normal map
- collision event list
- occlusion event list

## Derived Training Targets

The simulator exports exact truth, then a conversion step writes training episodes that match the current `phys_state_video` shape contract.

Current training arrays:

- `context_frames`: `[K, 3, H, W]`
- `future_frames`: `[T, 3, H, W]`
- `context_states`: `[K, N, 10]`
- `future_states`: `[T, N, 10]`
- `context_boxes`: `[K, N, 4]`
- `future_boxes`: `[T, N, 4]`
- `appearance`: `[N, A]`
- `camera`: `[K, C]`

Recommended state layout remains:

1. `center_x`
2. `center_y`
3. `relative_depth`
4. `log_scale`
5. `vel_x`
6. `vel_y`
7. `depth_vel`
8. `visibility`
9. `existence`
10. `confidence`

For simulation data:

- `confidence` can initially be `1.0`
- `existence` is distinct from `visibility`
- `relative_depth` should be camera-relative depth, normalized per sequence
- `log_scale` should be computed from projected box or mask area

Recommended auxiliary truth kept outside the `.npz`:

- exact world states
- masks
- depth maps
- contact events

This lets us later train stronger state extractors or physics-aware losses without regenerating the videos.

## Directory Layout

Recommended raw export root:

`/data/gaoya/AAA_test_video/Dataset_physV/sim_objstate_v1_raw/`

Recommended derived training root:

`/data/gaoya/AAA_test_video/0529/phys_state_video/datasets/sim_objstate_v1_episodes/`

Suggested raw layout:

```text
sim_objstate_v1_raw/
  train/
    F1_single_object/
      sample_000001/
        video.mp4
        meta.json
        states.npz
        masks/
        depth/
    F2_two_object/
    F3_chain_reaction/
    F4_occlusion/
    F5_drop_support/
  val/
  test/
  manifests/
```

Suggested derived episode layout:

```text
sim_objstate_v1_episodes/
  train/
    episode_000001.npz
    episode_000001.json
  val/
  test/
  manifest.json
```

## Dataset Size Recommendation

### Pilot

Use a pilot before large-scale generation:

- train: `2,000`
- val: `200`
- test: `200`

Distribution:

- F1: `25%`
- F2: `30%`
- F3: `20%`
- F4: `15%`
- F5: `10%`

### First usable training set

- train: `20,000`
- val: `1,000`
- test: `1,000`

This is enough to test whether exact simulator states materially help the predictor and adapter.

### If phase S1 works

Scale toward:

- train: `50,000` to `100,000`

Only do this after verifying that the pilot improves:

- future state prediction error
- identity persistence
- scale-depth consistency
- qualitative motion plausibility

## Split Strategy

Do not split randomly after generation without constraints.

Required split rules:

- no duplicate seeds across splits
- hold out some object appearance combinations
- hold out some camera viewpoints
- hold out some scenario parameter ranges
- keep at least one test subset with more objects than seen in most training samples

Suggested evaluation subsets:

- in-domain random test
- unseen appearance test
- unseen viewpoint test
- unseen parameter-range test
- higher-object-count stress test

## Caption Policy

Each sequence should have a short caption describing object category, motion, and interaction, for example:

- `a red rubber ball rolls forward and bounces off a wooden block`
- `two metal cylinders collide and separate`
- `a blue cube falls behind a larger block and reappears`

Do not use overly verbose captions. The caption should expose the main motion semantics but not every exact numeric parameter.

## Why This Dataset Fits The Current Method

It directly supports all major components of the project:

- state extraction supervision:
  exact boxes, masks, centers, depth, visibility
- future state prediction:
  exact future trajectories and event transitions
- consistency projection:
  exact scale-depth and velocity continuity checks
- state-conditioned generation:
  clean condition maps and object memory tokens

The current real-video pipeline had two main issues:

- too many human-centric or static clips
- pseudo states are noisy and often weakly aligned with true object dynamics

The simulation dataset addresses both issues while preserving perfect object identity and motion truth.

## Immediate Implementation Plan

1. Refactor the old `0526dp` simulator into a reusable generator with scene configs.
2. Add a frame-level exporter for object states, boxes, masks, depth, and camera.
3. Implement the five scenario families above.
4. Generate a pilot set of about `2,400` sequences.
5. Convert raw simulation outputs into `phys_state_video` training episodes.
6. Visualize the pilot set locally before training.
7. Train the predictor first, then the adapter, then compare against the current baseline.

## Non-Goals For The First Pass

- humans
- deformable objects
- fluids
- language-rich multi-step narratives
- photorealistic textures
- large outdoor scenes

These add cost and instability without helping the current object-state-conditioning question.
