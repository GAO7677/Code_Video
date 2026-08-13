# Stage 1: Causal xSSC State Audit

> Active causal-adaptation branch (2026-08-13): model-only fork from the
> noncausal MOVi-C `step-025000`, physical GPU 0, 24 raw frames / 12 causal
> states, optimizer steps 25000→35000.  A launch-time capacity probe chooses
> the largest safe single-GPU microbatch and changes gradient accumulation to
> keep the effective batch fixed at 384.  Outputs and W&B run are independent
> from both the source run and the earlier planned step-050000 branch.

## Scope

Stage 1 answers only three questions:

1. Does an xSSC object state remain temporally stable when every representation is
   computed under strict causal information constraints?
2. Does one current object state contain enough motion information, or is object
   history required?
3. After controlling for object history and model capacity, does information from
   other objects improve future-state prediction?

No ODE, Euler/RK4 integration, collision-specific model, Newtonian coordinate
constraint, current feature-query decoder metric, or future per-frame identity
matching is part of Stage 1.

## Fixed data and time axis

- Dataset: official local MOVi-C 1.0.0.
- Official splits: train 9737, validation 250, test 999 videos.
- Every video: 24 raw frames at 12 FPS.
- V-JEPA tubelet size: 2 raw frames.
- xSSC states: 12 states at 6 Hz, labelled by raw frames
  `[1, 3, 5, ..., 23]` (the second frame of each tubelet).
- xSSC trajectory shape: `[12, 11, 512]`.
- Static channels: `[0:384]`; dynamic channels: `[384:512]`.
- States 0--3 are the observed/calibration prefix. States 4--11 are the
  held-out future for the longest rollout.

MOVi-C does not provide 32--64 raw frames per official example. Stage 1 must not
create longer trajectories by joining videos or by overlapping tubelets. Longer
trajectories require a separately generated Kubric dataset and are outside this
implementation.

## Strict causal trajectory extraction

For tubelet state `t`, V-JEPA receives exactly raw frames `x[:2*t+2]`. Only the
last tubelet feature from that prefix is retained. The 12 retained features are
stacked in time and passed through the recurrent xSSC transition. xSSC at time
`t` therefore consumes only features and slots at times `<=t`.

The existing repeated-prefix encoder is the Stage 1 reference implementation.
It is deliberately preferred over a faster block-causal mask until a separate
parity test proves equivalence. The slot-only extraction path must skip the
feature decoder entirely and accept only the initial bbox condition, not a
future sequence of conditions.

Before a cache is accepted, future perturbation tests replace, shuffle, and
zero raw frames after multiple cut points. Frozen V-JEPA features and xSSC slots
at or before each cut point must remain equal within the configured numeric
tolerance.

## Causal-adapted representation checkpoint

Two checkpoints are reported separately:

- **Post-hoc causal control:** load the current noncausal xSSC weights and switch
  feature extraction to repeated-prefix V-JEPA. This is diagnostic because the
  xSSC head sees a shifted feature distribution.
- **Causal-adapted checkpoint:** initialize from the selected converged MOVi-C
  checkpoint, train with repeated-prefix V-JEPA and full 24-frame clips, and
  select the checkpoint using the official validation split only.

The original decoder may remain an observed-prefix representation-learning loss
during causal adaptation. It is never called during cache extraction, state
prediction, or future evaluation.

## Offline cache

Large artifacts live below `/data/gaoya/agent-data`:

```text
/data/gaoya/agent-data/cache/xssc_stage1_causal_state/
/data/gaoya/agent-data/checkpoints/xssc_stage1_causal_state/
/data/gaoya/agent-data/outputs/xssc_stage1_causal_state/
```

Each cache record stores fixed-shape tensors:

- causal slots `[12, 11, 512]`;
- slot aggregation attention `[12, 11, 16, 16]`;
- GT masks `[12, 10, 16, 16]`;
- GT 3-D position/velocity `[12, 10, 3]`;
- GT 2-D image position `[12, 10, 2]`;
- GT bbox `[12, 10, 4]`;
- visibility `[12, 10]` and object-valid mask `[10]`;
- bbox-conditioned slot-valid mask `[11]` (padded zero-condition slots excluded);
- one prefix-oracle mapping and one boundary-only frozen mapping `[11]`;
- source split/index/video name and an immutable provenance manifest.

RGB frames are not duplicated in the cache. Visualizers reopen the source
TFRecord from the stored split and index.

## Identity protocol

The prefix-oracle mapping minimizes mean mask-IoU cost over states 0--3. The
boundary-only mapping uses state 3 only. Both mappings are solved exactly once,
include dummy assignments, and are frozen for all future steps. Predictor
training always follows native recurrent slot indices; mappings only attach a
fixed GT identity for physical evaluation.

Future per-frame Hungarian is allowed only as a labelled diagnostic ceiling. It
must never reorder predictor inputs or targets and must never contribute to a
headline metric.

## Factorial predictor matrix

Representations:

- `dyn`: predict the 128 dynamic channels.
- `dyn_static`: static channels are a fixed condition; predict only dynamic
  channels and keep static fixed during rollout.
- `full`: predict all 512 channels with equal static/dynamic loss weights.

History settings: `H in {1, 2, 4}`.

Context settings:

- `individual`: each slot passes through the shared context block as a singleton
  set, so no cross-object information is available.
- `set`: all non-padded bbox-conditioned slots pass through the same context
  block together; zero-condition padded slots are attention-masked.

Both context settings instantiate the exact same modules and parameter count.
The only difference is tensor grouping. If the set-context effect is positive,
a separately labelled shuffled-context diagnostic should be added before making
a semantic interaction claim; it is not part of the primary factorial matrix.

Predictor architecture:

1. representation-specific input adapter to 256 dimensions;
2. two-layer temporal Transformer, 8 heads, FFN 1024, maximum history 4;
3. two-layer shared context Transformer, 8 heads, FFN 1024;
4. zero-initialized residual head predicting the next latent state.

Predictors are trained only with one-step normalized latent losses. Ground-truth
physical quantities are not predictor targets. Open-loop evaluation feeds each
predicted state back as input for horizons 1, 2, and 4; horizon 8 is explicitly
reported as a single-origin stress test.

## Frozen ground-truth probes

For each representation, simple frozen probes map real causal slots to:

- 3-D position;
- 3-D velocity;
- 2-D image position;
- bbox;
- presence.

Probe ceilings on real slots are reported before predictor results. Predicted
future slots are then passed through the same frozen probes. Position, velocity,
bbox, and presence targets never enter predictor training.

## Metrics and independent unit

Primary H1b metric: normalized 3-D velocity RMSE.

Secondary metrics: 3-D position RMSE, normalized 2-D center ADE/FDE, velocity
vector error, bbox IoU, presence balanced accuracy/F1, normalized latent MSE,
and latent cosine distance.

H1a audit: future-perturbation differences, static/dynamic drift, adjacent
same-object cosine, fixed-assignment mask IoU, ID switch rate, track coverage,
and fixed-vs-per-frame-oracle gap.

The video is the independent statistical unit. Objects, origins, and horizons
are averaged within video before paired video bootstrap. Formal comparisons use
three predictor seeds, 10,000 paired bootstrap samples, and Holm correction for
`H=2 vs H=1` and `H=4 vs H=1`.

## Required outputs

1. Data/checkpoint/cache provenance table.
2. Future perturbation and causal-invariance table.
3. Temporal stability table for boundary-frozen, prefix-oracle, and labelled
   per-frame oracle ceiling.
4. Frozen GT-probe ceiling table by representation.
5. `3 representations x 3 histories x 2 context settings` one-step table.
6. Paired history and context contrasts with confidence intervals.
7. Open-loop 1/2/4-step table and separate 8-step stress table.
8. Causal slot overlay with fixed identity colors, drift plots, history/context
   ablations, predicted-vs-GT trajectories, and failure cases.

## Acceptance gates

Stage 2 is not started unless all of the following hold:

- future perturbation invariance passes for V-JEPA features and xSSC slots;
- fixed-identity trajectories have usable coverage and a bounded oracle gap;
- real-slot frozen probes recover motion quantities above trivial baselines;
- at least one predictor beats latent copy on held-out videos;
- history/context conclusions are stable across seeds and video-level bootstrap.
