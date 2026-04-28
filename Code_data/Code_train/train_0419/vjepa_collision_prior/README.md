# V-JEPA Collision Prior

This directory contains a retrieval-style experiment for testing whether a frozen
V-JEPA world model is sensitive to collision events and whether its predicted
future latent is closest to the true post-collision future.

## Goal

For each collision sample:

1. Take a pre-collision `context` clip.
2. Use a frozen backbone to produce either:
   - a predicted future latent (`vjepa_predictor`), or
   - a context latent without prediction (`vjepa_context`, `videomae`, `dino`, `clip`).
3. Rank a set of candidate future clips containing:
   - the true future clip,
   - same-video wrong-time negatives,
   - same-object counterfactual case negatives,
   - no-collision negatives,
   - random negatives.

If the predicted future latent is consistently closest to the real future latent,
the model is a viable collision-window dynamic prior.

## Files

- `build_collision_manifest.py`
  Builds a JSONL retrieval manifest from Genesis rigid data.
- `eval_collision_retrieval.py`
  Runs ranking evaluation and writes per-query results plus summary metrics.
- `lib/data_utils.py`
  Dataset scanning, collision event parsing, kinematics access, bbox utilities.
- `lib/backends.py`
  Frozen feature backends and pooling logic.
- `lib/metrics.py`
  Ranking metrics.

## Default assumptions

- Primary event: first `object-object` collision in `collision_events.json`.
- Collision time `t_c`: the event `start_frame`.
- Context clip: `frames[t_c-L : t_c]`.
- Future clip for horizon `h`: `frames[t_c+h : t_c+h+W]`.

You can switch to include environment collisions with
`--include-environment-collisions`.

If `t_c < L`, the builder left-pads the context with repeated frame `0` so the
query still has a fixed clip length while remaining strictly pre-collision.

## Counterfactual case interface

The manifest builder keeps an explicit case-id hook for same-object
counterfactuals:

- `--counterfactual-caseids 000,001,002`
- `--counterfactual-scene-pattern '{object_id}__case{caseid}*'`

This lets you point the builder at future Genesis runs once more case IDs are
available. Missing requested case IDs are recorded in the manifest metadata
instead of hard-failing.

## Example: build manifest

```bash
/data/gaoya/miniconda3/envs/vjepa2/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_train/train_0419/vjepa_collision_prior/build_collision_manifest.py \
  --dataset-root /data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases \
  --eval-manifest /data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/mytest/manifest.jsonl \
  --output /home/gaoya/Code_Video/Code_data/Code_train/train_0419/vjepa_collision_prior/manifests/mytest_collision_retrieval.jsonl \
  --context-length 8 \
  --future-width 4 \
  --horizons 2,4,8,12 \
  --num-random-negatives 4 \
  --counterfactual-caseids 000,001,002,003,005,006,007,900,901
```

## Example: random baseline smoke test

```bash
/data/gaoya/miniconda3/envs/vjepa2/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_train/train_0419/vjepa_collision_prior/eval_collision_retrieval.py \
  --manifest /home/gaoya/Code_Video/Code_data/Code_train/train_0419/vjepa_collision_prior/manifests/mytest_collision_retrieval.jsonl \
  --backend random \
  --output-dir /home/gaoya/Code_Video/Code_data/Code_train/train_0419/vjepa_collision_prior/outputs/random_smoke
```

## Example: V-JEPA predictor

```bash
/data/gaoya/miniconda3/envs/vjepa2/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_train/train_0419/vjepa_collision_prior/eval_collision_retrieval.py \
  --manifest /home/gaoya/Code_Video/Code_data/Code_train/train_0419/vjepa_collision_prior/manifests/mytest_collision_retrieval.jsonl \
  --backend vjepa_predictor \
  --vjepa-checkpoint /data/gaoya/ckpt/Sylvest-vjepa2-vit-g/vitg-384.pt \
  --pooling object_pair \
  --output-dir /home/gaoya/Code_Video/Code_data/Code_train/train_0419/vjepa_collision_prior/outputs/vjepa_predictor_object_pair
```

## Backends

Implemented backends:

- `vjepa_predictor`
- `vjepa_context`
- `state_extrap`
- `random`

Pluggable HuggingFace backends:

- `videomae`
- `dino`
- `clip`

For the HuggingFace backends, pass the model IDs explicitly if you want a
specific checkpoint:

- `--videomae-model-id ...`
- `--dino-model-id ...`
- `--clip-model-id ...`

## Output metrics

The evaluator writes:

- `summary.json`
- `per_query.jsonl`

Main metrics:

- `top1_accuracy`
- `mean_gt_rank`
- `mean_positive_negative_margin`
- `per_horizon`

## Notes

- Object and object-pair pooling use candidate future bbox tracks to define the
  support region on token grids. This is intentional for retrieval: the scoring
  function is allowed to use each candidate's own support mask.
- The explicit state extrapolation baseline ranks candidate futures by negative
  constant-velocity trajectory MSE in COM space, not by latent similarity.
