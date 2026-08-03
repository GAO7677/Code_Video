# PhysRVG on Physics-IQ Verified

This directory is an adapter only. It does not modify either official codebase.

## Audit result

- The PhysRVG release archive on host `118` does not contain a Physics-IQ or
  Physics-IQ Verified runner. Its official inference example is `inference.py`.
- The files `generate_physics_iq_verified_official.py`,
  `prepare_physiq_submission_30fps.py`, and
  `eval_physics_iq_verified_official.sh` beside that checkout were added later;
  they are not present in the release zip.
- The old added generator submitted all 149 output frames as five seconds. Its
  first approximately 90 frames reproduced the input condition, so that score
  included the condition and represented only about two seconds of prediction.
  It is not comparable with the strict xSSC result.
- PhysRVG's untouched official pipeline fixes two temporal latents, corresponding
  to five frames. The adapter-local pipeline copy expands only this mask so all
  temporal latents encoded from the 72-frame input are effective conditions.

## Shared benchmark protocol

The adapter consumes the exact same 198-case list used by
`run_slot_dedup_step002000_physicsiq_verified_gpu2.sh`:

`/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/inputs/bpp/verified_v2v_bpp_198.txt`

The invariant settings are BPP prompts, V2V mode, 72 condition frames at 24 FPS
(3 seconds), 512x896 resolution, 40 inference steps, guidance 5, `do_cfg=False`,
and seed 42. PhysRVG produces one 189-frame sample. Wan's temporal VAE represents
the 72-frame input as a 69-frame clean prefix; that prefix is removed and only
the following 120 predicted frames are submitted at 24 FPS (5 seconds). These
condition, inference, temporal slicing, and output settings match the xSSC run.

The local and host-118 copies of these official scoring assets have identical
SHA256 values:

- `physiq/run_physics_iq.py`
- `physiq/aggregate_runs_from_csvs.py`
- `descriptions/best_practice/descriptions_base.csv`
- `descriptions/descriptions_original.csv`

## Foreground interface

Generation plus official scoring requires an explicit physical GPU:

```bash
GPU=3 /home/gaoya/Code_Video/Code_bench/physics-IQ_my/PhysRVG/run_physrvg_verified_remote118.sh all
```

A one-case fact test uses the same path and settings:

```bash
GPU=3 MAX_ITEMS=1 RUN_NAME=physrvg-strict-bpp-run_fact01 \
  /home/gaoya/Code_Video/Code_bench/physics-IQ_my/PhysRVG/run_physrvg_verified_remote118.sh generate
```

Scoring an already complete run does not require a GPU:

```bash
/home/gaoya/Code_Video/Code_bench/physics-IQ_my/PhysRVG/run_physrvg_verified_remote118.sh score
```

The default remote submission folder is:

`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physicsiq_verified/generated_videos_5s/physrvg-strict-bpp-run_01`

The default official evaluator output is:

`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physicsiq_verified/evaluation/physrvg-strict-bpp-run_01`
