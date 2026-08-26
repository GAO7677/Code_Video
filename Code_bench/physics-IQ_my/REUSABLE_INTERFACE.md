# Reusable Physics-IQ Verified Interface

The reusable layer separates model-specific generation from official scoring.
It does not modify the official benchmark repository.

## Contract

Every model adapter must produce one run folder containing only:

```text
0001_....mp4
0002_....mp4
...
0198_....mp4
```

The exact names are taken from the selected official descriptions CSV. All 198 videos must:

- be exactly `5.000 +/- 0.001` seconds,
- use one shared positive integer FPS,
- represent take-1 outputs,
- exclude V2V conditioning frames,
- use one prompt setting and one input mode per run.

Generation resolution, sampling steps, guidance, negative prompt, and model preprocessing belong to the adapter. They must remain fixed across the four independent runs.

For every newly launched Physics-IQ Verified `P0` (`bpp` + `v2v`) run, use the
canonical long negative prompt from
`common/physicsiq_p0_prompt.env`. Its version is
`physrvg-72f-adapted-long-v1` and its SHA-256 (without a trailing newline) is
`ce96e0324e4b54ce4b6e867f669ca520952e1a34cc116543516b1897f0d3c47e`. This is
the project comparison convention; it is not an additional official evaluator
input requirement. Historical run-specific configs remain frozen for
reproducibility.

## Generic commands

Validate any model output without running metrics:

```bash
bash /home/gaoya/Code_Video/Code_bench/physics-IQ_my/launch_verified_benchmark.sh \
  validate /path/to/model-bpp-run_01
```

Run official Verified scoring for one to four runs:

```bash
PHYSIQ_MODEL_NAME=my-model \
bash /home/gaoya/Code_Video/Code_bench/physics-IQ_my/launch_verified_benchmark.sh \
  score \
  /path/to/my-model-bpp-run_01 \
  /path/to/my-model-bpp-run_02 \
  /path/to/my-model-bpp-run_03 \
  /path/to/my-model-bpp-run_04
```

The `score` command first runs the reusable strict validator, then calls:

- official `physiq/run_physics_iq.py`,
- official `aggregate_runs_from_csvs.py --score-type verified`.

## Adapter launch

The current xSSC model can be launched through the same interface:

```bash
bash /home/gaoya/Code_Video/Code_bench/physics-IQ_my/launch_verified_benchmark.sh \
  generate \
  /home/gaoya/Code_Video/Code_bench/physics-IQ_my/adapters/xssc_full_sa_no_object.sh \
  1 \
  /path/to/checkpoint-directory \
  3
```

For another model, copy:

```text
/home/gaoya/Code_Video/Code_bench/physics-IQ_my/adapters/model_adapter_template.sh
```

The adapter receives stable `PHYSIQ_*` environment variables and must write the completed run-folder path to `PHYSIQ_RESULT_FILE`. The generic launcher then validates it automatically.

## Environment variables

- `PHYSIQ_WORKSPACE`: shared data/output root.
- `PHYSIQ_DATASET`: Physics-IQ Verified dataset root.
- `PHYSIQ_MODEL_NAME`: stable model identifier.
- `PHYSIQ_PROMPT_SETTING`: `bpp` or `op`.
- `PHYSIQ_INPUT_MODE`: normally `v2v` or `i2v`.
- `PHYSIQ_RUN_INDEX`: `1` through `4`.
- `PHYSIQ_RUN_TAG`: normalized `run_01` through `run_04`.
- `PHYSIQ_SEED`: default mapping `42` through `45`.
- `PHYSIQ_NEGATIVE_PROMPT`: canonical long prompt for new P0 adapters.
- `PHYSIQ_NEGATIVE_PROMPT_VERSION`: prompt version supplied by the launcher.
- `PHYSIQ_NEGATIVE_PROMPT_SHA256`: prompt digest supplied by the launcher.
- `PHYSIQ_RESULT_FILE`: adapter-to-launcher result path; adapters must write it.
