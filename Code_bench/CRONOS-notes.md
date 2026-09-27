# CRONOS local intake

Inspected on 2026-09-27. No evaluation environments or model weights installed.

## Locations and revisions

- Code: `/home/gaoya/Code_Video/Code_bench/CRONOS-benchmark`
- Git revision: `c56e317f8e433002f197d6fb1e410d7a0cb70fe2`
- Dataset: `/data/gaoya/dataset/CRONOS-benchmark`
- Dataset revision: `b29c3fb10474b38dd7c991f8a2762a0e8a1d0bac`
- Dataset download status: complete; archives are not extracted.
- All 15 shards passed remote size and SHA-256 verification on 2026-09-27.
- Exact archive size: 18,846,812,160 bytes (18.85 GB; 17.55 GiB).
- Archive inventory verified: 675 unique sequences, 182,250 JPEGs, 1,350 MP4s.
- Verification report: `/data/gaoya/dataset/CRONOS-benchmark/download-verification.json`.
- Remote shard manifest: `/data/gaoya/dataset/CRONOS-benchmark/download-manifest.json`.
- GitHub: https://github.com/GenIntel/CRONOS-benchmark
- Dataset: https://huggingface.co/datasets/genintel/CRONOS-benchmark
- Project: https://genintel.github.io/CRONOS/
- Paper: https://arxiv.org/abs/2605.23699

## Purpose

CRONOS evaluates counterfactual physical consistency in video generation/world
models. It varies scene, camera viewpoint, object appearance and object category
to test whether physical predictions remain appropriate under interventions.
Object-category changes may also change physical parameters; this is not simply
a demand for identical trajectories across all variants.

This repository provides evaluation scripts, not a video generator or a training
pipeline. Generated videos must be supplied separately. The published scripts
operate on one generated video and one reference sequence at a time; a full
benchmark requires an outer loop and aggregation across variants/seeds.

## Dataset (official dataset card)

- 675 synthetic Unreal Engine 5 sequences: fall 300, collision 300, occlusion 75.
- Five scenes: apartment, house, garden, pool, kitchen.
- Five objects: Bottle, Can, TennisBall, ToyTruck, SoccerBall.
- Three appearances; four views for fall/collision, side view only for occlusion.
- 90 frames per sequence, 1920 x 1080.
- RGB, depth and mask JPEG frames, complete.mp4, conditioning_5.mp4, metadata.json.
- 15 uncompressed TAR shards, approximately 18.8 GB in the current revision.
- Verified archive hierarchy:
  `dataset/{event}/{scene}/{object}/{appearance}/{view}/`.
- I2V uses the first image; V2V uses the five-frame conditioning clip.

Keep the archives for now: extracting a second full copy would require roughly
another dataset-sized allocation. Check free space before extraction.

## Evaluation flow

`scripts/run_predictions.sh --video VIDEO --ref REF_DIR --output OUTPUT_DIR`
executes:

1. SAM3 segmentation, initialized using the first reference mask.
2. CoTracker3 point tracking.
3. DINOv2 object appearance embeddings.
4. DisMo motion embeddings for the generated video.
5. DisMo motion embeddings for the reference video.
6. Qwen3-VL-32B event-specific questions.
7. SAM3D-Objects reconstruction, by default every 10 frames.

Then `metrics/metrics.py` computes:

| Raw metric | Meaning | Pass condition |
| --- | --- | --- |
| bg_mse | Background change relative to frame zero | <= 0.25 |
| motion_similarity | Generated/reference DisMo cosine similarity | >= 0.70 |
| object_consistency | Temporal object DINO similarity | >= 0.50 |
| mean_chamfer_distance | Aligned reconstructed mesh shape change | <= 0.80 |
| vlm_positive_fraction | Fraction of expected VLM answers | >= 0.35 |
| disappearance_fraction | Permanent mid-frame object disappearance | <= 0.10 |

`success` requires all the conditions above. Missing/nonfinite values fail.
Thresholds are applied to raw metrics; separately scaled values are also saved.
Several aggregations use the worst 5% rather than a simple overall mean.
Object appearance/shape evaluation excludes heavily occluded frames using the
configured visible-area threshold (0.25 of the frame-zero area).

## Setup considerations verified in source

- Main environment: Python 3.11, PyTorch 2.5.1/cu124 in setup_cronos.sh.
- SAM3D uses a separate environment due to conflicting dependencies. Pass its
  interpreter with `PYTHON_SAM3D`.
- `config.py` still contains placeholder paths for SAM3, SAM3D, CLIP and cache.
- `setup_cronos.sh` references `constraints-cronos.txt`, but that file is absent
  from this Git revision. The provided installer needs this resolved first.
- `scripts/generate_prompt.py --help` fails with ImportError: it imports
  `OBJECT_MATERIALS`, while config.py defines `OBJECT_APPEARANCES`. Dataset
  metadata already contains prompts, so this helper is not needed to read them.
- `setup_sam3d.sh` uses cu121 package sources, despite README cu124 wording;
  inspect its actual dependency/toolkit requirements before installation.
- `scripts/run_qwen.py` loads the 32B model with BF16 and device_map="auto",
  without a quantization configuration. Weights alone are roughly 64 GB
  (decimal), plus inference overhead. README's 24 GB figure is not sufficient
  for a fully GPU-resident default load on one GPU.
- Set explicit CUDA_VISIBLE_DEVICES before any future model run. Physical GPU 4
  is prohibited. Do not allow automatic device placement to expose GPU 4.
- Keep environments, weights, Torch Hub and HF caches, and evaluation outputs
  under `/data/gaoya/agent-data` or another authorized data path.
- README estimates another ~80 GB for environments/weights. Current disk space
  should be reassessed before any setup or evaluation.
- Code LICENSE is MIT. The dataset card is internally inconsistent: YAML says
  CC-BY-4.0, while its body says Apache 2.0. Clarify with the maintainers if needed.

## Repeat/resume dataset download

The dataset is public, so this command does not require a token:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  HF_ENDPOINT=https://hf-mirror.com \
  HF_HOME=/data/gaoya/agent-data/cache/huggingface \
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_XET=1 \
  /home/gaoya/miniconda3/envs/flux/bin/hf download \
  genintel/CRONOS-benchmark --repo-type dataset \
  --revision b29c3fb10474b38dd7c991f8a2762a0e8a1d0bac \
  --local-dir /data/gaoya/dataset/CRONOS-benchmark --max-workers 8
```

## Evaluation command template (not run)

After resolving setup issues, configuring paths and extracting the needed
reference sequence, use the two commands below. GPU 0 is only an example;
choose available permitted hardware with enough memory before running.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHON_SAM3D=/data/gaoya/agent-data/envs/cronos-sam3d/bin/python \
  bash scripts/run_predictions.sh \
  --video /path/to/generated.mp4 \
  --ref /path/to/extracted/event/scene/object/appearance/view \
  --output /data/gaoya/agent-data/outputs/cronos/example

python metrics/metrics.py \
  --video /path/to/generated.mp4 \
  --ref /path/to/extracted/event/scene/object/appearance/view \
  --output /data/gaoya/agent-data/outputs/cronos/example
```
