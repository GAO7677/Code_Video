# Wan2.2 Probe Scaffold

This directory contains the first-stage probing code for testing whether
`WMReward / V-JEPA surprise` can be decoded from `wan2.2-ti2v-5b`
intermediate representations.

Scripts:

- `extract_probe_features.py`
  - Generates a small set of Wan videos from manifest prompts.
  - Captures intermediate transformer block features at selected denoising
    steps and layers.
  - Saves one `probe_features.pt` plus `meta.json` per sample.
- `run_probe_smoke.py`
  - Thin wrapper around `extract_probe_features.py`.
  - Intended for single-sample smoke tests.
- `build_probe_index.py`
  - Builds a flat CSV/JSONL index over extracted feature files.

Current pooling saved per `(step, branch, layer)`:

- `h_post_global_mean`
- `delta_h_global_mean`
- `h_post_frame_mean`
- `delta_h_frame_mean`
- token-level L2 summary stats

Default output location for large artifacts:

- `/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22`

## Dataset Pipeline

This directory now also contains a three-stage dataset pipeline that reuses
the existing external inference scripts without modifying them:

- `run_generation_pipeline.py`
  - Reads GT input JSONs from `v2v_jsons`.
  - Calls the external generators through subprocess.
  - Produces `mp4 + same-name json` per model.
  - Writes generation registries under `pipeline_root/manifests/`.
- `backfill_wmreward_scores.py`
  - Runs `../batch_compute_wmreward.py` separately on generated outputs.
  - Writes `wmreward` fields back into each generated result JSON.
  - Refreshes the generation registries with `surprise_score` and
    `similarity_score`.
- `build_generation_manifest.py`
  - Groups rows by the same `basename`.
  - Auto-pairs the lowest- and highest-surprise generations.
  - Writes a probe-ready manifest CSV/JSONL.

### Default Models

The generation pipeline has three fixed model branches:

- `base`
  - Script:
    `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wanti2v.py`
  - Wan root:
    `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
- `openvid_lora`
  - Script:
    `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_openvid_lorav2v.py`
  - Weights root:
    `/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000`
- `pybullet_lora`
  - Script:
    `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_openvid_0613pybullet_lorav2v.py`
  - Weights root:
    `/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500`

### Default Output Root

By default all large generated artifacts go to:

- `/data/gaoya/agent-data/outputs/wmreward_probe_wan22`

Layout:

- `generations/base`
- `generations/openvid_lora`
- `generations/pybullet_lora`
- `wmreward/<model_key>`
- `manifests/`

### Run Order

1. Generate videos and result JSONs:

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/wmreward/probe_wan22/run_generation_pipeline.py \
  --input-json-dir /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons \
  --pipeline-root /data/gaoya/agent-data/outputs/wmreward_probe_wan22
```

2. Compute WMReward and write it back into each generated JSON:

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/wmreward/probe_wan22/backfill_wmreward_scores.py \
  --pipeline-root /data/gaoya/agent-data/outputs/wmreward_probe_wan22 \
  --wmreward-checkpoint-path /data/gaoya/ckpt/Sylvest-vjepa2-vit-g/vitg-384.pt
```

3. Build a probe-ready paired manifest:

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/wmreward/probe_wan22/build_generation_manifest.py \
  --pipeline-root /data/gaoya/agent-data/outputs/wmreward_probe_wan22 \
  --subset-name generated_probe_pairs
```

### Auto-Pair Rule

`build_generation_manifest.py` uses this rule:

- Group all successful rows by the same `basename`.
- Keep only rows where `wmreward_status == ok` and `surprise_score` exists.
- Inside each basename group, sort by ascending `surprise_score`.
- Assign `low` to the lowest-surprise row.
- Assign `high` to the highest-surprise row.
- Define `group_gap = high_surprise_score - low_surprise_score`.
- Rank pairs globally by descending `group_gap`.

Tie handling:

- By default ties are kept.
- If two rows share the same score, ordering falls back to
  `(model_key, output_video_path)`.
- `--drop-ties` removes groups where lowest and highest scores are equal.

### Produced Files

- `manifests/generation_run_config.json`
- `manifests/generation_registry_<model>.csv`
- `manifests/generation_registry_all.csv`
- `manifests/wmreward_run_config.json`
- `manifests/generated_probe_pairs.csv`
- `manifests/generated_probe_pairs.jsonl`
- `manifests/generated_probe_pairs_summary.json`
