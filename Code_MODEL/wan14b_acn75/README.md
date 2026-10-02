# Pusa-Wan2.2-14B A / C / N75

This experiment starts from native Wan2.2-T2V-A14B plus the released Pusa high/low adapters. Pusa is merged at the explicit scale in each config and frozen. New rank32 / alpha64 / dropout0 adapters train self-attention Q/K/V/O, cross-attention Q/K/V/O, and both FFN linear layers. There are 153,354,240 trainable parameters per expert, 306,708,480 total. No LightX2V adapter is loaded.

## Configuration and outputs

Edit `configs/A.json`, `configs/C.json`, and `configs/N75.json`. These contain model paths/revisions, Pusa scales, target modules, cache contract, dataset selection, optimizer, precision, W&B and output paths.

- Generic2175, `caption_initial0907`, 49 frames, 512×896, seed42.
- A: context shares future noise level; C: clean context; N75: nearest paired training sigma to 0.75×future sigma. All use future-only loss.
- Training scheduler follows the existing Wan5B run: shift1, 1000 discrete timesteps. Inference shift is a separate setting.
- Effective batch8 = 1 sample/GPU × 2 GPUs × 4 accumulation microsteps. The sampled timestep/expert is shared across ranks for each microstep; microsteps may use different experts.
- `paged_adamw8bit`, LR1e-4 constant, no warmup, weight decay0.01, max gradient norm1.
- Each mode runs to 3000 optimizer steps, saves paired high/low adapters every500. Each expert's actual update/sample counts are logged, because some sampled steps only use one expert.
- Online W&B is required; authentication failures stop the run. Project: `wan14b-acn75`.
- Frozen weights and fresh LoRAs materialize on CUDA. FSDP2 shards across the GPU pair. No CPU model offload.

Default output root:
`/data/gaoya/agent-data/outputs_v1/wan14b_acn75/20261002_pusa_lora32/`

Each mode has `train.log`, `metrics.jsonl`, `resolved_config.json`, `model_audit.json`, `checkpoints/step_XXXXXX/adapters.safetensors`, and `latest/` for optimizer/RNG/data-position resume. A single adapters file contains both expert namespaces. Historical checkpoints do not duplicate optimizer state.

## Cache contract

Old Wan5B VAE caches `[48,13,32,56]` cannot be reused. Native 14B latents are `[16,13,64,112]`. VAE and UMT5 execute in FP32 on CUDA; latents store BF16, text stores FP32 `[512,4096]`. Any online evaluation must use the same preprocessing, normalization and component compute precision.

The supplied RGB context is8 frames and the aligned latent context is2. Native causal Wan VAE consumes only RGB0 through4 for two complete latent steps; the last three supplied RGB frames form an incomplete block. The audit records this explicitly and compares a full49-frame encoding prefix with an independently encoded8-frame prefix. It never inserts the latent that would require RGB8 (the ninth frame).

Source video hashes and exact sampling indices are checked against the original Wan5B cache provenance. Each cache entry contains source/component/tokenizer/code hashes, finite-value checks and an online numerical comparison. File checksums and identity are checked by the training dataset. Unknown or modified entries fail closed.

Old text embeddings are compared caption by caption with native14B output. `cache.reuse_legacy_text=true` permits reuse only for entries that passed; it does not exempt any caption from online verification. The default writes fresh online outputs. A small successful sample never approves all711 captions.

Atol/rtol are explicit in `cache.validation`. BF16 quantization error is recorded separately: equality with cached latents is measured after the declared BF16 storage conversion, while context-causality equality is checked in FP32.

## Preparation

Run from this directory. The interpreter is:

```bash
PY=/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python
```

`prepare_assets.py` defaults to a read-only disk/identity preflight. It pins official revisions in `configs/assets.json`, reuses the existing UMT5 checkpoint only after full SHA256 equality, and requires space for missing weights plus60GiB reserve before downloading. Original DiT downloads are FP32; the native model stores approximately136GB including Pusa and encoder files, or approximately125GB of additional downloads when the existing identical UMT5 is reused. Reserve approximately200GB free for the full campaign. Do not delete existing task data to make room without explicit authorization.

```bash
$PY prepare_assets.py
$PY prepare_assets.py --download
```

Read-only dataset/cache inspection:

```bash
$PY audit_cache.py --config configs/A.json --mode inspect
```

Online sample tests, one component at a time (use an available GPU in the authorized pair):

```bash
CUDA_VISIBLE_DEVICES=1 $PY audit_cache.py --config configs/A.json --mode build --component text --device cuda:0 --limit 8
CUDA_VISIBLE_DEVICES=1 $PY audit_cache.py --config configs/A.json --mode verify --component text --device cuda:0 --limit 8
CUDA_VISIBLE_DEVICES=1 $PY audit_cache.py --config configs/A.json --mode build --component vae --device cuda:0 --limit 8
CUDA_VISIBLE_DEVICES=1 $PY audit_cache.py --config configs/A.json --mode verify --component vae --device cuda:0 --limit 8
```

Omit `--limit` to build the full cache after storage is available. `--num-shards` and `--shard-index` support splitting one component across devices; never run duplicate component shards on the same GPU. Code/contract changes require fresh validation and must not be silently accepted into an old cache.

## Validation and launch

A small runtime test exercises FSDP2, BNB, both experts and checkpoint round trips without downloading DiT weights. It does **not** approve production memory or data compatibility:

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 $PY -m torch.distributed.run --standalone --nproc_per_node=2 test_fsdp_runtime.py --output-dir /data/gaoya/agent-data/outputs_v1/wan14b_acn75/20261002_pusa_lora32/runtime_probe
```

After assets and the first8 cache cases are ready, perform the actual full-resolution14B test on GPU0,1:

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 $PY -m torch.distributed.run --standalone --nproc_per_node=2 train.py --config configs/A.json --smoke --output-dir /data/gaoya/agent-data/outputs_v1/wan14b_acn75/20261002_pusa_lora32/smoke
```

This performs A/C/N75 updates, covers both experts, checks finite loss/gradients and parameter changes, checks exact adapter/optimizer/RNG restoration, and records peak CUDA memory. Only `smoke_passed.json` from this full model test can authorize formal launch. Config/code/cache changes invalidate the gate.

Once the actual smoke passes and all2175 caches are verified, the recommended queue is A→N75 onGPU0,1 and C onGPU2,3. The launcher waits for three idle readings, never terminates an existing process, and accepts only these two pairs. The user's latest instruction explicitly reauthorizesGPU0,1; GPU4 remains forbidden.

```bash
SMOKE=/data/gaoya/agent-data/outputs_v1/wan14b_acn75/20261002_pusa_lora32/smoke/smoke_passed.json
$PY launch.py --config configs/A.json configs/N75.json --gpus 0,1 --smoke-report "$SMOKE" --session wan14b-A-N75 --execute
$PY launch.py --config configs/C.json --gpus 2,3 --smoke-report "$SMOKE" --session wan14b-C --execute
```

`--execute` creates the tmux workers. Without it the launcher only validates and prints the command. No formal queue should be launched from the small runtime test alone.

## Upstream provenance

- [Pusa official repository](https://github.com/Yaofang-Liu/Pusa-VidGen), minimal verified model files under sibling `Pusa-VidGen/`, pinned source manifest there.
- [Pusa adapters](https://huggingface.co/RaphaelLiu/Pusa-Wan2.2-V1), revision `d66132768e8783a60a53fb05e64938c75e21b2a4`.
- [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B), revision `c8c270b13ee05bfa474194ac9fb07a5868a97cea`.
- Official VAE/UMT5 source copies in `vendor/`; exact files are part of the cache contract.
