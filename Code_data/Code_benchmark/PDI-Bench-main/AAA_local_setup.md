# PDI-Bench Local Setup

This note records the local environment choice, asset layout, and download commands for this machine.

## Environment

Use the existing `sam` conda environment.

Current validated versions in `/home/gaoya/miniconda3/envs/sam`:

- `Python 3.10.19`
- `torch 2.1.0+cu118`
- `torch.version.cuda == 11.8`
- `nvcc 11.8`
- import checks passed for:
  - `sam2`
  - `cotracker`
  - `xformers`
  - `torch_scatter`

Rationale:

- This environment was the closest existing match to the official PDI-Bench requirement.
- It already had `sam2` installed before the upgrade.
- Upgrading this environment was lower-risk than downgrading `wan`.

## Central Storage Policy

Keep model weights under `/data/gaoya/ckpt`.

Keep benchmark data under `/data/gaoya/dataset`.

Do not redownload into the repo unless strictly necessary. Prefer symlinks from the repo to the central storage paths.

## Download Commands

All Hugging Face downloads should avoid local proxy variables and use the mirror endpoint:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
HF_ENDPOINT=https://hf-mirror.com \
HF_TOKEN='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' \
hf download facebook/cotracker3 \
--include scaled_offline.pth \
--local-dir /data/gaoya/ckpt/facebook-cotracker3
```

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
HF_ENDPOINT=https://hf-mirror.com \
HF_TOKEN='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' \
hf download AnteaWu/PDI-Dataset \
--repo-type dataset \
--include 'GT/**' \
--local-dir /data/gaoya/dataset/AnteaWu-PDI-Dataset
```

Optional: download the full PDI dataset, not only GT:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
HF_ENDPOINT=https://hf-mirror.com \
HF_TOKEN='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' \
hf download AnteaWu/PDI-Dataset \
--repo-type dataset \
--local-dir /data/gaoya/dataset/AnteaWu-PDI-Dataset
```

## Existing Local Assets

Already present locally:

- SAM2 checkpoint and config:
  - `/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt`
  - `/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_l.yaml`
- Depth-Anything weights:
  - `/data/gaoya/ckpt/LiheYoung-depth_anything_vitl14/pytorch_model.bin`
- RAFT weights:
  - `/data/gaoya/ckpt/RAFT-Things/models/raft-things.pth`

Still missing from the central storage unless downloaded separately:

- `/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth`
- `/data/gaoya/ckpt/mega-sam/megasam_final.pth`

## Non-HF Asset

`megasam_final.pth` is not documented in the local PDI-Bench README with a direct Hugging Face command.

The local README says to obtain it from the official Mega-SAM repository:

- `https://github.com/mega-sam/mega-sam`

Recommended central path after you obtain it:

- `/data/gaoya/ckpt/mega-sam/megasam_final.pth`

## Repo Expected Paths

The current PDI-Bench code expects:

- `checkpoints/sam2/sam2_hiera_large.pt`
- `checkpoints/sam2/sam2_hiera_l.yaml`
- `checkpoints/tracker/scaled_offline.pth`
- `third_party/mega_sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth`
- `third_party/mega_sam/checkpoints/megasam_final.pth`
- `third_party/mega_sam/cvd_opt/raft-things.pth`

It also expects the GT dataset under a repo-visible `videos/GT/...` tree when using the default batch scripts.

## Next Step

After all required assets are present, create symlinks from the repo to the central storage paths and then build Mega-SAM:

```bash
conda activate sam
bash scripts/build_mega_sam.sh
```
