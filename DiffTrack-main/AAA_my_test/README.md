# DiffTrack analysis on 0718ToyDataset

This directory adapts `analyze_real.py` to ordinary MP4 files without modifying
the original DiffTrack source. CoTracker tracks are pseudo-ground truth, not
renderer ground truth. Videos are resized using DiffTrack's `source -> 256 ->
480x720` preprocessing and truncated to the first 49 frames.

The seven manifest videos contain four unique video byte streams. Analysis runs
once per unique video; `tracks_manifest.json` records aliases for repeated
anchor/base files.

## 1. Prepare tracks

```bash
PYTHONPATH=/home/gaoya/Code_Video/co-tracker-main \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  AAA_my_test/prepare_toy_tracks.py \
  --dataset-root /data/gaoya/AAA_test_video/Dataset_physV/0718ToyDataset \
  --output-dir /data/gaoya/agent-data/outputs/difftrack_0718toy/tracks \
  --cotracker-root /home/gaoya/Code_Video/co-tracker-main \
  --checkpoint /data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth \
  --device cuda:0
```

## 2. Run all 50 noise probes

The completed run used ModelScope because the installed Hugging Face client
could not follow the repository redirect:

```bash
modelscope download \
  --model ZhipuAI/CogVideoX-2b \
  --local_dir /data/gaoya/agent-data/weights/CogVideoX-2b-modelscope \
  --max-workers 8
```

Alternatively, download the pinned Hugging Face snapshot with resumable `curl`:

```bash
bash AAA_my_test/download_cogvideox_2b.sh \
  /data/gaoya/agent-data/weights/CogVideoX-2b
```

```bash
HF_HOME=/data/gaoya/agent-data/cache/huggingface \
PYTHONPATH=/home/gaoya/Code_Video/DiffTrack-main/diffusers/src:/home/gaoya/Code_Video/DiffTrack-main \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  AAA_my_test/analyze_real_toy.py \
  --dataset-root /data/gaoya/AAA_test_video/Dataset_physV/0718ToyDataset \
  --track-dir /data/gaoya/agent-data/outputs/difftrack_0718toy/tracks \
  --output-dir /data/gaoya/agent-data/outputs/difftrack_0718toy/cogvideox_2b \
  --cache-dir /data/gaoya/agent-data/cache/huggingface \
  --model-path /data/gaoya/agent-data/weights/CogVideoX-2b-modelscope \
  --model cogvideox_t2v_2b \
  --device cuda:0 \
  --matching-accuracy \
  --conf-attn-score
```

Use `--start N --end M` to split unique videos across GPUs. Use
`--inverse-steps 0 10 20 30 40 49` for a smoke test or coarse sweep.

Run all four unique videos on GPUs 0-3 with a foreground parent process:

```bash
bash AAA_my_test/run_all_toy.sh
```

Each completed noise step is saved atomically to `step_state.npz`; rerunning the
same command resumes missing steps. Per-sample logs are stored under
`/data/gaoya/agent-data/outputs/difftrack_0718toy/cogvideox_2b/logs`.

Aggregate the completed PCK surfaces with:

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  AAA_my_test/summarize_results.py \
  --result-dir /data/gaoya/agent-data/outputs/difftrack_0718toy/cogvideox_2b
```

## 3. Per-object and background trajectory visualization

The first region-level experiment uses the dataset's lossless renderer instance
masks, samples 32 interior points independently for object A, object B, and the
background, and compares CoTracker with Q/K tracks at layer 17 / inverse step
49:

```bash
PYTHONPATH=/home/gaoya/Code_Video/DiffTrack-main/diffusers/src:/home/gaoya/Code_Video/DiffTrack-main \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  AAA_my_test/analyze_region_tracks.py \
  --case-key case_019_wheel_hits_block \
  --sample-type base \
  --regions object_a object_b background \
  --points-per-region 32 \
  --query-frame 0 \
  --layer 17 \
  --inverse-step 49 \
  --device cuda:0
```

Each region produces the sampled mask image, separate CoTracker and Q/K videos,
an overlay comparison video, raw tracks, visibility, and PCK/error metrics under
`/data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks`.
