#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
GPU_ID="${GPU_ID:-2}"
MODE="${1:-all}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

case "${MODE}" in
  mapping|forward|render|all) ;;
  *)
    echo "Usage: $0 [mapping|forward|render|all] [extra arguments]" >&2
    exit 2
    ;;
esac
if [[ "${GPU_ID}" == "4" ]]; then
  echo "GPU 4 is prohibited by workspace rules." >&2
  exit 2
fi
if [[ "${MODE}" == "forward" || "${MODE}" == "all" ]]; then
  used_mib="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "${used_mib}" -gt 2000 ]]; then
    echo "GPU ${GPU_ID} is using ${used_mib} MiB; refusing to overlap the diagnostic." >&2
    exit 2
  fi
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home/gaoya/Code_Video/co-tracker-main:/home/gaoya/Code_Video/DiffTrack-main:/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main:/home/gaoya/Grounded-SAM-2-main"

exec "${PYTHON}" "${HERE}/run_pybullet_latent_mask_correspondence_diagnostics.py" \
  "${MODE}" --device cuda:0 "$@"
