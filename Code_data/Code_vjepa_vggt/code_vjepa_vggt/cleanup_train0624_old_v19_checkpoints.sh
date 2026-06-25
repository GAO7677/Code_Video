#!/usr/bin/env bash
set -euo pipefail

TARGETS=(
  "/home/gaoya/AAA_train_cache/train0624/checkpoints/pybullet0624_diffsynth_object_v_newtrain_gpu67_lossfix_v19_priorfirst_boxgate_smoke"
  "/home/gaoya/AAA_train_cache/train0624/checkpoints/pybullet0624_diffsynth_object_v_newtrain_gpu67_lossfix_v19_priorfirst"
  "/home/gaoya/AAA_train_cache/train0624/checkpoints/pybullet0624_diffsynth_object_v_newtrain_gpu67_lossfix_v19_priorfirst_boxanchor_smoke"
  "/home/gaoya/AAA_train_cache/train0624/checkpoints/pybullet0624_diffsynth_object_v_newtrain_gpu67_lossfix_v19_priorfirst_smoke"
)

echo "Will remove:"
for path in "${TARGETS[@]}"; do
  if [[ -e "${path}" ]]; then
    du -sh "${path}" 2>/dev/null || true
  else
    echo "missing ${path}"
  fi
done

for path in "${TARGETS[@]}"; do
  if [[ -e "${path}" ]]; then
    rm -rf "${path}"
  fi
done

echo "Done."
