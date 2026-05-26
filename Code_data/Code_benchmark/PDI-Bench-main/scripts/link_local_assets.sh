#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

link_file() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  rm -f "$dst"
  ln -s "$src" "$dst"
  echo "linked: $dst -> $src"
}

link_dir() {
  local src="$1"
  local dst="$2"
  rm -rf "$dst"
  ln -s "$src" "$dst"
  echo "linked: $dst -> $src"
}

link_file /data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt \
  "$ROOT/checkpoints/sam2/sam2_hiera_large.pt"

link_file /data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_l.yaml \
  "$ROOT/checkpoints/sam2/sam2_hiera_l.yaml"

link_file /data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth \
  "$ROOT/checkpoints/tracker/scaled_offline.pth"

link_file /data/gaoya/ckpt/LiheYoung-depth_anything_vitl14_raw/checkpoints/depth_anything_vitl14.pth \
  "$ROOT/third_party/mega_sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth"

link_file /data/gaoya/ckpt/mega-sam/megasam_final.pth \
  "$ROOT/third_party/mega_sam/checkpoints/megasam_final.pth"

link_file /data/gaoya/ckpt/RAFT-Things/models/raft-things.pth \
  "$ROOT/third_party/mega_sam/cvd_opt/raft-things.pth"

link_dir /data/gaoya/dataset/AnteaWu-PDI-Dataset \
  "$ROOT/videos"
