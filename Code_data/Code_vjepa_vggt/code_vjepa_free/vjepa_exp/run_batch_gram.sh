#!/usr/bin/env bash
# Batch V-JEPA Gram extraction for GT and all generation methods
# Usage: bash run_batch_gram.sh [--limit N]

set -euo pipefail

PYTHON=/data/gaoya/miniconda3/envs/vjepa2/bin/python
SCRIPT="$(dirname "$0")/extract_vjepa_gram.py"
CKPT=/data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt
ASSETS_ROOT="/data/gaoya/AAA_test_video/Output_try0526/ABD_test/_viz_v2/A/baseline/assets"
OUT_ROOT="/data/gaoya/AAA_test_video/0626vjepa_free"
LIMIT=${1:-99999}

count=0
for case_dir in "$ASSETS_ROOT"/*/; do
    case_id=$(basename "$case_dir")

    # GT (gt_full)
    gt_video="$case_dir/gt_full.browser.mp4"
    if [[ -f "$gt_video" ]]; then
        out_dir="$OUT_ROOT/GT/$case_id"
        mkdir -p "$out_dir"
        if [[ ! -f "$out_dir/gt_full.browser_vjepa.npz" ]]; then
            echo "[GT] $case_id"
            "$PYTHON" "$SCRIPT" --video "$gt_video" --ckpt "$CKPT" --out-dir "$out_dir"
        else
            echo "[GT] $case_id  (cached)"
        fi
        count=$((count+1))
        [[ $count -ge $LIMIT ]] && exit 0
    fi

    # generation methods
    for method_dir in "$case_dir"/*/; do
        method=$(basename "$method_dir")
        vid="$method_dir/output.browser.mp4"
        [[ -f "$vid" ]] || continue

        stem=$(basename "$vid" .mp4)
        out_dir="$OUT_ROOT/$method/$case_id"
        mkdir -p "$out_dir"
        npz="$out_dir/${stem}_vjepa.npz"
        if [[ ! -f "$npz" ]]; then
            echo "[$method] $case_id"
            "$PYTHON" "$SCRIPT" --video "$vid" --ckpt "$CKPT" --out-dir "$out_dir"
        else
            echo "[$method] $case_id  (cached)"
        fi
        count=$((count+1))
        [[ $count -ge $LIMIT ]] && exit 0
    done
done

echo "Done. Total processed: $count"
