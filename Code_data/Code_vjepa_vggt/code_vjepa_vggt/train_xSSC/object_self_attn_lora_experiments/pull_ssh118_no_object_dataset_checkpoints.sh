#!/usr/bin/env bash
# Run in the foreground on the local host:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/pull_ssh118_no_object_dataset_checkpoints.sh
set -u

REMOTE_PYBULLET="/mnt/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/full_sa_no_object_pybullet100_gpu67_ssh118_resume500/ssh118_resume500_retry4_20260804/checkpoints/"
REMOTE_KUBRIC="/mnt/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/full_sa_no_object_kubric100_gpu67_ssh118_1000steps/ssh118_after_pybullet_20260804/checkpoints/"
LOCAL_PYBULLET="/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/full_sa_no_object_pybullet100_gpu67_1000steps/serial_20260804T115337Z/checkpoints/"
LOCAL_KUBRIC="/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/full_sa_no_object_kubric100_gpu67_1000steps/serial_20260804T115337Z/checkpoints/"

mkdir -p "${LOCAL_PYBULLET}" "${LOCAL_KUBRIC}"
while true; do
  rsync -a "118:${REMOTE_PYBULLET}" "${LOCAL_PYBULLET}" 2>/dev/null || true
  rsync -a "118:${REMOTE_KUBRIC}" "${LOCAL_KUBRIC}" 2>/dev/null || true
  date -u '+[%Y-%m-%dT%H:%M:%SZ] SSH 118 checkpoint pull complete'
  sleep 60
done
