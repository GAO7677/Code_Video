#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_ROOT/launch_xssc_experiment_verified_ssh118.sh" \
  "$SCRIPT_ROOT/feature_loss_step000500_physicsiq_verified_gpu67_remote.env"

