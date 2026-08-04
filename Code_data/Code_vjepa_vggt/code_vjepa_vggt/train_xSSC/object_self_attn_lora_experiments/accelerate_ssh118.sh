#!/usr/bin/env bash
set -euo pipefail

exec /mnt/data/gaoya/agent-data/envs/wan-cu128/bin/python \
  -m accelerate.commands.accelerate_cli "$@"
