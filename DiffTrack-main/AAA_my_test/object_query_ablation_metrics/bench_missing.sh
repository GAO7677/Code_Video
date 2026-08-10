
# bash /home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/object_query_ablation_metrics/bench_missing.sh \
#   /path/to/result_directory \
#   --gpu 5
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${WAN_PYTHON:-/data/gaoya/miniconda3/envs/wan/bin/python}"

exec "${PYTHON}" "${SCRIPT_DIR}/fill_missing_metrics.py" "$@"
