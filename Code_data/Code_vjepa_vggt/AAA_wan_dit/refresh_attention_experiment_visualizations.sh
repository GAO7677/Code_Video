#!/usr/bin/env bash
set -euo pipefail

# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/refresh_attention_experiment_visualizations.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/wan_attention_experiment_visualizations}"

"${PYTHON}" "${SCRIPT_DIR}/visualize_paired_query_50seeds.py" \
  --config "${SCRIPT_DIR}/paired_query_head_stability_test5_50seeds.json" \
  --root /data/gaoya/agent-data/outputs/wan_dit_paired_query_50seeds \
  --output-dir "${OUTPUT_ROOT}/paired_query_50seeds"

"${PYTHON}" "${SCRIPT_DIR}/visualize_head_ablation_metric_curves.py" \
  --config "${SCRIPT_DIR}/head_ablation_allblocks_test5_gpu56.json" \
  --output-dir "${OUTPUT_ROOT}/head_ablation_test5"

cat > "${OUTPUT_ROOT}/index.html" <<'EOF'
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wan attention experiments</title>
<style>
body{margin:0;background:#f4f6f6;color:#17212b;font:14px/1.5 Arial,sans-serif;letter-spacing:0}
header{background:#203732;color:#fff;padding:18px 28px}h1{font-size:22px;margin:0}
main{max-width:1000px;margin:0 auto;padding:28px 24px}.links{display:grid;grid-template-columns:1fr 1fr;gap:16px}
a{display:block;background:#fff;border:1px solid #d4dade;border-radius:6px;padding:18px;color:#165f52;text-decoration:none}
a strong{display:block;font-size:17px;margin-bottom:5px;color:#17212b}a span{color:#5f6973}
@media(max-width:720px){.links{grid-template-columns:1fr}}
</style></head><body><header><h1>Wan attention experiment visualizations</h1></header>
<main><div class="links">
<a href="paired_query_50seeds/"><strong>50-seed head-role statistics</strong>
<span>Fixed latent-t2 query and moving-query comparison</span></a>
<a href="head_ablation_test5/"><strong>All-block/all-head test_5 ablation</strong>
<span>Generation coverage and benchmark metric curves</span></a>
</div></main></body></html>
EOF

echo "${OUTPUT_ROOT}/index.html"
