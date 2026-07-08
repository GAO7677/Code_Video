from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_DATA_ROOT = Path("/data/gaoya/agent-data")
AGENT_CACHE_ROOT = AGENT_DATA_ROOT / "cache"
AGENT_OUTPUT_ROOT = AGENT_DATA_ROOT / "outputs"
DATA_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp")
A_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench")
A_OUTPUT = A_ROOT / "output"
ABC_REPORT_ROOT = DATA_ROOT / "abc_report"
TMP_ROOT = DATA_ROOT / "tmp_eval_all"
RUN_ROOT = DATA_ROOT / "eval_runs"

PDI_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_benchmark/PDI-Bench-main")
PDI_FLORENCE_MODEL = Path("/data/gaoya/ckpt/microsoft-Florence-2-base")
WMREWARD_ROOT = Path("/home/gaoya/Code_Video/WMReward-main1/WMReward-main")
WMREWARD_VJEPA2_ROOT = Path("/home/gaoya/.cache/torch/hub/facebookresearch_vjepa2_main")
PROXY_CKPT = Path("/data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt")
PROXY_REPO = Path("/home/gaoya/Code_Video/vjepa2-main")
VIDEOPHY_ROOT = Path("/home/gaoya/Code_Video/videophy-main")
VIDEOPHY2_CKPT = Path("/data/gaoya/ckpt/videophysics-videophy_2_auto")
VPHY_PYTHON = Path("/data/gaoya/miniconda3/envs/vphy/bin/python")
PHYGROUND_ROOT = REPO_ROOT / "PhyGround"
PHYJUDGE_ADAPTER = Path("/data/gaoya/ckpt/phyjudge-9B")
PHYJUDGE_INFER = PHYJUDGE_ADAPTER / "infer.py"
PHYJUDGE_BASE = Path("/data/gaoya/ckpt/Qwen-Qwen3.5-9B")
FLUX_PYTHON = Path("/home/gaoya/miniconda3/envs/flux/bin/python")
SAM_PYTHON = Path("/home/gaoya/miniconda3/envs/sam/bin/python")
COSMOS_COOKBOOK_ROOT = REPO_ROOT / "cosmos-cookbook"
COSMOS_REASON1_ROOT = REPO_ROOT / "cosmos-reason1"
COSMOS_REASON1_MODEL = Path("/home/gaoya/model_weights/Cosmos-Reason1-7B")
VBENCH_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526/VBench-master")
VBENCH2_ROOT = VBENCH_ROOT / "VBench-2.0"
VBENCH_FULL_INFO = VBENCH_ROOT / "vbench" / "VBench_full_info.json"
VBENCH2_FULL_INFO = VBENCH2_ROOT / "vbench2" / "VBench2_full_info.json"
