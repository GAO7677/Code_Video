from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp")
A_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench")
A_OUTPUT = A_ROOT / "output"
ABC_REPORT_ROOT = DATA_ROOT / "abc_report"
TMP_ROOT = DATA_ROOT / "tmp_eval_all"
RUN_ROOT = DATA_ROOT / "eval_runs"

PDI_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_benchmark/PDI-Bench-main")
PDI_FLORENCE_MODEL = Path("/data/gaoya/ckpt/microsoft-Florence-2-base")
WMREWARD_ROOT = Path("/home/gaoya/Code_Video/WMReward-main1/WMReward-main")
VJEPA2_ROOT = Path("/home/gaoya/.cache/torch/hub/facebookresearch_vjepa2_main")
WMREWARD_CKPT = Path("/data/gaoya/ckpt/Sylvest-vjepa2-vit-g/vitg-384.pt")
PROXY_CKPT = Path("/data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt")
PROXY_REPO = Path("/home/gaoya/Code_Video/vjepa2-main")
VIDEOPHY_ROOT = Path("/home/gaoya/Code_Video/videophy-main")
VIDEOPHY2_CKPT = Path("/data/gaoya/ckpt/videophysics-videophy_2_auto")
VPHY_PYTHON = Path("/data/gaoya/miniconda3/envs/vphy/bin/python")
